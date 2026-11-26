from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..episodic import EpisodeConsolidator, LLMEpisodeConsolidator
from ..extractors import JsonCompletionClient
from ..models import EpisodeRecord
from ..store import TimelineBranchConflictError

if TYPE_CHECKING:
    from .core import CanonArcEngine


def _active_provenance_episode_ids(engine: CanonArcEngine, step: int, timeline_id: str) -> set[str]:
    """Returns episode_ids referenced by active entities or edges at `step`."""
    pinned: set[str] = set()
    for ent in engine.store.get_all_active_entities(timeline_id, step):
        if ent.caused_by_episode_id:
            pinned.add(ent.caused_by_episode_id)
    for edge in engine.store.get_all_active_edges(timeline_id, step):
        if edge.caused_by_episode_id:
            pinned.add(edge.caused_by_episode_id)
    return pinned


def _select_consolidation_targets(
    engine: CanonArcEngine,
    step: int,
    *,
    before_step: int,
    max_salience: float = 4.0,
    predicate: str = "consolidated_summary",
    chapter_id: str | None = None,
    preserve_provenance: bool = True,
    preserve_deceived: bool = True,
    timeline_id: str,
) -> list[EpisodeRecord]:
    if before_step <= 0:
        return []
    candidates = engine.store.query_episodes(
        timeline_id=timeline_id,
        cutoff_step=before_step - 1,
    )
    pinned_ids = (
        _active_provenance_episode_ids(engine, step, timeline_id)
        if preserve_provenance
        else set()
    )
    targets = [
        ep
        for ep in candidates
        if ep.salience <= max_salience
        and ep.predicate != predicate
        and (chapter_id is None or ep.chapter_id == chapter_id)
        and (not preserve_provenance or ep.episode_id not in pinned_ids)
        and (not preserve_deceived or "deceived" not in ep.known_by.values())
    ]
    targets.sort(key=lambda ep: (ep.step, ep.episode_id))
    return targets


def _assert_consolidation_branch_safe(
    engine: CanonArcEngine,
    min_target_step: int,
    timeline_id: str,
    *,
    force: bool,
) -> None:
    if force:
        return
    for tl_info in engine.list_timelines():
        if (
            tl_info.parent_id == timeline_id
            and tl_info.fork_step is not None
            and tl_info.fork_step >= min_target_step
        ):
            raise TimelineBranchConflictError(
                f"Cannot consolidate episodes on '{timeline_id}' at step {min_target_step}: "
                f"child branch '{tl_info.timeline_id}' forked at step {tl_info.fork_step}. "
                "Pass force=True to override."
            )


def _commit_consolidated_batch(
    engine: CanonArcEngine,
    step: int,
    targets: Sequence[EpisodeRecord],
    *,
    summary: str,
    predicate: str,
    episode_id: str | None,
    chapter_id: str | None,
    salience: float,
    location_id: str | None,
    key_participant_ids: Sequence[str] | None,
    max_participants: int | None,
    epistemic_merge: Literal["union", "intersection"],
    anchor_step: Literal["current", "last_target"],
    force: bool,
    timeline_id: str,
) -> EpisodeRecord:
    min_target_step = min(ep.step for ep in targets)
    _assert_consolidation_branch_safe(engine, min_target_step, timeline_id, force=force)

    all_participants: dict[str, str] = {}
    participant_counts: Counter[str] = Counter()
    for ep in targets:
        for eid, role in ep.participants.items():
            all_participants.setdefault(eid, role)
            participant_counts[eid] += 1

    if key_participant_ids:
        selected_ids = [eid for eid in key_participant_ids if eid in all_participants]
        if max_participants is not None:
            selected_ids = selected_ids[: max(1, max_participants)]
        if not selected_ids:
            selected_ids = list(all_participants.keys())
            if max_participants is not None:
                selected_ids = [
                    eid for eid, _ in participant_counts.most_common(max(1, max_participants))
                ]
        participants = {eid: all_participants[eid] for eid in selected_ids}
    elif max_participants is not None and len(all_participants) > max_participants:
        top_ids = [eid for eid, _ in participant_counts.most_common(max(1, max_participants))]
        participants = {eid: all_participants[eid] for eid in top_ids}
    else:
        participants = all_participants

    restrict_witnesses = key_participant_ids is not None or max_participants is not None
    allowed_witnesses = set(participants.keys()) if restrict_witnesses else None
    witnesses: list[str] = []
    seen_witnesses: set[str] = set()
    for ep in targets:
        for wid in ep.witnesses:
            if allowed_witnesses is not None and wid not in allowed_witnesses:
                continue
            if wid not in seen_witnesses:
                seen_witnesses.add(wid)
                witnesses.append(wid)

    known_by: dict[str, Literal["direct", "told", "deceived"]] = {}
    if epistemic_merge == "intersection":
        per_ep_known: list[dict[str, Literal["direct", "told"]]] = []
        for ep in targets:
            ep_map: dict[str, Literal["direct", "told"]] = {
                eid: mode
                for eid, mode in ep.known_by.items()
                if mode != "deceived"
            }
            per_ep_known.append(ep_map)
        if per_ep_known:
            common_ids = set(per_ep_known[0].keys())
            for ep_map in per_ep_known[1:]:
                common_ids &= set(ep_map.keys())
            for eid in sorted(common_ids):
                modes = {ep_map[eid] for ep_map in per_ep_known}
                known_by[eid] = "direct" if modes == {"direct"} else "told"
    else:
        for ep in targets:
            for eid, mode in ep.known_by.items():
                if mode == "deceived":
                    continue
                if known_by.get(eid) != "direct":
                    known_by[eid] = mode

    eff_location = location_id
    if eff_location is None:
        distinct_locs = {ep.location_id for ep in targets if ep.location_id}
        if len(distinct_locs) == 1:
            eff_location = next(iter(distinct_locs))

    eff_chapter = chapter_id
    if eff_chapter is None:
        distinct_chapters = {ep.chapter_id for ep in targets if ep.chapter_id}
        if len(distinct_chapters) == 1:
            eff_chapter = next(iter(distinct_chapters))

    eff_step = max(ep.step for ep in targets) if anchor_step == "last_target" else step

    summary_ep = EpisodeRecord(
        episode_id=episode_id or f"ep_consolidated_{eff_step}_{uuid.uuid4().hex[:8]}",
        timeline_id=timeline_id,
        step=eff_step,
        predicate=predicate,
        summary=summary,
        participants=participants,
        witnesses=witnesses,
        known_by=known_by,
        location_id=eff_location,
        chapter_id=eff_chapter,
        consolidated_from=[ep.episode_id for ep in targets if ep.episode_id],
        salience=salience,
    )
    with engine._store_transaction():
        committed = engine.index_episodes(eff_step, [summary_ep], timeline_id=timeline_id)[0]
        delete_fn = getattr(engine.store, "delete_episodes", None)
        if callable(delete_fn):
            delete_fn([ep.episode_id for ep in targets], timeline_id=timeline_id)
    return committed


def count_consolidatable_episodes_impl(
    engine: CanonArcEngine,
    step: int,
    *,
    before_step: int,
    max_salience: float = 4.0,
    predicate: str = "consolidated_summary",
    chapter_id: str | None = None,
    preserve_provenance: bool = True,
    preserve_deceived: bool = True,
    timeline_id: str | None = None,
) -> int:
    if not engine._policy.has_tier(6):
        return 0
    tl = timeline_id or engine.active_timeline
    return len(
        _select_consolidation_targets(
            engine,
            step,
            before_step=before_step,
            max_salience=max_salience,
            predicate=predicate,
            chapter_id=chapter_id,
            preserve_provenance=preserve_provenance,
            preserve_deceived=preserve_deceived,
            timeline_id=tl,
        )
    )


def consolidate_episodes_impl(
    engine: CanonArcEngine,
    step: int,
    *,
    before_step: int,
    summary: str,
    max_salience: float = 4.0,
    predicate: str = "consolidated_summary",
    episode_id: str | None = None,
    chapter_id: str | None = None,
    salience: float = 7.5,
    location_id: str | None = None,
    key_participant_ids: Sequence[str] | None = None,
    max_participants: int | None = None,
    preserve_provenance: bool = True,
    preserve_deceived: bool = True,
    epistemic_merge: Literal["union", "intersection"] = "union",
    anchor_step: Literal["current", "last_target"] = "current",
    force: bool = False,
    timeline_id: str | None = None,
) -> EpisodeRecord | None:
    if not engine._policy.has_tier(6):
        return None
    tl = timeline_id or engine.active_timeline
    targets = _select_consolidation_targets(
        engine,
        step,
        before_step=before_step,
        max_salience=max_salience,
        predicate=predicate,
        chapter_id=chapter_id,
        preserve_provenance=preserve_provenance,
        preserve_deceived=preserve_deceived,
        timeline_id=tl,
    )
    if not targets:
        return None
    return _commit_consolidated_batch(
        engine,
        step,
        targets,
        summary=summary,
        predicate=predicate,
        episode_id=episode_id,
        chapter_id=chapter_id,
        salience=salience,
        location_id=location_id,
        key_participant_ids=key_participant_ids,
        max_participants=max_participants,
        epistemic_merge=epistemic_merge,
        anchor_step=anchor_step,
        force=force,
        timeline_id=tl,
    )


def resolve_episode_consolidator(
    engine: CanonArcEngine,
    consolidator: EpisodeConsolidator | None,
    client: JsonCompletionClient | None,
) -> EpisodeConsolidator:
    if consolidator is not None:
        return consolidator
    if client is not None:
        return LLMEpisodeConsolidator(client)
    if engine.episode_consolidator is not None:
        return engine.episode_consolidator
    if engine.extractor is not None and isinstance(engine.extractor, JsonCompletionClient):
        return LLMEpisodeConsolidator(engine.extractor)
    raise ValueError(
        "No EpisodeConsolidator or JsonCompletionClient provided. Pass consolidator=..., "
        "client=..., configure episode_consolidator on CanonArcEngine, or attach an "
        "extractor implementing JsonCompletionClient."
    )


def consolidate_episodes_auto_impl(
    engine: CanonArcEngine,
    step: int,
    *,
    before_step: int,
    consolidator: EpisodeConsolidator | None = None,
    client: JsonCompletionClient | None = None,
    max_salience: float = 4.0,
    min_episodes: int = 3,
    max_episodes_per_batch: int = 30,
    predicate: str = "consolidated_summary",
    chapter_id: str | None = None,
    salience: float | None = None,
    default_salience: float = 6.0,
    max_participants: int | None = 5,
    preserve_provenance: bool = True,
    preserve_deceived: bool = True,
    epistemic_merge: Literal["union", "intersection"] = "union",
    anchor_step: Literal["last_target", "current"] = "last_target",
    force: bool = False,
    timeline_id: str | None = None,
) -> list[EpisodeRecord]:
    if not engine._policy.has_tier(6):
        return []
    if max_episodes_per_batch < 1:
        raise ValueError("max_episodes_per_batch must be >= 1.")
    if min_episodes < 1:
        raise ValueError("min_episodes must be >= 1.")

    tl = timeline_id or engine.active_timeline
    targets = _select_consolidation_targets(
        engine,
        step,
        before_step=before_step,
        max_salience=max_salience,
        predicate=predicate,
        chapter_id=chapter_id,
        preserve_provenance=preserve_provenance,
        preserve_deceived=preserve_deceived,
        timeline_id=tl,
    )
    if len(targets) < min_episodes:
        return []

    _assert_consolidation_branch_safe(engine, targets[0].step, tl, force=force)

    active_consolidator = resolve_episode_consolidator(engine, consolidator, client)
    entity_names = {
        e.entity_id: e.canonical_name
        for e in engine.store.get_all_active_entities(tl, step)
    }

    batches: list[list[EpisodeRecord]] = [
        targets[i : i + max_episodes_per_batch]
        for i in range(0, len(targets), max_episodes_per_batch)
    ]
    if len(batches) > 1 and len(batches[-1]) < min_episodes:
        batches[-2].extend(batches[-1])
        batches.pop()

    committed_records: list[EpisodeRecord] = []
    for batch in batches:
        result = active_consolidator.summarize_episodes(
            batch,
            entity_names=entity_names,
            chapter_id=chapter_id,
        )
        eff_salience = (
            float(salience)
            if salience is not None
            else (result.salience if result.salience is not None else float(default_salience))
        )
        committed = _commit_consolidated_batch(
            engine,
            step,
            batch,
            summary=result.summary,
            predicate=predicate,
            episode_id=None,
            chapter_id=chapter_id,
            salience=eff_salience,
            location_id=result.location_id,
            key_participant_ids=result.key_participant_ids or None,
            max_participants=max_participants,
            epistemic_merge=epistemic_merge,
            anchor_step=anchor_step,
            force=force,
            timeline_id=tl,
        )
        committed_records.append(committed)
    return committed_records


def consolidate_chapter_impl(
    engine: CanonArcEngine,
    step: int,
    chapter_id: str,
    *,
    before_step: int | None = None,
    consolidator: EpisodeConsolidator | None = None,
    client: JsonCompletionClient | None = None,
    max_salience: float = 4.0,
    min_episodes: int = 3,
    max_episodes_per_batch: int = 30,
    predicate: str = "consolidated_summary",
    salience: float | None = None,
    default_salience: float = 6.0,
    max_participants: int | None = 5,
    preserve_provenance: bool = True,
    preserve_deceived: bool = True,
    epistemic_merge: Literal["union", "intersection"] = "union",
    anchor_step: Literal["last_target", "current"] = "last_target",
    force: bool = False,
    timeline_id: str | None = None,
) -> list[EpisodeRecord]:
    if not chapter_id or not chapter_id.strip():
        raise ValueError("chapter_id must be a non-empty string.")
    eff_before_step = (step + 1) if before_step is None else before_step
    return consolidate_episodes_auto_impl(
        engine,
        step,
        before_step=eff_before_step,
        consolidator=consolidator,
        client=client,
        max_salience=max_salience,
        min_episodes=min_episodes,
        max_episodes_per_batch=max_episodes_per_batch,
        predicate=predicate,
        chapter_id=chapter_id,
        salience=salience,
        default_salience=default_salience,
        max_participants=max_participants,
        preserve_provenance=preserve_provenance,
        preserve_deceived=preserve_deceived,
        epistemic_merge=epistemic_merge,
        anchor_step=anchor_step,
        force=force,
        timeline_id=timeline_id,
    )
