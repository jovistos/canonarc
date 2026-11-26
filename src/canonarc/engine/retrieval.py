from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..episodic import DEFAULT_EPISODIC_POLICY
from ..models import (
    DirectedEdgeRecord,
    EpisodeRecord,
    StepContext,
    VoiceExemplar,
)
from .epistemics import (
    apply_epistemic_disguises,
    project_pov_state,
    resolve_speaker_voice_targets,
)

if TYPE_CHECKING:
    from .core import CanonArcEngine


def retrieve_context_impl(
    engine: CanonArcEngine,
    step: int,
    *,
    text: str = "",
    prev_text: str = "",
    mentions: Sequence[str] = (),
    top_k_voice: int | None = None,
    top_k_episodes: int | None = None,
    pov_entity_ids: Sequence[str] | None = None,
    pov_mode: Literal["any", "all"] = "any",
    edge_types: Sequence[str] | None = None,
    location_id: str | None = None,
    inclusive: bool = False,
    timeline_id: str | None = None,
    fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only",
    story_time: float | str | None = None,
    chapter_id: str | None = None,
    scene_id: str | None = None,
    voice_constraints: dict[str, str] | None = None,
    lang: str | None = None,
) -> StepContext:
    tl = timeline_id or engine.active_timeline
    policy = engine._policy
    cutoff_step = step if inclusive else (step - 1)

    has_edges = policy.has_tier(2)
    has_state = policy.has_tier(3)
    has_voice = policy.has_tier(5)
    has_episodes = policy.has_tier(6)
    eff_top_k_voice = 0 if not has_voice else (3 if top_k_voice is None else top_k_voice)
    eff_top_k_episodes = 0 if not has_episodes else (5 if top_k_episodes is None else top_k_episodes)
    eff_edge_types: Sequence[str] | None = () if not has_edges else edge_types

    # 1. Single-pass fetch of active entities, aliases, and edges
    get_graph_fn = getattr(engine.store, "get_active_graph_state", None)
    if callable(get_graph_fn):
        active_entities, active_aliases, active_edges = get_graph_fn(
            tl, cutoff_step, edge_types=eff_edge_types
        )
    else:
        active_entities = engine.store.get_all_active_entities(tl, cutoff_step)
        active_aliases = engine.store.get_all_active_aliases(tl, cutoff_step)
        if has_edges:
            active_edges = engine.store.get_all_active_edges(tl, cutoff_step)
            if eff_edge_types is not None:
                active_edges = [e for e in active_edges if e.relation_type in eff_edge_types]
        else:
            active_edges = []

    # 2. Entity resolution via pluggable MentionResolver + POV epistemic disguise projection
    pov_set = set(pov_entity_ids) if pov_entity_ids else set()
    raw_mentions = engine._resolve_mentions(
        text=text,
        explicit_mentions=mentions,
        entities=active_entities,
        aliases=active_aliases,
        lang=lang,
    )
    resolved_mentions, disguise_masks, pierced_disguises = apply_epistemic_disguises(
        raw_mentions=raw_mentions,
        aliases=active_aliases,
        pov_set=pov_set,
        pov_mode=pov_mode,
    )

    entity_by_id = {e.entity_id: e for e in active_entities}
    primary_char_ids = {
        m.entity_id
        for m in resolved_mentions
        if m.entity_id.startswith("masked:")
        or ((ent := entity_by_id.get(m.entity_id)) is None or ent.kind == "character")
    }
    if (has_edges or has_state) and prev_text.strip() and len(primary_char_ids) < 2:
        halo_raw = engine._resolve_mentions(
            text=prev_text,
            explicit_mentions=(),
            entities=active_entities,
            aliases=active_aliases,
            lang=lang,
        )
        halo_mentions, halo_masks, halo_pierced = apply_epistemic_disguises(
            raw_mentions=halo_raw,
            aliases=active_aliases,
            pov_set=pov_set,
            pov_mode=pov_mode,
        )
        seen_resolved_ids = {m.entity_id for m in resolved_mentions}
        for m in halo_mentions:
            real_id = m.entity_id.removeprefix("masked:")
            if (
                m.entity_id in seen_resolved_ids
                or real_id in seen_resolved_ids
                or f"masked:{real_id}" in seen_resolved_ids
            ):
                continue
            seen_resolved_ids.add(m.entity_id)
            resolved_mentions.append(m)
            if real_id in halo_masks:
                disguise_masks.setdefault(real_id, halo_masks[real_id])
        for k, v in halo_pierced.items():
            if k not in disguise_masks:
                pierced_disguises.setdefault(k, v)

    resolved_entity_ids = {m.entity_id for m in resolved_mentions}
    masked_real_ids = set(disguise_masks.keys())

    # 3. Separate co-present scene entities from 1-hop social graph neighbors
    present_entity_ids = {
        eid for eid in resolved_entity_ids if not eid.startswith("masked:")
    }
    active_location_ids: set[str] = set()
    if location_id is not None:
        active_location_ids.add(location_id)
    for ent in active_entities:
        if ent.entity_id in present_entity_ids and ent.kind == "location":
            active_location_ids.add(ent.entity_id)
    if active_location_ids:
        for ent in active_entities:
            if ent.entity_id in masked_real_ids:
                continue
            ent_loc = ent.state.get("location_id") or ent.base_attributes.get("location_id")
            if ent_loc in active_location_ids:
                present_entity_ids.add(ent.entity_id)

    if not present_entity_ids and not disguise_masks:
        if fallback_on_empty_mentions == "all":
            present_entity_ids = {
                e.entity_id for e in active_entities if e.entity_id not in masked_real_ids
            }
        elif fallback_on_empty_mentions == "pov_only" and pov_set:
            present_entity_ids = {
                eid for eid in pov_set if eid not in masked_real_ids
            }

    in_scene_edges: list[DirectedEdgeRecord] = []
    background_edges: list[DirectedEdgeRecord] = []
    neighbor_entity_ids: set[str] = set()
    for edge in active_edges:
        if edge.relation_type == policy.belief_relation:
            continue
        if edge.source_id in masked_real_ids or edge.target_id in masked_real_ids:
            continue
        src_present = edge.source_id in present_entity_ids
        tgt_present = edge.target_id in present_entity_ids
        if src_present and tgt_present:
            in_scene_edges.append(edge)
        elif src_present or tgt_present:
            background_edges.append(edge)
            if not src_present:
                neighbor_entity_ids.add(edge.source_id)
            if not tgt_present:
                neighbor_entity_ids.add(edge.target_id)

    scene_entities = [e for e in active_entities if e.entity_id in present_entity_ids]
    neighbor_entities = [e for e in active_entities if e.entity_id in neighbor_entity_ids]

    pov_known_episode_ids: set[str] = set()
    fallback_lineage_episodes: list[tuple[EpisodeRecord, bytes]] | None = None
    if pov_set and has_episodes:
        provenance_ep_ids = {
            eid
            for eid in (
                *(e.caused_by_episode_id for e in scene_entities),
                *(e.caused_by_episode_id for e in neighbor_entities),
                *(e.caused_by_episode_id for e in in_scene_edges),
                *(e.caused_by_episode_id for e in background_edges),
            )
            if eid
        }
        if provenance_ep_ids:
            get_known_fn = getattr(engine.store, "get_known_episode_ids", None)
            if callable(get_known_fn):
                pov_known_episode_ids = get_known_fn(
                    tl,
                    cutoff_step,
                    pov_entity_ids or (),
                    pov_mode=pov_mode,
                    episode_ids=sorted(provenance_ep_ids),
                )
            else:
                fallback_lineage_episodes = engine.store.get_lineage_episodes(tl, cutoff_step)
                if pov_mode == "all":
                    pov_known_episode_ids = {
                        ep.episode_id
                        for ep, _ in fallback_lineage_episodes
                        if all(ep.known_by.get(pid) in ("direct", "told") for pid in pov_set)
                    }
                else:
                    pov_known_episode_ids = {
                        ep.episode_id
                        for ep, _ in fallback_lineage_episodes
                        if any(ep.known_by.get(pid) in ("direct", "told") for pid in pov_set)
                    }

    projected_entities, projected_edges = project_pov_state(
        scene_entities=scene_entities,
        scene_edges=in_scene_edges,
        all_active_edges=active_edges,
        pov_set=pov_set,
        pov_mode=pov_mode,
        pov_known_episode_ids=pov_known_episode_ids,
        disguise_masks=disguise_masks,
        pierced_disguises=pierced_disguises,
        policy=policy,
    )
    projected_related_entities, projected_background_edges = project_pov_state(
        scene_entities=neighbor_entities,
        scene_edges=background_edges,
        all_active_edges=active_edges,
        pov_set=pov_set,
        pov_mode=pov_mode,
        pov_known_episode_ids=pov_known_episode_ids,
        disguise_masks={},
        pierced_disguises=pierced_disguises,
        policy=policy,
    )

    # 4. Tier 5 Voice Exemplars retrieval
    voice_exemplars: list[VoiceExemplar] = []
    speaker_ids = resolved_entity_ids | {f"masked:{rid}" for rid in masked_real_ids}
    speakers = (
        [e for e in projected_entities if e.entity_id in speaker_ids]
        if speaker_ids
        else projected_entities
    )
    if eff_top_k_voice > 0 and speakers:
        in_scene_ids = resolved_entity_ids or {e.entity_id for e in projected_entities}
        for speaker in speakers:
            is_masked_speaker = speaker.entity_id.startswith("masked:")
            real_speaker_id = (
                speaker.entity_id.removeprefix("masked:")
                if is_masked_speaker
                else speaker.entity_id
            )
            edge_pool = active_edges if is_masked_speaker else (in_scene_edges + background_edges)
            voice_targets = resolve_speaker_voice_targets(
                real_speaker_id=real_speaker_id,
                is_masked_speaker=is_masked_speaker,
                edges=edge_pool,
                in_scene_ids=in_scene_ids,
                resolved_mentions=resolved_mentions,
                text=text,
                policy=policy,
                voice_constraints=voice_constraints,
            )

            candidates = engine.store.get_lineage_voice_exemplars(tl, cutoff_step, real_speaker_id)
            disguise_persona = (
                speaker.state.get(policy.entity_disguise_key) if is_masked_speaker else None
            )
            if disguise_persona:
                persona_candidates = [
                    (ex, emb)
                    for ex, emb in candidates
                    if ex.metadata.get(policy.exemplar_persona_key, "").lower()
                    == str(disguise_persona).lower()
                ]
                if persona_candidates:
                    candidates = persona_candidates

            ranked = engine.voice_bank.rank_exemplars_for_targets(
                query_text=engine._strip_text(text),
                candidates=candidates,
                targets=voice_targets,
                policy=policy,
                top_k=eff_top_k_voice,
            )
            if is_masked_speaker:
                ranked = [
                    ex.model_copy(update={"speaker_id": speaker.entity_id})
                    for ex in ranked
                ]
            voice_exemplars.extend(ranked)

    # 5. Tier 6 Episodic Memory retrieval
    episodes: list[EpisodeRecord] = []
    if eff_top_k_episodes > 0:
        pinned_episode_ids: set[str] = set()
        for ent in projected_entities:
            if ent.caused_by_episode_id:
                pinned_episode_ids.add(ent.caused_by_episode_id)
        for edge in projected_edges:
            if edge.caused_by_episode_id:
                pinned_episode_ids.add(edge.caused_by_episode_id)

        cue_entity_ids = (
            resolved_entity_ids
            if resolved_entity_ids
            else {e.entity_id for e in projected_entities}
        )
        if fallback_lineage_episodes is not None:
            candidate_episodes = fallback_lineage_episodes
        else:
            candidate_limit = max(eff_top_k_episodes * 20, 200)
            sql_cue_ids = sorted(cue_entity_ids | active_location_ids)
            clean_query = engine._strip_text(text)
            try:
                candidate_episodes = engine.store.get_lineage_episodes(
                    tl,
                    cutoff_step,
                    limit=candidate_limit,
                    pov_entity_ids=pov_entity_ids,
                    pov_mode=pov_mode,
                    cue_entity_ids=sql_cue_ids,
                    pinned_episode_ids=sorted(pinned_episode_ids),
                    query_text=clean_query,
                )
            except TypeError:
                candidate_episodes = engine.store.get_lineage_episodes(tl, cutoff_step)

        if candidate_episodes:
            episodic_policy = getattr(policy, "episodic_policy", DEFAULT_EPISODIC_POLICY)
            episodes = engine.voice_bank.rank_episodes(
                current_step=step,
                query_text=engine._strip_text(text),
                candidates=candidate_episodes,
                scene_entity_ids=cue_entity_ids,
                pinned_episode_ids=pinned_episode_ids,
                pov_entity_ids=pov_entity_ids,
                pov_mode=pov_mode,
                policy=episodic_policy,
                top_k=eff_top_k_episodes,
            )

    entity_names: dict[str, str] = {
        e.entity_id: e.canonical_name
        for e in (*projected_entities, *projected_related_entities)
    }
    referenced_ids: set[str] = set()
    for edge in (*projected_edges, *projected_background_edges):
        referenced_ids.add(edge.source_id)
        referenced_ids.add(edge.target_id)
    for ex in voice_exemplars:
        if ex.speaker_id:
            referenced_ids.add(ex.speaker_id)
        if ex.addressee_id:
            referenced_ids.add(ex.addressee_id)
    for ep in episodes:
        referenced_ids.update(ep.known_by.keys())
    for missing_id in sorted(referenced_ids - entity_names.keys()):
        if missing_id.startswith("masked:"):
            continue
        rec = engine.store.get_active_entity(
            missing_id, timeline_id=tl, cutoff_step=cutoff_step
        )
        if rec is not None:
            entity_names[missing_id] = rec.canonical_name

    return StepContext(
        timeline_id=tl,
        step=step,
        story_time=story_time,
        chapter_id=chapter_id,
        scene_id=scene_id,
        resolved_mentions=resolved_mentions,
        entities=projected_entities,
        active_edges=projected_edges,
        related_entities=projected_related_entities,
        background_edges=projected_background_edges,
        voice_exemplars=voice_exemplars,
        episodes=episodes,
        entity_names=entity_names,
    )
