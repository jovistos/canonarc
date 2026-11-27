from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..models import AliasRecord, EntityRecord, EpisodeRecord, StepContext, StepDelta
from .traces import KnownEntityScope

if TYPE_CHECKING:
    from .core import CanonArcEngine


def format_compact_entity_roster(
    engine: CanonArcEngine,
    ctx: StepContext,
    *,
    seen_entity_ids: set[str],
    scope: KnownEntityScope,
    max_roster_entities: int | None = None,
) -> tuple[str, str] | None:
    """Builds a terse cast/entity roster for out-of-scene active entities on ctx.timeline_id."""
    if scope == "scene":
        return None
    if max_roster_entities is not None and max_roster_entities < 0:
        raise ValueError("max_roster_entities must be non-negative.")

    cutoff_step = max(0, ctx.step - 1)
    active_entities = engine.store.get_all_active_entities(ctx.timeline_id, cutoff_step)
    candidates: list[EntityRecord] = []
    for e in active_entities:
        if e.entity_id.startswith("masked:") or e.entity_id in seen_entity_ids:
            continue
        if scope == "characters_compact" and e.kind != "character":
            continue
        candidates.append(e)

    if not candidates or max_roster_entities == 0:
        return None

    if max_roster_entities is not None and len(candidates) > max_roster_entities:
        candidates.sort(key=lambda e: (-e.revision_count, -e.valid_from_step, e.entity_id))
        candidates = candidates[:max_roster_entities]

    candidates.sort(key=lambda e: (e.valid_from_step, e.entity_id))

    def _entry_str(ent: EntityRecord) -> str:
        names: list[str] = []
        canon = ent.canonical_name.strip()
        if canon:
            names.append(canon)
        target_name = ent.base_attributes.get("target_name")
        if isinstance(target_name, str) and target_name.strip():
            tname = target_name.strip()
            if not any(n.lower() == tname.lower() for n in names):
                names.append(tname)
        label = "/".join(names) if names else ent.entity_id
        return f"{ent.entity_id}={label}"

    if scope == "characters_compact":
        header = (
            "### Other Known Characters (reference only — NOT necessarily in this scene; "
            "reuse these exact entity_ids if any reappear under a new name or epithet):"
        )
        body = ", ".join(_entry_str(e) for e in candidates)
        return header, body

    grouped: dict[str, list[EntityRecord]] = {}
    for e in candidates:
        grouped.setdefault(e.kind or "entity", []).append(e)
    kind_order = sorted(grouped.keys(), key=lambda k: (0 if k == "character" else 1, k))
    lines = [
        f"{kind}: {', '.join(_entry_str(e) for e in grouped[kind])}"
        for kind in kind_order
    ]
    header = (
        "### Other Known Entities (reference only — NOT necessarily in this scene; "
        "reuse these exact entity_ids if any reappear under a new name or epithet):"
    )
    return header, "\n".join(lines)


def build_extraction_messages_impl(
    engine: CanonArcEngine,
    ctx: StepContext,
    scene_text: str,
    *,
    prev_text: str = "",
    next_text: str = "",
    compact_schema: bool = False,
    known_entity_scope: KnownEntityScope = "scene",
    max_roster_entities: int | None = None,
) -> tuple[str, str]:
    """Builds prefix-cache-friendly (system_prompt, user_prompt) extraction messages."""
    if known_entity_scope not in ("scene", "characters_compact", "all_compact"):
        raise ValueError(
            f"Unsupported known_entity_scope: {known_entity_scope!r}. "
            "Expected 'scene', 'characters_compact', or 'all_compact'."
        )

    policy = engine._policy
    raw_schema = StepDelta.model_json_schema()
    props = raw_schema.get("properties", {})
    defs = raw_schema.get("$defs", {})
    if not policy.has_tier(6):
        for k in ("episodes", "inform", "is_flashback", "story_time", "chapter_id", "scene_id"):
            props.pop(k, None)
        for d in ("EpisodeRecord", "InformDelta"):
            defs.pop(d, None)
    if not policy.has_tier(2):
        props.pop("edges", None)
        defs.pop("EdgeDelta", None)

    schema_json = (
        json.dumps(raw_schema, separators=(",", ":"))
        if compact_schema
        else json.dumps(raw_schema, indent=2)
    )
    system_prompt = f"{engine.preset.extraction_instructions()}\n\n### Target StepDelta JSON Schema:\n\n{schema_json}"

    seen_entity_ids: set[str] = set()
    entity_lines: list[str] = []
    for e in (*ctx.entities, *ctx.related_entities):
        if e.entity_id.startswith("masked:") or e.entity_id in seen_entity_ids:
            continue
        seen_entity_ids.add(e.entity_id)
        entity_lines.append(f"- {e.entity_id} (name: {e.canonical_name}, kind: {e.kind})")
    episode_lines = (
        [
            f"- {ep.episode_id} (step {ep.step}): {ep.summary}"
            for ep in ctx.episodes
            if ep.episode_id
        ]
        if policy.has_tier(6)
        else []
    )

    user_parts = [
        "### Known Entity IDs (use exact entity_id values, not display names):",
        "\n".join(entity_lines) if entity_lines else "(none)",
    ]
    compact_roster = format_compact_entity_roster(
        engine,
        ctx,
        seen_entity_ids=seen_entity_ids,
        scope=known_entity_scope,
        max_roster_entities=max_roster_entities,
    )
    if compact_roster is not None:
        roster_header, roster_body = compact_roster
        user_parts.append(roster_header)
        user_parts.append(roster_body)
    if episode_lines:
        user_parts.append("### Recent Episode IDs (for caused_by_episode_id or InformDelta):")
        user_parts.append("\n".join(episode_lines))
    has_halo = bool(prev_text.strip() or next_text.strip())
    if prev_text.strip():
        user_parts.append(
            "### Preceding Context (Already Committed — Use ONLY to resolve pronouns like he/she/they; DO NOT re-extract events):"
        )
        user_parts.append(prev_text.strip())
    if has_halo:
        user_parts.append(
            "### Target Scene Chunk (EXTRACT ONLY state changes and episodes occurring in THIS text):"
        )
    else:
        user_parts.append("### Scene Text to Extract From:")
    user_parts.append(scene_text)
    if next_text.strip():
        user_parts.append(
            "### Following Context (Read-Only Lookahead — Use ONLY to identify speakers/addressees of trailing dialogue; DO NOT extract future events):"
        )
        user_parts.append(next_text.strip())
    return system_prompt, "\n\n".join(user_parts)


def _build_entity_lookup(
    entities: Sequence[Any],
    aliases: Sequence[Any],
    delta: StepDelta,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for e in entities:
        lookup[e.entity_id.lower()] = e.entity_id
        if e.canonical_name:
            lookup[e.canonical_name.lower()] = e.entity_id
        target_name = e.base_attributes.get("target_name")
        if isinstance(target_name, str) and target_name.strip():
            lookup.setdefault(target_name.strip().lower(), e.entity_id)
        merged_from = e.base_attributes.get("merged_from")
        if isinstance(merged_from, list):
            for old_id in merged_from:
                if isinstance(old_id, str) and old_id.strip():
                    lookup.setdefault(old_id.strip().lower(), e.entity_id)
    for a in aliases:
        if a.alias and a.kind != "disguise":
            lookup.setdefault(a.alias.lower(), a.entity_id)

    for ed in delta.entities:
        if ed.remove:
            continue
        lookup.setdefault(ed.entity_id.lower(), ed.entity_id)
        if ed.canonical_name:
            lookup.setdefault(ed.canonical_name.lower(), ed.entity_id)
        ed_target = ed.base_attributes.get("target_name")
        if isinstance(ed_target, str) and ed_target.strip():
            lookup.setdefault(ed_target.strip().lower(), ed.entity_id)
        for alias_item in ed.add_aliases:
            if isinstance(alias_item, AliasRecord):
                if alias_item.alias and alias_item.kind != "disguise":
                    lookup.setdefault(alias_item.alias.lower(), ed.entity_id)
            elif isinstance(alias_item, str) and alias_item.strip():
                lookup.setdefault(alias_item.strip().lower(), ed.entity_id)
    return lookup


def _normalize_episodes(
    episodes: Sequence[EpisodeRecord],
    lookup: dict[str, str],
    *,
    timeline_id: str,
    step: int | None,
) -> tuple[list[EpisodeRecord], dict[str, str]]:
    def _norm_id(raw: str) -> str:
        return lookup.get(raw.strip().lower(), raw)

    ep_id_map: dict[str, str] = {}
    norm_episodes: list[EpisodeRecord] = []
    for idx, ep in enumerate(episodes):
        old_id = ep.episode_id
        if old_id:
            new_id = old_id
            ep_id_map[old_id] = new_id
        else:
            digest = hashlib.sha256(
                f"{timeline_id}:{step if step is not None else 0}:{idx}:{ep.predicate}:{ep.summary}".encode()
            ).hexdigest()[:8]
            new_id = f"ep_{step if step is not None else 0}_{digest}"
        ep_updates: dict[str, Any] = {
            "episode_id": new_id,
            "participants": {_norm_id(k): v for k, v in ep.participants.items()},
            "witnesses": [_norm_id(w) for w in ep.witnesses],
            "known_by": {_norm_id(k): v for k, v in ep.known_by.items()},
            "believed_summaries": {_norm_id(k): v for k, v in ep.believed_summaries.items()},
            "location_id": _norm_id(ep.location_id) if ep.location_id else None,
        }
        if step is not None:
            ep_updates["step"] = step
        norm_episodes.append(ep.model_copy(update=ep_updates))
    return norm_episodes, ep_id_map


def _normalize_entity_state(
    state_dict: dict[str, Any],
    lookup: dict[str, str],
    location_ids: set[str],
) -> dict[str, Any]:
    if not state_dict:
        return state_dict

    def _norm_id(raw: str) -> str:
        return lookup.get(raw.strip().lower(), raw)

    out = dict(state_dict)
    if "location_id" in out and isinstance(out["location_id"], str):
        out["location_id"] = _norm_id(out["location_id"])
    else:
        for alt_loc_key in ("location", "current_location"):
            if alt_loc_key in out and isinstance(out[alt_loc_key], str):
                resolved_loc = _norm_id(out[alt_loc_key])
                if resolved_loc.startswith("loc_") or resolved_loc in location_ids:
                    out.pop(alt_loc_key, None)
                    out["location_id"] = resolved_loc
                    break
    if "mood" not in out:
        for alt_mood_key in ("emotional_state", "emotion"):
            if (
                alt_mood_key in out
                and isinstance(out[alt_mood_key], str)
                and out[alt_mood_key].strip()
            ):
                out["mood"] = out[alt_mood_key].strip()
                break
    return out


def _normalize_and_merge_entity_deltas(
    entity_deltas: Sequence[Any],
    lookup: dict[str, str],
    location_ids: set[str],
    ep_id_map: dict[str, str],
) -> list[Any]:
    merged_by_id: dict[str, Any] = {}
    for ed in entity_deltas:
        norm_ed = ed.model_copy(
            update={
                "entity_id": lookup.get(ed.entity_id.strip().lower(), ed.entity_id),
                "state": _normalize_entity_state(ed.state, lookup, location_ids),
                "caused_by_episode_id": ep_id_map.get(
                    ed.caused_by_episode_id, ed.caused_by_episode_id
                )
                if ed.caused_by_episode_id
                else None,
            }
        )
        existing = merged_by_id.get(norm_ed.entity_id)
        if existing is None or norm_ed.remove or existing.remove:
            merged_by_id[norm_ed.entity_id] = norm_ed
        else:
            merged_by_id[norm_ed.entity_id] = existing.model_copy(
                update={
                    "kind": norm_ed.kind or existing.kind,
                    "canonical_name": norm_ed.canonical_name or existing.canonical_name,
                    "base_attributes": {**existing.base_attributes, **norm_ed.base_attributes},
                    "state": {**existing.state, **norm_ed.state},
                    "private_state": {**existing.private_state, **norm_ed.private_state},
                    "add_aliases": [*existing.add_aliases, *norm_ed.add_aliases],
                    "remove_aliases": [*existing.remove_aliases, *norm_ed.remove_aliases],
                    "caused_by_episode_id": norm_ed.caused_by_episode_id
                    or existing.caused_by_episode_id,
                    "reason": norm_ed.reason or existing.reason,
                }
            )
    return list(merged_by_id.values())


def normalize_delta_impl(
    engine: CanonArcEngine,
    delta: StepDelta,
    *,
    step: int | None = None,
    timeline_id: str | None = None,
) -> StepDelta:
    """Normalizes canonical names/aliases into entity_ids and resolves intra-step episode IDs."""
    tl = timeline_id or engine.active_timeline
    entities = engine.store.get_all_active_entities(tl, step)
    aliases = engine.store.get_all_active_aliases(tl, step)

    lookup = _build_entity_lookup(entities, aliases, delta)
    norm_episodes, ep_id_map = _normalize_episodes(
        delta.episodes, lookup, timeline_id=tl, step=step
    )
    location_ids = {e.entity_id for e in entities if e.kind == "location"}
    norm_entities = _normalize_and_merge_entity_deltas(
        delta.entities, lookup, location_ids, ep_id_map
    )

    def _norm_id(raw: str) -> str:
        return lookup.get(raw.strip().lower(), raw)

    norm_edges = [
        rd.model_copy(
            update={
                "source_id": _norm_id(rd.source_id),
                "target_id": _norm_id(rd.target_id),
                "caused_by_episode_id": ep_id_map.get(
                    rd.caused_by_episode_id, rd.caused_by_episode_id
                )
                if rd.caused_by_episode_id
                else None,
            }
        )
        for rd in delta.edges
    ]
    norm_inform = [
        inf.model_copy(
            update={
                "entity_id": _norm_id(inf.entity_id),
                "episode_id": ep_id_map.get(inf.episode_id, inf.episode_id),
            }
        )
        for inf in delta.inform
    ]

    return StepDelta(
        entities=norm_entities,
        edges=norm_edges,
        episodes=norm_episodes,
        inform=norm_inform,
        is_flashback=delta.is_flashback,
        story_time=delta.story_time,
        chapter_id=delta.chapter_id,
        scene_id=delta.scene_id,
    )
