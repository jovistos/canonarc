from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ..models import AliasRecord, DirectedEdgeRecord, EntityRecord, ResolvedMention
from ..presets import DEFAULT_SCHEMA_POLICY, SchemaPolicy


def resolve_speaker_voice_targets(
    *,
    real_speaker_id: str,
    is_masked_speaker: bool,
    edges: Sequence[DirectedEdgeRecord],
    in_scene_ids: set[str],
    resolved_mentions: Sequence[ResolvedMention],
    text: str,
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
    voice_constraints: dict[str, str] | None = None,
) -> list[tuple[str | None, dict[str, str]]]:
    """Resolves ordered (addressee_id, constraints) dyadic voice targets for a speaker."""
    extra_constraints = {
        k: str(v) for k, v in (voice_constraints or {}).items() if str(v).strip()
    }
    outgoing = [
        e
        for e in edges
        if e.source_id == real_speaker_id
        and (not is_masked_speaker or e.relation_type == policy.facade_relation)
    ]
    if not outgoing:
        return [(None, dict(extra_constraints))]

    in_scene_edges = [
        e
        for e in outgoing
        if e.target_id in in_scene_ids and e.target_id != real_speaker_id
    ]
    active_pool = in_scene_edges if in_scene_edges else outgoing

    layer_ranks = {rel: idx + 1 for idx, rel in enumerate(policy.edge_layer_order)}

    def _layer_rank(rel_type: str) -> int:
        return layer_ranks.get(rel_type, 0)

    by_target: dict[str, list[DirectedEdgeRecord]] = {}
    for edge in active_pool:
        by_target.setdefault(edge.target_id, []).append(edge)

    text_lower = text.lower()
    mention_pos: dict[str, int] = {}
    for idx, m in enumerate(resolved_mentions):
        if m.entity_id in mention_pos:
            continue
        pos = text_lower.find(m.surface_text.lower()) if text_lower else -1
        mention_pos[m.entity_id] = pos if pos >= 0 else (1_000_000 + idx)

    constraint_keys = set(policy.hard_filter_keys) | set(policy.soft_boost_weights.keys())
    resolved_targets: list[tuple[tuple[int, int, str], str | None, dict[str, str]]] = []
    for target_id, target_edges in by_target.items():
        target_edges_sorted = sorted(
            target_edges,
            key=lambda e: (_layer_rank(e.relation_type), e.valid_from_step),
        )
        merged_constraints: dict[str, str] = {}
        max_step = 0
        for e in target_edges_sorted:
            for ck in constraint_keys:
                if e.state.get(ck):
                    merged_constraints[ck] = str(e.state[ck])
            max_step = max(max_step, e.valid_from_step)
        if extra_constraints:
            merged_constraints.update(extra_constraints)

        sort_key = (mention_pos.get(target_id, 2_000_000), -max_step, target_id)
        resolved_targets.append((sort_key, target_id, merged_constraints))

    resolved_targets.sort(key=lambda item: item[0])
    return [(addr, constraints) for _, addr, constraints in resolved_targets]


def project_pov_state(
    *,
    scene_entities: Sequence[EntityRecord],
    scene_edges: Sequence[DirectedEdgeRecord],
    all_active_edges: Sequence[DirectedEdgeRecord],
    pov_set: set[str],
    pov_mode: Literal["any", "all"] = "any",
    pov_known_episode_ids: set[str],
    disguise_masks: dict[str, AliasRecord],
    pierced_disguises: dict[str, AliasRecord],
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
) -> tuple[list[EntityRecord], list[DirectedEdgeRecord]]:
    """Projects entities and edges through the active POV's epistemic lens."""
    belief_overrides: dict[str, dict[str, Any]] = {}
    if pov_set:
        pov_edges = [
            edge
            for edge in all_active_edges
            if edge.relation_type == policy.belief_relation and edge.source_id in pov_set
        ]
        pov_edges.sort(key=lambda e: (e.source_id, e.valid_from_step))
        if len(pov_set) > 1 and pov_mode == "all":
            per_pov_beliefs: dict[str, dict[str, dict[str, Any]]] = {pid: {} for pid in pov_set}
            for edge in pov_edges:
                per_pov_beliefs[edge.source_id].setdefault(edge.target_id, {}).update(edge.state)
            common_targets = set.intersection(
                *(set(per_pov_beliefs[pid].keys()) for pid in pov_set)
            )
            for target_id in common_targets:
                first_pid = next(iter(pov_set))
                shared_state: dict[str, Any] = {}
                for k, v in per_pov_beliefs[first_pid][target_id].items():
                    if all(
                        k in per_pov_beliefs[pid][target_id]
                        and per_pov_beliefs[pid][target_id][k] == v
                        for pid in pov_set
                    ):
                        shared_state[k] = v
                if shared_state:
                    belief_overrides[target_id] = shared_state
        else:
            for edge in pov_edges:
                belief_overrides.setdefault(edge.target_id, {}).update(edge.state)

    facade_pairs: set[tuple[str, str]] = {
        (edge.source_id, edge.target_id)
        for edge in scene_edges
        if edge.relation_type == policy.facade_relation
    }

    is_multipov_all = len(pov_set) > 1 and pov_mode == "all"
    projected_entities: list[EntityRecord] = []
    for ent in scene_entities:
        ent_copy = ent.model_copy(deep=True)
        if ent_copy.entity_id in pierced_disguises:
            alias_rec = pierced_disguises[ent_copy.entity_id]
            ent_copy.disguised_as = alias_rec.persona_name or alias_rec.alias

        if pov_set:
            if ent_copy.entity_id not in pov_set or is_multipov_all:
                ent_copy.private_state = {}
                if (
                    ent_copy.caused_by_episode_id
                    and ent_copy.caused_by_episode_id not in pov_known_episode_ids
                ):
                    ent_copy.caused_by_episode_id = None
                    ent_copy.reason = None
            if ent_copy.entity_id in belief_overrides:
                ent_copy.state = {**ent_copy.state, **belief_overrides[ent_copy.entity_id]}

        projected_entities.append(ent_copy)

    for real_id, alias_rec in disguise_masks.items():
        persona_label = alias_rec.persona_name or alias_rec.alias
        projected_entities.append(
            EntityRecord(
                entity_id=f"masked:{real_id}",
                kind=policy.masked_entity_kind,
                canonical_name=persona_label,
                state={policy.entity_disguise_key: persona_label},
            )
        )

    projected_edges: list[DirectedEdgeRecord] = []
    for edge in scene_edges:
        is_self_edge = edge.source_id in pov_set and not is_multipov_all
        if (
            pov_set
            and edge.relation_type in policy.edge_layer_order
            and edge.relation_type != policy.facade_relation
            and not is_self_edge
            and (edge.source_id, edge.target_id) in facade_pairs
        ):
            continue
        edge_copy = edge.model_copy(deep=True)
        if (
            pov_set
            and not is_self_edge
            and edge_copy.caused_by_episode_id
            and edge_copy.caused_by_episode_id not in pov_known_episode_ids
        ):
            edge_copy.caused_by_episode_id = None
            edge_copy.reason = None
        projected_edges.append(edge_copy)

    return projected_entities, projected_edges


def apply_epistemic_disguises(
    *,
    raw_mentions: Sequence[ResolvedMention],
    aliases: Sequence[AliasRecord],
    pov_set: set[str] | None = None,
    pov_mode: Literal["any", "all"] = "any",
) -> tuple[list[ResolvedMention], dict[str, AliasRecord], dict[str, AliasRecord]]:
    """Applies POV-aware disguise masking & piercing on top of pluggable mention matches."""
    resolved: list[ResolvedMention] = []
    seen_entities: set[str] = set()
    disguise_masks: dict[str, AliasRecord] = {}
    pierced_disguises: dict[str, AliasRecord] = {}

    alias_by_key: dict[tuple[str, str], AliasRecord] = {
        (a.entity_id, a.alias.lower()): a for a in aliases if a.alias
    }
    alias_by_surface: dict[str, AliasRecord] = {
        a.alias.lower(): a for a in aliases if a.alias
    }

    ordered_mentions = sorted(
        raw_mentions,
        key=lambda m: (0 if (m.matched_via == "alias" or m.matched_source == "alias") else 1),
    )

    for m in ordered_mentions:
        is_alias_match = m.matched_via == "alias" or m.matched_source == "alias"
        lookup_key = (m.matched_phrase or m.surface_text).strip().lower()
        alias_rec = (
            alias_by_key.get((m.entity_id, lookup_key)) or alias_by_surface.get(lookup_key)
            if is_alias_match
            else None
        )
        if alias_rec is not None and alias_rec.kind == "disguise":
            ent_id = alias_rec.entity_id
            if alias_rec.known_to is not None and pov_set:
                known_set = set(alias_rec.known_to or ())
                pierced = (
                    pov_set.issubset(known_set)
                    if pov_mode == "all"
                    else bool(pov_set & known_set)
                )
                if not pierced:
                    masked_id = f"masked:{ent_id}"
                    if masked_id not in seen_entities:
                        seen_entities.add(masked_id)
                        disguise_masks[ent_id] = alias_rec
                        resolved.append(
                            ResolvedMention(
                                surface_text=m.surface_text,
                                entity_id=masked_id,
                                matched_via="alias",
                                confidence=m.confidence,
                                matched_source="alias",
                                matched_phrase=alias_rec.alias.lower(),
                                alias_kind="disguise",
                                lang=alias_rec.lang,
                            )
                        )
                    continue
            pierced_disguises[ent_id] = alias_rec

        if m.entity_id in disguise_masks:
            continue

        if m.entity_id not in seen_entities:
            seen_entities.add(m.entity_id)
            resolved.append(m)

    return resolved, disguise_masks, pierced_disguises
