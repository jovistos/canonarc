from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityMergeReport,
    EntityRecord,
    StepDelta,
)
from ..store import CausalStore
from ..store.protocols import fetch_active_graph_state

if TYPE_CHECKING:
    from .core import CanonArcEngine


@dataclass
class _AliasMigrationPlan:
    aliases_to_add: list[AliasRecord]
    dup_aliases_to_tombstone: list[str]
    other_alias_updates: dict[str, list[AliasRecord]]


@dataclass
class _AttributeMergePlan:
    merged_base_attrs: dict[str, Any]
    base_attrs_merged: list[str]
    merged_state: dict[str, Any]
    state_keys_merged: list[str]
    merged_private: dict[str, Any]


@dataclass
class _EdgeMigrationPlan:
    edges_to_write: list[EdgeDelta]
    edges_to_tombstone: list[DirectedEdgeRecord]
    edges_migrated: list[str]
    edges_merged_on_collision: list[str]
    edges_tombstoned_keys: list[str]
    self_loops_dropped: list[str]


def _is_empty_val(val: Any) -> bool:
    return val is None or (isinstance(val, str) and not val.strip())


def _remap_known_to(
    known_to: list[str] | None,
    dup_id: str,
    canon_id: str,
) -> list[str] | None:
    if known_to is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for eid in known_to:
        mapped = canon_id if eid == dup_id else eid
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _validate_merge_args(
    duplicate_id: str,
    canonical_id: str,
    conflict_strategy: Literal["canonical_wins", "duplicate_wins"],
) -> tuple[str, str]:
    dup_id = duplicate_id.strip()
    canon_id = canonical_id.strip()
    if not dup_id or not canon_id:
        raise ValueError("Both duplicate_id and canonical_id must be non-empty.")
    if dup_id == canon_id:
        raise ValueError(f"Cannot merge entity '{dup_id}' into itself.")
    if dup_id.startswith("masked:") or canon_id.startswith("masked:"):
        raise ValueError("Cannot merge synthetic 'masked:' disguise entity IDs.")
    if conflict_strategy not in ("canonical_wins", "duplicate_wins"):
        raise ValueError(
            f"Unsupported conflict_strategy: {conflict_strategy!r}. "
            "Expected 'canonical_wins' or 'duplicate_wins'."
        )
    return dup_id, canon_id


def _plan_alias_migration(
    dup_ent: EntityRecord,
    canon_ent: EntityRecord,
    active_aliases: Sequence[AliasRecord],
) -> _AliasMigrationPlan:
    dup_id = dup_ent.entity_id
    canon_id = canon_ent.entity_id
    canon_aliases = [a for a in active_aliases if a.entity_id == canon_id]
    dup_aliases = [a for a in active_aliases if a.entity_id == dup_id]
    existing_canon_alias_keys = {
        a.alias.strip().lower() for a in canon_aliases if a.alias.strip()
    }
    canon_name_lower = canon_ent.canonical_name.strip().lower()
    aliases_to_add: list[AliasRecord] = []

    def _queue_alias(rec: AliasRecord, *, allow_canonical: bool = False) -> None:
        clean = rec.alias.strip()
        if not clean:
            return
        key = clean.lower()
        if key in existing_canon_alias_keys:
            return
        if not allow_canonical and key == canon_name_lower:
            return
        existing_canon_alias_keys.add(key)
        aliases_to_add.append(
            rec.model_copy(
                update={
                    "alias": clean,
                    "entity_id": canon_id,
                    "known_to": _remap_known_to(rec.known_to, dup_id, canon_id),
                }
            )
        )

    _queue_alias(
        AliasRecord(alias=dup_ent.canonical_name, entity_id=canon_id, kind="permanent")
    )

    dup_target = dup_ent.base_attributes.get("target_name")
    if isinstance(dup_target, str) and dup_target.strip():
        raw_lang = dup_ent.base_attributes.get("target_lang") or dup_ent.base_attributes.get("lang")
        eff_lang = (
            raw_lang.strip().lower()
            if isinstance(raw_lang, str) and raw_lang.strip()
            else None
        )
        canon_target = canon_ent.base_attributes.get("target_name")
        canon_target_lower = (
            canon_target.strip().lower()
            if isinstance(canon_target, str) and canon_target.strip()
            else ""
        )
        if dup_target.strip().lower() != canon_target_lower:
            _queue_alias(
                AliasRecord(
                    alias=dup_target.strip(),
                    entity_id=canon_id,
                    kind="permanent",
                    lang=eff_lang,
                ),
                allow_canonical=eff_lang is not None,
            )

    for dup_alias in dup_aliases:
        _queue_alias(dup_alias, allow_canonical=dup_alias.lang is not None)

    other_alias_updates: dict[str, list[AliasRecord]] = {}
    for a in active_aliases:
        if a.entity_id in (dup_id, canon_id):
            continue
        if a.known_to and dup_id in a.known_to:
            other_alias_updates.setdefault(a.entity_id, []).append(
                a.model_copy(
                    update={"known_to": _remap_known_to(a.known_to, dup_id, canon_id)}
                )
            )

    return _AliasMigrationPlan(
        aliases_to_add=aliases_to_add,
        dup_aliases_to_tombstone=[a.alias for a in dup_aliases],
        other_alias_updates=other_alias_updates,
    )


def _merge_dict_fields(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    conflict_strategy: Literal["canonical_wins", "duplicate_wins"],
    skip_keys: frozenset[str] = frozenset(),
) -> list[str]:
    merged_keys: list[str] = []
    for k, v in source.items():
        if k in skip_keys or _is_empty_val(v):
            continue
        if conflict_strategy == "duplicate_wins" or k not in target or _is_empty_val(target[k]):
            if target.get(k) != v:
                target[k] = v
                merged_keys.append(k)
    return merged_keys


def _merge_entity_attributes(
    dup_ent: EntityRecord,
    canon_ent: EntityRecord,
    *,
    merge_attributes: bool,
    conflict_strategy: Literal["canonical_wins", "duplicate_wins"],
    has_tier3: bool,
) -> _AttributeMergePlan:
    merged_base_attrs: dict[str, Any] = dict(canon_ent.base_attributes)
    base_attrs_merged: list[str] = []
    merged_state: dict[str, Any] = dict(canon_ent.state)
    state_keys_merged: list[str] = []
    merged_private: dict[str, Any] = dict(canon_ent.private_state)

    if merge_attributes:
        base_attrs_merged = _merge_dict_fields(
            merged_base_attrs,
            dup_ent.base_attributes,
            conflict_strategy=conflict_strategy,
            skip_keys=frozenset({"merged_from"}),
        )
        if has_tier3:
            state_keys_merged = _merge_dict_fields(
                merged_state,
                dup_ent.state,
                conflict_strategy=conflict_strategy,
            )
            _merge_dict_fields(
                merged_private,
                dup_ent.private_state,
                conflict_strategy=conflict_strategy,
            )

    dup_id = dup_ent.entity_id
    canon_id = canon_ent.entity_id
    merged_from_list: list[str] = []
    for src_list in (
        canon_ent.base_attributes.get("merged_from"),
        dup_ent.base_attributes.get("merged_from"),
        [dup_id],
    ):
        if isinstance(src_list, list):
            for item in src_list:
                if (
                    isinstance(item, str)
                    and item.strip()
                    and item != canon_id
                    and item not in merged_from_list
                ):
                    merged_from_list.append(item)
    merged_base_attrs["merged_from"] = merged_from_list

    return _AttributeMergePlan(
        merged_base_attrs=merged_base_attrs,
        base_attrs_merged=base_attrs_merged,
        merged_state=merged_state,
        state_keys_merged=state_keys_merged,
        merged_private=merged_private,
    )


def _plan_location_ref_updates(
    dup_ent: EntityRecord,
    canon_id: str,
    active_entities: Sequence[EntityRecord],
    merged_state: dict[str, Any],
    *,
    has_tier3: bool,
    reason: str,
) -> tuple[list[EntityDelta], list[str]]:
    location_ref_deltas: list[EntityDelta] = []
    location_refs_updated: list[str] = []
    if dup_ent.kind != "location" or not has_tier3:
        return location_ref_deltas, location_refs_updated

    dup_id = dup_ent.entity_id
    for ent in active_entities:
        if ent.entity_id == dup_id or ent.state.get("location_id") != dup_id:
            continue
        location_refs_updated.append(ent.entity_id)
        if ent.entity_id == canon_id:
            merged_state["location_id"] = canon_id
        else:
            location_ref_deltas.append(
                EntityDelta(
                    entity_id=ent.entity_id,
                    state={"location_id": canon_id},
                    merge_state=True,
                    reason=reason,
                )
            )
    return location_ref_deltas, location_refs_updated


def _plan_edge_migration(
    dup_id: str,
    canon_id: str,
    active_edges: Sequence[DirectedEdgeRecord],
    *,
    conflict_strategy: Literal["canonical_wins", "duplicate_wins"],
    hard_keys: Sequence[str],
    reason: str,
    warnings_list: list[str],
) -> _EdgeMigrationPlan:
    plan = _EdgeMigrationPlan(
        edges_to_write=[],
        edges_to_tombstone=[],
        edges_migrated=[],
        edges_merged_on_collision=[],
        edges_tombstoned_keys=[],
        self_loops_dropped=[],
    )
    edge_index: dict[tuple[str, str, str], DirectedEdgeRecord] = {
        (e.source_id, e.target_id, e.relation_type): e for e in active_edges
    }
    sorted_hard_keys = sorted(set(hard_keys))

    for edge in active_edges:
        if edge.source_id != dup_id and edge.target_id != dup_id:
            continue
        old_key = CausalStore.edge_subject_key(edge.source_id, edge.target_id, edge.relation_type)
        plan.edges_to_tombstone.append(edge)
        plan.edges_tombstoned_keys.append(old_key)

        new_src = canon_id if edge.source_id == dup_id else edge.source_id
        new_tgt = canon_id if edge.target_id == dup_id else edge.target_id
        if new_src == new_tgt:
            plan.self_loops_dropped.append(old_key)
            continue

        new_key = CausalStore.edge_subject_key(new_src, new_tgt, edge.relation_type)
        existing_edge = edge_index.get((new_src, new_tgt, edge.relation_type))
        if existing_edge is None:
            plan.edges_to_write.append(
                EdgeDelta(
                    source_id=new_src,
                    target_id=new_tgt,
                    relation_type=edge.relation_type,
                    state=dict(edge.state),
                    merge=False,
                    caused_by_episode_id=edge.caused_by_episode_id,
                    reason=reason,
                )
            )
            plan.edges_migrated.append(new_key)
        else:
            for hk in sorted_hard_keys:
                dup_val = edge.state.get(hk)
                canon_val = existing_edge.state.get(hk)
                if (
                    dup_val is not None
                    and canon_val is not None
                    and str(dup_val).strip().lower() != str(canon_val).strip().lower()
                ):
                    winner = canon_val if conflict_strategy == "canonical_wins" else dup_val
                    warnings_list.append(
                        f"Edge collision on {new_key}: '{hk}' conflicted "
                        f"(canonical={canon_val!r} vs duplicate={dup_val!r}); kept {winner!r}."
                    )
            merged_edge_state = (
                {**edge.state, **existing_edge.state}
                if conflict_strategy == "canonical_wins"
                else {**existing_edge.state, **edge.state}
            )
            if merged_edge_state != existing_edge.state:
                plan.edges_to_write.append(
                    EdgeDelta(
                        source_id=new_src,
                        target_id=new_tgt,
                        relation_type=edge.relation_type,
                        state=merged_edge_state,
                        merge=False,
                        caused_by_episode_id=existing_edge.caused_by_episode_id
                        or edge.caused_by_episode_id,
                        reason=reason,
                    )
                )
            plan.edges_merged_on_collision.append(new_key)
    return plan


def _commit_merge_plan(
    engine: CanonArcEngine,
    *,
    dup_id: str,
    canon_id: str,
    timeline_id: str,
    step: int,
    reason: str,
    alias_plan: _AliasMigrationPlan,
    attr_plan: _AttributeMergePlan,
    location_ref_deltas: list[EntityDelta],
    edge_plan: _EdgeMigrationPlan,
    migrate_voice: bool,
    migrate_episodes: bool,
) -> tuple[int, int, int]:
    has_tier3 = engine._policy.has_tier(3)
    entity_deltas: list[EntityDelta] = [
        EntityDelta(
            entity_id=canon_id,
            base_attributes=attr_plan.merged_base_attrs,
            state=attr_plan.merged_state if has_tier3 else {},
            private_state=attr_plan.merged_private if has_tier3 else {},
            merge_state=not has_tier3,
            add_aliases=list(alias_plan.aliases_to_add),
            reason=reason,
        )
    ]
    if alias_plan.dup_aliases_to_tombstone:
        entity_deltas.append(
            EntityDelta(
                entity_id=dup_id,
                remove_aliases=alias_plan.dup_aliases_to_tombstone,
            )
        )
    for other_eid, updated_aliases in alias_plan.other_alias_updates.items():
        entity_deltas.append(
            EntityDelta(entity_id=other_eid, add_aliases=list(updated_aliases))
        )
    entity_deltas.extend(location_ref_deltas)

    tombstone_entity_fn = getattr(engine.store, "tombstone_entity", None)
    tombstone_edge_fn = getattr(engine.store, "tombstone_edge", None)
    migrate_voice_fn = getattr(engine.store, "migrate_entity_voice_exemplars", None)
    migrate_episodes_fn = getattr(engine.store, "migrate_entity_episodes", None)

    voice_migrated = 0
    episodes_migrated = 0
    epistemics_migrated = 0

    with engine._store_transaction():
        engine.store.apply_delta(
            step,
            StepDelta(entities=entity_deltas, edges=edge_plan.edges_to_write),
            timeline_id=timeline_id,
        )
        if edge_plan.edges_to_tombstone:
            if callable(tombstone_edge_fn):
                for old_edge in edge_plan.edges_to_tombstone:
                    tombstone_edge_fn(
                        old_edge.source_id,
                        old_edge.target_id,
                        old_edge.relation_type,
                        step=step,
                        timeline_id=timeline_id,
                    )
            else:
                engine.store.apply_delta(
                    step,
                    StepDelta(
                        edges=[
                            EdgeDelta(
                                source_id=old_edge.source_id,
                                target_id=old_edge.target_id,
                                relation_type=old_edge.relation_type,
                                remove=True,
                            )
                            for old_edge in edge_plan.edges_to_tombstone
                        ]
                    ),
                    timeline_id=timeline_id,
                )
        if callable(tombstone_entity_fn):
            tombstone_entity_fn(
                dup_id,
                step=step,
                timeline_id=timeline_id,
                payload={"merged_into": canon_id, "reason": reason},
            )
        else:
            engine.store.apply_delta(
                step,
                StepDelta(entities=[EntityDelta(entity_id=dup_id, remove=True)]),
                timeline_id=timeline_id,
            )
        if migrate_voice and engine._policy.has_tier(5) and callable(migrate_voice_fn):
            voice_migrated = migrate_voice_fn(
                dup_id, canon_id, step=step, timeline_id=timeline_id, dry_run=False
            )
        if migrate_episodes and engine._policy.has_tier(6) and callable(migrate_episodes_fn):
            episodes_migrated, epistemics_migrated = migrate_episodes_fn(
                dup_id, canon_id, step=step, timeline_id=timeline_id, dry_run=False
            )

    return voice_migrated, episodes_migrated, epistemics_migrated


def merge_entities_impl(
    engine: CanonArcEngine,
    duplicate_id: str,
    canonical_id: str,
    *,
    step: int | None = None,
    timeline_id: str | None = None,
    merge_attributes: bool = True,
    migrate_edges: bool = True,
    migrate_voice: bool = True,
    migrate_episodes: bool = True,
    conflict_strategy: Literal["canonical_wins", "duplicate_wins"] = "canonical_wins",
    allow_kind_mismatch: bool = False,
    reason: str | None = None,
    dry_run: bool = False,
) -> EntityMergeReport:
    """Atomically merges duplicate_id into canonical_id at (timeline_id, step)."""
    dup_id, canon_id = _validate_merge_args(duplicate_id, canonical_id, conflict_strategy)

    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")
    eff_step = tl_info.head_step if step is None else step
    if eff_step < 0:
        raise ValueError(f"step must be non-negative, got {eff_step}.")

    active_entities, active_aliases, active_edges = fetch_active_graph_state(
        engine.store, tl, eff_step
    )
    entity_by_id = {e.entity_id: e for e in active_entities}
    dup_ent = entity_by_id.get(dup_id)
    if dup_ent is None:
        raise ValueError(
            f"Duplicate entity '{dup_id}' is not active on timeline '{tl}' at step {eff_step}."
        )
    canon_ent = entity_by_id.get(canon_id)
    if canon_ent is None:
        raise ValueError(
            f"Canonical entity '{canon_id}' is not active on timeline '{tl}' at step {eff_step}."
        )

    warnings_list: list[str] = []
    if dup_ent.kind != canon_ent.kind:
        msg = (
            f"Kind mismatch: duplicate '{dup_id}' has kind='{dup_ent.kind}', "
            f"canonical '{canon_id}' has kind='{canon_ent.kind}'."
        )
        if not allow_kind_mismatch:
            raise ValueError(f"{msg} Pass allow_kind_mismatch=True to override.")
        warnings_list.append(msg)

    eff_reason = reason if reason is not None else f"Merged entity '{dup_id}' into '{canon_id}'"
    has_tier3 = engine._policy.has_tier(3)

    alias_plan = _plan_alias_migration(dup_ent, canon_ent, active_aliases)
    attr_plan = _merge_entity_attributes(
        dup_ent,
        canon_ent,
        merge_attributes=merge_attributes,
        conflict_strategy=conflict_strategy,
        has_tier3=has_tier3,
    )
    location_ref_deltas, location_refs_updated = _plan_location_ref_updates(
        dup_ent,
        canon_id,
        active_entities,
        attr_plan.merged_state,
        has_tier3=has_tier3,
        reason=eff_reason,
    )
    if migrate_edges and engine._policy.has_tier(2):
        edge_plan = _plan_edge_migration(
            dup_id,
            canon_id,
            active_edges,
            conflict_strategy=conflict_strategy,
            hard_keys=engine._policy.hard_filter_keys,
            reason=eff_reason,
            warnings_list=warnings_list,
        )
    else:
        edge_plan = _EdgeMigrationPlan([], [], [], [], [], [])

    if dry_run:
        migrate_voice_fn = getattr(engine.store, "migrate_entity_voice_exemplars", None)
        migrate_episodes_fn = getattr(engine.store, "migrate_entity_episodes", None)
        voice_migrated = (
            migrate_voice_fn(dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True)
            if migrate_voice and engine._policy.has_tier(5) and callable(migrate_voice_fn)
            else 0
        )
        episodes_migrated, epistemics_migrated = (
            migrate_episodes_fn(dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True)
            if migrate_episodes and engine._policy.has_tier(6) and callable(migrate_episodes_fn)
            else (0, 0)
        )
    else:
        voice_migrated, episodes_migrated, epistemics_migrated = _commit_merge_plan(
            engine,
            dup_id=dup_id,
            canon_id=canon_id,
            timeline_id=tl,
            step=eff_step,
            reason=eff_reason,
            alias_plan=alias_plan,
            attr_plan=attr_plan,
            location_ref_deltas=location_ref_deltas,
            edge_plan=edge_plan,
            migrate_voice=migrate_voice,
            migrate_episodes=migrate_episodes,
        )

    return EntityMergeReport(
        timeline_id=tl,
        step=eff_step,
        duplicate_id=dup_id,
        canonical_id=canon_id,
        dry_run=dry_run,
        aliases_added=[a.alias for a in alias_plan.aliases_to_add],
        aliases_tombstoned=alias_plan.dup_aliases_to_tombstone,
        base_attributes_merged=attr_plan.base_attrs_merged,
        state_keys_merged=attr_plan.state_keys_merged,
        edges_migrated=edge_plan.edges_migrated,
        edges_merged_on_collision=edge_plan.edges_merged_on_collision,
        edges_tombstoned=edge_plan.edges_tombstoned_keys,
        self_loops_dropped=edge_plan.self_loops_dropped,
        location_refs_updated=location_refs_updated,
        voice_exemplars_migrated=voice_migrated,
        episodes_migrated=episodes_migrated,
        epistemics_migrated=epistemics_migrated,
        warnings=warnings_list,
    )
