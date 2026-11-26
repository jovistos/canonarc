from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityMergeReport,
    StepDelta,
)
from ..store import CausalStore

if TYPE_CHECKING:
    from .core import CanonArcEngine


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

    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")
    eff_step = tl_info.head_step if step is None else step
    if eff_step < 0:
        raise ValueError(f"step must be non-negative, got {eff_step}.")

    get_graph_fn = getattr(engine.store, "get_active_graph_state", None)
    if callable(get_graph_fn):
        active_entities, active_aliases, active_edges = get_graph_fn(tl, eff_step)
    else:
        active_entities = engine.store.get_all_active_entities(tl, eff_step)
        active_aliases = engine.store.get_all_active_aliases(tl, eff_step)
        active_edges = engine.store.get_all_active_edges(tl, eff_step)

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

    def _remap_known_to(known_to: list[str] | None) -> list[str] | None:
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

    # 1. Collect aliases to promote onto canonical_id and old aliases to tombstone
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
                    "known_to": _remap_known_to(rec.known_to),
                }
            )
        )

    _queue_alias(AliasRecord(alias=dup_ent.canonical_name, entity_id=canon_id, kind="permanent"))

    dup_target = dup_ent.base_attributes.get("target_name")
    if isinstance(dup_target, str) and dup_target.strip():
        raw_lang = (
            dup_ent.base_attributes.get("target_lang")
            or dup_ent.base_attributes.get("lang")
        )
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

    dup_aliases_to_tombstone = [a.alias for a in dup_aliases]

    other_alias_updates: dict[str, list[AliasRecord]] = {}
    for a in active_aliases:
        if a.entity_id in (dup_id, canon_id):
            continue
        if a.known_to and dup_id in a.known_to:
            other_alias_updates.setdefault(a.entity_id, []).append(
                a.model_copy(update={"known_to": _remap_known_to(a.known_to)})
            )

    # 2. Merge base_attributes, state, and private_state
    merged_base_attrs: dict[str, Any] = dict(canon_ent.base_attributes)
    base_attrs_merged: list[str] = []
    merged_state: dict[str, Any] = dict(canon_ent.state)
    state_keys_merged: list[str] = []
    merged_private: dict[str, Any] = dict(canon_ent.private_state)

    def _is_empty_val(val: Any) -> bool:
        return val is None or (isinstance(val, str) and not val.strip())

    if merge_attributes:
        for k, v in dup_ent.base_attributes.items():
            if k == "merged_from" or _is_empty_val(v):
                continue
            if conflict_strategy == "duplicate_wins" or k not in merged_base_attrs or _is_empty_val(merged_base_attrs[k]):
                if merged_base_attrs.get(k) != v:
                    merged_base_attrs[k] = v
                    base_attrs_merged.append(k)

        if engine._policy.has_tier(3):
            for k, v in dup_ent.state.items():
                if _is_empty_val(v):
                    continue
                if conflict_strategy == "duplicate_wins" or k not in merged_state or _is_empty_val(merged_state[k]):
                    if merged_state.get(k) != v:
                        merged_state[k] = v
                        state_keys_merged.append(k)
            for k, v in dup_ent.private_state.items():
                if _is_empty_val(v):
                    continue
                if conflict_strategy == "duplicate_wins" or k not in merged_private or _is_empty_val(merged_private[k]):
                    if merged_private.get(k) != v:
                        merged_private[k] = v

    merged_from_list: list[str] = []
    for src_list in (
        canon_ent.base_attributes.get("merged_from"),
        dup_ent.base_attributes.get("merged_from"),
        [dup_id],
    ):
        if isinstance(src_list, list):
            for item in src_list:
                if isinstance(item, str) and item.strip() and item != canon_id and item not in merged_from_list:
                    merged_from_list.append(item)
    merged_base_attrs["merged_from"] = merged_from_list

    # 3. Update location_id references if merging locations
    location_ref_deltas: list[EntityDelta] = []
    location_refs_updated: list[str] = []
    if dup_ent.kind == "location" and engine._policy.has_tier(3):
        for ent in active_entities:
            if ent.entity_id == dup_id:
                continue
            if ent.state.get("location_id") == dup_id:
                location_refs_updated.append(ent.entity_id)
                if ent.entity_id == canon_id:
                    merged_state["location_id"] = canon_id
                else:
                    location_ref_deltas.append(
                        EntityDelta(
                            entity_id=ent.entity_id,
                            state={"location_id": canon_id},
                            merge_state=True,
                            reason=eff_reason,
                        )
                    )

    # 4. Migrate active edges
    edges_to_write: list[EdgeDelta] = []
    edges_to_tombstone: list[DirectedEdgeRecord] = []
    edges_migrated: list[str] = []
    edges_merged_on_collision: list[str] = []
    edges_tombstoned_keys: list[str] = []
    self_loops_dropped: list[str] = []

    if migrate_edges and engine._policy.has_tier(2):
        edge_index: dict[tuple[str, str, str], DirectedEdgeRecord] = {
            (e.source_id, e.target_id, e.relation_type): e for e in active_edges
        }
        hard_keys = set(engine._policy.hard_filter_keys)

        for edge in active_edges:
            if edge.source_id != dup_id and edge.target_id != dup_id:
                continue
            old_key = CausalStore.edge_subject_key(
                edge.source_id, edge.target_id, edge.relation_type
            )
            edges_to_tombstone.append(edge)
            edges_tombstoned_keys.append(old_key)

            new_src = canon_id if edge.source_id == dup_id else edge.source_id
            new_tgt = canon_id if edge.target_id == dup_id else edge.target_id
            if new_src == new_tgt:
                self_loops_dropped.append(old_key)
                continue

            new_key = CausalStore.edge_subject_key(new_src, new_tgt, edge.relation_type)
            existing_edge = edge_index.get((new_src, new_tgt, edge.relation_type))
            if existing_edge is None:
                edges_to_write.append(
                    EdgeDelta(
                        source_id=new_src,
                        target_id=new_tgt,
                        relation_type=edge.relation_type,
                        state=dict(edge.state),
                        merge=False,
                        caused_by_episode_id=edge.caused_by_episode_id,
                        reason=eff_reason,
                    )
                )
                edges_migrated.append(new_key)
            else:
                for hk in sorted(hard_keys):
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
                    edges_to_write.append(
                        EdgeDelta(
                            source_id=new_src,
                            target_id=new_tgt,
                            relation_type=edge.relation_type,
                            state=merged_edge_state,
                            merge=False,
                            caused_by_episode_id=existing_edge.caused_by_episode_id
                            or edge.caused_by_episode_id,
                            reason=eff_reason,
                        )
                    )
                edges_merged_on_collision.append(new_key)

    # 5. Execute (or preview for dry_run)
    migrate_voice_fn = getattr(engine.store, "migrate_entity_voice_exemplars", None)
    migrate_episodes_fn = getattr(engine.store, "migrate_entity_episodes", None)

    voice_migrated = 0
    episodes_migrated = 0
    epistemics_migrated = 0

    if dry_run:
        if migrate_voice and engine._policy.has_tier(5) and callable(migrate_voice_fn):
            voice_migrated = migrate_voice_fn(
                dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True
            )
        if migrate_episodes and engine._policy.has_tier(6) and callable(migrate_episodes_fn):
            episodes_migrated, epistemics_migrated = migrate_episodes_fn(
                dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=True
            )
    else:
        entity_deltas: list[EntityDelta] = [
            EntityDelta(
                entity_id=canon_id,
                base_attributes=merged_base_attrs,
                state=merged_state if engine._policy.has_tier(3) else {},
                private_state=merged_private if engine._policy.has_tier(3) else {},
                merge_state=False if engine._policy.has_tier(3) else True,
                add_aliases=list(aliases_to_add),
                reason=eff_reason,
            )
        ]
        if dup_aliases_to_tombstone:
            entity_deltas.append(
                EntityDelta(
                    entity_id=dup_id,
                    remove_aliases=dup_aliases_to_tombstone,
                )
            )
        for other_eid, updated_aliases in other_alias_updates.items():
            entity_deltas.append(
                EntityDelta(
                    entity_id=other_eid,
                    add_aliases=list(updated_aliases),
                )
            )
        entity_deltas.extend(location_ref_deltas)

        tombstone_entity_fn = getattr(engine.store, "tombstone_entity", None)
        tombstone_edge_fn = getattr(engine.store, "tombstone_edge", None)

        with engine._store_transaction():
            engine.store.apply_delta(
                eff_step,
                StepDelta(entities=entity_deltas, edges=edges_to_write),
                timeline_id=tl,
            )
            if edges_to_tombstone:
                if callable(tombstone_edge_fn):
                    for old_edge in edges_to_tombstone:
                        tombstone_edge_fn(
                            old_edge.source_id,
                            old_edge.target_id,
                            old_edge.relation_type,
                            step=eff_step,
                            timeline_id=tl,
                        )
                else:
                    engine.store.apply_delta(
                        eff_step,
                        StepDelta(
                            edges=[
                                EdgeDelta(
                                    source_id=old_edge.source_id,
                                    target_id=old_edge.target_id,
                                    relation_type=old_edge.relation_type,
                                    remove=True,
                                )
                                for old_edge in edges_to_tombstone
                            ]
                        ),
                        timeline_id=tl,
                    )
            if callable(tombstone_entity_fn):
                tombstone_entity_fn(
                    dup_id,
                    step=eff_step,
                    timeline_id=tl,
                    payload={"merged_into": canon_id, "reason": eff_reason},
                )
            else:
                engine.store.apply_delta(
                    eff_step,
                    StepDelta(entities=[EntityDelta(entity_id=dup_id, remove=True)]),
                    timeline_id=tl,
                )
            if migrate_voice and engine._policy.has_tier(5) and callable(migrate_voice_fn):
                voice_migrated = migrate_voice_fn(
                    dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=False
                )
            if migrate_episodes and engine._policy.has_tier(6) and callable(migrate_episodes_fn):
                episodes_migrated, epistemics_migrated = migrate_episodes_fn(
                    dup_id, canon_id, step=eff_step, timeline_id=tl, dry_run=False
                )

    return EntityMergeReport(
        timeline_id=tl,
        step=eff_step,
        duplicate_id=dup_id,
        canonical_id=canon_id,
        dry_run=dry_run,
        aliases_added=[a.alias for a in aliases_to_add],
        aliases_tombstoned=dup_aliases_to_tombstone,
        base_attributes_merged=base_attrs_merged,
        state_keys_merged=state_keys_merged,
        edges_migrated=edges_migrated,
        edges_merged_on_collision=edges_merged_on_collision,
        edges_tombstoned=edges_tombstoned_keys,
        self_loops_dropped=self_loops_dropped,
        location_refs_updated=location_refs_updated,
        voice_exemplars_migrated=voice_migrated,
        episodes_migrated=episodes_migrated,
        epistemics_migrated=epistemics_migrated,
        warnings=warnings_list,
    )
