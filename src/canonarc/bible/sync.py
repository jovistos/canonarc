from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..inflections import InflectionRequest, generate_inflections_batch
from ..models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    StepDelta,
)
from .models import (
    _INFLECTION_SENSITIVE_ATTRS,
    BibleImportFormat,
    BibleSyncMode,
    BibleSyncReport,
    _alias_signature,
    _edge_key,
    _ParsedBible,
    _ParsedEntityEntry,
    infer_import_format,
)
from .parsers import _parse_csv_bible, _parse_json_bible

if TYPE_CHECKING:
    from ..engine import CanonArcEngine


def _check_incoming_alias_collisions(parsed: _ParsedBible, report: BibleSyncReport) -> None:
    incoming_alias_owners: dict[str, list[str]] = {}
    for entry in parsed.entries:
        for al in entry.aliases:
            if al.kind == "disguise":
                continue
            incoming_alias_owners.setdefault(al.alias.strip().lower(), []).append(entry.entity.entity_id)
    for al_lower, owners in sorted(incoming_alias_owners.items()):
        unique_owners = sorted(set(owners))
        if len(unique_owners) > 1:
            report.warnings.append(
                f"Alias collision: '{al_lower}' is assigned to multiple entities ({', '.join(unique_owners)})."
            )


def _diff_edges(
    parsed: _ParsedBible,
    active_edges: dict[str, DirectedEdgeRecord],
    *,
    mode: BibleSyncMode,
    reason_tag: str,
    report: BibleSyncReport,
) -> tuple[list[EdgeDelta], list[str]]:
    edge_deltas: list[EdgeDelta] = []
    mutated_edge_keys: list[str] = []
    if not parsed.has_edges_section:
        return edge_deltas, mutated_edge_keys

    incoming_edge_keys: set[str] = set()
    for inc_edge in parsed.edges:
        ekey = _edge_key(inc_edge.source_id, inc_edge.target_id, inc_edge.relation_type)
        incoming_edge_keys.add(ekey)
        existing_edge = active_edges.get(ekey)
        if existing_edge is None:
            report.created_edges.append(ekey)
            edge_deltas.append(
                EdgeDelta(
                    source_id=inc_edge.source_id,
                    target_id=inc_edge.target_id,
                    relation_type=inc_edge.relation_type,
                    state=dict(inc_edge.state),
                    merge=False,
                    reason=reason_tag,
                )
            )
            continue

        if mode == "merge":
            merged_edge_state = dict(existing_edge.state)
            merged_edge_state.update(inc_edge.state)
            use_merge = True
        else:
            merged_edge_state = dict(inc_edge.state)
            use_merge = False

        if merged_edge_state != existing_edge.state:
            report.updated_edges.append(ekey)
            mutated_edge_keys.append(ekey)
            edge_deltas.append(
                EdgeDelta(
                    source_id=inc_edge.source_id,
                    target_id=inc_edge.target_id,
                    relation_type=inc_edge.relation_type,
                    state=dict(inc_edge.state),
                    merge=use_merge,
                    reason=reason_tag,
                )
            )
        else:
            report.unchanged_edges.append(ekey)

    if mode == "replace":
        for ekey, existing_edge in sorted(active_edges.items()):
            if ekey not in incoming_edge_keys:
                report.removed_edges.append(ekey)
                mutated_edge_keys.append(ekey)
                edge_deltas.append(
                    EdgeDelta(
                        source_id=existing_edge.source_id,
                        target_id=existing_edge.target_id,
                        relation_type=existing_edge.relation_type,
                        remove=True,
                        reason=reason_tag,
                    )
                )

    return edge_deltas, mutated_edge_keys


def _warn_shadowed_updates(
    engine: CanonArcEngine,
    *,
    timeline_id: str,
    step: int,
    head_step: int,
    mutated_entity_ids: list[str],
    mutated_edge_keys: list[str],
    report: BibleSyncReport,
) -> None:
    if step >= head_step:
        return
    for eid in mutated_entity_ids:
        hist = engine.store.get_history(eid, timeline_id=timeline_id, record_type="entity")
        later_steps = [h["valid_from_step"] for h in hist if h["valid_from_step"] > step]
        if later_steps:
            report.warnings.append(
                f"Shadowed update: entity '{eid}' has later revision(s) at step(s) {later_steps} "
                f"(head={head_step}). Syncing at step={step} will be shadowed at steps >= {min(later_steps)}; "
                f"sync at step={head_step} or roll back first."
            )
    for ekey in mutated_edge_keys:
        hist = engine.store.get_history(ekey, timeline_id=timeline_id, record_type="edge")
        later_steps = [h["valid_from_step"] for h in hist if h["valid_from_step"] > step]
        if later_steps:
            report.warnings.append(
                f"Shadowed update: edge '{ekey}' has later revision(s) at step(s) {later_steps} "
                f"(head={head_step}). Syncing at step={step} will be shadowed at steps >= {min(later_steps)}."
            )


def _sync_parsed_bible(
    engine: CanonArcEngine,
    parsed: _ParsedBible,
    *,
    source: str,
    eff_format: BibleImportFormat,
    reason_tag: str,
    step: int = 0,
    timeline_id: str | None = None,
    mode: BibleSyncMode = "upsert",
    target_lang: str | None = None,
    auto_inflect: bool = True,
    batch_size: int = 25,
    dry_run: bool = False,
    active_entities: dict[str, EntityRecord] | None = None,
) -> BibleSyncReport:
    """Commits an already-parsed Bible into the causal store with diff/upsert and batched inflections."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")

    eff_active_entities = (
        active_entities
        if active_entities is not None
        else {e.entity_id: e for e in engine.store.get_all_active_entities(tl, step)}
    )
    active_aliases_by_ent: dict[str, dict[str, AliasRecord]] = {}
    for a in engine.store.get_all_active_aliases(tl, step):
        active_aliases_by_ent.setdefault(a.entity_id, {})[a.alias.strip().lower()] = a
    active_edges = {
        _edge_key(ed.source_id, ed.target_id, ed.relation_type): ed
        for ed in engine.store.get_all_active_edges(tl, step)
    }

    report = BibleSyncReport(
        source=source,
        format=eff_format,
        timeline_id=tl,
        step=step,
        mode=mode,
        dry_run=dry_run,
        warnings=list(parsed.warnings),
    )

    _check_incoming_alias_collisions(parsed, report)

    entities_to_register: list[EntityRecord] = []
    entity_deltas: list[EntityDelta] = []
    mutated_entity_ids: list[str] = []
    incoming_entity_ids: set[str] = set()

    # Phase 1: Plan entity updates & collect all inflection requests for batched generation
    planned_entries: list[tuple[_ParsedEntityEntry, EntityRecord | None, EntityRecord, bool, bool, bool]] = []
    pending_inflect_indices: list[int] = []
    pending_inflect_requests: list[InflectionRequest] = []

    for idx, entry in enumerate(parsed.entries):
        inc = entry.entity
        eid = inc.entity_id
        incoming_entity_ids.add(eid)
        existing = eff_active_entities.get(eid)

        if existing is None:
            resolved_entity = EntityRecord(
                entity_id=eid,
                kind=inc.kind,
                canonical_name=inc.canonical_name,
                base_attributes=dict(inc.base_attributes),
                state=dict(inc.state) if entry.has_state else {},
                private_state=dict(inc.private_state) if entry.has_private_state else {},
                reason=reason_tag,
            )
            should_inflect_now = auto_inflect and not dry_run
            planned_entries.append((entry, None, resolved_entity, True, True, should_inflect_now))
        else:
            if mode == "merge":
                new_base = dict(existing.base_attributes)
                new_base.update(inc.base_attributes)
                new_state = dict(existing.state)
                if entry.has_state:
                    new_state.update(inc.state)
                new_priv = dict(existing.private_state)
                if entry.has_private_state:
                    new_priv.update(inc.private_state)
            else:
                new_base = dict(inc.base_attributes)
                new_state = dict(inc.state) if entry.has_state else dict(existing.state)
                new_priv = dict(inc.private_state) if entry.has_private_state else dict(existing.private_state)

            entity_changed = (
                inc.kind != existing.kind
                or inc.canonical_name != existing.canonical_name
                or new_base != existing.base_attributes
                or new_state != existing.state
                or new_priv != existing.private_state
            )

            inflection_inputs_changed = any(
                new_base.get(k) != existing.base_attributes.get(k) for k in _INFLECTION_SENSITIVE_ATTRS
            ) or (inc.kind != existing.kind)

            resolved_entity = EntityRecord(
                entity_id=eid,
                kind=inc.kind,
                canonical_name=inc.canonical_name,
                base_attributes=new_base,
                state=new_state,
                private_state=new_priv,
                reason=reason_tag,
            )
            should_inflect_now = auto_inflect and inflection_inputs_changed and not dry_run
            planned_entries.append(
                (entry, existing, resolved_entity, entity_changed, inflection_inputs_changed, should_inflect_now)
            )

        if should_inflect_now and engine.inflection_generator is not None:
            req = engine._make_inflection_request(resolved_entity, target_lang=target_lang)
            if req is not None:
                pending_inflect_indices.append(idx)
                pending_inflect_requests.append(req)

    # Phase 2: Generate all needed inflections in batched calls
    prefetched_forms_by_idx: dict[int, list[str]] = {}
    if pending_inflect_requests and engine.inflection_generator is not None:
        batched_forms = generate_inflections_batch(
            engine.inflection_generator,
            pending_inflect_requests,
            batch_size=batch_size,
        )
        for idx, forms in zip(pending_inflect_indices, batched_forms):
            prefetched_forms_by_idx[idx] = forms

    # Phase 3: Diff aliases & emit deltas
    for idx, (entry, existing, resolved_entity, entity_changed, inflection_inputs_changed, should_inflect_now) in enumerate(
        planned_entries
    ):
        eid = resolved_entity.entity_id

        if existing is None:
            desired_aliases = list(entry.aliases)
            extra_target_aliases = engine._build_target_aliases(
                resolved_entity,
                desired_aliases,
                target_lang=target_lang,
                auto_inflect=False,
                prefetched_forms=prefetched_forms_by_idx.get(idx),
            )
            if extra_target_aliases:
                report.inflections_generated += sum(
                    1 for a in extra_target_aliases if a.kind == "inflection"
                )
                desired_aliases.extend(extra_target_aliases)

            report.created_entities.append(eid)
            for al in desired_aliases:
                report.added_aliases.append((eid, al.alias))
            entities_to_register.append(resolved_entity)
            if desired_aliases:
                add_aliases_payload: list[str | AliasRecord] = list(desired_aliases)
                entity_deltas.append(
                    EntityDelta(
                        entity_id=eid,
                        add_aliases=add_aliases_payload,
                        reason=reason_tag,
                    )
                )
            continue

        desired_alias_map: dict[str, AliasRecord] = {
            a.alias.strip().lower(): a for a in entry.aliases
        }
        extra_target_aliases = engine._build_target_aliases(
            resolved_entity,
            list(desired_alias_map.values()),
            target_lang=target_lang,
            auto_inflect=False,
            prefetched_forms=prefetched_forms_by_idx.get(idx),
        )
        if extra_target_aliases:
            if should_inflect_now:
                report.inflections_generated += sum(
                    1 for a in extra_target_aliases if a.kind == "inflection"
                )
            for a in extra_target_aliases:
                desired_alias_map.setdefault(a.alias.strip().lower(), a)

        existing_alias_map = active_aliases_by_ent.get(eid, {})
        aliases_to_add: list[AliasRecord] = []
        aliases_to_remove: list[str] = []

        for al_key, al_rec in desired_alias_map.items():
            cur = existing_alias_map.get(al_key)
            if cur is None or _alias_signature(cur) != _alias_signature(al_rec):
                aliases_to_add.append(al_rec)
                report.added_aliases.append((eid, al_rec.alias))

        if mode in ("upsert", "replace"):
            for al_key, cur in existing_alias_map.items():
                if al_key in desired_alias_map:
                    continue
                if (
                    eff_format == "csv"
                    and not inflection_inputs_changed
                    and cur.kind == "inflection"
                ):
                    continue
                aliases_to_remove.append(cur.alias)
                report.removed_aliases.append((eid, cur.alias))

        if entity_changed:
            report.updated_entities.append(eid)
            mutated_entity_ids.append(eid)
            entities_to_register.append(resolved_entity)
        else:
            report.unchanged_entities.append(eid)

        if aliases_to_add or aliases_to_remove:
            add_payload: list[str | AliasRecord] = list(aliases_to_add)
            entity_deltas.append(
                EntityDelta(
                    entity_id=eid,
                    add_aliases=add_payload,
                    remove_aliases=aliases_to_remove,
                    reason=reason_tag,
                )
            )

    if mode == "replace":
        for eid in sorted(eff_active_entities.keys()):
            if eid not in incoming_entity_ids:
                report.removed_entities.append(eid)
                mutated_entity_ids.append(eid)
                entity_deltas.append(EntityDelta(entity_id=eid, remove=True, reason=reason_tag))

    edge_deltas, mutated_edge_keys = _diff_edges(
        parsed, active_edges, mode=mode, reason_tag=reason_tag, report=report
    )
    _warn_shadowed_updates(
        engine,
        timeline_id=tl,
        step=step,
        head_step=tl_info.head_step,
        mutated_entity_ids=mutated_entity_ids,
        mutated_edge_keys=mutated_edge_keys,
        report=report,
    )

    if dry_run or not report.has_changes:
        return report

    with engine._store_transaction():
        for ent in entities_to_register:
            engine.store.register_entity(ent, aliases=(), step=step, timeline_id=tl)
        if entity_deltas or edge_deltas:
            engine.store.apply_delta(
                step,
                StepDelta(entities=entity_deltas, edges=edge_deltas),
                timeline_id=tl,
            )

    return report


def sync_bible(
    engine: CanonArcEngine,
    path: str | Path,
    *,
    format: Literal["auto", "csv", "json"] = "auto",
    step: int = 0,
    timeline_id: str | None = None,
    mode: BibleSyncMode = "upsert",
    target_lang: str | None = None,
    auto_inflect: bool = True,
    batch_size: int = 25,
    dry_run: bool = False,
    column_map: dict[str, str] | None = None,
) -> BibleSyncReport:
    """Syncs a human-editable CSV or JSON Series Bible into the causal store with diff/upsert semantics."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")

    src_path = Path(path)
    eff_format = infer_import_format(src_path, format)
    encoding = "utf-8-sig" if eff_format == "csv" else "utf-8"
    raw_text = src_path.read_text(encoding=encoding)

    active_entities = {e.entity_id: e for e in engine.store.get_all_active_entities(tl, step)}
    if eff_format == "csv":
        parsed = _parse_csv_bible(raw_text, column_map=column_map, existing_ids=set(active_entities.keys()))
    else:
        parsed = _parse_json_bible(raw_text, existing_ids=set(active_entities.keys()))

    return _sync_parsed_bible(
        engine,
        parsed,
        source=str(src_path),
        eff_format=eff_format,
        reason_tag=f"bible_sync:{src_path.name}",
        step=step,
        timeline_id=tl,
        mode=mode,
        target_lang=target_lang,
        auto_inflect=auto_inflect,
        batch_size=batch_size,
        dry_run=dry_run,
        active_entities=active_entities,
    )
