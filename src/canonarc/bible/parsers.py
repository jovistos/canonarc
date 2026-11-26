from __future__ import annotations

import csv
import io
import json
from typing import Any

from ..models import AliasRecord, DirectedEdgeRecord, EntityRecord
from .models import (
    _CSV_RESERVED_COLS,
    _ParsedBible,
    _ParsedEntityEntry,
    _slugify_entity_id,
)


def _parse_csv_bible(
    raw_text: str,
    *,
    column_map: dict[str, str] | None = None,
    existing_ids: set[str],
) -> _ParsedBible:
    reader = csv.DictReader(io.StringIO(raw_text))
    if not reader.fieldnames:
        raise ValueError("CSV file is empty or missing a header row.")

    col_map = column_map or {}
    lower_col_map = {k.strip().lower(): v.strip() for k, v in col_map.items()}

    def _map_header(raw_header: str) -> str:
        stripped = raw_header.strip()
        if stripped in col_map:
            return col_map[stripped].strip()
        return lower_col_map.get(stripped.lower(), stripped)

    entries: list[_ParsedEntityEntry] = []
    warnings: list[str] = []
    seen_ids: set[str] = set(existing_ids)
    file_entity_ids: set[str] = set()

    for row_num, raw_row in enumerate(reader, start=2):
        mapped: dict[str, str] = {}
        for raw_k, raw_v in raw_row.items():
            if raw_k is None:
                continue
            key = _map_header(raw_k)
            val = (raw_v or "").strip()
            if key:
                mapped[key] = val

        if not any(mapped.values()):
            continue

        canonical_name = mapped.get("canonical_name", "")
        if not canonical_name:
            raise ValueError(f"CSV row {row_num}: missing required 'canonical_name'.")

        kind = (mapped.get("kind") or "character").strip()
        entity_id = mapped.get("entity_id", "")
        if not entity_id:
            entity_id = _slugify_entity_id(kind, canonical_name, seen_ids)
            warnings.append(
                f"Row {row_num}: auto-generated entity_id '{entity_id}' for '{canonical_name}'."
            )
        if entity_id in file_entity_ids:
            raise ValueError(f"CSV row {row_num}: duplicate entity_id '{entity_id}'.")
        file_entity_ids.add(entity_id)
        seen_ids.add(entity_id)

        base_attributes: dict[str, Any] = {}
        for k, v in mapped.items():
            if k in _CSV_RESERVED_COLS or not v:
                continue
            base_attributes[k] = v

        state: dict[str, Any] = {}
        has_state = False
        if "state" in mapped and mapped["state"]:
            try:
                parsed_state = json.loads(mapped["state"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"CSV row {row_num}: invalid JSON in 'state' column: {exc}") from exc
            if not isinstance(parsed_state, dict):
                raise ValueError(f"CSV row {row_num}: 'state' column must be a JSON object.")
            state = parsed_state
            has_state = True

        private_state: dict[str, Any] = {}
        has_private_state = False
        if "private_state" in mapped and mapped["private_state"]:
            try:
                parsed_priv = json.loads(mapped["private_state"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"CSV row {row_num}: invalid JSON in 'private_state' column: {exc}"
                ) from exc
            if not isinstance(parsed_priv, dict):
                raise ValueError(f"CSV row {row_num}: 'private_state' column must be a JSON object.")
            private_state = parsed_priv
            has_private_state = True

        aliases: list[AliasRecord] = []
        raw_aliases = mapped.get("aliases", "")
        if raw_aliases:
            seen_al: set[str] = set()
            for part in raw_aliases.split(";"):
                clean = part.strip()
                if not clean:
                    continue
                low = clean.lower()
                if low in seen_al:
                    continue
                seen_al.add(low)
                aliases.append(AliasRecord(alias=clean, entity_id=entity_id, kind="permanent"))

        entity = EntityRecord(
            entity_id=entity_id,
            kind=kind,
            canonical_name=canonical_name,
            base_attributes=base_attributes,
            state=state,
            private_state=private_state,
        )
        entries.append(
            _ParsedEntityEntry(
                entity=entity,
                aliases=aliases,
                has_state=has_state,
                has_private_state=has_private_state,
            )
        )

    return _ParsedBible(entries=entries, edges=[], has_edges_section=False, warnings=warnings)


def _parse_json_bible(raw_text: str, *, existing_ids: set[str]) -> _ParsedBible:
    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in bible file: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("JSON bible root must be an object.")

    raw_entities = doc.get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("JSON bible 'entities' must be a list.")

    entries: list[_ParsedEntityEntry] = []
    warnings: list[str] = []
    seen_ids: set[str] = set(existing_ids)
    file_entity_ids: set[str] = set()
    entries_by_id: dict[str, _ParsedEntityEntry] = {}

    for idx, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            raise ValueError(f"JSON entities[{idx}] must be an object.")
        canonical_name = str(item.get("canonical_name", "")).strip()
        if not canonical_name:
            raise ValueError(f"JSON entities[{idx}]: missing required 'canonical_name'.")
        kind = str(item.get("kind") or "character").strip()
        entity_id = str(item.get("entity_id") or "").strip()
        if not entity_id:
            entity_id = _slugify_entity_id(kind, canonical_name, seen_ids)
            warnings.append(
                f"JSON entities[{idx}]: auto-generated entity_id '{entity_id}' for '{canonical_name}'."
            )
        if entity_id in file_entity_ids:
            raise ValueError(f"JSON entities[{idx}]: duplicate entity_id '{entity_id}'.")
        file_entity_ids.add(entity_id)
        seen_ids.add(entity_id)

        base_attributes = item.get("base_attributes") or {}
        if not isinstance(base_attributes, dict):
            raise ValueError(f"JSON entities[{idx}]: 'base_attributes' must be an object.")

        has_state = "state" in item and item["state"] is not None
        state = item.get("state") or {}
        if not isinstance(state, dict):
            raise ValueError(f"JSON entities[{idx}]: 'state' must be an object.")

        has_private_state = "private_state" in item and item["private_state"] is not None
        private_state = item.get("private_state") or {}
        if not isinstance(private_state, dict):
            raise ValueError(f"JSON entities[{idx}]: 'private_state' must be an object.")

        aliases: list[AliasRecord] = []
        raw_aliases = item.get("aliases") or []
        if not isinstance(raw_aliases, list):
            raise ValueError(f"JSON entities[{idx}]: 'aliases' must be a list.")
        for a_idx, raw_al in enumerate(raw_aliases):
            if isinstance(raw_al, str):
                clean = raw_al.strip()
                if clean:
                    aliases.append(AliasRecord(alias=clean, entity_id=entity_id, kind="permanent"))
            elif isinstance(raw_al, dict):
                al_data = dict(raw_al)
                al_data.setdefault("entity_id", entity_id)
                al_data.setdefault("kind", "permanent")
                try:
                    aliases.append(AliasRecord.model_validate(al_data))
                except Exception as exc:
                    raise ValueError(
                        f"JSON entities[{idx}].aliases[{a_idx}]: invalid alias record: {exc}"
                    ) from exc
            else:
                raise ValueError(f"JSON entities[{idx}].aliases[{a_idx}]: expected string or object.")

        entity = EntityRecord(
            entity_id=entity_id,
            kind=kind,
            canonical_name=canonical_name,
            base_attributes=dict(base_attributes),
            state=dict(state),
            private_state=dict(private_state),
        )
        parsed_entry = _ParsedEntityEntry(
            entity=entity,
            aliases=aliases,
            has_state=has_state,
            has_private_state=has_private_state,
        )
        entries.append(parsed_entry)
        entries_by_id[entity_id] = parsed_entry

    top_aliases = doc.get("aliases")
    if top_aliases is not None:
        if not isinstance(top_aliases, list):
            raise ValueError("JSON bible 'aliases' must be a list.")
        for a_idx, raw_al in enumerate(top_aliases):
            if not isinstance(raw_al, dict):
                raise ValueError(f"JSON aliases[{a_idx}] must be an object.")
            al_data = dict(raw_al)
            al_data.setdefault("kind", "permanent")
            try:
                al_rec = AliasRecord.model_validate(al_data)
            except Exception as exc:
                raise ValueError(f"JSON aliases[{a_idx}]: invalid alias record: {exc}") from exc
            target_entry = entries_by_id.get(al_rec.entity_id)
            if target_entry is None:
                warnings.append(
                    f"JSON aliases[{a_idx}]: alias '{al_rec.alias}' references unknown entity_id '{al_rec.entity_id}'; skipped."
                )
            else:
                target_entry.aliases.append(al_rec)

    has_edges_section = "edges" in doc
    edges: list[DirectedEdgeRecord] = []
    raw_edges = doc.get("edges") or []
    if not isinstance(raw_edges, list):
        raise ValueError("JSON bible 'edges' must be a list.")
    for e_idx, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, dict):
            raise ValueError(f"JSON edges[{e_idx}] must be an object.")
        try:
            edges.append(DirectedEdgeRecord.model_validate(raw_edge))
        except Exception as exc:
            raise ValueError(f"JSON edges[{e_idx}]: invalid edge record: {exc}") from exc

    return _ParsedBible(
        entries=entries,
        edges=edges,
        has_edges_section=has_edges_section,
        warnings=warnings,
    )
