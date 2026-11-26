from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .models import (
    AliasRecord,
    DirectedEdgeRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    StepDelta,
)

if TYPE_CHECKING:
    from .engine import CanonArcEngine

BibleExportFormat = Literal["csv", "json", "md"]
BibleImportFormat = Literal["csv", "json"]
BibleSyncMode = Literal["upsert", "merge", "replace"]

BIBLE_SCHEMA_VERSION = 1

_KIND_PREFIXES: dict[str, str] = {
    "character": "char",
    "location": "loc",
    "artifact": "item",
    "item": "item",
    "faction": "fac",
    "term": "term",
}

_KIND_HEADINGS: dict[str, str] = {
    "character": "Characters",
    "location": "Locations",
    "faction": "Factions",
    "artifact": "Artifacts",
    "item": "Items",
    "term": "Terms & Glossary",
}

_CSV_STANDARD_ATTR_COLS: tuple[str, ...] = (
    "target_name",
    "target_lang",
    "gender",
    "declension",
)

_CSV_CORE_COLS: tuple[str, ...] = (
    "entity_id",
    "kind",
    "canonical_name",
    *_CSV_STANDARD_ATTR_COLS,
    "aliases",
)

_CSV_RESERVED_COLS: frozenset[str] = frozenset(
    {
        "entity_id",
        "kind",
        "canonical_name",
        "aliases",
        "state",
        "private_state",
    }
)

_INFLECTION_SENSITIVE_ATTRS: tuple[str, ...] = (
    "target_name",
    "target_lang",
    "lang",
    "gender",
    "declension",
)


class BibleExportResult(BaseModel):
    """Metadata returned by export_bible()."""

    path: str | None = None
    format: BibleExportFormat
    timeline_id: str
    at_step: int
    entity_count: int = 0
    alias_count: int = 0
    edge_count: int = 0


class BibleSyncReport(BaseModel):
    """Structured diff & audit report returned by sync_bible()."""

    source: str
    format: BibleImportFormat
    timeline_id: str
    step: int
    mode: BibleSyncMode
    dry_run: bool = False
    created_entities: list[str] = Field(default_factory=list)
    updated_entities: list[str] = Field(default_factory=list)
    removed_entities: list[str] = Field(default_factory=list)
    unchanged_entities: list[str] = Field(default_factory=list)
    added_aliases: list[tuple[str, str]] = Field(default_factory=list)
    removed_aliases: list[tuple[str, str]] = Field(default_factory=list)
    created_edges: list[str] = Field(default_factory=list)
    updated_edges: list[str] = Field(default_factory=list)
    removed_edges: list[str] = Field(default_factory=list)
    unchanged_edges: list[str] = Field(default_factory=list)
    inflections_generated: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.created_entities
            or self.updated_entities
            or self.removed_entities
            or self.added_aliases
            or self.removed_aliases
            or self.created_edges
            or self.updated_edges
            or self.removed_edges
        )

    def summary(self) -> str:
        prefix = "[DRY RUN] " if self.dry_run else ""
        parts = [
            f"{prefix}Bible sync ({self.format}, mode={self.mode}, timeline={self.timeline_id}, step={self.step}):",
            (
                f"  Entities: +{len(self.created_entities)} created, "
                f"~{len(self.updated_entities)} updated, "
                f"-{len(self.removed_entities)} removed, "
                f"={len(self.unchanged_entities)} unchanged"
            ),
            f"  Aliases:  +{len(self.added_aliases)} added, -{len(self.removed_aliases)} removed",
        ]
        if self.created_edges or self.updated_edges or self.removed_edges or self.unchanged_edges:
            parts.append(
                f"  Edges:    +{len(self.created_edges)} created, "
                f"~{len(self.updated_edges)} updated, "
                f"-{len(self.removed_edges)} removed, "
                f"={len(self.unchanged_edges)} unchanged"
            )
        if self.inflections_generated:
            parts.append(f"  Inflections generated: {self.inflections_generated}")
        if self.warnings:
            parts.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                parts.append(f"    - {w}")
        return "\n".join(parts)


class _ParsedEntityEntry(BaseModel):
    entity: EntityRecord
    aliases: list[AliasRecord] = Field(default_factory=list)
    has_state: bool = False
    has_private_state: bool = False


class _ParsedBible(BaseModel):
    entries: list[_ParsedEntityEntry] = Field(default_factory=list)
    edges: list[DirectedEdgeRecord] = Field(default_factory=list)
    has_edges_section: bool = False
    warnings: list[str] = Field(default_factory=list)


def infer_export_format(path: str | Path | None, fmt: Literal["auto", "csv", "json", "md"]) -> BibleExportFormat:
    if fmt != "auto":
        return fmt
    if path is None or str(path) == "-":
        raise ValueError("Cannot infer export format for stdout; specify format='csv'|'json'|'md'.")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in (".md", ".markdown"):
        return "md"
    raise ValueError(
        f"Cannot infer bible format from extension {suffix!r}; pass format='csv'|'json'|'md'."
    )


def infer_import_format(path: str | Path, fmt: Literal["auto", "csv", "json"]) -> BibleImportFormat:
    if fmt != "auto":
        if fmt not in ("csv", "json"):
            raise ValueError(f"Unsupported import format {fmt!r}; use 'csv' or 'json'.")
        return fmt
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in (".md", ".markdown"):
        raise ValueError("Markdown import is not supported in v1; use JSON or CSV for sync_bible().")
    raise ValueError(
        f"Cannot infer bible import format from extension {suffix!r}; pass format='csv'|'json'."
    )


def _slugify_entity_id(kind: str, canonical_name: str, existing_ids: set[str]) -> str:
    prefix = _KIND_PREFIXES.get(kind.strip().lower())
    if not prefix:
        clean_kind = re.sub(r"[^a-z0-9]+", "", kind.strip().lower())
        prefix = clean_kind[:4] if clean_kind else "ent"
    normalized = unicodedata.normalize("NFKD", canonical_name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        slug = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:8]
    candidate = f"{prefix}_{slug}"
    if candidate not in existing_ids:
        return candidate
    idx = 2
    while f"{candidate}_{idx}" in existing_ids:
        idx += 1
    return f"{candidate}_{idx}"


def _edge_key(source_id: str, target_id: str, relation_type: str) -> str:
    return f"{source_id}:{relation_type}->{target_id}"


def _alias_signature(a: AliasRecord) -> tuple[str, str, str | None, tuple[str, ...] | None, str | None]:
    return (
        a.alias.strip().lower(),
        a.kind,
        a.lang.strip().lower() if a.lang else None,
        tuple(sorted(a.known_to)) if a.known_to is not None else None,
        a.persona_name,
    )


def _serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


# --- Export / Rendering ---


def render_bible(
    engine: CanonArcEngine,
    *,
    format: BibleExportFormat,
    at_step: int | None = None,
    timeline_id: str | None = None,
    kinds: Sequence[str] | None = None,
    include_state: bool = False,
    include_edges: bool = True,
    include_aliases: bool = True,
    lang: str | None = None,
) -> tuple[str, BibleExportResult]:
    """Renders the active world Bible at `(timeline_id, at_step)` into CSV, JSON, or Markdown."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")
    eff_step = tl_info.head_step if at_step is None else at_step

    entities = engine.store.get_all_active_entities(tl, eff_step)
    kind_filter = {k.strip().lower() for k in kinds if k.strip()} if kinds else None
    if kind_filter is not None:
        entities = [e for e in entities if e.kind.lower() in kind_filter]
    entities.sort(key=lambda e: (e.kind.lower(), e.entity_id))
    kept_entity_ids = {e.entity_id for e in entities}

    aliases_by_entity: dict[str, list[AliasRecord]] = {e.entity_id: [] for e in entities}
    total_aliases = 0
    if include_aliases:
        raw_aliases = engine.store.get_all_active_aliases(tl, eff_step)
        lang_filter = lang.strip().lower() if lang and lang.strip() else None
        for a in raw_aliases:
            if a.entity_id not in kept_entity_ids:
                continue
            if lang_filter is not None and (a.lang or "").lower() != lang_filter:
                continue
            aliases_by_entity[a.entity_id].append(a)
        for al_list in aliases_by_entity.values():
            al_list.sort(key=lambda a: (a.alias.lower(), a.kind, a.lang or ""))
            total_aliases += len(al_list)

    edges: list[DirectedEdgeRecord] = []
    if include_edges and format in ("json", "md"):
        raw_edges = engine.store.get_all_active_edges(tl, eff_step)
        if kind_filter is not None:
            edges = [
                ed
                for ed in raw_edges
                if ed.source_id in kept_entity_ids and ed.target_id in kept_entity_ids
            ]
        else:
            edges = list(raw_edges)
        edges.sort(key=lambda ed: (ed.source_id, ed.relation_type, ed.target_id))

    if format == "json":
        content = _render_json(
            timeline_id=tl,
            at_step=eff_step,
            entities=entities,
            aliases_by_entity=aliases_by_entity,
            edges=edges,
            include_state=include_state,
            include_aliases=include_aliases,
            include_edges=include_edges,
        )
    elif format == "csv":
        content = _render_csv(
            entities=entities,
            aliases_by_entity=aliases_by_entity,
            include_state=include_state,
        )
    elif format == "md":
        content = _render_markdown(
            timeline_id=tl,
            at_step=eff_step,
            entities=entities,
            aliases_by_entity=aliases_by_entity,
            edges=edges,
            include_state=include_state,
            include_aliases=include_aliases,
            include_edges=include_edges,
        )
    else:
        raise ValueError(f"Unsupported export format: {format!r}")

    result = BibleExportResult(
        path=None,
        format=format,
        timeline_id=tl,
        at_step=eff_step,
        entity_count=len(entities),
        alias_count=total_aliases,
        edge_count=len(edges),
    )
    return content, result


def _render_json(
    *,
    timeline_id: str,
    at_step: int,
    entities: Sequence[EntityRecord],
    aliases_by_entity: dict[str, list[AliasRecord]],
    edges: Sequence[DirectedEdgeRecord],
    include_state: bool,
    include_aliases: bool,
    include_edges: bool,
) -> str:
    ent_payloads: list[dict[str, Any]] = []
    for e in entities:
        item: dict[str, Any] = {
            "entity_id": e.entity_id,
            "kind": e.kind,
            "canonical_name": e.canonical_name,
            "base_attributes": dict(e.base_attributes),
        }
        if include_state:
            item["state"] = dict(e.state)
            if e.private_state:
                item["private_state"] = dict(e.private_state)
        if include_aliases:
            item["aliases"] = [
                {
                    k: v
                    for k, v in {
                        "alias": a.alias,
                        "kind": a.kind,
                        "lang": a.lang,
                        "known_to": a.known_to,
                        "persona_name": a.persona_name,
                    }.items()
                    if v is not None
                }
                for a in aliases_by_entity.get(e.entity_id, ())
            ]
        ent_payloads.append(item)

    doc: dict[str, Any] = {
        "schema_version": BIBLE_SCHEMA_VERSION,
        "timeline_id": timeline_id,
        "at_step": at_step,
        "entities": ent_payloads,
    }
    if include_edges:
        doc["edges"] = [
            {
                "source_id": ed.source_id,
                "target_id": ed.target_id,
                "relation_type": ed.relation_type,
                "state": dict(ed.state),
            }
            for ed in edges
        ]
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def _render_csv(
    *,
    entities: Sequence[EntityRecord],
    aliases_by_entity: dict[str, list[AliasRecord]],
    include_state: bool,
) -> str:
    extra_attr_keys: set[str] = set()
    any_private_state = False
    for e in entities:
        for k in e.base_attributes:
            if k not in _CSV_STANDARD_ATTR_COLS and k not in _CSV_RESERVED_COLS:
                extra_attr_keys.add(k)
        if include_state and e.private_state:
            any_private_state = True

    fieldnames: list[str] = [*_CSV_CORE_COLS, *sorted(extra_attr_keys)]
    if include_state:
        fieldnames.append("state")
        if any_private_state:
            fieldnames.append("private_state")

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for e in entities:
        row: dict[str, str] = {
            "entity_id": e.entity_id,
            "kind": e.kind,
            "canonical_name": e.canonical_name,
            "aliases": "; ".join(a.alias for a in aliases_by_entity.get(e.entity_id, ())),
        }
        for col in _CSV_STANDARD_ATTR_COLS:
            row[col] = _serialize_csv_value(e.base_attributes.get(col))
        for col in sorted(extra_attr_keys):
            row[col] = _serialize_csv_value(e.base_attributes.get(col))
        if include_state:
            row["state"] = json.dumps(e.state, ensure_ascii=False, separators=(",", ":")) if e.state else ""
            if any_private_state:
                row["private_state"] = (
                    json.dumps(e.private_state, ensure_ascii=False, separators=(",", ":"))
                    if e.private_state
                    else ""
                )
        writer.writerow(row)
    return buf.getvalue()


def _render_markdown(
    *,
    timeline_id: str,
    at_step: int,
    entities: Sequence[EntityRecord],
    aliases_by_entity: dict[str, list[AliasRecord]],
    edges: Sequence[DirectedEdgeRecord],
    include_state: bool,
    include_aliases: bool,
    include_edges: bool,
) -> str:
    lines: list[str] = [f"# Series Bible — `{timeline_id}` (Step {at_step})", ""]

    if not entities and not edges:
        lines.append("_No active entities or relationships at this step._")
        return "\n".join(lines) + "\n"

    grouped: dict[str, list[EntityRecord]] = {}
    for e in entities:
        grouped.setdefault(e.kind.lower(), []).append(e)

    kind_order = list(_KIND_HEADINGS.keys()) + sorted(k for k in grouped if k not in _KIND_HEADINGS)
    for kind_key in kind_order:
        group = grouped.get(kind_key)
        if not group:
            continue
        heading = _KIND_HEADINGS.get(kind_key, kind_key.replace("_", " ").title())
        lines.append(f"## {heading}")
        lines.append("")

        headers = ["ID", "Canonical Name", "Attributes"]
        if include_aliases:
            headers.append("Aliases")
        if include_state:
            headers.append("State")

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        for e in group:
            attr_str = ", ".join(f"`{k}`={v}" for k, v in e.base_attributes.items()) or "—"
            cells = [
                f"`{_escape_md_cell(e.entity_id)}`",
                _escape_md_cell(e.canonical_name),
                _escape_md_cell(attr_str),
            ]
            if include_aliases:
                al_parts: list[str] = []
                for a in aliases_by_entity.get(e.entity_id, ()):
                    tags = [t for t in (a.kind if a.kind != "permanent" else None, a.lang) if t]
                    suffix = f" ({', '.join(tags)})" if tags else ""
                    al_parts.append(f"{a.alias}{suffix}")
                cells.append(_escape_md_cell("; ".join(al_parts) or "—"))
            if include_state:
                state_parts = [f"`{k}`={v}" for k, v in e.state.items()]
                if e.private_state:
                    priv = ", ".join(f"{k}={v}" for k, v in e.private_state.items())
                    state_parts.append(f"private: {priv}")
                cells.append(_escape_md_cell(", ".join(state_parts) or "—"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if include_edges and edges:
        lines.append("## Relationships & Registers")
        lines.append("")
        for ed in edges:
            pairs = ", ".join(f"`{k}`={v}" for k, v in ed.state.items()) or "active"
            lines.append(f"- `{ed.source_id}` → `{ed.target_id}` [`{ed.relation_type}`]: {pairs}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_bible(
    engine: CanonArcEngine,
    path: str | Path,
    *,
    format: Literal["auto", "csv", "json", "md"] = "auto",
    at_step: int | None = None,
    timeline_id: str | None = None,
    kinds: Sequence[str] | None = None,
    include_state: bool = False,
    include_edges: bool = True,
    include_aliases: bool = True,
    lang: str | None = None,
    excel_bom: bool = False,
) -> BibleExportResult:
    """Exports the active world Bible at `(timeline_id, at_step)` to CSV, JSON, or Markdown."""
    eff_format = infer_export_format(path, format)
    content, result = render_bible(
        engine,
        format=eff_format,
        at_step=at_step,
        timeline_id=timeline_id,
        kinds=kinds,
        include_state=include_state,
        include_edges=include_edges,
        include_aliases=include_aliases,
        lang=lang,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if (eff_format == "csv" and excel_bom) else "utf-8"
    out_path.write_text(content, encoding=encoding)
    return result.model_copy(update={"path": str(out_path)})


# --- Parsing ---


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

        # Skip completely blank rows
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

    # Also accept optional top-level 'aliases' list for convenience
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


# --- Sync / Diff Engine ---


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
    active_aliases_by_ent: dict[str, dict[str, AliasRecord]] = {}
    for a in engine.store.get_all_active_aliases(tl, step):
        active_aliases_by_ent.setdefault(a.entity_id, {})[a.alias.strip().lower()] = a
    active_edges = {
        _edge_key(ed.source_id, ed.target_id, ed.relation_type): ed
        for ed in engine.store.get_all_active_edges(tl, step)
    }

    if eff_format == "csv":
        parsed = _parse_csv_bible(raw_text, column_map=column_map, existing_ids=set(active_entities.keys()))
    else:
        parsed = _parse_json_bible(raw_text, existing_ids=set(active_entities.keys()))

    report = BibleSyncReport(
        source=str(src_path),
        format=eff_format,
        timeline_id=tl,
        step=step,
        mode=mode,
        dry_run=dry_run,
        warnings=list(parsed.warnings),
    )

    # Check for alias collisions across incoming entities
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

    reason_tag = f"bible_sync:{src_path.name}"

    entities_to_register: list[EntityRecord] = []
    entity_deltas: list[EntityDelta] = []
    edge_deltas: list[EdgeDelta] = []
    mutated_entity_ids: list[str] = []
    mutated_edge_keys: list[str] = []

    incoming_entity_ids: set[str] = set()

    for entry in parsed.entries:
        inc = entry.entity
        eid = inc.entity_id
        incoming_entity_ids.add(eid)
        existing = active_entities.get(eid)

        if existing is None:
            merged_base = dict(inc.base_attributes)
            merged_state = dict(inc.state) if entry.has_state else {}
            merged_priv = dict(inc.private_state) if entry.has_private_state else {}
            resolved_entity = EntityRecord(
                entity_id=eid,
                kind=inc.kind,
                canonical_name=inc.canonical_name,
                base_attributes=merged_base,
                state=merged_state,
                private_state=merged_priv,
                reason=reason_tag,
            )

            desired_aliases = list(entry.aliases)
            extra_target_aliases = engine._build_target_aliases(
                resolved_entity,
                desired_aliases,
                target_lang=target_lang,
                auto_inflect=auto_inflect and not dry_run,
            )
            if extra_target_aliases:
                raw_target = str(resolved_entity.base_attributes.get("target_name", "")).strip().lower()
                inflection_count = sum(
                    1 for a in extra_target_aliases if a.alias.strip().lower() != raw_target
                )
                report.inflections_generated += inflection_count
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

        # Existing entity: compute target state based on sync mode
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
            # 'upsert' or 'replace': row is authoritative for base_attributes
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

        # Smart auto_inflect gating: only fire inflection generator if target_name/lang/gender/declension changed
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

        # Build desired alias map for this entity
        desired_alias_map: dict[str, AliasRecord] = {
            a.alias.strip().lower(): a for a in entry.aliases
        }
        should_inflect_now = auto_inflect and inflection_inputs_changed and not dry_run
        extra_target_aliases = engine._build_target_aliases(
            resolved_entity,
            list(desired_alias_map.values()),
            target_lang=target_lang,
            auto_inflect=should_inflect_now,
        )
        if extra_target_aliases:
            raw_target = str(resolved_entity.base_attributes.get("target_name", "")).strip().lower()
            if should_inflect_now:
                report.inflections_generated += sum(
                    1 for a in extra_target_aliases if a.alias.strip().lower() != raw_target
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
            # In CSV mode, preserve existing target-language inflections if target_name inputs didn't change
            raw_lang = (
                target_lang
                or new_base.get("target_lang")
                or new_base.get("lang")
            )
            eff_lang = raw_lang.strip().lower() if isinstance(raw_lang, str) and raw_lang.strip() else None
            for al_key, cur in existing_alias_map.items():
                if al_key in desired_alias_map:
                    continue
                if (
                    eff_format == "csv"
                    and auto_inflect
                    and not inflection_inputs_changed
                    and eff_lang is not None
                    and cur.lang == eff_lang
                    and cur.kind == "permanent"
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

    # Handle entity removals in 'replace' mode
    if mode == "replace":
        for eid in sorted(active_entities.keys()):
            if eid not in incoming_entity_ids:
                report.removed_entities.append(eid)
                mutated_entity_ids.append(eid)
                entity_deltas.append(EntityDelta(entity_id=eid, remove=True, reason=reason_tag))

    # Diff edges (JSON only in v1)
    if parsed.has_edges_section:
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

    # Materialized-snapshot shadowing guardrail: warn if step < head_step and later revisions exist
    if step < tl_info.head_step:
        for eid in mutated_entity_ids:
            hist = engine.store.get_history(eid, timeline_id=tl, record_type="entity")
            later_steps = [h["valid_from_step"] for h in hist if h["valid_from_step"] > step]
            if later_steps:
                report.warnings.append(
                    f"Shadowed update: entity '{eid}' has later revision(s) at step(s) {later_steps} "
                    f"(head={tl_info.head_step}). Syncing at step={step} will be shadowed at steps >= {min(later_steps)}; "
                    f"sync at step={tl_info.head_step} or roll back first."
                )
        for ekey in mutated_edge_keys:
            hist = engine.store.get_history(ekey, timeline_id=tl, record_type="edge")
            later_steps = [h["valid_from_step"] for h in hist if h["valid_from_step"] > step]
            if later_steps:
                report.warnings.append(
                    f"Shadowed update: edge '{ekey}' has later revision(s) at step(s) {later_steps} "
                    f"(head={tl_info.head_step}). Syncing at step={step} will be shadowed at steps >= {min(later_steps)}."
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
