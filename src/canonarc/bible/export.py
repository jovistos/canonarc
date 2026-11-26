from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..models import AliasRecord, DirectedEdgeRecord, EntityRecord
from .models import (
    _CSV_CORE_COLS,
    _CSV_RESERVED_COLS,
    _CSV_STANDARD_ATTR_COLS,
    _KIND_HEADINGS,
    BIBLE_SCHEMA_VERSION,
    BibleExportFormat,
    BibleExportResult,
    _escape_md_cell,
    _serialize_csv_value,
    infer_export_format,
)

if TYPE_CHECKING:
    from ..engine import CanonArcEngine


def render_bible(
    engine: CanonArcEngine,
    *,
    format: BibleExportFormat,
    at_step: int | None = None,
    timeline_id: str | None = None,
    kinds: Sequence[str] | None = None,
    include_state: bool = False,
    include_edges: bool | None = None,
    include_aliases: bool = True,
    include_inflections: bool | None = None,
    lang: str | None = None,
) -> tuple[str, BibleExportResult]:
    """Renders the active world Bible at `(timeline_id, at_step)` into CSV, JSON, or Markdown."""
    eff_include_edges = (
        engine._policy.has_tier(2) if include_edges is None else include_edges
    )
    eff_include_inflections = (
        (format == "json") if include_inflections is None else include_inflections
    )
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
            if not eff_include_inflections and a.kind == "inflection":
                continue
            if lang_filter is not None and (a.lang or "").lower() != lang_filter:
                continue
            aliases_by_entity[a.entity_id].append(a)
        for al_list in aliases_by_entity.values():
            al_list.sort(key=lambda a: (a.alias.lower(), a.kind, a.lang or ""))
            total_aliases += len(al_list)

    edges: list[DirectedEdgeRecord] = []
    if eff_include_edges and format in ("json", "md"):
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
            include_edges=eff_include_edges,
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
            include_edges=eff_include_edges,
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
    include_edges: bool | None = None,
    include_aliases: bool = True,
    include_inflections: bool | None = None,
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
        include_inflections=include_inflections,
        lang=lang,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if (eff_format == "csv" and excel_bom) else "utf-8"
    out_path.write_text(content, encoding=encoding)
    return result.model_copy(update={"path": str(out_path)})
