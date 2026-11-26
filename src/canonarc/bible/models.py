from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..models import (
    ENTITY_KIND_HEADINGS,
    ENTITY_KIND_PREFIXES,
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
)

BibleExportFormat = Literal["csv", "json", "md"]
BibleImportFormat = Literal["csv", "json"]
BibleSyncMode = Literal["upsert", "merge", "replace"]

BIBLE_SCHEMA_VERSION = 1

_KIND_PREFIXES: dict[str, str] = ENTITY_KIND_PREFIXES

_KIND_HEADINGS: dict[str, str] = ENTITY_KIND_HEADINGS

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

_CSV_PROVENANCE_COLS: tuple[str, ...] = (
    "first_seen_step",
    "first_seen_source",
)

_CSV_READONLY_COLS: frozenset[str] = frozenset(_CSV_PROVENANCE_COLS)

_PROVENANCE_METADATA_KEYS: tuple[str, ...] = (
    "source_ref",
    "chapter_id",
    "filename",
    "source_file",
    "label",
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
    skipped: bool = False
    content_hash: str | None = None
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
        if self.skipped:
            return (
                f"{prefix}[SKIPPED - unchanged] Bible sync "
                f"({self.format}, mode={self.mode}, timeline={self.timeline_id}, step={self.step}): "
                f"{self.source}"
            )
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
