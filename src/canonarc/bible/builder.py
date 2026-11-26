from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..audit.models import DuplicateCandidate
from ..audit.runner import find_duplicate_candidates
from ..extractors import (
    JsonCompletionClient,
    TokenUsage,
    _coerce_usage,
    _strip_json_fences,
    resolve_client_model,
    validate_json_completion_client,
)
from ..models import AliasRecord, EntityRecord
from ..text_masking import compile_ignore_patterns
from .export import _render_csv, _render_json
from .models import (
    BibleSyncMode,
    BibleSyncReport,
    _ParsedBible,
    _ParsedEntityEntry,
    _slugify_entity_id,
    infer_import_format,
)
from .sync import _sync_parsed_bible

if TYPE_CHECKING:
    from ..engine import CanonArcEngine

DEFAULT_ALLOWED_KINDS: tuple[str, ...] = (
    "character",
    "location",
    "faction",
    "organization",
    "artifact",
    "term",
)

_RESERVED_ENTITY_KEYS: frozenset[str] = frozenset(
    {
        "entity_id",
        "canonical_name",
        "kind",
        "aliases",
        "attributes",
        "base_attributes",
        "target_name",
        "target_lang",
        "gender",
        "declension",
        "notes",
    }
)


class BibleSection(BaseModel):
    """A single ordered chunk of source text or NER-filtered evidence fed to BibleBuilder."""

    section_id: str
    text: str
    title: str | None = None


class BibleConflict(BaseModel):
    """Records a contradiction between a locked entity field and a later section's proposal."""

    section_id: str
    entity_id: str
    canonical_name: str
    field: str
    locked_value: Any
    proposed_value: Any
    resolution: Literal["kept_locked", "updated"] = "kept_locked"


class BibleDraft(BaseModel):
    """In-memory Series Bible draft produced by BibleBuilder before or alongside Step 0 commit."""

    entities: list[EntityRecord] = Field(default_factory=list)
    aliases: dict[str, list[AliasRecord]] = Field(default_factory=dict)
    conflicts: list[BibleConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sections_processed: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)

    def to_parsed_bible(self) -> _ParsedBible:
        """Converts this draft into the internal _ParsedBible structure accepted by sync_bible."""
        parsed_entries = [
            _ParsedEntityEntry(
                entity=ent.model_copy(deep=True),
                aliases=[a.model_copy(deep=True) for a in self.aliases.get(ent.entity_id, ())],
                has_state=bool(ent.state),
                has_private_state=bool(ent.private_state),
            )
            for ent in self.entities
        ]
        return _ParsedBible(
            entries=parsed_entries,
            edges=[],
            has_edges_section=False,
            warnings=list(self.warnings),
        )

    def render_json(self, *, timeline_id: str = "main", at_step: int = 0) -> str:
        """Renders this draft in CanonArc's canonical JSON bible format (round-trip with sync_bible)."""
        sorted_entities = sorted(self.entities, key=lambda e: (e.kind.lower(), e.entity_id))
        aliases_map = {
            e.entity_id: sorted(
                self.aliases.get(e.entity_id, ()),
                key=lambda a: (a.alias.lower(), a.kind, a.lang or ""),
            )
            for e in sorted_entities
        }
        return _render_json(
            timeline_id=timeline_id,
            at_step=at_step,
            entities=sorted_entities,
            aliases_by_entity=aliases_map,
            edges=[],
            include_state=False,
            include_aliases=True,
            include_edges=False,
        )

    def render_csv(self) -> str:
        """Renders this draft in CanonArc's canonical CSV glossary format (round-trip with sync_bible)."""
        sorted_entities = sorted(self.entities, key=lambda e: (e.kind.lower(), e.entity_id))
        aliases_map = {
            e.entity_id: sorted(
                self.aliases.get(e.entity_id, ()),
                key=lambda a: (a.alias.lower(), a.kind, a.lang or ""),
            )
            for e in sorted_entities
        }
        return _render_csv(
            entities=sorted_entities,
            aliases_by_entity=aliases_map,
            include_state=False,
        )

    def export(
        self,
        path: str | Path,
        *,
        format: Literal["auto", "csv", "json"] = "auto",
        timeline_id: str = "main",
        at_step: int = 0,
        excel_bom: bool = False,
    ) -> Path:
        """Writes this draft to a JSON or CSV file compatible with sync_bible()."""
        out_path = Path(path)
        eff_format = infer_import_format(out_path, format)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if eff_format == "json":
            content = self.render_json(timeline_id=timeline_id, at_step=at_step)
            out_path.write_text(content, encoding="utf-8")
        else:
            content = self.render_csv()
            encoding = "utf-8-sig" if excel_bom else "utf-8"
            out_path.write_text(content, encoding=encoding)
        return out_path

    def summary(self) -> str:
        total_aliases = sum(len(v) for v in self.aliases.values())
        parts = [
            f"BibleDraft: {len(self.entities)} entities, {total_aliases} aliases "
            f"({self.sections_processed} sections, {self.api_calls} API calls, {self.cache_hits} cache hits)",
        ]
        if self.conflicts:
            parts.append(f"  Conflicts ({len(self.conflicts)}):")
            for c in self.conflicts:
                parts.append(
                    f"    - [{c.section_id}] {c.canonical_name} ({c.entity_id}).{c.field}: "
                    f"kept {c.locked_value!r} over {c.proposed_value!r}"
                )
        if self.warnings:
            parts.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                parts.append(f"    - {w}")
        return "\n".join(parts)


class BibleBootstrapReport(BaseModel):
    """Combined result of bootstrap_bible(): draft + optional Step 0 sync + duplicate candidates."""

    draft: BibleDraft
    sync_report: BibleSyncReport | None = None
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    export_path: str | None = None

    def summary(self) -> str:
        lines = [self.draft.summary()]
        if self.export_path:
            lines.append(f"  Exported draft: {self.export_path}")
        if self.sync_report is not None:
            lines.append(self.sync_report.summary())
        if self.duplicate_candidates:
            lines.append(f"  Duplicate candidates ({len(self.duplicate_candidates)}):")
            for cand in self.duplicate_candidates:
                lines.append(
                    f"    - {cand.duplicate_name} ({cand.duplicate_id}) <-> "
                    f"{cand.canonical_name} ({cand.canonical_id}) [score={cand.score:.1f}]"
                )
        return "\n".join(lines)


def _coerce_section(item: BibleSection | Mapping[str, Any] | str, idx: int) -> BibleSection:
    if isinstance(item, BibleSection):
        return item
    if isinstance(item, str):
        return BibleSection(section_id=f"section_{idx + 1:03d}", text=item)
    if isinstance(item, Mapping):
        section_id = str(item.get("section_id") or f"section_{idx + 1:03d}").strip()
        text = str(item.get("text") or "")
        raw_title = item.get("title")
        title = str(raw_title).strip() if raw_title is not None and str(raw_title).strip() else None
        return BibleSection(section_id=section_id, text=text, title=title)
    raise TypeError(f"Unsupported BibleSection item type: {type(item).__name__}")


def _format_locked_ledger(
    entities: Sequence[EntityRecord],
    aliases_by_entity: Mapping[str, Sequence[AliasRecord]],
) -> str:
    """Renders a compact one-line-per-entity locked ledger for forward propagation."""
    if not entities:
        return "(none yet)"
    lines: list[str] = []
    for ent in entities:
        parts = [f"- {ent.entity_id} | {ent.canonical_name} [{ent.kind}]"]
        target_name = ent.base_attributes.get("target_name")
        if isinstance(target_name, str) and target_name.strip():
            parts.append(f" -> '{target_name.strip()}'")
        meta: list[str] = []
        gender = ent.base_attributes.get("gender")
        if isinstance(gender, str) and gender.strip():
            meta.append(gender.strip())
        decl = ent.base_attributes.get("declension")
        if isinstance(decl, str) and decl.strip():
            meta.append(decl.strip())
        if meta:
            parts.append(f" ({', '.join(meta)})")
        ent_aliases = [
            a.alias
            for a in aliases_by_entity.get(ent.entity_id, ())
            if a.kind != "inflection" and a.alias.strip()
        ]
        if ent_aliases:
            parts.append(f" | aliases: {', '.join(ent_aliases[:8])}")
        lines.append("".join(parts))
    return "\n".join(lines)


def _parse_builder_response(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parses a section extraction JSON payload into (entities, alias_updates)."""
    cleaned = _strip_json_fences(raw).strip()
    if not cleaned.startswith(("{", "[")):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1]
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        entities = [item for item in parsed if isinstance(item, dict)]
        return entities, []
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object or list from BibleBuilder, got {type(parsed).__name__}.")
    raw_entities = parsed.get("entities")
    if raw_entities is None:
        raw_entities = parsed.get("entries") or parsed.get("items") or []
    if not isinstance(raw_entities, list):
        raise ValueError("BibleBuilder JSON field 'entities' must be a list.")
    raw_updates = parsed.get("alias_updates") or []
    if not isinstance(raw_updates, list):
        raise ValueError("BibleBuilder JSON field 'alias_updates' must be a list.")
    entities = [item for item in raw_entities if isinstance(item, dict)]
    updates = [item for item in raw_updates if isinstance(item, dict)]
    return entities, updates


def _merge_notes(existing: str, incoming: str, *, max_chars: int) -> str:
    clean_existing = existing.strip()
    clean_incoming = incoming.strip()
    if not clean_incoming:
        return clean_existing
    if not clean_existing:
        return clean_incoming[:max_chars] if max_chars > 0 else clean_incoming
    if clean_incoming.lower() in clean_existing.lower():
        return clean_existing
    if clean_existing.lower() in clean_incoming.lower():
        merged = clean_incoming
    else:
        merged = f"{clean_existing}; {clean_incoming}"
    if max_chars > 0 and len(merged) > max_chars:
        return merged[: max_chars - 1].rstrip() + "…"
    return merged


class BibleBuilder:
    """Domain-agnostic sequential Series Bible accumulator backed by any JsonCompletionClient.

    Feeds sections in reading order, propagates a compact locked-entity ledger forward,
    enforces immutable locks on canonical/target names with explicit conflict logging,
    and optionally caches per-section LLM responses on disk.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise Series Bible & terminology extraction engine. "
        "Maintain strict consistency with already-locked entities and respond ONLY with valid JSON."
    )

    def __init__(
        self,
        client: JsonCompletionClient,
        *,
        source_lang: str | None = None,
        target_lang: str | None = None,
        allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
        instructions: str = "",
        system_prompt: str | None = None,
        prompt_builder: Callable[[BibleSection, str, Sequence[EntityRecord]], str] | None = None,
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        max_notes_chars: int = 300,
        cache_dir: str | Path | None = None,
        bypass_cache: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 1,
    ) -> None:
        self.client: JsonCompletionClient = validate_json_completion_client(client)
        self.source_lang = source_lang.strip().lower() if source_lang and source_lang.strip() else None
        self.target_lang = target_lang.strip().lower() if target_lang and target_lang.strip() else None
        self.allowed_kinds: tuple[str, ...] = tuple(
            k.strip().lower() for k in allowed_kinds if k and k.strip()
        ) or DEFAULT_ALLOWED_KINDS
        self.instructions = instructions.strip()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt_builder = prompt_builder
        self._compiled_ignore = compile_ignore_patterns(ignore_patterns)
        self.max_notes_chars = max(0, max_notes_chars)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bypass_cache = bypass_cache
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max(0, max_retries)

        self.api_calls: int = 0
        self.cache_hits: int = 0
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def _model_name(self) -> str:
        return resolve_client_model(self.client)

    def _is_ignored_name(self, name: str) -> bool:
        clean = name.strip()
        if not clean:
            return True
        for pat in self._compiled_ignore:
            if pat.fullmatch(clean):
                return True
        return False

    def build_section_prompt(
        self,
        section: BibleSection,
        locked_ledger: str,
        locked_entities: Sequence[EntityRecord],
    ) -> str:
        """Builds the user prompt for a single section given the current locked-entity ledger."""
        if self.prompt_builder is not None:
            return self.prompt_builder(section, locked_ledger, locked_entities)

        kinds_str = ", ".join(self.allowed_kinds)
        header_title = f" ({section.title})" if section.title else ""
        if self.target_lang:
            lang_desc = (
                f"{self.source_lang} -> {self.target_lang}"
                if self.source_lang
                else f"target language '{self.target_lang}'"
            )
            schema_desc = (
                '{"entities": [{"canonical_name": "...", "kind": "...", '
                '"target_name": "...", "gender": "...", "declension": "...", '
                '"aliases": ["..."], "notes": "..."}], '
                '"alias_updates": [{"entity_id": "...", "add_aliases": ["..."]}]}'
            )
            mode_guidance = (
                f"Localization pair: {lang_desc}.\n"
                "- Provide `target_name` (canonical localized rendering), `gender` (when applicable), "
                "`declension` (if relevant for target language morphology), `aliases` (other source-text names/epithets), "
                "and concise `notes`.\n"
            )
        else:
            schema_desc = (
                '{"entities": [{"canonical_name": "...", "kind": "...", '
                '"aliases": ["..."], "attributes": {}, "notes": "..."}], '
                '"alias_updates": [{"entity_id": "...", "add_aliases": ["..."]}]}'
            )
            mode_guidance = (
                "- Provide `canonical_name`, `kind`, `aliases` (other names/titles/epithets), "
                "optional `attributes` dict, and concise `notes`.\n"
            )

        extra_block = f"\nEditorial guidelines:\n{self.instructions}\n" if self.instructions else ""
        return (
            f"Extract proper named entities from section '{section.section_id}'{header_title} "
            f"to build a canonical Series Bible.\n"
            f"Allowed kinds: {kinds_str}.\n"
            f"{mode_guidance}"
            "Rules:\n"
            "- Reuse already-locked entities below; DO NOT invent duplicate entities or change locked `target_name`s.\n"
            "- Only output newly introduced entities, OR existing entities for which this section reveals "
            "missing attributes (e.g. gender/declension/notes) or new aliases.\n"
            "- If a new epithet, title, or variant name refers to an already-locked entity, emit it in `alias_updates` "
            "referencing that entity's `entity_id`.\n"
            "- Ignore generic unnamed nouns (e.g. 'the guard', 'the street') unless they are recurring named entities.\n"
            f"{extra_block}\n"
            f"Already-locked entities:\n{locked_ledger}\n\n"
            f"Return ONLY valid JSON matching:\n{schema_desc}\n\n"
            f"Section '{section.section_id}' text:\n{section.text.strip()}"
        )

    def _cache_key(self, section: BibleSection, prompt: str) -> str:
        raw = (
            f"model={self._model_name()}\0"
            f"system={self.system_prompt}\0"
            f"section={section.section_id}\0"
            f"prompt={prompt}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _call_section(
        self,
        section: BibleSection,
        prompt: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], TokenUsage, bool]:
        key = self._cache_key(section, prompt)
        cache_path = (self.cache_dir / f"{key}.json") if self.cache_dir is not None else None
        if not self.bypass_cache and cache_path is not None and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                entities = [item for item in cached.get("entities", []) if isinstance(item, dict)]
                updates = [item for item in cached.get("alias_updates", []) if isinstance(item, dict)]
                self.cache_hits += 1
                return entities, updates, TokenUsage(cache_hits=1), True
            except Exception:  # noqa: BLE001, S110
                pass

        last_exc: Exception | None = None
        section_usage = TokenUsage()
        parsed_result: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None

        for attempt in range(self.max_retries + 1):
            content, raw_usage = self.client.call_json_api(
                prompt,
                system_prompt=self.system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.api_calls += 1
            call_usage = _coerce_usage(raw_usage, latency_ms=0.0)
            if attempt > 0:
                call_usage.retry_count += 1
            section_usage.add(call_usage)
            self.last_usage = call_usage
            self.total_usage.add(call_usage)
            try:
                parsed_result = _parse_builder_response(content)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                section_usage.validation_errors += 1
                self.total_usage.validation_errors += 1

        if parsed_result is None:
            assert last_exc is not None
            raise last_exc

        entities, updates = parsed_result
        if cache_path is not None:
            payload = {
                "cache_key": key,
                "model": self._model_name(),
                "section_id": section.section_id,
                "entities": entities,
                "alias_updates": updates,
                "usage": section_usage.to_dict(),
            }
            tmp_path = cache_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(cache_path)

        return entities, updates, section_usage, False

    def build(
        self,
        sections: Sequence[BibleSection | Mapping[str, Any] | str],
        *,
        initial_entities: Sequence[EntityRecord] = (),
        initial_aliases: Sequence[AliasRecord] = (),
    ) -> BibleDraft:
        """Runs the sequential accumulator across sections and returns an inspectable BibleDraft."""
        entities_by_id: dict[str, EntityRecord] = {}
        entity_order: list[str] = []
        aliases_by_entity: dict[str, list[AliasRecord]] = {}
        alias_keys_by_entity: dict[str, set[str]] = {}
        name_index: dict[str, list[str]] = {}  # lower(name or alias) -> list[entity_id]

        def _index_name(name: str, eid: str) -> None:
            key = name.strip().lower()
            if not key:
                return
            owners = name_index.setdefault(key, [])
            if eid not in owners:
                owners.append(eid)

        def _add_alias(eid: str, alias_rec: AliasRecord) -> bool:
            clean = alias_rec.alias.strip()
            if not clean or self._is_ignored_name(clean):
                return False
            low = clean.lower()
            ent = entities_by_id[eid]
            if low == ent.canonical_name.strip().lower() and alias_rec.lang is None:
                return False
            seen = alias_keys_by_entity.setdefault(eid, set())
            if low in seen:
                return False
            seen.add(low)
            aliases_by_entity.setdefault(eid, []).append(
                alias_rec.model_copy(update={"alias": clean, "entity_id": eid})
            )
            _index_name(clean, eid)
            return True

        # Pre-seed with initial_entities (e.g., Book 1 cast or manually curated locks)
        for ent in initial_entities:
            copy_ent = ent.model_copy(deep=True)
            if copy_ent.entity_id not in entities_by_id:
                entity_order.append(copy_ent.entity_id)
            entities_by_id[copy_ent.entity_id] = copy_ent
            aliases_by_entity.setdefault(copy_ent.entity_id, [])
            alias_keys_by_entity.setdefault(copy_ent.entity_id, set())
            _index_name(copy_ent.canonical_name, copy_ent.entity_id)
            raw_target = copy_ent.base_attributes.get("target_name")
            if isinstance(raw_target, str) and raw_target.strip():
                _index_name(raw_target, copy_ent.entity_id)

        for al in initial_aliases:
            if al.entity_id in entities_by_id:
                _add_alias(al.entity_id, al)

        conflicts: list[BibleConflict] = []
        warnings: list[str] = []
        run_usage = TokenUsage()
        run_api_calls = 0
        run_cache_hits = 0
        sections_processed = 0
        allowed_set = set(self.allowed_kinds)

        def _find_matching_entity(
            *,
            explicit_id: str | None,
            canonical_name: str,
            kind: str,
        ) -> EntityRecord | None:
            if explicit_id and explicit_id in entities_by_id:
                return entities_by_id[explicit_id]
            owners = name_index.get(canonical_name.strip().lower(), ())
            if not owners:
                return None
            for eid in owners:
                if entities_by_id[eid].kind.lower() == kind.lower():
                    return entities_by_id[eid]
            return entities_by_id[owners[0]]

        for idx, raw_sec in enumerate(sections):
            section = _coerce_section(raw_sec, idx)
            if not section.text.strip():
                continue

            ordered_entities = [entities_by_id[eid] for eid in entity_order]
            locked_ledger = _format_locked_ledger(ordered_entities, aliases_by_entity)
            prompt = self.build_section_prompt(section, locked_ledger, ordered_entities)

            raw_entities, raw_updates, sec_usage, was_cached = self._call_section(section, prompt)
            sections_processed += 1
            run_usage.add(sec_usage)
            if was_cached:
                run_cache_hits += 1
            else:
                run_api_calls += sec_usage.api_calls or 1

            for item in raw_entities:
                canonical_name = str(item.get("canonical_name") or item.get("name") or "").strip()
                if not canonical_name or self._is_ignored_name(canonical_name):
                    continue

                raw_kind = str(item.get("kind") or "character").strip().lower()
                if raw_kind not in allowed_set:
                    warnings.append(
                        f"[{section.section_id}] Skipping '{canonical_name}': kind '{raw_kind}' "
                        f"not in allowed_kinds {self.allowed_kinds}."
                    )
                    continue

                explicit_id = str(item.get("entity_id") or "").strip() or None
                incoming_attrs: dict[str, Any] = {}
                raw_nested_attrs = item.get("attributes") or item.get("base_attributes")
                if isinstance(raw_nested_attrs, dict):
                    for k, v in raw_nested_attrs.items():
                        if v is not None and str(v).strip():
                            incoming_attrs[str(k).strip()] = v

                for std_key in ("target_name", "target_lang", "gender", "declension", "notes"):
                    val = item.get(std_key)
                    if val is not None and str(val).strip():
                        incoming_attrs[std_key] = str(val).strip()

                for k, v in item.items():
                    if k not in _RESERVED_ENTITY_KEYS and v is not None and str(v).strip():
                        incoming_attrs[str(k).strip()] = v

                if self.target_lang and "target_name" in incoming_attrs:
                    incoming_attrs.setdefault("target_lang", self.target_lang)

                raw_aliases_list = item.get("aliases") or []
                incoming_aliases: list[str] = []
                if isinstance(raw_aliases_list, list):
                    for al_val in raw_aliases_list:
                        if isinstance(al_val, str) and al_val.strip():
                            incoming_aliases.append(al_val.strip())
                        elif isinstance(al_val, dict) and str(al_val.get("alias") or "").strip():
                            incoming_aliases.append(str(al_val["alias"]).strip())

                existing = _find_matching_entity(
                    explicit_id=explicit_id,
                    canonical_name=canonical_name,
                    kind=raw_kind,
                )

                if existing is None:
                    eid = explicit_id or _slugify_entity_id(
                        raw_kind, canonical_name, set(entities_by_id.keys())
                    )
                    if "notes" in incoming_attrs and self.max_notes_chars > 0:
                        incoming_attrs["notes"] = _merge_notes(
                            "", str(incoming_attrs["notes"]), max_chars=self.max_notes_chars
                        )
                    new_ent = EntityRecord(
                        entity_id=eid,
                        kind=raw_kind,
                        canonical_name=canonical_name,
                        base_attributes=incoming_attrs,
                    )
                    entities_by_id[eid] = new_ent
                    entity_order.append(eid)
                    aliases_by_entity[eid] = []
                    alias_keys_by_entity[eid] = set()
                    _index_name(canonical_name, eid)
                    if isinstance(incoming_attrs.get("target_name"), str):
                        _index_name(incoming_attrs["target_name"], eid)
                    for al_str in incoming_aliases:
                        _add_alias(
                            eid,
                            AliasRecord(
                                alias=al_str,
                                entity_id=eid,
                                kind="permanent",
                                lang=self.source_lang,
                            ),
                        )
                else:
                    eid = existing.entity_id
                    if existing.kind.lower() != raw_kind:
                        conflicts.append(
                            BibleConflict(
                                section_id=section.section_id,
                                entity_id=eid,
                                canonical_name=existing.canonical_name,
                                field="kind",
                                locked_value=existing.kind,
                                proposed_value=raw_kind,
                                resolution="kept_locked",
                            )
                        )
                    if canonical_name.lower() != existing.canonical_name.strip().lower():
                        _add_alias(
                            eid,
                            AliasRecord(
                                alias=canonical_name,
                                entity_id=eid,
                                kind="permanent",
                                lang=self.source_lang,
                            ),
                        )

                    for attr_k, attr_v in incoming_attrs.items():
                        if attr_k == "notes":
                            existing.base_attributes["notes"] = _merge_notes(
                                str(existing.base_attributes.get("notes") or ""),
                                str(attr_v),
                                max_chars=self.max_notes_chars,
                            )
                            continue
                        cur_val = existing.base_attributes.get(attr_k)
                        if cur_val is None or (isinstance(cur_val, str) and not cur_val.strip()):
                            existing.base_attributes[attr_k] = attr_v
                            if attr_k == "target_name" and isinstance(attr_v, str):
                                _index_name(attr_v, eid)
                        elif str(cur_val).strip().lower() != str(attr_v).strip().lower():
                            conflicts.append(
                                BibleConflict(
                                    section_id=section.section_id,
                                    entity_id=eid,
                                    canonical_name=existing.canonical_name,
                                    field=attr_k,
                                    locked_value=cur_val,
                                    proposed_value=attr_v,
                                    resolution="kept_locked",
                                )
                            )

                    for al_str in incoming_aliases:
                        _add_alias(
                            eid,
                            AliasRecord(
                                alias=al_str,
                                entity_id=eid,
                                kind="permanent",
                                lang=self.source_lang,
                            ),
                        )

            for upd in raw_updates:
                target_eid = str(upd.get("entity_id") or "").strip()
                if not target_eid or target_eid not in entities_by_id:
                    warnings.append(
                        f"[{section.section_id}] alias_updates referenced unknown entity_id '{target_eid}'; skipped."
                    )
                    continue
                add_list = upd.get("add_aliases") or upd.get("aliases") or []
                if isinstance(add_list, list):
                    for raw_al in add_list:
                        if isinstance(raw_al, str) and raw_al.strip():
                            _add_alias(
                                target_eid,
                                AliasRecord(
                                    alias=raw_al.strip(),
                                    entity_id=target_eid,
                                    kind="permanent",
                                    lang=self.source_lang,
                                ),
                            )

        final_entities = [entities_by_id[eid] for eid in entity_order]
        final_aliases = {eid: list(aliases_by_entity.get(eid, ())) for eid in entity_order}
        return BibleDraft(
            entities=final_entities,
            aliases=final_aliases,
            conflicts=conflicts,
            warnings=warnings,
            sections_processed=sections_processed,
            cache_hits=run_cache_hits,
            api_calls=run_api_calls,
            usage=run_usage,
        )


def bootstrap_bible(
    engine: CanonArcEngine,
    sections: Sequence[BibleSection | Mapping[str, Any] | str],
    *,
    client: JsonCompletionClient | None = None,
    builder: BibleBuilder | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    instructions: str = "",
    system_prompt: str | None = None,
    prompt_builder: Callable[[BibleSection, str, Sequence[EntityRecord]], str] | None = None,
    ignore_patterns: Sequence[str | re.Pattern[str]] = (),
    max_notes_chars: int = 300,
    cache_dir: str | Path | None = None,
    bypass_cache: bool = False,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    max_retries: int = 1,
    step: int = 0,
    timeline_id: str | None = None,
    mode: BibleSyncMode = "upsert",
    inherit_existing: bool = True,
    auto_inflect: bool = True,
    batch_size: int = 25,
    commit: bool = True,
    dry_run: bool = False,
    export_path: str | Path | None = None,
    export_format: Literal["auto", "csv", "json"] = "auto",
    check_duplicates: bool = True,
    fuzzy_threshold: float = 88.0,
    same_kind_only: bool = True,
) -> BibleBootstrapReport:
    """Runs BibleBuilder across ordered sections, optionally exports a checkpoint draft,
    commits into the engine at `step` (with batched inflections), and checks for near-duplicates.
    """
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")

    eff_builder = builder
    if eff_builder is None:
        eff_client: Any = client if client is not None else engine.extractor
        if eff_client is None:
            raise ValueError(
                "bootstrap_bible() requires either `builder`, `client`, or an `engine.extractor` "
                "implementing JsonCompletionClient."
            )
        eff_builder = BibleBuilder(
            eff_client,
            source_lang=source_lang,
            target_lang=target_lang,
            allowed_kinds=allowed_kinds,
            instructions=instructions,
            system_prompt=system_prompt,
            prompt_builder=prompt_builder,
            ignore_patterns=ignore_patterns,
            max_notes_chars=max_notes_chars,
            cache_dir=cache_dir,
            bypass_cache=bypass_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )

    initial_entities: list[EntityRecord] = []
    initial_aliases: list[AliasRecord] = []
    if inherit_existing:
        initial_entities = engine.store.get_all_active_entities(tl, step)
        initial_aliases = engine.store.get_all_active_aliases(tl, step)

    draft = eff_builder.build(
        sections,
        initial_entities=initial_entities,
        initial_aliases=initial_aliases,
    )

    exported_str: str | None = None
    if export_path is not None:
        written = draft.export(
            export_path,
            format=export_format,
            timeline_id=tl,
            at_step=step,
        )
        exported_str = str(written)

    sync_report: BibleSyncReport | None = None
    if commit:
        eff_target_lang = target_lang or eff_builder.target_lang
        sync_report = _sync_parsed_bible(
            engine,
            draft.to_parsed_bible(),
            source=exported_str or "bible_builder",
            eff_format="json",
            reason_tag="bible_bootstrap",
            step=step,
            timeline_id=tl,
            mode=mode,
            target_lang=eff_target_lang,
            auto_inflect=auto_inflect,
            batch_size=batch_size,
            dry_run=dry_run,
        )

    duplicate_candidates: list[DuplicateCandidate] = []
    if commit and not dry_run and check_duplicates:
        duplicate_candidates = find_duplicate_candidates(
            engine,
            at_step=step,
            timeline_id=tl,
            fuzzy_threshold=fuzzy_threshold,
            same_kind_only=same_kind_only,
        )

    return BibleBootstrapReport(
        draft=draft,
        sync_report=sync_report,
        duplicate_candidates=duplicate_candidates,
        export_path=exported_str,
    )
