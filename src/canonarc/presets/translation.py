from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from ..models import ENTITY_KIND_HEADINGS, DirectedEdgeRecord, EntityRecord, StepContext
from .formatting import (
    _cap_voice_exemplars_per_speaker,
    _render_budgeted_blocks,
    _truncate_attr_value,
)
from .policy import DEFAULT_SCHEMA_POLICY, SchemaPolicy

logger = logging.getLogger(__name__)

EntityGrouping = Literal["flat", "cast_and_glossary", "by_kind"]

DEFAULT_TRANSLATION_SECTION_HEADERS: dict[str, str] = {
    "state": "## Scene character state & register — binding",
    "glossary": "## Terminology & glossary — binding",
    "background_edges": "## Off-screen character ties — reference only",
    "episodes": "## Relevant narrative episodes — context",
    "voice": "## Character voice bank — stylistic anchor",
}


class TranslationPreset:
    """Preset for literary and game localization (Tier 1 base declension/gender + Tier 2 register)."""

    def __init__(
        self,
        policy: SchemaPolicy | None = None,
        *,
        include_background_edges: bool = False,
        show_entity_ids: bool = False,
        include_kind: bool | Literal["non_character"] = True,
        extra_attributes: Sequence[str] = ("notes",),
        max_attr_chars: int | None = 160,
        empty_text: str = "",
        max_voice_per_speaker: int | None = None,
        max_context_chars: int | None = None,
        entity_grouping: EntityGrouping = "flat",
        section_headers: Mapping[str, str] | None = None,
    ) -> None:
        if include_kind not in (True, False, "non_character"):
            raise ValueError(
                f"Invalid include_kind: {include_kind!r}. Expected True, False, or 'non_character'."
            )
        if max_attr_chars is not None and max_attr_chars <= 0:
            raise ValueError("max_attr_chars must be a positive integer or None.")
        if max_voice_per_speaker is not None and max_voice_per_speaker < 0:
            raise ValueError("max_voice_per_speaker must be non-negative or None.")
        if max_context_chars is not None and max_context_chars < 0:
            raise ValueError("max_context_chars must be non-negative or None.")
        if entity_grouping not in ("flat", "cast_and_glossary", "by_kind"):
            raise ValueError(
                f"Invalid entity_grouping: {entity_grouping!r}. "
                "Expected 'flat', 'cast_and_glossary', or 'by_kind'."
            )
        merged_headers = dict(DEFAULT_TRANSLATION_SECTION_HEADERS)
        if section_headers is not None:
            unknown = sorted(set(section_headers) - set(DEFAULT_TRANSLATION_SECTION_HEADERS))
            if unknown:
                raise ValueError(
                    f"Unknown section_headers key(s): {unknown}. "
                    f"Expected subset of {sorted(DEFAULT_TRANSLATION_SECTION_HEADERS)}."
                )
            for key, val in section_headers.items():
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(
                        f"Invalid section_headers[{key!r}]: expected a non-empty string."
                    )
                merged_headers[key] = val

        self.policy: SchemaPolicy = policy or DEFAULT_SCHEMA_POLICY
        self.include_background_edges: bool = include_background_edges
        self.show_entity_ids: bool = show_entity_ids
        self.include_kind: bool | Literal["non_character"] = include_kind
        self.extra_attributes: tuple[str, ...] = tuple(extra_attributes)
        self.max_attr_chars: int | None = max_attr_chars
        self.empty_text: str = empty_text
        self.max_voice_per_speaker: int | None = max_voice_per_speaker
        self.max_context_chars: int | None = max_context_chars
        self.entity_grouping: EntityGrouping = entity_grouping
        self.section_headers: dict[str, str] = merged_headers

    def _should_show_kind(self, kind: str) -> bool:
        if not kind:
            return False
        if self.include_kind is True:
            return True
        if self.include_kind == "non_character":
            return kind != "character"
        return False

    @staticmethod
    def _build_display_labels(ctx: StepContext) -> dict[str, str]:
        """Builds human-readable labels for entity_ids, auto-disambiguating colliding names."""
        name_by_id: dict[str, str] = dict(ctx.entity_names)
        kind_by_id: dict[str, str] = {}
        for e in (*ctx.entities, *ctx.related_entities):
            name_by_id[e.entity_id] = e.canonical_name
            if e.kind:
                kind_by_id[e.entity_id] = e.kind

        name_counts = Counter(name.casefold() for name in name_by_id.values())
        name_kind_counts = Counter(
            (name.casefold(), kind_by_id.get(eid, ""))
            for eid, name in name_by_id.items()
        )

        labels: dict[str, str] = {}
        for eid, name in name_by_id.items():
            key = name.casefold()
            if name_counts[key] <= 1:
                labels[eid] = name
            else:
                kind = kind_by_id.get(eid, "")
                if kind and name_kind_counts[(key, kind)] == 1:
                    labels[eid] = f"{name} [{kind}]"
                else:
                    labels[eid] = f"{name} ({eid})"
        return labels

    @staticmethod
    def _format_edge_line(
        edge: DirectedEdgeRecord,
        *,
        id_labels: dict[str, str] | None = None,
    ) -> str:
        src = id_labels.get(edge.source_id, edge.source_id) if id_labels is not None else edge.source_id
        dst = id_labels.get(edge.target_id, edge.target_id) if id_labels is not None else edge.target_id
        if edge.relation_type in ("stance", "facade") or ("register" in edge.state or "tone" in edge.state):
            parts: list[str] = []
            if edge.state.get("register"):
                parts.append(f"register={edge.state['register']}")
            if edge.state.get("tone"):
                parts.append(f"tone={edge.state['tone']}")
            for k, v in edge.state.items():
                if k not in ("register", "tone") and v not in (None, ""):
                    parts.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
            tag = " [facade]" if edge.relation_type == "facade" else ""
            detail = ", ".join(parts) if parts else "active"
            return f"- {src} → {dst}{tag}: {detail}"
        pairs = ", ".join(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in edge.state.items())
        return f"- {src} → {dst} [{edge.relation_type}]: {pairs}"

    def _format_entity_line(
        self,
        e: EntityRecord,
        *,
        is_lexicon_only: bool,
        id_labels: dict[str, str] | None,
        suppress_kind_tag: bool = False,
    ) -> str:
        show_kind = (not suppress_kind_tag) and self._should_show_kind(e.kind)
        if self.show_entity_ids:
            kind_tag = f" [{e.kind}]" if show_kind else ""
            subject = f"{e.canonical_name} ({e.entity_id}){kind_tag}"
        else:
            base_label = id_labels.get(e.entity_id, e.canonical_name) if id_labels is not None else e.canonical_name
            if show_kind and f"[{e.kind}]" not in base_label:
                subject = f"{base_label} [{e.kind}]"
            else:
                subject = base_label

        gender = e.base_attributes.get("gender", "")
        decl = e.base_attributes.get("declension", "")
        target_name = e.base_attributes.get("target_name", e.canonical_name)
        attr_str = (
            f"target='{target_name}'"
            + (f", {gender}" if gender else "")
            + (f", {decl}" if decl else "")
        )
        mood = "" if is_lexicon_only else e.state.get("mood", "")
        mood_str = f" [mood: {mood}]" if mood else ""

        extra_parts: list[str] = []
        for attr_key in self.extra_attributes:
            if attr_key in ("target_name", "gender", "declension"):
                continue
            raw_val = e.base_attributes.get(attr_key)
            if raw_val is None or raw_val == "" or raw_val == [] or raw_val == {}:
                continue
            rendered_val = _truncate_attr_value(raw_val, self.max_attr_chars)
            if rendered_val:
                extra_parts.append(f"{attr_key}: {rendered_val}")
        extra_str = f"; {'; '.join(extra_parts)}" if extra_parts else ""
        return f"- {subject}: {attr_str}{mood_str}{extra_str}"

    def _group_entity_lines(
        self,
        entities: Sequence[EntityRecord],
        *,
        is_lexicon_only: bool,
        id_labels: dict[str, str] | None,
    ) -> list[tuple[str | None, list[str]]]:
        """Groups entities into deterministic ``### <Kind>`` subsections."""
        grouped: dict[str, list[EntityRecord]] = {}
        for e in entities:
            kind_key = (e.kind or "").strip().lower() or "other"
            grouped.setdefault(kind_key, []).append(e)

        ordered_kinds = [
            k for k in ENTITY_KIND_HEADINGS if k in grouped
        ] + sorted(k for k in grouped if k not in ENTITY_KIND_HEADINGS)

        sections: list[tuple[str | None, list[str]]] = []
        for kind_key in ordered_kinds:
            heading = ENTITY_KIND_HEADINGS.get(
                kind_key, kind_key.replace("_", " ").title()
            )
            lines = [
                self._format_entity_line(
                    e,
                    is_lexicon_only=is_lexicon_only,
                    id_labels=id_labels,
                    suppress_kind_tag=True,
                )
                for e in grouped[kind_key]
            ]
            sections.append((f"### {heading}", lines))
        return sections

    def format_prompt_context(self, ctx: StepContext) -> str:
        id_labels: dict[str, str] | None = (
            None if self.show_entity_ids else self._build_display_labels(ctx)
        )

        # Tier 1 & 2 binding state block (or Tier 1 lexicon block in LEXICON_ONLY)
        is_lexicon_only = not self.policy.has_tier(2) and not self.policy.has_tier(3)
        state_header = (
            self.section_headers["glossary"]
            if is_lexicon_only
            else self.section_headers["state"]
        )
        glossary_header = self.section_headers["glossary"]
        glossary_sections: list[tuple[str | None, list[str]]] | None = None

        if self.entity_grouping == "flat":
            entity_lines = [
                self._format_entity_line(e, is_lexicon_only=is_lexicon_only, id_labels=id_labels)
                for e in ctx.entities
            ]
        elif self.entity_grouping == "cast_and_glossary":
            if is_lexicon_only:
                entity_lines = [
                    self._format_entity_line(e, is_lexicon_only=True, id_labels=id_labels)
                    for e in ctx.entities
                ]
            else:
                char_entities = [
                    e for e in ctx.entities if (e.kind or "").strip().lower() == "character"
                ]
                non_char_entities = [
                    e for e in ctx.entities if (e.kind or "").strip().lower() != "character"
                ]
                entity_lines = [
                    self._format_entity_line(e, is_lexicon_only=False, id_labels=id_labels)
                    for e in char_entities
                ]
                if non_char_entities:
                    glossary_sections = [
                        (
                            None,
                            [
                                self._format_entity_line(
                                    e, is_lexicon_only=False, id_labels=id_labels
                                )
                                for e in non_char_entities
                            ],
                        )
                    ]
        else:  # "by_kind"
            if is_lexicon_only:
                entity_lines = []
                glossary_sections = self._group_entity_lines(
                    ctx.entities, is_lexicon_only=True, id_labels=id_labels
                )
            else:
                char_entities = [
                    e for e in ctx.entities if (e.kind or "").strip().lower() == "character"
                ]
                non_char_entities = [
                    e for e in ctx.entities if (e.kind or "").strip().lower() != "character"
                ]
                entity_lines = [
                    self._format_entity_line(
                        e,
                        is_lexicon_only=False,
                        id_labels=id_labels,
                        suppress_kind_tag=True,
                    )
                    for e in char_entities
                ]
                if non_char_entities:
                    glossary_sections = self._group_entity_lines(
                        non_char_entities, is_lexicon_only=False, id_labels=id_labels
                    )

        active_edge_lines = (
            []
            if is_lexicon_only
            else [self._format_edge_line(edge, id_labels=id_labels) for edge in ctx.active_edges]
        )

        bg_header = self.section_headers["background_edges"]
        bg_lines = (
            [self._format_edge_line(edge, id_labels=id_labels) for edge in ctx.background_edges]
            if self.include_background_edges and ctx.background_edges
            else []
        )

        # Tier 6 Episodic context block (when present)
        ep_header = self.section_headers["episodes"]
        ep_lines: list[str] = []
        if ctx.episodes:
            scene_char_ids = {
                e.entity_id
                for e in ctx.entities
                if e.kind == "character" and not e.entity_id.startswith("masked:")
            }
            for ep in ctx.episodes:
                if ep.known_by:
                    known_ids = sorted(ep.known_by.keys())
                    known_labels = (
                        known_ids
                        if id_labels is None
                        else [id_labels.get(eid, eid) for eid in known_ids]
                    )
                    known_str = f" | known to: {', '.join(known_labels)}"
                    ignorant_ids = sorted(scene_char_ids - set(ep.known_by.keys()))
                    ignorant_labels = (
                        ignorant_ids
                        if id_labels is None
                        else [id_labels.get(eid, eid) for eid in ignorant_ids]
                    )
                    ignorant_str = f" | ignorant: {', '.join(ignorant_labels)}" if ignorant_labels else ""
                else:
                    known_str = ""
                    ignorant_str = ""
                ep_lines.append(f"- [Step {ep.step}{known_str}{ignorant_str}] {ep.summary}")

        # Tier 5 Voice Bank anchor block
        voice_header = self.section_headers["voice"]
        voice_lines: list[str] = []
        capped_exemplars = _cap_voice_exemplars_per_speaker(
            ctx.voice_exemplars, self.max_voice_per_speaker
        )
        for ex in capped_exemplars:
            speaker = ex.speaker_id if id_labels is None else id_labels.get(ex.speaker_id, ex.speaker_id)
            reg = ex.metadata.get("register", "")
            reg_str = f" ({reg})" if reg else ""
            fallback_note = ex.metadata.get("fallback_note", "")
            note_str = f" {fallback_note}" if fallback_note else ""
            voice_lines.append(f'- {speaker}{reg_str}{note_str}: "{ex.text}"')

        rendered, dropped_binding = _render_budgeted_blocks(
            state_header=state_header,
            entity_lines=entity_lines,
            active_edge_lines=active_edge_lines,
            bg_header=bg_header,
            bg_lines=bg_lines,
            ep_header=ep_header,
            ep_lines=ep_lines,
            voice_header=voice_header,
            voice_lines=voice_lines,
            empty_text=self.empty_text,
            max_context_chars=self.max_context_chars,
            glossary_header=glossary_header,
            glossary_sections=glossary_sections,
        )
        if dropped_binding > 0:
            logger.warning(
                "TranslationPreset truncated %d binding entity/edge line(s) to fit max_context_chars=%d",
                dropped_binding,
                self.max_context_chars,
            )
        return rendered

    def extraction_instructions(self) -> str:
        return (
            "Analyze the literary prose chunk and emit a strict StepDelta JSON diff for localization consistency:\n"
            "1. **Entities (`entities`) — Reuse Existing IDs & Delta Discipline**:\n"
            "   - ALWAYS reuse existing canonical `entity_id`s from the known entities/aliases list. "
            "NEVER invent a new entity for an epithet, title, or disguise that already aliases an existing character "
            "(for example, if 'hooded courier' aliases `char_lyra`, emit updates on `char_lyra`, NOT a new `char_courier`).\n"
            "   - Emit an `EntityDelta` ONLY when `mood`, `location_id`, or `add_aliases` changed in this chunk.\n"
            "   - Set `state['mood']` to a concise lowercase word or short phrase (e.g., `'calm'`, `'furious'`, `'panicked'`, `'suspicious'`, `'grief'`).\n"
            "   - When a disguised character unmasks to someone, emit `add_aliases` with `kind='disguise'` and updated `known_to`.\n"
            "2. **Edges (`edges`) — Binding T–V Register & Tone on `relation_type='stance'`**:\n"
            "   - Whenever a speaker's grammatical address register (e.g., formal `'Vi'` vs. informal `'ti'`) or interpersonal tone shifts "
            "toward an addressee (such as an angry accusation, unmasking, or furious reprimand), emit "
            "`EdgeDelta(source_id='<speaker_id>', target_id='<addressee_id>', relation_type='stance', "
            "state={'register': '<Vi|ti|formal|informal>', 'tone': '<concise_tone>'}, caused_by_episode_id='<ep_id>')`.\n"
            "   - ALWAYS use `relation_type='stance'` and ALWAYS include `'register'` and `'tone'` keys for interpersonal register shifts "
            "(do not hide register shifts under ad-hoc relation types like `'accusation'` or omit `'register'`).\n"
            "3. **Episodes (`episodes`) — Concise Plot Beats & Strict Witness Scoping**:\n"
            "   - Emit 1–2 primary `EpisodeRecord`s for the major plot beats in the chunk.\n"
            "   - Only include characters physically present in the room during the beat in `participants`/`witnesses` "
            "(exclude anyone who already stepped outside, e.g., onto a balcony; include silent/disguised observers inside the room using their canonical `entity_id`).\n"
            "4. **Flashbacks (`is_flashback=True`)**:\n"
            "   - If the chunk depicts a memory or flashback to an earlier time (e.g., ten years ago), set `is_flashback=True` "
            "and `story_time` on the `StepDelta` so present-day character moods and T–V registers are preserved.\n"
            "5. **Epistemic Transitions (`inform` / `InformDelta`)**:\n"
            "   - Do NOT emit `InformDelta` with `epistemic='direct'` (direct witnesses belong in `episode.witnesses`).\n"
            "   - Use `epistemic='deceived'` (with `believed_summary`), `'told'`, or `'forgotten'` when characters lie about, reveal, or lose memory of an earlier episode."
        )
