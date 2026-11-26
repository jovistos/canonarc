from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .embedders import _CJK_CHAR_RE
from .models import (
    ENTITY_KIND_PREFIXES,
    AliasRecord,
    DirectedEdgeRecord,
    EntityRecord,
    EpisodeRecord,
    ResolvedMention,
    StepContext,
)
from .presets import DEFAULT_SCHEMA_POLICY, DomainPreset, SchemaPolicy
from .resolver import DefaultLexicalResolver, _VOWELS

if TYPE_CHECKING:
    from .engine import CanonArcEngine

AuditSeverity = Literal["info", "warning", "error"]

_SEVERITY_RANK: dict[AuditSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
}

_ENTITY_ID_PREFIXES: tuple[str, ...] = (
    *dict.fromkeys(f"{prefix}_" for prefix in ENTITY_KIND_PREFIXES.values()),
    "ent_",
)

_LEADING_ARTICLES_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)

_TV_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"vi", "ti", "vous", "tu", "sie", "du", "usted", "lei", "vy", "ty"}
)
_GENERIC_REGISTER_TOKENS: frozenset[str] = frozenset(
    {"formal", "informal", "casual", "polite", "intimate", "familiar"}
)

_SCRIPT_ALIASES: dict[str, str] = {
    "latin": "latin",
    "latn": "latin",
    "cyrillic": "cyrillic",
    "cyrl": "cyrillic",
    "greek": "greek",
    "grek": "greek",
    "arabic": "arabic",
    "arab": "arabic",
    "hebrew": "hebrew",
    "hebr": "hebrew",
    "devanagari": "devanagari",
    "deva": "devanagari",
    "han": "han",
    "cjk": "han",
    "hani": "han",
    "hans": "han",
    "hant": "han",
    "hiragana": "hiragana",
    "hira": "hiragana",
    "katakana": "katakana",
    "kana": "katakana",
    "hangul": "hangul",
    "hang": "hangul",
    "thai": "thai",
    "armenian": "armenian",
    "armn": "armenian",
    "georgian": "georgian",
    "geor": "georgian",
    "bengali": "bengali",
    "beng": "bengali",
    "tamil": "tamil",
    "taml": "tamil",
}

_SUPPORTED_SCRIPTS: frozenset[str] = frozenset(_SCRIPT_ALIASES.values())

_UNAMBIGUOUS_LANG_SCRIPTS: dict[str, tuple[str, ...]] = {
    "ru": ("cyrillic",),
    "uk": ("cyrillic",),
    "bg": ("cyrillic",),
    "be": ("cyrillic",),
    "mk": ("cyrillic",),
    "el": ("greek",),
    "ar": ("arabic",),
    "fa": ("arabic",),
    "ur": ("arabic",),
    "he": ("hebrew",),
    "iw": ("hebrew",),
    "hi": ("devanagari",),
    "th": ("thai",),
    "ko": ("hangul",),
    "ja": ("han", "hiragana", "katakana"),
    "hy": ("armenian",),
    "ka": ("georgian",),
    "bn": ("bengali",),
    "ta": ("tamil",),
}

_URL_OR_EMAIL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b)",
    re.IGNORECASE,
)
_MARKUP_OR_PLACEHOLDER_RE = re.compile(
    r"</?[A-Za-z][^>\n]*>|\{[^{}\s]+\}|%(?:\([^)\s]+\))?[#0\- +]?(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?[diouxXeEfFgGcrsa%]"
)
_ROMAN_NUMERAL_RE = re.compile(
    r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
_LETTER_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class AuditIssue(BaseModel):
    """Single continuity or localization QA finding emitted by an AuditRule."""

    code: str = Field(description="Machine-readable issue code, e.g. 'exonym_leak'")
    rule_name: str = Field(description="Name of the AuditRule that emitted this issue")
    severity: AuditSeverity = "warning"
    step: int
    timeline_id: str = "main"
    message: str
    subject_key: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggestion: str | None = None


class AuditReport(BaseModel):
    """Structured result returned by engine.audit_step() and engine.audit_timeline()."""

    timeline_id: str
    steps_audited: list[int] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)
    rules_executed: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no error-severity issues were detected."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

    def has_issues_at_or_above(self, min_severity: AuditSeverity) -> bool:
        threshold = _SEVERITY_RANK[min_severity]
        return any(_SEVERITY_RANK[i.severity] >= threshold for i in self.issues)

    def by_severity(self, severity: AuditSeverity) -> list[AuditIssue]:
        return [i for i in self.issues if i.severity == severity]

    def by_code(self, code: str) -> list[AuditIssue]:
        return [i for i in self.issues if i.code == code]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        steps_desc = (
            f"step {self.steps_audited[0]}"
            if len(self.steps_audited) == 1
            else f"{len(self.steps_audited)} step(s)"
        )
        header = (
            f"Audit [{status}] timeline={self.timeline_id} ({steps_desc}): "
            f"{self.error_count} error(s), {self.warning_count} warning(s), {self.info_count} info"
        )
        if not self.issues:
            return header
        lines = [header]
        for issue in self.issues:
            subj = f" [{issue.subject_key}]" if issue.subject_key else ""
            lines.append(
                f"  - [{issue.severity.upper()}] step {issue.step} ({issue.code}){subj}: {issue.message}"
            )
            if issue.suggestion:
                lines.append(f"      suggestion: {issue.suggestion}")
        return "\n".join(lines)

    def format_repair_instructions(
        self,
        *,
        min_severity: AuditSeverity = "error",
        max_issues: int | None = None,
        header: str | None = (
            "Fix the following terminology/continuity errors in your translation "
            "while preserving everything else:"
        ),
        include_snippets: bool = True,
    ) -> str:
        """Formats failed issues at or above min_severity as a compact LLM-ready repair bullet list."""
        threshold = _SEVERITY_RANK[min_severity]
        filtered = [i for i in self.issues if _SEVERITY_RANK[i.severity] >= threshold]
        if not filtered:
            return ""

        filtered.sort(key=lambda i: _SEVERITY_RANK[i.severity], reverse=True)

        deduped: list[AuditIssue] = []
        seen_term_subjects: set[str] = set()
        term_codes = {"exonym_leak", "target_term_mismatch", "target_term_missing"}
        for issue in filtered:
            if issue.subject_key and issue.code in term_codes:
                if issue.subject_key in seen_term_subjects:
                    continue
                seen_term_subjects.add(issue.subject_key)
            deduped.append(issue)

        if max_issues is not None and max_issues >= 0:
            deduped = deduped[:max_issues]
        if not deduped:
            return ""

        lines: list[str] = []
        if header:
            lines.append(header)
        for issue in deduped:
            subj = f"[{issue.subject_key}] " if issue.subject_key else ""
            body = (
                issue.suggestion.strip()
                if issue.suggestion and issue.suggestion.strip()
                else issue.message.strip()
            )
            snippet = issue.evidence.get("snippet") if include_snippets else None
            if isinstance(snippet, str) and snippet.strip():
                lines.append(f"- {subj}{body} (at: {snippet!r})")
            else:
                lines.append(f"- {subj}{body}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AuditContext:
    """Read-only point-in-time snapshot passed to every AuditRule."""

    timeline_id: str
    step: int
    source_text: str = ""
    target_text: str = ""
    entities: tuple[EntityRecord, ...] = ()
    new_entities: tuple[EntityRecord, ...] = ()
    aliases: tuple[AliasRecord, ...] = ()
    edges: tuple[DirectedEdgeRecord, ...] = ()
    mutated_edges: tuple[DirectedEdgeRecord, ...] = ()
    episodes: tuple[EpisodeRecord, ...] = ()
    all_episode_ids: frozenset[str] = frozenset()
    resolved_mentions: tuple[ResolvedMention, ...] = ()
    policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY
    lang: str | None = None


class AuditRule(Protocol):
    """Protocol for pluggable QA and continuity audit rules."""

    name: str
    required_tiers: frozenset[int]

    def check(self, ctx: AuditContext) -> Sequence[AuditIssue]: ...


class CallableAuditRule:
    """Adapter turning a plain Python function into an AuditRule."""

    def __init__(
        self,
        name: str,
        fn: Callable[[AuditContext], Sequence[AuditIssue]],
        *,
        required_tiers: frozenset[int] = frozenset({1}),
    ) -> None:
        self.name = name
        self._fn = fn
        self.required_tiers = frozenset(required_tiers)

    def check(self, ctx: AuditContext) -> Sequence[AuditIssue]:
        return list(self._fn(ctx))


def _phrase_pattern(phrase: str) -> str:
    prefix = "" if _CJK_CHAR_RE.match(phrase[0]) else r"(?<!\w)"
    suffix = "" if _CJK_CHAR_RE.match(phrase[-1]) else r"(?!\w)"
    return rf"{prefix}{re.escape(phrase)}{suffix}"


def _extract_snippet(text: str, start: int, end: int, radius: int = 30) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", " ", text.strip().lower()).strip()
    return _LEADING_ARTICLES_RE.sub("", cleaned).strip()


def _entity_id_slug(entity_id: str) -> str:
    low = entity_id.strip().lower()
    for prefix in _ENTITY_ID_PREFIXES:
        if low.startswith(prefix):
            low = low[len(prefix) :]
            break
    return _normalize_label(low)


class ExonymLeakRule:
    """Detects untranslated canonical_name leakage in target_text when target_name != canonical_name."""

    name: str = "exonym_leak"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "error",
        case_sensitive: bool = True,
        only_source_mentioned: bool = True,
        snippet_radius: int = 30,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.case_sensitive = case_sensitive
        self.only_source_mentioned = only_source_mentioned
        self.snippet_radius = snippet_radius

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if not ctx.target_text.strip():
            return []

        issues: list[AuditIssue] = []
        aliases_by_entity: dict[str, list[str]] = {}
        for a in ctx.aliases:
            if a.alias.strip():
                aliases_by_entity.setdefault(a.entity_id, []).append(a.alias.strip())

        flags = 0 if self.case_sensitive else re.IGNORECASE
        source_text = ctx.source_text
        has_source = bool(source_text.strip())

        for entity in ctx.entities:
            if entity.base_attributes.get("allow_source_name"):
                continue
            canonical = entity.canonical_name.strip()
            raw_target = entity.base_attributes.get("target_name")
            if not canonical or not isinstance(raw_target, str) or not raw_target.strip():
                continue
            target_name = raw_target.strip()
            if canonical.lower() == target_name.lower():
                continue

            # Guard against canonical_name appearing as a valid word inside target_name
            if re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE):
                continue

            if self.only_source_mentioned and has_source:
                cues = [canonical, *aliases_by_entity.get(entity.entity_id, ())]
                if not any(
                    re.search(_phrase_pattern(cue), source_text, re.IGNORECASE)
                    for cue in cues
                    if cue
                ):
                    continue

            matches = list(re.finditer(_phrase_pattern(canonical), ctx.target_text, flags))
            if not matches:
                continue

            spans = [(m.start(), m.end()) for m in matches]
            first = matches[0]
            issues.append(
                AuditIssue(
                    code="exonym_leak",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=entity.entity_id,
                    entity_ids=[entity.entity_id],
                    message=(
                        f"Untranslated canonical_name '{canonical}' leaked into target_text "
                        f"({len(matches)} occurrence(s)); expected target_name '{target_name}'."
                    ),
                    evidence={
                        "canonical_name": canonical,
                        "expected_target_name": target_name,
                        "occurrence_count": len(matches),
                        "matched_spans": spans,
                        "snippet": _extract_snippet(
                            ctx.target_text, first.start(), first.end(), self.snippet_radius
                        ),
                    },
                    suggestion=f"Replace '{canonical}' with '{target_name}' (or its appropriate inflected form).",
                )
            )

        return issues


class TargetTermComplianceRule:
    """Verifies that entities mentioned in source_text appear in target_text using their locked target_name (or valid inflections)."""

    name: str = "target_term_compliance"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        mismatch_severity: AuditSeverity = "error",
        missing_severity: AuditSeverity = "warning",
        case_sensitive: bool = False,
        fuzzy_stem_threshold: float | None = 85.0,
        near_miss_threshold: float = 62.0,
        allow_partial_target_match: bool = True,
        kinds: Sequence[str] | None = None,
        snippet_radius: int = 30,
    ) -> None:
        self.mismatch_severity: AuditSeverity = mismatch_severity
        self.missing_severity: AuditSeverity = missing_severity
        self.case_sensitive = case_sensitive
        self.fuzzy_stem_threshold = fuzzy_stem_threshold
        self.near_miss_threshold = near_miss_threshold
        self.allow_partial_target_match = allow_partial_target_match
        self.kinds: frozenset[str] | None = (
            frozenset(k.strip().lower() for k in kinds if k.strip())
            if kinds is not None
            else None
        )
        self.snippet_radius = snippet_radius

    @staticmethod
    def _effective_target_lang(entity: EntityRecord, ctx_lang: str | None) -> str | None:
        if ctx_lang and ctx_lang.strip():
            return ctx_lang.strip().lower()
        raw = entity.base_attributes.get("target_lang") or entity.base_attributes.get("lang")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        return None

    @staticmethod
    def _word_matches_token(
        ref_word_lower: str,
        raw_token_spans: Sequence[tuple[str, str, int, int]],
        *,
        fuzzy_stem_threshold: float | None,
    ) -> bool:
        if not ref_word_lower:
            return False
        # 1. Exact token match
        for _, lower_tok, _, _ in raw_token_spans:
            if lower_tok == ref_word_lower:
                return True

        if _CJK_CHAR_RE.match(ref_word_lower[0]):
            return False

        # 2. Short-name inflection stems (3-6 chars) via DefaultLexicalResolver
        if 3 <= len(ref_word_lower) <= 6:
            stems = DefaultLexicalResolver._short_name_stems(ref_word_lower)
            for raw_tok, lower_tok, _, _ in raw_token_spans:
                if not raw_tok or not raw_tok[0].isupper():
                    continue
                for stem, allowed_suffixes in stems:
                    if lower_tok.startswith(stem) and len(lower_tok) > len(stem):
                        if lower_tok[len(stem) :] in allowed_suffixes:
                            return True

        # 3. Conservative leading-stem + rapidfuzz inflection match (for >= 5 char names, e.g. Julijan -> Julijanu, Poaro -> Poaroa)
        if fuzzy_stem_threshold is not None and len(ref_word_lower) >= 5:
            stem = (
                ref_word_lower[:-1]
                if ref_word_lower[-1] in _VOWELS
                else ref_word_lower
            )
            if len(stem) >= 4:
                for raw_tok, lower_tok, _, _ in raw_token_spans:
                    if not raw_tok or not raw_tok[0].isupper():
                        continue
                    if lower_tok.startswith(stem) and len(lower_tok) <= len(ref_word_lower) + 4:
                        if float(fuzz.ratio(ref_word_lower, lower_tok)) >= fuzzy_stem_threshold:
                            return True
        return False

    def _matches_target(
        self,
        target_name: str,
        accepted_forms: Sequence[str],
        source_cue: str,
        target_text: str,
        raw_token_spans: Sequence[tuple[str, str, int, int]],
    ) -> bool:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        target_words = [w for w in re.findall(r"\w+", target_name) if len(w) >= 3]
        source_words = re.findall(r"\w+", source_cue)
        require_full_multiword = len(target_words) >= 2 and len(source_words) >= 2

        # 1. Exact word-boundary match against target_name or accepted target aliases/inflections.
        # When both source_cue and target_name are multi-word (e.g. "Hercule Poirot" -> "Erkil Poaro"),
        # ignore single-word aliases (e.g. "Poaroa") so "Herkul Poaro" doesn't falsely pass via surname alone.
        for form in accepted_forms:
            if not form:
                continue
            if require_full_multiword and len(re.findall(r"\w+", form)) < 2:
                continue
            if re.search(_phrase_pattern(form), target_text, flags):
                return True

        # 2. Single-word stem/inflection match across accepted forms (only when not requiring full multi-word)
        if not require_full_multiword:
            for form in accepted_forms:
                if " " not in form and self._word_matches_token(
                    form.lower(),
                    raw_token_spans,
                    fuzzy_stem_threshold=self.fuzzy_stem_threshold,
                ):
                    return True

        # 3. Multi-word target_name token-by-token stem/inflection match
        if len(target_words) >= 2:
            if self.allow_partial_target_match and not require_full_multiword:
                # Source only used a single-word alias (e.g. 'Poirot'); matching any significant token ('Poaro'/'Poaroa') suffices
                if any(
                    self._word_matches_token(
                        w.lower(),
                        raw_token_spans,
                        fuzzy_stem_threshold=self.fuzzy_stem_threshold,
                    )
                    for w in target_words
                ):
                    return True
            else:
                # Multi-word source cue (e.g. 'Hercule Poirot'): every significant token must match
                # so 'Herkul Poaro' does NOT falsely pass for 'Erkil Poaro'
                if all(
                    self._word_matches_token(
                        w.lower(),
                        raw_token_spans,
                        fuzzy_stem_threshold=self.fuzzy_stem_threshold,
                    )
                    for w in target_words
                ):
                    return True

        return False

    def _find_suspected_variant(
        self,
        canonical: str,
        target_name: str,
        target_text: str,
        raw_token_spans: Sequence[tuple[str, str, int, int]],
    ) -> tuple[str, int, int, float] | None:
        """Scans capitalized token windows in target_text for a likely wrong transliteration/spelling."""
        if not raw_token_spans:
            return None
        refs = [r.strip().lower() for r in (target_name, canonical) if r and r.strip()]
        if not refs:
            return None

        target_word_count = max(1, len(re.findall(r"\w+", target_name)))
        window_sizes = sorted({1, target_word_count})

        best: tuple[str, int, int, float] | None = None
        n_tokens = len(raw_token_spans)
        for win in window_sizes:
            if win > n_tokens:
                continue
            for idx in range(n_tokens - win + 1):
                slice_spans = raw_token_spans[idx : idx + win]
                # Require capitalized proper-noun signal
                if not any(tok and tok[0].isupper() for tok, _, _, _ in slice_spans):
                    continue
                if win > 1:
                    # Only join adjacent tokens separated by whitespace/punctuation (not across sentences)
                    start = slice_spans[0][2]
                    end = slice_spans[-1][3]
                    cand_raw = target_text[start:end]
                    if re.search(r"[.!?\n;:]", cand_raw):
                        continue
                    # For multi-word window, require first token capitalized
                    if not slice_spans[0][0][0].isupper():
                        continue
                else:
                    cand_raw, _, start, end = slice_spans[0]

                cand_lower = cand_raw.strip().lower()
                if len(cand_lower) < 3:
                    continue

                for ref in refs:
                    ref_wc = max(1, len(re.findall(r"\w+", ref)))
                    if win != ref_wc:
                        continue
                    max_len_diff = max(3, len(ref) // 2)
                    if abs(len(cand_lower) - len(ref)) > max_len_diff:
                        continue
                    score = float(fuzz.ratio(cand_lower, ref))
                    if score >= self.near_miss_threshold:
                        if best is None or score > best[3] or (score == best[3] and len(cand_raw) > len(best[0])):
                            best = (cand_raw, start, end, score)
        return best

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if not ctx.source_text.strip() or not ctx.target_text.strip():
            return []

        aliases_by_entity: dict[str, list[AliasRecord]] = {}
        for a in ctx.aliases:
            if a.alias.strip():
                aliases_by_entity.setdefault(a.entity_id, []).append(a)

        mentions_by_entity: dict[str, list[ResolvedMention]] = {}
        for rm in ctx.resolved_mentions:
            mentions_by_entity.setdefault(rm.entity_id, []).append(rm)

        raw_token_spans = [
            (m.group(0), m.group(0).lower(), m.start(), m.end())
            for m in re.finditer(r"\w+", ctx.target_text)
        ]

        issues: list[AuditIssue] = []
        for entity in ctx.entities:
            if entity.base_attributes.get("allow_omission"):
                continue
            if self.kinds is not None and entity.kind.strip().lower() not in self.kinds:
                continue

            canonical = entity.canonical_name.strip()
            raw_target = entity.base_attributes.get("target_name")
            if not canonical or not isinstance(raw_target, str) or not raw_target.strip():
                continue
            target_name = raw_target.strip()
            target_lang = self._effective_target_lang(entity, ctx.lang)
            ent_aliases = aliases_by_entity.get(entity.entity_id, [])

            # 1. Determine if entity was mentioned in source_text
            source_cue: str | None = None
            ent_mentions = mentions_by_entity.get(entity.entity_id)
            if ent_mentions:
                source_cue = ent_mentions[0].surface_text.strip() or ent_mentions[0].matched_phrase
            else:
                source_cues = [canonical]
                for a in ent_aliases:
                    if a.kind == "inflection":
                        continue
                    if target_lang is not None and a.lang and a.lang.strip().lower() == target_lang:
                        continue
                    source_cues.append(a.alias.strip())
                for cue in source_cues:
                    if cue and re.search(_phrase_pattern(cue), ctx.source_text, re.IGNORECASE):
                        source_cue = cue
                        break

            if not source_cue:
                continue

            # 2. Respect allow_source_name and avoid double-reporting verbatim exonym leaks
            if entity.base_attributes.get("allow_source_name"):
                if re.search(_phrase_pattern(canonical), ctx.target_text, re.IGNORECASE):
                    continue
            elif canonical.lower() != target_name.lower():
                if not re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE):
                    if re.search(_phrase_pattern(canonical), ctx.target_text, 0):
                        continue

            # 3. Build accepted target-side surface forms
            accepted_forms: list[str] = [target_name]
            seen_forms = {target_name.lower()}
            for a in ent_aliases:
                al_str = a.alias.strip()
                if not al_str or al_str.lower() in seen_forms:
                    continue
                al_lang = a.lang.strip().lower() if a.lang and a.lang.strip() else None
                is_target_inflection = a.kind == "inflection" and (
                    target_lang is None or al_lang is None or al_lang == target_lang
                )
                is_target_lang_alias = target_lang is not None and al_lang == target_lang
                if is_target_inflection or is_target_lang_alias:
                    seen_forms.add(al_str.lower())
                    accepted_forms.append(al_str)

            if self._matches_target(
                target_name, accepted_forms, source_cue, ctx.target_text, raw_token_spans
            ):
                continue

            # 4. Classify failure: suspected wrong spelling/transliteration (error) vs pure omission (warning)
            near_miss = self._find_suspected_variant(
                canonical, target_name, ctx.target_text, raw_token_spans
            )
            inflection_examples = [f for f in accepted_forms[1:5]]
            inflection_hint = (
                f" (or valid inflections such as {', '.join(inflection_examples)})"
                if inflection_examples
                else " (or its appropriate inflected form)"
            )

            if near_miss is not None:
                variant_str, start, end, score = near_miss
                snippet = _extract_snippet(ctx.target_text, start, end, self.snippet_radius)
                issues.append(
                    AuditIssue(
                        code="target_term_mismatch",
                        rule_name=self.name,
                        severity=self.mismatch_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key=entity.entity_id,
                        entity_ids=[entity.entity_id],
                        message=(
                            f"Source mentioned '{source_cue}' ({entity.entity_id}), but locked "
                            f"target_name '{target_name}' was not found in target_text "
                            f"(suspected wrong spelling/transliteration: '{variant_str}')."
                        ),
                        evidence={
                            "canonical_name": canonical,
                            "matched_source_cue": source_cue,
                            "expected_target_name": target_name,
                            "accepted_target_forms": accepted_forms[:8],
                            "suspected_variant": variant_str,
                            "variant_similarity": round(score, 2),
                            "snippet": snippet,
                        },
                        suggestion=(
                            f"Replace '{variant_str}' with locked target form '{target_name}'"
                            f"{inflection_hint}."
                        ),
                    )
                )
            else:
                issues.append(
                    AuditIssue(
                        code="target_term_missing",
                        rule_name=self.name,
                        severity=self.missing_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key=entity.entity_id,
                        entity_ids=[entity.entity_id],
                        message=(
                            f"Source mentioned '{source_cue}' ({entity.entity_id}), but locked "
                            f"target_name '{target_name}' was not found in target_text."
                        ),
                        evidence={
                            "canonical_name": canonical,
                            "matched_source_cue": source_cue,
                            "expected_target_name": target_name,
                            "accepted_target_forms": accepted_forms[:8],
                            "suspected_variant": None,
                        },
                        suggestion=(
                            f"Ensure '{source_cue}' is rendered as '{target_name}'"
                            f"{inflection_hint} unless intentionally pronominalized."
                        ),
                    )
                )

        return issues


def _normalize_script_name(script: str) -> str:
    key = script.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _SCRIPT_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported script '{script}'. Supported scripts: {sorted(_SUPPORTED_SCRIPTS)}."
        )
    return normalized


def _infer_scripts_from_lang(lang: str | None) -> frozenset[str] | None:
    """Infers expected script(s) only from unambiguous BCP-47 tags (e.g. 'sr-Latn', 'ru', 'el')."""
    if not lang or not lang.strip():
        return None
    parts = [p.strip().lower() for p in re.split(r"[-_]", lang.strip()) if p.strip()]
    if not parts:
        return None
    # Explicit BCP-47 script subtag (e.g. sr-Latn, sr-Cyrl, uz-Latn, zh-Hans)
    for subtag in parts[1:]:
        if subtag in _SCRIPT_ALIASES:
            return frozenset({_SCRIPT_ALIASES[subtag]})
    # Unambiguous primary language subtag (e.g. ru, uk, bg, el, ja)
    primary = parts[0]
    inferred = _UNAMBIGUOUS_LANG_SCRIPTS.get(primary)
    if inferred is not None:
        return frozenset(inferred)
    return None


def _char_script(ch: str) -> str | None:
    """Classifies a Unicode letter into a canonical script name, or None for non-letters/common chars."""
    if not unicodedata.category(ch).startswith("L"):
        return None
    cp = ord(ch)
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "latin"
    if (0x0400 <= cp <= 0x052F) or (0x2DE0 <= cp <= 0x2DFF) or (0xA640 <= cp <= 0xA69F) or (0x1C80 <= cp <= 0x1C8F):
        return "cyrillic"
    if (0x0370 <= cp <= 0x03FF) or (0x1F00 <= cp <= 0x1FFF):
        return "greek"
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
        return "arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if (0x30A0 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF) or (0xFF65 <= cp <= 0xFF9F):
        return "katakana"
    if (0xAC00 <= cp <= 0xD7AF) or (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F):
        return "hangul"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if _CJK_CHAR_RE.match(ch):
        return "han"

    name = unicodedata.name(ch, "")
    if not name:
        return None
    first_word = name.split(" ", 1)[0].lower()
    if first_word in _SCRIPT_ALIASES:
        return _SCRIPT_ALIASES[first_word]
    if "CJK" in name:
        return "han"
    return None


def _mask_spans(text: str, pattern: re.Pattern[str]) -> str:
    """Replaces matched spans with spaces of identical length so character offsets stay aligned."""
    return pattern.sub(lambda m: " " * (m.end() - m.start()), text)


class ScriptConsistencyRule:
    """Detects unexpected script drift and mixed-script homoglyphs in target_text."""

    name: str = "script_consistency"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        expected_script: str | Sequence[str] | None = None,
        *,
        severity: AuditSeverity = "error",
        mixed_script_severity: AuditSeverity = "error",
        allowed_tokens: Sequence[str] | None = None,
        allowed_patterns: Sequence[str | re.Pattern[str]] | None = None,
        allow_entity_names: bool = True,
        allow_roman_numerals: bool = True,
        ignore_urls: bool = True,
        ignore_markup_and_placeholders: bool = True,
        skip_exonym_leaks: bool = True,
        snippet_radius: int = 30,
    ) -> None:
        if expected_script is None:
            self.expected_scripts: frozenset[str] | None = None
        elif isinstance(expected_script, str):
            self.expected_scripts = frozenset({_normalize_script_name(expected_script)})
        else:
            normalized = frozenset(_normalize_script_name(s) for s in expected_script if s.strip())
            if not normalized:
                raise ValueError("expected_script sequence must contain at least one script name.")
            self.expected_scripts = normalized

        self.severity: AuditSeverity = severity
        self.mixed_script_severity: AuditSeverity = mixed_script_severity
        self.allowed_tokens: tuple[str, ...] = tuple(
            t.strip() for t in (allowed_tokens or ()) if t and t.strip()
        )
        self.allowed_patterns: tuple[re.Pattern[str], ...] = tuple(
            p if isinstance(p, re.Pattern) else re.compile(p)
            for p in (allowed_patterns or ())
        )
        self.allow_entity_names = allow_entity_names
        self.allow_roman_numerals = allow_roman_numerals
        self.ignore_urls = ignore_urls
        self.ignore_markup_and_placeholders = ignore_markup_and_placeholders
        self.skip_exonym_leaks = skip_exonym_leaks
        self.snippet_radius = snippet_radius

    def _resolve_expected_scripts(self, ctx: AuditContext) -> frozenset[str] | None:
        if self.expected_scripts is not None:
            return self.expected_scripts
        return _infer_scripts_from_lang(ctx.lang)

    def _mask_exempt_regions(self, ctx: AuditContext, text: str) -> str:
        masked = text
        if self.ignore_urls:
            masked = _mask_spans(masked, _URL_OR_EMAIL_RE)
        if self.ignore_markup_and_placeholders:
            masked = _mask_spans(masked, _MARKUP_OR_PLACEHOLDER_RE)
        for pat in self.allowed_patterns:
            masked = _mask_spans(masked, pat)

        # Mask user-provided allowlisted tokens/phrases (longest first to handle multi-word phrases)
        if self.allowed_tokens:
            for phrase in sorted(self.allowed_tokens, key=len, reverse=True):
                masked = _mask_spans(masked, re.compile(_phrase_pattern(phrase), re.IGNORECASE))

        # Mask canon-approved entity names/aliases so glossary terms don't trigger false positives
        if self.allow_entity_names and ctx.entities:
            exempt_phrases: set[str] = set()
            ctx_lang = ctx.lang.strip().lower() if ctx.lang and ctx.lang.strip() else None
            for entity in ctx.entities:
                raw_target = entity.base_attributes.get("target_name")
                if isinstance(raw_target, str) and raw_target.strip():
                    exempt_phrases.add(raw_target.strip())
                if entity.base_attributes.get("allow_source_name") and entity.canonical_name.strip():
                    exempt_phrases.add(entity.canonical_name.strip())
            for al in ctx.aliases:
                al_str = al.alias.strip()
                if not al_str:
                    continue
                al_lang = al.lang.strip().lower() if al.lang and al.lang.strip() else None
                if al.kind == "inflection" or (ctx_lang is not None and al_lang == ctx_lang):
                    exempt_phrases.add(al_str)
            for phrase in sorted(exempt_phrases, key=len, reverse=True):
                masked = _mask_spans(masked, re.compile(_phrase_pattern(phrase), re.IGNORECASE))

        # Avoid double-reporting verbatim canonical_name leaks already owned by ExonymLeakRule
        if self.skip_exonym_leaks and ctx.entities:
            aliases_by_entity: dict[str, list[str]] = {}
            for a in ctx.aliases:
                if a.alias.strip():
                    aliases_by_entity.setdefault(a.entity_id, []).append(a.alias.strip())
            has_source = bool(ctx.source_text.strip())
            for entity in ctx.entities:
                if entity.base_attributes.get("allow_source_name"):
                    continue
                canonical = entity.canonical_name.strip()
                raw_target = entity.base_attributes.get("target_name")
                if not canonical or not isinstance(raw_target, str) or not raw_target.strip():
                    continue
                target_name = raw_target.strip()
                if canonical.lower() == target_name.lower():
                    continue
                if re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE):
                    continue
                if has_source:
                    cues = [canonical, *aliases_by_entity.get(entity.entity_id, ())]
                    if not any(
                        re.search(_phrase_pattern(cue), ctx.source_text, re.IGNORECASE)
                        for cue in cues
                        if cue
                    ):
                        continue
                masked = _mask_spans(masked, re.compile(_phrase_pattern(canonical)))

        return masked

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if not ctx.target_text.strip():
            return []

        expected = self._resolve_expected_scripts(ctx)
        if not expected:
            return []

        masked = self._mask_exempt_regions(ctx, ctx.target_text)
        expected_sorted = sorted(expected)
        expected_label = "/".join(expected_sorted)

        # Group pure-foreign tokens by detected script, and mixed-script homoglyphs separately
        by_script: dict[str, list[tuple[str, int, int, int]]] = {}
        mixed_hits: list[tuple[str, int, int, list[str]]] = []

        for match in _LETTER_TOKEN_RE.finditer(masked):
            tok = match.group(0)
            scripts_in_tok: list[str] = []
            seen_in_tok: set[str] = set()
            unexpected_char_count = 0
            for ch in tok:
                sc = _char_script(ch)
                if sc is None:
                    continue
                if sc not in seen_in_tok:
                    seen_in_tok.add(sc)
                    scripts_in_tok.append(sc)
                if sc not in expected:
                    unexpected_char_count += 1

            if unexpected_char_count == 0 or not scripts_in_tok:
                continue

            # Exempt standard uppercase Roman numerals (e.g. IV, XII) when Latin is not expected
            if (
                self.allow_roman_numerals
                and seen_in_tok == {"latin"}
                and tok.isupper()
                and _ROMAN_NUMERAL_RE.match(tok) is not None
            ):
                continue

            start, end = match.start(), match.end()
            if len(seen_in_tok) >= 2:
                mixed_hits.append((tok, start, end, sorted(seen_in_tok)))
            else:
                found_script = scripts_in_tok[0]
                by_script.setdefault(found_script, []).append(
                    (tok, start, end, unexpected_char_count)
                )

        issues: list[AuditIssue] = []

        if mixed_hits:
            first_tok, first_start, first_end, _ = mixed_hits[0]
            unique_tokens = list(dict.fromkeys(tok for tok, _, _, _ in mixed_hits))
            scripts_by_tok: dict[str, list[str]] = {}
            for tok, _, _, scs in mixed_hits:
                if tok not in scripts_by_tok and len(scripts_by_tok) < 10:
                    scripts_by_tok[tok] = scs
            sample_str = ", ".join(repr(t) for t in unique_tokens[:5])
            issues.append(
                AuditIssue(
                    code="mixed_script_token",
                    rule_name=self.name,
                    severity=self.mixed_script_severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key="script:mixed",
                    message=(
                        f"Detected {len(mixed_hits)} mixed-script token(s) in target_text "
                        f"({sample_str}); expected only {expected_label} script."
                    ),
                    evidence={
                        "expected_scripts": expected_sorted,
                        "mixed_tokens": unique_tokens[:10],
                        "mixed_scripts_by_token": scripts_by_tok,
                        "offending_token_count": len(mixed_hits),
                        "snippet": _extract_snippet(
                            ctx.target_text, first_start, first_end, self.snippet_radius
                        ),
                    },
                    suggestion=(
                        f"Fix mixed-script homoglyph characters in {sample_str} so all letters "
                        f"use {expected_label} script."
                    ),
                )
            )

        for found_script in sorted(by_script.keys()):
            hits = by_script[found_script]
            _, first_start, first_end, _ = hits[0]
            unique_tokens = list(dict.fromkeys(tok for tok, _, _, _ in hits))
            total_chars = sum(c for _, _, _, c in hits)
            sample_str = ", ".join(repr(t) for t in unique_tokens[:5])
            issues.append(
                AuditIssue(
                    code="script_drift",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=f"script:{found_script}",
                    message=(
                        f"Unexpected {found_script} script detected in target_text "
                        f"({len(hits)} token(s), {total_chars} char(s): {sample_str}); "
                        f"expected {expected_label}."
                    ),
                    evidence={
                        "expected_scripts": expected_sorted,
                        "found_script": found_script,
                        "offending_tokens": unique_tokens[:10],
                        "offending_token_count": len(hits),
                        "offending_char_count": total_chars,
                        "snippet": _extract_snippet(
                            ctx.target_text, first_start, first_end, self.snippet_radius
                        ),
                    },
                    suggestion=(
                        f"Rewrite the unexpected {found_script} text ({sample_str}) "
                        f"in {expected_label} script."
                    ),
                )
            )

        return issues


class NearDuplicateEntityRule:
    """Detects near-duplicate EntityRecord creation via fuzzy name or alias/exonym collision."""

    name: str = "near_duplicate_entity"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 88.0,
        severity: AuditSeverity = "warning",
        same_kind_only: bool = True,
        check_all_entities: bool = False,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.severity: AuditSeverity = severity
        self.same_kind_only = same_kind_only
        self.check_all_entities = check_all_entities

    @staticmethod
    def _ignored_targets(entity: EntityRecord) -> set[str]:
        raw = entity.base_attributes.get("ignore_duplicate_of")
        if isinstance(raw, str) and raw.strip():
            return {raw.strip()}
        if isinstance(raw, (list, tuple, set)):
            return {str(x).strip() for x in raw if str(x).strip()}
        return set()

    @staticmethod
    def _candidate_labels(entity: EntityRecord) -> list[tuple[str, str]]:
        labels: list[tuple[str, str]] = []
        if entity.canonical_name.strip():
            labels.append(("canonical_name", _normalize_label(entity.canonical_name)))
        tname = entity.base_attributes.get("target_name")
        if isinstance(tname, str) and tname.strip():
            norm_t = _normalize_label(tname)
            if norm_t and norm_t not in {lbl for _, lbl in labels}:
                labels.append(("target_name", norm_t))
        slug = _entity_id_slug(entity.entity_id)
        if slug and len(slug) >= 4 and slug not in {lbl for _, lbl in labels}:
            labels.append(("entity_id", slug))
        return [(src, lbl) for src, lbl in labels if lbl]

    def _compare_pair(
        self,
        candidate: EntityRecord,
        existing: EntityRecord,
        existing_aliases: Sequence[AliasRecord],
    ) -> tuple[float, str, str, str] | None:
        """Returns (similarity, candidate_field, existing_field, matched_value) if colliding."""
        if existing.entity_id in self._ignored_targets(candidate):
            return None
        if candidate.entity_id in self._ignored_targets(existing):
            return None

        cand_labels = self._candidate_labels(candidate)

        # Build existing labels: canonical_name, target_name, and aliases/personas
        exist_labels: list[tuple[str, str]] = []
        if existing.canonical_name.strip():
            exist_labels.append(("canonical_name", _normalize_label(existing.canonical_name)))
        etname = existing.base_attributes.get("target_name")
        if isinstance(etname, str) and etname.strip():
            exist_labels.append(("target_name", _normalize_label(etname)))
        for al in existing_aliases:
            if al.alias.strip():
                exist_labels.append((f"alias:{al.alias}", _normalize_label(al.alias)))
            if al.persona_name and al.persona_name.strip():
                exist_labels.append((f"persona:{al.persona_name}", _normalize_label(al.persona_name)))

        best: tuple[float, str, str, str] | None = None
        for c_src, c_lbl in cand_labels:
            for e_src, e_lbl in exist_labels:
                if not e_lbl:
                    continue
                is_alias = e_src.startswith("alias:") or e_src.startswith("persona:")
                # Respect same_kind_only for name-to-name comparisons; aliases still catch cross-kind splits
                if self.same_kind_only and not is_alias and candidate.kind != existing.kind:
                    continue
                if c_lbl == e_lbl:
                    score = 100.0
                elif len(c_lbl) >= 4 and len(e_lbl) >= 4:
                    # Use ratio (strict) rather than WRatio to avoid partial-substring false positives
                    score = float(fuzz.ratio(c_lbl, e_lbl))
                else:
                    continue

                if score >= self.fuzzy_threshold:
                    if best is None or score > best[0]:
                        best = (score, c_src, e_src, e_lbl)
        return best

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if len(ctx.entities) < 2:
            return []

        aliases_by_entity: dict[str, list[AliasRecord]] = {}
        for al in ctx.aliases:
            aliases_by_entity.setdefault(al.entity_id, []).append(al)

        issues: list[AuditIssue] = []
        seen_pairs: set[tuple[str, str]] = set()

        if self.check_all_entities:
            sorted_entities = sorted(
                ctx.entities, key=lambda e: (e.valid_from_step, e.entity_id)
            )
            pairs_to_check: list[tuple[EntityRecord, EntityRecord]] = []
            for idx, cand in enumerate(sorted_entities):
                for prev in sorted_entities[:idx]:
                    pairs_to_check.append((cand, prev))
        else:
            if not ctx.new_entities:
                return []
            new_ids = {e.entity_id for e in ctx.new_entities}
            prior_entities = [e for e in ctx.entities if e.entity_id not in new_ids]
            pairs_to_check = []
            sorted_new = sorted(ctx.new_entities, key=lambda e: e.entity_id)
            for idx, cand in enumerate(sorted_new):
                for existing in prior_entities:
                    pairs_to_check.append((cand, existing))
                for earlier_new in sorted_new[:idx]:
                    pairs_to_check.append((cand, earlier_new))

        for cand, existing in pairs_to_check:
            pair_key = (
                min(cand.entity_id, existing.entity_id),
                max(cand.entity_id, existing.entity_id),
            )
            if pair_key in seen_pairs:
                continue
            hit = self._compare_pair(
                cand, existing, aliases_by_entity.get(existing.entity_id, ())
            )
            if hit is None:
                continue
            score, c_src, e_src, matched_val = hit
            seen_pairs.add(pair_key)
            issues.append(
                AuditIssue(
                    code="near_duplicate_entity",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=cand.entity_id,
                    entity_ids=[cand.entity_id, existing.entity_id],
                    message=(
                        f"Entity '{cand.entity_id}' ({cand.canonical_name}) collides with existing "
                        f"entity '{existing.entity_id}' ({existing.canonical_name}) via {e_src} "
                        f"(similarity={score:.1f})."
                    ),
                    evidence={
                        "new_entity_id": cand.entity_id,
                        "new_canonical_name": cand.canonical_name,
                        "existing_entity_id": existing.entity_id,
                        "existing_canonical_name": existing.canonical_name,
                        "candidate_field": c_src,
                        "matched_against": e_src,
                        "matched_value": matched_val,
                        "similarity": round(score, 2),
                    },
                    suggestion=(
                        f"Merge '{cand.entity_id}' into '{existing.entity_id}' via "
                        f"engine.merge_entities('{cand.entity_id}', '{existing.entity_id}') "
                        f"(or register '{cand.canonical_name}' as an AliasRecord on '{existing.entity_id}')."
                    ),
                )
            )

        return issues


class UnexplainedRegisterFlipRule:
    """Flags mutations to hard-filter edge keys (e.g. 'register') lacking causal provenance."""

    name: str = "unexplained_register_flip"
    required_tiers: frozenset[int] = frozenset({2, 3})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "warning",
        keys: Sequence[str] | None = None,
        skip_step_zero: bool = True,
        check_dangling_episode: bool = True,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.keys: tuple[str, ...] | None = tuple(keys) if keys is not None else None
        self.skip_step_zero = skip_step_zero
        self.check_dangling_episode = check_dangling_episode

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if self.skip_step_zero and ctx.step == 0:
            return []

        tracked_keys = self.keys if self.keys is not None else ctx.policy.hard_filter_keys
        if not tracked_keys:
            return []

        issues: list[AuditIssue] = []
        for edge in ctx.mutated_edges:
            if edge.prior_state is None or edge.revision_count <= 1:
                continue
            subject_key = f"{edge.source_id}:{edge.relation_type}->{edge.target_id}"

            for key in tracked_keys:
                prior_val = edge.prior_state.get(key)
                new_val = edge.state.get(key)
                if prior_val is None or new_val is None:
                    continue
                if str(prior_val).strip().lower() == str(new_val).strip().lower():
                    continue

                has_ep = bool(edge.caused_by_episode_id and edge.caused_by_episode_id.strip())
                has_reason = bool(edge.reason and edge.reason.strip())

                if not has_ep and not has_reason:
                    issues.append(
                        AuditIssue(
                            code="unexplained_register_flip",
                            rule_name=self.name,
                            severity=self.severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} ({edge.relation_type}) "
                                f"flipped '{key}' from '{prior_val}' to '{new_val}' without "
                                "caused_by_episode_id or reason."
                            ),
                            evidence={
                                "key": key,
                                "prior_value": prior_val,
                                "new_value": new_val,
                                "prior_step": edge.prior_step,
                                "caused_by_episode_id": edge.caused_by_episode_id,
                                "reason": edge.reason,
                            },
                            suggestion=(
                                "Attach caused_by_episode_id or a concise reason on the EdgeDelta "
                                "to document why this register shift occurred."
                            ),
                        )
                    )
                elif (
                    self.check_dangling_episode
                    and has_ep
                    and ctx.policy.has_tier(6)
                    and edge.caused_by_episode_id not in ctx.all_episode_ids
                ):
                    issues.append(
                        AuditIssue(
                            code="dangling_causal_episode",
                            rule_name=self.name,
                            severity="error",
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} ({edge.relation_type}) "
                                f"flipped '{key}' referencing unknown caused_by_episode_id "
                                f"'{edge.caused_by_episode_id}'."
                            ),
                            evidence={
                                "key": key,
                                "prior_value": prior_val,
                                "new_value": new_val,
                                "caused_by_episode_id": edge.caused_by_episode_id,
                            },
                            suggestion="Ensure caused_by_episode_id matches a committed EpisodeRecord.episode_id.",
                        )
                    )

        return issues


class RegisterVocabularyDriftRule:
    """Detects mixed register vocabularies (e.g. Vi/ti mixed with formal/informal) or invalid values."""

    name: str = "register_vocabulary_drift"
    required_tiers: frozenset[int] = frozenset({2})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "warning",
        key: str = "register",
        allowed_values: Sequence[str] | None = None,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.key = key
        self.allowed_values: frozenset[str] | None = (
            frozenset(v.strip().lower() for v in allowed_values if v.strip())
            if allowed_values is not None
            else None
        )

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        if self.allowed_values is not None:
            for edge in ctx.edges:
                val = edge.state.get(self.key)
                if val is None or not str(val).strip():
                    continue
                if str(val).strip().lower() not in self.allowed_values:
                    subject_key = f"{edge.source_id}:{edge.relation_type}->{edge.target_id}"
                    issues.append(
                        AuditIssue(
                            code="register_vocabulary_drift",
                            rule_name=self.name,
                            severity=self.severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=subject_key,
                            entity_ids=[edge.source_id, edge.target_id],
                            message=(
                                f"Edge {edge.source_id} → {edge.target_id} has {self.key}='{val}', "
                                f"which is outside allowed values {sorted(self.allowed_values)}."
                            ),
                            evidence={
                                "key": self.key,
                                "value": val,
                                "allowed_values": sorted(self.allowed_values),
                            },
                            suggestion=f"Normalize '{val}' to one of {sorted(self.allowed_values)}.",
                        )
                    )
            return issues

        # Auto-detect mixed T-V tokens (Vi/ti, vous/tu, Sie/du) alongside generic English tokens (formal/informal)
        tv_seen: set[str] = set()
        generic_seen: set[str] = set()
        for edge in ctx.edges:
            val = edge.state.get(self.key)
            if val is None:
                continue
            low = str(val).strip().lower()
            if low in _TV_REGISTER_TOKENS:
                tv_seen.add(str(val).strip())
            elif low in _GENERIC_REGISTER_TOKENS:
                generic_seen.add(str(val).strip())

        if tv_seen and generic_seen:
            issues.append(
                AuditIssue(
                    code="register_vocabulary_drift",
                    rule_name=self.name,
                    severity=self.severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    message=(
                        f"Mixed '{self.key}' vocabularies active on timeline '{ctx.timeline_id}': "
                        f"T-V tokens {sorted(tv_seen)} mixed with generic labels {sorted(generic_seen)}."
                    ),
                    evidence={
                        "key": self.key,
                        "tv_values": sorted(tv_seen),
                        "generic_values": sorted(generic_seen),
                    },
                    suggestion=(
                        "Standardize on one register vocabulary (e.g. Vi/ti or formal/informal) "
                        "so VoiceBank hard-filtering stays consistent."
                    ),
                )
            )
        return issues


def default_audit_rules(preset: DomainPreset | None = None) -> list[AuditRule]:
    """Returns the default built-in AuditRule suite."""
    _ = preset  # Reserved for future preset-specific defaults; lexical rules are no-ops without target_text.
    return [
        ExonymLeakRule(),
        TargetTermComplianceRule(),
        NearDuplicateEntityRule(),
        UnexplainedRegisterFlipRule(),
        RegisterVocabularyDriftRule(),
    ]


def _load_graph_state(
    engine: CanonArcEngine,
    timeline_id: str,
    step: int,
) -> tuple[list[EntityRecord], list[AliasRecord], list[DirectedEdgeRecord]]:
    get_state = getattr(engine.store, "get_active_graph_state", None)
    if callable(get_state):
        return get_state(timeline_id, step)
    entities = engine.store.get_all_active_entities(timeline_id, step)
    aliases = engine.store.get_all_active_aliases(timeline_id, step)
    edges = engine.store.get_all_active_edges(timeline_id, step)
    return entities, aliases, edges


def build_audit_context(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    timeline_id: str | None = None,
    lang: str | None = None,
) -> AuditContext:
    """Constructs a read-only AuditContext snapshot for a single step."""
    tl = timeline_id or engine.active_timeline
    entities, aliases, edges = _load_graph_state(engine, tl, step)
    new_entities = tuple(
        e for e in entities if e.valid_from_step == step and e.revision_count <= 1
    )
    mutated_edges = tuple(e for e in edges if e.valid_from_step == step)

    episodes_at_step: tuple[EpisodeRecord, ...] = ()
    all_episode_ids: frozenset[str] = frozenset()
    if engine._policy.has_tier(6):
        ep_rows = engine.store.get_lineage_episodes(tl, step)
        all_eps = [ep for ep, _ in ep_rows]
        all_episode_ids = frozenset(ep.episode_id for ep in all_eps if ep.episode_id)
        episodes_at_step = tuple(ep for ep in all_eps if ep.step == step)

    return AuditContext(
        timeline_id=tl,
        step=step,
        source_text=source_text,
        target_text=target_text,
        entities=tuple(entities),
        new_entities=new_entities,
        aliases=tuple(aliases),
        edges=tuple(edges),
        mutated_edges=mutated_edges,
        episodes=episodes_at_step,
        all_episode_ids=all_episode_ids,
        policy=engine._policy,
        lang=lang,
    )


def _select_rules(
    engine: CanonArcEngine,
    rules: Sequence[AuditRule] | None,
    *,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
) -> list[AuditRule]:
    if rules is not None:
        selected = list(rules)
    elif engine.audit_rules is not None:
        selected = list(engine.audit_rules)
    else:
        selected = default_audit_rules(engine.preset)

    if include_rules is not None:
        allowed = {r.strip() for r in include_rules if r.strip()}
        selected = [r for r in selected if r.name in allowed]
    if exclude_rules is not None:
        blocked = {r.strip() for r in exclude_rules if r.strip()}
        selected = [r for r in selected if r.name not in blocked]

    active_tiers = engine._policy.active_tiers
    return [r for r in selected if r.required_tiers.issubset(active_tiers)]


def run_audit_step(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    timeline_id: str | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
) -> AuditReport:
    """Runs all applicable AuditRules against a single timeline step."""
    tl = timeline_id or engine.active_timeline
    ctx = build_audit_context(
        engine,
        step,
        source_text=source_text,
        target_text=target_text,
        timeline_id=tl,
        lang=lang,
    )
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    min_rank = _SEVERITY_RANK[min_severity]

    issues: list[AuditIssue] = []
    for rule in active_rules:
        for issue in rule.check(ctx):
            if _SEVERITY_RANK[issue.severity] >= min_rank:
                issues.append(issue)

    return AuditReport(
        timeline_id=tl,
        steps_audited=[step],
        issues=issues,
        rules_executed=[r.name for r in active_rules],
    )


def run_audit_draft(
    engine: CanonArcEngine,
    step: int,
    *,
    source_text: str = "",
    target_text: str = "",
    context: StepContext | None = None,
    timeline_id: str | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
    include_all_timeline_entities: bool = False,
) -> AuditReport:
    """Audits a pre-commit draft translation against the pre-commit state (step - 1 / StepContext)."""
    tl = timeline_id or (context.timeline_id if context is not None else engine.active_timeline)
    cutoff_step = max(0, step - 1)
    all_entities, all_aliases, all_edges = _load_graph_state(engine, tl, cutoff_step)

    if context is not None and not include_all_timeline_entities:
        # Use entities already resolved/active in tx.context (plus related_entities) and their aliases
        ctx_entity_map: dict[str, EntityRecord] = {e.entity_id: e for e in context.entities}
        for rel in context.related_entities:
            ctx_entity_map.setdefault(rel.entity_id, rel)
        entities = tuple(ctx_entity_map.values())
        active_ids = set(ctx_entity_map.keys())
        aliases = tuple(a for a in all_aliases if a.entity_id in active_ids)
        edges = tuple(context.active_edges)
        resolved_mentions = tuple(context.resolved_mentions)
    else:
        entities = tuple(all_entities)
        aliases = tuple(all_aliases)
        edges = tuple(all_edges)
        resolved_mentions = tuple(context.resolved_mentions) if context is not None else ()

    ctx = AuditContext(
        timeline_id=tl,
        step=step,
        source_text=source_text,
        target_text=target_text,
        entities=entities,
        new_entities=(),
        aliases=aliases,
        edges=edges,
        mutated_edges=(),
        episodes=tuple(context.episodes) if context is not None else (),
        all_episode_ids=frozenset(
            ep.episode_id for ep in (context.episodes if context is not None else ()) if ep.episode_id
        ),
        resolved_mentions=resolved_mentions,
        policy=engine._policy,
        lang=lang,
    )
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    min_rank = _SEVERITY_RANK[min_severity]

    issues: list[AuditIssue] = []
    for rule in active_rules:
        for issue in rule.check(ctx):
            if _SEVERITY_RANK[issue.severity] >= min_rank:
                issues.append(issue)

    return AuditReport(
        timeline_id=tl,
        steps_audited=[step],
        issues=issues,
        rules_executed=[r.name for r in active_rules],
    )


def run_audit_timeline(
    engine: CanonArcEngine,
    *,
    timeline_id: str | None = None,
    from_step: int = 0,
    to_step: int | None = None,
    texts: Mapping[int, str | tuple[str, str]] | None = None,
    rules: Sequence[AuditRule] | None = None,
    include_rules: Sequence[str] | None = None,
    exclude_rules: Sequence[str] | None = None,
    min_severity: AuditSeverity = "info",
    lang: str | None = None,
) -> AuditReport:
    """Audits every step in [from_step..to_step] on a timeline and aggregates issues."""
    tl = timeline_id or engine.active_timeline
    tl_info = engine.store.get_timeline(tl)
    if tl_info is None:
        raise ValueError(f"Timeline '{tl}' does not exist.")

    end_step = tl_info.head_step if to_step is None else to_step
    if texts:
        max_text_step = max(texts.keys())
        if to_step is None and max_text_step > end_step:
            end_step = max_text_step

    steps = list(range(max(0, from_step), end_step + 1))
    active_rules = _select_rules(
        engine, rules, include_rules=include_rules, exclude_rules=exclude_rules
    )
    min_rank = _SEVERITY_RANK[min_severity]

    all_issues: list[AuditIssue] = []
    seen_global_keys: set[tuple[str, str, str | None]] = set()

    for s in steps:
        src = ""
        tgt = ""
        if texts and s in texts:
            entry = texts[s]
            if isinstance(entry, tuple):
                src, tgt = entry[0], entry[1]
            else:
                tgt = entry

        ctx = build_audit_context(
            engine,
            s,
            source_text=src,
            target_text=tgt,
            timeline_id=tl,
            lang=lang,
        )
        for rule in active_rules:
            for issue in rule.check(ctx):
                if _SEVERITY_RANK[issue.severity] < min_rank:
                    continue
                # Deduplicate timeline-wide drift warnings across consecutive steps
                if issue.code == "register_vocabulary_drift":
                    dedup_key = (issue.code, issue.message, issue.subject_key)
                    if dedup_key in seen_global_keys:
                        continue
                    seen_global_keys.add(dedup_key)
                all_issues.append(issue)

    return AuditReport(
        timeline_id=tl,
        steps_audited=steps,
        issues=all_issues,
        rules_executed=[r.name for r in active_rules],
    )
