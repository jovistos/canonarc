from __future__ import annotations

import re
from collections.abc import Sequence

from rapidfuzz import fuzz

from ..embedders import _CJK_CHAR_RE
from ..models import AliasRecord, EntityRecord, ResolvedMention
from ..resolver import _VOWELS, DefaultLexicalResolver
from .models import AuditContext, AuditIssue, AuditSeverity
from .text_utils import (
    _LETTER_TOKEN_RE,
    _MARKUP_OR_PLACEHOLDER_RE,
    _ROMAN_NUMERAL_RE,
    _URL_OR_EMAIL_RE,
    _char_script,
    _extract_snippet,
    _infer_scripts_from_lang,
    _mask_spans,
    _normalize_script_name,
    _phrase_pattern,
)


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
        target_text = ctx.masked_target_text
        if not target_text.strip():
            return []

        issues: list[AuditIssue] = []
        aliases_by_entity: dict[str, list[str]] = {}
        for a in ctx.aliases:
            if a.alias.strip():
                aliases_by_entity.setdefault(a.entity_id, []).append(a.alias.strip())

        flags = 0 if self.case_sensitive else re.IGNORECASE
        source_text = ctx.masked_source_text
        has_source = bool(source_text.strip())
        snippet_text = ctx.target_snippet_text

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

            if self.only_source_mentioned and has_source:
                cues = [canonical, *aliases_by_entity.get(entity.entity_id, ())]
                if not any(
                    re.search(_phrase_pattern(cue), source_text, re.IGNORECASE)
                    for cue in cues
                    if cue
                ):
                    continue

            matches = list(re.finditer(_phrase_pattern(canonical), target_text, flags))
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
                            snippet_text, first.start(), first.end(), self.snippet_radius
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
        for _, lower_tok, _, _ in raw_token_spans:
            if lower_tok == ref_word_lower:
                return True

        if _CJK_CHAR_RE.match(ref_word_lower[0]):
            return False

        if 3 <= len(ref_word_lower) <= 6:
            stems = DefaultLexicalResolver._short_name_stems(ref_word_lower)
            for raw_tok, lower_tok, _, _ in raw_token_spans:
                if not raw_tok or not raw_tok[0].isupper():
                    continue
                for stem, allowed_suffixes in stems:
                    if (
                        lower_tok.startswith(stem)
                        and len(lower_tok) > len(stem)
                        and lower_tok[len(stem) :] in allowed_suffixes
                    ):
                        return True

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
                    if (
                        lower_tok.startswith(stem)
                        and len(lower_tok) <= len(ref_word_lower) + 4
                        and float(fuzz.ratio(ref_word_lower, lower_tok)) >= fuzzy_stem_threshold
                    ):
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

        for form in accepted_forms:
            if not form:
                continue
            if require_full_multiword and len(re.findall(r"\w+", form)) < 2:
                continue
            if re.search(_phrase_pattern(form), target_text, flags):
                return True

        if not require_full_multiword:
            for form in accepted_forms:
                if " " not in form and self._word_matches_token(
                    form.lower(),
                    raw_token_spans,
                    fuzzy_stem_threshold=self.fuzzy_stem_threshold,
                ):
                    return True

        if len(target_words) >= 2:
            if self.allow_partial_target_match and not require_full_multiword:
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
                if not any(tok and tok[0].isupper() for tok, _, _, _ in slice_spans):
                    continue
                if win > 1:
                    start = slice_spans[0][2]
                    end = slice_spans[-1][3]
                    cand_raw = target_text[start:end]
                    if re.search(r"[.!?\n;:]", cand_raw):
                        continue
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
                    if score >= self.near_miss_threshold and (
                        best is None or score > best[3] or (score == best[3] and len(cand_raw) > len(best[0]))
                    ):
                        best = (cand_raw, start, end, score)
        return best

    @staticmethod
    def _find_source_cue(
        canonical: str,
        ent_aliases: Sequence[AliasRecord],
        ent_mentions: Sequence[ResolvedMention] | None,
        target_lang: str | None,
        source_text: str,
    ) -> str | None:
        if ent_mentions:
            return ent_mentions[0].surface_text.strip() or ent_mentions[0].matched_phrase
        source_cues = [canonical]
        for a in ent_aliases:
            if a.kind == "inflection":
                continue
            if target_lang is not None and a.lang and a.lang.strip().lower() == target_lang:
                continue
            source_cues.append(a.alias.strip())
        for cue in source_cues:
            if cue and re.search(_phrase_pattern(cue), source_text, re.IGNORECASE):
                return cue
        return None

    @staticmethod
    def _collect_accepted_target_forms(
        target_name: str,
        ent_aliases: Sequence[AliasRecord],
        target_lang: str | None,
    ) -> list[str]:
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
        return accepted_forms

    def _build_compliance_issue(
        self,
        ctx: AuditContext,
        *,
        entity: EntityRecord,
        canonical: str,
        source_cue: str,
        target_name: str,
        accepted_forms: list[str],
        near_miss: tuple[str, int, int, float] | None,
        snippet_text: str,
    ) -> AuditIssue:
        inflection_examples = accepted_forms[1:5]
        inflection_hint = (
            f" (or valid inflections such as {', '.join(inflection_examples)})"
            if inflection_examples
            else " (or its appropriate inflected form)"
        )
        if near_miss is not None:
            variant_str, start, end, score = near_miss
            snippet = _extract_snippet(snippet_text, start, end, self.snippet_radius)
            return AuditIssue(
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
        return AuditIssue(
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

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        source_text = ctx.masked_source_text
        target_text = ctx.masked_target_text
        if not source_text.strip() or not target_text.strip():
            return []
        snippet_text = ctx.target_snippet_text

        aliases_by_entity: dict[str, list[AliasRecord]] = {}
        for a in ctx.aliases:
            if a.alias.strip():
                aliases_by_entity.setdefault(a.entity_id, []).append(a)

        mentions_by_entity: dict[str, list[ResolvedMention]] = {}
        for rm in ctx.resolved_mentions:
            mentions_by_entity.setdefault(rm.entity_id, []).append(rm)

        raw_token_spans = [
            (m.group(0), m.group(0).lower(), m.start(), m.end())
            for m in re.finditer(r"\w+", target_text)
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

            source_cue = self._find_source_cue(
                canonical,
                ent_aliases,
                mentions_by_entity.get(entity.entity_id),
                target_lang,
                source_text,
            )
            if not source_cue:
                continue

            if entity.base_attributes.get("allow_source_name"):
                if re.search(_phrase_pattern(canonical), target_text, re.IGNORECASE):
                    continue
            elif (
                canonical.lower() != target_name.lower()
                and not re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE)
                and re.search(_phrase_pattern(canonical), target_text, 0)
            ):
                continue

            accepted_forms = self._collect_accepted_target_forms(
                target_name, ent_aliases, target_lang
            )
            if self._matches_target(
                target_name, accepted_forms, source_cue, target_text, raw_token_spans
            ):
                continue

            near_miss = self._find_suspected_variant(
                canonical, target_name, target_text, raw_token_spans
            )
            issues.append(
                self._build_compliance_issue(
                    ctx,
                    entity=entity,
                    canonical=canonical,
                    source_cue=source_cue,
                    target_name=target_name,
                    accepted_forms=accepted_forms,
                    near_miss=near_miss,
                    snippet_text=snippet_text,
                )
            )

        return issues


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

        if self.allowed_tokens:
            for phrase in sorted(self.allowed_tokens, key=len, reverse=True):
                masked = _mask_spans(masked, re.compile(_phrase_pattern(phrase), re.IGNORECASE))

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

        if self.skip_exonym_leaks and ctx.entities:
            aliases_by_entity: dict[str, list[str]] = {}
            for a in ctx.aliases:
                if a.alias.strip():
                    aliases_by_entity.setdefault(a.entity_id, []).append(a.alias.strip())
            source_text = ctx.masked_source_text
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
                if re.search(_phrase_pattern(canonical), target_name, re.IGNORECASE):
                    continue
                if has_source:
                    cues = [canonical, *aliases_by_entity.get(entity.entity_id, ())]
                    if not any(
                        re.search(_phrase_pattern(cue), source_text, re.IGNORECASE)
                        for cue in cues
                        if cue
                    ):
                        continue
                masked = _mask_spans(masked, re.compile(_phrase_pattern(canonical)))

        return masked

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        target_text = ctx.masked_target_text
        if not target_text.strip():
            return []

        expected = self._resolve_expected_scripts(ctx)
        if not expected:
            return []

        masked = self._mask_exempt_regions(ctx, target_text)
        snippet_text = ctx.target_snippet_text
        expected_sorted = sorted(expected)
        expected_label = "/".join(expected_sorted)

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
            _first_tok, first_start, first_end, _ = mixed_hits[0]
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
                            snippet_text, first_start, first_end, self.snippet_radius
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
                            snippet_text, first_start, first_end, self.snippet_radius
                        ),
                    },
                    suggestion=(
                        f"Rewrite the unexpected {found_script} text ({sample_str}) "
                        f"in {expected_label} script."
                    ),
                )
            )

        return issues
