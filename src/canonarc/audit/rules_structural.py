from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from rapidfuzz import fuzz

from ..text_masking import (
    COMMON_PLACEHOLDER_PATTERNS,
    compile_ignore_patterns,
    mask_ignored_spans,
    strip_ignored_spans,
)
from .models import AuditContext, AuditIssue, AuditSeverity
from .text_utils import _extract_snippet, _mask_spans, _phrase_pattern

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_DIGITS_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"(?:[.!?…]+(?=\s|$)|[。！？]+)")
_EDGE_PUNCT_STRIP = " \t\r\n\"'“”‘’„«»()[]{}.,;:!?—–-"


def _extract_placeholder_hits(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> list[tuple[str, int, int, int]]:
    """Extracts (token, pattern_idx, start, end) broadest-first without double-counting nested spans."""
    if not text or not patterns:
        return []
    remaining = text
    hits: list[tuple[str, int, int, int]] = []
    for pat_idx, pat in enumerate(patterns):
        for match in pat.finditer(remaining):
            hits.append((match.group(0), pat_idx, match.start(), match.end()))
        remaining = mask_ignored_spans(remaining, (pat,))
    hits.sort(key=lambda item: (item[2], item[3]))
    return hits


def _mutation_pair_score(src_tok: str, tgt_tok: str) -> float:
    src_nums = _DIGITS_RE.findall(src_tok)
    tgt_nums = _DIGITS_RE.findall(tgt_tok)
    num_bonus = 0.0
    if src_nums and tgt_nums:
        num_bonus = 40.0 if src_nums == tgt_nums else -20.0
    return num_bonus + float(fuzz.ratio(src_tok.lower(), tgt_tok.lower()))


def _normalize_for_echo(text: str) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return collapsed.strip(_EDGE_PUNCT_STRIP)


def _letter_word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(_SENTENCE_END_RE.findall(stripped)))


class PlaceholderPreservationRule:
    """Deterministic rule verifying that placeholders in source_text appear verbatim with identical multiplicity in target_text.

    Inspects unmasked ``ctx.source_text`` and ``ctx.target_text``. Defaults to
    ``COMMON_PLACEHOLDER_PATTERNS`` (``[[...]]``, ``{var}``, printf ``%s``) rather than
    inheriting ``ctx.ignore_patterns`` automatically, avoiding false positives on
    translatable markup such as Markdown image alt text or HTML comments. Pass
    ``patterns=...`` or ``use_context_ignore_patterns=True`` to customize.
    """

    name: str = "placeholder_preservation"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        patterns: Sequence[str | re.Pattern[str]] | None = None,
        use_context_ignore_patterns: bool = False,
        check_order: bool = False,
        missing_severity: AuditSeverity = "error",
        unexpected_severity: AuditSeverity = "error",
        duplicated_severity: AuditSeverity = "error",
        order_severity: AuditSeverity = "warning",
        snippet_radius: int = 30,
    ) -> None:
        self._explicit_patterns: tuple[re.Pattern[str], ...] | None = (
            compile_ignore_patterns(patterns) if patterns is not None else None
        )
        self.use_context_ignore_patterns = use_context_ignore_patterns
        self.check_order = check_order
        self.missing_severity: AuditSeverity = missing_severity
        self.unexpected_severity: AuditSeverity = unexpected_severity
        self.duplicated_severity: AuditSeverity = duplicated_severity
        self.order_severity: AuditSeverity = order_severity
        self.snippet_radius = snippet_radius

    def _resolve_patterns(self, ctx: AuditContext) -> tuple[re.Pattern[str], ...]:
        if self._explicit_patterns is not None:
            return self._explicit_patterns
        if self.use_context_ignore_patterns:
            return ctx.ignore_patterns
        return COMMON_PLACEHOLDER_PATTERNS

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        if not ctx.source_text.strip() or not ctx.target_text.strip():
            return []

        patterns = self._resolve_patterns(ctx)
        if not patterns:
            return []

        src_hits = _extract_placeholder_hits(ctx.source_text, patterns)
        tgt_hits = _extract_placeholder_hits(ctx.target_text, patterns)
        if not src_hits and not tgt_hits:
            return []

        src_counts: Counter[str] = Counter(tok for tok, _, _, _ in src_hits)
        tgt_counts: Counter[str] = Counter(tok for tok, _, _, _ in tgt_hits)

        src_pat_by_tok: dict[str, int] = {}
        for tok, pat_idx, _, _ in src_hits:
            src_pat_by_tok.setdefault(tok, pat_idx)

        tgt_pat_by_tok: dict[str, int] = {}
        tgt_first_span: dict[str, tuple[int, int]] = {}
        for tok, pat_idx, start, end in tgt_hits:
            tgt_pat_by_tok.setdefault(tok, pat_idx)
            tgt_first_span.setdefault(tok, (start, end))

        missing: Counter[str] = src_counts - tgt_counts
        extra: Counter[str] = tgt_counts - src_counts

        issues: list[AuditIssue] = []

        # Pair missing source placeholders with novel extra target placeholders from the same pattern family
        # to emit a single actionable 'placeholder_mutated' finding (e.g. [[IMG_1]] -> [[SLIKA_1]]).
        novel_extras = [tok for tok in extra if src_counts[tok] == 0]
        candidate_pairs: list[tuple[float, str, str]] = []
        for s_tok in missing:
            s_pat = src_pat_by_tok[s_tok]
            for t_tok in novel_extras:
                if tgt_pat_by_tok[t_tok] == s_pat:
                    candidate_pairs.append((_mutation_pair_score(s_tok, t_tok), s_tok, t_tok))
        candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        for _, s_tok, t_tok in candidate_pairs:
            paired = min(missing[s_tok], extra[t_tok])
            if paired <= 0:
                continue
            missing[s_tok] -= paired
            extra[t_tok] -= paired
            t_start, t_end = tgt_first_span[t_tok]
            issues.append(
                AuditIssue(
                    code="placeholder_mutated",
                    rule_name=self.name,
                    severity=self.missing_severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=f"placeholder:{s_tok}",
                    message=(
                        f"Placeholder '{s_tok}' appears to have been altered or translated to "
                        f"'{t_tok}' ({paired} occurrence(s))."
                    ),
                    evidence={
                        "token": s_tok,
                        "suspected_replacement": t_tok,
                        "expected_count": src_counts[s_tok],
                        "actual_count": tgt_counts[s_tok],
                        "mutated_count": paired,
                        "snippet": _extract_snippet(
                            ctx.target_text, t_start, t_end, self.snippet_radius
                        ),
                    },
                    suggestion=f"Restore exact placeholder '{s_tok}' in place of '{t_tok}'.",
                )
            )

        for tok, count in missing.items():
            if count <= 0:
                continue
            issues.append(
                AuditIssue(
                    code="placeholder_missing",
                    rule_name=self.name,
                    severity=self.missing_severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key=f"placeholder:{tok}",
                    message=(
                        f"Placeholder '{tok}' is missing in target_text "
                        f"(expected {src_counts[tok]}, found {tgt_counts[tok]})."
                    ),
                    evidence={
                        "token": tok,
                        "expected_count": src_counts[tok],
                        "actual_count": tgt_counts[tok],
                        "missing_count": count,
                    },
                    suggestion=(
                        f"Restore missing placeholder '{tok}' "
                        f"(expected {src_counts[tok]} occurrence(s), found {tgt_counts[tok]})."
                    ),
                )
            )

        for tok, count in extra.items():
            if count <= 0:
                continue
            t_start, t_end = tgt_first_span[tok]
            snippet = _extract_snippet(ctx.target_text, t_start, t_end, self.snippet_radius)
            if src_counts[tok] > 0:
                issues.append(
                    AuditIssue(
                        code="placeholder_duplicated",
                        rule_name=self.name,
                        severity=self.duplicated_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key=f"placeholder:{tok}",
                        message=(
                            f"Placeholder '{tok}' is duplicated in target_text "
                            f"(expected {src_counts[tok]}, found {tgt_counts[tok]})."
                        ),
                        evidence={
                            "token": tok,
                            "expected_count": src_counts[tok],
                            "actual_count": tgt_counts[tok],
                            "extra_count": count,
                            "snippet": snippet,
                        },
                        suggestion=(
                            f"Remove extra duplicate of placeholder '{tok}' "
                            f"(expected {src_counts[tok]} occurrence(s), found {tgt_counts[tok]})."
                        ),
                    )
                )
            else:
                issues.append(
                    AuditIssue(
                        code="placeholder_unexpected",
                        rule_name=self.name,
                        severity=self.unexpected_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key=f"placeholder:{tok}",
                        message=(
                            f"Unexpected placeholder '{tok}' in target_text "
                            f"({tgt_counts[tok]} occurrence(s)) not present in source_text."
                        ),
                        evidence={
                            "token": tok,
                            "expected_count": 0,
                            "actual_count": tgt_counts[tok],
                            "snippet": snippet,
                        },
                        suggestion=f"Remove unexpected placeholder '{tok}' not present in source_text.",
                    )
                )

        if self.check_order and not issues:
            src_seq = [tok for tok, _, _, _ in src_hits]
            tgt_seq = [tok for tok, _, _, _ in tgt_hits]
            if src_seq != tgt_seq:
                issues.append(
                    AuditIssue(
                        code="placeholder_order_mismatch",
                        rule_name=self.name,
                        severity=self.order_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key="placeholder:order",
                        message=(
                            f"Placeholder order changed in target_text: "
                            f"expected {src_seq}, found {tgt_seq}."
                        ),
                        evidence={
                            "expected_order": src_seq,
                            "actual_order": tgt_seq,
                        },
                        suggestion=f"Keep positional placeholders in source order: {src_seq}.",
                    )
                )

        return issues


class UntranslatedSourceRule:
    """Opt-in heuristic rule flagging verbatim or near-verbatim source_text echoes in target_text.

    Complements ``ScriptConsistencyRule`` for same-script language pairs (e.g., EN -> SR-Latn/DE/FR).
    Operates on masked text and optionally exempts locked entity names that are intentionally
    identical between source and target. Intended as a coarse guardrail; disable or raise
    thresholds when source-language quotes/lyrics are expected verbatim.
    """

    name: str = "untranslated_source"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        severity: AuditSeverity = "error",
        paragraph_severity: AuditSeverity = "error",
        similarity_threshold: float = 92.0,
        min_chars: int = 25,
        min_words: int = 5,
        check_paragraphs: bool = True,
        min_paragraph_words: int = 6,
        exempt_entity_names: bool = True,
        snippet_radius: int = 40,
    ) -> None:
        self.severity: AuditSeverity = severity
        self.paragraph_severity: AuditSeverity = paragraph_severity
        self.similarity_threshold = similarity_threshold
        self.min_chars = min_chars
        self.min_words = min_words
        self.check_paragraphs = check_paragraphs
        self.min_paragraph_words = min_paragraph_words
        self.exempt_entity_names = exempt_entity_names
        self.snippet_radius = snippet_radius

    def _exempt_phrases(self, ctx: AuditContext) -> list[str]:
        if not self.exempt_entity_names or not ctx.entities:
            return []
        phrases: set[str] = set()
        for entity in ctx.entities:
            canonical = entity.canonical_name.strip()
            if not canonical:
                continue
            raw_target = entity.base_attributes.get("target_name")
            target_name = raw_target.strip() if isinstance(raw_target, str) else ""
            if entity.base_attributes.get("allow_source_name") or (
                target_name and canonical.lower() == target_name.lower()
            ):
                phrases.add(canonical)
        return sorted(phrases, key=len, reverse=True)

    @staticmethod
    def _mask_phrases(text: str, phrases: Sequence[str]) -> str:
        masked = text
        for phrase in phrases:
            masked = _mask_spans(masked, re.compile(_phrase_pattern(phrase), re.IGNORECASE))
        return masked

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        masked_src = ctx.masked_source_text
        masked_tgt = ctx.masked_target_text
        if not masked_src.strip() or not masked_tgt.strip():
            return []

        exempt = self._exempt_phrases(ctx)
        if exempt:
            masked_src = self._mask_phrases(masked_src, exempt)
            masked_tgt = self._mask_phrases(masked_tgt, exempt)

        norm_src = _normalize_for_echo(masked_src)
        norm_tgt = _normalize_for_echo(masked_tgt)
        if not norm_src or not norm_tgt:
            return []

        if len(norm_src) >= self.min_chars and _letter_word_count(norm_src) >= self.min_words:
            sim = float(fuzz.ratio(norm_src, norm_tgt))
            if sim >= self.similarity_threshold:
                snippet_src = ctx.target_snippet_text
                return [
                    AuditIssue(
                        code="untranslated_source_echo",
                        rule_name=self.name,
                        severity=self.severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key="structural:untranslated_source",
                        message=(
                            f"Target text appears to echo source_text untranslated "
                            f"(similarity {sim:.1f}% >= {self.similarity_threshold:.1f}%)."
                        ),
                        evidence={
                            "similarity": round(sim, 2),
                            "threshold": self.similarity_threshold,
                            "source_chars": len(norm_src),
                            "snippet": _extract_snippet(
                                snippet_src, 0, min(len(snippet_src), 80), self.snippet_radius
                            ),
                        },
                        suggestion=(
                            "Translate the source text into the target language instead of "
                            "echoing the original text verbatim."
                        ),
                    )
                ]

        if not self.check_paragraphs:
            return []

        src_paras = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(masked_src) if p.strip()]
        if len(src_paras) < 2:
            return []

        tgt_para_spans: list[tuple[str, int, int]] = []
        cursor = 0
        for match in _PARAGRAPH_SPLIT_RE.finditer(masked_tgt):
            seg = masked_tgt[cursor : match.start()]
            if seg.strip():
                tgt_para_spans.append((seg, cursor, match.start()))
            cursor = match.end()
        tail = masked_tgt[cursor:]
        if tail.strip():
            tgt_para_spans.append((tail, cursor, len(masked_tgt)))

        if not tgt_para_spans:
            return []

        norm_tgt_paras = [(_normalize_for_echo(seg), start, end) for seg, start, end in tgt_para_spans]
        snippet_text = ctx.target_snippet_text
        issues: list[AuditIssue] = []
        flagged_tgt_indices: set[int] = set()

        for sp_idx, sp in enumerate(src_paras):
            norm_sp = _normalize_for_echo(sp)
            if len(norm_sp) < self.min_chars or _letter_word_count(norm_sp) < self.min_paragraph_words:
                continue
            for tp_idx, (norm_tp, t_start, t_end) in enumerate(norm_tgt_paras):
                if tp_idx in flagged_tgt_indices or not norm_tp:
                    continue
                sim = float(fuzz.ratio(norm_sp, norm_tp))
                is_contained = len(norm_sp) >= self.min_chars and norm_sp in norm_tp
                if sim >= self.similarity_threshold or is_contained:
                    flagged_tgt_indices.add(tp_idx)
                    eff_sim = 100.0 if is_contained and sim < 100.0 else sim
                    issues.append(
                        AuditIssue(
                            code="untranslated_paragraph_echo",
                            rule_name=self.name,
                            severity=self.paragraph_severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key=f"structural:untranslated_paragraph:{sp_idx}",
                            message=(
                                f"Paragraph {sp_idx + 1} of source_text appears untranslated "
                                f"in target_text (similarity {eff_sim:.1f}%)."
                            ),
                            evidence={
                                "source_paragraph_index": sp_idx,
                                "target_paragraph_index": tp_idx,
                                "similarity": round(eff_sim, 2),
                                "snippet": _extract_snippet(
                                    snippet_text,
                                    t_start,
                                    min(t_end, t_start + 80),
                                    self.snippet_radius,
                                ),
                            },
                            suggestion=(
                                "Translate the untranslated source paragraph into the target "
                                "language rather than leaving it in the source language."
                            ),
                        )
                    )
                    break

        return issues


class OmissionGuardRule:
    """Opt-in coarse structural guardrail catching empty translations, severe truncation, or structure collapse.

    Operates on masked/stripped text so markup and placeholders do not skew ratios.
    Softer structural signals (bloat, paragraph/sentence collapse) default to ``"warning"``
    so automated repair loops do not thrash on natural literary restructuring. Tune
    ``min_length_ratio`` / ``max_length_ratio`` for cross-script pairs (e.g., EN <-> CJK).
    """

    name: str = "omission_guard"
    required_tiers: frozenset[int] = frozenset({1})

    def __init__(
        self,
        *,
        min_length_ratio: float | None = 0.45,
        max_length_ratio: float | None = 2.5,
        min_source_chars: int = 80,
        min_paragraph_ratio: float | None = 0.5,
        min_source_paragraphs: int = 3,
        min_sentence_ratio: float | None = 0.45,
        min_source_sentences: int = 4,
        empty_severity: AuditSeverity = "error",
        truncation_severity: AuditSeverity = "error",
        bloat_severity: AuditSeverity = "warning",
        structure_severity: AuditSeverity = "warning",
    ) -> None:
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.min_source_chars = min_source_chars
        self.min_paragraph_ratio = min_paragraph_ratio
        self.min_source_paragraphs = min_source_paragraphs
        self.min_sentence_ratio = min_sentence_ratio
        self.min_source_sentences = min_source_sentences
        self.empty_severity: AuditSeverity = empty_severity
        self.truncation_severity: AuditSeverity = truncation_severity
        self.bloat_severity: AuditSeverity = bloat_severity
        self.structure_severity: AuditSeverity = structure_severity

    def check(self, ctx: AuditContext) -> list[AuditIssue]:
        src_clean = strip_ignored_spans(
            ctx.source_text, ctx.ignore_patterns, normalizer=ctx.text_normalizer
        ).strip()
        if not src_clean:
            return []

        tgt_clean = strip_ignored_spans(
            ctx.target_text, ctx.ignore_patterns, normalizer=ctx.text_normalizer
        ).strip()
        if not tgt_clean:
            return [
                AuditIssue(
                    code="empty_translation",
                    rule_name=self.name,
                    severity=self.empty_severity,
                    step=ctx.step,
                    timeline_id=ctx.timeline_id,
                    subject_key="structural:empty",
                    message=(
                        f"Target text is empty while source_text contains {len(src_clean)} character(s)."
                    ),
                    evidence={
                        "source_chars": len(src_clean),
                        "target_chars": 0,
                    },
                    suggestion="Provide a complete translation of the source text.",
                )
            ]

        issues: list[AuditIssue] = []

        src_len = len(src_clean)
        tgt_len = len(tgt_clean)
        length_ratio = tgt_len / src_len
        if src_len >= self.min_source_chars:
            if self.min_length_ratio is not None and length_ratio < self.min_length_ratio:
                issues.append(
                    AuditIssue(
                        code="translation_truncated",
                        rule_name=self.name,
                        severity=self.truncation_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key="structural:length_low",
                        message=(
                            f"Target text is severely shorter than source_text "
                            f"({tgt_len} vs {src_len} chars, ratio {length_ratio:.2f} < "
                            f"{self.min_length_ratio:.2f}); possible omission or truncation."
                        ),
                        evidence={
                            "source_chars": src_len,
                            "target_chars": tgt_len,
                            "length_ratio": round(length_ratio, 3),
                            "min_length_ratio": self.min_length_ratio,
                        },
                        suggestion=(
                            "Check for omitted sentences or truncated ending and translate "
                            "all content from the source text."
                        ),
                    )
                )
            elif self.max_length_ratio is not None and length_ratio > self.max_length_ratio:
                issues.append(
                    AuditIssue(
                        code="translation_bloated",
                        rule_name=self.name,
                        severity=self.bloat_severity,
                        step=ctx.step,
                        timeline_id=ctx.timeline_id,
                        subject_key="structural:length_high",
                        message=(
                            f"Target text is unusually longer than source_text "
                            f"({tgt_len} vs {src_len} chars, ratio {length_ratio:.2f} > "
                            f"{self.max_length_ratio:.2f}); possible repetition or hallucination."
                        ),
                        evidence={
                            "source_chars": src_len,
                            "target_chars": tgt_len,
                            "length_ratio": round(length_ratio, 3),
                            "max_length_ratio": self.max_length_ratio,
                        },
                        suggestion=(
                            "Remove duplicated or hallucinated text not present in the source text."
                        ),
                    )
                )

        if self.min_paragraph_ratio is not None:
            src_paras = [p for p in _PARAGRAPH_SPLIT_RE.split(src_clean) if p.strip()]
            tgt_paras = [p for p in _PARAGRAPH_SPLIT_RE.split(tgt_clean) if p.strip()]
            if len(src_paras) >= self.min_source_paragraphs:
                para_ratio = len(tgt_paras) / len(src_paras)
                if para_ratio < self.min_paragraph_ratio:
                    issues.append(
                        AuditIssue(
                            code="paragraph_collapse",
                            rule_name=self.name,
                            severity=self.structure_severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key="structural:paragraphs",
                            message=(
                                f"Paragraph count collapsed from {len(src_paras)} in source_text "
                                f"to {len(tgt_paras)} in target_text (ratio {para_ratio:.2f} < "
                                f"{self.min_paragraph_ratio:.2f})."
                            ),
                            evidence={
                                "source_paragraphs": len(src_paras),
                                "target_paragraphs": len(tgt_paras),
                                "paragraph_ratio": round(para_ratio, 3),
                                "min_paragraph_ratio": self.min_paragraph_ratio,
                            },
                            suggestion=(
                                "Preserve the paragraph structure of the source text and verify "
                                "no paragraphs were omitted."
                            ),
                        )
                    )

        if self.min_sentence_ratio is not None:
            src_sents = _count_sentences(src_clean)
            tgt_sents = _count_sentences(tgt_clean)
            if src_sents >= self.min_source_sentences:
                sent_ratio = tgt_sents / src_sents
                if sent_ratio < self.min_sentence_ratio:
                    issues.append(
                        AuditIssue(
                            code="sentence_collapse",
                            rule_name=self.name,
                            severity=self.structure_severity,
                            step=ctx.step,
                            timeline_id=ctx.timeline_id,
                            subject_key="structural:sentences",
                            message=(
                                f"Sentence count collapsed from {src_sents} in source_text "
                                f"to {tgt_sents} in target_text (ratio {sent_ratio:.2f} < "
                                f"{self.min_sentence_ratio:.2f})."
                            ),
                            evidence={
                                "source_sentences": src_sents,
                                "target_sentences": tgt_sents,
                                "sentence_ratio": round(sent_ratio, 3),
                                "min_sentence_ratio": self.min_sentence_ratio,
                            },
                            suggestion=(
                                "Verify that no sentences were omitted or overly condensed "
                                "compared to the source text."
                            ),
                        )
                    )

        return issues
