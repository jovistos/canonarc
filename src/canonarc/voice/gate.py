from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

from ..embedders import _tokenize_units
from ..models import DialogueCandidate, VoiceExemplar
from ..text_masking import TextNormalizer, compile_ignore_patterns, strip_ignored_spans

# Quoted dialogue patterns across English, European, Balkan, and CJK typographic conventions
_OUTWARD_GUILLEMET_PATTERN = re.compile(r'«([^»]+)»')
_INWARD_GUILLEMET_PATTERN = re.compile(r'»([^«]+)«')
_DASH_DIALOGUE_PATTERN = re.compile(r'(?:^|\n)\s*[—–―]\s*([^\n]+)')
_DASH_INTERJECTION_SPLIT_RE = re.compile(r'\s+[—–―]\s+')
_DASH_LINE_PREFIX_RE = re.compile(r'^\s*[—–―]\s*')

DIALOGUE_PATTERNS = [
    re.compile(r'"([^"]+)"'),                   # Straight double quotes
    re.compile(r'„([^“”]+)[“”]'),               # Serbian / Croatian / Polish / Hungarian / German
    re.compile(r'“([^”]+)”'),                   # English curved quotes
    _OUTWARD_GUILLEMET_PATTERN,                 # French / Russian guillemets
    re.compile(r'「([^」]+)」'),                  # Japanese / Traditional Chinese
    re.compile(r'『([^』]+)』'),                  # Japanese / Traditional Chinese inner quotes
    _DASH_DIALOGUE_PATTERN,                     # Em-dash / en-dash / quotation-dash dialogue lines
]


def _split_dash_dialogue(raw_line: str) -> tuple[str, str]:
    """Splits a dash-dialogue line into (spoken_dialogue, inline_narrator_beats)."""
    stripped = raw_line.strip()
    parts = [p.strip() for p in _DASH_INTERJECTION_SPLIT_RE.split(stripped)]
    if len(parts) <= 1:
        return stripped, ""

    spoken_parts: list[str] = []
    narrator_parts: list[str] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        if idx % 2 == 1 and not part.endswith(("?", "!")):
            narrator_parts.append(part)
        else:
            spoken_parts.append(part)

    return " ".join(spoken_parts).strip(), " ".join(narrator_parts).strip()


class IngestionGate:
    """Two-stage anti-drift gate to prevent early generation drift and error compounding."""

    def __init__(
        self,
        min_words: int = 12,
        max_words: int = 60,
        min_distinctiveness: float = 0.40,
        min_quality: float = 8.0,
        *,
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        text_normalizer: TextNormalizer | None = None,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_distinctiveness = min_distinctiveness
        self.min_quality = min_quality
        self.ignore_patterns: tuple[re.Pattern[str], ...] = compile_ignore_patterns(ignore_patterns)
        self.text_normalizer: TextNormalizer | None = text_normalizer

    @staticmethod
    def _attribution_windows(
        text: str,
        start: int,
        end: int,
        *,
        prev_text: str = "",
        next_text: str = "",
        is_dash_dialogue: bool = False,
    ) -> tuple[str, str]:
        """Extracts local pre-quote and post-quote attribution clauses."""
        before_raw = text[max(0, start - 140) : start]
        if start < 140 and prev_text:
            prefix = prev_text.rstrip()
            if prefix:
                head = text[:start]
                combined_before = f"{prefix} {head}" if head else f"{prefix} "
                before_raw = combined_before[-140:]
        for sep in ("\n\n", '"', "”", "“", "»", "«", "」", "』"):
            idx = before_raw.rfind(sep)
            if idx != -1:
                before_raw = before_raw[idx + len(sep) :]
        if is_dash_dialogue and before_raw:
            lines = before_raw.splitlines()
            last_dash_idx = -1
            for i, line in enumerate(lines):
                if _DASH_LINE_PREFIX_RE.match(line):
                    last_dash_idx = i
            if last_dash_idx != -1:
                before_raw = "\n".join(lines[last_dash_idx + 1 :])

        after_raw = text[end : min(len(text), end + 100)]
        if len(text) - end < 100 and next_text:
            suffix = next_text.lstrip()
            if suffix:
                tail = text[end:]
                combined_after = f"{tail} {suffix}" if tail else f" {suffix}"
                after_raw = combined_after[:100]
        for sep in ("\n\n", "\n", '"', "”", "“", "„", "«", "»", "「", "『"):
            idx = after_raw.find(sep)
            if idx != -1:
                after_raw = after_raw[:idx]

        return before_raw.strip(), after_raw.strip()

    def extract_candidates(
        self,
        text: str,
        *,
        source_text: str = "",
        prev_text: str = "",
        next_text: str = "",
        speaker_resolver: Callable[[str, str], tuple[str | None, str | None, str | None]] | None = None,
        on_rejected: Callable[[str, str, dict[str, Any]], None] | None = None,
        ignore_patterns: Sequence[str | re.Pattern[str]] | None = None,
        text_normalizer: TextNormalizer | None = None,
    ) -> list[DialogueCandidate]:
        """Extracts quoted dialogue spans and applies Gate 1 (length + distinctiveness pre-filter)."""
        extra_patterns = compile_ignore_patterns(ignore_patterns) if ignore_patterns else ()
        if self.ignore_patterns and extra_patterns:
            seen_pat = set(self.ignore_patterns)
            eff_patterns = self.ignore_patterns + tuple(
                p for p in extra_patterns if p not in seen_pat
            )
        else:
            eff_patterns = extra_patterns or self.ignore_patterns
        eff_normalizer = text_normalizer if text_normalizer is not None else self.text_normalizer

        if eff_patterns or eff_normalizer is not None:
            text = strip_ignored_spans(text, eff_patterns, normalizer=eff_normalizer)
            if source_text:
                source_text = strip_ignored_spans(
                    source_text, eff_patterns, normalizer=eff_normalizer
                )
            if prev_text:
                prev_text = strip_ignored_spans(prev_text, eff_patterns, normalizer=eff_normalizer)
            if next_text:
                next_text = strip_ignored_spans(next_text, eff_patterns, normalizer=eff_normalizer)

        candidates: list[DialogueCandidate] = []
        seen_spans: set[str] = set()
        consumed_quote_delimiters: set[int] = set()

        first_open_g = text.find("«")
        first_close_g = text.find("»")
        use_inward_guillemets = (
            first_open_g != -1 and first_close_g != -1 and first_close_g < first_open_g
        )

        for pattern in DIALOGUE_PATTERNS:
            active_pattern = (
                _INWARD_GUILLEMET_PATTERN
                if (pattern is _OUTWARD_GUILLEMET_PATTERN and use_inward_guillemets)
                else pattern
            )
            is_dash_pattern = active_pattern is _DASH_DIALOGUE_PATTERN

            for match in active_pattern.finditer(text):
                if not is_dash_pattern:
                    if (
                        match.start() in consumed_quote_delimiters
                        or (match.end() - 1) in consumed_quote_delimiters
                    ):
                        continue
                    consumed_quote_delimiters.add(match.start())
                    consumed_quote_delimiters.add(match.end() - 1)

                raw_span = match.group(1).strip()
                if is_dash_pattern:
                    span, inline_attribution = _split_dash_dialogue(raw_span)
                else:
                    span, inline_attribution = raw_span, ""

                if not span or span in seen_spans:
                    continue
                seen_spans.add(span)

                units, is_cjk = _tokenize_units(span)
                unit_count = len(units)
                effective_max = int(self.max_words * 1.8) if is_cjk else self.max_words
                if not (self.min_words <= unit_count <= effective_max):
                    if on_rejected is not None:
                        on_rejected(
                            span,
                            f"word_count {unit_count} outside [{self.min_words}, {effective_max}]",
                            {"unit_count": unit_count, "min_words": self.min_words, "max_words": effective_max},
                        )
                    continue

                unique_units = set(units)
                root_ttr = len(unique_units) / math.sqrt(unit_count)
                lexical_diversity = len(unique_units) / unit_count
                bigrams = set(pairwise(units))
                bigram_diversity = len(bigrams) / max(1, unit_count - 1)

                if is_cjk:
                    distinctiveness = min(1.0, (root_ttr / 5.0) * 0.6 + bigram_diversity * 0.4)
                else:
                    long_words = [w for w in units if len(w) >= 6]
                    long_ratio = len(long_words) / unit_count
                    distinctiveness = min(
                        1.0,
                        (root_ttr / 5.0) * 0.55 + long_ratio * 0.25 + bigram_diversity * 0.20,
                    )

                if distinctiveness < self.min_distinctiveness:
                    if on_rejected is not None:
                        on_rejected(
                            span,
                            f"distinctiveness {distinctiveness:.3f} < {self.min_distinctiveness}",
                            {"distinctiveness": distinctiveness, "threshold": self.min_distinctiveness},
                        )
                    continue

                speaker_id: str | None = None
                addressee_id: str | None = None
                speaker_surface: str | None = None
                if speaker_resolver is not None:
                    before_win, after_win = self._attribution_windows(
                        text,
                        match.start(),
                        match.end(),
                        prev_text=prev_text,
                        next_text=next_text,
                        is_dash_dialogue=is_dash_pattern,
                    )
                    if inline_attribution:
                        after_win = (
                            f"{inline_attribution} {after_win}".strip()
                            if after_win
                            else inline_attribution
                        )
                    speaker_id, addressee_id, speaker_surface = speaker_resolver(
                        before_win, after_win
                    )
                candidates.append(
                    DialogueCandidate(
                        candidate_id=str(uuid.uuid4())[:8],
                        text=span,
                        source_text=source_text,
                        word_count=unit_count,
                        distinctiveness_score=round(distinctiveness, 3),
                        lexical_diversity=round(lexical_diversity, 3),
                        speaker_id=speaker_id,
                        addressee_id=addressee_id,
                        speaker_surface=speaker_surface,
                    )
                )

        return candidates

    def passes_quality_gate(self, exemplar: VoiceExemplar) -> bool:
        """Gate 2 check: post-polish / quality score >= threshold."""
        return exemplar.quality_score >= self.min_quality
