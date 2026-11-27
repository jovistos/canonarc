from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from rapidfuzz import fuzz, process

from .embedders import _CJK_CHAR_RE
from .models import AliasRecord, EntityRecord, ResolvedMention
from .text_masking import TextNormalizer, apply_text_masking, compile_ignore_patterns

_VOWELS: frozenset[str] = frozenset("aeiouáéíóúàèìòùäëïöüâêîôûāēīōū")

# Conservative inflectional & possessive suffix inventory for 2-char stems (e.g. Ana -> an + i/e/om/in/ina)
_TWO_CHAR_STEM_SUFFIXES: frozenset[str] = frozenset(
    {
        "a",
        "e",
        "i",
        "o",
        "u",
        "om",
        "em",
        "oj",
        "ej",
        "in",
        "ina",
        "ino",
        "ine",
        "inu",
        "ov",
        "ova",
        "ovo",
        "ove",
        "ovu",
        "ev",
        "eva",
        "evo",
        "eve",
        "evu",
    }
)

# Inflectional & possessive suffix inventory for 3-6 char stems (Slavic, Germanic, Romance, Uralic)
_SHORT_INFLECTION_SUFFIXES: frozenset[str] = _TWO_CHAR_STEM_SUFFIXES | frozenset(
    {
        "y",
        "s",
        "n",
        "im",
        "am",
        "um",
        "aj",
        "ih",
        "ah",
        "uh",
        "an",
        "en",
        "oy",
        "ey",
        "ou",
        "au",
        "iu",
        "ia",
        "ie",
        "io",
        "owi",
        "iem",
        "ami",
        "ach",
        "ego",
        "emu",
        "ych",
        "ymi",
    }
)


class MentionResolver(Protocol):
    """Protocol for pluggable entity mention & coreference resolution."""

    def resolve(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
    ) -> list[ResolvedMention]: ...


class DefaultLexicalResolver:
    """Word-boundary lexical + alias resolver with bilingual target_name indexing,
    prefix/stem-aware short-name inflection fallback, and rapidfuzz inflection fallback."""

    def __init__(
        self,
        fuzzy_threshold: float | None = 88.0,
        *,
        short_name_inflection: bool = True,
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        text_normalizer: TextNormalizer | None = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.short_name_inflection = short_name_inflection
        self.ignore_patterns: tuple[re.Pattern[str], ...] = compile_ignore_patterns(ignore_patterns)
        self.text_normalizer: TextNormalizer | None = text_normalizer

    @staticmethod
    def _phrase_pattern(phrase_lower: str) -> str:
        prefix = "" if _CJK_CHAR_RE.match(phrase_lower[0]) else r"(?<!\w)"
        suffix = "" if _CJK_CHAR_RE.match(phrase_lower[-1]) else r"(?!\w)"
        return rf"{prefix}{re.escape(phrase_lower)}{suffix}"

    @classmethod
    def _find_whole_phrase(cls, text_lower: str, phrase_lower: str) -> int:
        if not phrase_lower:
            return -1
        match = re.search(cls._phrase_pattern(phrase_lower), text_lower)
        return match.start() if match else -1

    @staticmethod
    def _short_name_stems(phrase_lower: str) -> list[tuple[str, frozenset[str]]]:
        """Builds candidate (stem, allowed_suffixes) pairs for a 3-6 char single-word name."""
        n = len(phrase_lower)
        if n < 3 or n > 6 or " " in phrase_lower or _CJK_CHAR_RE.match(phrase_lower[0]):
            return []
        stems: list[tuple[str, frozenset[str]]] = []
        if phrase_lower[-1] in _VOWELS:
            stem = phrase_lower[:-1]
            suffix_set = _TWO_CHAR_STEM_SUFFIXES if len(stem) <= 2 else _SHORT_INFLECTION_SUFFIXES
            stems.append((stem, suffix_set))
        else:
            stems.append((phrase_lower, _SHORT_INFLECTION_SUFFIXES))
            # Slavic fleeting vowel in penultimate position (e.g. Petar -> petr-, Mitar -> mitr-)
            if n >= 5 and phrase_lower[-2] in _VOWELS and phrase_lower[-3] not in _VOWELS:
                fleeting_stem = phrase_lower[:-2] + phrase_lower[-1]
                if len(fleeting_stem) >= 4:
                    stems.append((fleeting_stem, _SHORT_INFLECTION_SUFFIXES))
        return stems

    @classmethod
    def _match_short_inflection(
        cls,
        phrase_lower: str,
        raw_token_spans: Sequence[tuple[str, str, int, int]],
        matched_spans: dict[tuple[int, int], str],
    ) -> tuple[str, int, int] | None:
        """Matches a capitalized short inflected proper noun token against a 3-6 char name stem."""
        stems = cls._short_name_stems(phrase_lower)
        if not stems:
            return None
        for raw_tok, lower_tok, start, end in raw_token_spans:
            if not raw_tok or not raw_tok[0].isupper():
                continue
            if lower_tok == phrase_lower:
                continue
            if any(start < e and end > s for s, e in matched_spans):
                continue
            for stem, allowed_suffixes in stems:
                if lower_tok.startswith(stem) and len(lower_tok) > len(stem):
                    suffix = lower_tok[len(stem) :]
                    if suffix in allowed_suffixes:
                        return lower_tok, start, end
        return None

    @staticmethod
    def _build_lookup_tables(
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
        lang_norm: str | None,
    ) -> tuple[dict[str, str], dict[str, list[AliasRecord]]]:
        name_to_id: dict[str, str] = {}
        for e in entities:
            if e.canonical_name:
                name_to_id[e.canonical_name.lower()] = e.entity_id
            target_name = e.base_attributes.get("target_name")
            if isinstance(target_name, str) and target_name.strip():
                ent_target_lang = e.base_attributes.get("target_lang") or e.base_attributes.get("lang")
                ent_target_lang_norm = (
                    ent_target_lang.strip().lower()
                    if isinstance(ent_target_lang, str) and ent_target_lang.strip()
                    else None
                )
                if (
                    lang_norm is None
                    or ent_target_lang_norm is None
                    or ent_target_lang_norm == lang_norm
                ):
                    name_to_id.setdefault(target_name.strip().lower(), e.entity_id)

        alias_map: dict[str, list[AliasRecord]] = {}
        for a in aliases:
            if not a.alias:
                continue
            if lang_norm is not None and a.lang is not None and a.lang.strip().lower() != lang_norm:
                continue
            alias_map.setdefault(a.alias.lower(), []).append(a)
        return name_to_id, alias_map

    @staticmethod
    def _build_phrase_candidates(
        name_to_id: dict[str, str],
        alias_map: dict[str, list[AliasRecord]],
    ) -> list[
        tuple[
            str,
            str,
            Literal["canonical", "alias"],
            Literal["permanent", "epithet", "disguise", "title", "inflection"] | None,
            str | None,
        ]
    ]:
        candidates: list[
            tuple[
                str,
                str,
                Literal["canonical", "alias"],
                Literal["permanent", "epithet", "disguise", "title", "inflection"] | None,
                str | None,
            ]
        ] = []
        for name, ent_id in name_to_id.items():
            candidates.append((name, ent_id, "canonical", None, None))
        for alias_str, alias_recs in alias_map.items():
            for alias_rec in alias_recs:
                candidates.append(
                    (alias_str, alias_rec.entity_id, "alias", alias_rec.kind, alias_rec.lang)
                )
        candidates.sort(key=lambda item: len(item[0]), reverse=True)
        return candidates

    def _fuzzy_match_candidates(
        self,
        candidates: Sequence[
            tuple[
                str,
                str,
                Literal["canonical", "alias"],
                Literal["permanent", "epithet", "disguise", "title", "inflection"] | None,
                str | None,
            ]
        ],
        raw_token_spans: Sequence[tuple[str, str, int, int]],
        matched_spans: dict[tuple[int, int], str],
        matched_entity_ids: set[str],
        add_fn: Any,
    ) -> None:
        assert self.fuzzy_threshold is not None
        token_spans = [(lower_tok, s, e) for _, lower_tok, s, e in raw_token_spans]
        n_tokens = len(token_spans)
        windows_cache: dict[int, tuple[list[str], dict[str, set[int]]]] = {}

        def _get_windows(n_words: int) -> tuple[list[str], dict[str, set[int]]]:
            cached = windows_cache.get(n_words)
            if cached is not None:
                return cached
            wins: list[str] = []
            seen_wins: set[str] = set()
            tri_index: dict[str, set[int]] = {}
            for idx in range(n_tokens - n_words + 1):
                w_start = token_spans[idx][1]
                w_end = token_spans[idx + n_words - 1][2]
                if any(w_start < e and w_end > s for s, e in matched_spans):
                    continue
                window = " ".join(token_spans[j][0] for j in range(idx, idx + n_words))
                if window in seen_wins:
                    continue
                seen_wins.add(window)
                wi = len(wins)
                wins.append(window)
                for k in range(len(window) - 2):
                    tri_index.setdefault(window[k : k + 3], set()).add(wi)
            cached = (wins, tri_index)
            windows_cache[n_words] = cached
            return cached

        for phrase, ent_id, candidate_via, a_kind, a_lang in candidates:
            if ent_id in matched_entity_ids or len(phrase) < 4:
                continue
            n_words = len(phrase.split())
            if n_words <= 0 or n_tokens < n_words:
                continue
            wins, tri_index = _get_windows(n_words)
            if not wins:
                continue
            max_len_diff = max(3, int(len(phrase) * 0.4))
            phrase_len = len(phrase)
            if phrase_len >= 6 and self.fuzzy_threshold >= 80.0:
                cand_indices: set[int] = set()
                for k in range(phrase_len - 2):
                    hits = tri_index.get(phrase[k : k + 3])
                    if hits:
                        cand_indices.update(hits)
                if not cand_indices:
                    continue
                sub_wins = [
                    wins[wi]
                    for wi in sorted(cand_indices)
                    if abs(phrase_len - len(wins[wi])) <= max_len_diff
                ]
            else:
                sub_wins = [w for w in wins if abs(phrase_len - len(w)) <= max_len_diff]
            if not sub_wins:
                continue
            fuzzy_match = process.extractOne(
                phrase,
                sub_wins,
                scorer=fuzz.WRatio,
                score_cutoff=self.fuzzy_threshold,
            )
            if fuzzy_match is not None:
                best_window, score, _ = fuzzy_match
                add_fn(
                    best_window,
                    ent_id,
                    "fuzzy",
                    conf=round(score / 100.0, 3),
                    matched_source=candidate_via,
                    matched_phrase=phrase,
                    alias_kind=a_kind,
                    alias_lang=a_lang,
                )
                matched_entity_ids.add(ent_id)

    def resolve(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
        *,
        lang: str | None = None,
    ) -> list[ResolvedMention]:
        resolved: list[ResolvedMention] = []
        seen_pairs: set[tuple[str, str]] = set()  # (entity_id, surface_lower)
        lang_norm = lang.strip().lower() if lang and lang.strip() else None
        name_to_id, alias_map = self._build_lookup_tables(entities, aliases, lang_norm)

        def _add(
            surface: str,
            entity_id: str,
            via: Literal["canonical", "alias", "fuzzy"],
            conf: float = 1.0,
            *,
            matched_source: Literal["canonical", "alias"] | None = None,
            matched_phrase: str | None = None,
            alias_kind: Literal["permanent", "epithet", "disguise", "title", "inflection"] | None = None,
            alias_lang: str | None = None,
        ) -> None:
            key = (entity_id, surface.strip().lower())
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            resolved.append(
                ResolvedMention(
                    surface_text=surface,
                    entity_id=entity_id,
                    matched_via=via,
                    confidence=conf,
                    matched_source=matched_source,
                    matched_phrase=matched_phrase,
                    alias_kind=alias_kind,
                    lang=alias_lang,
                )
            )

        # 1. Explicit mentions
        for m in explicit_mentions:
            m_clean = m.strip().lower()
            if not m_clean:
                continue
            if m_clean in name_to_id:
                _add(
                    m,
                    name_to_id[m_clean],
                    "canonical",
                    matched_source="canonical",
                    matched_phrase=m_clean,
                )
            elif m_clean in alias_map:
                for alias_rec in alias_map[m_clean]:
                    _add(
                        m,
                        alias_rec.entity_id,
                        "alias",
                        matched_source="alias",
                        matched_phrase=m_clean,
                        alias_kind=alias_rec.kind,
                        alias_lang=alias_rec.lang,
                    )

        if not text:
            return resolved
        if self.ignore_patterns or self.text_normalizer is not None:
            text = apply_text_masking(
                text, self.ignore_patterns, normalizer=self.text_normalizer
            )
        if not text:
            return resolved

        # 2. Word-boundary text scan (longest phrases first to avoid partial shadowing)
        text_lower = text.lower()
        candidates = self._build_phrase_candidates(name_to_id, alias_map)
        matched_spans: dict[tuple[int, int], str] = {}
        for phrase, ent_id, via, a_kind, a_lang in candidates:
            pattern = self._phrase_pattern(phrase)
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                if any(
                    start < e and end > s and (s, e) != (start, end)
                    for s, e in matched_spans
                ):
                    continue
                matched_spans[(start, end)] = phrase
                _add(
                    phrase,
                    ent_id,
                    via,
                    matched_source=via,
                    matched_phrase=phrase,
                    alias_kind=a_kind,
                    alias_lang=a_lang,
                )
                break

        if self.fuzzy_threshold is None:
            return resolved

        raw_token_spans = [
            (m.group(0), m.group(0).lower(), m.start(), m.end())
            for m in re.finditer(r"\w+", text)
        ]
        matched_entity_ids = {r.entity_id for r in resolved}

        # 3. Prefix/stem-aware short-name inflection fallback (3-6 char proper names)
        if self.short_name_inflection:
            for phrase, ent_id, candidate_via, a_kind, a_lang in candidates:
                if ent_id in matched_entity_ids or not (3 <= len(phrase) <= 6):
                    continue
                hit = self._match_short_inflection(phrase, raw_token_spans, matched_spans)
                if hit is not None:
                    matched_tok, start, end = hit
                    matched_spans[(start, end)] = phrase
                    _add(
                        matched_tok,
                        ent_id,
                        "fuzzy",
                        conf=0.9,
                        matched_source=candidate_via,
                        matched_phrase=phrase,
                        alias_kind=a_kind,
                        alias_lang=a_lang,
                    )
                    matched_entity_ids.add(ent_id)

        # 4. Optional rapidfuzz WRatio fallback for longer inflected names / typos
        self._fuzzy_match_candidates(
            candidates, raw_token_spans, matched_spans, matched_entity_ids, _add
        )
        return resolved
