from __future__ import annotations

import re
from typing import Literal, Protocol, Sequence

from rapidfuzz import fuzz, process

from .embedders import _CJK_CHAR_RE
from .models import AliasRecord, EntityRecord, ResolvedMention


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
    """Word-boundary lexical + alias resolver with optional rapidfuzz inflection fallback."""

    def __init__(self, fuzzy_threshold: float | None = 88.0) -> None:
        self.fuzzy_threshold = fuzzy_threshold

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

    def resolve(
        self,
        text: str,
        explicit_mentions: Sequence[str],
        entities: Sequence[EntityRecord],
        aliases: Sequence[AliasRecord],
    ) -> list[ResolvedMention]:
        resolved: list[ResolvedMention] = []
        seen_pairs: set[tuple[str, str]] = set()  # (entity_id, surface_lower)

        name_to_id = {e.canonical_name.lower(): e.entity_id for e in entities if e.canonical_name}
        alias_map: dict[str, list[AliasRecord]] = {}
        for a in aliases:
            if a.alias:
                alias_map.setdefault(a.alias.lower(), []).append(a)

        def _add(
            surface: str,
            entity_id: str,
            via: Literal["canonical", "alias", "fuzzy"],
            conf: float = 1.0,
            *,
            matched_source: Literal["canonical", "alias"] | None = None,
            matched_phrase: str | None = None,
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
                    )

        # 2. Word-boundary text scan (longest phrases first to avoid partial shadowing)
        if text:
            text_lower = text.lower()
            candidates: list[tuple[str, str, Literal["canonical", "alias"]]] = []
            for name, ent_id in name_to_id.items():
                candidates.append((name, ent_id, "canonical"))
            for alias_str, alias_recs in alias_map.items():
                for alias_rec in alias_recs:
                    candidates.append((alias_str, alias_rec.entity_id, "alias"))
            candidates.sort(key=lambda item: len(item[0]), reverse=True)

            matched_spans: dict[tuple[int, int], str] = {}
            for phrase, ent_id, via in candidates:
                pattern = self._phrase_pattern(phrase)
                for match in re.finditer(pattern, text_lower):
                    start, end = match.start(), match.end()
                    # Allow identical spans (shared aliases across entities), block partial/nested overlaps
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
                    )
                    break

            # 3. Optional fuzzy fallback for inflected languages / typos
            if self.fuzzy_threshold is not None:
                matched_entity_ids = {r.entity_id for r in resolved}
                token_spans = [
                    (m.group(0), m.start(), m.end()) for m in re.finditer(r"\w+", text_lower)
                ]
                n_tokens = len(token_spans)
                # Lazily cache deduplicated non-overlapping windows and trigram index per word count
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

                for phrase, ent_id, candidate_via in candidates:
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
                        sub_wins = [
                            w for w in wins if abs(phrase_len - len(w)) <= max_len_diff
                        ]
                    if not sub_wins:
                        continue
                    match = process.extractOne(
                        phrase,
                        sub_wins,
                        scorer=fuzz.WRatio,
                        score_cutoff=self.fuzzy_threshold,
                    )
                    if match is not None:
                        best_window, score, _ = match
                        _add(
                            best_window,
                            ent_id,
                            "fuzzy",
                            conf=round(score / 100.0, 3),
                            matched_source=candidate_via,
                            matched_phrase=phrase,
                        )
                        matched_entity_ids.add(ent_id)

        return resolved
