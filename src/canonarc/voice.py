from __future__ import annotations

import inspect
import json
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal, Protocol

import numpy as np

from .embedders import (
    _CJK_CHAR_RE,
    _TOKEN_UNIT_RE,
    Embedder,
    OfflineNgramEmbedder,
    _tokenize_units,
)
from .episodic import (
    DEFAULT_EPISODIC_POLICY,
    DefaultEpisodeRanker,
    EpisodeRanker,
    EpisodicScoringPolicy,
    cosine_similarity,
)
from .extractors import (
    JsonCompletionClient,
    TokenUsage,
    _coerce_usage,
    validate_json_completion_client,
)
from .models import DialogueCandidate, EpisodeRecord, StepContext, VoiceExemplar
from .presets import DEFAULT_SCHEMA_POLICY, SchemaPolicy

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
    """Splits a dash-dialogue line into (spoken_dialogue, inline_narrator_beats).

    Handles European/Balkan dash dialogue such as:
    - 'Ne znam šta da radim — reče Artur.'
    - 'Ne znam — reče Artur — šta da radim.'
    Preserves unspaced dashes (e.g. '10–15') and in-speech rhetorical asides
    ending in '?' or '!'.
    """
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
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.min_distinctiveness = min_distinctiveness
        self.min_quality = min_quality

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
            # Drop any preceding dash-dialogue turn(s) so the previous speaker's
            # line does not bleed into this turn's attribution window, while
            # keeping a normal narrative lead-in line (e.g. "Elena uzdahnu:\n").
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
    ) -> list[DialogueCandidate]:
        """Extracts quoted dialogue spans and applies Gate 1 (length + distinctiveness pre-filter)."""
        candidates: list[DialogueCandidate] = []
        seen_spans: set[str] = set()
        consumed_quote_delimiters: set[int] = set()

        # Detect inward guillemets (»…«, common in Croatian/Slovenian/German)
        # vs outward guillemets («…», French/Russian) to prevent matching
        # narrator text between two outward quotes.
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


class VoiceBank:
    """Tier 5 Gated Voice Bank (with optional backwards-compatible delegation to EpisodeRanker)."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        episode_ranker: EpisodeRanker | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or OfflineNgramEmbedder()
        self.gate: IngestionGate = gate or IngestionGate()
        self.episode_ranker: EpisodeRanker = episode_ranker or DefaultEpisodeRanker(
            embedder=self.embedder
        )

    def _cosine_sim(
        self,
        q_vec: np.ndarray | None,
        q_norm: float,
        emb_bytes: bytes,
        fallback_text: str,
    ) -> float:
        return cosine_similarity(self.embedder, q_vec, q_norm, emb_bytes, fallback_text)

    def rank_exemplars(
        self,
        query_text: str,
        candidates: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        target_register: str = "",
        target_tone: str = "",
        target_addressee: str | None = None,
        target_constraints: dict[str, str] | None = None,
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
        top_k: int = 3,
    ) -> list[VoiceExemplar]:
        """Applies policy-driven hard filters + soft attribute boosts + vector cosine similarity."""
        if not candidates or top_k <= 0:
            return []

        q_vec = self.embedder.embed_text(query_text) if query_text else None
        q_norm = float(np.linalg.norm(q_vec)) if q_vec is not None else 0.0

        constraints: dict[str, str] = {}
        if target_register:
            constraints["register"] = target_register
        if target_tone:
            constraints["tone"] = target_tone
        if target_constraints:
            constraints.update(target_constraints)
        norm_constraints = {
            k.lower(): str(v).strip().lower()
            for k, v in constraints.items()
            if str(v).strip()
        }

        def _score_pass(allow_hard_mismatch: bool) -> list[tuple[float, VoiceExemplar]]:
            scored_items: list[tuple[float, VoiceExemplar]] = []
            for ex, emb_bytes in candidates:
                meta = {k.lower(): str(v).strip().lower() for k, v in ex.metadata.items()}
                penalty = 0.0
                rejected = False
                mismatched_keys: list[str] = []
                for hk in policy.hard_filter_keys:
                    hk_lower = hk.lower()
                    req_val = norm_constraints.get(hk_lower, "")
                    ex_val = meta.get(hk_lower, "")
                    if req_val and ex_val and ex_val != req_val:
                        if not allow_hard_mismatch:
                            rejected = True
                            break
                        penalty += policy.hard_filter_mismatch_penalty
                        mismatched_keys.append(hk)
                if rejected:
                    continue

                sim = self._cosine_sim(q_vec, q_norm, emb_bytes, ex.text) if q_vec is not None else 0.5

                soft_bonus = 0.0
                for sk, weight in policy.soft_boost_weights.items():
                    sk_lower = sk.lower()
                    req_val = norm_constraints.get(sk_lower, "")
                    if req_val and meta.get(sk_lower, "") == req_val:
                        soft_bonus += weight

                addr_bonus = (
                    policy.addressee_boost
                    if (target_addressee and ex.addressee_id == target_addressee)
                    else 0.0
                )
                quality_factor = policy.quality_weight * (ex.quality_score / 10.0)

                total_score = sim + soft_bonus + addr_bonus + quality_factor - penalty
                ex_copy = ex.model_copy()
                ex_copy.match_score = round(total_score, 4)
                if mismatched_keys and policy.hard_filter_fallback_note:
                    note = policy.hard_filter_fallback_note.format(keys=", ".join(mismatched_keys))
                    ex_copy.metadata = {**ex_copy.metadata, "fallback_note": note}
                scored_items.append((total_score, ex_copy))
            return scored_items

        scored_items = _score_pass(allow_hard_mismatch=False)
        if not scored_items and policy.fallback_on_empty_hard_filter:
            scored_items = _score_pass(allow_hard_mismatch=True)

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_items[:top_k]]

    def rank_exemplars_for_targets(
        self,
        query_text: str,
        candidates: Sequence[tuple[VoiceExemplar, bytes]],
        *,
        targets: Sequence[tuple[str | None, dict[str, str]] | tuple[str | None, str, str]],
        policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
        top_k: int = 3,
    ) -> list[VoiceExemplar]:
        """Ranks exemplars across multiple co-present dyadic targets via Round-Robin allocation."""
        if not candidates or top_k <= 0:
            return []
        if not targets:
            return self.rank_exemplars(query_text, candidates, policy=policy, top_k=top_k)

        normalized_targets: list[tuple[str | None, dict[str, str]]] = []
        for t in targets:
            if len(t) == 2:
                normalized_targets.append((t[0], t[1]))
            else:
                addr, reg, tone = t
                c: dict[str, str] = {}
                if reg:
                    c["register"] = reg
                if tone:
                    c["tone"] = tone
                normalized_targets.append((addr, c))

        if len(normalized_targets) == 1:
            addr, c = normalized_targets[0]
            return self.rank_exemplars(
                query_text=query_text,
                candidates=candidates,
                target_constraints=c,
                target_addressee=addr,
                policy=policy,
                top_k=top_k,
            )

        per_target_queues: list[list[VoiceExemplar]] = []
        for addr, c in normalized_targets:
            ranked = self.rank_exemplars(
                query_text=query_text,
                candidates=candidates,
                target_constraints=c,
                target_addressee=addr,
                policy=policy,
                top_k=top_k,
            )
            for ex in ranked:
                if addr and "for_addressee" not in ex.metadata:
                    ex.metadata = {**ex.metadata, "for_addressee": addr}
            if ranked:
                per_target_queues.append(ranked)

        merged: list[VoiceExemplar] = []
        seen_ids: set[str] = set()
        max_len = max((len(q) for q in per_target_queues), default=0)
        for depth in range(max_len):
            for queue in per_target_queues:
                if depth < len(queue):
                    ex = queue[depth]
                    dedup_key = ex.exemplar_id or ex.text
                    if dedup_key not in seen_ids:
                        seen_ids.add(dedup_key)
                        merged.append(ex)
                        if len(merged) >= top_k:
                            return merged

        return merged

    def rank_episodes(
        self,
        *,
        current_step: int,
        query_text: str,
        candidates: Sequence[tuple[EpisodeRecord, bytes]],
        scene_entity_ids: set[str],
        pinned_episode_ids: set[str] | None = None,
        pov_entity_ids: Sequence[str] | None = None,
        pov_mode: Literal["any", "all"] = "any",
        policy: EpisodicScoringPolicy = DEFAULT_EPISODIC_POLICY,
        top_k: int = 5,
    ) -> list[EpisodeRecord]:
        """Delegates to the configured EpisodeRanker (kept here for backwards compatibility)."""
        return self.episode_ranker.rank_episodes(
            current_step=current_step,
            query_text=query_text,
            candidates=candidates,
            scene_entity_ids=scene_entity_ids,
            pinned_episode_ids=pinned_episode_ids,
            pov_entity_ids=pov_entity_ids,
            pov_mode=pov_mode,
            policy=policy,
            top_k=top_k,
        )


_ARTIFACT_RE = re.compile(r"(\[[^\]]*\]|\{[^}]*\}|<[^>]*>|https?://|\b(?:TODO|FIXME|TBD)\b)", re.IGNORECASE)
_EXCESSIVE_PUNCT_RE = re.compile(r"([!?]{3,}|\.{4,})")
_EXPRESSIVE_PUNCT_RE = re.compile(r"[!?—…;:！？]")
_CLAUSE_PUNCT_RE = re.compile(r"[,،、;:]|[.!?。！？]\s+\S")


@dataclass(frozen=True)
class VoiceScoreResult:
    """Structured result returned by a Gate-2 VoiceQualityScorer."""

    score: float
    reason: str = ""
    metadata_updates: dict[str, str] = field(default_factory=dict)


class VoiceQualityScorer(Protocol):
    """Protocol for Gate-2 voice exemplar quality scoring (0.0 to 10.0)."""

    def score_candidate(
        self,
        candidate: DialogueCandidate,
        *,
        metadata: dict[str, str],
        context: StepContext | None = None,
    ) -> float | VoiceScoreResult: ...


def _coerce_voice_score_result(raw: float | int | VoiceScoreResult) -> VoiceScoreResult:
    if isinstance(raw, VoiceScoreResult):
        clamped = round(max(0.0, min(10.0, float(raw.score))), 3)
        clean_updates = {
            str(k).strip(): str(v).strip()
            for k, v in raw.metadata_updates.items()
            if str(k).strip() and str(v).strip()
        }
        return VoiceScoreResult(
            score=clamped,
            reason=str(raw.reason or "").strip(),
            metadata_updates=clean_updates,
        )
    if isinstance(raw, (int, float)):
        clamped = round(max(0.0, min(10.0, float(raw))), 3)
        return VoiceScoreResult(score=clamped)
    raise TypeError(f"VoiceQualityScorer must return float or VoiceScoreResult, got {type(raw).__name__}")


def score_voice_candidates_batch(
    scorer: VoiceQualityScorer,
    candidates: Sequence[DialogueCandidate],
    metadatas: Sequence[dict[str, str]],
    *,
    context: StepContext | None = None,
) -> list[VoiceScoreResult]:
    """Dispatches to `scorer.score_candidates_batch()` when available, with per-item fallback."""
    if not candidates:
        return []
    if len(candidates) != len(metadatas):
        raise ValueError("candidates and metadatas must have the same length.")
    batch_fn = getattr(scorer, "score_candidates_batch", None)
    if callable(batch_fn):
        raw_results = batch_fn(candidates, metadatas, context=context)
        return [_coerce_voice_score_result(r) for r in raw_results]
    return [
        _coerce_voice_score_result(
            scorer.score_candidate(cand, metadata=meta, context=context)
        )
        for cand, meta in zip(candidates, metadatas)
    ]


class HeuristicVoiceQualityScorer:
    """Zero-LLM deterministic Gate-2 voice quality scorer (0.0 to 10.0).

    Scores Gate-1 DialogueCandidates using lexical distinctiveness, lexical diversity,
    length sweet-spot balance, attribution/metadata completeness, dialogue expressiveness,
    and artifact penalties.
    """

    def __init__(
        self,
        *,
        optimal_min_words: int = 16,
        optimal_max_words: int = 45,
        distinctiveness_weight: float = 4.2,
        lexical_diversity_weight: float = 2.3,
        length_weight: float = 1.5,
        attribution_weight: float = 1.2,
        expressiveness_weight: float = 0.8,
        artifact_penalty: float = 2.5,
    ) -> None:
        self.optimal_min_words = optimal_min_words
        self.optimal_max_words = optimal_max_words
        self.distinctiveness_weight = distinctiveness_weight
        self.lexical_diversity_weight = lexical_diversity_weight
        self.length_weight = length_weight
        self.attribution_weight = attribution_weight
        self.expressiveness_weight = expressiveness_weight
        self.artifact_penalty = artifact_penalty

    def _length_score(self, word_count: int) -> float:
        if self.optimal_min_words <= word_count <= self.optimal_max_words:
            return self.length_weight
        if word_count < self.optimal_min_words:
            # Linear ramp from 60% weight at 12 words up to 100% at optimal_min_words
            floor_words = max(1, self.optimal_min_words - 4)
            ratio = max(0.0, (word_count - floor_words) / max(1, self.optimal_min_words - floor_words))
            return self.length_weight * (0.60 + 0.40 * ratio)
        # Gentle taper above optimal_max_words
        excess = word_count - self.optimal_max_words
        taper = max(0.55, 1.0 - (excess / max(1, self.optimal_max_words)) * 0.45)
        return self.length_weight * taper

    def score_candidate(
        self,
        candidate: DialogueCandidate,
        *,
        metadata: dict[str, str],
        context: StepContext | None = None,
    ) -> VoiceScoreResult:
        del context  # Reserved for custom/LLM scorers
        reasons: list[str] = []

        dist = max(0.0, min(1.0, float(candidate.distinctiveness_score)))
        # Gate-1 distinctiveness for 12-20 word spans naturally peaks around 0.72-0.78 due to root_ttr/5.0
        norm_dist = max(0.0, min(1.0, (dist - 0.30) / 0.45))
        lex = max(0.0, min(1.0, float(candidate.lexical_diversity)))
        score = norm_dist * self.distinctiveness_weight + lex * self.lexical_diversity_weight
        score += self._length_score(candidate.word_count)

        # Attribution & dyadic metadata completeness
        attr_frac = 0.0
        if candidate.speaker_id:
            attr_frac += 0.50
        else:
            reasons.append("missing_speaker")
        if candidate.addressee_id:
            attr_frac += 0.35
        if any(str(v).strip() for v in metadata.values()):
            attr_frac += 0.15
        score += attr_frac * self.attribution_weight

        # Dialogue expressiveness & clause rhythm
        text = candidate.text.strip()
        expr_frac = 0.0
        if _EXPRESSIVE_PUNCT_RE.search(text):
            expr_frac += 0.50
        if _CLAUSE_PUNCT_RE.search(text):
            expr_frac += 0.50
        score += expr_frac * self.expressiveness_weight

        # Artifact / drift penalties
        if _ARTIFACT_RE.search(text):
            score -= self.artifact_penalty
            reasons.append("template_or_markup_artifact")
        if _EXCESSIVE_PUNCT_RE.search(text):
            score -= 0.8
            reasons.append("excessive_punctuation")
        if "\n" in text:
            score -= 0.6
            reasons.append("multiline_span")

        clamped = round(max(0.0, min(10.0, score)), 3)
        return VoiceScoreResult(
            score=clamped,
            reason=", ".join(reasons) if reasons else "heuristic_score",
        )


class CallableVoiceQualityScorer:
    """Adapter turning any Python callable into a Gate-2 VoiceQualityScorer."""

    def __init__(
        self,
        fn: Callable[..., float | int | VoiceScoreResult],
    ) -> None:
        self.fn = fn
        self._params = list(inspect.signature(fn).parameters.values())

    def score_candidate(
        self,
        candidate: DialogueCandidate,
        *,
        metadata: dict[str, str],
        context: StepContext | None = None,
    ) -> VoiceScoreResult:
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in self._params)
        accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in self._params)
        positional = [
            p
            for p in self._params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        param_names = {p.name for p in self._params}

        if accepts_kwargs:
            raw = self.fn(candidate, metadata=metadata, context=context)
        elif accepts_varargs or len(positional) >= 3:
            raw = self.fn(candidate, metadata, context)
        elif len(positional) == 2:
            raw = self.fn(candidate, metadata)
        elif len(positional) == 1:
            kwargs: dict[str, Any] = {}
            if "metadata" in param_names:
                kwargs["metadata"] = metadata
            if "context" in param_names:
                kwargs["context"] = context
            raw = self.fn(candidate, **kwargs)
        else:
            raw = self.fn(candidate)
        return _coerce_voice_score_result(raw)


def _parse_llm_voice_scores(
    raw: str | dict[str, Any] | list[Any],
    candidates: Sequence[DialogueCandidate],
) -> dict[str, VoiceScoreResult]:
    """Parses a batched LLM JSON response into candidate_id -> VoiceScoreResult."""
    from .extractors import _strip_json_fences

    parsed: Any = raw
    if isinstance(raw, str):
        cleaned = _strip_json_fences(raw).strip()
        if not cleaned.startswith(("{", "[")):
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            first_bracket = cleaned.find("[")
            last_bracket = cleaned.rfind("]")
            if first_brace != -1 and last_brace > first_brace and (
                first_bracket == -1 or first_brace < first_bracket
            ):
                cleaned = cleaned[first_brace : last_brace + 1]
            elif first_bracket != -1 and last_bracket > first_bracket:
                cleaned = cleaned[first_bracket : last_bracket + 1]
        parsed = json.loads(cleaned)

    if isinstance(parsed, dict):
        container: Any = (
            parsed.get("items")
            or parsed.get("scores")
            or parsed.get("results")
            or next((v for v in parsed.values() if isinstance(v, list)), [])
        )
    elif isinstance(parsed, list):
        container = parsed
    else:
        raise ValueError(f"Unsupported LLM voice score payload: {type(parsed).__name__}")

    valid_ids = {c.candidate_id for c in candidates}
    results: dict[str, VoiceScoreResult] = {}
    if not isinstance(container, list):
        return results

    for pos, entry in enumerate(container):
        if not isinstance(entry, dict):
            continue
        cid_raw = entry.get("candidate_id")
        if isinstance(cid_raw, str) and cid_raw in valid_ids:
            cid = cid_raw
        elif isinstance(entry.get("id"), int) and 0 <= entry["id"] < len(candidates):
            cid = candidates[entry["id"]].candidate_id
        elif pos < len(candidates) and candidates[pos].candidate_id not in results:
            cid = candidates[pos].candidate_id
        else:
            continue

        raw_score = entry.get("score")
        if not isinstance(raw_score, (int, float)):
            continue
        reason = str(entry.get("reason") or "").strip()
        raw_updates = entry.get("metadata_updates") or entry.get("metadata") or {}
        updates: dict[str, str] = {}
        if isinstance(raw_updates, Mapping):
            for k, v in raw_updates.items():
                ks, vs = str(k).strip(), str(v).strip()
                if ks and vs:
                    updates[ks] = vs
        # Convenience: allow top-level "tone" or "register" keys if emitted directly
        for direct_key in ("tone", "register"):
            val = entry.get(direct_key)
            if isinstance(val, str) and val.strip() and direct_key not in updates:
                updates[direct_key] = val.strip()

        results[cid] = _coerce_voice_score_result(
            VoiceScoreResult(score=float(raw_score), reason=reason, metadata_updates=updates)
        )
    return results


class LLMVoiceQualityScorer:
    """Optional LLM-backed Gate-2 voice quality judge backed by any JsonCompletionClient."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a strict literary voice-bank curator performing Gate-2 quality scoring. "
        "Grade each dialogue candidate from 0.0 to 10.0 on: (1) distinctive character voice "
        "and natural phrasing, (2) adherence to the expected interpersonal register/tone, and "
        "(3) fidelity if source_text is provided. Reserve scores >= 8.0 for strong, reusable "
        "voice anchors free of artifacts or register errors. "
        'Respond ONLY with valid JSON: {"items": [{"candidate_id": "...", "score": 8.5, '
        '"reason": "short justification", "metadata_updates": {}}]}.'
    )

    def __init__(
        self,
        extractor: JsonCompletionClient,
        *,
        batch_size: int = 15,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 1,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        fallback_scorer: VoiceQualityScorer | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.extractor: JsonCompletionClient = validate_json_completion_client(extractor)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max(0, max_retries)
        self.system_prompt = system_prompt
        self.fallback_scorer = fallback_scorer
        self.api_calls: int = 0
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def _build_batch_prompt(
        self,
        candidates: Sequence[DialogueCandidate],
        metadatas: Sequence[dict[str, str]],
    ) -> str:
        payload = [
            {
                "candidate_id": c.candidate_id,
                "speaker_id": c.speaker_id,
                "addressee_id": c.addressee_id,
                "expected_metadata": meta,
                "text": c.text,
                **({"source_text": c.source_text} if c.source_text else {}),
            }
            for c, meta in zip(candidates, metadatas)
        ]
        return (
            "Score each dialogue candidate below from 0.0 to 10.0. "
            "Preserve every candidate_id exactly.\n"
            f"Candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def score_candidates_batch(
        self,
        candidates: Sequence[DialogueCandidate],
        metadatas: Sequence[dict[str, str]],
        *,
        context: StepContext | None = None,
    ) -> list[VoiceScoreResult]:
        if not candidates:
            return []
        if len(candidates) != len(metadatas):
            raise ValueError("candidates and metadatas must have the same length.")

        resolved: dict[str, VoiceScoreResult] = {}
        for start in range(0, len(candidates), self.batch_size):
            chunk_cands = candidates[start : start + self.batch_size]
            chunk_metas = metadatas[start : start + self.batch_size]
            prompt = self._build_batch_prompt(chunk_cands, chunk_metas)

            parsed_chunk: dict[str, VoiceScoreResult] | None = None
            last_exc: Exception | None = None
            for _attempt in range(self.max_retries + 1):
                content, raw_usage = self.extractor.call_json_api(
                    prompt,
                    system_prompt=self.system_prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.api_calls += 1
                call_usage = _coerce_usage(raw_usage, latency_ms=0.0)
                self.last_usage = call_usage
                self.total_usage.add(call_usage)
                try:
                    parsed_chunk = _parse_llm_voice_scores(content, chunk_cands)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
            if parsed_chunk is None:
                if self.fallback_scorer is not None:
                    fallback_res = score_voice_candidates_batch(
                        self.fallback_scorer, chunk_cands, chunk_metas, context=context
                    )
                    for c, res in zip(chunk_cands, fallback_res):
                        resolved[c.candidate_id] = res
                    continue
                assert last_exc is not None
                raise last_exc
            resolved.update(parsed_chunk)

        out: list[VoiceScoreResult] = []
        for cand, meta in zip(candidates, metadatas):
            if cand.candidate_id in resolved:
                out.append(resolved[cand.candidate_id])
            elif self.fallback_scorer is not None:
                out.append(
                    _coerce_voice_score_result(
                        self.fallback_scorer.score_candidate(cand, metadata=meta, context=context)
                    )
                )
            else:
                out.append(VoiceScoreResult(score=0.0, reason="missing_from_llm_response"))
        return out

    def score_candidate(
        self,
        candidate: DialogueCandidate,
        *,
        metadata: dict[str, str],
        context: StepContext | None = None,
    ) -> VoiceScoreResult:
        return self.score_candidates_batch([candidate], [metadata], context=context)[0]


__all__ = [
    "DIALOGUE_PATTERNS",
    "_CJK_CHAR_RE",
    "_TOKEN_UNIT_RE",
    "CallableVoiceQualityScorer",
    "Embedder",
    "HeuristicVoiceQualityScorer",
    "IngestionGate",
    "LLMVoiceQualityScorer",
    "OfflineNgramEmbedder",
    "VoiceBank",
    "VoiceQualityScorer",
    "VoiceScoreResult",
    "_tokenize_units",
    "score_voice_candidates_batch",
]
