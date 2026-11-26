from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..extractors import (
    JsonCompletionClient,
    TokenUsage,
    _coerce_usage,
    _strip_json_fences,
    validate_json_completion_client,
)
from ..models import DialogueCandidate, StepContext

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
    """Zero-LLM deterministic Gate-2 voice quality scorer (0.0 to 10.0)."""

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
            floor_words = max(1, self.optimal_min_words - 4)
            ratio = max(0.0, (word_count - floor_words) / max(1, self.optimal_min_words - floor_words))
            return self.length_weight * (0.60 + 0.40 * ratio)
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
        del context
        reasons: list[str] = []

        dist = max(0.0, min(1.0, float(candidate.distinctiveness_score)))
        norm_dist = max(0.0, min(1.0, (dist - 0.30) / 0.45))
        lex = max(0.0, min(1.0, float(candidate.lexical_diversity)))
        score = norm_dist * self.distinctiveness_weight + lex * self.lexical_diversity_weight
        score += self._length_score(candidate.word_count)

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

        text = candidate.text.strip()
        expr_frac = 0.0
        if _EXPRESSIVE_PUNCT_RE.search(text):
            expr_frac += 0.50
        if _CLAUSE_PUNCT_RE.search(text):
            expr_frac += 0.50
        score += expr_frac * self.expressiveness_weight

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
