from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..audit import AuditReport
from ..extractors import TokenUsage, resolve_client_model
from ..models import (
    EpisodeRecord,
    StageUsageSummary,
    StepContextSummary,
    StepDelta,
    TransactionUsageSummary,
    VoiceHarvestTrace,
)

KnownEntityScope = Literal["scene", "characters_compact", "all_compact"]

_ACTIVE_COMMIT_EPISODES: contextvars.ContextVar[list[EpisodeRecord] | None] = (
    contextvars.ContextVar("_canonarc_active_commit_episodes", default=None)
)


def compute_step_hash(
    text: str,
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Computes a deterministic SHA-256 hex digest for step resumability checks."""
    payload: dict[str, Any] = {"text": text}
    if model is not None:
        payload["model"] = model
    if prompt_version is not None:
        payload["prompt_version"] = prompt_version
    if extra:
        payload["extra"] = extra
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DraftAuditTrace(BaseModel):
    """Recorded pre-commit audit_draft() execution within a StepTransaction."""

    attempt: int
    label: str | None = None
    passed: bool
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    report: AuditReport


class StepTransactionTrace(BaseModel):
    """Standardized, JSON-ready provenance trace for a StepTransaction."""

    schema_version: str = "1.0"
    timeline_id: str
    step: int
    lang: str | None = None
    rerun_clean: bool = True
    status: Literal["open", "committed", "skipped", "rolled_back"] = "open"
    content_hash: str | None = None
    checkpoint_metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    context: StepContextSummary
    audit_reports: list[DraftAuditTrace] = Field(default_factory=list)
    extraction_skipped_by_policy: bool = False
    raw_delta: StepDelta | None = None
    staged_delta: StepDelta | None = None
    delta_counts: dict[str, int] = Field(default_factory=dict)
    staged_extra_episodes: int = 0
    staged_extra_exemplars: int = 0
    voice_harvest: VoiceHarvestTrace = Field(default_factory=VoiceHarvestTrace)
    harvested_voice_count: int = 0
    committed_episode_ids: list[str] = Field(default_factory=list)
    indexed_exemplar_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    usage: TransactionUsageSummary = Field(default_factory=TransactionUsageSummary)


def _get_component_usage_counter(component: Any) -> TokenUsage | None:
    """Returns the billable/total TokenUsage counter on an extractor, scorer, or generator."""
    if component is None:
        return None
    billable = getattr(component, "billable_usage", None)
    if isinstance(billable, TokenUsage):
        return billable
    total = getattr(component, "total_usage", None)
    if isinstance(total, TokenUsage):
        return total
    return None


def _snapshot_component_usage(component: Any) -> TokenUsage | None:
    counter = _get_component_usage_counter(component)
    if counter is None:
        return None
    return TokenUsage.from_dict(counter.to_dict())


def _resolve_component_model(component: Any) -> str:
    """Resolves model identifier from an extractor, wrapper, voice scorer, or inflection generator."""
    model = getattr(component, "model", None) or getattr(component, "model_id", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    inner = getattr(component, "extractor", None)
    if inner is not None and inner is not component:
        return _resolve_component_model(inner)
    return resolve_client_model(component)


def _diff_component_usage(
    stage: str,
    component: Any,
    before: TokenUsage | None,
) -> StageUsageSummary | None:
    if before is None:
        return None
    after = _get_component_usage_counter(component)
    if after is None:
        return None
    delta = TokenUsage(
        prompt_tokens=max(0, after.prompt_tokens - before.prompt_tokens),
        completion_tokens=max(0, after.completion_tokens - before.completion_tokens),
        total_tokens=max(0, after.total_tokens - before.total_tokens),
        cached_prompt_tokens=max(0, after.cached_prompt_tokens - before.cached_prompt_tokens),
        reasoning_tokens=max(0, after.reasoning_tokens - before.reasoning_tokens),
        api_calls=max(0, after.api_calls - before.api_calls),
        cache_hits=max(0, after.cache_hits - before.cache_hits),
        cache_misses=max(0, after.cache_misses - before.cache_misses),
        retry_count=max(0, after.retry_count - before.retry_count),
        validation_errors=max(0, after.validation_errors - before.validation_errors),
        latency_ms=round(max(0.0, after.latency_ms - before.latency_ms), 3),
    )
    if (
        delta.api_calls == 0
        and delta.cache_hits == 0
        and delta.cache_misses == 0
        and delta.total_tokens == 0
    ):
        return None
    model = _resolve_component_model(component)
    cost = float(delta.estimate_cost_usd(model=model)["total_cost_usd"])
    return StageUsageSummary(
        stage=stage,
        model=model,
        prompt_tokens=delta.prompt_tokens,
        completion_tokens=delta.completion_tokens,
        total_tokens=delta.total_tokens,
        cached_prompt_tokens=delta.cached_prompt_tokens,
        reasoning_tokens=delta.reasoning_tokens,
        api_calls=delta.api_calls,
        cache_hits=delta.cache_hits,
        cache_misses=delta.cache_misses,
        retry_count=delta.retry_count,
        validation_errors=delta.validation_errors,
        latency_ms=delta.latency_ms,
        estimated_cost_usd=cost,
    )


def _merge_stage_usage(
    existing: StageUsageSummary | None,
    added: StageUsageSummary,
) -> StageUsageSummary:
    if existing is None:
        return added
    merged_model = existing.model if existing.model == added.model else "mixed"
    return StageUsageSummary(
        stage=existing.stage,
        model=merged_model,
        prompt_tokens=existing.prompt_tokens + added.prompt_tokens,
        completion_tokens=existing.completion_tokens + added.completion_tokens,
        total_tokens=existing.total_tokens + added.total_tokens,
        cached_prompt_tokens=existing.cached_prompt_tokens + added.cached_prompt_tokens,
        reasoning_tokens=existing.reasoning_tokens + added.reasoning_tokens,
        api_calls=existing.api_calls + added.api_calls,
        cache_hits=existing.cache_hits + added.cache_hits,
        cache_misses=existing.cache_misses + added.cache_misses,
        retry_count=existing.retry_count + added.retry_count,
        validation_errors=existing.validation_errors + added.validation_errors,
        latency_ms=round(existing.latency_ms + added.latency_ms, 3),
        estimated_cost_usd=round(existing.estimated_cost_usd + added.estimated_cost_usd, 6),
    )


def _build_transaction_usage_summary(
    stages: Mapping[str, StageUsageSummary],
) -> TransactionUsageSummary:
    stage_copies = {k: v.model_copy(deep=True) for k, v in stages.items()}
    return TransactionUsageSummary(
        stages=stage_copies,
        prompt_tokens=sum(s.prompt_tokens for s in stage_copies.values()),
        completion_tokens=sum(s.completion_tokens for s in stage_copies.values()),
        total_tokens=sum(s.total_tokens for s in stage_copies.values()),
        cached_prompt_tokens=sum(s.cached_prompt_tokens for s in stage_copies.values()),
        reasoning_tokens=sum(s.reasoning_tokens for s in stage_copies.values()),
        api_calls=sum(s.api_calls for s in stage_copies.values()),
        cache_hits=sum(s.cache_hits for s in stage_copies.values()),
        cache_misses=sum(s.cache_misses for s in stage_copies.values()),
        retry_count=sum(s.retry_count for s in stage_copies.values()),
        validation_errors=sum(s.validation_errors for s in stage_copies.values()),
        latency_ms=round(sum(s.latency_ms for s in stage_copies.values()), 3),
        estimated_cost_usd=round(sum(s.estimated_cost_usd for s in stage_copies.values()), 6),
    )
