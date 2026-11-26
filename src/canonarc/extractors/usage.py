from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_dotenv(
    dotenv_path: str | Path = ".env",
    *,
    override: bool = False,
) -> dict[str, str]:
    """Zero-dependency .env file parser and loader."""
    path = Path(dotenv_path)
    if not path.is_file():
        # Also check repository root relative to this package if running from subdirectories
        alt_path = Path(__file__).resolve().parents[3] / ".env"
        if alt_path.is_file():
            path = alt_path
        else:
            return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        if key:
            loaded[key] = val
            if override or key not in os.environ:
                os.environ[key] = val
    return loaded


MODEL_PRICING_PER_1M_USD: dict[str, tuple[float, float, float]] = {
    # (input_uncached_per_1m, input_cached_per_1m, output_per_1m)
    "deepseek-flash": (0.14, 0.014, 0.28),
    "deepseek-chat": (0.14, 0.014, 0.28),
    "deepseek-v3": (0.14, 0.014, 0.28),
    "deepseek-reasoner": (0.55, 0.14, 2.19),
    "deepseek-r1": (0.55, 0.14, 2.19),
    "gemini-4-hydrogen": (0.10, 0.025, 0.40),
    "gemini-2.5-flash": (0.10, 0.025, 0.40),
    "gemini-2.0-flash": (0.10, 0.025, 0.40),
    "gemini-2.5-pro": (1.25, 0.3125, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    "rule-based-non-llm": (0.0, 0.0, 0.0),
    "deterministic_compiler": (0.0, 0.0, 0.0),
    "local-model": (0.0, 0.0, 0.0),
}

_LOCAL_OR_FREE_MODEL_PREFIXES: tuple[str, ...] = (
    "qwen/",
    "qwen-",
    "llama",
    "meta-llama/",
    "gemma",
    "mistral",
    "phi-",
    "local",
    "lmstudio",
    "ollama",
    "rule-based",
    "deterministic",
    "sim-",
    "tier1-",
    "tier2-",
    "tier3-",
    "tier4-",
    "cached-test",
)


def get_model_pricing_per_1m_usd(model: str | None = None) -> tuple[float, float, float]:
    """Returns (input_uncached, input_cached, output) USD rates per 1M tokens for a given model."""
    if not model:
        return (0.14, 0.014, 0.28)
    clean = model.strip().lower()
    if clean in MODEL_PRICING_PER_1M_USD:
        return MODEL_PRICING_PER_1M_USD[clean]
    if any(clean.startswith(pref) for pref in _LOCAL_OR_FREE_MODEL_PREFIXES) or "@" in clean:
        return (0.0, 0.0, 0.0)
    if "gemini" in clean and ("flash" in clean or "hydrogen" in clean):
        return (0.10, 0.025, 0.40)
    if "gemini" in clean:
        return (1.25, 0.3125, 10.00)
    if "deepseek" in clean:
        return (0.14, 0.014, 0.28)
    if "gpt-4o-mini" in clean:
        return (0.15, 0.075, 0.60)
    return (0.14, 0.014, 0.28)


@dataclass
class TokenUsage:
    """Tracks token counts, caching metrics, retries, latency, and estimated API cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    validation_errors: int = 0
    latency_ms: float = 0.0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.api_calls += other.api_calls
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.retry_count += other.retry_count
        self.validation_errors += other.validation_errors
        self.latency_ms = round(self.latency_ms + other.latency_ms, 3)

    def estimate_cost_usd(
        self,
        *,
        model: str | None = None,
        input_price_per_1m: float | None = None,
        cached_input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ) -> dict[str, Any]:
        """Estimates USD price from token counts and model pricing registry."""
        default_in, default_cached, default_out = get_model_pricing_per_1m_usd(model)
        eff_in = default_in if input_price_per_1m is None else input_price_per_1m
        eff_cached = default_cached if cached_input_price_per_1m is None else cached_input_price_per_1m
        eff_out = default_out if output_price_per_1m is None else output_price_per_1m

        uncached_prompt = max(0, self.prompt_tokens - self.cached_prompt_tokens)
        input_cost = (uncached_prompt / 1_000_000.0) * eff_in
        cached_cost = (self.cached_prompt_tokens / 1_000_000.0) * eff_cached
        output_cost = (self.completion_tokens / 1_000_000.0) * eff_out
        total_cost = input_cost + cached_cost + output_cost
        return {
            "input_cost_usd": round(input_cost, 6),
            "cached_input_cost_usd": round(cached_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "rates_per_1m_usd": {
                "input_uncached": eff_in,
                "input_cached": eff_cached,
                "output": eff_out,
            },
        }

    def to_dict(self, *, model: str | None = None) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retry_count": self.retry_count,
            "validation_errors": self.validation_errors,
            "latency_ms": round(self.latency_ms, 3),
            "cost_estimate": self.estimate_cost_usd(model=model),
        }

    @classmethod
    def from_api_usage(cls, usage_dict: Mapping[str, Any] | None, *, latency_ms: float = 0.0) -> TokenUsage:
        if not usage_dict:
            return cls(api_calls=1, latency_ms=round(latency_ms, 3))
        raw_prompt = usage_dict.get("prompt_tokens")
        if raw_prompt is None:
            raw_prompt = usage_dict.get("input_tokens", 0)
        prompt_tokens = int(raw_prompt or 0)

        raw_completion = usage_dict.get("completion_tokens")
        if raw_completion is None:
            raw_completion = usage_dict.get("output_tokens", 0)
        completion_tokens = int(raw_completion or 0)

        raw_total = usage_dict.get("total_tokens")
        total_tokens = int(prompt_tokens + completion_tokens if raw_total is None else raw_total)

        prompt_details = usage_dict.get("prompt_tokens_details") or {}
        input_details = usage_dict.get("input_token_details") or {}
        raw_cached = usage_dict.get("prompt_cache_hit_tokens")
        if raw_cached is None:
            raw_cached = usage_dict.get("cached_prompt_tokens")
        if raw_cached is None:
            raw_cached = usage_dict.get("cache_read_input_tokens")
        if raw_cached is None and isinstance(prompt_details, Mapping):
            raw_cached = prompt_details.get("cached_tokens")
        if raw_cached is None and isinstance(input_details, Mapping):
            raw_cached = input_details.get("cache_read")
        cached_tokens = int(raw_cached or 0)

        comp_details = usage_dict.get("completion_tokens_details") or {}
        output_details = usage_dict.get("output_token_details") or {}
        raw_reasoning = usage_dict.get("reasoning_tokens")
        if raw_reasoning is None and isinstance(comp_details, Mapping):
            raw_reasoning = comp_details.get("reasoning_tokens")
        if raw_reasoning is None and isinstance(output_details, Mapping):
            raw_reasoning = output_details.get("reasoning")
        reasoning_tokens = int(raw_reasoning or 0)

        eff_latency = latency_ms
        if eff_latency <= 0 and usage_dict.get("latency_ms") is not None:
            eff_latency = float(usage_dict["latency_ms"])
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            api_calls=1,
            latency_ms=round(eff_latency, 3),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TokenUsage:
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            cached_prompt_tokens=int(data.get("cached_prompt_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
            api_calls=int(data.get("api_calls", 0)),
            cache_hits=int(data.get("cache_hits", 0)),
            cache_misses=int(data.get("cache_misses", 0)),
            retry_count=int(data.get("retry_count", 0)),
            validation_errors=int(data.get("validation_errors", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


UsageInput = TokenUsage | Mapping[str, Any] | None


def _coerce_usage(raw_usage: UsageInput, *, latency_ms: float) -> TokenUsage:
    if raw_usage is None:
        return TokenUsage(api_calls=1, latency_ms=round(latency_ms, 3))
    if isinstance(raw_usage, TokenUsage):
        cloned = TokenUsage.from_dict(raw_usage.to_dict())
        if cloned.api_calls <= 0:
            cloned.api_calls = 1
        if cloned.latency_ms <= 0:
            cloned.latency_ms = round(latency_ms, 3)
        return cloned
    if isinstance(raw_usage, Mapping):
        return TokenUsage.from_api_usage(raw_usage, latency_ms=latency_ms)
    raise TypeError(f"Unsupported usage payload type: {type(raw_usage).__name__}")
