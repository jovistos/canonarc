from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import StepDelta


class DeltaExtractor(Protocol):
    """Protocol for LLM-backed structured StepDelta extraction."""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta: ...


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_ANYWHERE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_json_fences(raw: str) -> str:
    stripped = _THINK_BLOCK_RE.sub("", raw).strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    any_match = _ANYWHERE_FENCE_RE.search(stripped)
    if any_match:
        return any_match.group(1).strip()
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace and not stripped.startswith("{"):
        return stripped[first_brace : last_brace + 1].strip()
    return stripped


def _coerce_step_delta(raw: str | dict[str, Any] | StepDelta) -> StepDelta:
    if isinstance(raw, StepDelta):
        return raw
    if isinstance(raw, dict):
        return StepDelta.model_validate(raw)
    if isinstance(raw, str):
        cleaned = _strip_json_fences(raw)
        return StepDelta.model_validate_json(cleaned)
    raise TypeError(f"Unsupported StepDelta payload type: {type(raw).__name__}")


def load_dotenv(
    dotenv_path: str | Path = ".env",
    *,
    override: bool = False,
) -> dict[str, str]:
    """Zero-dependency .env file parser and loader."""
    path = Path(dotenv_path)
    if not path.is_file():
        # Also check repository root relative to this package if running from subdirectories
        alt_path = Path(__file__).resolve().parents[2] / ".env"
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


RawStepDelta = str | dict[str, Any] | StepDelta
UsageInput = TokenUsage | Mapping[str, Any] | None
CallableReturn = RawStepDelta | tuple[RawStepDelta, UsageInput]


def _unpack_callable_result(raw: Any) -> tuple[Any, UsageInput]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0], raw[1]
    return raw, None


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


class CallableExtractor:
    """Adapter turning any (prompt, schema) or (system_prompt, user_prompt, schema) callable into a DeltaExtractor.

    Automatically strips Markdown JSON fences, validates against StepDelta, supports
    automatic retries or custom repair callbacks on malformed responses, and bridges
    optional TokenUsage or SDK usage dicts (OpenAI, DeepSeek, Anthropic, LangChain)
    into ``last_usage``, ``total_usage``, and ``CachedDeltaExtractor``.
    """

    def __init__(
        self,
        fn: Callable[[str, dict[str, Any]], CallableReturn] | None = None,
        *,
        messages_fn: Callable[[str, str, dict[str, Any]], CallableReturn] | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        max_retries: int = 1,
        repair_fn: Callable[[str, Exception], CallableReturn] | None = None,
        on_usage: Callable[[TokenUsage], None] | None = None,
    ) -> None:
        if fn is None and messages_fn is None:
            raise ValueError("CallableExtractor requires at least one of 'fn' or 'messages_fn'.")
        self.fn = fn
        self.messages_fn = messages_fn
        self.model: str | None = model.strip() if isinstance(model, str) and model.strip() else None
        self.system_prompt: str = system_prompt or ""
        self.max_retries = max(0, max_retries)
        self.repair_fn = repair_fn
        self.on_usage = on_usage
        self.last_retries: int = 0
        self.last_validation_errors: int = 0
        self.total_retries: int = 0
        self.total_validation_errors: int = 0
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    def _extract_with_retries(
        self,
        *,
        base_user_prompt: str,
        system_prompt: str | None,
    ) -> StepDelta:
        schema = StepDelta.model_json_schema()
        current_prompt = base_user_prompt
        last_exc: Exception | None = None
        last_raw_str = ""
        retries_used = 0
        val_errors = 0
        call_usage = TokenUsage()

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                retries_used += 1
                self.total_retries += 1
            t0 = time.perf_counter()
            if attempt > 0 and self.repair_fn is not None and last_exc is not None:
                raw_result = self.repair_fn(last_raw_str, last_exc)
            elif system_prompt is not None and self.messages_fn is not None:
                raw_result = self.messages_fn(system_prompt, current_prompt, schema)
            else:
                assert self.fn is not None
                raw_result = self.fn(current_prompt, schema)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_payload, raw_usage = _unpack_callable_result(raw_result)
            attempt_usage = _coerce_usage(raw_usage, latency_ms=elapsed_ms)
            if attempt > 0:
                attempt_usage.retry_count += 1
            call_usage.add(attempt_usage)

            last_raw_str = (
                raw_payload if isinstance(raw_payload, str) else json.dumps(raw_payload, default=str)
            )
            try:
                parsed = _coerce_step_delta(raw_payload)
                self.last_retries = retries_used
                self.last_validation_errors = val_errors
                self.last_usage = call_usage
                self.total_usage.add(call_usage)
                if self.on_usage is not None:
                    self.on_usage(TokenUsage.from_dict(call_usage.to_dict()))
                return parsed
            except Exception as exc:  # noqa: BLE001
                val_errors += 1
                call_usage.validation_errors += 1
                self.total_validation_errors += 1
                self.last_retries = retries_used
                self.last_validation_errors = val_errors
                last_exc = exc
                current_prompt = (
                    f"{base_user_prompt}\n\n"
                    f"Previous output failed StepDelta JSON validation with error:\n{exc}\n"
                    "Return ONLY valid JSON matching the StepDelta schema."
                )

        self.last_usage = call_usage
        self.total_usage.add(call_usage)
        if self.on_usage is not None:
            self.on_usage(TokenUsage.from_dict(call_usage.to_dict()))
        assert last_exc is not None
        raise last_exc

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        if self.fn is not None:
            return self._extract_with_retries(base_user_prompt=prompt, system_prompt=None)
        return self._extract_with_retries(
            base_user_prompt=prompt,
            system_prompt=self.system_prompt,
        )

    def extract_delta_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        scene_text: str,
    ) -> StepDelta:
        """Extracts a StepDelta with separate system and user prompts for prefix-cache reuse."""
        if self.messages_fn is not None:
            return self._extract_with_retries(
                base_user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
        return self.extract_delta(f"{system_prompt}\n\n{user_prompt}", scene_text)


def resolve_lmstudio_model(
    base_url: str = "http://localhost:1234/v1",
    *,
    timeout: float = 5.0,
) -> str:
    """Auto-discovers the currently loaded LLM/VLM in LM Studio without requiring a model name.

    1. Queries LM Studio's `/api/v0/models` endpoint to find the model with `"state": "loaded"`
       (ignoring embedding models).
    2. Falls back to `/v1/models` and selects the first non-embedding model.
    3. Returns `"local-model"` if the server is not reachable yet.
    """
    clean_base = base_url.rstrip("/")
    host_root = clean_base.removesuffix("/v1")

    # 1. Try LM Studio native /api/v0/models to detect which model has state == "loaded"
    try:
        req_v0 = urllib.request.Request(f"{host_root}/api/v0/models", method="GET")
        with urllib.request.urlopen(req_v0, timeout=timeout) as resp:
            data_v0 = json.loads(resp.read().decode("utf-8")).get("data") or []
        loaded_models = [
            str(m["id"])
            for m in data_v0
            if m.get("id")
            and m.get("state") == "loaded"
            and m.get("type") not in ("embeddings", "embedding")
        ]
        if loaded_models:
            return loaded_models[0]
        non_emb_v0 = [
            str(m["id"])
            for m in data_v0
            if m.get("id") and m.get("type") not in ("embeddings", "embedding")
        ]
        if non_emb_v0:
            return non_emb_v0[0]
    except Exception:  # noqa: BLE001, S110
        pass

    # 2. Fallback to standard OpenAI-compatible /v1/models
    try:
        v1_url = f"{clean_base}/models" if clean_base.endswith("/v1") else f"{clean_base}/v1/models"
        req_v1 = urllib.request.Request(v1_url, method="GET")
        with urllib.request.urlopen(req_v1, timeout=timeout) as resp:
            data_v1 = json.loads(resp.read().decode("utf-8")).get("data") or []
        candidates = [
            str(m["id"])
            for m in data_v1
            if m.get("id") and "embed" not in str(m["id"]).lower()
        ]
        if candidates:
            return candidates[0]
    except Exception:  # noqa: BLE001, S110
        pass

    return "local-model"


class OpenAICompatibleExtractor:
    """Zero-dependency HTTP extractor for DeepSeek, OpenAI, LM Studio, LiteLLM, vLLM, or Ollama."""

    def __init__(
        self,
        *,
        model: str | None = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        use_json_object_format: bool = True,
        timeout: float = 120.0,
        max_retries: int = 1,
        system_prompt: str = (
            "You are a precise narrative state extraction engine. "
            "Respond only with valid JSON conforming to the requested StepDelta schema."
        ),
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if model is None or str(model).strip().lower() in ("", "auto", "default"):
            if "1234" in self.base_url or "lmstudio" in self.base_url.lower():
                self.model = resolve_lmstudio_model(self.base_url)
            else:
                self.model = "local-model"
        else:
            self.model = str(model).strip()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        if response_format is not None:
            self.response_format: dict[str, Any] | None = response_format
        elif use_json_object_format:
            self.response_format = {"type": "json_object"}
        else:
            self.response_format = None
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.system_prompt = system_prompt
        self.headers = dict(headers) if headers else {}
        self.last_usage: TokenUsage = TokenUsage()
        self.total_usage: TokenUsage = TokenUsage()

    @classmethod
    def from_lmstudio(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
        reasoning_effort: str | None = "none",
        timeout: float = 600.0,
        max_retries: int = 2,
        dotenv_path: str | Path = ".env",
    ) -> OpenAICompatibleExtractor:
        """Creates an OpenAICompatibleExtractor for LM Studio.

        If `model` is omitted (or `'auto'` / `'default'`), automatically queries LM Studio's
        `/api/v0/models` (and `/v1/models`) to use the currently loaded model in LM Studio.
        """
        load_dotenv(dotenv_path)
        resolved_url: str = (
            base_url or os.environ.get("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
        )
        raw_model = model or os.environ.get("LMSTUDIO_MODEL")
        if raw_model is None or str(raw_model).strip().lower() in ("", "auto", "default"):
            resolved_model = resolve_lmstudio_model(resolved_url)
        else:
            resolved_model = str(raw_model).strip()
        return cls(
            model=resolved_model,
            api_key=api_key or os.environ.get("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=resolved_url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            use_json_object_format=False,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_gemini(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
        reasoning_effort: str | None = "low",
        timeout: float = 180.0,
        max_retries: int = 2,
        dotenv_path: str | Path = ".env",
    ) -> OpenAICompatibleExtractor:
        """Creates an OpenAICompatibleExtractor pre-configured for Google Gemini (default: gemini-4-hydrogen)."""
        load_dotenv(dotenv_path)
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        resolved_url: str = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        resolved_model: str = (
            model
            or os.environ.get("GEMINI_EXTRACT_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-4-hydrogen"
        )
        return cls(
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_deepseek(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
        reasoning_effort: str | None = "low",
        timeout: float = 180.0,
        max_retries: int = 2,
        dotenv_path: str | Path = ".env",
    ) -> OpenAICompatibleExtractor:
        """Creates an OpenAICompatibleExtractor pre-configured for DeepSeek Flash / DeepSeek API."""
        load_dotenv(dotenv_path)
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        resolved_url: str = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        )
        resolved_model: str = (
            model
            or os.environ.get("DEEPSEEK_EXTRACT_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or "deepseek-flash"
        )
        return cls(
            model=resolved_model,
            api_key=resolved_key,
            base_url=resolved_url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            max_retries=max_retries,
        )

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Executes a JSON-mode chat completion and returns (content_str, token_usage)."""
        body_dict: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "messages": [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.model and self.model not in ("auto", "default"):
            body_dict["model"] = self.model
        if self.response_format is not None:
            body_dict["response_format"] = self.response_format
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None:
            body_dict["max_tokens"] = effective_max_tokens
        effective_effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        if effective_effort is not None:
            body_dict["reasoning_effort"] = effective_effort

        req_headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            req_headers.setdefault("Authorization", f"Bearer {self.api_key}")

        def _send(payload_dict: dict[str, Any]) -> tuple[dict[str, Any], float]:
            payload = json.dumps(payload_dict).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers=req_headers,
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                parsed_body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return parsed_body, (time.perf_counter() - t0) * 1000.0

        transient_codes = {429, 500, 502, 503, 504}
        max_http_attempts = 4
        for http_attempt in range(max_http_attempts):
            try:
                body, elapsed_ms = _send(body_dict)
                break
            except urllib.error.HTTPError as exc:
                err_text = ""
                try:
                    err_text = exc.read().decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001, S110
                    pass
                # LM Studio rejects {"type": "json_object"}; automatically fallback without response_format
                if exc.code == 400 and "response_format" in body_dict and "response_format" in err_text:
                    self.response_format = None
                    body_dict.pop("response_format", None)
                    body, elapsed_ms = _send(body_dict)
                    break
                if (
                    exc.code in transient_codes
                    and http_attempt + 1 < max_http_attempts
                    and "limit: 0" not in err_text
                ):
                    time.sleep(2.0 * (2 ** http_attempt))
                    continue
                raise RuntimeError(
                    f"HTTP {exc.code} from {self.base_url}/chat/completions: {err_text}"
                ) from exc

        if body.get("model") and self.model in ("", "auto", "default", "local-model"):
            self.model = str(body["model"])
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible endpoint returned no choices.")
        usage = TokenUsage.from_api_usage(body.get("usage"), latency_ms=elapsed_ms)
        self.last_usage = usage
        self.total_usage.add(usage)
        content = str(choices[0].get("message", {}).get("content", ""))
        return content, usage

    def _call_api(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> str:
        content, _ = self.call_json_api(prompt, system_prompt=system_prompt)
        return content

    def extract_delta(
        self,
        prompt: str,
        scene_text: str,
        *,
        system_prompt: str | None = None,
    ) -> StepDelta:
        def _call_with_system(cur_prompt: str, schema: dict[str, Any]) -> str:
            return self._call_api(cur_prompt, schema, system_prompt=system_prompt)

        call_fn: Callable[[str, dict[str, Any]], CallableReturn] = (
            self._call_api if system_prompt is None else _call_with_system
        )
        helper = CallableExtractor(call_fn, max_retries=self.max_retries)
        try:
            delta = helper.extract_delta(prompt, scene_text)
        finally:
            self.last_usage.retry_count += helper.last_retries
            self.last_usage.validation_errors += helper.last_validation_errors
            self.total_usage.retry_count += helper.last_retries
            self.total_usage.validation_errors += helper.last_validation_errors
        return delta

    def extract_delta_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        scene_text: str,
    ) -> StepDelta:
        """Extracts a StepDelta with separate system and user prompts for prefix-cache reuse."""
        return self.extract_delta(user_prompt, scene_text, system_prompt=system_prompt)


class CachedDeltaExtractor:
    """Content-addressed disk-caching wrapper around any DeltaExtractor.

    Caches validated StepDelta extractions keyed by (model, system_prompt, prompt, scene_text)
    so repeated runs with the same chunk, prompt, and model reuse existing extractions with zero API calls.
    Also records token usage metadata inside each cache file for accurate cost estimation even on cache hits.
    """

    def __init__(
        self,
        extractor: DeltaExtractor,
        cache_dir: str | Path,
        *,
        model_id: str | None = None,
        system_prompt: str | None = None,
        bypass_cache: bool = False,
    ) -> None:
        self.extractor = extractor
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id or getattr(extractor, "model", None) or type(extractor).__name__
        self.system_prompt = system_prompt or getattr(extractor, "system_prompt", "")
        self.bypass_cache = bypass_cache

        self.cache_hits: int = 0
        self.cache_misses: int = 0
        # Billable usage incurred during the current process run (0 on cache hits)
        self.billable_usage: TokenUsage = TokenUsage()
        # Cumulative usage including historical tokens saved by cache hits
        self.historical_usage: TokenUsage = TokenUsage()
        self.step_usage_log: list[dict[str, Any]] = []

    def compute_cache_key(
        self,
        prompt: str,
        scene_text: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        effective_system = self.system_prompt if system_prompt is None else system_prompt
        raw_material = (
            f"model={self.model_id}\0"
            f"system={effective_system}\0"
            f"prompt={prompt.strip()}\0"
            f"scene_text={scene_text.strip()}"
        )
        return hashlib.sha256(raw_material.encode("utf-8")).hexdigest()

    def _cache_file_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def extract_delta_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        scene_text: str,
    ) -> StepDelta:
        """Caches and delegates a split (system_prompt, user_prompt) extraction."""
        return self.extract_delta(user_prompt, scene_text, system_prompt=system_prompt)

    def extract_delta(
        self,
        prompt: str,
        scene_text: str,
        *,
        system_prompt: str | None = None,
    ) -> StepDelta:
        cache_key = self.compute_cache_key(prompt, scene_text, system_prompt=system_prompt)
        cache_path = self._cache_file_path(cache_key)

        if not self.bypass_cache and cache_path.is_file():
            try:
                cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                delta = StepDelta.model_validate(cached_payload["delta"])
                saved_usage = TokenUsage.from_dict(cached_payload.get("usage"))
                self.cache_hits += 1
                self.billable_usage.cache_hits += 1
                self.historical_usage.add(saved_usage)
                self.historical_usage.cache_hits += 1
                self.step_usage_log.append(
                    {
                        "cache_key": cache_key,
                        "cached": True,
                        "model": self.model_id,
                        "scene_preview": scene_text.strip()[:90],
                        "usage": saved_usage.to_dict(),
                    }
                )
                return delta
            except Exception:  # noqa: BLE001, S110
                # If cache file is corrupted or schema-incompatible, fall through and re-extract
                pass

        # Cache miss: invoke underlying extractor
        before_total = (
            TokenUsage.from_dict(self.extractor.total_usage.to_dict())
            if hasattr(self.extractor, "total_usage")
            else None
        )
        t0 = time.perf_counter()
        if system_prompt is not None:
            extract_messages = getattr(self.extractor, "extract_delta_messages", None)
            if callable(extract_messages):
                delta = extract_messages(system_prompt, prompt, scene_text)
            else:
                delta = self.extractor.extract_delta(f"{system_prompt}\n\n{prompt}", scene_text)
        else:
            delta = self.extractor.extract_delta(prompt, scene_text)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if hasattr(self.extractor, "total_usage") and before_total is not None:
            after_total: TokenUsage = self.extractor.total_usage
            raw_lat = after_total.latency_ms - before_total.latency_ms
            call_usage = TokenUsage(
                prompt_tokens=after_total.prompt_tokens - before_total.prompt_tokens,
                completion_tokens=after_total.completion_tokens - before_total.completion_tokens,
                total_tokens=after_total.total_tokens - before_total.total_tokens,
                cached_prompt_tokens=after_total.cached_prompt_tokens - before_total.cached_prompt_tokens,
                reasoning_tokens=after_total.reasoning_tokens - before_total.reasoning_tokens,
                api_calls=max(1, after_total.api_calls - before_total.api_calls),
                retry_count=after_total.retry_count - before_total.retry_count,
                validation_errors=after_total.validation_errors - before_total.validation_errors,
                latency_ms=round(raw_lat if raw_lat > 0 else wall_ms, 3),
            )
        else:
            last_u = getattr(self.extractor, "last_usage", None)
            if isinstance(last_u, TokenUsage):
                call_usage = TokenUsage.from_dict(last_u.to_dict())
                if call_usage.latency_ms <= 0:
                    call_usage.latency_ms = round(wall_ms, 3)
            else:
                retries = int(getattr(self.extractor, "last_retries", 0))
                val_errs = int(getattr(self.extractor, "last_validation_errors", 0))
                call_usage = TokenUsage(
                    api_calls=1 + retries,
                    retry_count=retries,
                    validation_errors=val_errs,
                    latency_ms=round(wall_ms, 3),
                )

        self.cache_misses += 1
        self.billable_usage.add(call_usage)
        self.billable_usage.cache_misses += 1
        self.historical_usage.add(call_usage)
        self.historical_usage.cache_misses += 1

        record = {
            "cache_key": cache_key,
            "model": self.model_id,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
            "scene_text_sha256": hashlib.sha256(scene_text.strip().encode("utf-8")).hexdigest(),
            "scene_text": scene_text,
            "usage": call_usage.to_dict(),
            "delta": delta.model_dump(mode="json"),
        }
        tmp_path = cache_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(cache_path)

        self.step_usage_log.append(
            {
                "cache_key": cache_key,
                "cached": False,
                "model": self.model_id,
                "scene_preview": scene_text.strip()[:90],
                "usage": call_usage.to_dict(),
            }
        )
        return delta
