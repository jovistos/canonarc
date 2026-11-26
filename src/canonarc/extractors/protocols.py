from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from ..models import StepDelta
from .usage import TokenUsage, UsageInput

RawStepDelta = str | dict[str, Any] | StepDelta
CallableReturn = RawStepDelta | tuple[RawStepDelta, UsageInput]

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_ANYWHERE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class DeltaExtractor(Protocol):
    """Protocol for LLM-backed structured StepDelta extraction."""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta: ...


@runtime_checkable
class JsonCompletionClient(Protocol):
    """Protocol for any LLM client capable of returning a JSON string and TokenUsage."""

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]: ...


def resolve_client_model(client: Any) -> str:
    """Safely resolves a model identifier from any JsonCompletionClient or extractor."""
    model = getattr(client, "model", None) or getattr(client, "model_id", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return type(client).__name__


def validate_json_completion_client(client: Any) -> JsonCompletionClient:
    """Validates eagerly that ``client`` (and any CachedDeltaExtractor wrapper) supports call_json_api."""
    from .cached import CachedDeltaExtractor

    if not isinstance(client, JsonCompletionClient):
        raise TypeError(
            f"{type(client).__name__} does not implement JsonCompletionClient.call_json_api()."
        )
    if isinstance(client, CachedDeltaExtractor):
        validate_json_completion_client(client.extractor)
    return client


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


def _unpack_callable_result(raw: Any) -> tuple[Any, UsageInput]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw[0], raw[1]
    return raw, None


def _invoke_with_optional_kwargs(
    fn: Callable[..., Any],
    *args: Any,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if max_tokens is not None or temperature is not None:
        try:
            params: tuple[inspect.Parameter, ...] = tuple(
                inspect.signature(fn).parameters.values()
            )
        except (TypeError, ValueError):
            params = ()
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        kw_names = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if max_tokens is not None and (accepts_var_kw or "max_tokens" in kw_names):
            kwargs["max_tokens"] = max_tokens
        if temperature is not None and (accepts_var_kw or "temperature" in kw_names):
            kwargs["temperature"] = temperature
    return fn(*args, **kwargs)


def _coerce_json_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, StepDelta):
        return raw.model_dump_json()
    if isinstance(raw, (Mapping, list)):
        return json.dumps(raw, ensure_ascii=False)
    raise TypeError(f"Unsupported JSON completion payload type: {type(raw).__name__}")
