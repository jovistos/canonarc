from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Callable, Protocol

from .models import StepDelta


class DeltaExtractor(Protocol):
    """Protocol for LLM-backed structured StepDelta extraction."""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta: ...


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def _strip_json_fences(raw: str) -> str:
    stripped = raw.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
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


class CallableExtractor:
    """Adapter turning any (prompt, schema) -> str | dict | StepDelta callable into a DeltaExtractor.

    Automatically strips Markdown JSON fences, validates against StepDelta, and supports
    automatic retries or custom repair callbacks on malformed responses.
    """

    def __init__(
        self,
        fn: Callable[[str, dict[str, Any]], str | dict[str, Any] | StepDelta],
        *,
        max_retries: int = 1,
        repair_fn: Callable[[str, Exception], str | dict[str, Any]] | None = None,
    ) -> None:
        self.fn = fn
        self.max_retries = max(0, max_retries)
        self.repair_fn = repair_fn

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        schema = StepDelta.model_json_schema()
        current_prompt = prompt
        last_exc: Exception | None = None
        last_raw_str = ""

        for attempt in range(self.max_retries + 1):
            raw: str | dict[str, Any] | StepDelta
            if attempt > 0 and self.repair_fn is not None and last_exc is not None:
                raw = self.repair_fn(last_raw_str, last_exc)
            else:
                raw = self.fn(current_prompt, schema)
            last_raw_str = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            try:
                return _coerce_step_delta(raw)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                current_prompt = (
                    f"{prompt}\n\n"
                    f"Previous output failed StepDelta JSON validation with error:\n{exc}\n"
                    "Return ONLY valid JSON matching the StepDelta schema."
                )
        assert last_exc is not None
        raise last_exc


class OpenAICompatibleExtractor:
    """Zero-dependency HTTP extractor for OpenAI, LiteLLM, vLLM, or Ollama /v1/chat/completions."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 1,
        system_prompt: str = (
            "You are a precise narrative state extraction engine. "
            "Respond only with valid JSON conforming to the requested StepDelta schema."
        ),
        headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.system_prompt = system_prompt
        self.headers = dict(headers) if headers else {}

    def _call_api(self, prompt: str, schema: dict[str, Any]) -> str:  # noqa: ARG002
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            req_headers.setdefault("Authorization", f"Bearer {self.api_key}")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=req_headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible endpoint returned no choices.")
        return str(choices[0].get("message", {}).get("content", ""))

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        helper = CallableExtractor(self._call_api, max_retries=self.max_retries)
        return helper.extract_delta(prompt, scene_text)
