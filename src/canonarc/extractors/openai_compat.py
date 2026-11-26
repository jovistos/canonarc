from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..models import StepDelta
from .callable import CallableExtractor
from .protocols import CallableReturn
from .usage import TokenUsage, load_dotenv


def resolve_lmstudio_model(
    base_url: str = "http://localhost:1234/v1",
    *,
    timeout: float = 5.0,
) -> str:
    """Auto-discovers the currently loaded LLM/VLM in LM Studio without requiring a model name."""
    clean_base = base_url.rstrip("/")
    host_root = clean_base.removesuffix("/v1")

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
        on_usage: Callable[[TokenUsage], None] | None = None,
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
        self.on_usage = on_usage
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
        """Creates an OpenAICompatibleExtractor for LM Studio."""
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
        if self.on_usage is not None:
            self.on_usage(TokenUsage.from_dict(usage.to_dict()))
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
