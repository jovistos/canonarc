from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from ..models import StepDelta
from .protocols import (
    CallableReturn,
    _coerce_json_content,
    _coerce_step_delta,
    _invoke_with_optional_kwargs,
    _unpack_callable_result,
)
from .usage import TokenUsage, _coerce_usage


class CallableExtractor:
    """Adapter turning any (prompt, schema) or (system_prompt, user_prompt, schema) callable into a DeltaExtractor and JsonCompletionClient."""

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

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple[str, TokenUsage]:
        """Executes a single JSON completion through fn/messages_fn and returns (content_str, usage)."""
        eff_schema: dict[str, Any] = {} if schema is None else schema
        eff_system = self.system_prompt if system_prompt is None else system_prompt

        t0 = time.perf_counter()
        if self.messages_fn is not None:
            raw_result = _invoke_with_optional_kwargs(
                self.messages_fn,
                eff_system,
                prompt,
                eff_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            assert self.fn is not None
            combined_prompt = f"{eff_system}\n\n{prompt}" if eff_system else prompt
            raw_result = _invoke_with_optional_kwargs(
                self.fn,
                combined_prompt,
                eff_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_payload, raw_usage = _unpack_callable_result(raw_result)
        usage = _coerce_usage(raw_usage, latency_ms=elapsed_ms)
        self.last_usage = usage
        self.total_usage.add(usage)
        if self.on_usage is not None:
            self.on_usage(TokenUsage.from_dict(usage.to_dict()))
        content = _coerce_json_content(raw_payload)
        return content, usage
