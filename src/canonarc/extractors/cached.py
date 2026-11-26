from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..models import StepDelta
from .protocols import DeltaExtractor, validate_json_completion_client
from .usage import TokenUsage


class CachedDeltaExtractor:
    """Content-addressed disk-caching wrapper around any DeltaExtractor."""

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
        self.billable_usage: TokenUsage = TokenUsage()
        self.historical_usage: TokenUsage = TokenUsage()
        self.step_usage_log: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        """Model identifier alias matching OpenAICompatibleExtractor / CallableExtractor."""
        return self.model_id

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]:
        """Passes generic JSON completions through to the wrapped JsonCompletionClient."""
        client = validate_json_completion_client(self.extractor)
        return client.call_json_api(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

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
                pass

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
