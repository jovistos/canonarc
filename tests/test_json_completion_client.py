from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self
from unittest.mock import patch

import pytest

from canonarc import (
    CachedDeltaExtractor,
    CallableExtractor,
    DialogueCandidate,
    HeuristicVoiceQualityScorer,
    InflectionRequest,
    JsonCompletionClient,
    LLMInflectionGenerator,
    LLMVoiceQualityScorer,
    OpenAICompatibleExtractor,
    StepDelta,
    TokenUsage,
)


class _DeltaOnlyExtractor:
    """Minimal DeltaExtractor that does NOT implement call_json_api."""

    def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
        return StepDelta()


class _MinimalJsonClient:
    """Minimal custom JsonCompletionClient with no .model attribute."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def call_json_api(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, TokenUsage]:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self.payload, TokenUsage(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
            api_calls=1,
        )


def test_json_completion_client_protocol_checks(tmp_path: Path) -> None:
    callable_ext = CallableExtractor(lambda prompt, schema: "{}")
    openai_ext = OpenAICompatibleExtractor(model="test-model")
    cached_ext = CachedDeltaExtractor(callable_ext, tmp_path / "delta_cache")
    minimal_client = _MinimalJsonClient("{}")

    assert isinstance(callable_ext, JsonCompletionClient)
    assert isinstance(openai_ext, JsonCompletionClient)
    assert isinstance(cached_ext, JsonCompletionClient)
    assert isinstance(minimal_client, JsonCompletionClient)
    assert not isinstance(_DeltaOnlyExtractor(), JsonCompletionClient)


def test_callable_extractor_call_json_api_messages_fn_and_kwargs_forwarding() -> None:
    seen: list[dict[str, Any]] = []
    usage_events: list[TokenUsage] = []

    def messages_fn(
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        seen.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return {"ok": True}, {"input_tokens": 30, "output_tokens": 10}

    ext = CallableExtractor(
        messages_fn=messages_fn,
        model="claude-test",
        on_usage=usage_events.append,
    )

    content, usage = ext.call_json_api(
        "Return JSON",
        system_prompt="Be strict",
        max_tokens=256,
        temperature=0.0,
    )

    assert json.loads(content) == {"ok": True}
    assert seen == [
        {
            "system_prompt": "Be strict",
            "user_prompt": "Return JSON",
            "schema": {},
            "max_tokens": 256,
            "temperature": 0.0,
        }
    ]
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 40
    assert ext.total_usage.total_tokens == 40
    assert len(usage_events) == 1
    assert usage_events[0].total_tokens == 40


def test_callable_extractor_call_json_api_fn_ignores_unaccepted_kwargs() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def plain_fn(prompt: str, schema: dict[str, Any]) -> str:
        seen.append((prompt, schema))
        return '{"forms": ["Kael", "Kaela"]}'

    ext = CallableExtractor(plain_fn)
    content, usage = ext.call_json_api(
        "Generate forms",
        system_prompt="Morphology engine",
        max_tokens=128,
        temperature=0.2,
    )

    assert json.loads(content) == {"forms": ["Kael", "Kaela"]}
    assert seen == [("Morphology engine\n\nGenerate forms", {})]
    assert usage.api_calls == 1


def test_cached_delta_extractor_passthrough_and_eager_validation(tmp_path: Path) -> None:
    usage_events: list[TokenUsage] = []
    base = CallableExtractor(
        messages_fn=lambda sys_p, usr_p, schema: (
            {"forms": ["Vuk", "Vuka", "Vuku"]},
            {"prompt_tokens": 20, "completion_tokens": 5},
        ),
        model="gemini-custom",
        on_usage=usage_events.append,
    )
    cached = CachedDeltaExtractor(base, tmp_path / "delta_cache")
    assert cached.model == "gemini-custom"

    inflector = LLMInflectionGenerator(cached, cache_dir=tmp_path / "inflections")
    forms = inflector.generate_inflections("Vuk", "sr", gender="masculine")
    assert "Vuka" in forms
    assert inflector.api_calls == 1
    assert inflector.total_usage.total_tokens == 25
    assert len(usage_events) == 1

    # Eager validation rejects plain DeltaExtractor and CachedDeltaExtractor wrapping one
    delta_only = _DeltaOnlyExtractor()
    cached_delta_only = CachedDeltaExtractor(delta_only, tmp_path / "delta_only_cache")

    with pytest.raises(TypeError, match="JsonCompletionClient"):
        LLMInflectionGenerator(delta_only)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JsonCompletionClient"):
        LLMInflectionGenerator(cached_delta_only)
    with pytest.raises(TypeError, match="JsonCompletionClient"):
        LLMVoiceQualityScorer(delta_only)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JsonCompletionClient"):
        LLMVoiceQualityScorer(cached_delta_only)


def test_llm_inflection_generator_with_minimal_client_no_model_attr(tmp_path: Path) -> None:
    client = _MinimalJsonClient(
        json.dumps({"items": [{"id": 0, "forms": ["Mira", "Mire"]}, {"id": 1, "forms": ["Vuk", "Vuka"]}]})
    )
    gen = LLMInflectionGenerator(client, cache_dir=tmp_path / "inflections", max_tokens=100)

    batch_out = gen.generate_inflections_batch(
        [
            InflectionRequest(name="Mira", lang="sr", gender="feminine"),
            InflectionRequest(name="Vuk", lang="sr", gender="masculine"),
        ]
    )
    assert "Mire" in batch_out[0]
    assert "Vuka" in batch_out[1]
    assert gen.api_calls == 1
    assert gen.total_usage.total_tokens == 20
    assert client.calls[0]["max_tokens"] == 200

    # Disk cache hit on a fresh generator instance should make 0 API calls
    gen2 = LLMInflectionGenerator(client, cache_dir=tmp_path / "inflections")
    cached_out = gen2.generate_inflections("Mira", "sr", gender="feminine")
    assert "Mire" in cached_out
    assert gen2.api_calls == 0
    assert gen2.cache_hits == 1
    assert gen2.total_usage.total_tokens == 0


def test_llm_voice_quality_scorer_with_callable_extractor_and_fallback() -> None:
    usage_events: list[TokenUsage] = []
    attempts = {"count": 0}

    def messages_fn(system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> tuple[str, dict[str, int]]:
        attempts["count"] += 1
        assert schema == {}
        if attempts["count"] == 1:
            return "not-valid-json", {"prompt_tokens": 40, "completion_tokens": 5}
        return (
            json.dumps(
                {
                    "items": [
                        {
                            "candidate_id": "c1",
                            "score": 8.7,
                            "reason": "distinctive register",
                            "tone": "dry",
                        }
                    ]
                }
            ),
            {"prompt_tokens": 40, "completion_tokens": 25},
        )

    client = CallableExtractor(messages_fn=messages_fn, on_usage=usage_events.append)
    scorer = LLMVoiceQualityScorer(
        client,
        max_retries=1,
        fallback_scorer=HeuristicVoiceQualityScorer(),
    )
    cand = DialogueCandidate(
        candidate_id="c1",
        speaker_id="kael",
        addressee_id="mira",
        text="You really think the council will listen to us?",
        word_count=9,
        distinctiveness_score=0.8,
        lexical_diversity=0.9,
    )
    res = scorer.score_candidate(cand, metadata={"register": "informal"})

    assert res.score == pytest.approx(8.7)
    assert res.metadata_updates["tone"] == "dry"
    assert scorer.api_calls == 2
    assert scorer.total_usage.prompt_tokens == 80
    assert scorer.total_usage.completion_tokens == 30
    assert scorer.total_usage.total_tokens == 110
    assert len(usage_events) == 2
    assert sum(u.total_tokens for u in usage_events) == 110


def test_unified_callable_extractor_across_all_three_subsystems(tmp_path: Path) -> None:
    usage_events: list[TokenUsage] = []
    seen_schemas: list[bool] = []

    def unified_router(
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> tuple[str, dict[str, int]]:
        seen_schemas.append(bool(schema))
        if schema:
            # DeltaExtractor path receives StepDelta JSON schema
            assert "properties" in schema
            return (
                json.dumps({"entities": [], "edges": [], "informs": []}),
                {"prompt_tokens": 100, "completion_tokens": 20},
            )
        if "morphological inflection" in system_prompt.lower():
            return (
                json.dumps({"forms": ["Kael", "Kaela"]}),
                {"prompt_tokens": 50, "completion_tokens": 10},
            )
        return (
            json.dumps({"items": [{"candidate_id": "c1", "score": 9.1, "reason": "sharp voice"}]}),
            {"prompt_tokens": 60, "completion_tokens": 15},
        )

    client = CallableExtractor(
        messages_fn=unified_router,
        model="unified-sdk-model",
        on_usage=usage_events.append,
    )

    # 1. StepDelta extraction
    delta = client.extract_delta_messages("Extract state", "Kael enters.", "Kael enters.")
    assert isinstance(delta, StepDelta)

    # 2. Inflections
    inflector = LLMInflectionGenerator(client, cache_dir=tmp_path / "inflections")
    forms = inflector.generate_inflections("Kael", "sr")
    assert "Kaela" in forms

    # 3. Gate-2 voice scoring
    scorer = LLMVoiceQualityScorer(client)
    cand = DialogueCandidate(
        candidate_id="c1",
        speaker_id="kael",
        addressee_id="mira",
        text="Hold the line.",
        word_count=3,
        distinctiveness_score=0.75,
        lexical_diversity=1.0,
    )
    score = scorer.score_candidate(cand, metadata={"register": "informal"})
    assert score.score == pytest.approx(9.1)

    # Verify schema distinction and unified + per-subsystem accounting
    assert seen_schemas == [True, False, False]
    assert len(usage_events) == 3
    assert sum(u.total_tokens for u in usage_events) == 255
    assert client.total_usage.total_tokens == 255
    assert inflector.total_usage.total_tokens == 60
    assert scorer.total_usage.total_tokens == 75


def test_openai_compatible_extractor_on_usage_callback() -> None:
    usage_events: list[TokenUsage] = []

    class _FakeResp:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "choices": [{"message": {"content": '{"forms": ["A", "B"]}'}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22},
                }
            ).encode("utf-8")

    ext = OpenAICompatibleExtractor(
        model="gpt-4o-mini",
        api_key="test-key",
        on_usage=usage_events.append,
    )
    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        content, usage = ext.call_json_api("Generate forms")

    assert json.loads(content) == {"forms": ["A", "B"]}
    assert usage.total_tokens == 22
    assert len(usage_events) == 1
    assert usage_events[0].total_tokens == 22
