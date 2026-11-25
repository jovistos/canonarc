from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from canonarc import (
    CachedDeltaExtractor,
    CallableExtractor,
    CanonArcEngine,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    OpenAICompatibleExtractor,
    StepDelta,
    TokenUsage,
)


def _seed_entities(engine: CanonArcEngine) -> None:
    engine.register_entity(
        EntityRecord(
            entity_id="char_elena",
            canonical_name="Elena",
            kind="character",
            state={"mood": "calm", "location_id": "loc_hall"},
        )
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_julian",
            canonical_name="Julian",
            kind="character",
            state={"mood": "watchful", "location_id": "loc_hall"},
        )
    )


def test_build_extraction_messages_separates_static_prefix_from_dynamic_context() -> None:
    """system_prompt holds only static rules + schema and stays identical across steps."""
    with CanonArcEngine(":memory:") as engine:
        _seed_entities(engine)

        ctx1 = engine.retrieve_context(step=1, text="Elena entered the hall.")
        sys1, user1 = engine.build_extraction_messages(ctx1, "Elena entered the hall.")

        # Commit an episode and mutate context for step 2
        engine.commit_step(
            step=1,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_elena", state={"mood": "furious"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_1",
                        step=1,
                        predicate="confrontation",
                        summary="Elena accused Julian.",
                        participants={"agent": "char_elena", "target": "char_julian"},
                        witnesses=["char_elena", "char_julian"],
                    )
                ],
            ),
        )
        ctx2 = engine.retrieve_context(step=2, text="Julian fled into the courtyard.")
        sys2, user2 = engine.build_extraction_messages(
            ctx2,
            "Julian fled into the courtyard.",
            prev_text="Elena entered the hall.",
            next_text="Marcus watched from above.",
        )

        # Static system_prompt is byte-for-byte identical across steps (prefix-cacheable)
        assert sys1 == sys2
        assert engine.preset.extraction_instructions() in sys1
        assert "### Target StepDelta JSON Schema:" in sys1
        assert '"StepDelta"' in sys1

        # Dynamic content lives strictly in user_prompt, never in system_prompt
        assert "char_elena" not in sys1
        assert "ep_1" not in sys2
        assert "Elena entered the hall." not in sys1
        assert "### Known Entity IDs" in user1
        assert "char_elena" in user1
        assert "### Target StepDelta JSON Schema:" not in user1
        assert "### Recent Episode IDs" in user2
        assert "ep_1" in user2
        assert "### Preceding Context" in user2
        assert "### Following Context" in user2

        # Single-string build_extraction_prompt preserves prefix-cache order
        combined = engine.build_extraction_prompt(ctx1, "Elena entered the hall.")
        assert combined == f"{sys1}\n\n{user1}"

        # compact_schema reduces static prefix length while preserving valid schema
        sys_compact, user_compact = engine.build_extraction_messages(
            ctx1, "Elena entered the hall.", compact_schema=True
        )
        assert len(sys_compact) < len(sys1)
        assert user_compact == user1

        # TimelineSession mirrors build_extraction_messages
        session = engine.session("main")
        assert session.build_extraction_messages(ctx1, "Elena entered the hall.") == (sys1, user1)


def test_step_transaction_routes_split_messages_to_openai_compatible_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """StepTransaction sends static system_prompt in role='system' and dynamic user_prompt in role='user'."""
    captured_payloads: list[dict[str, Any]] = []

    class _FakeHTTPResponse:
        def __init__(self, body: dict[str, Any]) -> None:
            self._raw = json.dumps(body).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self) -> _FakeHTTPResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    responses = [
        # 1st attempt: malformed JSON to verify retry preserves system_prompt
        {
            "choices": [{"message": {"content": "not valid json"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
        # 2nd attempt: valid StepDelta JSON
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "entities": [
                                    {"entity_id": "char_elena", "state": {"mood": "furious"}}
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 25,
                "total_tokens": 145,
                "prompt_cache_hit_tokens": 100,
            },
        },
    ]

    def _fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeHTTPResponse:
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeHTTPResponse(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    extractor = OpenAICompatibleExtractor(
        model="deepseek-flash",
        api_key="test-key",
        base_url="https://api.example.com/v1",
        max_retries=1,
    )

    with CanonArcEngine(":memory:", extractor=extractor) as engine:
        _seed_entities(engine)
        scene = "Elena slammed her hand on the table in fury."
        with engine.step_transaction(step=1, text=scene) as tx:
            delta = tx.extract_and_commit(scene)

        assert len(delta.entities) == 1
        assert delta.entities[0].state["mood"] == "furious"

    assert len(captured_payloads) == 2
    first_msgs = captured_payloads[0]["messages"]
    retry_msgs = captured_payloads[1]["messages"]

    assert first_msgs[0]["role"] == "system"
    assert first_msgs[1]["role"] == "user"
    assert "### Target StepDelta JSON Schema:" in first_msgs[0]["content"]
    assert "### Known Entity IDs" in first_msgs[1]["content"]
    assert "### Target StepDelta JSON Schema:" not in first_msgs[1]["content"]

    # Retry keeps system_prompt identical and appends error hint to user_prompt
    assert retry_msgs[0]["content"] == first_msgs[0]["content"]
    assert retry_msgs[1]["content"].startswith(first_msgs[1]["content"])
    assert "Previous output failed StepDelta JSON validation" in retry_msgs[1]["content"]


def test_cached_delta_extractor_forwards_and_keys_split_messages(tmp_path: Path) -> None:
    """CachedDeltaExtractor keys by per-call system_prompt and forwards split messages when supported."""
    message_calls: list[tuple[str, str, str]] = []

    class _SplitExtractor:
        def __init__(self) -> None:
            self.model = "deepseek-flash"
            self.system_prompt = "default"
            self.total_usage = TokenUsage()

        def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
            raise AssertionError("Should use extract_delta_messages when available")

        def extract_delta_messages(
            self, system_prompt: str, user_prompt: str, scene_text: str
        ) -> StepDelta:
            message_calls.append((system_prompt, user_prompt, scene_text))
            self.total_usage.add(TokenUsage(prompt_tokens=500, completion_tokens=50, total_tokens=550, api_calls=1))
            return StepDelta(scene_id="s1")

    cached = CachedDeltaExtractor(_SplitExtractor(), cache_dir=tmp_path / "cache")

    # 1st call: miss
    cached.extract_delta_messages("SYS_V1", "USER_1", "Scene A")
    assert len(message_calls) == 1
    assert message_calls[0] == ("SYS_V1", "USER_1", "Scene A")
    assert cached.cache_misses == 1
    assert cached.cache_hits == 0

    # 2nd call with identical split messages: hit
    cached.extract_delta_messages("SYS_V1", "USER_1", "Scene A")
    assert len(message_calls) == 1
    assert cached.cache_hits == 1

    # 3rd call with changed system_prompt: miss
    cached.extract_delta_messages("SYS_V2", "USER_1", "Scene A")
    assert len(message_calls) == 2
    assert cached.cache_misses == 2


def test_cached_delta_extractor_and_transaction_fallback_for_legacy_extractor(
    tmp_path: Path,
) -> None:
    """Legacy DeltaExtractors without extract_delta_messages still work directly and under CachedDeltaExtractor."""
    captured: list[str] = []

    def _legacy_fn(prompt: str, schema: dict[str, Any]) -> StepDelta:
        captured.append(prompt)
        return StepDelta(entities=[EntityDelta(entity_id="char_elena", state={"mood": "furious"})])

    legacy = CallableExtractor(_legacy_fn)
    cached_legacy = CachedDeltaExtractor(legacy, cache_dir=tmp_path / "legacy_cache")

    with CanonArcEngine(":memory:", extractor=cached_legacy) as engine:
        _seed_entities(engine)
        scene = "Elena glared across the chamber."
        with engine.step_transaction(step=1, text=scene) as tx:
            delta = tx.extract_and_commit(scene)

        assert delta.entities[0].state["mood"] == "furious"
        assert len(captured) == 1
        assert "### Target StepDelta JSON Schema:" in captured[0]
        assert "### Known Entity IDs" in captured[0]
        # Static schema appears before dynamic entity list
        assert captured[0].index("### Target StepDelta JSON Schema:") < captured[0].index(
            "### Known Entity IDs"
        )
