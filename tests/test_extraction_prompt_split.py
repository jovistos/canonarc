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


def test_known_entity_scope_compact_cast_roster_modes() -> None:
    """known_entity_scope appends a terse cast/entity roster for out-of-scene entities without leaking future steps."""
    with CanonArcEngine(":memory:") as engine:
        # Step 0 seeded cast + location + faction
        engine.register_entity(
            EntityRecord(
                entity_id="char_elena",
                canonical_name="Elena",
                kind="character",
                base_attributes={"target_name": "Jelena"},
                state={"location_id": "loc_hall"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_ivan",
                canonical_name="Ivan",
                kind="character",
                base_attributes={"target_name": "Ivan"},  # same as canonical -> no duplicate slash
                state={"location_id": "loc_docks"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="char_marcus",
                canonical_name="Marcus",
                kind="character",
                base_attributes={"target_name": "Markus"},
                state={"location_id": "loc_tower"},
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="loc_docks",
                canonical_name="Old Docks",
                kind="location",
            ),
            step=0,
        )
        engine.register_entity(
            EntityRecord(
                entity_id="fac_guild",
                canonical_name="Ferrymen Guild",
                kind="faction",
            ),
            step=0,
        )
        # Future character introduced at step 5 (must not leak into step 2)
        engine.register_entity(
            EntityRecord(
                entity_id="char_future",
                canonical_name="Future Stranger",
                kind="character",
            ),
            step=5,
        )

        scene = "Elena heard the old ferryman cough in the dark."
        ctx = engine.retrieve_context(step=2, text=scene)

        # 1. Default 'scene' scope: only Elena (in-scene), no compact roster
        sys_scene, user_scene = engine.build_extraction_messages(ctx, scene)
        assert "char_elena" in user_scene
        assert "char_ivan" not in user_scene
        assert "char_marcus" not in user_scene
        assert "Other Known" not in user_scene

        # 2. 'characters_compact': adds 1-line roster of out-of-scene characters only
        sys_chars, user_chars = engine.build_extraction_messages(
            ctx, scene, known_entity_scope="characters_compact"
        )
        assert sys_chars == sys_scene  # static system_prompt unchanged (prefix cache safe)
        assert "### Other Known Characters" in user_chars
        assert "char_ivan=Ivan, char_marcus=Marcus/Markus" in user_chars
        # In-scene Elena is not duplicated in the compact roster
        roster_section = user_chars.split("### Other Known Characters", 1)[1].split("### ", 1)[0]
        assert "char_elena=" not in roster_section
        # Non-characters and future entities are excluded
        assert "loc_docks" not in user_chars
        assert "fac_guild" not in user_chars
        assert "char_future" not in user_chars

        # 3. 'all_compact': includes characters + locations + factions grouped by kind
        sys_all, user_all = engine.build_extraction_messages(
            ctx, scene, known_entity_scope="all_compact"
        )
        assert sys_all == sys_scene
        assert "### Other Known Entities" in user_all
        assert "character: char_ivan=Ivan, char_marcus=Marcus/Markus" in user_all
        assert "faction: fac_guild=Ferrymen Guild" in user_all
        assert "location: loc_docks=Old Docks" in user_all
        assert "char_future" not in user_all

        # 4. TimelineSession and build_extraction_prompt forward the option
        session = engine.session("main")
        assert session.build_extraction_messages(
            ctx, scene, known_entity_scope="characters_compact"
        ) == (sys_chars, user_chars)
        assert engine.build_extraction_prompt(
            ctx, scene, known_entity_scope="characters_compact"
        ) == f"{sys_chars}\n\n{user_chars}"

        # 5. max_roster_entities cap & validation
        _, user_capped = engine.build_extraction_messages(
            ctx, scene, known_entity_scope="characters_compact", max_roster_entities=1
        )
        capped_line = user_capped.split("### Other Known Characters", 1)[1].splitlines()[1]
        assert len(capped_line.split(", ")) == 1

        _, user_zero = engine.build_extraction_messages(
            ctx, scene, known_entity_scope="characters_compact", max_roster_entities=0
        )
        assert "### Other Known Characters" not in user_zero

        with pytest.raises(ValueError, match="Unsupported known_entity_scope"):
            engine.build_extraction_messages(ctx, scene, known_entity_scope="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_roster_entities"):
            engine.build_extraction_messages(
                ctx, scene, known_entity_scope="characters_compact", max_roster_entities=-1
            )


def test_step_transaction_forwards_known_entity_scope() -> None:
    """StepTransaction.extract_and_commit forwards known_entity_scope to build_extraction_messages."""
    captured_prompts: list[str] = []

    def _extractor_fn(prompt: str, schema: dict[str, Any]) -> StepDelta:
        captured_prompts.append(prompt)
        return StepDelta(entities=[EntityDelta(entity_id="char_ivan", state={"mood": "weary"})])

    with CanonArcEngine(":memory:", extractor=CallableExtractor(_extractor_fn)) as engine:
        engine.register_entity(EntityRecord(entity_id="char_elena", canonical_name="Elena", kind="character"))
        engine.register_entity(EntityRecord(entity_id="char_ivan", canonical_name="Ivan", kind="character"))

        scene = "Elena spotted the old ferryman."
        with engine.step_transaction(step=1, text=scene) as tx:
            delta = tx.extract_and_commit(scene, known_entity_scope="characters_compact")

        assert delta.entities[0].entity_id == "char_ivan"
        assert len(captured_prompts) == 1
        assert "### Other Known Characters" in captured_prompts[0]
        assert "char_ivan=Ivan" in captured_prompts[0]


def test_callable_extractor_messages_fn_preserves_split_prefix_and_bridges_usage(
    tmp_path: Path,
) -> None:
    """CallableExtractor(messages_fn=...) preserves split prompts across retries and bridges usage into CachedDeltaExtractor."""
    captured_calls: list[tuple[str, str, dict[str, Any]]] = []
    forwarded_usage: list[TokenUsage] = []

    responses: list[tuple[Any, dict[str, Any]]] = [
        # Attempt 1: invalid JSON, LangChain-style usage_metadata
        (
            "not valid json",
            {
                "input_tokens": 1200,
                "output_tokens": 20,
                "total_tokens": 1220,
                "input_token_details": {"cache_read": 0},
            },
        ),
        # Attempt 2: valid StepDelta, prefix cache hit + reasoning tokens
        (
            StepDelta(entities=[EntityDelta(entity_id="char_elena", state={"mood": "furious"})]),
            {
                "input_tokens": 1250,
                "output_tokens": 80,
                "total_tokens": 1330,
                "input_token_details": {"cache_read": 1000},
                "output_token_details": {"reasoning": 30},
            },
        ),
    ]

    def _messages_fn(
        system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        captured_calls.append((system_prompt, user_prompt, schema))
        return responses.pop(0)

    extractor = CallableExtractor(
        messages_fn=_messages_fn,
        model="gemini-2.5-flash",
        max_retries=1,
        on_usage=forwarded_usage.append,
    )
    cached = CachedDeltaExtractor(extractor, cache_dir=tmp_path / "callable_cache")
    assert cached.model_id == "gemini-2.5-flash"

    with CanonArcEngine(":memory:", extractor=cached) as engine:
        _seed_entities(engine)
        scene = "Elena slammed her hand on the table in fury."
        with engine.step_transaction(step=1, text=scene) as tx:
            delta = tx.extract_and_commit(scene)

        assert len(delta.entities) == 1
        assert delta.entities[0].state["mood"] == "furious"

    # Two attempts: split messages preserved, system_prompt byte-identical on retry
    assert len(captured_calls) == 2
    sys_1, user_1, schema_1 = captured_calls[0]
    sys_2, user_2, _ = captured_calls[1]
    assert "### Target StepDelta JSON Schema:" in sys_1
    assert "### Known Entity IDs" not in sys_1
    assert "### Known Entity IDs" in user_1
    assert "### Target StepDelta JSON Schema:" not in user_1
    assert "properties" in schema_1
    assert sys_2 == sys_1
    assert user_2.startswith(user_1)
    assert "Previous output failed StepDelta JSON validation" in user_2

    # Usage accumulated across retry and bridged to last_usage, total_usage, on_usage, and CachedDeltaExtractor
    assert extractor.last_usage.prompt_tokens == 2450
    assert extractor.last_usage.completion_tokens == 100
    assert extractor.last_usage.total_tokens == 2550
    assert extractor.last_usage.cached_prompt_tokens == 1000
    assert extractor.last_usage.reasoning_tokens == 30
    assert extractor.last_usage.api_calls == 2
    assert extractor.last_usage.retry_count == 1
    assert extractor.last_usage.validation_errors == 1
    assert extractor.total_usage.prompt_tokens == 2450

    assert len(forwarded_usage) == 1
    assert forwarded_usage[0].cached_prompt_tokens == 1000

    assert cached.cache_misses == 1
    assert cached.billable_usage.prompt_tokens == 2450
    assert cached.billable_usage.cached_prompt_tokens == 1000
    assert cached.billable_usage.reasoning_tokens == 30
    assert cached.billable_usage.api_calls == 2
    assert cached.billable_usage.retry_count == 1
    assert cached.billable_usage.validation_errors == 1
    assert cached.billable_usage.estimate_cost_usd(model=cached.model_id)["total_cost_usd"] > 0

    # Disk cache hit on identical split call does not re-invoke messages_fn or on_usage
    cached_delta = cached.extract_delta_messages(sys_1, user_1, scene)
    assert cached_delta.entities[0].state["mood"] == "furious"
    assert len(captured_calls) == 2
    assert len(forwarded_usage) == 1
    assert cached.cache_hits == 1
    assert cached.historical_usage.prompt_tokens == 4900


def test_callable_extractor_usage_formats_and_validation() -> None:
    """Supports Anthropic usage dicts, TokenUsage returns, direct extract_delta with messages_fn, and validates args."""
    with pytest.raises(ValueError, match="at least one of 'fn' or 'messages_fn'"):
        CallableExtractor()

    # Anthropic-style usage dict via legacy 2-arg fn
    def _anthropic_fn(prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"scene_id": "s_anth"},
            {
                "input_tokens": 900,
                "output_tokens": 120,
                "cache_read_input_tokens": 700,
            },
        )

    ext_anth = CallableExtractor(_anthropic_fn)
    delta_anth = ext_anth.extract_delta("prompt", "scene")
    assert delta_anth.scene_id == "s_anth"
    assert ext_anth.last_usage.prompt_tokens == 900
    assert ext_anth.last_usage.completion_tokens == 120
    assert ext_anth.last_usage.total_tokens == 1020
    assert ext_anth.last_usage.cached_prompt_tokens == 700

    # TokenUsage instance return + messages_fn called via direct extract_delta()
    seen: list[tuple[str, str]] = []

    def _msg_only(sys_p: str, user_p: str, schema: dict[str, Any]) -> tuple[StepDelta, TokenUsage]:
        seen.append((sys_p, user_p))
        return StepDelta(scene_id="s_tok"), TokenUsage(
            prompt_tokens=400,
            completion_tokens=50,
            total_tokens=450,
            cached_prompt_tokens=300,
        )

    ext_msg = CallableExtractor(messages_fn=_msg_only, system_prompt="STATIC_SYS")
    delta_msg = ext_msg.extract_delta("DYNAMIC_USER", "scene")
    assert delta_msg.scene_id == "s_tok"
    assert seen == [("STATIC_SYS", "DYNAMIC_USER")]
    assert ext_msg.total_usage.prompt_tokens == 400
    assert ext_msg.total_usage.cached_prompt_tokens == 300
    assert ext_msg.total_usage.api_calls == 1
