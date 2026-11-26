from __future__ import annotations

import json
from typing import Any

import pytest

from canonarc import (
    CachedDeltaExtractor,
    CallableEpisodeConsolidator,
    CallableExtractor,
    CanonArcEngine,
    ConsolidationSummary,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    HeuristicEpisodeConsolidator,
    SchemaPolicy,
    StepDelta,
    TimelineBranchConflictError,
    TokenUsage,
    TranslationPreset,
    build_consolidation_prompt,
)


def _seed_basic_cast(engine: CanonArcEngine) -> None:
    engine.seed_entities(
        [
            EntityRecord(entity_id="char_elena", canonical_name="Elena", kind="character"),
            EntityRecord(entity_id="char_marcus", canonical_name="Marcus", kind="character"),
            EntityRecord(entity_id="char_julian", canonical_name="Julian", kind="character"),
            EntityRecord(entity_id="char_guard", canonical_name="Watchman", kind="character"),
            EntityRecord(entity_id="loc_archive", canonical_name="Archive Vault", kind="location"),
        ],
        step=0,
    )


def test_llm_consolidation_auto_with_retry_usage_and_cache(tmp_path):
    """LLMEpisodeConsolidator retries malformed JSON, resolves entity names, tracks usage, and caches."""
    calls: list[tuple[str, str]] = []
    usage_events: list[TokenUsage] = []

    def fake_messages_fn(
        system_prompt: str, user_prompt: str, schema: dict[str, Any]
    ) -> tuple[str, dict[str, int]]:
        calls.append((system_prompt, user_prompt))
        if len(calls) == 1:
            return "not valid json at all", {"prompt_tokens": 100, "completion_tokens": 10}
        payload = {
            "summary": "Elena and Marcus completed three routine archive checks in the Archive Vault.",
            "salience": 9.5,  # should be clamped to max_salience=7.5
            "key_participant_ids": ["char_elena", "char_marcus", "char_nonexistent"],
            "location_id": "loc_archive",
        }
        return f"```json\n{json.dumps(payload)}\n```", {
            "prompt_tokens": 120,
            "completion_tokens": 45,
        }

    extractor = CallableExtractor(
        messages_fn=fake_messages_fn,
        model="sim-consolidator",
        on_usage=usage_events.append,
    )
    cached = CachedDeltaExtractor(extractor, cache_dir=tmp_path / "cache")

    with CanonArcEngine(":memory:", extractor=cached) as engine:
        _seed_basic_cast(engine)
        for s in (1, 2, 3):
            engine.index_episodes(
                step=s,
                episodes=[
                    EpisodeRecord(
                        episode_id=f"ep_patrol_{s}",
                        summary=f"Routine archive check {s}.",
                        participants={"char_elena": "actor", "char_marcus": "partner"},
                        witnesses=["char_elena", "char_marcus", "char_guard"],
                        known_by={"char_elena": "direct", "char_marcus": "direct"},
                        location_id="loc_archive",
                        chapter_id="ch_1",
                        salience=2.5,
                    )
                ],
            )
        # High-salience episode should remain untouched
        engine.index_episodes(
            step=4,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_betrayal",
                    summary="Julian stole the cipher key from the Archive Vault.",
                    participants={"char_julian": "actor"},
                    chapter_id="ch_1",
                    salience=9.0,
                )
            ],
        )

        assert engine.count_consolidatable_episodes(step=10, before_step=5, chapter_id="ch_1") == 3

        results = engine.consolidate_episodes_auto(
            step=10,
            before_step=5,
            chapter_id="ch_1",
            max_salience=4.0,
            min_episodes=3,
        )
        assert len(results) == 1
        ep = results[0]
        assert ep.summary.startswith("Elena and Marcus completed three routine archive checks")
        assert ep.salience == 7.5  # clamped
        assert ep.step == 3  # anchor_step="last_target"
        assert ep.chapter_id == "ch_1"
        assert ep.location_id == "loc_archive"
        assert set(ep.participants.keys()) == {"char_elena", "char_marcus"}
        assert "char_nonexistent" not in ep.participants
        assert set(ep.consolidated_from) == {"ep_patrol_1", "ep_patrol_2", "ep_patrol_3"}

        # Prompt included resolved canonical names
        assert "Elena (char_elena: actor)" in calls[0][1]
        assert "Archive Vault (loc_archive)" in calls[0][1]

        # Retried once on malformed JSON and tracked usage
        assert len(calls) == 2
        assert len(usage_events) == 2
        assert extractor.total_usage.prompt_tokens == 220

        remaining = {e.episode_id for e in engine.query_episodes(cutoff_step=20)}
        assert "ep_betrayal" in remaining
        assert ep.episode_id in remaining
        assert "ep_patrol_1" not in remaining

        # Running again is a zero-LLM no-op
        again = engine.consolidate_episodes_auto(step=11, before_step=5, chapter_id="ch_1")
        assert again == []
        assert len(calls) == 2


def test_consolidate_chapter_and_timeline_session():
    """consolidate_chapter only compacts the specified chapter and mirrors cleanly on TimelineSession."""
    with CanonArcEngine(
        ":memory:",
        episode_consolidator=HeuristicEpisodeConsolidator(default_salience=6.0),
    ) as engine:
        _seed_basic_cast(engine)
        for s in (1, 2, 3):
            engine.index_episodes(
                step=s,
                episodes=[
                    EpisodeRecord(
                        episode_id=f"ep_ch1_{s}",
                        summary=f"Chapter 1 walk {s}.",
                        participants={"char_elena": "actor"},
                        chapter_id="ch_1",
                        salience=2.0,
                    )
                ],
            )
        for s in (4, 5, 6):
            engine.index_episodes(
                step=s,
                episodes=[
                    EpisodeRecord(
                        episode_id=f"ep_ch2_{s}",
                        summary=f"Chapter 2 talk {s}.",
                        participants={"char_marcus": "actor"},
                        chapter_id="ch_2",
                        salience=2.0,
                    )
                ],
            )

        session = engine.session("main")
        assert session.count_consolidatable_episodes(step=6, before_step=7, chapter_id="ch_1") == 3
        compacted = session.consolidate_chapter(step=6, chapter_id="ch_1")
        assert len(compacted) == 1
        assert compacted[0].chapter_id == "ch_1"
        assert compacted[0].step == 3
        assert set(compacted[0].consolidated_from) == {"ep_ch1_1", "ep_ch1_2", "ep_ch1_3"}

        remaining_ids = {e.episode_id for e in session.query_episodes(cutoff_step=10)}
        assert {"ep_ch2_4", "ep_ch2_5", "ep_ch2_6"}.issubset(remaining_ids)
        assert "ep_ch1_1" not in remaining_ids


def test_min_episodes_noop_and_batch_chunking():
    """min_episodes avoids wasted LLM calls; max_episodes_per_batch chunks large backlogs."""
    batch_sizes: list[int] = []

    def summarize(eps: list[EpisodeRecord]) -> ConsolidationSummary:
        batch_sizes.append(len(eps))
        return ConsolidationSummary(
            summary=f"Compacted {len(eps)} episodes ({eps[0].step}..{eps[-1].step}).",
            salience=5.5,
        )

    consolidator = CallableEpisodeConsolidator(summarize)
    with CanonArcEngine(":memory:", episode_consolidator=consolidator) as engine:
        _seed_basic_cast(engine)
        for s in (1, 2):
            engine.index_episodes(
                step=s,
                episodes=[EpisodeRecord(episode_id=f"ep_{s}", summary=f"Event {s}", salience=2.0)],
            )

        # Fewer than min_episodes=3 => zero calls
        assert engine.consolidate_episodes_auto(step=10, before_step=10, min_episodes=3) == []
        assert batch_sizes == []

        # Add 5 more (total 7); with batch=3 and min_episodes=2 -> [3, 4] (trailing 1 merged)
        for s in (3, 4, 5, 6, 7):
            engine.index_episodes(
                step=s,
                episodes=[EpisodeRecord(episode_id=f"ep_{s}", summary=f"Event {s}", salience=2.0)],
            )

        out = engine.consolidate_episodes_auto(
            step=10,
            before_step=10,
            min_episodes=2,
            max_episodes_per_batch=3,
        )
        assert len(out) == 2
        assert batch_sizes == [3, 4]
        assert out[0].step == 3
        assert out[1].step == 7


def test_provenance_and_deceived_epistemic_protection():
    """Compaction skips episodes pinned by caused_by_episode_id or containing deceived POVs."""
    with CanonArcEngine(
        ":memory:",
        episode_consolidator=HeuristicEpisodeConsolidator(),
    ) as engine:
        _seed_basic_cast(engine)
        # Step 1: low-salience episode that caused an active mood state
        engine.commit_step(
            step=1,
            delta=StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="char_elena",
                        state={"mood": "suspicious"},
                        caused_by_episode_id="ep_pinned",
                    )
                ],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_pinned",
                        summary="Elena spotted a torn seal.",
                        participants={"char_elena": "actor"},
                        salience=3.0,
                    )
                ],
            ),
        )
        # Step 2: low-salience episode with a deceived observer
        engine.index_episodes(
            step=2,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_deceived",
                    summary="Julian secretly swapped the ledger.",
                    participants={"char_julian": "actor"},
                    salience=3.0,
                )
            ],
        )
        engine.inform_entity(
            step=2,
            episode_id="ep_deceived",
            entity_id="char_elena",
            epistemic="deceived",
            believed_summary="Julian filed the ledger properly.",
        )
        # Steps 3, 4: ordinary low-salience episodes with different POVs
        engine.index_episodes(
            step=3,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_plain_1",
                    summary="Elena and Marcus checked the south gate.",
                    participants={"char_elena": "actor", "char_marcus": "partner"},
                    known_by={"char_elena": "direct", "char_marcus": "direct"},
                    salience=2.0,
                )
            ],
        )
        engine.index_episodes(
            step=4,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_plain_2",
                    summary="Elena logged the shift change alone.",
                    participants={"char_elena": "actor"},
                    known_by={"char_elena": "direct"},
                    salience=2.0,
                )
            ],
        )

        # With preserve_provenance=True and preserve_deceived=True, only ep_plain_1 & ep_plain_2 match
        assert engine.count_consolidatable_episodes(step=10, before_step=10) == 2

        compacted = engine.consolidate_episodes_auto(
            step=10,
            before_step=10,
            min_episodes=2,
            epistemic_merge="intersection",
        )
        assert len(compacted) == 1
        assert set(compacted[0].consolidated_from) == {"ep_plain_1", "ep_plain_2"}
        # Intersection prevents leaking ep_plain_2 (Elena-only) to Marcus
        assert compacted[0].known_by == {"char_elena": "direct"}

        remaining_ids = {e.episode_id for e in engine.query_episodes(cutoff_step=15)}
        assert "ep_pinned" in remaining_ids
        assert "ep_deceived" in remaining_ids


def test_participant_cap_and_chronological_ranking():
    """Participant cap prevents overlap flooding and anchor_step='last_target' preserves recency."""
    with CanonArcEngine(
        ":memory:",
        episode_consolidator=HeuristicEpisodeConsolidator(max_participants=2),
    ) as engine:
        _seed_basic_cast(engine)
        for s in (1, 2, 3):
            engine.index_episodes(
                step=s,
                episodes=[
                    EpisodeRecord(
                        episode_id=f"ep_act1_{s}",
                        summary=f"Act 1 patrol {s}.",
                        participants={
                            "char_elena": "actor",
                            "char_marcus": "partner",
                            f"char_cameo_{s}": "bystander",
                        },
                        salience=2.0,
                    )
                ],
            )
        # Recent Act 5 event
        engine.index_episodes(
            step=50,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_act5_recent",
                    summary="Elena and Marcus confronted Julian in the Archive Vault.",
                    participants={"char_elena": "actor", "char_marcus": "partner"},
                    salience=6.0,
                )
            ],
        )

        compacted = engine.consolidate_episodes_auto(
            step=51,
            before_step=10,
            max_participants=2,
            anchor_step="last_target",
        )
        assert len(compacted) == 1
        assert len(compacted[0].participants) == 2
        assert set(compacted[0].participants.keys()) == {"char_elena", "char_marcus"}
        assert compacted[0].step == 3

        # Recent Act 5 episode should outrank the older Act 1 summary
        ctx = engine.retrieve_context(
            step=52,
            text="Elena and Marcus in the Archive Vault",
            mentions=["Elena", "Marcus"],
            top_k_episodes=2,
        )
        assert ctx.episodes[0].episode_id == "ep_act5_recent"


def test_branch_safety_and_tier6_disabled_noop():
    """Consolidation blocks corrupting child branches unless force=True, and no-ops when Tier 6 is disabled."""
    called = False

    def fail_if_called(eps: list[EpisodeRecord]) -> str:
        nonlocal called
        called = True
        return "should not run"

    with CanonArcEngine(
        ":memory:",
        episode_consolidator=CallableEpisodeConsolidator(fail_if_called),
    ) as engine:
        _seed_basic_cast(engine)
        for s in (1, 2, 3):
            engine.index_episodes(
                step=s,
                episodes=[EpisodeRecord(episode_id=f"ep_{s}", summary=f"S{s}", salience=2.0)],
            )
        engine.branch(at_step=5, name="alt_branch")

        with pytest.raises(TimelineBranchConflictError):
            engine.consolidate_episodes_auto(step=10, before_step=5, min_episodes=3)
        # Branch check happens BEFORE spending LLM tokens
        assert called is False

        # force=True overrides
        res = engine.consolidate_episodes_auto(step=10, before_step=5, min_episodes=3, force=True)
        assert len(res) == 1
        assert called is True

    # Tier 6 disabled (STATE_ONLY) => clean no-op without IndexError
    with CanonArcEngine(
        ":memory:",
        preset=TranslationPreset(policy=SchemaPolicy.state_only()),
        episode_consolidator=HeuristicEpisodeConsolidator(),
    ) as engine_no_t6:
        assert engine_no_t6.count_consolidatable_episodes(step=10, before_step=5) == 0
        assert (
            engine_no_t6.consolidate_episodes(step=10, before_step=5, summary="noop")
            is None
        )
        assert engine_no_t6.consolidate_episodes_auto(step=10, before_step=5) == []
        assert engine_no_t6.consolidate_chapter(step=10, chapter_id="ch_1") == []


def test_build_consolidation_prompt_format():
    """build_consolidation_prompt includes chapter header, step metadata, and resolved entity names."""
    ep = EpisodeRecord(
        episode_id="ep_1",
        step=4,
        predicate="patrol",
        summary="Checked the vault lock.",
        participants={"char_elena": "actor"},
        location_id="loc_archive",
        chapter_id="ch_1",
        scene_id="sc_2",
        salience=2.5,
    )
    sys_prompt, user_prompt = build_consolidation_prompt(
        [ep],
        entity_names={"char_elena": "Elena", "loc_archive": "Archive Vault"},
        chapter_id="ch_1",
    )
    assert "JSON" in sys_prompt
    assert "Chapter: ch_1" in user_prompt
    assert "step=4, scene=sc_2, type=patrol, salience=2.5" in user_prompt
    assert "Elena (char_elena: actor)" in user_prompt
    assert "Archive Vault (loc_archive)" in user_prompt
