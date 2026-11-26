from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from canonarc import (
    ALL_TIERS,
    PROFILE_TIERS,
    AliasRecord,
    CallableExtractor,
    CanonArcEngine,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    SchemaPolicy,
    StepDelta,
    TierProfile,
    TranslationPreset,
    VoiceExemplar,
    compute_step_hash,
)


class _SpyEmbedder:
    """Embedder that counts every call so we can assert zero embedding work."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: int = 0

    def embed_text(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.ones(self.dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.calls += 1
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


def test_schema_policy_tier_profiles_and_validation() -> None:
    assert SchemaPolicy().profile == TierProfile.FULL
    assert SchemaPolicy().active_tiers == ALL_TIERS
    assert SchemaPolicy.lexicon_only().active_tiers == frozenset({1, 4})
    assert SchemaPolicy.state_only().active_tiers == frozenset({1, 2, 3, 4})
    assert SchemaPolicy.full().active_tiers == ALL_TIERS
    assert SchemaPolicy(profile="lexicon_only").profile is TierProfile.LEXICON_ONLY  # type: ignore[arg-type]
    assert PROFILE_TIERS[TierProfile.LEXICON_ONLY] == frozenset({1, 4})

    custom = SchemaPolicy.with_tiers({1, 2, 3, 4, 5})
    assert custom.has_tier(5) is True
    assert custom.has_tier(6) is False

    with pytest.raises(ValueError, match="Tier 1"):
        SchemaPolicy.with_tiers({2, 3, 4})

    with pytest.raises(ValueError, match="Invalid tier"):
        SchemaPolicy.with_tiers({1, 7})


def test_lexicon_only_zero_llm_zero_embedder_and_checkpoint_resumability(tmp_path: Path) -> None:
    spy_embedder = _SpyEmbedder()
    extractor_calls = 0

    def _fail_if_called(prompt: str, text: str) -> StepDelta:
        nonlocal extractor_calls
        extractor_calls += 1
        raise AssertionError("Extractor must not be called in LEXICON_ONLY profile")

    preset = TranslationPreset(policy=SchemaPolicy.lexicon_only())
    db_path = tmp_path / "nonfiction.sqlite"

    with CanonArcEngine(
        db_path,
        preset=preset,
        embedder=spy_embedder,
        extractor=CallableExtractor(_fail_if_called),
    ) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="term_gradient_descent",
                kind="term",
                canonical_name="gradient descent",
                base_attributes={
                    "target_name": "gradijentni spust",
                    "gender": "masculine",
                    "declension": "inanimate",
                },
            ),
            aliases=[
                AliasRecord(
                    alias="gradijentnog spusta",
                    entity_id="term_gradient_descent",
                    kind="permanent",
                    lang="sr",
                )
            ],
        )
        engine.register_entity(
            EntityRecord(
                entity_id="term_backprop",
                kind="term",
                canonical_name="backpropagation",
                base_attributes={"target_name": "propagacija unazad"},
            )
        )

        # prev_text mentions backpropagation, current text only mentions gradient descent:
        # LEXICON_ONLY must skip the character-halo pass over prev_text.
        chunk_text = "We optimize the loss using gradient descent."
        chunk_hash = compute_step_hash(chunk_text, model="test-model")
        with engine.step_transaction(
            1,
            text=chunk_text,
            prev_text="Earlier we introduced backpropagation.",
            content_hash=chunk_hash,
        ) as tx:
            ctx = tx.context
            assert [e.entity_id for e in ctx.entities] == ["term_gradient_descent"]
            assert ctx.active_edges == []
            assert ctx.related_entities == []
            assert ctx.background_edges == []
            assert ctx.voice_exemplars == []
            assert ctx.episodes == []

            formatted = engine.format_context(ctx)
            assert "## Terminology & glossary — binding" in formatted
            assert "target='gradijentni spust'" in formatted
            assert "[mood:" not in formatted

            # extract_and_commit must be a zero-LLM no-op that still commits the checkpoint
            delta = tx.extract_and_commit(chunk_text)
            assert delta == StepDelta()

        assert extractor_calls == 0
        assert spy_embedder.calls == 0
        assert engine.is_step_committed(1, content_hash=chunk_hash) is True
        assert engine.last_committed_step() == 1

        # Explicit calls to disabled tier writers are safe no-ops with zero embeddings
        assert engine.index_episodes(
            2, [EpisodeRecord(summary="ignored", predicate="note")]
        ) == []
        assert engine.index_voice_exemplars(
            2,
            [
                VoiceExemplar(
                    step=2,
                    speaker_id="term_gradient_descent",
                    text="ignored dialogue line",
                )
            ],
            bypass_gate=True,
        ) == []
        assert spy_embedder.calls == 0

        # apply_delta strips Tier 2/3/6 payloads while keeping Tier 1/4 glossary updates
        engine.apply_delta(
            2,
            StepDelta(
                entities=[
                    EntityDelta(
                        entity_id="term_gradient_descent",
                        state={"mood": "should_be_stripped"},
                        add_aliases=["GD"],
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="term_gradient_descent",
                        target_id="term_backprop",
                        relation_type="stance",
                        state={"register": "formal"},
                    )
                ],
                episodes=[EpisodeRecord(summary="should be dropped")],
                inform=[InformDelta(episode_id="ep_1", entity_id="term_gradient_descent")],
            ),
        )
        assert spy_embedder.calls == 0
        ctx2 = engine.retrieve_context(3, text="GD converges quickly.")
        assert len(ctx2.entities) == 1
        assert ctx2.entities[0].entity_id == "term_gradient_descent"
        assert ctx2.entities[0].state == {}
        assert ctx2.active_edges == []
        assert ctx2.episodes == []

        # render_bible defaults include_edges=False in LEXICON_ONLY
        bible_json, res = engine.render_bible(format="json")
        parsed = json.loads(bible_json)
        assert "edges" not in parsed
        assert res.edge_count == 0


def test_lexicon_only_requires_no_extractor() -> None:
    """extract_and_commit works in LEXICON_ONLY even when extractor=None."""
    preset = TranslationPreset(policy=SchemaPolicy.lexicon_only())
    with CanonArcEngine(preset=preset) as engine:
        with engine.step_transaction(1, text="Intro paragraph.", content_hash="h1") as tx:
            delta = tx.extract_and_commit("Intro paragraph.")
            assert delta == StepDelta()
        assert engine.is_step_committed(1, content_hash="h1") is True


def test_state_only_enables_tiers_1_to_4_and_prunes_extraction_schema() -> None:
    spy_embedder = _SpyEmbedder()
    captured_prompts: list[str] = []

    def _extractor(prompt: str, text: str) -> StepDelta:
        captured_prompts.append(prompt)
        return StepDelta(
            entities=[
                EntityDelta(entity_id="char_arthur", state={"mood": "furious"}),
            ],
            edges=[
                EdgeDelta(
                    source_id="char_arthur",
                    target_id="char_merlin",
                    relation_type="stance",
                    state={"register": "ti", "tone": "angry"},
                )
            ],
            # Even if LLM emits an episode, STATE_ONLY must drop it and skip embedding
            episodes=[EpisodeRecord(summary="Arthur shouted at Merlin")],
        )

    preset = TranslationPreset(policy=SchemaPolicy.state_only())
    with CanonArcEngine(
        preset=preset,
        embedder=spy_embedder,
        extractor=CallableExtractor(_extractor),
    ) as engine:
        engine.seed_entities(
            [
                EntityRecord(entity_id="char_arthur", canonical_name="Arthur"),
                EntityRecord(entity_id="char_merlin", canonical_name="Merlin"),
            ]
        )

        # build_extraction_messages prunes Tier 6 fields from JSON schema
        ctx0 = engine.retrieve_context(1, text="Arthur faced Merlin.")
        sys_prompt, user_prompt = engine.build_extraction_messages(ctx0, "Arthur faced Merlin.")
        schema_section = sys_prompt.split("### Target StepDelta JSON Schema:\n\n", 1)[1]
        schema = json.loads(schema_section)
        assert "entities" in schema["properties"]
        assert "edges" in schema["properties"]
        assert "episodes" not in schema["properties"]
        assert "inform" not in schema["properties"]
        assert "is_flashback" not in schema["properties"]
        assert "EpisodeRecord" not in schema.get("$defs", {})
        assert "InformDelta" not in schema.get("$defs", {})
        assert "### Recent Episode IDs" not in user_prompt

        with engine.step_transaction(1, text="Arthur shouted at Merlin.") as tx:
            tx.extract_and_commit(
                "Arthur shouted at Merlin.",
                exemplars=[
                    VoiceExemplar(
                        step=1,
                        speaker_id="char_arthur",
                        text="How dare you deceive me!",
                    )
                ],
                bypass_gate=True,
            )

        assert len(captured_prompts) == 1
        assert spy_embedder.calls == 0

        # Next step retrieves Tier 2 edges and Tier 3 mood, but zero voice/episodes
        ctx2 = engine.retrieve_context(2, text="Arthur looked at Merlin.")
        arthur = next(e for e in ctx2.entities if e.entity_id == "char_arthur")
        assert arthur.state["mood"] == "furious"
        assert len(ctx2.active_edges) == 1
        assert ctx2.active_edges[0].state["register"] == "ti"
        assert ctx2.voice_exemplars == []
        assert ctx2.episodes == []

        # render_bible defaults include_edges=True in STATE_ONLY
        bible_json, res = engine.render_bible(format="json")
        parsed = json.loads(bible_json)
        assert "edges" in parsed
        assert res.edge_count == 1


def test_unified_pipeline_and_mid_project_upgrade_lexicon_to_full(tmp_path: Path) -> None:
    """A single loop handles LEXICON_ONLY and FULL, and a SQLite DB can upgrade cleanly."""
    db_path = tmp_path / "book.sqlite"

    # Phase 1: Start in LEXICON_ONLY
    with CanonArcEngine(
        db_path,
        preset=TranslationPreset(policy=SchemaPolicy.lexicon_only()),
    ) as engine:
        engine.register_entity(
            EntityRecord(
                entity_id="char_lyra",
                canonical_name="Lyra",
                base_attributes={"target_name": "Lira", "gender": "feminine"},
            )
        )
        with engine.step_transaction(1, text="Lyra enters.", content_hash="s1") as tx:
            tx.extract_and_commit("Lyra enters.")
        assert engine.is_step_committed(1, content_hash="s1") is True

    # Phase 2: Reopen same SQLite DB in FULL profile without migration
    with CanonArcEngine(
        db_path,
        preset=TranslationPreset(policy=SchemaPolicy.full()),
    ) as engine:
        assert engine.is_step_committed(1, content_hash="s1") is True
        engine.commit_step(
            2,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_lyra", state={"mood": "determined"})],
                episodes=[
                    EpisodeRecord(
                        episode_id="ep_vault",
                        predicate="discovery",
                        summary="Lyra found the hidden key.",
                        participants={"char_lyra": "actor"},
                        witnesses=["char_lyra"],
                    )
                ],
            ),
            exemplars=[
                VoiceExemplar(
                    step=2,
                    speaker_id="char_lyra",
                    text="We leave at dawn, no matter the cost.",
                    metadata={"register": "formal"},
                )
            ],
            bypass_gate=True,
        )
        ctx = engine.retrieve_context(3, text="Lyra checked the hidden key.")
        assert ctx.entities[0].state["mood"] == "determined"
        assert len(ctx.episodes) == 1
        assert ctx.episodes[0].episode_id == "ep_vault"
        assert len(ctx.voice_exemplars) == 1
