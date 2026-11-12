import subprocess
import sys

import numpy as np
import pytest

from narrative_memory import (
    AliasRecord,
    CausalStore,
    DefaultLexicalResolver,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    NarrativeMemoryEngine,
    OfflineNgramEmbedder,
    ResolvedMention,
    SchemaPolicy,
    SimulationPreset,
    StepDelta,
    TimelineBranchConflictError,
    VoiceBank,
    VoiceExemplar,
)


def test_custom_store_injection():
    custom_store = CausalStore(":memory:")
    engine = NarrativeMemoryEngine(store=custom_store)
    assert engine.store is custom_store
    engine.register_entity(EntityRecord(entity_id="a1", canonical_name="Trader Alice"), step=0)
    ctx = engine.retrieve_context(step=1, mentions=["Trader Alice"])
    assert len(ctx.entities) == 1
    assert ctx.entities[0].entity_id == "a1"
    engine.close()


def test_schema_policy_domain_keys_and_sparse_fallback():
    policy = SchemaPolicy(
        hard_filter_keys=("channel",),
        soft_boost_weights={"urgency": 0.25, "stance": 0.15},
        fallback_on_empty_hard_filter=True,
        hard_filter_mismatch_penalty=0.2,
    )
    preset = SimulationPreset(policy=policy)
    engine = NarrativeMemoryEngine(":memory:", preset=preset)

    engine.register_entity(EntityRecord(entity_id="agent_risk", canonical_name="RiskOfficer"), step=0)
    engine.register_entity(EntityRecord(entity_id="agent_desk", canonical_name="TradingDesk"), step=0)
    engine.commit_step(
        step=1,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="agent_risk",
                    target_id="agent_desk",
                    relation_type="stance",
                    state={"channel": "slack", "urgency": "high", "stance": "hawkish"},
                )
            ]
        ),
        exemplars=[
            VoiceExemplar(
                step=1,
                speaker_id="agent_risk",
                text="Escalate margin call immediately before market close.",
                metadata={"channel": "email", "urgency": "low", "stance": "dovish"},
            ),
            VoiceExemplar(
                step=1,
                speaker_id="agent_risk",
                text="Cut exposure on desk alpha right now.",
                metadata={"channel": "slack", "urgency": "high", "stance": "hawkish"},
            ),
        ],
        bypass_gate=True,
    )

    ctx = engine.retrieve_context(step=2, text="RiskOfficer pinged TradingDesk", top_k_voice=2)
    assert len(ctx.voice_exemplars) == 1
    assert "Cut exposure" in ctx.voice_exemplars[0].text

    # Now switch channel to 'phone' where 0 exemplars exist -> fallback_on_empty_hard_filter prevents starvation
    engine.commit_step(
        step=2,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="agent_risk",
                    target_id="agent_desk",
                    relation_type="stance",
                    state={"channel": "phone"},
                )
            ]
        ),
    )
    ctx_fallback = engine.retrieve_context(step=3, text="RiskOfficer called TradingDesk", top_k_voice=1)
    assert len(ctx_fallback.voice_exemplars) == 1


def test_word_boundary_and_fuzzy_mention_resolver():
    engine = NarrativeMemoryEngine(
        ":memory:",
        mention_resolver=DefaultLexicalResolver(fuzzy_threshold=85.0),
    )
    engine.register_entity(EntityRecord(entity_id="char_art", canonical_name="Art"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_artur", canonical_name="Artur"), step=0)

    # "started" must NOT match "Art"
    ctx_false = engine.retrieve_context(
        step=1,
        text="The storm started at midnight.",
        fallback_on_empty_mentions="none",
    )
    assert ctx_false.resolved_mentions == []
    assert ctx_false.entities == []

    # Inflected form "Artura" should fuzzy-match "Artur"
    ctx_inflected = engine.retrieve_context(
        step=1,
        text="Videli smo Artura na trgu.",
        fallback_on_empty_mentions="none",
    )
    assert [m.entity_id for m in ctx_inflected.resolved_mentions] == ["char_artur"]
    assert ctx_inflected.resolved_mentions[0].matched_via == "fuzzy"

    # Default engine (without explicit fallback_on_empty_mentions) must NOT dump all entities on empty match
    default_engine = NarrativeMemoryEngine(":memory:")
    default_engine.register_entity(EntityRecord(entity_id="char_art", canonical_name="Art"), step=0)
    default_engine.register_entity(EntityRecord(entity_id="char_zh", canonical_name="李雷"), step=0)

    ctx_empty = default_engine.retrieve_context(step=1, text="The storm started at midnight.")
    assert ctx_empty.entities == []

    # CJK entity without spaces around it should still match
    ctx_zh = default_engine.retrieve_context(step=1, text="李雷去了学校。")
    assert [e.entity_id for e in ctx_zh.entities] == ["char_zh"]
    default_engine.close()


def test_custom_mention_resolver_and_pov_fallback():
    class StubNERResolver:
        def resolve(self, text, explicit_mentions, entities, aliases):
            return [ResolvedMention(surface_text="he", entity_id="char_bob", matched_via="alias")]

    engine = NarrativeMemoryEngine(":memory:", mention_resolver=StubNERResolver())
    engine.register_entity(EntityRecord(entity_id="char_alice", canonical_name="Alice"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_bob", canonical_name="Bob"), step=0)

    ctx = engine.retrieve_context(step=1, text="He nodded.")
    assert [e.entity_id for e in ctx.entities] == ["char_bob"]

    # Test pov_only fallback when resolver returns nothing
    class EmptyResolver:
        def resolve(self, text, explicit_mentions, entities, aliases):
            return []

    engine.mention_resolver = EmptyResolver()
    ctx_pov = engine.retrieve_context(
        step=1,
        text="Silence.",
        pov_entity_ids=["char_alice"],
        fallback_on_empty_mentions="pov_only",
    )
    assert [e.entity_id for e in ctx_pov.entities] == ["char_alice"]


def test_extraction_prompt_normalization_and_transaction():
    class MockExtractor:
        def extract_delta(self, prompt: str, scene_text: str) -> StepDelta:
            assert "char_arthur" in prompt
            assert "StepDelta" in prompt
            # Simulate LLM emitting display names & temporary episode ID
            return StepDelta(
                episodes=[
                    EpisodeRecord(
                        episode_id="temp_ep_1",
                        predicate="argument",
                        summary="Arthur confronted Merlin.",
                        participants={"Arthur": "actor", "Merlin": "target"},
                    )
                ],
                edges=[
                    EdgeDelta(
                        source_id="Arthur",
                        target_id="Merlin",
                        state={"tone": "angry"},
                        caused_by_episode_id="temp_ep_1",
                    )
                ],
            )

    engine = NarrativeMemoryEngine(":memory:", extractor=MockExtractor())
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)

    with engine.step_transaction(1, text="Arthur confronted Merlin.") as tx:
        delta = tx.extract_and_commit("Arthur slammed the table.")
        assert delta.edges[0].source_id == "char_arthur"
        assert delta.edges[0].target_id == "char_merlin"
        assert "char_arthur" in delta.episodes[0].participants

    ctx = engine.retrieve_context(step=2, text="Arthur spoke to Merlin.")
    assert len(ctx.active_edges) == 1
    assert ctx.active_edges[0].state["tone"] == "angry"
    assert ctx.active_edges[0].caused_by_episode_id == "temp_ep_1"


def test_deterministic_ngram_embedder_across_subprocesses():
    import os
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": src_dir}
    code = (
        "from narrative_memory.voice import OfflineNgramEmbedder; "
        "import json; "
        "v = OfflineNgramEmbedder(dim=32).embed_text('Arthur drew Excalibur').tolist(); "
        "print(json.dumps(v))"
    )
    out1 = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    out2 = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    assert out1 == out2


def test_dimension_mismatch_guard_in_voice_bank():
    bank = VoiceBank(embedder=OfflineNgramEmbedder(dim=64))
    wrong_dim_bytes = np.ones(128, dtype=np.float32).tobytes()
    ex = VoiceExemplar(step=1, speaker_id="s1", text="Hold the line!")
    ranked = bank.rank_exemplars("Hold the line!", [(ex, wrong_dim_bytes)], top_k=1)
    assert len(ranked) == 1
    assert ranked[0].match_score > 0.9


def test_branch_isolation_rollback_guard_snapshot_and_state_key_deletion():
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="c1", canonical_name="Alice", state={"mood": "calm", "disguise": "bard"}),
        step=0,
    )
    engine.commit_step(
        step=2,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="c1", state={"disguise": None})],
            episodes=[EpisodeRecord(episode_id="ep_shared", summary="Shared step 2 event")],
        ),
        exemplars=[VoiceExemplar(exemplar_id="vx_1", step=2, speaker_id="c1", text="A quiet tune.")],
        bypass_gate=True,
    )

    # Verify None removed 'disguise' while keeping 'mood'
    ctx3 = engine.retrieve_context(step=3, mentions=["Alice"])
    assert ctx3.entities[0].state == {"mood": "calm"}

    # Branch at step 2: voice exemplar at step 2 MUST be visible on child branch
    engine.branch(at_step=2, name="alt", switch=True)
    ctx_alt = engine.retrieve_context(step=3, mentions=["Alice"], top_k_voice=1)
    assert len(ctx_alt.voice_exemplars) == 1

    # Committing same episode_id on 'alt' must NOT overwrite 'main'
    engine.commit_step(
        step=3,
        delta=StepDelta(episodes=[EpisodeRecord(episode_id="ep_shared", summary="Alt branch event")]),
    )
    main_eps = engine.query_episodes(timeline_id="main")
    alt_eps = engine.query_episodes(timeline_id="alt")
    assert [e.summary for e in main_eps] == ["Shared step 2 event"]
    assert any(e.summary == "Alt branch event" for e in alt_eps)

    # Rolling back 'main' before child branch fork_step=2 must raise TimelineBranchConflictError
    with pytest.raises(TimelineBranchConflictError):
        engine.rollback_to(1, timeline_id="main")

    # Export snapshot must preserve voice exemplars and valid_from_step
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = Path(tmp) / "snap.db"
        engine.export_snapshot(snap_path, at_step=3, timeline_id="alt")
        snap_engine = NarrativeMemoryEngine(snap_path)
        snap_ctx = snap_engine.retrieve_context(step=4, mentions=["Alice"], top_k_voice=1)
        assert len(snap_ctx.voice_exemplars) == 1
        assert snap_ctx.entities[0].valid_from_step == 2
        snap_engine.close()
    engine.close()
