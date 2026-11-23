import warnings

import numpy as np
import pytest

from canonarc import (
    AliasRecord,
    BaseEngineHooks,
    CallableEmbedder,
    CallableExtractor,
    CausalStore,
    DefaultEpisodeRanker,
    DefaultLexicalResolver,
    EdgeDelta,
    EmbedderMismatchError,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    EpisodicScoringPolicy,
    InformDelta,
    CanonArcEngine,
    OfflineNgramEmbedder,
    SchemaPolicy,
    StepDelta,
    TimelineBranchConflictError,
    VoiceExemplar,
)
from canonarc.cli import main as cli_main


def test_disguise_fuzzy_alias_and_comention_do_not_leak():
    """Fuzzy-matched disguise aliases and canonical co-mentions must not leak true identity."""
    engine = CanonArcEngine(
        ":memory:",
        mention_resolver=DefaultLexicalResolver(fuzzy_threshold=82.0),
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_odysseus",
            canonical_name="Odysseus",
            private_state={"secret": "king"},
        ),
        aliases=[
            AliasRecord(
                alias="the old beggar",
                entity_id="char_odysseus",
                kind="disguise",
                known_to=["char_odysseus", "char_eurycleia"],
                persona_name="Old Beggar",
            )
        ],
        step=0,
    )
    engine.register_entity(EntityRecord(entity_id="char_suitor", canonical_name="Antinous"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_eurycleia", canonical_name="Eurycleia"), step=0)
    engine.commit_step(
        step=1,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_suitor",
                    target_id="char_odysseus",
                    state={"tone": "hostile"},
                )
            ]
        ),
    )

    # Case A: Inflected disguise alias ("old beggars") matched via fuzzy fallback
    ctx_fuzzy = engine.retrieve_context(
        step=5,
        text="Antinous laughed at the old beggars by the hearth.",
        pov_entity_ids=["char_suitor"],
    )
    ids_fuzzy = {e.entity_id for e in ctx_fuzzy.entities}
    assert "masked:char_odysseus" in ids_fuzzy
    assert "char_odysseus" not in ids_fuzzy
    assert ctx_fuzzy.active_edges == []

    # Case B: Co-mentioning canonical "Odysseus" alongside disguise alias still masks for fooled POV
    ctx_comention = engine.retrieve_context(
        step=5,
        text="Antinous mocked the old beggar, never suspecting Odysseus.",
        mentions=["Antinous", "the old beggar", "Odysseus"],
        pov_entity_ids=["char_suitor"],
    )
    ids_comention = {e.entity_id for e in ctx_comention.entities}
    assert "masked:char_odysseus" in ids_comention
    assert "char_odysseus" not in ids_comention

    # Piercing POV still resolves true identity with pierced annotation
    ctx_eury = engine.retrieve_context(
        step=5,
        text="Eurycleia looked at the old beggars.",
        pov_entity_ids=["char_eurycleia"],
    )
    eury_ids = {e.entity_id for e in ctx_eury.entities}
    assert "char_odysseus" in eury_ids
    odysseus = next(e for e in ctx_eury.entities if e.entity_id == "char_odysseus")
    assert odysseus.disguised_as == "Old Beggar"
    engine.close()


def test_shared_alias_text_scan_and_partial_overlap_guard():
    """Shared aliases in text resolve all matching entities while partial span overlaps are blocked."""
    resolver = DefaultLexicalResolver(fuzzy_threshold=None)
    entities = [
        EntityRecord(entity_id="char_a", canonical_name="Alice"),
        EntityRecord(entity_id="char_b", canonical_name="Bob"),
        EntityRecord(entity_id="char_arthur", canonical_name="King Arthur Pendragon"),
        EntityRecord(entity_id="char_fake", canonical_name="King Arthur"),
    ]
    aliases = [
        AliasRecord(alias="the captain", entity_id="char_a"),
        AliasRecord(alias="the captain", entity_id="char_b"),
    ]

    mentions = resolver.resolve(
        text="The captain saluted King Arthur Pendragon.",
        explicit_mentions=(),
        entities=entities,
        aliases=aliases,
    )
    matched_ids = {m.entity_id for m in mentions}
    assert "char_a" in matched_ids
    assert "char_b" in matched_ids
    assert "char_arthur" in matched_ids
    assert "char_fake" not in matched_ids


def test_deceived_pov_semantic_search_uses_believed_summary():
    """Deceived POV semantic scoring matches against believed_summary, not hidden ground truth."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)

    engine.commit_step(
        step=10,
        delta=StepDelta(
            episodes=[
                EpisodeRecord(
                    episode_id="ep_egg",
                    summary="Merlin hid the dragon egg in the cellar.",
                    participants={"char_merlin": "actor"},
                    known_by={"char_arthur": "deceived", "char_merlin": "direct"},
                    believed_summaries={
                        "char_arthur": "The dragon fled to the northern mountains."
                    },
                    salience=5.0,
                ),
                EpisodeRecord(
                    episode_id="ep_feast",
                    summary="Merlin baked bread in the hall.",
                    participants={"char_merlin": "actor"},
                    known_by={"char_arthur": "direct", "char_merlin": "direct"},
                    salience=5.0,
                ),
            ]
        ),
    )

    # Querying "northern mountains" under Arthur's POV must match ep_egg via believed_summary!
    ctx_mountains = engine.retrieve_context(
        step=15,
        text="Merlin looked toward the northern mountains dragon.",
        pov_entity_ids=["char_arthur"],
        top_k_episodes=1,
    )
    assert len(ctx_mountains.episodes) == 1
    assert ctx_mountains.episodes[0].episode_id == "ep_egg"
    assert ctx_mountains.episodes[0].summary == "The dragon fled to the northern mountains."

    # Querying "cellar" under Arthur's POV must NOT match hidden ground truth ("cellar"),
    # unlike Merlin's POV which knows the ground truth.
    ctx_arthur_cellar = engine.retrieve_context(
        step=15,
        text="cellar",
        pov_entity_ids=["char_arthur"],
        top_k_episodes=2,
    )
    ctx_merlin_cellar = engine.retrieve_context(
        step=15,
        text="cellar",
        pov_entity_ids=["char_merlin"],
        top_k_episodes=2,
    )
    arthur_egg_score = next(e.match_score for e in ctx_arthur_cellar.episodes if e.episode_id == "ep_egg")
    merlin_egg_score = next(e.match_score for e in ctx_merlin_cellar.episodes if e.episode_id == "ep_egg")
    assert merlin_egg_score > arthur_egg_score + 0.10
    engine.close()


def test_offline_embedder_cjk_tokenization_and_signed_hashing():
    """OfflineNgramEmbedder splits CJK units cleanly and avoids positive collision drift."""
    emb = OfflineNgramEmbedder(dim=128)
    v_zh1 = emb.embed_text("苹果香蕉橙子")
    v_zh2 = emb.embed_text("银河宇宙星辰")
    v_zh1_dup = emb.embed_text("苹果香蕉橙子")

    assert np.allclose(v_zh1, v_zh1_dup)
    sim_same = float(np.dot(v_zh1, v_zh1_dup))
    sim_diff = float(np.dot(v_zh1, v_zh2))
    assert sim_same > 0.99
    assert sim_diff < 0.25

    # Adjacent CJK bigrams distinguish reversed 2-char compounds and respect boundaries
    v_kingdom = emb.embed_text("王国")
    v_king = emb.embed_text("国王")
    v_spaced = emb.embed_text("王 国")
    assert float(np.dot(v_kingdom, v_king)) < 0.85
    assert not np.allclose(v_kingdom, v_spaced)


def test_cte_same_step_branch_override_and_snapshot_revision_count(tmp_path):
    """Same-step branch overrides do not leak into prior_state or inflate revision_count,
    and export_snapshot preserves revision_count and deduplicates branch epistemics."""
    store = CausalStore(":memory:")
    store.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "calm"}),
        step=0,
    )
    store.apply_delta(
        step=5,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "alert"})]),
    )
    store.apply_delta(
        step=10,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "angry"})],
            episodes=[
                EpisodeRecord(
                    episode_id="ep_1",
                    summary="Secret council meeting",
                    known_by={"char_arthur": "direct"},
                )
            ],
        ),
    )

    # Fork at step 10 and override char_arthur and epistemic at step 10 on alt
    store.create_timeline("alt", parent_id="main", fork_step=10)
    store.apply_delta(
        step=10,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "joyful"})]),
        timeline_id="alt",
    )
    store.record_epistemic(
        InformDelta(
            episode_id="ep_1",
            entity_id="char_arthur",
            epistemic="deceived",
            believed_summary="Peaceful banquet",
        ),
        step=10,
        timeline_id="alt",
    )

    alt_arthur = store.get_active_entity("char_arthur", timeline_id="alt", cutoff_step=10)
    assert alt_arthur is not None
    assert alt_arthur.state == {"mood": "joyful"}
    # Prior state must be step 5 ('alert'), NOT main's overridden step 10 ('angry')!
    assert alt_arthur.prior_state == {"mood": "alert"}
    assert alt_arthur.prior_step == 5
    assert alt_arthur.revision_count == 3  # step 0, step 5, step 10 (alt)

    # Add one more revision at step 15 -> revision_count == 4
    store.apply_delta(
        step=15,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_arthur", state={"mood": "tired"})]),
        timeline_id="alt",
    )
    alt_15 = store.get_active_entity("char_arthur", timeline_id="alt", cutoff_step=15)
    assert alt_15 is not None
    assert alt_15.revision_count == 4

    snap_path = tmp_path / "snapshot.sqlite"
    store.export_snapshot(snap_path, at_step=15, timeline_id="alt")

    with CausalStore(snap_path) as snap_store:
        loaded_arthur = snap_store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=15)
        assert loaded_arthur is not None
        assert loaded_arthur.state == {"mood": "tired"}
        assert loaded_arthur.prior_state == {"mood": "joyful"}
        assert loaded_arthur.revision_count == 4

        ep = snap_store.get_episode("ep_1", cutoff_step=15)
        assert ep is not None
        assert ep.known_by["char_arthur"] == "deceived"
        assert ep.believed_summaries["char_arthur"] == "Peaceful banquet"
        ep_rows = snap_store.conn.execute("SELECT COUNT(*) FROM episode_epistemics").fetchone()[0]
        assert ep_rows == 1
    store.close()


def test_normalize_delta_intra_step_new_entity_and_alias():
    """normalize_delta resolves entities and non-disguise aliases created within the same StepDelta."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)

    raw = StepDelta(
        entities=[
            EntityDelta(
                entity_id="char_lancelot",
                canonical_name="Lancelot",
                add_aliases=["the silver knight"],
            )
        ],
        edges=[
            EdgeDelta(
                source_id="Arthur",
                target_id="Lancelot",
                state={"tone": "respectful"},
            )
        ],
        episodes=[
            EpisodeRecord(
                summary="Arthur knighted Lancelot.",
                participants={"Arthur": "actor", "the silver knight": "recipient"},
            )
        ],
    )
    norm = engine.normalize_delta(raw, step=1)
    assert norm.edges[0].source_id == "char_arthur"
    assert norm.edges[0].target_id == "char_lancelot"
    assert norm.episodes[0].participants == {"char_arthur": "actor", "char_lancelot": "recipient"}
    engine.close()


def test_multipov_all_mode_belief_state_consistency():
    """In pov_mode='all', belief_state only overrides attributes shared by all POV observers."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(
            entity_id="char_arthur",
            canonical_name="Arthur",
            state={"status": "dead", "location": "avalon"},
        ),
        step=0,
    )
    engine.register_entity(EntityRecord(entity_id="char_alice", canonical_name="Alice"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_bob", canonical_name="Bob"), step=0)

    engine.commit_step(
        step=5,
        delta=StepDelta(
            edges=[
                EdgeDelta(
                    source_id="char_alice",
                    target_id="char_arthur",
                    relation_type="belief_state",
                    state={"status": "alive", "location": "camelot"},
                ),
                EdgeDelta(
                    source_id="char_bob",
                    target_id="char_arthur",
                    relation_type="belief_state",
                    state={"status": "alive", "location": "north"},
                ),
            ]
        ),
    )

    ctx_all = engine.retrieve_context(
        step=10,
        mentions=["Arthur"],
        pov_entity_ids=["char_alice", "char_bob"],
        pov_mode="all",
    )
    arthur = ctx_all.entities[0]
    assert arthur.state["status"] == "alive"  # agreed by both
    assert arthur.state["location"] == "avalon"  # diverged -> ground truth preserved
    engine.close()


def test_epistemic_forgetting_and_suppression():
    """Episodes can be forgotten, suppressed, or overwritten to deceived at a later step."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(EntityRecord(entity_id="char_merlin", canonical_name="Merlin"), step=0)

    engine.index_episodes(
        step=5,
        episodes=[
            EpisodeRecord(
                episode_id="ep_spell",
                summary="Merlin cast a forbidden spell.",
                participants={"char_merlin": "actor"},
                witnesses=["char_arthur"],
                salience=9.0,
            )
        ],
    )

    # At step 10, Arthur remembers witnessing it
    ctx_before = engine.retrieve_context(
        step=10,
        text="Merlin",
        pov_entity_ids=["char_arthur"],
    )
    assert [e.episode_id for e in ctx_before.episodes] == ["ep_spell"]

    # At step 20, Merlin wipes Arthur's memory
    engine.inform_entity(step=20, episode_id="ep_spell", entity_id="char_arthur", epistemic="forgotten")

    # At step 25, Arthur no longer recalls ep_spell
    ctx_after = engine.retrieve_context(
        step=25,
        text="Merlin",
        pov_entity_ids=["char_arthur"],
    )
    assert ctx_after.episodes == []

    # Time-travel back to step 10 still shows Arthur knew it before step 20!
    ep_at_10 = engine.store.get_episode("ep_spell", cutoff_step=10)
    assert ep_at_10 is not None
    assert ep_at_10.known_by["char_arthur"] == "direct"
    engine.close()


def test_flashback_mode_narrative_time_and_episode_consolidation():
    """Flashbacks do not clobber present-day entity/edge states, and consolidate_episodes compacts old events."""
    engine = CanonArcEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "furious"}),
        step=0,
    )
    engine.commit_step(
        step=10,
        delta=StepDelta(
            is_flashback=True,
            story_time="10 years earlier",
            chapter_id="ch_1",
            scene_id="sc_flashback",
            entities=[
                EntityDelta(entity_id="char_arthur", state={"mood": "innocent"}),
                EntityDelta(entity_id="char_uther", canonical_name="Uther", state={"mood": "proud"}),
            ],
            episodes=[
                EpisodeRecord(
                    episode_id="ep_flashback",
                    summary="Young Arthur trained with Uther.",
                    participants={"char_arthur": "actor", "char_uther": "mentor"},
                    salience=8.5,
                )
            ],
        ),
    )

    # Present-day Arthur mood was NOT overwritten by the flashback
    arthur = engine.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=11)
    assert arthur is not None
    assert arthur.state["mood"] == "furious"

    # Brand new flashback entity Uther WAS created
    uther = engine.store.get_active_entity("char_uther", timeline_id="main", cutoff_step=11)
    assert uther is not None
    assert uther.canonical_name == "Uther"

    # Flashback episode preserved narrative time & hierarchy fields
    ep_fb = engine.store.get_episode("ep_flashback", cutoff_step=11)
    assert ep_fb is not None
    assert ep_fb.story_time == "10 years earlier"
    assert ep_fb.chapter_id == "ch_1"
    assert ep_fb.scene_id == "sc_flashback"

    # Add minor low-salience episodes and consolidate them
    for s in (12, 13, 14):
        engine.index_episodes(
            step=s,
            episodes=[
                EpisodeRecord(
                    episode_id=f"ep_minor_{s}",
                    summary=f"Routine patrol {s}.",
                    participants={"char_arthur": "actor"},
                    chapter_id="ch_1",
                    salience=2.0,
                )
            ],
        )

    consolidated = engine.consolidate_episodes(
        step=20,
        before_step=15,
        max_salience=4.0,
        chapter_id="ch_1",
        summary="Chapter 1 Summary: Arthur completed routine patrols.",
    )
    assert consolidated is not None
    assert set(consolidated.consolidated_from) == {"ep_minor_12", "ep_minor_13", "ep_minor_14"}
    remaining_ids = {e.episode_id for e in engine.query_episodes(cutoff_step=25)}
    assert "ep_flashback" in remaining_ids  # high salience kept
    assert consolidated.episode_id in remaining_ids
    assert "ep_minor_12" not in remaining_ids
    engine.close()


def test_delete_timeline_cascade_and_branch_conflict():
    """delete_timeline enforces branch safety unless cascade=True and rejects deleting main."""
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice"), step=0)
        engine.branch(at_step=0, name="route_a")
        engine.branch(at_step=1, name="route_a_sub", from_timeline="route_a")

        with pytest.raises(ValueError, match="main"):
            engine.delete_timeline("main")

        with pytest.raises(TimelineBranchConflictError):
            engine.delete_timeline("route_a")

        deleted = engine.delete_timeline("route_a", cascade=True)
        assert deleted >= 2
        assert [t.timeline_id for t in engine.list_timelines()] == ["main"]


def test_sessions_hooks_batch_embedder_extractor_ranker_and_cli(tmp_path):
    """Tests TimelineSession, ContextVar use_timeline, hooks, batch embedding, CallableExtractor, EpisodeRanker, and CLI."""
    batch_calls: list[int] = []

    class CountingBatchEmbedder(OfflineNgramEmbedder):
        def embed_batch(self, texts):
            batch_calls.append(len(texts))
            return [self.embed_text(t) for t in texts]

    events: list[str] = []

    class RecordingHooks(BaseEngineHooks):
        def on_before_commit(self, step, timeline_id, delta, exemplars, episodes):
            events.append(f"before:{timeline_id}:{step}")

        def on_after_commit(self, step, timeline_id, committed_episodes, indexed_exemplars):
            events.append(f"after:{timeline_id}:{step}:{len(committed_episodes)}:{len(indexed_exemplars)}")

        def on_branch(self, timeline_info):
            events.append(f"branch:{timeline_info.timeline_id}")

        def on_rollback(self, step, timeline_id, deleted_count):
            events.append(f"rollback:{timeline_id}:{step}")

        def on_gate_rejected(self, stage, text, reason, metadata):
            events.append(f"rejected:{stage}")

    attempts = {"n": 0}

    def flaky_llm(prompt: str, schema: dict) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "not valid json"
        return '```json\n{"entities": [{"entity_id": "char_arthur", "state": {"mood": "bold"}}]}\n```'

    extractor = CallableExtractor(flaky_llm, max_retries=1)
    custom_policy = SchemaPolicy(
        episodic_policy=EpisodicScoringPolicy(salience_weight=0.5),
    )
    from canonarc.presets import InteractiveFictionPreset

    db_file = tmp_path / "cli_test.sqlite"
    with CanonArcEngine(
        db_file,
        embedder=CountingBatchEmbedder(),
        extractor=extractor,
        preset=InteractiveFictionPreset(policy=custom_policy),
        episode_ranker=DefaultEpisodeRanker(),
        hooks=[RecordingHooks()],
    ) as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "calm"}),
            step=0,
        )
        engine.commit_step(
            step=5,
            delta=StepDelta(
                episodes=[
                    EpisodeRecord(episode_id="ep_a", summary="Arthur entered the hall."),
                    EpisodeRecord(episode_id="ep_b", summary="Arthur inspected the banners."),
                ]
            ),
            exemplars=[
                VoiceExemplar(speaker_id="char_arthur", step=5, text="Too short", quality_score=5.0),
                VoiceExemplar(
                    speaker_id="char_arthur",
                    step=5,
                    text="Stand ready, knights of Camelot, for dawn brings our decisive trial!",
                    quality_score=9.2,
                ),
            ],
        )
        # 1 batch call for 2 episodes + 1 batch call for 1 surviving exemplar
        assert batch_calls == [2, 1]
        assert "rejected:gate2" in events
        assert "before:main:5" in events
        assert "after:main:5:2:1" in events

        # Gate 1 rejection hook
        engine.extract_dialogue_candidates('Arthur said, "Yes."', step=6)
        assert "rejected:gate1" in events

        # Branch + session + use_timeline isolation
        engine.branch(at_step=5, name="route_b")
        assert "branch:route_b" in events

        sess_b = engine.session("route_b")
        with sess_b.step_transaction(step=10, text="Arthur drew his sword.") as tx:
            delta = tx.extract_and_commit("Arthur stepped forward boldly.")
            assert delta.entities[0].state["mood"] == "bold"

        # ContextVar scoping leaves default timeline untouched
        with engine.use_timeline("route_b"):
            assert engine.active_timeline == "route_b"
        assert engine.active_timeline == "main"

        # Main was not mutated by route_b's step 10
        main_art = engine.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=10)
        b_art = engine.store.get_active_entity("char_arthur", timeline_id="route_b", cutoff_step=10)
        assert main_art is not None and main_art.state["mood"] == "calm"
        assert b_art is not None and b_art.state["mood"] == "bold"

    # Verify CLI subcommands on persisted DB
    assert cli_main(["tree", str(db_file)]) == 0
    assert cli_main(["inspect", str(db_file), "--timeline", "route_b", "--step", "10", "--inclusive"]) == 0
    assert cli_main(["diff", str(db_file), "main", "route_b", "--step", "10"]) == 0
    assert cli_main(["history", str(db_file), "char_arthur", "--timeline", "route_b"]) == 0


def test_embedder_metadata_mismatch_and_reindex(tmp_path):
    """Store records embedder_id & dim; reopening with another embedder warns, raises, or reindexes."""
    db_file = tmp_path / "embedder_meta.db"

    with CanonArcEngine(db_file, embedder=OfflineNgramEmbedder(dim=64)) as engine:
        engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
        engine.commit_step(
            step=1,
            episodes=[
                EpisodeRecord(
                    episode_id="ep_1",
                    summary="Arthur found the ancient crown.",
                    participants={"char_arthur": "actor"},
                    witnesses=["char_arthur"],
                )
            ],
            exemplars=[
                VoiceExemplar(
                    step=1,
                    speaker_id="char_arthur",
                    text="By the old oath, this crown belongs to Camelot alone!",
                    quality_score=9.5,
                )
            ],
        )
        assert engine.store.get_embedding_metadata() == ("OfflineNgramEmbedder(dim=64)", 64)

    # Different dimension -> warn by default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with CanonArcEngine(db_file, embedder=OfflineNgramEmbedder(dim=128)) as engine:
            assert any("OfflineNgramEmbedder(dim=64)" in str(w.message) for w in caught)

    # Same dimension (64), different embedder_id -> raise when strict
    alt_same_dim = CallableEmbedder(
        lambda text: np.ones(64, dtype=np.float32),
        embedder_id="custom-model-v1",
        dim=64,
    )
    with pytest.raises(EmbedderMismatchError, match="custom-model-v1"):
        CanonArcEngine(db_file, embedder=alt_same_dim, on_embedder_mismatch="raise").close()

    # Different dimension + on_embedder_mismatch="reindex" migrates stored BLOBs in batches
    with CanonArcEngine(
        db_file, embedder=OfflineNgramEmbedder(dim=128), on_embedder_mismatch="reindex"
    ) as engine:
        assert engine.store.get_embedding_metadata() == ("OfflineNgramEmbedder(dim=128)", 128)
        ep_rows = engine.store.get_lineage_episodes("main", 10)
        assert len(ep_rows[0][1]) == 128 * 4
        vo_rows = engine.store.get_lineage_voice_exemplars("main", 10, "char_arthur")
        assert len(vo_rows[0][1]) == 128 * 4

    # Re-opening with the migrated embedder now succeeds cleanly without warnings
    with warnings.catch_warnings(record=True) as caught_after:
        warnings.simplefilter("always")
        with CanonArcEngine(
            db_file, embedder=OfflineNgramEmbedder(dim=128), on_embedder_mismatch="raise"
        ) as engine:
            counts = engine.reindex_embeddings()
            assert counts == {"episodes": 1, "voice_exemplars": 1}
        assert not [w for w in caught_after if issubclass(w.category, UserWarning)]
