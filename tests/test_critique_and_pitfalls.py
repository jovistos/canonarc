import sqlite3
import tempfile
from pathlib import Path

import pytest

from narrative_memory import (
    AliasRecord,
    EdgeDelta,
    EntityDelta,
    EntityRecord,
    EpisodeRecord,
    InformDelta,
    NarrativeMemoryEngine,
    StepDelta,
    VoiceExemplar,
)
from narrative_memory.store import CausalStore, TimelineBranchConflictError


def test_sqlite_pragmas_and_foreign_keys():
    """PRAGMA foreign_keys=ON is enforced and file-backed DBs use WAL mode."""
    store = CausalStore(":memory:")
    fk_on = store.conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    assert fk_on == 1
    with pytest.raises(sqlite3.IntegrityError):
        with store.conn:
            store.conn.execute(
                "INSERT INTO graph_revisions (timeline_id, step, record_type, subject_key, is_tombstone, payload) "
                "VALUES ('nonexistent_tl', 1, 'entity', 'e1', 0, '{}')"
            )
    store.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "wal_test.sqlite"
        file_store = CausalStore(db_file)
        mode = file_store.conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
        file_store.close()


def test_timeline_head_step_and_fork_step_required():
    """TimelineInfo.head_step reflects max step across tables, and fork_step=None is rejected for branches."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice"), step=0)
    engine.commit_step(
        step=12,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_a", state={"mood": "happy"})]),
    )
    main_info = engine.store.get_timeline("main")
    assert main_info is not None
    assert main_info.head_step == 12

    with pytest.raises(ValueError, match="fork_step"):
        engine.store.create_timeline("bad_branch", parent_id="main", fork_step=None)

    branch_info = engine.branch(at_step=12, name="alt_route")
    assert branch_info.head_step == 12

    engine.index_episodes(
        step=25,
        episodes=[EpisodeRecord(episode_id="ep_alt", summary="Branch event")],
        timeline_id="alt_route",
    )
    timelines = {t.timeline_id: t for t in engine.list_timelines()}
    assert timelines["main"].head_step == 12
    assert timelines["alt_route"].head_step == 25
    engine.close()


def test_alias_scoping_case_normalization_and_safe_removal():
    """Shared epithets across entities do not collide, and mixed-case aliases tombstone cleanly."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_a", canonical_name="Alice"),
        aliases=["The Captain", "The Old Wizard"],
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_b", canonical_name="Bob"),
        aliases=["the captain"],
        step=0,
    )

    aliases_0 = engine.store.get_all_active_aliases("main", 0)
    captain_entities = {a.entity_id for a in aliases_0 if a.alias.lower() == "the captain"}
    assert captain_entities == {"char_a", "char_b"}

    # Remove mixed-case aliases from Alice only
    engine.commit_step(
        step=5,
        delta=StepDelta(
            entities=[
                EntityDelta(
                    entity_id="char_a",
                    remove_aliases=["the captain", "the old wizard"],
                )
            ]
        ),
    )

    aliases_5 = engine.store.get_all_active_aliases("main", 5)
    assert [(a.entity_id, a.alias.lower()) for a in aliases_5] == [("char_b", "the captain")]
    engine.close()


def test_get_history_branch_dedup_tombstones_and_large_steps():
    """get_history handles branch overrides, record_type isolation, tombstones, and steps > 999999999."""
    store = CausalStore(":memory:")
    store.register_entity(
        EntityRecord(entity_id="hero", canonical_name="Hero", state={"hp": 100}),
        aliases=["hero"],
        step=0,
    )
    store.apply_delta(
        step=10,
        delta=StepDelta(entities=[EntityDelta(entity_id="hero", state={"hp": 80})]),
    )
    store.create_timeline("alt", parent_id="main", fork_step=10)
    store.apply_delta(
        step=10,
        delta=StepDelta(entities=[EntityDelta(entity_id="hero", state={"hp": 999})]),
        timeline_id="alt",
    )
    store.apply_delta(
        step=2_000_000_000,
        delta=StepDelta(entities=[EntityDelta(entity_id="hero", remove=True)]),
        timeline_id="alt",
    )

    hist = store.get_history("hero", timeline_id="alt", record_type="entity")
    assert len(hist) == 3
    assert hist[0]["valid_from_step"] == 0
    assert hist[0]["valid_to_step"] == 9
    assert hist[1]["valid_from_step"] == 10
    assert hist[1]["timeline_id"] == "alt"
    assert hist[1]["payload"]["state"]["hp"] == 999
    assert hist[1]["valid_to_step"] == 1_999_999_999
    assert hist[2]["is_tombstone"] is True
    assert hist[2]["valid_from_step"] == 2_000_000_000
    assert hist[2]["valid_to_step"] == 2_000_000_000
    store.close()


def test_export_snapshot_preserves_epistemics_trajectory_and_voice():
    """export_snapshot preserves raw episode_epistemics, prior_state trajectory, and voice_exemplars."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(entity_id="char_arthur", canonical_name="Arthur", state={"mood": "calm"}),
        step=0,
    )
    engine.commit_step(
        step=5,
        delta=StepDelta(
            entities=[EntityDelta(entity_id="char_arthur", state={"mood": "furious"})],
            episodes=[
                EpisodeRecord(
                    episode_id="ep_secret",
                    summary="The vault was robbed.",
                    participants={"char_merlin": "witness"},
                    known_by={"char_arthur": "deceived"},
                    believed_summaries={"char_arthur": "The vault is safe."},
                )
            ],
        ),
        exemplars=[
            VoiceExemplar(
                exemplar_id="ex_1",
                step=5,
                speaker_id="char_arthur",
                text="Nobody touches my vault without answering to me!",
                quality_score=9.5,
            )
        ],
        bypass_gate=True,
    )
    # At step 20, Arthur learns the truth
    engine.inform_entity(step=20, episode_id="ep_secret", entity_id="char_arthur", epistemic="told")

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "save_25.sqlite"
        engine.export_snapshot(snap_path, at_step=25)
        loaded = NarrativeMemoryEngine(snap_path)

        # 1. Voice exemplar survived
        voice = loaded.store.get_lineage_voice_exemplars("main", 25, "char_arthur")
        assert len(voice) == 1
        assert voice[0][0].exemplar_id == "ex_1"

        # 2. Prior state trajectory survived
        arthur = loaded.store.get_active_entity("char_arthur", timeline_id="main", cutoff_step=25)
        assert arthur is not None
        assert arthur.state == {"mood": "furious"}
        assert arthur.prior_state == {"mood": "calm"}
        assert arthur.prior_step == 0

        # 3. Time-traveling epistemics survived: at step 10 Arthur is still deceived; at step 25 he is told
        ep_at_10 = loaded.store.get_episode("ep_secret", cutoff_step=10)
        assert ep_at_10 is not None
        assert ep_at_10.known_by["char_arthur"] == "deceived"
        assert ep_at_10.believed_summaries["char_arthur"] == "The vault is safe."

        ep_at_25 = loaded.store.get_episode("ep_secret", cutoff_step=25)
        assert ep_at_25 is not None
        assert ep_at_25.known_by["char_arthur"] == "told"
        loaded.close()
    engine.close()


def test_step_transaction_preserves_data_on_exception_and_guards_branches():
    """StepTransaction does not delete >= step data if an exception occurs inside the with-block."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice", state={"hp": 10}), step=0)
    engine.commit_step(
        step=5,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_a", state={"hp": 5})]),
    )

    with pytest.raises(RuntimeError, match="LLM timeout"):
        with engine.step_transaction(step=5, rerun_clean=True):
            raise RuntimeError("LLM timeout")

    # Step 5 data must still be intact!
    alice = engine.store.get_active_entity("char_a", timeline_id="main", cutoff_step=5)
    assert alice is not None
    assert alice.state["hp"] == 5

    # Child branch guard still triggers on __enter__
    engine.branch(at_step=5, name="child")
    with pytest.raises(TimelineBranchConflictError):
        with engine.step_transaction(step=5, rerun_clean=True):
            pass
    engine.close()


def test_deceived_believed_summary_preserved_and_multipov_all_isolation():
    """Re-asserting deceived without summary preserves believed_summary; pov_mode='all' isolates secrets."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(
        EntityRecord(
            entity_id="char_alice",
            canonical_name="Alice",
            private_state={"secret": "alice_secret"},
        ),
        step=0,
    )
    engine.register_entity(
        EntityRecord(
            entity_id="char_bob",
            canonical_name="Bob",
            private_state={"secret": "bob_secret"},
        ),
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_spy", canonical_name="Mallory"),
        aliases=[
            AliasRecord(
                alias="the merchant",
                entity_id="char_spy",
                kind="disguise",
                known_to=["char_alice"],
                persona_name="Innocent Merchant",
            )
        ],
        step=0,
    )
    engine.commit_step(
        step=2,
        delta=StepDelta(
            episodes=[
                EpisodeRecord(
                    episode_id="ep_lie",
                    summary="Mallory stole the key.",
                    known_by={"char_alice": "direct", "char_bob": "deceived"},
                    believed_summaries={"char_bob": "The key was lost."},
                )
            ],
            entities=[
                EntityDelta(
                    entity_id="char_spy",
                    state={"mood": "nervous"},
                    caused_by_episode_id="ep_lie",
                    reason="Stole the key",
                )
            ],
        ),
    )
    # Re-assert deceived with believed_summary=None -> should NOT wipe out "The key was lost."
    engine.inform_entity(step=3, episode_id="ep_lie", entity_id="char_bob", epistemic="deceived")
    ep = engine.store.get_episode("ep_lie", cutoff_step=5)
    assert ep is not None
    assert ep.believed_summaries["char_bob"] == "The key was lost."

    # Multi-POV with pov_mode="all": Bob doesn't know ep_lie or the disguise, so both stay hidden
    ctx = engine.retrieve_context(
        step=5,
        text="Alice and Bob spoke with the merchant.",
        mentions=["Alice", "Bob"],
        pov_entity_ids=["char_alice", "char_bob"],
        pov_mode="all",
    )
    ids = {e.entity_id for e in ctx.entities}
    assert "masked:char_spy" in ids
    assert "char_spy" not in ids
    alice = next(e for e in ctx.entities if e.entity_id == "char_alice")
    bob = next(e for e in ctx.entities if e.entity_id == "char_bob")
    assert alice.private_state == {}
    assert bob.private_state == {}
    engine.close()


def test_lexical_resolver_finditer_and_subphrase_guard():
    """Shorter canonical name appearing after a longer alias is still found; shorter alias inside longer phrase is skipped."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_arthur", canonical_name="Arthur"), step=0)
    engine.register_entity(
        EntityRecord(entity_id="char_impostor", canonical_name="Mordred"),
        aliases=[
            AliasRecord(
                alias="King Arthur of Camelot",
                entity_id="char_impostor",
                kind="disguise",
                persona_name="False King",
            ),
            AliasRecord(alias="the guard", entity_id="char_impostor", kind="epithet"),
        ],
        step=0,
    )
    engine.register_entity(
        EntityRecord(entity_id="char_captain", canonical_name="The Guard Captain"),
        step=0,
    )

    ctx = engine.retrieve_context(
        step=1,
        text="King Arthur of Camelot greeted Arthur while The Guard Captain watched.",
        inclusive=True,
    )
    resolved_ids = {m.entity_id for m in ctx.resolved_mentions}
    # Both char_impostor (via "King Arthur of Camelot"), char_arthur (second "Arthur"), and char_captain
    assert "char_impostor" in resolved_ids
    assert "char_arthur" in resolved_ids
    assert "char_captain" in resolved_ids
    # "the guard" inside "The Guard Captain" must NOT have matched char_impostor as "the guard"
    assert not any(m.surface_text == "the guard" for m in ctx.resolved_mentions)
    engine.close()


def test_provenance_not_inherited_on_unrelated_state_mutation():
    """Mutating state without caused_by_episode_id/reason does not carry over stale provenance."""
    engine = NarrativeMemoryEngine(":memory:")
    engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice", state={"mood": "calm"}), step=0)
    engine.commit_step(
        step=5,
        delta=StepDelta(
            entities=[
                EntityDelta(
                    entity_id="char_a",
                    state={"mood": "furious"},
                    caused_by_episode_id="ep_5",
                    reason="Betrayed at step 5",
                )
            ]
        ),
    )
    engine.commit_step(
        step=10,
        delta=StepDelta(entities=[EntityDelta(entity_id="char_a", state={"mood": "calm"})]),
    )
    alice = engine.store.get_active_entity("char_a", timeline_id="main", cutoff_step=10)
    assert alice is not None
    assert alice.state == {"mood": "calm"}
    assert alice.caused_by_episode_id is None
    assert alice.reason is None
    engine.close()
