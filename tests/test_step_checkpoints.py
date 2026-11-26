from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from canonarc import (
    CanonArcEngine,
    EntityDelta,
    EntityRecord,
    StepCheckpoint,
    StepDelta,
    compute_step_hash,
)


def test_empty_delta_commit_advances_head_step_and_checkpoints() -> None:
    """Pure scenery/narration steps with empty deltas still advance head_step and record checkpoints."""
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="char_arthur", canonical_name="Arthur"),
            step=0,
        )
        # Entity seeding alone does not count as a committed step checkpoint
        assert engine.last_committed_step() is None
        assert engine.is_step_committed(0) is False

        # Step 1 has real mutations
        engine.commit_step(
            step=1,
            delta=StepDelta(
                entities=[EntityDelta(entity_id="char_arthur", state={"mood": "calm"})]
            ),
            content_hash="hash_step_1",
            checkpoint_metadata={"model": "gpt-4.1-mini"},
        )
        assert engine.last_committed_step() == 1
        assert engine.is_step_committed(1) is True

        # Step 2 is pure scenery: empty StepDelta, no episodes, no dialogue
        engine.commit_step(
            step=2,
            delta=StepDelta(),
            content_hash="hash_step_2",
            checkpoint_metadata={"model": "gpt-4.1-mini", "chunk_chars": 420},
        )
        assert engine.last_committed_step() == 2
        assert engine.is_step_committed(2) is True
        assert engine.is_step_committed(3) is False

        # TimelineInfo.head_step reflects step 2 in both get_timeline() and list_timelines()
        tl_info = engine.store.get_timeline("main")
        assert tl_info is not None
        assert tl_info.head_step == 2
        listed = {t.timeline_id: t for t in engine.list_timelines()}
        assert listed["main"].head_step == 2

        # Step 3 committed with delta=None also records a checkpoint
        engine.commit_step(step=3, content_hash="hash_step_3")
        assert engine.last_committed_step() == 3
        assert engine.is_step_committed(3, content_hash="hash_step_3") is True


def test_content_hash_metadata_match_and_compute_step_hash() -> None:
    """Verifies content_hash matching, metadata_match filtering, idempotent overwrite, and compute_step_hash."""
    h1 = compute_step_hash("Rain fell over the valley.", model="gpt-4.1-mini", prompt_version="v1")
    h1_same = compute_step_hash("Rain fell over the valley.", model="gpt-4.1-mini", prompt_version="v1", extra={})
    h1_diff_model = compute_step_hash("Rain fell over the valley.", model="gpt-4.1", prompt_version="v1")
    h1_extra_a = compute_step_hash("Rain", extra={"b": 2, "a": 1})
    h1_extra_b = compute_step_hash("Rain", extra={"a": 1, "b": 2})

    assert h1 == h1_same
    assert h1 != h1_diff_model
    assert h1_extra_a == h1_extra_b
    assert len(h1) == 64

    with CanonArcEngine(":memory:") as engine:
        engine.commit_step(
            step=14,
            delta=StepDelta(),
            content_hash=h1,
            checkpoint_metadata={"model": "gpt-4.1-mini", "prompt_version": "v1", "tokens": 310},
        )

        assert engine.is_step_committed(14) is True
        assert engine.is_step_committed(14, content_hash=h1) is True
        assert engine.is_step_committed(14, content_hash=h1_diff_model) is False
        assert engine.is_step_committed(14, metadata_match={"model": "gpt-4.1-mini"}) is True
        assert (
            engine.is_step_committed(
                14,
                content_hash=h1,
                metadata_match={"model": "gpt-4.1-mini", "prompt_version": "v1"},
            )
            is True
        )
        assert engine.is_step_committed(14, metadata_match={"model": "gpt-4.1"}) is False
        assert engine.is_step_committed(14, metadata_match={"missing_key": True}) is False

        cp = engine.get_step_checkpoint(14)
        assert isinstance(cp, StepCheckpoint)
        assert cp.timeline_id == "main"
        assert cp.step == 14
        assert cp.content_hash == h1
        assert cp.metadata == {"model": "gpt-4.1-mini", "prompt_version": "v1", "tokens": 310}

        # Re-committing step 14 updates hash and metadata idempotently
        engine.commit_step(
            step=14,
            delta=StepDelta(),
            content_hash=h1_diff_model,
            checkpoint_metadata={"model": "gpt-4.1", "prompt_version": "v1"},
        )
        assert engine.is_step_committed(14, content_hash=h1) is False
        assert engine.is_step_committed(14, content_hash=h1_diff_model) is True
        assert engine.is_step_committed(14, metadata_match={"model": "gpt-4.1"}) is True

        # Step committed with content_hash=None fails when a specific content_hash is required
        engine.commit_step(step=15, delta=StepDelta())
        assert engine.is_step_committed(15) is True
        assert engine.is_step_committed(15, content_hash="anything") is False


def test_step_transaction_checkpoints_and_atomic_rollback() -> None:
    """StepTransaction commits empty-delta checkpoints, ignores uncommitted exits, and rolls back on error."""
    with CanonArcEngine(":memory:") as engine:
        # Exiting without calling tx.commit() must not write a checkpoint
        with engine.step_transaction(step=1, rerun_clean=False) as tx:
            assert tx.context.step == 1
        assert engine.is_step_committed(1) is False
        assert engine.last_committed_step() is None

        # Calling tx.commit() with delta=None still records the checkpoint
        with engine.step_transaction(
            step=1,
            content_hash="tx_hash_1",
            checkpoint_metadata={"source": "init"},
        ) as tx:
            tx.commit()
        assert engine.is_step_committed(1, content_hash="tx_hash_1") is True
        assert engine.is_step_committed(1, metadata_match={"source": "init"}) is True

        # Passing content_hash / checkpoint_metadata directly to tx.commit() overrides transaction defaults
        with engine.step_transaction(step=2, content_hash="old_hash") as tx:
            tx.commit(
                StepDelta(),
                content_hash="tx_hash_2",
                checkpoint_metadata={"source": "commit_arg"},
            )
        assert engine.is_step_committed(2, content_hash="tx_hash_2") is True
        assert engine.is_step_committed(2, metadata_match={"source": "commit_arg"}) is True

        # Exception raised inside StepTransaction must not leave a checkpoint behind
        with pytest.raises(RuntimeError, match="boom"):
            with engine.step_transaction(step=3, content_hash="tx_hash_3") as tx:
                tx.commit(StepDelta())
                raise RuntimeError("boom")
        assert engine.is_step_committed(3) is False
        assert engine.last_committed_step() == 2


def test_rollback_to_cleans_up_step_checkpoints() -> None:
    """rollback_to(step) deletes checkpoints > step, includes them in deleted count, and rewinds head_step."""
    with CanonArcEngine(":memory:") as engine:
        engine.commit_step(step=1, delta=StepDelta(), content_hash="h1")
        engine.commit_step(step=2, delta=StepDelta(), content_hash="h2")
        engine.commit_step(step=3, delta=StepDelta(), content_hash="h3")
        assert engine.last_committed_step() == 3

        deleted = engine.rollback_to(step=1)
        assert deleted == 2
        assert engine.is_step_committed(1, content_hash="h1") is True
        assert engine.is_step_committed(2) is False
        assert engine.is_step_committed(3) is False
        assert engine.last_committed_step() == 1
        tl = engine.store.get_timeline("main")
        assert tl is not None
        assert tl.head_step == 1

        # Rolling back to 0 clears step 1 as well
        deleted_all = engine.rollback_to(step=0)
        assert deleted_all == 1
        assert engine.last_committed_step() is None
        assert engine.is_step_committed(1) is False


def test_branching_lineage_delete_timeline_and_session_parity() -> None:
    """Checkpoints obey copy-on-write branching, child overrides, delete_timeline FK cleanup, and TimelineSession."""
    with CanonArcEngine(":memory:") as engine:
        engine.commit_step(step=1, delta=StepDelta(), content_hash="main_1")
        engine.commit_step(step=2, delta=StepDelta(), content_hash="main_2")

        engine.branch(at_step=2, name="alt")

        # Main advances past fork_step
        engine.commit_step(step=3, delta=StepDelta(), content_hash="main_3")

        alt_session = engine.session("alt")
        # Child inherits parent checkpoints <= fork_step (1, 2) and ignores parent step 3
        assert alt_session.is_step_committed(1, content_hash="main_1") is True
        assert alt_session.is_step_committed(2, content_hash="main_2") is True
        assert alt_session.is_step_committed(3) is False
        assert alt_session.last_committed_step() == 2

        # Child overrides step 2 and commits step 4 without affecting main
        alt_session.commit_step(
            step=2,
            delta=StepDelta(),
            content_hash="alt_2_override",
            checkpoint_metadata={"branch": "alt"},
        )
        alt_session.commit_step(step=4, delta=StepDelta(), content_hash="alt_4")

        assert alt_session.is_step_committed(2, content_hash="alt_2_override") is True
        assert alt_session.is_step_committed(4, content_hash="alt_4") is True
        assert alt_session.last_committed_step() == 4
        assert [cp.step for cp in alt_session.list_step_checkpoints()] == [1, 2, 4]
        assert alt_session.get_step_checkpoint(2) is not None
        assert alt_session.get_step_checkpoint(2).timeline_id == "alt"  # type: ignore[union-attr]

        # Main remains unaffected
        assert engine.is_step_committed(2, content_hash="main_2") is True
        assert engine.is_step_committed(3, content_hash="main_3") is True
        assert engine.is_step_committed(4) is False
        assert engine.last_committed_step() == 3

        # Deleting the child timeline cleans up its step_checkpoints without FK errors
        deleted = engine.delete_timeline("alt")
        assert deleted >= 3  # 2 checkpoints + 1 timeline row
        assert engine.is_step_committed(2, content_hash="main_2") is True


def test_export_snapshot_preserves_checkpoints_up_to_at_step(tmp_path: Path) -> None:
    """export_snapshot copies lineage-resolved checkpoints <= at_step and excludes > at_step."""
    src_db = tmp_path / "src.sqlite"
    snap_db = tmp_path / "snap.sqlite"

    with CanonArcEngine(src_db) as engine:
        engine.commit_step(step=1, delta=StepDelta(), content_hash="m1", checkpoint_metadata={"v": 1})
        engine.commit_step(step=2, delta=StepDelta(), content_hash="m2")
        engine.branch(at_step=2, name="route_b")
        engine.commit_step(step=3, delta=StepDelta(), content_hash="b3", timeline_id="route_b")
        engine.commit_step(step=4, delta=StepDelta(), content_hash="b4", timeline_id="route_b")
        engine.export_snapshot(snap_db, at_step=3, timeline_id="route_b")

    with CanonArcEngine(snap_db) as snap_engine:
        assert snap_engine.last_committed_step() == 3
        assert snap_engine.is_step_committed(1, content_hash="m1", metadata_match={"v": 1}) is True
        assert snap_engine.is_step_committed(2, content_hash="m2") is True
        assert snap_engine.is_step_committed(3, content_hash="b3") is True
        assert snap_engine.is_step_committed(4) is False
        tl = snap_engine.store.get_timeline("main")
        assert tl is not None
        assert tl.head_step == 3


def test_legacy_database_migrates_cleanly(tmp_path: Path) -> None:
    """Opening an existing SQLite DB created without step_checkpoints adds the table automatically."""
    db_path = tmp_path / "legacy.sqlite"
    with CanonArcEngine(db_path) as engine:
        engine.register_entity(EntityRecord(entity_id="char_a", canonical_name="Alice"), step=0)
        engine.commit_step(
            step=1,
            delta=StepDelta(entities=[EntityDelta(entity_id="char_a", state={"mood": "happy"})]),
        )

    # Simulate a pre-checkpoint database by dropping step_checkpoints
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE step_checkpoints")
    conn.commit()
    conn.close()

    with CanonArcEngine(db_path) as migrated:
        assert migrated.last_committed_step() is None
        assert migrated.is_step_committed(1) is False
        migrated.commit_step(step=2, delta=StepDelta(), content_hash="after_migration")
        assert migrated.last_committed_step() == 2
        assert migrated.is_step_committed(2, content_hash="after_migration") is True


def test_custom_store_without_checkpoint_methods_degrades_gracefully() -> None:
    """Custom MemoryStore backends without CheckpointStore methods continue working for normal commits."""

    class MinimalStore:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def apply_delta(self, step: int, delta: StepDelta, *, timeline_id: str = "main", episode_embeddings: dict[str, bytes] | None = None) -> list[Any]:
            return []

    store = MinimalStore()
    engine = CanonArcEngine(store=store)  # type: ignore[arg-type]
    # Normal commit_step without checkpoint args succeeds silently
    engine.commit_step(step=1, delta=StepDelta())

    # Explicitly requesting checkpoint hash/metadata or querying checkpoints raises NotImplementedError
    with pytest.raises(NotImplementedError):
        engine.commit_step(step=2, delta=StepDelta(), content_hash="h2")
    with pytest.raises(NotImplementedError):
        engine.is_step_committed(1)
    with pytest.raises(NotImplementedError):
        engine.last_committed_step()
    with pytest.raises(NotImplementedError):
        engine.get_step_checkpoint(1)
    with pytest.raises(NotImplementedError):
        engine.list_step_checkpoints()
