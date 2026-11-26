from __future__ import annotations

import pytest

from canonarc import (
    BaseEngineHooks,
    CanonArcEngine,
    EntityDelta,
    EntityRecord,
    ResumePlan,
    SchemaPolicy,
    StepDelta,
    TimelineBranchConflictError,
    TranslationPreset,
)


class _RecordingHooks(BaseEngineHooks):
    def __init__(self) -> None:
        self.rollbacks: list[tuple[int, str, int]] = []

    def on_rollback(self, step: int, timeline_id: str, deleted_rows: int) -> None:
        self.rollbacks.append((step, timeline_id, deleted_rows))


def _commit_entity(
    engine: CanonArcEngine,
    step: int,
    entity_id: str,
    name: str,
    content_hash: str,
    *,
    metadata: dict | None = None,
    timeline_id: str | None = None,
) -> None:
    engine.commit_step(
        step,
        delta=StepDelta(
            entities=[
                EntityDelta(
                    entity_id=entity_id,
                    kind="character",
                    canonical_name=name,
                )
            ]
        ),
        content_hash=content_hash,
        checkpoint_metadata=metadata,
        timeline_id=timeline_id,
    )


def test_plan_resume_empty_and_clean_interruption() -> None:
    with CanonArcEngine(":memory:") as engine:
        # Empty DB (1-based)
        plan0 = engine.plan_resume([(1, "h1"), (2, "h2"), (3, "h3")])
        assert isinstance(plan0, ResumePlan)
        assert plan0.causal is True
        assert plan0.reusable_steps == []
        assert plan0.pending_steps == [1, 2, 3]
        assert plan0.missing_steps == [1, 2, 3]
        assert plan0.stale_steps == []
        assert plan0.first_stale_step is None
        assert plan0.rollback_to_step is None
        assert plan0.needs_rollback is False
        assert plan0.is_up_to_date is False

        # Empty DB (0-based) must not trigger false rollback
        plan_zero = engine.plan_resume([(0, "h0"), (1, "h1")])
        assert plan_zero.rollback_to_step is None
        assert plan_zero.needs_rollback is False

        # Commit steps 1..2, leave 3..4 missing (clean interruption)
        _commit_entity(engine, 1, "e1", "Alice", "h1")
        _commit_entity(engine, 2, "e2", "Bob", "h2")

        plan_resume = engine.plan_resume([(1, "h1"), (2, "h2"), (3, "h3"), (4, "h4")])
        assert plan_resume.reusable_steps == [1, 2]
        assert plan_resume.pending_steps == [3, 4]
        assert plan_resume.missing_steps == [3, 4]
        assert plan_resume.stale_steps == []
        assert plan_resume.invalidated_steps == []
        assert plan_resume.first_stale_step is None
        assert plan_resume.rollback_to_step is None
        assert plan_resume.needs_rollback is False

        # Fully up-to-date
        plan_done = engine.plan_resume([(1, "h1"), (2, "h2")])
        assert plan_done.is_up_to_date is True
        assert plan_done.reusable_steps == [1, 2]
        assert plan_done.pending_steps == []
        assert plan_done.rollback_to_step is None


def test_plan_resume_causal_edit_dry_run_and_apply_rollback() -> None:
    hooks = _RecordingHooks()
    with CanonArcEngine(":memory:", hooks=[hooks]) as engine:
        for i in range(1, 6):
            _commit_entity(engine, i, f"e{i}", f"Char{i}", f"h{i}")

        edited = [(1, "h1"), (2, "h2"), (3, "h3_EDITED"), (4, "h4"), (5, "h5")]

        # Dry run (default apply_rollback=False) does not mutate DB
        dry = engine.plan_resume(edited)
        assert dry.causal is True
        assert dry.reusable_steps == [1, 2]
        assert dry.first_stale_step == 3
        assert dry.stale_steps == [3]
        assert dry.invalidated_steps == [4, 5]
        assert dry.pending_steps == [3, 4, 5]
        assert dry.steps_to_purge == [3, 4, 5]
        assert dry.rollback_to_step == 2
        assert dry.needs_rollback is True
        assert dry.rollback_applied is False
        assert dry.deleted_rows == 0
        assert engine.is_step_committed(5, content_hash="h5") is True
        assert hooks.rollbacks == []

        # Apply rollback
        applied = engine.plan_resume(edited, apply_rollback=True)
        assert applied.rollback_to_step == 2
        assert applied.rollback_applied is True
        assert applied.needs_rollback is False
        assert applied.deleted_rows > 0
        assert len(hooks.rollbacks) == 1
        assert hooks.rollbacks[0][0] == 2
        assert hooks.rollbacks[0][1] == "main"

        assert engine.is_step_committed(2, content_hash="h2") is True
        assert engine.is_step_committed(3) is False
        assert engine.is_step_committed(4) is False
        assert engine.is_step_committed(5) is False
        assert engine.store.get_active_entity("e2", timeline_id="main", cutoff_step=10) is not None
        assert engine.store.get_active_entity("e3", timeline_id="main", cutoff_step=10) is None


def test_plan_resume_first_step_stale_preserves_step_zero_seed() -> None:
    with CanonArcEngine(":memory:") as engine:
        engine.register_entity(
            EntityRecord(entity_id="seed_0", kind="term", canonical_name="GlossaryTerm"),
            step=0,
        )
        _commit_entity(engine, 1, "e1", "Alice", "h1")
        _commit_entity(engine, 2, "e2", "Bob", "h2")

        plan = engine.plan_resume([(1, "h1_new"), (2, "h2")], apply_rollback=True)
        assert plan.reusable_steps == []
        assert plan.first_stale_step == 1
        assert plan.rollback_to_step == 0
        assert plan.rollback_applied is True
        assert engine.is_step_committed(1) is False
        assert engine.is_step_committed(2) is False
        # Step 0 seed entity is preserved!
        assert engine.store.get_active_entity("seed_0", timeline_id="main", cutoff_step=10) is not None


def test_plan_resume_shortened_input_prune_tail() -> None:
    with CanonArcEngine(":memory:") as engine:
        for i in range(1, 6):
            _commit_entity(engine, i, f"e{i}", f"Char{i}", f"h{i}")

        # prune_tail=False keeps steps 4..5 untouched
        keep = engine.plan_resume([(1, "h1"), (2, "h2"), (3, "h3")], prune_tail=False)
        assert keep.is_up_to_date is True
        assert keep.orphaned_steps == []
        assert keep.rollback_to_step is None

        # prune_tail=True (default) rolls back orphaned tail 4..5
        pruned = engine.plan_resume([(1, "h1"), (2, "h2"), (3, "h3")], apply_rollback=True)
        assert pruned.reusable_steps == [1, 2, 3]
        assert pruned.pending_steps == []
        assert pruned.orphaned_steps == [4, 5]
        assert pruned.steps_to_purge == [4, 5]
        assert pruned.rollback_to_step == 3
        assert pruned.rollback_applied is True
        assert engine.is_step_committed(3, content_hash="h3") is True
        assert engine.is_step_committed(4) is False
        assert engine.is_step_committed(5) is False


def test_plan_resume_detects_partial_uncheckpointed_writes() -> None:
    with CanonArcEngine(":memory:") as engine:
        _commit_entity(engine, 1, "e1", "Alice", "h1")
        _commit_entity(engine, 2, "e2", "Bob", "h2")
        # Simulate crash after writing graph_revisions at step 3 without recording checkpoint
        engine.store.apply_delta(
            3,
            StepDelta(entities=[EntityDelta(entity_id="e3_partial", kind="character", canonical_name="Partial")]),
            timeline_id="main",
        )
        assert engine.is_step_committed(3) is False

        plan = engine.plan_resume([(1, "h1"), (2, "h2"), (3, "h3")], apply_rollback=True)
        assert plan.reusable_steps == [1, 2]
        assert plan.pending_steps == [3]
        assert plan.rollback_to_step == 2
        assert plan.rollback_applied is True
        assert engine.store.get_active_entity("e3_partial", timeline_id="main", cutoff_step=10) is None


def test_plan_resume_non_causal_lexicon_only_purges_only_stale_steps() -> None:
    with CanonArcEngine(
        ":memory:", preset=TranslationPreset(policy=SchemaPolicy.lexicon_only())
    ) as engine:
        for i in range(1, 5):
            _commit_entity(engine, i, f"term_{i}", f"Term{i}", f"h{i}")

        edited = [(1, "h1"), (2, "h2_EDITED"), (3, "h3"), (4, "h4")]

        # Defaults causal=False for LEXICON_ONLY
        dry = engine.plan_resume(edited)
        assert dry.causal is False
        assert dry.reusable_steps == [1, 3, 4]
        assert dry.pending_steps == [2]
        assert dry.stale_steps == [2]
        assert dry.invalidated_steps == []
        assert dry.first_stale_step == 2
        assert dry.rollback_to_step is None
        assert dry.needs_rollback is True
        assert dry.steps_to_purge == [2]

        # apply_rollback=True purges ONLY step 2 via delete_steps, keeping 1, 3, 4 intact
        applied = engine.plan_resume(edited, apply_rollback=True)
        assert applied.rollback_applied is True
        assert applied.needs_rollback is False
        assert applied.deleted_rows > 0

        assert engine.is_step_committed(1, content_hash="h1") is True
        assert engine.is_step_committed(2) is False
        assert engine.is_step_committed(3, content_hash="h3") is True
        assert engine.is_step_committed(4, content_hash="h4") is True
        assert engine.store.get_active_entity("term_2", timeline_id="main", cutoff_step=10) is None
        assert engine.store.get_active_entity("term_3", timeline_id="main", cutoff_step=10) is not None
        assert engine.store.get_active_entity("term_4", timeline_id="main", cutoff_step=10) is not None


def test_plan_resume_metadata_match_and_causal_override() -> None:
    with CanonArcEngine(":memory:") as engine:
        _commit_entity(engine, 1, "e1", "Alice", "h1", metadata={"model": "v1"})
        _commit_entity(engine, 2, "e2", "Bob", "h2", metadata={"model": "v1"})
        _commit_entity(engine, 3, "e3", "Cara", "h3", metadata={"model": "v2"})

        # Metadata mismatch at step 1..2 when requiring model=v2
        plan = engine.plan_resume(
            [(1, "h1"), (2, "h2"), (3, "h3")],
            metadata_match={"model": "v2"},
        )
        assert plan.reusable_steps == []
        assert plan.first_stale_step == 1
        assert plan.stale_steps == [1, 2]
        assert plan.invalidated_steps == [3]
        assert plan.rollback_to_step == 0

        # Explicit causal=False override on a FULL engine
        non_causal = engine.plan_resume(
            [(1, "h1"), (2, "h2"), (3, "h3")],
            metadata_match={"model": "v2"},
            causal=False,
        )
        assert non_causal.causal is False
        assert non_causal.reusable_steps == [3]
        assert non_causal.pending_steps == [1, 2]
        assert non_causal.rollback_to_step is None


def test_plan_resume_branched_timeline_and_fork_conflict() -> None:
    with CanonArcEngine(":memory:") as engine:
        _commit_entity(engine, 1, "e1", "Alice", "h1")
        _commit_entity(engine, 2, "e2", "Bob", "h2")
        _commit_entity(engine, 3, "e3", "Cara", "h3")

        alt = engine.session("main").branch(at_step=2, name="alt")
        _commit_entity(engine, 3, "e3_alt", "CaraAlt", "h3_alt", timeline_id="alt")
        _commit_entity(engine, 4, "e4_alt", "DanAlt", "h4_alt", timeline_id="alt")

        # Edit after fork_step (step 3 changed on alt) -> rolls back alt to 2 without touching main
        ok_plan = alt.plan_resume(
            [(1, "h1"), (2, "h2"), (3, "h3_alt_v2"), (4, "h4_alt")],
            apply_rollback=True,
        )
        assert ok_plan.timeline_id == "alt"
        assert ok_plan.rollback_to_step == 2
        assert ok_plan.fork_conflict_step is None
        assert ok_plan.rollback_applied is True
        assert alt.is_step_committed(3) is False
        assert engine.is_step_committed(3, content_hash="h3") is True

        # Edit before fork_step (step 2 changed on alt, which forked at 2) -> watermark=1 < fork_step=2
        conflict_dry = alt.plan_resume([(1, "h1"), (2, "h2_NEW"), (3, "h3_alt_v2")])
        assert conflict_dry.rollback_to_step == 1
        assert conflict_dry.fork_conflict_step == 2
        assert conflict_dry.rollback_applied is False

        with pytest.raises(TimelineBranchConflictError, match="forked at step 2"):
            alt.plan_resume(
                [(1, "h1"), (2, "h2_NEW"), (3, "h3_alt_v2")],
                apply_rollback=True,
            )

        # Rolling back main past child fork_step=2 raises unless force=True
        with pytest.raises(TimelineBranchConflictError):
            engine.plan_resume([(1, "h1"), (2, "h2_NEW"), (3, "h3")], apply_rollback=True)

        forced = engine.plan_resume(
            [(1, "h1"), (2, "h2_NEW"), (3, "h3")],
            apply_rollback=True,
            force=True,
        )
        assert forced.rollback_applied is True
        assert forced.rollback_to_step == 1


def test_plan_resume_validates_step_ordering() -> None:
    with CanonArcEngine(":memory:") as engine:
        with pytest.raises(ValueError, match="Duplicate step"):
            engine.plan_resume([(1, "a"), (1, "b")])
        with pytest.raises(ValueError, match="strictly ascending"):
            engine.plan_resume([(2, "a"), (1, "b")])
