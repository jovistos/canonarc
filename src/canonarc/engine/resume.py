from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..models import ResumePlan, StepCheckpoint
from ..store import TimelineBranchConflictError

if TYPE_CHECKING:
    from .core import CanonArcEngine


def checkpoint_matches(
    cp: StepCheckpoint | None,
    *,
    content_hash: str | None,
    metadata_match: Mapping[str, Any] | None,
) -> bool:
    if cp is None:
        return False
    if content_hash is not None and cp.content_hash != content_hash:
        return False
    if metadata_match is not None:
        for key, expected in metadata_match.items():
            if key not in cp.metadata or cp.metadata[key] != expected:
                return False
    return True


def plan_resume_impl(
    engine: CanonArcEngine,
    step_hashes: Sequence[tuple[int, str]],
    *,
    causal: bool | None = None,
    metadata_match: Mapping[str, Any] | None = None,
    timeline_id: str | None = None,
    apply_rollback: bool = False,
    force: bool = False,
    prune_tail: bool = True,
) -> ResumePlan:
    """Plans which ordered (step, content_hash) pairs can be reused vs. re-run."""
    tl = timeline_id or engine.active_timeline
    is_causal = (
        bool(causal)
        if causal is not None
        else any(engine._policy.has_tier(t) for t in (2, 3, 5, 6))
    )

    normalized: list[tuple[int, str]] = []
    prev_step: int | None = None
    for raw_step, content_hash in step_hashes:
        step = int(raw_step)
        if prev_step is not None:
            if step == prev_step:
                raise ValueError(f"Duplicate step {step} in step_hashes.")
            if step < prev_step:
                raise ValueError(
                    f"step_hashes must be ordered by strictly ascending step numbers "
                    f"(got {step} after {prev_step})."
                )
        normalized.append((step, content_hash))
        prev_step = step

    checkpoints = {cp.step: cp for cp in engine.list_step_checkpoints(timeline_id=tl)}
    tl_info = engine.store.get_timeline(tl)
    fork_step = tl_info.fork_step if tl_info is not None else None
    head_step = tl_info.head_step if tl_info is not None else None

    if prune_tail:
        if normalized:
            max_planned = normalized[-1][0]
            orphaned_steps = sorted(s for s in checkpoints if s > max_planned)
        else:
            orphaned_steps = sorted(checkpoints.keys())
    else:
        orphaned_steps = []

    reusable_steps: list[int] = []
    pending_steps: list[int] = []
    missing_steps: list[int] = []
    stale_steps: list[int] = []
    invalidated_steps: list[int] = []
    first_stale_step: int | None = None
    rollback_to_step: int | None = None
    fork_conflict_step: int | None = None
    rollback_applied = False
    deleted_rows = 0

    if is_causal:
        diverged = False
        for step, content_hash in normalized:
            cp = checkpoints.get(step)
            matched = checkpoint_matches(
                cp, content_hash=content_hash, metadata_match=metadata_match
            )
            if not diverged:
                if matched:
                    reusable_steps.append(step)
                else:
                    diverged = True
                    pending_steps.append(step)
                    if cp is not None:
                        stale_steps.append(step)
                        first_stale_step = step
                    else:
                        missing_steps.append(step)
            else:
                pending_steps.append(step)
                if cp is not None:
                    if matched:
                        invalidated_steps.append(step)
                    else:
                        stale_steps.append(step)
                        if first_stale_step is None:
                            first_stale_step = step
                else:
                    missing_steps.append(step)

        if reusable_steps:
            watermark: int | None = reusable_steps[-1]
        elif normalized:
            watermark = normalized[0][0] - 1
        elif orphaned_steps:
            watermark = orphaned_steps[0] - 1
        else:
            watermark = None

        if watermark is not None:
            has_stale_tail = bool(stale_steps or invalidated_steps or orphaned_steps)
            has_uncheckpointed_writes = (
                head_step is not None
                and head_step > watermark
                and (watermark >= 0 or head_step > 0)
                and (fork_step is None or head_step > fork_step)
                and (prune_tail or (bool(normalized) and head_step <= normalized[-1][0]))
            )
            if has_stale_tail or has_uncheckpointed_writes:
                rollback_to_step = watermark

        if (
            rollback_to_step is not None
            and fork_step is not None
            and rollback_to_step < fork_step
        ):
            fork_conflict_step = fork_step
            if apply_rollback:
                parent_hint = (
                    f" Roll back parent '{tl_info.parent_id}' or fork a new branch at step {rollback_to_step}."
                    if tl_info is not None and tl_info.parent_id
                    else ""
                )
                raise TimelineBranchConflictError(
                    f"Cannot roll back branch '{tl}' to step {rollback_to_step}: "
                    f"branch forked at step {fork_step}.{parent_hint}"
                )

        if apply_rollback and rollback_to_step is not None:
            deleted_rows = engine.rollback_to(rollback_to_step, timeline_id=tl, force=force)
            rollback_applied = True
    else:
        for step, content_hash in normalized:
            cp = checkpoints.get(step)
            if checkpoint_matches(
                cp, content_hash=content_hash, metadata_match=metadata_match
            ):
                reusable_steps.append(step)
            else:
                pending_steps.append(step)
                if cp is not None:
                    stale_steps.append(step)
                    if first_stale_step is None:
                        first_stale_step = step
                else:
                    missing_steps.append(step)

        if apply_rollback:
            purge_steps = sorted(set(stale_steps) | set(orphaned_steps))
            if purge_steps:
                delete_steps_fn = getattr(engine.store, "delete_steps", None)
                if callable(delete_steps_fn):
                    deleted_rows = int(delete_steps_fn(purge_steps, timeline_id=tl, force=force))
                    rollback_applied = True

    return ResumePlan(
        timeline_id=tl,
        causal=is_causal,
        reusable_steps=reusable_steps,
        pending_steps=pending_steps,
        missing_steps=missing_steps,
        stale_steps=stale_steps,
        invalidated_steps=invalidated_steps,
        orphaned_steps=orphaned_steps,
        first_stale_step=first_stale_step,
        rollback_to_step=rollback_to_step,
        fork_conflict_step=fork_conflict_step,
        rollback_applied=rollback_applied,
        deleted_rows=deleted_rows,
    )
