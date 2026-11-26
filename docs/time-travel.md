# Causal Time Travel: Branching, Rollback & Snapshots

In narrative fiction, game engines (like Ren'Py), and simulations, time is discrete, branchable, and rewindable across both property graph states and episodic events.

---

## 1. Zero-Copy Branching (`branch(at_step, name)`)

When a player reaches a branching choice (e.g., *Accept the Dark Pact* vs. *Refuse*), `canonarc` forks the timeline with zero data copying:

```python
# Main storyline proceeds
engine.commit_step(step=20, delta=...)

# Branch at Step 20
engine.branch(at_step=20, name="dark_pact_route")

# Switch default active timeline
engine.checkout("dark_pact_route")
```

### How Zero-Copy Works
- A new row is inserted into `timelines`: `(timeline_id='dark_pact_route', parent_id='main', fork_step=20)`.
- Revisions, voice exemplars, and episodes from `0..20` remain on `main` and are shared transparently along the recursive lineage CTE.
- New revisions and episodes committed to `dark_pact_route` at `step > 20` live strictly on the branch.
- Any future changes on `main` at `step > 20` are blocked by `parent.step <= child.fork_step`.

---

## 2. Instant Rollback (`rollback_to(step)`)

In visual novels, users frequently scroll the mouse-wheel backward to reread previous lines or explore alternate choices.

`rollback_to(step)` deletes all graph revisions, voice exemplars, and episodic records at `> step` on the target timeline. If child timelines forked from `timeline_id` at `fork_step > step`, `rollback_to` raises `TimelineBranchConflictError` to prevent corrupting descendant histories unless `force=True` is passed:

```python
# Rolls back 'main' timeline to step 15
deleted_count = engine.rollback_to(step=15, timeline_id="main")
```

### SQL Implementation
```sql
DELETE FROM graph_revisions WHERE timeline_id = :tl AND step > :step;
DELETE FROM voice_exemplars WHERE timeline_id = :tl AND step > :step;
DELETE FROM episodes WHERE timeline_id = :tl AND step > :step;
DELETE FROM episode_epistemics WHERE timeline_id = :tl AND step > :step;
DELETE FROM step_checkpoints WHERE timeline_id = :tl AND step > :step;
```
Execution takes **< 1ms**, making it suitable for 60 FPS game engine main loops.

### Step Checkpoints & Resumability
Every `commit_step()` records a lineage-aware row in `step_checkpoints(timeline_id, step, content_hash, metadata_json)`—even for empty-delta steps—so `TimelineInfo.head_step` advances accurately and resumable pipelines can query `engine.is_step_committed(step, content_hash=...)` and `engine.last_committed_step()`. Rolling back with `rollback_to(step)` automatically removes checkpoints at `> step`.

---

## 3. Comparing Counterfactual Branches (`diff_timelines`)

To understand how two routes or simulation runs diverged:

```python
diff = engine.diff_timelines("main", "dark_pact_route", at_step=50)

print("Entities changed:", diff["changed"])
print("Keys only in dark pact:", diff["only_in_b"])
```

---

## 4. Standalone Save-Slot Snapshots (`export_snapshot`)

To create an isolated `.sqlite` save file (e.g., `save_slot_01.sqlite`) that can be emailed, backed up, or loaded independently:

```python
engine.export_snapshot("saves/slot_01.sqlite", at_step=50, timeline_id="dark_pact_route")
```

The exported database materializes the exact active world state and all lineage episodes up to `step=50` flattened into a fresh `step=0` baseline (preserving `revision_count` across the snapshot).

---

## 5. Deleting & Pruning Timelines (`delete_timeline`)

Remove finished or abandoned branches while keeping parent histories intact:

```python
# Delete a leaf branch (raises TimelineBranchConflictError if it has children)
engine.delete_timeline("dark_pact_route")

# Recursively delete a branch and all descendant branches
engine.delete_timeline("dark_pact_route", cascade=True)
```

Note: Deleting `"main"` is disallowed unless handled explicitly; deleting the currently checked-out branch automatically resets `active_timeline` back to `"main"`.

---

## 6. Concurrency-Safe Timeline Sessions (`use_timeline` / `session`)

Avoid mutating global `engine.checkout(...)` state across concurrent async tasks or worker threads:

```python
# Task-local ContextVar scoping (ideal for async web handlers)
with engine.use_timeline("dark_pact_route") as tl_session:
    ctx = engine.retrieve_context(step=30, mentions=["Arthur"])
    tl_session.commit_step(step=30, delta=...)

# Explicit immutable timeline handle
route_a = engine.session("route_a")
ctx_a = route_a.retrieve_context(step=30, mentions=["Arthur"])
```

---

## 7. Non-Linear Narrative Time & Flashbacks

`canonarc` distinguishes **discourse order** (`step: int`) from **diegetic story order** (`story_time`, `chapter_id`, `scene_id`):

```python
engine.commit_step(
    step=40,
    delta=StepDelta(
        is_flashback=True,
        story_time="10 years earlier",
        chapter_id="ch_04",
        scene_id="flashback_citadel",
        episodes=[
            EpisodeRecord(
                episode_id="ep_oath_past",
                summary="Young Arthur swore the Secret Citadel Oath.",
                participants={"char_arthur": "actor"},
            )
        ],
    ),
)
```

When `is_flashback=True`, existing entities and edges at step 40 are **never rolled backward** to historical state, while episodic events, epistemic records, and brand-new flashback-only entities are cleanly indexed into causal memory.

---

## 8. CLI Timeline Inspection

Inspect save files without writing Python scripts:

```bash
canonarc tree saves/story.sqlite
canonarc inspect saves/story.sqlite --timeline dark_pact_route --step 50 --pov char_arthur
canonarc diff saves/story.sqlite main dark_pact_route --step 50
canonarc history saves/story.sqlite char_arthur --timeline dark_pact_route
```
