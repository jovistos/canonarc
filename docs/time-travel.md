# Causal Time Travel: Branching, Rollback & Snapshots

In narrative fiction, game engines (like Ren'Py), and simulations, time is discrete and branchable.

---

## 1. Zero-Copy Branching (`branch(at_step, name)`)

When a player reaches a branching choice (e.g., *Accept the Dark Pact* vs. *Refuse*), `narrative-memory` forks the timeline with zero data copying:

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
- Revisions `0..20` remain on `main` and are shared transparently.
- New revisions committed to `dark_pact_route` at `step > 20` live strictly on the branch.
- Any future changes on `main` at `step > 20` are blocked by `parent.step <= child.fork_step`.

---

## 2. Instant Rollback (`rollback_to(step)`)

In visual novels, users frequently scroll the mouse-wheel backward to reread previous lines or explore alternate choices.

`rollback_to(step)` deletes all revisions and exemplars at `> step` on the target timeline:

```python
# Rolls back 'main' timeline to step 15
deleted_count = engine.rollback_to(step=15, timeline_id="main")
```

### SQL Implementation
```sql
DELETE FROM graph_revisions WHERE timeline_id = :tl AND step > :step;
DELETE FROM voice_exemplars WHERE timeline_id = :tl AND step > :step;
```
Execution takes **< 1ms**, making it suitable for 60 FPS game engine main loops.

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

The exported database materializes the exact active world state at `step=50` flattened into a fresh `step=0` baseline.
