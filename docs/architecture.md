# Architecture & Storage Philosophy

`narrative-memory` models narrative worlds as a **Tier 1–5 superset** organized along a causal timeline tree.

---

## 1. The 5-Tier Architectural Superset

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 5: Gated Voice Bank (Empirical Style & Syntax Anchors) │
├─────────────────────────────────────────────────────────────┤
│ Tier 4: Temporal Versioning & 1-Hop Alias Coreference       │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Rolling State Deltas (Resolve-on-Write Materialized)│
├─────────────────────────────────────────────────────────────┤
│ Tier 2: Pairwise Stance & Register (Asymmetric A -> B)      │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: Base Entity Attributes (Immutable Archetypes/Rules) │
└─────────────────────────────────────────────────────────────┘
```

### Tier 1: Base Entity Attributes
- Fixed metadata attached to entities on creation at `step = 0`.
- Includes grammatical properties (declension class, gender), archetypes, faction affiliations, and target localized renderings.
- Immutable by default, but branch-safe if patched later.

### Tier 2: Pairwise Register & Stance
- Interpersonal relationships are **directed** (`source_id -> target_id`).
- Arthur's register toward Merlin (`informal, contemptuous`) is independent of Merlin's register toward Arthur (`formal, patient`).

### Tier 3: Rolling State Deltas
- Ingestion pipeline extracts JSON diffs (`StepDelta`) after each scene or chunk.
- **Resolve-on-Write**: Instead of storing unmerged deltas, `apply_delta()` resolves the active state at `step` and materializes the complete state dictionary into a new revision row.

### Tier 4: Temporal Versioning & 1-Hop Expansion
- All revisions are indexed by `step: int`.
- The query engine resolves active states at `step < current_step` (no future spoiler leakage).
- Evaluates 1-hop aliases (`"the old wizard" -> char_merlin`) and pulls 1-hop edge neighbors connected to active characters in the scene.

### Tier 5: Gated Voice Bank
- Stores empirical past dialogue lines.
- Filtered by active graph properties (`speaker_id`, `register`, `tone`, `step < current_step`).
- Protected by a 2-stage anti-drift gate to prevent bad generations from compounding.

---

## 2. Storage Architecture: Unified 3-Table SQLite Schema

Rather than spreading state across multiple tables, `narrative-memory` uses three lean tables:

```sql
-- 1. Copy-on-Write Timeline Tree
CREATE TABLE timelines (
    timeline_id TEXT PRIMARY KEY,
    parent_id   TEXT REFERENCES timelines(timeline_id),
    fork_step   INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

-- 2. Unified Append-Only Materialized Revisions
CREATE TABLE graph_revisions (
    revision_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    timeline_id  TEXT NOT NULL REFERENCES timelines(timeline_id),
    step         INTEGER NOT NULL,
    record_type  TEXT NOT NULL, -- 'entity' | 'edge' | 'alias'
    subject_key  TEXT NOT NULL, -- 'char_arthur', 'char_arthur->char_merlin', 'the old wizard'
    source_id    TEXT,          -- entity_id for entity/alias; source_id for edge
    target_id    TEXT,          -- target_id for edge
    is_tombstone INTEGER NOT NULL DEFAULT 0,
    payload      TEXT NOT NULL  -- Full materialized JSON
);

-- 3. Tier 5 Voice Exemplars
CREATE TABLE voice_exemplars (
    exemplar_id           TEXT PRIMARY KEY,
    timeline_id           TEXT NOT NULL REFERENCES timelines(timeline_id),
    step                  INTEGER NOT NULL,
    speaker_id            TEXT NOT NULL,
    addressee_id          TEXT,
    text                  TEXT NOT NULL,
    source_text           TEXT NOT NULL DEFAULT '',
    metadata              TEXT NOT NULL DEFAULT '{}',
    quality_score         REAL NOT NULL,
    distinctiveness_score REAL NOT NULL,
    embedding             BLOB NOT NULL
);
```

---

## 3. The Recursive CTE Lineage Engine

Point-in-time state resolution is executed in a single recursive CTE:

```sql
WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
    SELECT timeline_id, parent_id, :cutoff_step, 0
    FROM timelines
    WHERE timeline_id = :timeline_id
    UNION ALL
    SELECT
        t.timeline_id,
        t.parent_id,
        MIN(l.max_step, t_child.fork_step),
        l.depth + 1
    FROM lineage l
    JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
    JOIN timelines t       ON t.timeline_id = l.parent_id
    WHERE l.parent_id IS NOT NULL
),
ranked_revisions AS (
    SELECT
        r.*,
        ROW_NUMBER() OVER (
            PARTITION BY r.record_type, r.subject_key
            ORDER BY l.depth ASC, r.step DESC, r.revision_id DESC
        ) AS rn
    FROM graph_revisions r
    JOIN lineage l
      ON r.timeline_id = l.timeline_id
     AND r.step <= l.max_step
)
SELECT record_type, subject_key, source_id, target_id, step AS valid_from_step, payload
FROM ranked_revisions
WHERE rn = 1 AND is_tombstone = 0;
```

### Why this query is fast and robust:
1. `depth ASC`: Revisions made on the child branch (`depth = 0`) strictly supersede parent revisions.
2. `r.step <= l.max_step`: Parent revisions committed *after* `fork_step` are completely invisible to the child branch.
3. `rn = 1`: Picks the latest revision for each key in sub-millisecond execution.
