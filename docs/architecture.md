# Architecture & Storage Philosophy

`canonarc` models narrative worlds as a **Tier 1–6 superset** organized along a causal timeline tree.

---

## 1. The 6-Tier Architectural Superset

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 6: Episodic Event Ledger (Causal & Epistemic History)  │
├─────────────────────────────────────────────────────────────┤
│ Tier 5: Gated Voice Bank (Empirical Style & Syntax Anchors) │
├─────────────────────────────────────────────────────────────┤
│ Tier 4: Temporal Versioning & 1-Hop Alias Coreference       │
├─────────────────────────────────────────────────────────────┤
│ Tier 3: Rolling State Deltas & Windowed Trajectories        │
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

### Tier 3: Rolling State Deltas & Windowed Trajectory
- Ingestion pipeline extracts JSON diffs (`StepDelta`) after each scene or chunk.
- **Resolve-on-Write**: Instead of storing unmerged deltas, `apply_delta()` resolves the active state at `step` and materializes the complete state dictionary into a new revision row.
- **Windowed Trajectory Tracking**: The query engine uses `LEAD(...) OVER` and `COUNT(*) OVER` window functions to attach `prior_state`, `prior_step`, and `revision_count` directly to active entity and edge records. State resets (e.g. Arthur's mood cooling from `"furious"` back to `"calm"`) do not erase knowledge of the prior emotional state.

### Tier 4: Temporal Versioning & 1-Hop Expansion
- All revisions are indexed by `step: int`.
- The query engine resolves active states at `step < current_step` (no future spoiler leakage).
- Evaluates 1-hop aliases (`"the old wizard" -> char_merlin`) and partitions the social graph into **co-present scene entities/edges** (`StepContext.entities`, `StepContext.active_edges`—matched in text, explicitly passed, or sharing the active `location_id`) vs. **1-hop background neighbors/edges** (`StepContext.related_entities`, `StepContext.background_edges`), preventing off-screen characters from polluting scene presence, `| ignorant:` dramatic irony tags, or causal episode pinning.

### Tier 5: Gated Voice Bank
- Stores empirical past dialogue lines.
- Filtered by active graph properties (`speaker_id`, `register`, `tone`, `step < current_step`).
- Protected by a 2-stage anti-drift gate to prevent bad generations from compounding.

### Tier 6: Episodic Event Ledger & Epistemic Visibility
- Stores discrete, N-ary plot events (betrayals, arguments, item transfers, discoveries) anchored at `step`, with optional non-linear narrative coordinates (`story_time`, `chapter_id`, `scene_id`) and flashback-safe ingestion (`StepDelta.is_flashback=True`).
- **Causal Provenance**: Entities and edges point to triggering episodes via `caused_by_episode_id` and `reason`, ensuring the LLM knows *why* a character is furious or untrusting.
- **Epistemic Tracking, Retraction & POV Projection**:
  - Tracks who witnessed an event (`"direct"`), who learned secondhand (`"told"`), who holds a false belief (`"deceived"` with `believed_summary`), and who later forgot or had memory erased (`"forgotten"` / `"suppressed"` via `InformDelta`).
  - When `pov_entity_ids` is specified in `retrieve_context()`, the engine projects Tiers 1–6 through the observer's epistemic lens (`pov_mode="any"` or shared intersection `pov_mode="all"`): unpierced `disguise` aliases (`AliasRecord.known_to`) are masked to `persona_name`, `private_state` and unwitnessed causal provenance (`reason` / `caused_by_episode_id`) are redacted, `belief_state` edges override stale ground-truth state, and `facade` edges hide private `stance` edges while driving outward voice exemplar selection.
  - **Deceived-POV Vector Isolation**: For deceived observers, `DefaultEpisodeRanker` embeds `believed_summary` rather than ground-truth `emb_bytes` so secret event vectors never leak into semantic retrieval.
- **Hybrid 4-Signal Scorer & Compaction**: `DefaultEpisodeRanker` (configured via `EpisodicScoringPolicy`) ranks candidate episodes using POV visibility, causal provenance pinning, entity/location cue overlap, semantic cosine similarity, and salience-modulated recency decay. `retrieve_context()` bounds candidate loading in SQL via `CausalStore.get_lineage_episodes(..., limit=...)` (combining POV filtering, pinned provenance, `participants`/`witnesses`/`location_id` cue matching, and `episodes_fts` FTS5 summary matching) and checks provenance visibility via `get_known_episode_ids()`. Older low-salience episodes can be compacted manually via `consolidate_episodes()` or automatically via `consolidate_episodes_auto()` / `consolidate_chapter()` using pluggable `EpisodeConsolidator` implementations (`LLMEpisodeConsolidator`, `CallableEpisodeConsolidator`, `HeuristicEpisodeConsolidator`) with provenance, deceived-POV, participant-cap, chronological-anchoring, and child-branch safety guards (`consolidated_from`).

---

## 2. Storage Architecture: SQLite Schema

Rather than spreading state across disconnected services, `canonarc` maintains its temporal graph and episodic multiverse in a single `.sqlite` database:

```sql
-- 1. Copy-on-Write Timeline Tree
CREATE TABLE IF NOT EXISTS timelines (
    timeline_id TEXT PRIMARY KEY,
    parent_id   TEXT REFERENCES timelines(timeline_id),
    fork_step   INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

-- 2. Unified Append-Only Materialized Revisions (Typed Multigraph)
CREATE TABLE IF NOT EXISTS graph_revisions (
    revision_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    timeline_id   TEXT NOT NULL REFERENCES timelines(timeline_id),
    step          INTEGER NOT NULL,
    record_type   TEXT NOT NULL, -- 'entity' | 'edge' | 'alias'
    subject_key   TEXT NOT NULL, -- 'char_arthur', 'char_arthur:stance->char_merlin', 'the old wizard'
    relation_type TEXT,          -- predicate for 'edge' ('stance', 'kinship', 'debt'); NULL for entity/alias
    source_id     TEXT,          -- entity_id for entity/alias; source_id for edge
    target_id     TEXT,          -- target_id for edge
    is_tombstone  INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL  -- Full materialized JSON
);

CREATE INDEX IF NOT EXISTS idx_revisions_lookup
    ON graph_revisions(timeline_id, record_type, subject_key, step DESC, revision_id DESC);

CREATE INDEX IF NOT EXISTS idx_revisions_edge_type
    ON graph_revisions(timeline_id, record_type, relation_type, source_id, target_id);

-- 3. Tier 5 Voice Exemplars
CREATE TABLE IF NOT EXISTS voice_exemplars (
    exemplar_id           TEXT NOT NULL,
    timeline_id           TEXT NOT NULL REFERENCES timelines(timeline_id),
    step                  INTEGER NOT NULL,
    speaker_id            TEXT NOT NULL,
    addressee_id          TEXT,
    text                  TEXT NOT NULL,
    source_text           TEXT NOT NULL DEFAULT '',
    metadata              TEXT NOT NULL DEFAULT '{}',
    quality_score         REAL NOT NULL,
    distinctiveness_score REAL NOT NULL,
    embedding             BLOB NOT NULL,
    PRIMARY KEY (timeline_id, exemplar_id)
);

CREATE INDEX IF NOT EXISTS idx_voice_lookup
    ON voice_exemplars(timeline_id, speaker_id, step DESC);

-- 4. Tier 6 Episodic Event Ledger
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT NOT NULL,
    timeline_id  TEXT NOT NULL REFERENCES timelines(timeline_id),
    step         INTEGER NOT NULL,
    predicate    TEXT NOT NULL DEFAULT 'event',
    summary      TEXT NOT NULL,
    location_id  TEXT,
    salience     REAL NOT NULL DEFAULT 5.0,
    participants TEXT NOT NULL DEFAULT '{}', -- JSON map: entity_id -> role
    witnesses    TEXT NOT NULL DEFAULT '[]', -- JSON list of entity_ids
    metadata     TEXT NOT NULL DEFAULT '{}',
    embedding    BLOB NOT NULL DEFAULT X'',
    PRIMARY KEY (timeline_id, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_episodes_lookup
    ON episodes(timeline_id, step DESC);

-- 5. Tier 6 Epistemic Awareness Join Table
CREATE TABLE IF NOT EXISTS episode_epistemics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timeline_id      TEXT NOT NULL REFERENCES timelines(timeline_id),
    step             INTEGER NOT NULL,
    episode_id       TEXT NOT NULL,
    entity_id        TEXT NOT NULL,
    epistemic        TEXT NOT NULL DEFAULT 'direct', -- 'direct' | 'told' | 'deceived' | 'forgotten' | 'suppressed'
    believed_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_episode_epistemics_lookup
    ON episode_epistemics(timeline_id, episode_id, step DESC);

-- 6. FTS5 Summary Index & Sync Triggers for Bounded Candidate Pre-Filtering
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    timeline_id UNINDEXED,
    episode_id UNINDEXED,
    summary
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    DELETE FROM episodes_fts WHERE timeline_id = NEW.timeline_id AND episode_id = NEW.episode_id;
    INSERT INTO episodes_fts(timeline_id, episode_id, summary)
    VALUES (NEW.timeline_id, NEW.episode_id, NEW.summary);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    DELETE FROM episodes_fts WHERE timeline_id = OLD.timeline_id AND episode_id = OLD.episode_id;
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF summary ON episodes BEGIN
    DELETE FROM episodes_fts WHERE timeline_id = OLD.timeline_id AND episode_id = OLD.episode_id;
    INSERT INTO episodes_fts(timeline_id, episode_id, summary)
    VALUES (NEW.timeline_id, NEW.episode_id, NEW.summary);
END;

-- 7. Store Metadata (Embedder Provenance & Dimension Tracking)
CREATE TABLE IF NOT EXISTS store_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

---

## 3. The Recursive CTE Lineage Engine with Windowed Trajectory

Point-in-time state resolution is executed in a single recursive CTE that simultaneously resolves active records (`rn = 1`) and extracts historical trajectory:

```sql
WITH RECURSIVE lineage(timeline_id, parent_id, max_step, depth) AS (
    SELECT timeline_id, parent_id, :cutoff_step, 0
    FROM timelines
    WHERE timeline_id = :timeline_id
    UNION ALL
    SELECT
        t.timeline_id,
        t.parent_id,
        CASE
            WHEN l.max_step IS NULL THEN t_child.fork_step
            WHEN t_child.fork_step IS NULL THEN l.max_step
            ELSE MIN(l.max_step, t_child.fork_step)
        END,
        l.depth + 1
    FROM lineage l
    JOIN timelines t_child ON t_child.timeline_id = l.timeline_id
    JOIN timelines t       ON t.timeline_id = l.parent_id
    WHERE l.parent_id IS NOT NULL
),
step_deduped AS (
    SELECT
        r.revision_id,
        r.timeline_id,
        r.step,
        r.record_type,
        r.subject_key,
        r.relation_type,
        r.source_id,
        r.target_id,
        r.is_tombstone,
        r.payload,
        l.depth,
        ROW_NUMBER() OVER (
            PARTITION BY r.record_type, r.subject_key, r.step
            ORDER BY l.depth ASC, r.revision_id DESC
        ) AS step_rn
    FROM graph_revisions r
    JOIN lineage l
      ON r.timeline_id = l.timeline_id
     AND (l.max_step IS NULL OR r.step <= l.max_step)
),
ranked_revisions AS (
    SELECT
        revision_id,
        timeline_id,
        step AS valid_from_step,
        record_type,
        subject_key,
        relation_type,
        source_id,
        target_id,
        is_tombstone,
        payload,
        ROW_NUMBER() OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY record_type, subject_key
        ) AS revision_count,
        LEAD(payload) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_payload,
        LEAD(step) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_step,
        LEAD(is_tombstone) OVER (
            PARTITION BY record_type, subject_key
            ORDER BY depth ASC, step DESC, revision_id DESC
        ) AS prior_is_tombstone
    FROM step_deduped
    WHERE step_rn = 1
)
SELECT
    record_type,
    subject_key,
    relation_type,
    source_id,
    target_id,
    valid_from_step,
    payload,
    revision_count,
    prior_payload,
    prior_step,
    prior_is_tombstone
FROM ranked_revisions
WHERE rn = 1 AND is_tombstone = 0;
```

### Why this architecture is fast and robust:
1. `step_deduped` (`step_rn = 1`): If a child branch overrides a record at the exact same `step` as its parent, the parent row at that `step` is deduplicated before windowing so `prior_payload` and `revision_count` remain uncorrupted.
2. `depth ASC`: Revisions made on the child branch (`depth = 0`) strictly supersede parent revisions.
3. `r.step <= l.max_step`: Parent revisions committed *after* `fork_step` are completely invisible to the child branch.
4. `rn = 1`: Picks the latest revision for each key in sub-millisecond execution while `prior_payload` preserves the preceding state without requiring a second database query.
5. **Episodic Isolation**: Events are not forced into key-value entity states, eliminating prompt bloat and schema decay.

---

## 4. Modular Subsystem Map (`src/canonarc/`)

| Module | Responsibility |
| :--- | :--- |
| `models.py` | Pydantic v2 schemas for Tiers 1–6, `EpistemicMode`, `StepDelta`, `StepContext`, and `ResolvedMention` |
| `store/` | `MemoryStore` protocol & SQLite `CausalStore` (`schema`, `graph`, `episodes`, `timelines`, `checkpoints`, `migrations`) |
| `embedders.py` | `Embedder` protocol, `OfflineNgramEmbedder` (CJK + signed hashing), `CallableEmbedder`, `OpenAICompatibleEmbedder`, `OllamaEmbedder` |
| `voice/` | `IngestionGate`, `VoiceBank`, and `VoiceQualityScorer` (`gate`, `bank`, `scoring`) |
| `episodic.py` | `EpisodeRanker`, `DefaultEpisodeRanker`, `EpisodicScoringPolicy`, and `EpisodeConsolidator` implementations |
| `resolver.py` | `MentionResolver` & `DefaultLexicalResolver` (word-boundary, fuzzy inflection, alias span matching) |
| `extractors/` | `DeltaExtractor`, `JsonCompletionClient`, `CallableExtractor`, `OpenAICompatibleExtractor`, `CachedDeltaExtractor`, and usage tracking |
| `hooks.py` | `EngineHooks` & `BaseEngineHooks` lifecycle callbacks (commit, rollback, branch, gate rejection) |
| `presets/` | `TierProfile`, `SchemaPolicy`, `InteractiveFictionPreset`, `TranslationPreset`, and `SimulationPreset` |
| `bible/` | Sequential LLM Series Bible bootstrapping (`builder`), lossless CSV/JSON/Markdown export, and idempotent diff/upsert sync (`export`, `parsers`, `sync`, `models`) |
| `audit/` | Pluggable deterministic QA & continuity audit engine (`models`, `runner`, `rules_continuity`, `rules_translation`, `text_utils`) |
| `engine/` | `CanonArcEngine`, `StepTransaction`, `TimelineSession`, retrieval, epistemics, seeding, mutations, embeddings, harvesting, normalization, consolidation, merging, resume, traces |
| `cli.py` | `canonarc` CLI inspector (`tree`, `inspect`, `diff`, `history`, `export`, `import`, `audit`, `duplicates`, `merge`) |
