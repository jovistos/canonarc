# API Reference

## `CanonArcEngine`

The central facade for state tracking, episodic memory, voice retrieval, mutation, and causal time travel.

```python
class CanonArcEngine:
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        store: MemoryStore | None = None,
        timeline_id: str = "main",
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        preset: DomainPreset | None = None,
        mention_resolver: MentionResolver | None = None,
        extractor: DeltaExtractor | None = None,
        episode_ranker: EpisodeRanker | None = None,
        episode_consolidator: EpisodeConsolidator | None = None,
        inflection_generator: InflectionGenerator | None = None,
        voice_scorer: VoiceQualityScorer | None = None,
        hooks: Sequence[EngineHooks] = (),
        audit_rules: Sequence[AuditRule] | None = None,
        on_embedder_mismatch: Literal["raise", "warn", "reindex", "ignore"] = "warn",
        ignore_patterns: Sequence[str | re.Pattern[str]] = (),
        text_normalizer: TextNormalizer | None = None,
    ) -> None: ...
```

Supports Python context management (`with CanonArcEngine("save.sqlite") as engine:`) as well as concurrency-safe timeline isolation via `engine.session(timeline_id)` (`TimelineSession`) and `with engine.use_timeline(timeline_id):`.

### Seeding Methods
- `register_entity(entity: EntityRecord, aliases: Sequence[str | AliasRecord] = (), step: int = 0, timeline_id: str | None = None, target_lang: str | None = None, auto_inflect: bool = True) -> None`: Registers an entity and its aliases at `step`. Auto-registers `base_attributes["target_name"]` as `kind="permanent"` and any generated inflected/possessive forms as `kind="inflection"`.
- `seed_entities(entities: Sequence[EntityRecord], step: int = 0, timeline_id: str | None = None, target_lang: str | None = None, auto_inflect: bool = True, batch_size: int = 25) -> None`: Batch-registers entities in a single store transaction and generates inflections via `generate_inflections_batch(..., batch_size=batch_size)`.

### Series Bible Import/Export & Sync (`canonarc.bible`)
- `export_bible(path: str | Path, *, format: Literal["auto", "csv", "json", "md"] = "auto", at_step: int | None = None, timeline_id: str | None = None, kinds: Sequence[str] | None = None, include_state: bool = False, include_edges: bool | None = None, include_aliases: bool = True, include_inflections: bool | None = None, lang: str | None = None, excel_bom: bool = False) -> BibleExportResult`: Exports the active Series Bible at `(timeline_id, at_step)` (defaults to timeline head when `at_step=None`) to JSON (lossless entities + aliases + edges), CSV (spreadsheet-friendly entities + `;`-separated aliases + flattened `base_attributes`), or Markdown (export-only human reference doc). When `include_edges=None`, defaults to `policy.has_tier(2)` (`False` in `LEXICON_ONLY`, `True` in `STATE_ONLY`/`FULL`). When `include_inflections=None`, defaults to `True` for `json` and `False` for `csv`/`md` (hiding `kind="inflection"` grammar forms from human spreadsheets while keeping them in JSON backups and lexical resolution).
- `render_bible(*, format: Literal["csv", "json", "md"] = "md", at_step: int | None = None, timeline_id: str | None = None, kinds: Sequence[str] | None = None, include_state: bool = False, include_edges: bool | None = None, include_aliases: bool = True, include_inflections: bool | None = None, lang: str | None = None) -> tuple[str, BibleExportResult]`: Renders the Series Bible to a string without writing to disk.
- `sync_bible(path: str | Path, *, format: Literal["auto", "csv", "json"] = "auto", step: int = 0, timeline_id: str | None = None, mode: Literal["upsert", "merge", "replace"] = "upsert", target_lang: str | None = None, auto_inflect: bool = True, batch_size: int = 25, dry_run: bool = False, column_map: dict[str, str] | None = None) -> BibleSyncReport`: Diffs a CSV or JSON Series Bible against active state at `(timeline_id, step)` and atomically applies only changed revisions (idempotent re-syncs emit zero revisions). Supports `merge` (non-destructive), `upsert` (default, row-authoritative without tombstoning absent entities), and `replace` (strict mirror with tombstones). Emits warnings for alias collisions, auto-generated IDs, and shadowed revisions when `step < head_step`. Only invokes `inflection_generator` when `target_name`/`target_lang`/`gender`/`declension`/`kind` change, batches needed entities via `generate_inflections_batch(..., batch_size=batch_size)`, and preserves existing `kind="inflection"` aliases on unchanged CSV rows.

*(All three methods are also available on `TimelineSession`.)*

### Pluggable QA & Continuity Audit (`canonarc.audit`)
- `audit_step(step: int, source_text: str = "", target_text: str = "", *, timeline_id: str | None = None, rules: Sequence[AuditRule] | None = None, include_rules: Sequence[str] | None = None, exclude_rules: Sequence[str] | None = None, min_severity: AuditSeverity = "info", lang: str | None = None) -> AuditReport`: Runs deterministic, zero-LLM continuity and localization QA rules against a single committed step (and optional source/target text pair). Automatically skips rules whose `required_tiers` are disabled by the active `SchemaPolicy`.
- `audit_timeline(*, timeline_id: str | None = None, from_step: int = 0, to_step: int | None = None, texts: Mapping[int, str | tuple[str, str]] | None = None, rules: Sequence[AuditRule] | None = None, include_rules: Sequence[str] | None = None, exclude_rules: Sequence[str] | None = None, min_severity: AuditSeverity = "info", lang: str | None = None) -> AuditReport`: Audits every step in `[from_step..to_step]` on a timeline, optionally passing per-step `target_text` or `(source_text, target_text)` pairs for text-level rules.
- `find_duplicate_candidates(*, at_step: int | None = None, timeline_id: str | None = None, fuzzy_threshold: float = 88.0, same_kind_only: bool = True) -> list[DuplicateCandidate]`: Read-only discovery helper that scans the entire active entity roster at `(timeline_id, at_step)` (defaulting to the timeline's `head_step`; pass `at_step=0` for post-glossary checks) using `NearDuplicateEntityRule(check_all_entities=True)` and returns typed `DuplicateCandidate` records sorted by descending score, ready for review or explicit `merge_entities(cand.duplicate_id, cand.canonical_id)` calls. Also available on `TimelineSession`.

*(Both methods are also available on `TimelineSession`.)*

### Retrieval & Query Methods
- `retrieve_context(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int | None = None, top_k_episodes: int | None = None, pov_entity_ids: Sequence[str] | None = None, pov_mode: Literal["any", "all"] = "any", edge_types: Sequence[str] | None = None, location_id: str | None = None, inclusive: bool = False, timeline_id: str | None = None, fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only", voice_constraints: dict[str, str] | None = None) -> StepContext`: Executes unified retrieval for all active tiers in `SchemaPolicy` at `step` (`top_k_voice` defaults to `3` when Tier 5 is enabled or `0` when disabled; `top_k_episodes` defaults to `5` when Tier 6 is enabled or `0` when disabled). Co-present scene entities (matched in text/mentions, sharing the active `location_id`, or POV fallback) and edges between them populate `entities` and `active_edges`, while 1-hop off-screen social graph neighbors populate `related_entities` and `background_edges`. Optional `voice_constraints` merge on top of edge-derived attributes for Tier 5 hard-filter/soft-boost ranking (e.g., `{"channel": "radio", "urgency": "high"}`).
- `query_episodes(cutoff_step: int = 999999999, participants: Sequence[str] = (), match_all_participants: bool = False, predicate: str | None = None, location_id: str | None = None, known_to: Sequence[str] = (), min_salience: float = 0.0, step_range: tuple[int, int] | None = None, timeline_id: str | None = None) -> list[EpisodeRecord]`: Queries episodic ledger by participants, predicate, location, and epistemic visibility.
- `format_context(ctx: StepContext) -> str`: Formats `StepContext` using the configured `DomainPreset`.

### Extraction & Normalization Methods
- `build_extraction_messages(ctx: StepContext, scene_text: str, *, prev_text: str = "", next_text: str = "", compact_schema: bool = False, known_entity_scope: KnownEntityScope = "scene", max_roster_entities: int | None = None) -> tuple[str, str]`: Builds a prefix-cache-friendly `(system_prompt, user_prompt)` pair with static extraction rules and `StepDelta` JSON schema in `system_prompt` (automatically pruning disabled Tier 2/6 fields when using e.g. `TierProfile.STATE_ONLY`) and dynamic per-step context in `user_prompt`. `known_entity_scope` controls which known IDs are shown: `"scene"` (default: `ctx.entities` + `ctx.related_entities`), `"characters_compact"` (appends a terse 1-line `id=Name/TargetName` cast roster of all other active characters on `ctx.timeline_id` at `max(0, ctx.step - 1)`), or `"all_compact"` (all other active entities grouped by `kind`). Use `max_roster_entities` to cap the compact roster size on very large corpora.
- `build_extraction_prompt(ctx: StepContext, scene_text: str, *, prev_text: str = "", next_text: str = "", compact_schema: bool = False, known_entity_scope: KnownEntityScope = "scene", max_roster_entities: int | None = None) -> str`: Single-string wrapper around `build_extraction_messages()` preserving static-prefix-first ordering.
- `normalize_delta(delta: StepDelta, step: int | None = None, timeline_id: str | None = None) -> StepDelta`: Normalizes canonical names, aliases, and retired IDs recorded in `base_attributes["merged_from"]` into canonical `entity_id`s, and resolves intra-step symbolic `episode_id` references.

*(Extraction and normalization methods are also available on `TimelineSession`.)*

### Transaction & Mutation Methods
- `step_transaction(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int | None = None, top_k_episodes: int | None = None, pov_entity_ids: Sequence[str] | None = None, pov_mode: Literal["any", "all"] = "any", edge_types: Sequence[str] | None = None, location_id: str | None = None, timeline_id: str | None = None, rerun_clean: bool = True) -> StepTransaction`: Context manager for clean read-and-commit workflows. Supports:
  - `tx.audit_draft(target_text, *, source_text=None, rules=None, include_rules=None, exclude_rules=None, min_severity="info", lang=None, include_all_timeline_entities=False, label=None) -> AuditReport`: Audits a pre-commit draft translation against the pre-commit state (`tx.context` / `step - 1`) before `tx.commit()` or `tx.extract_and_commit()`, enabling deterministic retry loops via `report.format_repair_instructions()`. Every call is appended to the transaction's audit trace in attempt order (tagged with optional `label`, e.g. `"initial"`, `"repair_1"`).
  - `tx.extract_and_commit(scene_text, *, compact_schema=False, known_entity_scope="scene", max_roster_entities=None, ...)`: Runs the configured `DeltaExtractor`, normalizes IDs, and stages the `StepDelta` (zero-LLM-call no-op when Tier 3 is disabled, e.g., `TierProfile.LEXICON_ONLY`).
  - `tx.harvest_voice(text, *, source_text="", lang=None, prev_text=None, next_text=None, scorer=None, min_quality=None, require_speaker=True, infer_addressee=True, extra_metadata=None, stage=True) -> list[VoiceExemplar]`: Runs Gate-1 extraction, attaches active dyadic edge/persona metadata (overlaying any staged `StepDelta` in the same transaction), scores candidates with Gate-2 `VoiceQualityScorer`, emits `on_gate_rejected("gate2", ...)` for rejections, and stages passing `VoiceExemplar`s for atomic commit (zero-work no-op when Tier 5 is disabled).
  - `tx.trace(*, include_raw_delta=False) -> StepTransactionTrace` & `tx.to_trace_dict(*, include_raw_delta=False, exclude_none=True) -> dict[str, Any]`: Builds a standardized, JSON-ready provenance snapshot capturing `StepContextSummary` (resolved mentions, injected entity/edge/episode/voice IDs & counts), all `audit_draft()` reports, `staged_delta` & `delta_counts` (plus opt-in pre-normalization `raw_delta`), `VoiceHarvestTrace` counts, post-commit status/IDs (`committed_episode_ids`, `indexed_exemplar_ids`), and per-stage model-specific `TransactionUsageSummary`. Safe to call inside the `with` block (`status="open"`) or after exit (`status="committed" | "skipped" | "rolled_back"`; call after exit for final committed IDs and `duration_ms`).
- `apply_delta(step: int, delta: StepDelta, timeline_id: str | None = None) -> list[EpisodeRecord]`: Directly applies state deltas, episodes, and inform actions. When `delta.is_flashback=True`, skips overwriting existing entities/edges while still indexing episodes and new entities.
- `merge_entities(duplicate_id: str, canonical_id: str, *, step: int | None = None, timeline_id: str | None = None, merge_attributes: bool = True, migrate_edges: bool = True, migrate_voice: bool = True, migrate_episodes: bool = True, conflict_strategy: Literal["canonical_wins", "duplicate_wins"] = "canonical_wins", allow_kind_mismatch: bool = False, reason: str | None = None, dry_run: bool = False) -> EntityMergeReport`: Atomically merges `duplicate_id` into `canonical_id` at `(timeline_id, step)` (defaults to timeline head). Promotes `duplicate_id`'s `canonical_name`, `base_attributes["target_name"]`, and active `AliasRecord`s onto `canonical_id`, tombstones `duplicate_id` and its old aliases, backfills `base_attributes`/`state`/`private_state` (recording `base_attributes["merged_from"]` for `normalize_delta()` redirects), and optionally migrates active edges (dropping self-loops and warning on `hard_filter_keys` collisions), location references, Tier 5 voice exemplars, and Tier 6 episodes/epistemics lineage-safely. Also available on `TimelineSession`.
- `index_episodes(step: int, episodes: Sequence[EpisodeRecord], timeline_id: str | None = None) -> list[EpisodeRecord]`: Embeds (using `embed_batch()` when available) and indexes episodes into Tier 6.
- `count_consolidatable_episodes(step: int, *, before_step: int, max_salience: float = 4.0, predicate: str = "consolidated_summary", chapter_id: str | None = None, preserve_provenance: bool = True, preserve_deceived: bool = True, timeline_id: str | None = None) -> int`: Zero-LLM readiness check returning how many episodes match consolidation criteria (excluding episodes pinned by active `caused_by_episode_id` or `"deceived"` epistemics when enabled). Returns `0` when Tier 6 is disabled. Also available on `TimelineSession`.
- `consolidate_episodes(step: int, *, before_step: int, summary: str, max_salience: float = 4.0, predicate: str = "consolidated_summary", episode_id: str | None = None, chapter_id: str | None = None, salience: float = 7.5, location_id: str | None = None, key_participant_ids: Sequence[str] | None = None, max_participants: int | None = None, preserve_provenance: bool = True, preserve_deceived: bool = True, epistemic_merge: Literal["union", "intersection"] = "union", anchor_step: Literal["current", "last_target"] = "current", force: bool = False, timeline_id: str | None = None) -> EpisodeRecord | None`: Low-level primitive that compacts older low-salience episodes (`< before_step`) into a single summary episode. Protects active causal provenance (`caused_by_episode_id`), `"deceived"` POV records, and child branches (raising `TimelineBranchConflictError` unless `force=True`), and returns `None` when Tier 6 is disabled. Also available on `TimelineSession`.
- `consolidate_episodes_auto(step: int, *, before_step: int, consolidator: EpisodeConsolidator | None = None, client: JsonCompletionClient | None = None, max_salience: float = 4.0, min_episodes: int = 3, max_episodes_per_batch: int = 30, predicate: str = "consolidated_summary", chapter_id: str | None = None, salience: float | None = None, default_salience: float = 6.0, max_participants: int | None = 5, preserve_provenance: bool = True, preserve_deceived: bool = True, epistemic_merge: Literal["union", "intersection"] = "union", anchor_step: Literal["last_target", "current"] = "last_target", force: bool = False, timeline_id: str | None = None) -> list[EpisodeRecord]`: High-level one-call consolidation helper that queries eligible episodes, resolves canonical entity names, summarizes them in bounded batches via `consolidator` / `client` / `engine.episode_consolidator` / `engine.extractor`, and commits compacted episodes atomically. Skips LLM calls when fewer than `min_episodes` match or Tier 6 is disabled, and verifies child-branch safety before calling the LLM. Also available on `TimelineSession`.
- `consolidate_chapter(step: int, chapter_id: str, *, before_step: int | None = None, ...) -> list[EpisodeRecord]`: Convenience wrapper around `consolidate_episodes_auto()` scoped to a single `chapter_id` (`before_step` defaults to `step + 1`). Also available on `TimelineSession`.
- `reindex_embeddings(batch_size: int = 128) -> dict[str, int]`: Re-embeds all stored episodes and voice exemplars in batches using the active embedder and updates `store_metadata` (`embedder_id`, `embedding_dim`). Returns `{"episodes": n_ep, "voice_exemplars": n_vo}`.
- `inform_entity(step: int, episode_id: str, entity_id: str, epistemic: EpistemicMode = "told", believed_summary: str | None = None, timeline_id: str | None = None) -> None`: Records secondhand, delayed, deceptive (`"deceived"`), or erased (`"forgotten"` / `"suppressed"`) epistemic awareness of an episode.
- `extract_dialogue_candidates(text: str, source_text: str = "") -> list[DialogueCandidate]`: Extracts candidate dialogue lines passing Gate 1.
- `index_voice_exemplars(step: int, exemplars: Sequence[VoiceExemplar], bypass_gate: bool = False, timeline_id: str | None = None) -> list[VoiceExemplar]`: Indexes voice exemplars into Tier 5 Vector DB.
- `commit_step(step: int, delta: StepDelta | None = None, exemplars: Sequence[VoiceExemplar] = (), episodes: Sequence[EpisodeRecord] = (), bypass_gate: bool = False, timeline_id: str | None = None, content_hash: str | None = None, checkpoint_metadata: dict[str, Any] | None = None) -> list[VoiceExemplar]`: Atomically commits deltas, episodes, voice exemplars, and a `step_checkpoints` row for the step (even when all mutations are empty).
- `is_step_committed(step: int, content_hash: str | None = None, metadata_match: dict[str, Any] | None = None, timeline_id: str | None = None) -> bool`: Returns `True` if `step` is checkpointed along the timeline lineage and matches `content_hash` / `metadata_match` when provided.
- `last_committed_step(timeline_id: str | None = None) -> int | None`: Returns the highest checkpointed step along the timeline lineage, or `None` if no step has been committed yet.
- `get_step_checkpoint(step: int, timeline_id: str | None = None) -> StepCheckpoint | None`: Returns the lineage-resolved `StepCheckpoint` (`timeline_id`, `step`, `content_hash`, `metadata`) for `step`.
- `list_step_checkpoints(timeline_id: str | None = None, cutoff_step: int | None = None) -> list[StepCheckpoint]`: Returns lineage-resolved checkpoints in ascending step order.
- `plan_resume(step_hashes: Sequence[tuple[int, str]], *, causal: bool | None = None, metadata_match: Mapping[str, Any] | None = None, timeline_id: str | None = None, apply_rollback: bool = False, force: bool = False, prune_tail: bool = True) -> ResumePlan`: Evaluates an ordered sequence of `(step, content_hash)` pairs against lineage-resolved checkpoints in a single query and returns a `ResumePlan`. Defaults `causal` to `True` when any stateful tier (`2`, `3`, `5`, or `6`) is active (`STATE_ONLY` / `FULL`) and `False` for `LEXICON_ONLY`. In causal mode, the first missing or mismatched step invalidates the suffix and computes `rollback_to_step` (also detecting uncheckpointed writes via `head_step` and orphaned tail steps when `prune_tail=True`). In non-causal mode, steps are evaluated independently and `apply_rollback=True` purges only stale/orphaned steps via `store.delete_steps()`. Also available on `TimelineSession`.
- `compute_step_hash(text: str, *, model: str | None = None, prompt_version: str | None = None, extra: dict[str, Any] | None = None) -> str`: Top-level helper producing a deterministic SHA-256 hex digest for step resumability checks.

### Time-Travel Methods
- `rollback_to(step: int, timeline_id: str | None = None, force: bool = False) -> int`: Rewinds the timeline to `step`, deleting all future graph revisions, voice exemplars, episodes, and step checkpoints. Raises `TimelineBranchConflictError` if a child branch forked at `fork_step > step` unless `force=True`.
- `branch(at_step: int, name: str, from_timeline: str | None = None, switch: bool = False, metadata: dict | None = None) -> TimelineInfo`: Forks an alternate timeline at `at_step`.
- `delete_timeline(timeline_id: str, cascade: bool = False) -> int`: Deletes a timeline branch (and optionally its descendants when `cascade=True`).
- `checkout(timeline_id: str) -> None`: Switches the active timeline.
- `use_timeline(timeline_id: str) -> Iterator[TimelineSession]`: Context manager scoping `active_timeline` via `contextvars` for concurrent requests.
- `session(timeline_id: str | None = None) -> TimelineSession`: Returns an immutable-timeline session view bound to `timeline_id`.
- `list_timelines() -> list[TimelineInfo]`: Lists all existing timelines.
- `diff_timelines(timeline_a: str, timeline_b: str, at_step: int) -> dict[str, Any]`: Compares active states between two branches.
- `export_snapshot(target_path: str | Path, at_step: int, timeline_id: str | None = None) -> Path`: Exports a standalone, flattened `.sqlite` save file preserving active state and lineage episodes.

---

## Core Models

### `ResumePlan`
```python
class ResumePlan(BaseModel):
    timeline_id: str = "main"
    causal: bool = True
    reusable_steps: list[int] = []
    pending_steps: list[int] = []
    missing_steps: list[int] = []
    stale_steps: list[int] = []
    invalidated_steps: list[int] = []
    orphaned_steps: list[int] = []
    first_stale_step: int | None = None
    rollback_to_step: int | None = None
    fork_conflict_step: int | None = None
    rollback_applied: bool = False
    deleted_rows: int = 0
    # Convenience properties:
    # .steps_to_purge -> sorted(stale | invalidated | orphaned)
    # .is_up_to_date  -> not pending_steps and rollback_to_step is None and not orphaned_steps
    # .needs_rollback -> True when a rollback or targeted purge is pending
```

### `EntityRecord`
```python
class EntityRecord(BaseModel):
    entity_id: str
    kind: str = "character"              # Standard kinds: 'character', 'location', 'faction', 'organization', 'artifact', 'item', 'term'
    canonical_name: str
    base_attributes: dict[str, Any] = {}
    state: dict[str, Any] = {}
    private_state: dict[str, Any] = {}
    disguised_as: str | None = None
    valid_from_step: int = 0
    valid_to_step: int | None = None
    caused_by_episode_id: str | None = None
    reason: str | None = None
    prior_state: dict[str, Any] | None = None
    prior_step: int | None = None
    revision_count: int = 1
```
Standard auto-generated ID prefixes (shared across `canonarc.bible` and `canonarc.audit` via `canonarc.models.ENTITY_KIND_PREFIXES`): `char_`, `loc_`, `fac_`, `org_`, `item_` (`artifact`/`item`), and `term_`.

### `AliasRecord`
```python
class AliasRecord(BaseModel):
    alias: str
    entity_id: str
    kind: Literal["permanent", "epithet", "disguise", "title", "inflection"] = "epithet"
    lang: str | None = None              # Optional ISO/BCP-47 tag (e.g. 'en', 'sr'); None = universal
    known_to: list[str] | None = None    # None = universal; list[str] = scoped to observers
    persona_name: str | None = None      # Surface identity shown to fooled observers
    valid_from_step: int = 0
    valid_to_step: int | None = None
```

### `DirectedEdgeRecord`
```python
class DirectedEdgeRecord(BaseModel):
    source_id: str
    target_id: str
    relation_type: str = "stance"
    state: dict[str, Any] = {}
    valid_from_step: int = 0
    valid_to_step: int | None = None
    caused_by_episode_id: str | None = None
    reason: str | None = None
    prior_state: dict[str, Any] | None = None
    prior_step: int | None = None
    revision_count: int = 1
```

### `EpisodeRecord` (Tier 6)
```python
EpistemicMode = Literal["direct", "told", "deceived", "forgotten", "suppressed"]

class EpisodeRecord(BaseModel):
    episode_id: str = ""
    timeline_id: str = "main"
    step: int = 0
    predicate: str = "event"
    summary: str
    participants: dict[str, str] = {}    # entity_id -> semantic role
    witnesses: list[str] = []             # entity_ids with direct observation
    known_by: dict[str, Literal["direct", "told", "deceived"]] = {}
    believed_summaries: dict[str, str] = {}
    location_id: str | None = None
    story_time: int | float | str | None = None
    chapter_id: str | None = None
    scene_id: str | None = None
    consolidated_from: list[str] = []
    salience: float = 5.0                # 0.0 to 10.0
    metadata: dict[str, Any] = {}
    match_score: float = 0.0
```

### `InformDelta`
```python
class InformDelta(BaseModel):
    episode_id: str
    entity_id: str
    epistemic: EpistemicMode = "told"    # "direct" | "told" | "deceived" | "forgotten" | "suppressed"
    believed_summary: str | None = None
```

### `VoiceExemplar`
```python
class VoiceExemplar(BaseModel):
    exemplar_id: str = ""
    timeline_id: str = "main"
    step: int
    speaker_id: str
    addressee_id: str | None = None
    text: str
    source_text: str = ""
    metadata: dict[str, str] = {}
    quality_score: float = 10.0
    distinctiveness_score: float = 0.0
    match_score: float = 0.0
```

### `ResolvedMention`
```python
class ResolvedMention(BaseModel):
    surface_text: str
    entity_id: str
    matched_via: Literal["canonical", "alias", "fuzzy"]
    confidence: float = 1.0
    matched_source: Literal["canonical", "alias"] | None = None
    matched_phrase: str | None = None
    alias_kind: Literal["permanent", "epithet", "disguise", "title", "inflection"] | None = None
    lang: str | None = None
```

### `StepDelta`
```python
class StepDelta(BaseModel):
    entities: list[EntityDelta] = []
    edges: list[EdgeDelta] = []
    episodes: list[EpisodeRecord] = []
    inform: list[InformDelta] = []
    is_flashback: bool = False
    story_time: int | float | str | None = None
    chapter_id: str | None = None
    scene_id: str | None = None
```

### `StepContext` & `StepContextSummary`
```python
class StepContext(BaseModel):
    timeline_id: str
    step: int
    story_time: int | float | str | None = None
    chapter_id: str | None = None
    scene_id: str | None = None
    resolved_mentions: list[ResolvedMention] = []
    entities: list[EntityRecord] = []               # Co-present scene entities
    active_edges: list[DirectedEdgeRecord] = []      # Edges strictly between co-present scene entities
    related_entities: list[EntityRecord] = []       # 1-hop off-screen social graph neighbors
    background_edges: list[DirectedEdgeRecord] = []  # Edges from scene entities to off-screen neighbors
    voice_exemplars: list[VoiceExemplar] = []
    episodes: list[EpisodeRecord] = []
    entity_names: dict[str, str] = {}                # entity_id -> canonical_name lookup (includes off-screen episode/edge refs)
    # Helper: ctx.to_summary() -> StepContextSummary (compact IDs, edge keys, counts, and resolved_mentions)
```

### `StepTransactionTrace`
```python
class StepTransactionTrace(BaseModel):
    schema_version: str = "1.0"
    timeline_id: str
    step: int
    lang: str | None = None
    rerun_clean: bool = True
    status: Literal["open", "committed", "skipped", "rolled_back"] = "open"
    content_hash: str | None = None
    checkpoint_metadata: dict[str, Any] = {}
    duration_ms: float | None = None
    context: StepContextSummary                      # resolved_mentions, injected IDs, edge keys, counts
    audit_reports: list[DraftAuditTrace] = []        # ordered pre-commit audit_draft() attempts
    extraction_skipped_by_policy: bool = False
    raw_delta: StepDelta | None = None               # populated when tx.trace(include_raw_delta=True)
    staged_delta: StepDelta | None = None
    delta_counts: dict[str, int] = {}                # entities, edges, episodes, inform
    staged_extra_episodes: int = 0
    staged_extra_exemplars: int = 0
    voice_harvest: VoiceHarvestTrace                 # candidates_extracted, gate1/gate2 rejections, staged_count, IDs
    harvested_voice_count: int = 0
    committed_episode_ids: list[str] = []
    indexed_exemplar_ids: list[str] = []
    error: str | None = None
    usage: TransactionUsageSummary                   # per-stage StageUsageSummary + summed model-specific cost
```

### `BibleExportResult` & `BibleSyncReport`
```python
class BibleExportResult(BaseModel):
    path: str | None = None
    format: Literal["csv", "json", "md"]
    timeline_id: str
    at_step: int
    entity_count: int = 0
    alias_count: int = 0
    edge_count: int = 0

class BibleSyncReport(BaseModel):
    source: str
    format: Literal["csv", "json"]
    timeline_id: str
    step: int
    mode: Literal["upsert", "merge", "replace"]
    dry_run: bool = False
    created_entities: list[str] = []
    updated_entities: list[str] = []
    removed_entities: list[str] = []
    unchanged_entities: list[str] = []
    added_aliases: list[tuple[str, str]] = []
    removed_aliases: list[tuple[str, str]] = []
    created_edges: list[str] = []
    updated_edges: list[str] = []
    removed_edges: list[str] = []
    unchanged_edges: list[str] = []
    inflections_generated: int = 0
    warnings: list[str] = []
    # Properties / helpers: report.has_changes, report.summary()
```

### `AuditIssue`, `AuditReport` & `DuplicateCandidate`
```python
AuditSeverity = Literal["info", "warning", "error"]

class DuplicateCandidate(BaseModel):
    duplicate_id: str
    canonical_id: str
    score: float                         # similarity score in [0.0, 100.0]
    reason: str                          # e.g. "canonical_name matched alias:Mr. Darcy (100.0%)"
    duplicate_name: str = ""
    canonical_name: str = ""
    duplicate_kind: str = ""
    canonical_kind: str = ""
    candidate_field: str = ""
    matched_against: str = ""
    matched_value: str = ""
    step: int = 0
    timeline_id: str = "main"

class AuditIssue(BaseModel):
    code: str                            # e.g. "exonym_leak", "near_duplicate_entity", "unexplained_register_flip"
    rule_name: str
    severity: AuditSeverity = "warning"
    step: int
    timeline_id: str = "main"
    message: str
    subject_key: str | None = None
    entity_ids: list[str] = []
    evidence: dict[str, Any] = {}
    suggestion: str | None = None

class AuditReport(BaseModel):
    timeline_id: str
    steps_audited: list[int] = []
    issues: list[AuditIssue] = []
    rules_executed: list[str] = []
    # Properties / helpers: report.passed, report.error_count, report.warning_count,
    # report.info_count, report.has_issues_at_or_above(min_severity),
    # report.by_severity(sev), report.by_code(code), report.duplicate_candidates(min_score=0.0),
    # report.summary(),
    # report.format_repair_instructions(min_severity="error", max_issues=None, header=..., include_snippets=True)
```

### `EntityMergeReport`
```python
class EntityMergeReport(BaseModel):
    timeline_id: str
    step: int
    duplicate_id: str
    canonical_id: str
    dry_run: bool = False
    aliases_added: list[str] = []
    aliases_tombstoned: list[str] = []
    base_attributes_merged: list[str] = []
    state_keys_merged: list[str] = []
    edges_migrated: list[str] = []
    edges_merged_on_collision: list[str] = []
    edges_tombstoned: list[str] = []
    self_loops_dropped: list[str] = []
    location_refs_updated: list[str] = []
    voice_exemplars_migrated: int = 0
    episodes_migrated: int = 0
    epistemics_migrated: int = 0
    warnings: list[str] = []
    # Helper: report.summary()
```

---

## Extension Protocols, Adapters & Policies

- `MemoryStore`: Storage protocol implemented by `CausalStore`. Inject a custom backend via `CanonArcEngine(store=my_store)`. `CausalStore` also provides `get_known_episode_ids()` and bounded/pre-filtered `get_lineage_episodes(..., limit=..., pov_entity_ids=..., cue_entity_ids=..., pinned_episode_ids=..., query_text=...)` backed by SQLite FTS5 (`episodes_fts`).
- `MentionResolver`: Pluggable entity/coreference resolution protocol (`resolve(text, explicit_mentions, entities, aliases) -> list[ResolvedMention]`). Defaults to `DefaultLexicalResolver(fuzzy_threshold=88.0, *, short_name_inflection=True, ignore_patterns=(), text_normalizer=None)`, which uses word-boundary matching, short-name inflection fallback, and optional `rapidfuzz` inflection matching.
- **Markup & Placeholder Stripping (`canonarc.text_masking`)**: Configure `CanonArcEngine(..., ignore_patterns=..., text_normalizer=...)` once to automatically mask or strip non-prose tokens across lexical resolution (`retrieve_context()` + `prev_text` halo), Gate-1/Gate-2 voice harvesting (`harvest_voice()`), vector query embeddings, and text-level audit rules (`ExonymLeakRule`, `TargetTermComplianceRule`, `ScriptConsistencyRule`, plus `AuditContext.masked_source_text` / `masked_target_text` for custom rules).
  - **Offset-preserving masking vs. prose stripping:** Audit rules and lexical resolution use `mask_ignored_spans()` (replacing matched spans with spaces of identical length while preserving `\n`, keeping `matched_spans` and `_extract_snippet()` aligned with raw text); voice harvesting and query embeddings use `strip_ignored_spans()` (removing spans and collapsing horizontal whitespace so long markup does not starve attribution windows or pollute `VoiceExemplar.text`).
  - **Vetted built-in pattern constants:** `COMMON_MARKUP_PATTERNS` bundles `MARKUP_HTML_COMMENTS`, `MARKUP_MARKDOWN_IMAGES`, `MARKUP_MARKDOWN_LINK_URLS` (masks `(url)` while keeping visible `[anchor]` prose), `MARKUP_HTML_TAGS` (masks tags without erasing inner prose), `MARKUP_BRACKET_PLACEHOLDERS` (`[[IMG_1]]`), `MARKUP_BRACE_PLACEHOLDERS` (`{player_name}`), and `MARKUP_PRINTF_PLACEHOLDERS` (`%s`). `COMMON_PLACEHOLDER_PATTERNS` bundles only the strict verbatim placeholder patterns (`MARKUP_BRACKET_PLACEHOLDERS`, `MARKUP_BRACE_PLACEHOLDERS`, `MARKUP_PRINTF_PLACEHOLDERS`) for use with `PlaceholderPreservationRule`.
  - **Caveat:** Prefer specific non-greedy patterns (avoid `<b>.*?</b>` which erases inner prose), and note that `text_normalizer: Callable[[str], str]` changes string length so audit snippets fall back to `AuditContext.target_snippet_text` (normalized text).
- `EpisodeRanker` / `EpisodicScoringPolicy`: Pluggable Tier 6 episodic memory ranker (`DefaultEpisodeRanker`) decoupled from `VoiceBank`, configurable via `SchemaPolicy.episodic_policy`.
- `EpisodeConsolidator` / `ConsolidationSummary` (`canonarc.episodic`): Pluggable Tier 6 episode summarizer protocol used by `consolidate_episodes_auto()` and `consolidate_chapter()`. Ships with `LLMEpisodeConsolidator(client: JsonCompletionClient, max_retries=1, max_tokens=512, temperature=0.0, min_salience=4.5, max_salience=7.5)` (reuses any `OpenAICompatibleExtractor`, `CallableExtractor`, or `CachedDeltaExtractor` with unified `TokenUsage` tracking), `CallableEpisodeConsolidator(fn)`, `HeuristicEpisodeConsolidator(max_chars=600, default_salience=6.0, max_participants=5)`, and `build_consolidation_prompt(episodes, entity_names=..., chapter_id=...)`.
- `EngineHooks` / `BaseEngineHooks`: Middleware callbacks (`on_before_commit`, `on_after_commit`, `on_rollback`, `on_branch`, `on_gate_rejected`).
- `AuditRule` / `CallableAuditRule` (`canonarc.audit`): Pluggable QA rule protocol (`name`, `required_tiers`, `check(ctx: AuditContext) -> Sequence[AuditIssue]`). Also exposes `run_audit_draft(...)` for pre-commit draft auditing. Built-in rules (`default_audit_rules()`):
  - `ExonymLeakRule`: Detects untranslated `canonical_name` leakage in `target_text` when `target_name != canonical_name` (case-sensitive CJK/word-boundary matching, source-mention scoping, `base_attributes["allow_source_name"]` opt-out).
  - `TargetTermComplianceRule`: Verifies that every entity matched in `source_text` with a `target_name` appears in `target_text` via its locked `target_name`, `lang=target_lang` / `kind="inflection"` aliases, short-name stems, or conservative leading-stem fuzzy matching. Emits `target_term_mismatch` (`severity="error"` by default) when a suspected wrong spelling/transliteration is found (e.g. `"Febi"` for `"Fibi"` or `"Herkul Poaro"` for `"Erkil Poaro"`), and `target_term_missing` (`severity="warning"` by default) for pure omissions/pronominalization. Supports `base_attributes["allow_omission"]` and `kinds` filtering.
  - `NearDuplicateEntityRule`: Detects alias/persona/exonym hijacks and `rapidfuzz` name collisions against existing entities (`fuzzy_threshold=88.0`, `same_kind_only=True`, `base_attributes["ignore_duplicate_of"]` opt-out).
  - `UnexplainedRegisterFlipRule`: Flags mutations to `policy.hard_filter_keys` (e.g., `register`) lacking `caused_by_episode_id` and `reason` (skipping step `0`), plus `dangling_causal_episode` errors.
  - `RegisterVocabularyDriftRule`: Detects mixed register vocabularies (e.g., `Vi`/`ti` mixed with `formal`/`informal`) or values outside an explicit `allowed_values` set.
  - Opt-in built-in rules (not in `default_audit_rules()`):
    - `ScriptConsistencyRule(expected_script=None, *, severity="error", mixed_script_severity="error", allowed_tokens=None, allowed_patterns=None, allow_entity_names=True, allow_roman_numerals=True, ignore_urls=True, ignore_markup_and_placeholders=True, skip_exonym_leaks=True, snippet_radius=30)`: Zero-dependency Unicode script checker for `target_text`. Accepts a single script name (`"latin"`, `"cyrillic"`, `"greek"`, etc.), a sequence of allowed scripts (e.g. `("han", "hiragana", "katakana")` for Japanese), or `None` to infer from unambiguous BCP-47 `ctx.lang` tags (`sr-Latn`, `sr-Cyrl`, `ru`, `uk`, `bg`, `el`, `ja`, etc.; safe no-op on ambiguous tags like `sr`). Emits aggregated `script_drift` (at most one per unexpected script) and `mixed_script_token` (homoglyph contamination) issues with snippets for `format_repair_instructions()`. Auto-exempts canon-locked entity names/aliases, Roman numerals, URLs, markup/placeholders, and verbatim `ExonymLeakRule` leaks to minimize false positives and duplicate repair bullets.
    - `PlaceholderPreservationRule(*, patterns=None, use_context_ignore_patterns=False, check_order=False, missing_severity="error", unexpected_severity="error", duplicated_severity="error", order_severity="warning", snippet_radius=30)`: Deterministic rule inspecting unmasked `source_text` and `target_text` broadest-first (without double-counting nested spans) to verify exact placeholder multiplicity. Defaults to `COMMON_PLACEHOLDER_PATTERNS` rather than `ctx.ignore_patterns` to avoid false positives on translatable markup (Markdown image alt text, HTML comments); pass `patterns=...` or `use_context_ignore_patterns=True` to customize. Pairs missing tokens with novel same-family target tokens into `placeholder_mutated` (e.g. `[[IMG_1]]` → `[[SLIKA_1]]`), and emits `placeholder_missing`, `placeholder_duplicated`, `placeholder_unexpected`, and (when `check_order=True`) `placeholder_order_mismatch`.
    - `UntranslatedSourceRule(*, severity="error", paragraph_severity="error", similarity_threshold=92.0, min_chars=25, min_words=5, check_paragraphs=True, min_paragraph_words=6, exempt_entity_names=True, snippet_radius=40)`: Opt-in heuristic rule complementing `ScriptConsistencyRule` for same-script language pairs (e.g. EN → SR-Latn/DE/FR). Operates on masked text, exempts `allow_source_name` / identical-`target_name` entities and short utterances, and flags whole-step (`untranslated_source_echo`) or paragraph-level (`untranslated_paragraph_echo`) verbatim echoes.
    - `OmissionGuardRule(*, min_length_ratio=0.45, max_length_ratio=2.5, min_source_chars=80, min_paragraph_ratio=0.5, min_source_paragraphs=3, min_sentence_ratio=0.45, min_source_sentences=4, empty_severity="error", truncation_severity="error", bloat_severity="warning", structure_severity="warning")`: Opt-in coarse structural guardrail operating on `strip_ignored_spans()` text so markup/placeholders do not skew ratios. Flags `empty_translation` and `translation_truncated` as `"error"`, and softer signals (`translation_bloated`, `paragraph_collapse`, `sentence_collapse`) as `"warning"`. Tune `min_length_ratio` / `max_length_ratio` for cross-script pairs (e.g. EN ↔ CJK).
- `TierProfile` & `SchemaPolicy`: `TierProfile` (`LEXICON_ONLY` = Tiers 1+4, `STATE_ONLY` = Tiers 1–4, `FULL` = Tiers 1–6) and frozen `SchemaPolicy` dataclass on `DomainPreset.policy` defining `profile`, optional custom `enabled_tiers`, `active_tiers`, `has_tier(tier)`, convenience factories (`SchemaPolicy.lexicon_only()`, `state_only()`, `full()`, `with_tiers(...)`), `edge_layer_order`, `facade_relation`, `belief_relation`, `hard_filter_keys`, `soft_boost_weights`, `fallback_on_empty_hard_filter`, `hard_filter_mismatch_penalty`, and `episodic_policy`.
- `TranslationPreset(policy=None, *, include_background_edges=False, show_entity_ids=False, include_kind=True, extra_attributes=("notes",), max_attr_chars=160, empty_text="", max_voice_per_speaker=None, max_context_chars=None)`: Localization preset that renders human-readable display names by default (auto-disambiguating colliding names and resolving off-screen episode/edge IDs via `StepContext.entity_names`), includes entity `[kind]` (`True`, `"non_character"`, or `False`) and truncated extra `base_attributes` (default `("notes",)`), omits empty section headers and returns `empty_text` on empty contexts, caps per-speaker voice exemplars, and trims whole lines in reverse priority order (background edges → episodes → voice → active edges → entities) when `max_context_chars` is set.

---

## Embedders (`canonarc.embedders`)

- `Embedder` (Protocol): `embed_text(text: str) -> np.ndarray`, plus optional `embed_batch(texts: Sequence[str]) -> list[np.ndarray]`.
- `OfflineNgramEmbedder(dim: int = 128)`: Default zero-dependency character/word n-gram embedder with CJK unit tokenization and Weinberger signed CRC32 feature hashing.
- `CallableEmbedder(fn, *, batch_fn=None, embedder_id=None, dim=None)`: Wraps any custom Python function or in-process model (e.g. `SentenceTransformer.encode`).
- `OpenAICompatibleEmbedder(*, model="text-embedding-3-small", api_key=None, base_url="https://api.openai.com/v1", timeout=30.0, headers=None, dim=None)`: Zero-dependency `urllib.request` client for OpenAI, LM Studio, LiteLLM, or vLLM `/v1/embeddings` endpoints.
- `OllamaEmbedder(*, model="nomic-embed-text", base_url="http://localhost:11434", timeout=30.0, dim=None)`: Zero-dependency client for local Ollama `/api/embed` endpoints.
- `embed_texts_batch(embedder, texts)` & `get_embedder_info(embedder)`: Helper utilities for batch vectorization and provenance metadata (`embedder_id`, `embedding_dim`) stored in `store_metadata`. Raises `EmbedderMismatchError` when `on_embedder_mismatch="raise"`.

---

## Extractors & Token Telemetry (`canonarc.extractors`)

- `DeltaExtractor` (Protocol): `extract_delta(prompt: str, scene_text: str) -> StepDelta` (extractors may also implement `extract_delta_messages(system_prompt: str, user_prompt: str, scene_text: str) -> StepDelta` for prefix-cache-friendly split prompts).
- `JsonCompletionClient` (`@runtime_checkable` Protocol): `call_json_api(prompt: str, *, system_prompt: str | None = None, max_tokens: int | None = None, temperature: float | None = None) -> tuple[str, TokenUsage]`. Implemented by `OpenAICompatibleExtractor`, `CallableExtractor`, and `CachedDeltaExtractor` so all three LLM subsystems (`DeltaExtractor`, `LLMInflectionGenerator`, `LLMVoiceQualityScorer`) can share one adapter and unified `on_usage` accounting.
- `CallableExtractor(fn=None, *, messages_fn=None, model=None, system_prompt=None, max_retries=1, repair_fn=None, on_usage=None)`:
  - Wraps either a 2-arg `(prompt: str, schema: dict)` callable (`fn`) or a 3-arg `(system_prompt: str, user_prompt: str, schema: dict)` callable (`messages_fn`) with `<think>` block stripping, Markdown JSON fence stripping, Pydantic validation, and automatic validation-error retry prompts.
  - Implements `extract_delta_messages(...)` so LangChain and custom SDK wrappers receive split `(system_prompt, user_prompt)` messages with `system_prompt` kept byte-identical across retries for provider prompt caching.
  - Implements `call_json_api(prompt, *, system_prompt=None, max_tokens=None, temperature=None, schema=None) -> tuple[str, TokenUsage]` for generic JSON completions (`LLMInflectionGenerator`, `LLMVoiceQualityScorer`), passing `schema={}` by default (instead of `StepDelta`'s schema) and forwarding `max_tokens`/`temperature` only when accepted by the underlying callable.
  - Callables may return either `str | dict | StepDelta` or a `(payload, usage)` tuple (accepting `TokenUsage` or OpenAI/DeepSeek/Anthropic/LangChain usage dicts), automatically populating `last_usage`, `total_usage`, optional `on_usage(TokenUsage)` callback, and `CachedDeltaExtractor` cache keys (`model`) and usage logs.
- `OpenAICompatibleExtractor(*, model="gpt-4o-mini", api_key=None, base_url="https://api.openai.com/v1", temperature=0.0, max_tokens=None, reasoning_effort=None, response_format=None, use_json_object_format=True, timeout=120.0, max_retries=1, system_prompt=..., headers=None, on_usage=None)`:
  - Zero-dependency HTTP client for OpenAI-compatible `/chat/completions` endpoints with transient HTTP retry (`429`, `5xx`), automatic fallback when an endpoint rejects `response_format={"type": "json_object"}`, and `last_usage` / `total_usage` / `on_usage(TokenUsage)` tracking.
  - **Convenience Factory Methods:**
    - `OpenAICompatibleExtractor.from_lmstudio(*, model=None, base_url=None, ...)`: Connects to local LM Studio (`http://localhost:1234/v1`) and auto-discovers the currently loaded LLM via `resolve_lmstudio_model()` when `model` is omitted.
    - `OpenAICompatibleExtractor.from_gemini(*, model=None, api_key=None, ...)`: Pre-configured for Google Gemini's OpenAI-compatible endpoint (`GEMINI_API_KEY`).
    - `OpenAICompatibleExtractor.from_deepseek(*, model=None, api_key=None, ...)`: Pre-configured for the DeepSeek API (`DEEPSEEK_API_KEY`).
- `CachedDeltaExtractor(extractor, cache_dir, *, model_id=None, system_prompt=None, bypass_cache=False)`:
  - Content-addressed SHA-256 disk-caching wrapper around any `DeltaExtractor`. Tracks both `billable_usage` (0 tokens on cache hits) and `historical_usage` (`TokenUsage`).
  - Exposes `.model` (alias for `model_id`) and passes `call_json_api(...)` through to the wrapped `JsonCompletionClient` so the same handle can be shared with `LLMInflectionGenerator` and `LLMVoiceQualityScorer`.
- `TokenUsage`: Dataclass recording `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_prompt_tokens`, `reasoning_tokens`, `api_calls`, `cache_hits`, `cache_misses`, `retry_count`, `validation_errors`, `latency_ms`, `.estimate_cost_usd(...)`, and `.from_api_usage(usage_dict)` (normalizing OpenAI, DeepSeek, Anthropic, and LangChain `usage_metadata` dicts).
- `load_dotenv(dotenv_path=".env", *, override=False) -> dict[str, str]`: Zero-dependency `.env` loader.

---

## Inflection Generators (`canonarc.inflections`)

- `InflectionRequest`: Frozen dataclass (`name`, `lang`, `gender=None`, `kind="character"`, `declension=None`) for batched morphological inflection requests.
- `InflectionGenerator` (Protocol): `generate_inflections(name, lang, *, gender=None, kind="character", declension=None) -> list[str]`.
- `CallableInflectionGenerator(fn, *, use_cache=True)`: Wraps any Python callable with in-memory caching and `generate_inflections_batch(items, *, batch_size=25)`.
- `LLMInflectionGenerator(extractor, *, cache_dir=None, max_tokens=None, temperature=0.0, max_retries=1, system_prompt=...)`: One-shot and batched (`generate_inflections_batch(items, *, batch_size=25)`) LLM inflection generator backed by any `JsonCompletionClient` (`OpenAICompatibleExtractor`, `CallableExtractor`, `CachedDeltaExtractor`, or custom client), with shared per-item memory + disk caching, language-grouped integer-indexed batch prompts, per-item fallback on dropped IDs, and `last_usage` / `total_usage` tracking.
- `generate_inflections_batch(generator, items, *, batch_size=25) -> list[list[str]]`: Dispatcher helper that calls `generator.generate_inflections_batch()` when available and falls back to per-item calls for custom generators.
- `build_inflection_prompt(...)` / `build_batch_inflection_prompt(items)`: Deterministic single-item and batched JSON prompt builders.

---

## Voice Quality Scorers (`canonarc.voice`)

- `VoiceScoreResult`: Frozen dataclass (`score: float`, `reason: str = ""`, `metadata_updates: dict[str, str] = {}`) returned by Gate-2 voice quality scorers (scores clamped to `[0.0, 10.0]`).
- `VoiceQualityScorer` (Protocol): `score_candidate(candidate: DialogueCandidate, *, metadata: dict[str, str], context: StepContext | None = None) -> float | VoiceScoreResult` (plus optional `score_candidates_batch(...)`).
- `HeuristicVoiceQualityScorer(*, optimal_min_words=16, optimal_max_words=45, distinctiveness_weight=4.2, lexical_diversity_weight=2.3, length_weight=1.5, attribution_weight=1.2, expressiveness_weight=0.8, artifact_penalty=2.5)`: Default zero-LLM deterministic Gate-2 scorer used by `CanonArcEngine` when `voice_scorer` is not provided.
- `CallableVoiceQualityScorer(fn)`: Wraps any 1-, 2-, or 3-argument Python callable returning `float | VoiceScoreResult`.
- `LLMVoiceQualityScorer(extractor, *, batch_size=15, max_tokens=None, temperature=0.0, max_retries=1, system_prompt=..., fallback_scorer=None)`: Batched LLM Gate-2 judge backed by any `JsonCompletionClient` (`OpenAICompatibleExtractor`, `CallableExtractor`, `CachedDeltaExtractor`, or custom client) with `last_usage` / `total_usage` tracking.
- `score_voice_candidates_batch(scorer, candidates, metadatas, *, context=None) -> list[VoiceScoreResult]`: Dispatcher helper calling `scorer.score_candidates_batch()` when available with per-item fallback.

---

## CLI Reference (`canonarc`)

Installed as a console script via `[project.scripts]` (`canonarc.cli:main`):

```bash
# Render the hierarchical timeline branch tree (pass --json for machine-readable output)
canonarc tree story_save.sqlite [--json]

# Inspect active point-in-time world state on a timeline
canonarc inspect story_save.sqlite [--timeline main] [--step 85] [--inclusive] [--pov char_arthur] [--text "scene cue"] [--json]

# Diff active entities, edges, and aliases between two branches at a specific step
canonarc diff story_save.sqlite timeline_a timeline_b --step 85 [--json]

# Show causal revision trajectory (valid_from..valid_to) for an entity, edge, or alias key
canonarc history story_save.sqlite char_arthur [--timeline main] [--type entity|edge|alias] [--json]

# Export Series Bible to CSV, JSON, or Markdown (use '-' for stdout)
canonarc export story_save.sqlite bible.csv [--format auto|csv|json|md] [--timeline main] [--step 0] [--kinds character,term] [--lang sr] [--include-state] [--no-edges] [--no-aliases] [--include-inflections|--exclude-inflections] [--excel-bom] [--json]

# Sync a CSV or JSON Series Bible into the database (offline; auto_inflect disabled in CLI)
canonarc import story_save.sqlite bible.csv [--format auto|csv|json] [--timeline main] [--step 0] [--mode upsert|merge|replace] [--target-lang sr] [--column-map SRC=DST] [--dry-run] [--json]

# Run deterministic continuity and localization QA audit rules (exits 1 if --fail-on threshold matched)
canonarc audit story_save.sqlite [--timeline main] [--step 2 --source s2_en.txt --target s2_sr.txt] [--from-step 0 --to-step 10] [--pairs-jsonl pairs.jsonl] [--rules exonym_leak,near_duplicate_entity] [--exclude-rules ...] [--min-severity info|warning|error] [--fail-on error|warning|none] [--json]

# Scan all active entities for near-duplicate merge candidates (read-only; prints ready-to-run merge hints)
canonarc duplicates story_save.sqlite [--timeline main] [--step N] [--threshold 88.0] [--allow-cross-kind] [--json]

# Atomically merge a duplicate entity into a canonical entity
canonarc merge story_save.sqlite char_poirot char_hercule_poirot [--timeline main] [--step N] [--no-attributes] [--no-edges] [--no-voice] [--no-episodes] [--conflict-strategy canonical_wins|duplicate_wins] [--allow-kind-mismatch] [--reason "..."] [--dry-run] [--json]
```
