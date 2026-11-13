# API Reference

## `NarrativeMemoryEngine`

The central facade for state tracking, episodic memory, voice retrieval, mutation, and causal time travel.

```python
class NarrativeMemoryEngine:
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
    ) -> None: ...
```

### Seeding Methods
- `register_entity(entity: EntityRecord, aliases: Sequence[str | AliasRecord] = (), step: int = 0, timeline_id: str | None = None) -> None`: Registers an entity and its aliases at `step`.
- `seed_entities(entities: Sequence[EntityRecord], step: int = 0, timeline_id: str | None = None) -> None`: Batch registers entities.

### Retrieval & Query Methods
- `retrieve_context(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int = 3, top_k_episodes: int = 5, pov_entity_ids: Sequence[str] | None = None, pov_mode: Literal["any", "all"] = "any", edge_types: Sequence[str] | None = None, location_id: str | None = None, inclusive: bool = False, timeline_id: str | None = None, fallback_on_empty_mentions: Literal["all", "pov_only", "none"] = "pov_only") -> StepContext`: Executes unified Tier 1–6 retrieval at `step`. Co-present scene entities (matched in text/mentions, sharing the active `location_id`, or POV fallback) and edges between them populate `entities` and `active_edges`, while 1-hop off-screen social graph neighbors populate `related_entities` and `background_edges`.
- `query_episodes(cutoff_step: int = 999999999, participants: Sequence[str] = (), match_all_participants: bool = False, predicate: str | None = None, location_id: str | None = None, known_to: Sequence[str] = (), min_salience: float = 0.0, step_range: tuple[int, int] | None = None, timeline_id: str | None = None) -> list[EpisodeRecord]`: Queries episodic ledger by participants, predicate, location, and epistemic visibility.
- `format_context(ctx: StepContext) -> str`: Formats `StepContext` using the configured `DomainPreset`.

### Extraction & Normalization Methods
- `build_extraction_prompt(ctx: StepContext, scene_text: str) -> str`: Combines `preset.extraction_instructions()`, active entity/episode IDs, and `StepDelta.model_json_schema()` into an LLM extraction prompt.
- `normalize_delta(delta: StepDelta, step: int | None = None, timeline_id: str | None = None) -> StepDelta`: Normalizes canonical names/aliases into canonical `entity_id`s and resolves intra-step symbolic `episode_id` references.

### Transaction & Mutation Methods
- `step_transaction(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int = 3, top_k_episodes: int = 5, pov_entity_ids: Sequence[str] | None = None, pov_mode: Literal["any", "all"] = "any", edge_types: Sequence[str] | None = None, location_id: str | None = None, timeline_id: str | None = None, rerun_clean: bool = True) -> StepTransaction`: Context manager for clean read-and-commit workflows. Supports `tx.extract_and_commit(scene_text)` when a `DeltaExtractor` is configured.
- `apply_delta(step: int, delta: StepDelta, timeline_id: str | None = None) -> list[EpisodeRecord]`: Directly applies state deltas, episodes, and inform actions.
- `index_episodes(step: int, episodes: Sequence[EpisodeRecord], timeline_id: str | None = None) -> list[EpisodeRecord]`: Embeds and indexes episodes into Tier 6.
- `inform_entity(step: int, episode_id: str, entity_id: str, epistemic: Literal["direct", "told", "deceived"] = "told", believed_summary: str | None = None, timeline_id: str | None = None) -> None`: Records secondhand, delayed, or deceptive epistemic awareness of an episode.
- `extract_dialogue_candidates(text: str, source_text: str = "") -> list[DialogueCandidate]`: Extracts candidate dialogue lines passing Gate 1.
- `index_voice_exemplars(step: int, exemplars: Sequence[VoiceExemplar], bypass_gate: bool = False, timeline_id: str | None = None) -> list[VoiceExemplar]`: Indexes voice exemplars into Tier 5 Vector DB.
- `commit_step(step: int, delta: StepDelta | None = None, exemplars: Sequence[VoiceExemplar] = (), episodes: Sequence[EpisodeRecord] = (), bypass_gate: bool = False, timeline_id: str | None = None) -> list[VoiceExemplar]`: Atomically commits deltas, episodes, and voice exemplars for a step.

### Time-Travel Methods
- `rollback_to(step: int, timeline_id: str | None = None, force: bool = False) -> int`: Rewinds the timeline to `step`, deleting all future graph revisions, voice exemplars, and episodes. Raises `TimelineBranchConflictError` if a child branch forked at `fork_step > step` unless `force=True`.
- `branch(at_step: int, name: str, from_timeline: str | None = None, switch: bool = False, metadata: dict | None = None) -> TimelineInfo`: Forks an alternate timeline at `at_step`.
- `checkout(timeline_id: str) -> None`: Switches the active timeline.
- `list_timelines() -> list[TimelineInfo]`: Lists all existing timelines.
- `diff_timelines(timeline_a: str, timeline_b: str, at_step: int) -> dict[str, Any]`: Compares active states between two branches.
- `export_snapshot(target_path: str | Path, at_step: int, timeline_id: str | None = None) -> Path`: Exports a standalone, flattened `.sqlite` save file preserving active state and lineage episodes.

---

## Core Models

### `EntityRecord`
```python
class EntityRecord(BaseModel):
    entity_id: str
    kind: str = "character"
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

### `AliasRecord`
```python
class AliasRecord(BaseModel):
    alias: str
    entity_id: str
    kind: Literal["permanent", "epithet", "disguise", "title"] = "epithet"
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
    salience: float = 5.0                # 0.0 to 10.0
    metadata: dict[str, Any] = {}
    match_score: float = 0.0
```

### `InformDelta`
```python
class InformDelta(BaseModel):
    episode_id: str
    entity_id: str
    epistemic: Literal["direct", "told", "deceived"] = "told"
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

### `StepDelta`
```python
class StepDelta(BaseModel):
    entities: list[EntityDelta] = []
    edges: list[EdgeDelta] = []
    episodes: list[EpisodeRecord] = []
    inform: list[InformDelta] = []
```

### `StepContext`
```python
class StepContext(BaseModel):
    timeline_id: str
    step: int
    resolved_mentions: list[ResolvedMention] = []
    entities: list[EntityRecord] = []               # Co-present scene entities
    active_edges: list[DirectedEdgeRecord] = []      # Edges strictly between co-present scene entities
    related_entities: list[EntityRecord] = []       # 1-hop off-screen social graph neighbors
    background_edges: list[DirectedEdgeRecord] = []  # Edges from scene entities to off-screen neighbors
    voice_exemplars: list[VoiceExemplar] = []
    episodes: list[EpisodeRecord] = []
```

---

## Extension Protocols & Policies

- `MemoryStore`: Storage protocol implemented by `CausalStore`. Inject a custom backend via `NarrativeMemoryEngine(store=my_store)`.
- `MentionResolver`: Pluggable entity/coreference resolution protocol (`resolve(text, explicit_mentions, entities, aliases) -> list[ResolvedMention]`). Defaults to `DefaultLexicalResolver(fuzzy_threshold=None)`, which uses word-boundary matching and optional `rapidfuzz` inflection matching.
- `DeltaExtractor`: LLM extraction protocol (`extract_delta(prompt, scene_text) -> StepDelta`). Used automatically by `StepTransaction.extract_and_commit(...)`.
- `SchemaPolicy`: Frozen dataclass on `DomainPreset.policy` defining `edge_layer_order`, `facade_relation`, `belief_relation`, `hard_filter_keys`, `soft_boost_weights`, `fallback_on_empty_hard_filter`, and `hard_filter_mismatch_penalty`.
