# API Reference

## `NarrativeMemoryEngine`

The central facade for state tracking, retrieval, mutation, and time travel.

```python
class NarrativeMemoryEngine:
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        timeline_id: str = "main",
        embedder: Embedder | None = None,
        gate: IngestionGate | None = None,
        preset: DomainPreset | None = None,
    ) -> None: ...
```

### Seeding Methods
- `register_entity(entity: EntityRecord, aliases: Sequence[str] = (), step: int = 0, timeline_id: str | None = None) -> None`: Registers an entity and its aliases at `step`.
- `seed_entities(entities: Sequence[EntityRecord], step: int = 0, timeline_id: str | None = None) -> None`: Batch registers entities.

### Retrieval Methods
- `retrieve_context(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int = 3, inclusive: bool = False, timeline_id: str | None = None) -> StepContext`: Executes unified Tier 1–5 retrieval at `step`.
- `format_context(ctx: StepContext) -> str`: Formats `StepContext` using the configured `DomainPreset`.

### Transaction & Mutation Methods
- `step_transaction(step: int, text: str = "", mentions: Sequence[str] = (), top_k_voice: int = 3, timeline_id: str | None = None, rerun_clean: bool = True) -> StepTransaction`: Context manager for clean read-and-commit workflows.
- `apply_delta(step: int, delta: StepDelta, timeline_id: str | None = None) -> None`: Directly applies state deltas.
- `extract_dialogue_candidates(text: str, source_text: str = "") -> list[DialogueCandidate]`: Extracts candidate dialogue lines passing Gate 1.
- `index_voice_exemplars(step: int, exemplars: Sequence[VoiceExemplar], bypass_gate: bool = False, timeline_id: str | None = None) -> list[VoiceExemplar]`: Indexes voice exemplars into Tier 5 Vector DB.
- `commit_step(step: int, delta: StepDelta | None = None, exemplars: Sequence[VoiceExemplar] = (), bypass_gate: bool = False, timeline_id: str | None = None) -> list[VoiceExemplar]`: Atomically commits delta and exemplars for a step.

### Time-Travel Methods
- `rollback_to(step: int, timeline_id: str | None = None) -> int`: Rewinds the timeline to `step`, deleting future records.
- `branch(at_step: int, name: str, from_timeline: str | None = None, switch: bool = False, metadata: dict | None = None) -> TimelineInfo`: Forks an alternate timeline at `at_step`.
- `checkout(timeline_id: str) -> None`: Switches the active timeline.
- `list_timelines() -> list[TimelineInfo]`: Lists all existing timelines.
- `diff_timelines(timeline_a: str, timeline_b: str, at_step: int) -> dict[str, Any]`: Compares active states between two branches.
- `export_snapshot(target_path: str | Path, at_step: int, timeline_id: str | None = None) -> Path`: Exports a standalone, flattened `.sqlite` save file.

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
    valid_from_step: int = 0
    valid_to_step: int | None = None
```

### `DirectedEdgeRecord`
```python
class DirectedEdgeRecord(BaseModel):
    source_id: str
    target_id: str
    state: dict[str, Any] = {}
    valid_from_step: int = 0
    valid_to_step: int | None = None
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
```
