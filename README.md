# `narrative-memory`

**Tier 1–5 Temporal Graph, Voice Bank & Causal Time-Travel Engine for LLMs**

`narrative-memory` is a standalone, local-first Python library that gives LLM applications **stateful character memory, directed interpersonal relationships, empirical voice anchoring, and Git-like discrete time travel (`rollback_to(step)` and `branch(step)`)**.

## Core Features

- **Discrete Causal Time**: State transitions, pairwise relationships, and dialogue exemplars are indexed by discrete causal steps (turns, scenes, ticks, or chunks).
- **Zero-Copy Branching (`branch(at_step, name)`)**: Fork alternate reality timelines with zero data duplication.
- **Instant Rollback (`rollback_to(step)`)**: Rewind timeline state atomically in `< 1ms`.
- **Tiers 1–4 Temporal Graph**: Track immutable base attributes, dynamic pairwise registers/stances, and 1-hop coreference aliases.
- **Tier 5 Gated Voice Bank**: Style and syntax anchoring without "error compounding", protected by a 2-stage anti-drift gate.
- **Single-File SQLite Portability**: Everything lives in a single portable `.sqlite` file (or `:memory:`).

## Quickstart

```python
from narrative_memory import NarrativeMemoryEngine, EntityRecord, EdgeDelta, StepDelta

engine = NarrativeMemoryEngine(":memory:")

# Tier 1 Seeding
engine.register_entity(
    EntityRecord(
        entity_id="char_arthur",
        canonical_name="Arthur",
        base_attributes={"gender": "M", "title": "King"},
    )
)
engine.register_entity(
    EntityRecord(
        entity_id="char_merlin",
        canonical_name="Merlin",
        base_attributes={"gender": "M", "title": "Court Wizard"},
    ),
    aliases=["the old wizard"],
)

# Step 1: Initial relationship
engine.commit_step(
    step=1,
    delta=StepDelta(
        edges=[
            EdgeDelta(
                source_id="char_arthur",
                target_id="char_merlin",
                state={"register": "formal", "tone": "respectful"},
            )
        ]
    ),
)

# Branch a counterfactual timeline at step 1
branch = engine.branch(at_step=1, name="rebellion_route")
```
