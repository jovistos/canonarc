# `narrative-memory`

<p align="center">
  <strong>Tier 1–6 Temporal Graph, Episodic Ledger, Voice Bank & Causal Time-Travel Engine for LLMs</strong>
</p>

<p align="center">
  <a href="docs/tutorial-and-study-book.md"><strong>Study Book & Deep Tutorial</strong></a> &bull;
  <a href="docs/architecture.md">Architecture</a> &bull;
  <a href="docs/time-travel.md">Time Travel Guide</a> &bull;
  <a href="docs/voice-bank.md">Voice Bank & Anti-Drift</a> &bull;
  <a href="docs/presets.md">Domain Presets</a> &bull;
  <a href="docs/api-reference.md">API Reference</a>
</p>

---

## What is `narrative-memory`?

Most LLM memory frameworks (`mem0`, `zep`, `langmem`) are designed for **conversational chatbots** operating in continuous wall-clock time. They fall short when building **Interactive Fiction, Visual Novels (Ren'Py / Ink), Multi-Agent Simulations, Creative Writing Co-Pilots, and Sequential Localization Pipelines**, where time moves in discrete **causal steps** (turns, scenes, ticks, or chunks) and narrative state must be **rewindable, branchable, and spoiler-free**.

`narrative-memory` gives your application:
- **Tiers 1–4 Temporal Property Graph**: Tracks immutable base attributes (gender, declension, archetypes), asymmetric directed relationships (`A -> B != B -> A`), 1-hop coreference aliases (`"the old wizard"` -> `char_merlin`), and rolling state deltas with trajectory history (`prior_state`, `revision_count`).
- **Tier 5 Gated Dynamic Voice Bank**: Anchors dialogue style and syntax using historical in-character lines without "error compounding" or self-conditioning drift.
- **Tier 6 Episodic Event Ledger & Epistemic Visibility**: Tracks N-ary plot events (actions, betrayals, item transfers), causal provenance pointers (`caused_by_episode_id`), and POV witness tracking (`"direct"` vs. `"told"`) to eliminate metagaming and state-dict bloat.
- **First-Class Causal Time Travel**: Atomic `< 1ms` rollback (`rollback_to(step)`) and zero-copy branching (`branch(at_step, name)`) via an append-only SQLite lineage tree.
- **Single-File Portability**: The entire multiverse of timelines, graph revisions, episodic events, and vector embeddings lives inside a **single `.sqlite` file** (or `:memory:`).

---

## Quickstart

### Installation

```bash
pip install narrative-memory
```

*(Dependencies: `pydantic>=2.0`, `numpy>=1.26`, `rapidfuzz>=3.0`, and standard library `sqlite3`)*

---

### Basic Usage

```python
from narrative_memory import (
    NarrativeMemoryEngine,
    EntityRecord,
    StepDelta,
    EntityDelta,
    EdgeDelta,
    EpisodeRecord,
    VoiceExemplar,
)

# 1. Initialize Engine (file path or in-memory)
engine = NarrativeMemoryEngine("story_save.sqlite")

# 2. Tier 1: Seed Root Entities & Aliases (Step 0)
engine.register_entity(
    EntityRecord(
        entity_id="char_arthur",
        canonical_name="Arthur",
        base_attributes={"title": "King", "gender": "M"},
    ),
    step=0,
)
engine.register_entity(
    EntityRecord(
        entity_id="char_merlin",
        canonical_name="Merlin",
        base_attributes={"title": "Court Wizard", "gender": "M"},
    ),
    aliases=["the old wizard"],
    step=0,
)

# 3. Tier 2 & 3: Commit Scene State Shifts
engine.commit_step(
    step=10,
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

# Index an empirical voice exemplar into Tier 5 Voice Bank
engine.index_voice_exemplars(
    step=15,
    exemplars=[
        VoiceExemplar(
            speaker_id="char_arthur",
            text="I beseech your counsel, venerable wizard, for the kingdom stands upon the precipice.",
            metadata={"register": "formal", "tone": "respectful"},
            quality_score=9.5,
        )
    ],
)
```

---

### Step 82 Argument & Step 85 Retrieval (With Tier 6 Episodic Ledger)

```python
# Step 82: Argument occurs -> Arthur's mood shifts, stance changes, and an episode is recorded
engine.commit_step(
    step=82,
    delta=StepDelta(
        entities=[
            EntityDelta(
                entity_id="char_arthur",
                state={"mood": "furious"},
                reason="Discovered Merlin concealed the prophecy",
            )
        ],
        edges=[
            EdgeDelta(
                source_id="char_arthur",
                target_id="char_merlin",
                state={"register": "informal", "tone": "furious"},
                caused_by_episode_id="ep_step82_prophecy_dispute",
            )
        ],
        episodes=[
            EpisodeRecord(
                episode_id="ep_step82_prophecy_dispute",
                predicate="argument",
                summary="Arthur confronted Merlin over the concealed dragon prophecy in the council chamber.",
                participants={"char_arthur": "actor", "char_merlin": "target"},
                salience=8.5,
            )
        ],
    ),
)

# Step 85: Unified Tier 1-6 Retrieval Pass
context = engine.retrieve_context(
    step=85,
    text='Arthur glared at the old wizard. "Speak truth now!"',
    mentions=["Arthur", "the old wizard"],
    top_k_voice=3,
    top_k_episodes=5,
)

# 1. 1-Hop Coreference: "the old wizard" -> char_merlin
assert any(m.entity_id == "char_merlin" for m in context.resolved_mentions)

# 2. Temporal Edge: Arthur -> Merlin is informal & furious with causal link
edge = context.active_edges[0]
print(edge.state)  # {'register': 'informal', 'tone': 'furious'}
print(edge.caused_by_episode_id)  # 'ep_step82_prophecy_dispute'

# 3. Tier 6 Episodic Memory: Pinpointed relevant narrative episode
assert context.episodes[0].episode_id == "ep_step82_prophecy_dispute"

# 4. Formatted Prompt for LLM Injection
print(engine.format_context(context))
```

---

### Time Travel: Branching & Rollback

```python
# Create an alternate timeline branch forked at Step 50
engine.branch(at_step=50, name="reconciliation_route")

# Switch to the new branch
engine.checkout("reconciliation_route")

# The branch inherits state and episodes up to Step 50, but ignores Step 82 on 'main'!
branch_context = engine.retrieve_context(step=85, mentions=["Arthur", "Merlin"])
assert branch_context.active_edges[0].state["register"] == "formal"

# Rewind 'main' back to Step 10 in < 1ms (revisions and episodes at > 10 atomically deleted)
engine.rollback_to(step=10, timeline_id="main")
```

---

## Transactional Step Execution (`step_transaction`)

For interactive game loops, visual novels, and automated pipelines, use the built-in context manager:

```python
with engine.step_transaction(step=85, text=scene_text) as tx:
    # 1. Access point-in-time state (guaranteed clean, no forward leakage)
    ctx = tx.context
    prompt = engine.format_context(ctx)

    # 2. Call your LLM
    response = call_llm(prompt)

    # 3. Extract deltas, episodes & voice lines, then stage for atomic commit
    tx.commit(
        delta=extracted_delta,
        exemplars=candidate_lines,
        episodes=extracted_episodes,
    )
# Automatically commits on exit or rolls back if an exception occurs
```

---

## CLI Timeline Inspector

Inspect branching timelines, world state, diffs, and entity trajectories directly from `.sqlite` saves:

```bash
narrative-memory tree story_save.sqlite
narrative-memory inspect story_save.sqlite --timeline main --step 85 --pov char_arthur
narrative-memory diff story_save.sqlite main reconciliation_route --step 85
narrative-memory history story_save.sqlite char_arthur --timeline main
```

---

## Plug-and-Play Adapters, Sessions & Lifecycle Hooks

- **Zero-dependency Embedders** (`narrative_memory.embedders`): `OfflineNgramEmbedder` (with CJK tokenization & signed feature hashing), `CallableEmbedder`, `OpenAICompatibleEmbedder`, and `OllamaEmbedder` with `embed_batch()` support.
- **Zero-dependency Extractors** (`narrative_memory.extractors`): `CallableExtractor` and `OpenAICompatibleExtractor` with JSON-fence stripping and schema validation retries.
- **Tier 6 Episodic Policy & Consolidation** (`narrative_memory.episodic`): Tunable `EpisodicScoringPolicy`, `DefaultEpisodeRanker`, flashback-safe `StepDelta(is_flashback=True)`, `"forgotten"` / `"suppressed"` epistemic updates, and `engine.consolidate_episodes(...)`.
- **Concurrency-Safe Sessions & Hooks** (`narrative_memory.hooks`): Use `with NarrativeMemoryEngine("save.sqlite") as engine:`, `engine.session("branch_a")`, `with engine.use_timeline("branch_a"):`, and `BaseEngineHooks` (`on_before_commit`, `on_after_commit`, `on_rollback`, `on_branch`, `on_gate_rejected`).

---

## Documentation

- [**Comprehensive Tutorial & Study Book**](docs/tutorial-and-study-book.md): Masterclass guide covering ontology, motivation, recursive CTE lineage, anti-drift gates, episodic recall, and end-to-end examples.
- [**Architecture Guide**](docs/architecture.md): Deep dive into the 6-tier state superset and SQLite schema.
- [**Causal Time Travel**](docs/time-travel.md): Lineage trees, copy-on-write branching, atomic rollback, and save-slot exports.
- [**Voice Bank & Anti-Drift**](docs/voice-bank.md): The 2-stage ingestion gate, n-gram vector retrieval, and hard-register cascade.
- [**Domain Presets**](docs/presets.md): Pre-built prompt formatters for Interactive Fiction, Localization, and Multi-Agent Simulations.
- [**API Reference**](docs/api-reference.md): Detailed reference for classes, methods, and Pydantic schemas.

---

## License

MIT License. Free and open source.
