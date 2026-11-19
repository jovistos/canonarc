<p align="center">
  <img src="docs/assets/banner.jpg" alt="canonarc — Temporal Graph, Episodic Ledger, Voice Bank & Causal Time-Travel Engine for LLMs" width="100%" />
</p>

<h1 align="center"><code>canonarc</code></h1>

<p align="center">
  <strong>Tier 1–6 Temporal Graph, Episodic Ledger, Voice Bank & Causal Time-Travel Engine for LLMs</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/storage-SQLite%20Single--File-003B57?style=flat-square&logo=sqlite" alt="SQLite Single-File" />
  <img src="https://img.shields.io/badge/architecture-Tier%201--6%20Causal%20Graph-6366f1?style=flat-square" alt="Tier 1-6 Causal Graph" />
  <img src="https://img.shields.io/badge/typing-PEP%20561%20py.typed-3776AB?style=flat-square" alt="PEP 561 Typed" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" /></a>
</p>

<p align="center">
  <a href="docs/tutorial-and-study-book.md"><strong>Study Book & Deep Tutorial</strong></a> &bull;
  <a href="docs/architecture.md">Architecture</a> &bull;
  <a href="docs/epistemics-and-pov.md">Epistemics & POV</a> &bull;
  <a href="docs/time-travel.md">Time Travel</a> &bull;
  <a href="docs/voice-bank.md">Voice Bank</a> &bull;
  <a href="docs/presets.md">Presets</a> &bull;
  <a href="docs/benchmarks.md">Benchmarks</a> &bull;
  <a href="examples/README.md">Examples</a> &bull;
  <a href="docs/api-reference.md">API Reference</a>
</p>

---

## What is `canonarc`?

Most LLM memory frameworks (`mem0`, `zep`, `langmem`) are designed for **conversational chatbots** operating in continuous wall-clock time. They fall short when building **Interactive Fiction, Visual Novels (Ren'Py / Ink), Multi-Agent Simulations, Creative Writing Co-Pilots, and Sequential Localization Pipelines**, where time moves in discrete **causal steps** (turns, scenes, ticks, or chunks) and narrative state must be **rewindable, branchable, and spoiler-free**.

`canonarc` gives your application:
- **Tiers 1–4 Temporal Property Graph**: Tracks immutable base attributes (gender, declension, archetypes), asymmetric directed relationships (`A -> B != B -> A`), 1-hop coreference aliases (`"the old wizard"` -> `char_merlin`), and rolling state deltas with trajectory history (`prior_state`, `revision_count`).
- **Tier 5 Gated Dynamic Voice Bank**: Anchors dialogue style and syntax using historical in-character lines without "error compounding" or self-conditioning drift.
- **Tier 6 Episodic Event Ledger & Epistemic Visibility**: Tracks N-ary plot events (actions, betrayals, item transfers), causal provenance pointers (`caused_by_episode_id`), and POV witness tracking (`"direct"`, `"told"`, `"deceived"`, `"forgotten"`, `"suppressed"`) to eliminate metagaming and state-dict bloat.
- **First-Class Causal Time Travel**: Atomic `< 1ms` rollback (`rollback_to(step)`) and zero-copy branching (`branch(at_step, name)`) via an append-only SQLite lineage tree.
- **Single-File Portability**: The entire multiverse of timelines, graph revisions, episodic events, and vector embeddings lives inside a **single `.sqlite` file** (or `:memory:`).

### How `canonarc` Compares

| Capability | Chatbot Memory (`mem0`, `Zep`, `LangMem`) | Storyworld / GraphRAG (`Graphiti`, `ComoRAG`) | `canonarc` |
| :--- | :--- | :--- | :--- |
| **Time Model** | Continuous wall-clock timestamps | Wall-clock or linear chapter order | **Discrete causal steps (`step: int`)** + flashback-safe `story_time` |
| **Branching & Rollback** | None (linear history) | None or heavy subgraph copies | **Zero-copy SQLite lineage tree** + **`< 1ms` atomic rollback** |
| **Character POV & Secrets** | Omniscient queries | Omniscient world graph | **Per-POV epistemic projection** (`direct`, `told`, `deceived`, disguise masking) |
| **Dialogue Voice Control** | None | Unfiltered prose retrieval (prone to drift) | **Tier 5 Voice Bank** (2-stage anti-drift gate + dyadic register filtering) |
| **State + Event Provenance** | Flat facts or key-value dicts | Nodes/edges or text chunks | **Tiers 1–4 State Graph** linked to **Tier 6 Episodes** via `caused_by_episode_id` |
| **Storage & Dependencies** | Cloud APIs / Vector DB + Graph DB | Neo4j / Kuzu / FalkorDB + Vector DB | **Single `.sqlite` file** (`pydantic`, `numpy`, `rapidfuzz` only) |

---

## Quickstart

### Installation

```bash
pip install canonarc
```

*(Dependencies: `pydantic>=2.0`, `numpy>=1.26`, `rapidfuzz>=3.0`, and standard library `sqlite3`)*

---

### 1. Seed Entities, Aliases, Edges & Voice Anchors

```python
from canonarc import (
    CanonArcEngine,
    EntityRecord,
    StepDelta,
    EntityDelta,
    EdgeDelta,
    EpisodeRecord,
    VoiceExemplar,
)

# Initialize Engine (file path or ":memory:")
engine = CanonArcEngine("story_save.sqlite")

# Tier 1: Seed Root Entities & Aliases (Step 0)
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
engine.register_entity(
    EntityRecord(
        entity_id="char_mordred",
        canonical_name="Mordred",
        base_attributes={"title": "Knight", "gender": "M"},
    ),
    step=0,
)

# Tier 2 & 3: Commit Initial Relationship State (Step 10)
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

# Tier 5: Index an Empirical Voice Exemplar
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

### 2. Commit Narrative Events & Retrieve Point-in-Time Context

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
                reason="Concealed the dragon prophecy",
            )
        ],
        episodes=[
            EpisodeRecord(
                episode_id="ep_step82_prophecy_dispute",
                predicate="argument",
                summary="Arthur confronted Merlin over the concealed dragon prophecy in the council chamber.",
                participants={"char_arthur": "actor", "char_merlin": "target"},
                witnesses=["char_arthur", "char_merlin"],
                salience=8.5,
            )
        ],
    ),
)

# Step 85: Unified Tier 1-6 Retrieval Pass (Arthur, Merlin & Mordred in scene)
context = engine.retrieve_context(
    step=85,
    text='Arthur glared at the old wizard while Mordred watched. "Speak truth now!"',
    mentions=["Arthur", "the old wizard", "Mordred"],
    top_k_voice=3,
    top_k_episodes=5,
)

# 1. 1-Hop Coreference: "the old wizard" -> char_merlin
assert any(m.entity_id == "char_merlin" for m in context.resolved_mentions)

# 2. Temporal Edge: Arthur -> Merlin is informal & furious with causal link
edge = context.active_edges[0]
assert edge.state == {"register": "informal", "tone": "furious"}
assert edge.caused_by_episode_id == "ep_step82_prophecy_dispute"

# 3. Formatted Prompt Block Ready for LLM Injection
print(engine.format_context(context))
```

**Rendered output of `engine.format_context(context)`:**

```markdown
### Active Scene Narrative Context (Step: 85)

#### Characters Present & State:
- **Arthur** (char_arthur): gender: M, title: King | mood: furious (because: Discovered Merlin concealed the prophecy)
- **Merlin** (char_merlin): gender: M, title: Court Wizard
- **Mordred** (char_mordred): gender: M, title: Knight

#### Interpersonal Stances & Register:
- char_arthur → char_merlin: register=informal, tone=furious (because: Concealed the dragon prophecy)

#### Episodic Memory (Relevant Past Events):
- [Step 82 | argument | known to: char_arthur, char_merlin | ignorant: char_mordred] Arthur confronted Merlin over the concealed dragon prophecy in the council chamber.

#### Voice Bank (Historical Dialogue Anchors):
- char_arthur [formal, respectful]: "I beseech your counsel, venerable wizard, for the kingdom stands upon the precipice."
```

---

### 3. Anti-Metagaming & POV Epistemic Projection (`pov_entity_ids`)

Project the entire 6-tier world state through a specific character's eyes so NPCs never act on secrets they didn't witness (see [Epistemic State, Disguises & POV Projection](docs/epistemics-and-pov.md)):

```python
# Query Step 85 strictly from Mordred's POV (who was NOT in the council chamber at Step 82)
mordred_pov = engine.retrieve_context(
    step=85,
    text="Mordred observed Arthur and Merlin.",
    mentions=["Arthur", "Merlin", "Mordred"],
    pov_entity_ids=["char_mordred"],
)

# Mordred sees Arthur is furious, but the secret council episode and causal reason are redacted!
assert mordred_pov.episodes == []
arthur_view = next(e for e in mordred_pov.entities if e.entity_id == "char_arthur")
assert arthur_view.state["mood"] == "furious"
assert arthur_view.reason is None

# Feed Mordred a deceptive cover story at Step 86
engine.inform_entity(
    step=86,
    episode_id="ep_step82_prophecy_dispute",
    entity_id="char_mordred",
    epistemic="deceived",
    believed_summary="Arthur reprimanded Merlin over a missing supply ledger.",
)

# Now Mordred recalls the cover story (and vector search embeds believed_summary, never the secret!)
mordred_deceived = engine.retrieve_context(
    step=87,
    text="Mordred asked about the dispute.",
    mentions=["Mordred", "Arthur"],
    pov_entity_ids=["char_mordred"],
)
assert mordred_deceived.episodes[0].summary == "Arthur reprimanded Merlin over a missing supply ledger."
```

---

### 4. Causal Time Travel: Zero-Copy Branching & `< 1ms` Rollback

```python
# Fork an alternate timeline branch at Step 50 (zero data copying)
engine.branch(at_step=50, name="reconciliation_route")

# Switch to the new branch
engine.checkout("reconciliation_route")

# The branch inherits state and episodes up to Step 50, but ignores Step 82 on 'main'!
branch_context = engine.retrieve_context(step=85, mentions=["Arthur", "Merlin"])
assert branch_context.active_edges[0].state["register"] == "formal"

# Compare divergent timelines at Step 85
diff = engine.diff_timelines("main", "reconciliation_route", at_step=85)

# Rewind 'main' back to Step 10 in < 1ms (revisions and episodes at > 10 atomically deleted)
engine.rollback_to(step=10, timeline_id="main")
```

---

## Transactional Step Execution (`step_transaction`)

For interactive game loops, visual novels, and automated LLM pipelines, use the built-in transactional context manager:

```python
from canonarc import CanonArcEngine, OpenAICompatibleExtractor

engine = CanonArcEngine(
    "story_save.sqlite",
    extractor=OpenAICompatibleExtractor.from_gemini(),  # or .from_lmstudio(), .from_deepseek()
)

with engine.step_transaction(step=85, text=scene_text, pov_entity_ids=["char_arthur"]) as tx:
    # 1. Access point-in-time state (guaranteed clean on reruns, no forward spoiler leakage)
    prompt = engine.format_context(tx.context)

    # 2. Generate next scene or dialogue with your LLM
    generated_scene = call_llm(prompt)

    # 3. Automatically extract StepDelta JSON + Gate 1/2 voice exemplars & stage atomic commit
    tx.extract_and_commit(generated_scene)
    # Or commit deterministic game-engine deltas directly:
    # tx.commit(delta=my_step_delta, exemplars=candidate_lines)
```

---

## Interactive Example Notebooks

Self-contained, offline-runnable Jupyter notebooks in [`examples/`](examples/README.md) walk through each primary domain:

| Notebook | Domain | Preset | Key Capabilities Demonstrated |
| :--- | :--- | :--- | :--- |
| [`01_interactive_fiction_and_rpg.ipynb`](examples/01_interactive_fiction_and_rpg.ipynb) | Interactive Fiction, Visual Novels & RPGs | `InteractiveFictionPreset` | Tiers 1–6, 1-hop alias coreference, 2-stage anti-drift `IngestionGate`, `prior_state` trajectory, dramatic irony (`\| ignorant:`), `branch()`, `diff_timelines()`, `rollback_to()`, `export_snapshot()` |
| [`02_literary_and_game_localization.ipynb`](examples/02_literary_and_game_localization.ipynb) | Literary & Video Game Localization | `TranslationPreset` | Tier 1 locked grammar/exonyms (`target_name`, `gender`, `declension`), asymmetric dyadic T–V register (`Vi` vs. `ti`), multi-party voice anchoring, flashback-safe ingestion (`is_flashback=True`) |
| [`03_multi_agent_simulation.ipynb`](examples/03_multi_agent_simulation.ipynb) | Multi-Agent Autonomous Simulations | `SimulationPreset` | Custom `SchemaPolicy` & `EpisodicScoringPolicy`, per-tick directed affinity edges, partial observability (`pov_entity_ids`), secondhand gossip (`InformDelta`), `consolidate_episodes()` |
| [`04_mystery_and_social_deduction.ipynb`](examples/04_mystery_and_social_deduction.ipynb) | Mystery, Espionage & Social Deduction | `InteractiveFictionPreset` | Disguise & persona masking (`AliasRecord.known_to` → `masked:<id>`), `facade` vs. hidden `stance` edges, `private_state` redaction, `deceived` beliefs (`believed_summary` vector isolation), disguise piercing |

```bash
pip install -e ".[notebooks]"
jupyter lab examples/
```

---

## CLI Timeline Inspector

Inspect branching timelines, world state, diffs, and entity trajectories directly from `.sqlite` saves (pass `--json` to any subcommand for machine-readable output):

```bash
canonarc tree story_save.sqlite
canonarc inspect story_save.sqlite --timeline main --step 85 --pov char_arthur
canonarc diff story_save.sqlite main reconciliation_route --step 85
canonarc history story_save.sqlite char_arthur --timeline main
```

---

## Plug-and-Play Adapters, Sessions & Lifecycle Hooks

- **Zero-dependency Embedders** (`canonarc.embedders`): `OfflineNgramEmbedder` (with CJK tokenization & signed feature hashing), `CallableEmbedder`, `OpenAICompatibleEmbedder`, and `OllamaEmbedder` with `embed_batch()` and automatic embedder-mismatch detection (`on_embedder_mismatch="warn" | "raise" | "reindex" | "ignore"`).
- **Zero-dependency Extractors** (`canonarc.extractors`): `CallableExtractor`, `OpenAICompatibleExtractor` (with `.from_lmstudio()`, `.from_gemini()`, `.from_deepseek()`), and content-addressed `CachedDeltaExtractor` with `TokenUsage` cost estimation.
- **Tier 6 Episodic Policy & Consolidation** (`canonarc.episodic`): Tunable `EpisodicScoringPolicy`, `DefaultEpisodeRanker`, flashback-safe `StepDelta(is_flashback=True)`, `"forgotten"` / `"suppressed"` epistemic updates, and `engine.consolidate_episodes(...)`.
- **Concurrency-Safe Sessions & Hooks** (`canonarc.hooks`): Use `with CanonArcEngine("save.sqlite") as engine:`, `engine.session("branch_a")`, `with engine.use_timeline("branch_a"):`, and `BaseEngineHooks` (`on_before_commit`, `on_after_commit`, `on_rollback`, `on_branch`, `on_gate_rejected`).

---

## Documentation

- [**Comprehensive Tutorial & Study Book**](docs/tutorial-and-study-book.md): Masterclass guide covering ontology, motivation, recursive CTE lineage, anti-drift gates, episodic recall, and end-to-end examples.
- [**Architecture Guide**](docs/architecture.md): Deep dive into the 6-tier state superset, SQLite schema, and windowed recursive CTE.
- [**Epistemic State, Disguises & POV Projection**](docs/epistemics-and-pov.md): Guide to anti-metagaming, `"direct"` / `"told"` / `"deceived"` / `"forgotten"` modes, scoped disguises (`known_to`), and `facade` vs. `stance` edges.
- [**Causal Time Travel**](docs/time-travel.md): Lineage trees, copy-on-write branching, atomic rollback, non-linear flashbacks, and save-slot exports.
- [**Voice Bank & Anti-Drift**](docs/voice-bank.md): The 2-stage ingestion gate, multi-party dyadic register resolution, and embedder migration.
- [**Domain Presets**](docs/presets.md): Pre-built prompt formatters and `SchemaPolicy` configuration for Interactive Fiction, Localization, and Multi-Agent Simulations.
- [**Domains of Application**](docs/domains-of-application.md): Primary and emerging use cases, domain-to-capability matrix, and when to choose another tool.
- [**Synthetic Story Extraction Benchmarks**](docs/benchmarks.md): Multi-model extraction accuracy, epistemic leak metrics, token costs, and reproduction guide across the 3K-token synthetic story suite.
- [**API Reference**](docs/api-reference.md): Detailed reference for classes, methods, extractors, embedders, CLI commands, and Pydantic schemas.

---

## Development & Contributing

Contributions, bug reports, and new domain presets or adapters are welcome! To set up a local development environment:

```bash
git clone https://github.com/canonarc/canonarc.git
cd canonarc
pip install -e ".[dev]"

# Run linter, strict type checker, and test suite
ruff check src tests
mypy src
pytest
```

---

## License

[MIT License](LICENSE). Free and open source.
