<p align="center">
  <img src="https://raw.githubusercontent.com/jovistos/canonarc/main/docs/assets/banner.jpg" alt="canonarc — Branchable, spoiler-free narrative memory and world-state engine for LLMs" width="100%" />
</p>

<h1 align="center"><code>canonarc</code></h1>

<p align="center">
  <strong>Branchable, spoiler-free narrative memory &amp; world-state engine for LLM games, interactive fiction, localization, and multi-agent simulations — in a single SQLite file.</strong>
</p>

<p align="center">
  <a href="https://github.com/jovistos/canonarc/actions/workflows/ci.yml"><img src="https://github.com/jovistos/canonarc/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/typing-PEP%20561%20py.typed-3776AB?style=flat-square" alt="PEP 561 Typed" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" /></a>
</p>

<p align="center">
  <a href="docs/handbook.md"><strong>Handbook</strong></a> &bull;
  <a href="docs/architecture.md">Architecture</a> &bull;
  <a href="docs/epistemics-and-pov.md">Epistemics &amp; POV</a> &bull;
  <a href="examples/README.md">Examples</a> &bull;
  <a href="docs/benchmarks.md">Benchmarks</a> &bull;
  <a href="docs/api-reference.md">API Reference</a>
</p>

---

## Why `canonarc`?

Most LLM memory frameworks (`mem0`, `Zep`, `LangMem`) are built for **conversational chatbots** running in continuous wall-clock time. They break down in **Interactive Fiction, Visual Novels (Ren'Py / Ink), Multi-Agent Simulations, Creative Writing Co-Pilots, and Sequential Localization Pipelines**, where time advances in discrete **causal steps** (turns, scenes, ticks, or chunks) and world state must be **rewindable, branchable, character-scoped, and spoiler-free**.

### The 6-Tier World-State Model

`canonarc` unifies narrative state, dialogue voice, and episodic memory inside a single append-only SQLite database (or `:memory:`):

| Tier | Layer | What It Tracks |
| :---: | :--- | :--- |
| **Tier 1** | **Base Entity Attributes** | Immutable character/item metadata seeded at step 0 (gender, declension, archetypes, exonyms). |
| **Tier 2** | **Asymmetric Dyadic Edges** | Directed relationships (`A → B ≠ B → A`) such as register, tone, trust, or hidden `facade` vs. `stance`. |
| **Tier 3** | **Rolling State & Trajectories** | Point-in-time entity/edge state deltas with windowed history (`prior_state`, `prior_step`, `revision_count`). |
| **Tier 4** | **Temporal Versioning & 1-Hop Coreference** | Strict `step < current_step` resolution, alias linking (`"the old wizard"` → `char_merlin`), and co-present vs. background separation. |
| **Tier 5** | **Gated Voice Bank** | Empirical historical dialogue anchors filtered by speaker and active register/tone, protected by a 2-stage anti-drift gate. |
| **Tier 6** | **Episodic Ledger & Epistemics** | N-ary plot events with causal provenance (`caused_by_episode_id`) and per-character knowledge (`direct`, `told`, `deceived`, `forgotten`, `suppressed`). |

### How It Fits in Your Loop

```text
  Scene Text / Tick
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  step_transaction(step, text, pov_entity_ids=[...])      │
  │   1. Retrieve Tiers 1–6 strictly at (step < current)     │
  │   2. Project through POV lens (redact secrets/disguises) │
  └──────────────────────────┬───────────────────────────────┘
                             │  formatted prompt block
                             ▼
                     Your LLM / Game Loop
                             │  generated scene / delta
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Extract StepDelta + 2-Stage Anti-Drift Voice Gate       │
  │  Atomic commit to single-file SQLite lineage tree        │
  └──────────────────────────────────────────────────────────┘
```

---

## How `canonarc` Compares

| Capability | Chatbot Memory (`mem0`, `Zep`, `LangMem`) | Storyworld / GraphRAG (`Graphiti`, `ComoRAG`) | `canonarc` |
| :--- | :--- | :--- | :--- |
| **Time Model** | Continuous wall-clock timestamps | Wall-clock or linear chapter order | **Discrete causal steps (`step: int`)** + flashback-safe `story_time` |
| **Branching & Rollback** | Linear history only | None or heavy subgraph copies | **Zero-copy SQLite lineage tree** (`0.09 ms` branch, `0.07 ms` undo) |
| **Character POV & Secrets** | Omniscient queries | Omniscient world graph | **Per-POV epistemic projection** (`direct`, `told`, `deceived`, disguise masking) |
| **Dialogue Voice Control** | None | Unfiltered prose retrieval (prone to drift) | **Tier 5 Voice Bank** (2-stage anti-drift gate + dyadic register filtering) |
| **State + Event Provenance** | Flat facts or key-value dicts | Nodes/edges or text chunks | **Tiers 1–4 State Graph** linked to **Tier 6 Episodes** via `caused_by_episode_id` |
| **Storage & Dependencies** | Cloud APIs / Vector DB + Graph DB | Neo4j / Kuzu / FalkorDB + Vector DB | **Single `.sqlite` file** (`pydantic`, `numpy`, `rapidfuzz` only) |

### When to Choose Something Else
`canonarc` is intentionally scoped to storyworlds, simulations, and discrete-step pipelines (see [Domains of Application](docs/domains-of-application.md)). Consider another tool if you need:
- **Continuous wall-clock enterprise chatbots or CRM logs**: Systems built around real-world validity timestamps (`t_valid` / `t_invalid`) like `Zep` or `Graphiti` are a better fit.
- **Million-document static corpus RAG or multi-hop analytical QA**: `canonarc` uses in-memory NumPy scoring over storyworld-scale histories and 1-hop graph expansion; dedicated vector stores or full graph databases (e.g., Neo4j) suit massive static corpora.
- **High-concurrency multi-writer distributed backends**: Single-file SQLite excels for local saves, game loops, and per-session sandboxes, not horizontally sharded multi-tenant writes.

---

## Quickstart

### Installation

```bash
pip install canonarc
```

*(Requires Python 3.11+. Core dependencies: `pydantic>=2.0`, `numpy>=1.26`, `rapidfuzz>=3.0`, and stdlib `sqlite3`. To install from source: `pip install git+https://github.com/jovistos/canonarc.git`.)*

### 1. The Transactional LLM Loop (`step_transaction`)

For interactive game loops, visual novels, and automated LLM pipelines, `step_transaction()` handles spoiler-free retrieval, LLM extraction, anti-drift voice gating, and atomic commits in one block:

```python
from canonarc import CanonArcEngine, OpenAICompatibleExtractor

engine = CanonArcEngine(
    "story_save.sqlite",
    extractor=OpenAICompatibleExtractor.from_gemini(),  # or .from_lmstudio(), .from_deepseek()
)

with engine.step_transaction(step=85, text=scene_text, pov_entity_ids=["char_arthur"]) as tx:
    # 1. Point-in-time context (clean on reruns, zero forward spoiler leakage)
    prompt = engine.format_context(tx.context)

    # 2. Generate the next scene or dialogue with your LLM
    generated_scene = call_llm(prompt)

    # 3. Extract StepDelta JSON + Gate 1/2 voice exemplars & commit atomically
    tx.extract_and_commit(generated_scene)
    # Or commit deterministic game-engine deltas directly:
    # tx.commit(delta=my_step_delta, exemplars=candidate_lines)
```

### 2. Explicit World-State Walkthrough (No LLM Required)

You can also drive `canonarc` deterministically from a game engine or script:

```python
from canonarc import (
    CanonArcEngine, EntityRecord, StepDelta,
    EntityDelta, EdgeDelta, EpisodeRecord, VoiceExemplar,
)

engine = CanonArcEngine("story_save.sqlite")  # or ":memory:"

# Seed entities & 1-hop alias (Step 0)
engine.register_entity(
    EntityRecord(entity_id="char_arthur", canonical_name="Arthur", base_attributes={"title": "King", "gender": "M"}),
    step=0,
)
engine.register_entity(
    EntityRecord(entity_id="char_merlin", canonical_name="Merlin", base_attributes={"title": "Court Wizard", "gender": "M"}),
    aliases=["the old wizard"],
    step=0,
)
engine.register_entity(
    EntityRecord(entity_id="char_mordred", canonical_name="Mordred", base_attributes={"title": "Knight", "gender": "M"}),
    step=0,
)

# Seed initial relationship (Step 10) & Voice Bank anchors (Step 15)
engine.commit_step(
    step=10,
    delta=StepDelta(edges=[
        EdgeDelta(source_id="char_arthur", target_id="char_merlin", state={"register": "formal", "tone": "respectful"})
    ]),
)
engine.index_voice_exemplars(
    step=15,
    exemplars=[
        VoiceExemplar(
            speaker_id="char_arthur", step=15,
            text="I beseech your counsel, venerable wizard, for the kingdom stands upon the precipice.",
            metadata={"register": "formal", "tone": "respectful"}, quality_score=9.5,
        ),
        VoiceExemplar(
            speaker_id="char_arthur", step=15,
            text="Enough riddles, old man — answer me plainly or leave this hall!",
            metadata={"register": "informal", "tone": "furious"}, quality_score=9.2,
        ),
    ],
)

# Step 82: Argument occurs -> mood, stance, and causal episode committed together
engine.commit_step(
    step=82,
    delta=StepDelta(
        entities=[EntityDelta(
            entity_id="char_arthur", state={"mood": "furious"},
            caused_by_episode_id="ep_step82_prophecy_dispute",
            reason="Discovered Merlin concealed the prophecy",
        )],
        edges=[EdgeDelta(
            source_id="char_arthur", target_id="char_merlin",
            state={"register": "informal", "tone": "furious"},
            caused_by_episode_id="ep_step82_prophecy_dispute",
            reason="Concealed the dragon prophecy",
        )],
        episodes=[EpisodeRecord(
            episode_id="ep_step82_prophecy_dispute",
            predicate="argument",
            summary="Arthur confronted Merlin over the concealed dragon prophecy in the council chamber.",
            participants={"char_arthur": "actor", "char_merlin": "target"},
            witnesses=["char_arthur", "char_merlin"],
            salience=8.5,
        )],
    ),
)

# Step 85: Unified Tier 1–6 retrieval ("the old wizard" resolves to char_merlin)
context = engine.retrieve_context(
    step=85,
    text='Arthur glared at the old wizard while Mordred watched. "Speak truth now!"',
    mentions=["Arthur", "the old wizard", "Mordred"],
    top_k_voice=3,
    top_k_episodes=5,
)
print(engine.format_context(context))
```

**Rendered output of `engine.format_context(context)`:**

```markdown
### Active Scene Narrative Context (Step: 85)

#### Characters Present & State:
- **Arthur** (char_arthur): mood: furious [reason: Discovered Merlin concealed the prophecy]
- **Merlin** (char_merlin): mood: neutral
- **Mordred** (char_mordred): mood: neutral

#### Interpersonal Stances & Register:
- char_arthur → char_merlin: register=informal, tone=furious, trust=0.0 (because: Concealed the dragon prophecy)

#### Episodic Memory (Relevant Past Events):
- [Step 82 | argument | known to: char_arthur, char_merlin | ignorant: char_mordred] Arthur confronted Merlin over the concealed dragon prophecy in the council chamber.

#### Voice Bank (Historical Dialogue Anchors):
- char_arthur [informal, furious]: "Enough riddles, old man — answer me plainly or leave this hall!"
```

---

## Core Capabilities

### Anti-Metagaming, POV Projection & Deception

Project the entire 6-tier world state through a character's eyes so NPCs never act on secrets they didn't witness (see [Epistemic State, Disguises & POV Projection](docs/epistemics-and-pov.md)):

```python
# Query Step 85 strictly from Mordred's POV (absent at Step 82)
mordred_pov = engine.retrieve_context(
    step=85,
    text="Mordred observed Arthur and Merlin.",
    mentions=["Arthur", "Merlin", "Mordred"],
    pov_entity_ids=["char_mordred"],
)
# Mordred sees Arthur is furious, but the secret episode & causal reason are redacted:
# mordred_pov.episodes == []
# arthur.reason is None

# Feed Mordred a deceptive cover story at Step 86
engine.inform_entity(
    step=86,
    episode_id="ep_step82_prophecy_dispute",
    entity_id="char_mordred",
    epistemic="deceived",
    believed_summary="Arthur reprimanded Merlin over a missing supply ledger.",
)

# Mordred recalls the cover story — and vector search embeds believed_summary, never the secret!
mordred_deceived = engine.retrieve_context(
    step=87,
    text="Mordred asked about the dispute.",
    mentions=["Mordred", "Arthur"],
    pov_entity_ids=["char_mordred"],
)
print(mordred_deceived.episodes[0].summary)
# -> "Arthur reprimanded Merlin over a missing supply ledger."
```

### Zero-Copy Branching & Fast Rollback

```python
# Fork an alternate timeline at Step 50 (zero data copying)
engine.branch(at_step=50, name="reconciliation_route")
engine.checkout("reconciliation_route")

# Inherits state up to Step 50, ignoring Step 82 on 'main'
branch_ctx = engine.retrieve_context(step=85, mentions=["Arthur", "Merlin"])
# branch_ctx.active_edges[0].state["register"] == "formal"

# Compare divergent timelines at Step 85
diff = engine.diff_timelines("main", "reconciliation_route", at_step=85)

# Atomically rewind 'main' back to Step 10 (revisions & episodes > 10 deleted)
engine.rollback_to(step=10, timeline_id="main")
```

### CLI Timeline Inspector

Inspect branching timelines, POV world state, diffs, and entity trajectories directly from `.sqlite` saves (pass `--json` to any subcommand for machine-readable output):

```bash
$ canonarc tree story_save.sqlite
main [head=86]
└─ reconciliation_route (fork@50) [head=50]

$ canonarc diff story_save.sqlite main reconciliation_route --step 85
Diff main vs reconciliation_route @ step 85:
  Changed (2):
    - char_arthur: {... 'mood': 'furious', 'reason': 'Discovered Merlin concealed the prophecy'} vs {... 'state': {}}
    - char_arthur:stance->char_merlin: {'register': 'informal', 'tone': 'furious', ...} vs {'register': 'formal', 'tone': 'respectful'}
  Identical keys: 3

# Additional commands:
canonarc inspect story_save.sqlite --timeline main --step 85 --pov char_arthur
canonarc history story_save.sqlite char_arthur --timeline main
```

---

## Performance & Benchmarks at a Glance

Engine algorithmic scaling micro-benchmarks from [`scripts/benchmark_engine_performance.py`](scripts/benchmark_engine_performance.py) (see [Benchmarks](docs/benchmarks.md) for full LLM extraction & long-horizon marathon tables):

| Metric (1,500 Steps, 120 Characters) | Result | Notes |
| :--- | :---: | :--- |
| **Zero-Copy `branch()`** | **`0.087 ms`** | Constant-time lineage fork |
| **1-Step Undo `rollback_to()`** | **`0.067 ms`** | `4.48 ms` for 50% deep rollback (-750 steps) |
| **`retrieve_context()` P50** | **`26.9 ms`** | **`21.3 ms`** after `consolidate_episodes()` |
| **SQLite Save File Size** | **`2.03 MB`** | **`436 KB`** (-78.5%) post-consolidation |
| **Deceived-POV Vector Isolation** | **`0%` leakage** | `100%` Recall@1 on `believed_summary`, adversarial secret queries blocked |

---

## Interactive Example Notebooks

Self-contained Jupyter notebooks in [`examples/`](examples/README.md) walk through each primary domain:

| Notebook | Domain | Preset | Key Capabilities Demonstrated |
| :--- | :--- | :--- | :--- |
| [`01_interactive_fiction_and_rpg.ipynb`](examples/01_interactive_fiction_and_rpg.ipynb) | Interactive Fiction, Visual Novels & RPGs | `InteractiveFictionPreset` | Tiers 1–6, 1-hop alias coreference, 2-stage anti-drift `IngestionGate`, `prior_state` trajectory, dramatic irony (`\| ignorant:`), `branch()`, `diff_timelines()`, `rollback_to()`, `export_snapshot()` |
| [`02_literary_translation.ipynb`](examples/02_literary_translation.ipynb) | Long-Form Literary Translation | `TranslationPreset` | Pure-English A/B comparison, transactional `step_transaction()` pipeline, verified `Vi`→`ti` register propagation, automated Tier 5 Voice Bank harvesting & anti-drift gating, flashback safety, linguistic glosses & QA audit |
| [`03_multi_agent_simulation.ipynb`](examples/03_multi_agent_simulation.ipynb) | Multi-Agent Autonomous Simulations | `SimulationPreset` | Custom `SchemaPolicy` & `EpisodicScoringPolicy`, per-tick directed affinity edges, partial observability (`pov_entity_ids`), secondhand gossip (`InformDelta`), `consolidate_episodes()` |
| [`04_mystery_and_social_deduction.ipynb`](examples/04_mystery_and_social_deduction.ipynb) | Mystery, Espionage & Social Deduction | `InteractiveFictionPreset` | Disguise & persona masking (`AliasRecord.known_to` → `masked:<id>`), `facade` vs. hidden `stance` edges, `private_state` redaction, `deceived` beliefs (`believed_summary` vector isolation), disguise piercing |

```bash
pip install -e ".[notebooks]"
jupyter lab examples/
```

---

## Built-In Adapters, Sessions & Lifecycle Hooks

- **Embedders (no extra SDKs required)** (`canonarc.embedders`): `OfflineNgramEmbedder` (with CJK tokenization & signed feature hashing), `CallableEmbedder`, `OpenAICompatibleEmbedder`, and `OllamaEmbedder` with `embed_batch()` and automatic embedder-mismatch detection (`on_embedder_mismatch="warn" | "raise" | "reindex" | "ignore"`).
- **Extractors (no extra SDKs required)** (`canonarc.extractors`): `CallableExtractor`, `OpenAICompatibleExtractor` (with `.from_lmstudio()`, `.from_gemini()`, `.from_deepseek()`), and content-addressed `CachedDeltaExtractor` with `TokenUsage` cost estimation.
- **Tier 6 Episodic Policy & Consolidation** (`canonarc.episodic`): Tunable `EpisodicScoringPolicy`, `DefaultEpisodeRanker`, flashback-safe `StepDelta(is_flashback=True)`, `"forgotten"` / `"suppressed"` epistemic updates, and `engine.consolidate_episodes(...)`.
- **Concurrency-Safe Sessions & Hooks** (`canonarc.hooks`): Use `with CanonArcEngine("save.sqlite") as engine:`, `engine.session("branch_a")`, `with engine.use_timeline("branch_a"):`, and `BaseEngineHooks` (`on_before_commit`, `on_after_commit`, `on_rollback`, `on_branch`, `on_gate_rejected`).

---

## Documentation

- [**The `canonarc` Handbook**](docs/handbook.md): End-to-end tutorial & architectural book covering ontology, recursive CTE lineage, anti-drift gates, epistemic POV projection, live LLM extraction, and worked examples.
- [**Architecture Guide**](docs/architecture.md): Deep dive into the 6-tier state superset, SQLite schema, and windowed recursive CTE.
- [**Epistemic State, Disguises & POV Projection**](docs/epistemics-and-pov.md): Guide to anti-metagaming, `"direct"` / `"told"` / `"deceived"` / `"forgotten"` modes, scoped disguises (`known_to`), and `facade` vs. `stance` edges.
- [**Causal Time Travel**](docs/time-travel.md): Lineage trees, copy-on-write branching, atomic rollback, non-linear flashbacks, and save-slot exports.
- [**Voice Bank & Anti-Drift**](docs/voice-bank.md): The 2-stage ingestion gate, multi-party dyadic register resolution, and embedder migration.
- [**Domain Presets**](docs/presets.md): Pre-built prompt formatters and `SchemaPolicy` configuration for Interactive Fiction, Localization, and Multi-Agent Simulations.
- [**Domains of Application**](docs/domains-of-application.md): Primary and emerging use cases, domain-to-capability matrix, and when to choose another tool.
- [**Synthetic Story & Engine Benchmarks**](docs/benchmarks.md): Multi-model extraction accuracy, epistemic leak metrics, engine scaling micro-benchmarks, and reproduction guide.
- [**API Reference**](docs/api-reference.md): Detailed reference for classes, methods, extractors, embedders, CLI commands, and Pydantic schemas.

---

## Development & Contributing

Contributions, bug reports, and new domain presets or adapters are welcome:

```bash
git clone https://github.com/jovistos/canonarc.git
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
