# Domains of Application

`canonarc` is designed for applications where **time moves in discrete causal steps** (turns, scenes, ticks, or chunks), **world state must be rewindable and branchable**, and **who knows what—and how characters speak to each other—matters**.

This guide outlines the primary domains with built-in support, adjacent domains enabled by `SchemaPolicy` and custom presets, and scenarios where a different memory architecture is a better fit.

---

## 1. Primary Domains (Built-In Presets)

These domains are directly supported by built-in [Domain Presets](presets.md) and core engine features.

### Interactive Fiction, Visual Novels & Narrative RPGs
- **Preset:** `InteractiveFictionPreset`
- **Typical targets:** Ren'Py, Ink, Twine, text adventures, and LLM-driven RPGs.
- **Why `canonarc` fits:**
  - **Zero-copy branching & `< 1ms` rollback:** `branch()` and `rollback_to()` support save-scumming, alternate story routes, and 60 FPS game loops (see [Causal Time Travel](time-travel.md)).
  - **Portable save slots:** The entire multiverse of timelines, graph revisions, episodic events, and embeddings lives in a single `.sqlite` file, with standalone snapshots via `export_snapshot()`.
  - **Anti-metagaming:** Per-character POV projection (`pov_entity_ids`) ensures NPCs only act on events they witnessed or were told about.

### Literary & Video Game Localization / Translation Pipelines
- **Preset:** `TranslationPreset`
- **Typical targets:** Sequential novel translation, JRPG/visual novel localization, and dubbing or subtitle pipelines.
- **Real-world example:** [blebel-ai-book-translator](https://github.com/jovistos/blebel-ai-book-translator) — book translation app powered by `canonarc`.
- **Why `canonarc` fits:**
  - **Grammar & exonym locking (Tier 1):** Stores immutable base attributes such as grammatical gender, declension class, and target-language exonyms.
  - **Asymmetric dyadic register (Tier 2):** Models directed relationships (`A -> B != B -> A`) for T–V distinctions (e.g., *ti* vs. *Vi*), Japanese/Korean honorifics, and politeness shifts across a story.
  - **Anti-drift voice anchoring (Tier 5):** The gated [Voice Bank](voice-bank.md) preserves each character's diction and syntax without compounding translation drift.

### Multi-Agent Autonomous Simulations & Social Dynamics
- **Preset:** `SimulationPreset`
- **Typical targets:** Generative agent towns, economic/diplomatic simulations, faction politics, and social-behavior research.
- **Why `canonarc` fits:**
  - **Per-tick state & directed affinity:** Tracks internal agent states, directed trust/affinity edges, and N-ary shared events (`item_transfer`, `alliance`, `betrayal`).
  - **Partial observability:** Models `direct`, `told`, `deceived`, `forgotten`, and `suppressed` knowledge per agent.
  - **Counterfactual experiments:** Fork a simulation at any tick with `branch()` to run "what-if" interventions and compare divergent outcomes with `diff_timelines()`.

### Creative Writing & Long-Form Narrative Co-Pilots
- **Preset:** `InteractiveFictionPreset` (or custom `DomainPreset`)
- **Typical targets:** Novel drafting assistants, TV writers' room continuity trackers, and serialized fiction tools.
- **Why `canonarc` fits:**
  - **Non-linear storytelling:** Supports `story_time`, `chapter_id`, `scene_id`, and flashback-safe ingestion (`StepDelta(is_flashback=True)`).
  - **Continuity & dramatic irony:** 1-hop alias coreference (`"the old wizard"` → `char_merlin`), windowed emotional trajectories (`prior_state`), causal provenance (`caused_by_episode_id`), and `| ignorant:` annotations keep long-running narratives coherent.

---

## 2. Adjacent & Emerging Domains

Because the 6-tier model is domain-agnostic and customizable via `SchemaPolicy` and custom `DomainPreset` implementations (see [Domain Presets](presets.md)), it also fits well in:

### Tabletop RPG (TTRPG) & AI Game Master Assistants
- Tracks campaign lore, faction intrigue, NPC disguises (`AliasRecord.known_to`), hidden agendas (`private_state` and `facade` edges), and strict separation between player knowledge and GM ground truth.

### Mystery, Espionage & Social Deduction Games
- Fits detective games, *Werewolf/Mafia*-style LLM benchmarks, and hidden-role scenarios where false beliefs (`deceived` epistemics with `believed_summary` vector isolation), information asymmetry, and unpierced disguises are core mechanics.

### Branching Roleplay & Training Simulators
- Supports clinical, legal, crisis-negotiation, or customer-support roleplay where trainees rewind a scenario to test a different strategy and instructors inspect or diff the resulting decision trees.

### Counterfactual Planning & Tree-of-Thought Agent Search
- Serves as a lightweight, isolated state substrate for agentic planners exploring multiple hypothetical action branches in discrete steps without cloning a database per branch.

---

## 3. Domain-to-Capability Matrix

| Domain | Recommended Preset | Key `canonarc` Capabilities Used |
| :--- | :--- | :--- |
| Interactive Fiction / Visual Novels / RPGs | `InteractiveFictionPreset` | `branch()`, `rollback_to()`, `export_snapshot()`, `pov_entity_ids`, Tier 5 Voice Bank |
| Literary & Game Localization | `TranslationPreset` | Tier 1 base grammar/exonyms, Tier 2 asymmetric register, Tier 5 gated Voice Bank |
| Multi-Agent Simulations | `SimulationPreset` | Per-tick deltas, directed affinity edges, Tier 6 epistemics, `diff_timelines()` |
| Creative Writing Co-Pilots | `InteractiveFictionPreset` | Flashbacks (`is_flashback`), alias coreference, `prior_state`, `caused_by_episode_id` |
| TTRPG / AI Game Master | `InteractiveFictionPreset` | Disguise masking (`known_to`), `private_state`, `facade` edges, POV projection |
| Mystery & Social Deduction | Custom / `SimulationPreset` | `deceived` epistemics (`believed_summary`), dramatic irony tags, POV isolation |
| Branching Training Simulators | Custom / `InteractiveFictionPreset` | `rollback_to()`, `branch()`, `diff_timelines()`, CLI `tree`/`inspect` |
| Counterfactual Agent Search | Custom / `SimulationPreset` | Zero-copy SQLite lineage tree, atomic `step_transaction()` |

---

## 4. When to Choose a Different Tool

`canonarc` is intentionally scoped to storyworlds, simulations, and discrete-step pipelines (see [Architecture & Storage Philosophy](architecture.md)). It is generally **not** the right fit for:

- **Continuous wall-clock enterprise chatbots or CRM logs:** Systems built around continuous real-world timestamps (`t_valid` / `t_invalid`) and linear conversation logs (such as Graphiti or Zep) are a better match.
- **Million-document static corpus RAG or deep multi-hop analytical QA:** `canonarc` uses in-memory NumPy scoring over storyworld-scale histories and 1-hop graph expansion; large static corpora or arbitrary multi-hop Cypher/PageRank queries are better served by dedicated vector databases or full graph databases (e.g., Neo4j / HippoRAG).
