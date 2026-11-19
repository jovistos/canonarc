# `canonarc` Example Notebooks

Self-contained, offline-runnable Jupyter notebooks demonstrating how `canonarc` works across its core domains of application—each with explanations of **why** the architecture fits that domain and **how** the 6-tier engine operates under the hood.

All notebooks run offline out of the box using the zero-dependency `OfflineNgramEmbedder` and an in-memory SQLite store (`":memory:"`).

## Notebooks

| Notebook | Domain | Preset | Key Features Demonstrated |
| :--- | :--- | :--- | :--- |
| [01_interactive_fiction_and_rpg.ipynb](01_interactive_fiction_and_rpg.ipynb) | Interactive Fiction, Visual Novels & Narrative RPGs | `InteractiveFictionPreset` | Tiers 1–6, 1-hop alias coreference, 2-stage anti-drift `IngestionGate`, `prior_state` trajectory, dramatic irony (`\| ignorant:`), `branch()`, `diff_timelines()`, `rollback_to()`, `export_snapshot()` |
| [02_literary_and_game_localization.ipynb](02_literary_and_game_localization.ipynb) | Literary & Video Game Localization Pipelines | `TranslationPreset` | Tier 1 locked grammar/exonyms (`target_name`, `gender`, `declension`), asymmetric dyadic T–V register (`Vi` vs. `ti`), multi-party voice anchoring, dynamic register shifts, flashback-safe ingestion (`is_flashback=True`) |
| [03_multi_agent_simulation.ipynb](03_multi_agent_simulation.ipynb) | Multi-Agent Autonomous Simulations & Counterfactuals | `SimulationPreset` | Custom `SchemaPolicy` & `EpisodicScoringPolicy`, per-tick directed affinity edges, partial observability (`pov_entity_ids`), secondhand gossip (`InformDelta`), counterfactual `branch()` + `diff_timelines()`, `consolidate_episodes()` |
| [04_mystery_and_social_deduction.ipynb](04_mystery_and_social_deduction.ipynb) | Mystery, Espionage, TTRPG & Social Deduction | `InteractiveFictionPreset` | Disguise & persona masking (`AliasRecord.known_to` → `masked:<id>`), `facade` vs. hidden `stance` edges, `private_state` redaction, `deceived` beliefs (`believed_summary` vector isolation), disguise piercing & truth reveals |

## Running the Notebooks

Install the package (editable or from PyPI) and launch Jupyter:

```bash
pip install -e .
jupyter lab examples/
```
