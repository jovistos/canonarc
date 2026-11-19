# `canonarc` Example Notebooks

Self-contained, offline-runnable Jupyter notebooks demonstrating how `canonarc` works across its core domains of application—each with explanations of **why** the architecture fits that domain, **how** the 6-tier engine operates under the hood, and a final **automated LLM `StepDelta` extraction** step.

All notebooks run cleanly offline out of the box using the zero-dependency `OfflineNgramEmbedder` and an in-memory SQLite store (`":memory:"`). The final cell in each notebook demonstrates live LLM extraction via `tx.extract_and_commit(...)`—deliberately pairing **cloud DeepSeek API** and **local Qwen via LM Studio** across different domains to showcase `canonarc`'s backend diversity—and gracefully falls back to a prompt preview when credentials or a local server are unavailable.

## Notebooks

| Notebook | Domain | Preset | LLM Extractor | Key Features Demonstrated |
| :--- | :--- | :--- | :--- | :--- |
| [01_interactive_fiction_and_rpg.ipynb](01_interactive_fiction_and_rpg.ipynb) | Interactive Fiction, Visual Novels & Narrative RPGs | `InteractiveFictionPreset` | **DeepSeek API** (`from_deepseek`) | Tiers 1–6, 1-hop alias coreference, 2-stage anti-drift `IngestionGate`, `prior_state` trajectory, dramatic irony (`\| ignorant:`), `branch()`, `diff_timelines()`, `rollback_to()`, `export_snapshot()`, live DeepSeek `StepDelta` extraction |
| [02_literary_and_game_localization.ipynb](02_literary_and_game_localization.ipynb) | Literary & Video Game Localization Pipelines | `TranslationPreset` | **Local Qwen via LM Studio** (`from_lmstudio`) | Tier 1 locked grammar/exonyms (`target_name`, `gender`, `declension`), asymmetric dyadic T–V register (`Vi` vs. `ti`), multi-party voice anchoring, dynamic register shifts, flashback-safe ingestion (`is_flashback=True`), privacy-first local Qwen extraction |
| [03_multi_agent_simulation.ipynb](03_multi_agent_simulation.ipynb) | Multi-Agent Autonomous Simulations & Counterfactuals | `SimulationPreset` | **Local Qwen via LM Studio** (`from_lmstudio` + `CachedDeltaExtractor`) | Custom `SchemaPolicy` & `EpisodicScoringPolicy`, per-tick directed affinity edges, partial observability (`pov_entity_ids`), secondhand gossip (`InformDelta`), counterfactual `branch()` + `diff_timelines()`, `consolidate_episodes()`, `$0.00` local tick extraction with disk caching |
| [04_mystery_and_social_deduction.ipynb](04_mystery_and_social_deduction.ipynb) | Mystery, Espionage, TTRPG & Social Deduction | `InteractiveFictionPreset` | **DeepSeek API** (`from_deepseek`) | Disguise & persona masking (`AliasRecord.known_to` → `masked:<id>`), `facade` vs. hidden `stance` edges, `private_state` redaction, `deceived` beliefs (`believed_summary` vector isolation), disguise piercing & truth reveals, epistemic deception extraction |

## Running the Notebooks

Install the package (editable or from PyPI) and launch Jupyter:

```bash
pip install -e .
jupyter lab examples/
```

### Optional: Running the Live LLM Extraction Cells

Every notebook runs end-to-end without network access, automatically printing a prompt preview if a backend is not configured. To execute the live extraction cells:

- **DeepSeek API (Notebooks 01 & 04):** Set `DEEPSEEK_API_KEY` (and optionally `DEEPSEEK_MODEL=deepseek-flash`) in `.env` at the repository root (see [`.env.example`](../.env.example)).
- **Local Qwen via LM Studio (Notebooks 02 & 03):** Start LM Studio's local server on `http://localhost:1234/v1` (or set `LMSTUDIO_BASE_URL`) with a Qwen model loaded (e.g., `qwen/qwen3.8-27b` or `qwen/qwen3.5-9b`). `OpenAICompatibleExtractor.from_lmstudio()` auto-discovers the loaded model via `/api/v0/models`.
- **Swapping backends:** All extractors share the `DeltaExtractor` protocol, so switching any notebook between `from_deepseek()`, `from_lmstudio()`, and `from_gemini()` is a one-line change.
