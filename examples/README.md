# `canonarc` Example Notebooks

Self-contained Jupyter notebooks demonstrating how `canonarc` works across its core domains of application—each with explanations of **why** the architecture fits that domain, **how** the 6-tier engine operates under the hood, and live **automated LLM `StepDelta` extraction**.

The notebooks pair **cloud DeepSeek API** and **local Qwen via LM Studio** across different domains to showcase `canonarc`'s backend diversity, using the zero-dependency `OfflineNgramEmbedder` and an in-memory SQLite store (`":memory:"`).

## Notebooks

| Notebook | Domain | Preset | LLM Backend | Key Features Demonstrated |
| :--- | :--- | :--- | :--- | :--- |
| [01_interactive_fiction_and_rpg.ipynb](01_interactive_fiction_and_rpg.ipynb) | Interactive Fiction, Visual Novels & Narrative RPGs | `InteractiveFictionPreset` | **DeepSeek API (`deepseek-flash`)** (`from_deepseek` + `CachedDeltaExtractor`) | Stateless vs. `canonarc` POV A/B metagaming comparison, 6-tier RPG mapping & Mermaid turn loop, non-character entities (`location`/`item`) & `possesses`/`background_edges`, `BaseEngineHooks` telemetry, 2-stage `IngestionGate` (`gate1`/`gate2`) + prose harvesting + fallback note, non-linear flashback safety (`is_flashback`), full epistemic spectrum (`direct`/`deceived`/`told`/`forgotten` + `pov_mode="all"` + HTML matrix), multi-route `branch()`/`session()`/`use_timeline()` + inline SVG route & trust chart, `diff_timelines()` HTML diff, `export_snapshot()` reload verification, timed `< 1ms` `rollback_to()`, `consolidate_episodes()`, closed-loop `play_rpg_turn()` with `CachedDeltaExtractor`, and 14-point automated QA audit |
| [02_literary_translation.ipynb](02_literary_translation.ipynb) | Long-Form Literary Translation Pipelines | `TranslationPreset` | **DeepSeek API** (`from_deepseek` + DeepSeek translator) | Pure-English A/B comparison (stateless vs. `canonarc`), 6-Tier translation mental model, transactional `step_transaction()` pipeline, verified `Vi`→`ti` register propagation, automated Tier 5 Voice Bank harvesting & `IngestionGate` filtering, flashback safety, non-Serbian linguistic gloss table, and automated localization QA audit |
| [03_multi_agent_simulation.ipynb](03_multi_agent_simulation.ipynb) | Multi-Agent Autonomous Simulations & Counterfactuals | `SimulationPreset` | **DeepSeek API (`deepseek-flash`)** (`from_deepseek` + `CachedDeltaExtractor`) | Omniscient vs. POV-isolated A/B comparison, closed-loop tick runner, custom `SchemaPolicy` & `EpisodicScoringPolicy`, `BaseEngineHooks` telemetry, callsign aliases, `private_state` & `facade` edges, full epistemic spectrum (`direct`/`told`/`deceived`/`forgotten`/`pov_mode="all"`), parallel `session()`/`use_timeline()` counterfactuals, `rollback_to()`, `export_snapshot()`, `consolidate_episodes()`, inline SVG affinity & trajectory charts, cached `deepseek-flash` extraction & QA audit |
| [04_mystery_and_social_deduction.ipynb](04_mystery_and_social_deduction.ipynb) | Mystery, Espionage, TTRPG & Social Deduction | `InteractiveFictionPreset` | **DeepSeek API (`deepseek-flash`)** (`from_deepseek` + `CachedDeltaExtractor`) | Omniscient vs. POV-isolated live A/B metagaming test, 6-tier Mermaid epistemic pipeline & ground-truth vs. perceived graph, 4 cleanly separated boundaries (scoped `disguise` → `masked:<id>`, unmasked `facade` vs. `stance`, stale `belief_state` override, causal provenance `reason` redaction), all 5 `EpistemicMode` states (`direct`/`told`/`deceived`/`forgotten`/`suppressed`) + Deceived-POV vector isolation proof + HTML Epistemic Matrix, multi-POV `pov_mode="any"` vs. `"all"` + dramatic irony, persona/facade Tier 5 `VoiceBank`, 3-round social deduction ballots + inline SVG trust/suspicion chart, `is_flashback` safety, `branch()`/`session()`/`diff_timelines()` + inline SVG branch tree + `< 1ms` `rollback_to()`, `EpistemicAuditHook`, cached DeepSeek extraction & 14-point automated QA scorecard |

## Running the Notebooks

Install the package (editable or from PyPI) and launch Jupyter:

```bash
pip install -e .
jupyter lab examples/
```

### Configuring LLM Backends

To execute the live LLM translation and extraction cells:

- **DeepSeek API (Notebooks 01–04):** Set `DEEPSEEK_API_KEY` (and optionally `DEEPSEEK_MODEL=deepseek-flash`) in `.env` at the repository root (see [`.env.example`](../.env.example)).
- **Local Qwen via LM Studio (optional):** Start LM Studio's local server on `http://localhost:1234/v1` (or set `LMSTUDIO_BASE_URL`) and swap `from_deepseek()` to `OpenAICompatibleExtractor.from_lmstudio()` (which auto-discovers the loaded model via `/api/v0/models`).
- **Swapping backends:** All extractors share the `DeltaExtractor` protocol, so switching any notebook between `from_deepseek()`, `from_lmstudio()`, and `from_gemini()` is a one-line change.
