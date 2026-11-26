# Tier 5 Voice Bank & Anti-Drift Architecture

## The Problem: Error Compounding & Generation Drift

In sequential LLM generation, feeding past outputs back into the prompt without filtering creates a dangerous feedback loop:
1. In Chapter 1, the LLM generates a slightly awkward or generic dialogue line.
2. In Chapter 4, this awkward line is retrieved as a few-shot exemplar.
3. The LLM mimics the awkward phrasing, reinforcing the error until character voice collapses.

---

## The Solution: Two-Stage Anti-Drift Gate

`canonarc` guards the Voice Bank behind two strict gates:

```
Raw Output Prose
       │
       ▼
[Gate 1: Pre-LLM Deterministic Gate]
  - Extract quoted & dash dialogue across typographic styles:
    "...", “...”, „...“ / „...” (Serbian/Croatian/Polish/German),
    «...» / »...« (guillemets), 「...」 / 『...』 (CJK), and — / – / ― dash lines
  - Strip inline narrator interjections from dash dialogue (e.g. "– Ne znam – reče Artur – šta da radim.")
    and route stripped beats into speaker attribution windows
  - Word count: 12 <= words <= 60
  - Lexical distinctiveness: Root-TTR (|V| / sqrt(N)) >= threshold
  - Long word density (>= 6 chars)
       │
   (Passes)
       ▼
CandidateDialogueSpans
       │
       ▼
[Gate 2: Post-Polish / Quality Gate]
  - Candidate line undergoes post-polish or quality scoring
  - Quality score >= 8.0 (on 1.0 - 10.0 scale)
       │
   (Passes)
       ▼
Indexed into Tier 5 Voice Bank
```

Rejected Gate 1 / Gate 2 candidates can be observed for diagnostics via `IngestionGate.extract_candidates(..., on_rejected=callback)` or the `on_gate_rejected` hook in `BaseEngineHooks`.

---

## Automated Gate-2 Harvesting (`StepTransaction.harvest_voice`)

Inside `engine.step_transaction(...)`, call `tx.harvest_voice(...)` to run the entire Gate-1 → dyadic tagging → Gate-2 scoring → atomic staging pipeline in one call:

```python
with engine.step_transaction(step=12, text=source_chunk, lang="sr") as tx:
    prompt = engine.format_context(tx.context)
    translated = translate(prompt, source_chunk)

    tx.extract_and_commit(source_chunk)
    harvested = tx.harvest_voice(
        translated,
        source_text=source_chunk,
        lang="sr",  # resolves Serbian inflected names/aliases around „…“ and —
    )
# On exit: StepDelta, episodes, passing Gate-2 VoiceExemplars, and StepCheckpoint commit atomically.
```

What `tx.harvest_voice()` does automatically:
1. **Tier-5 Policy Short-Circuit**: No-op returning `[]` when Tier 5 is disabled (e.g., `TierProfile.LEXICON_ONLY` or `STATE_ONLY`).
2. **Gate-1 Extraction with Staged-Delta Visibility**: Extracts dialogue spans (`"..."`, `“…”`, `„…“` / `„…”`, `«…»` / `»…«`, `「…」`, `—` / `–` / `―` dash lines with inline narrator-beat stripping) and resolves `(speaker_id, addressee_id)` with quote-boundary-isolated attribution windows, including any brand-new entities/aliases introduced in `tx._staged_delta` during the same step.
3. **Graph-Aware Dyadic Tagging**: Looks up the active `(speaker_id → addressee_id)` edge across `SchemaPolicy.edge_layer_order` (`stance` → `facade`) **overlaid with any intra-step `StepDelta` edge/entity mutations** (so a `Vi` → `ti` shift or new `disguise` in the current scene is reflected immediately). In two-character scenes where Gate 1 only resolves the speaker, `infer_addressee=True` (default) infers the co-present addressee automatically.
4. **Gate-2 Quality Scoring (`VoiceQualityScorer`)**: Scores candidates on `[0.0, 10.0]` via the configured scorer (defaults to `HeuristicVoiceQualityScorer`), fires `on_gate_rejected("gate2", ...)` for rejected lines, and stages passing `VoiceExemplar`s for atomic commit (or pass `stage=False` for dry-run inspection).

### Pluggable `VoiceQualityScorer` Implementations
- **`HeuristicVoiceQualityScorer`** *(default, zero-LLM)*: Deterministic scorer combining normalized Gate-1 distinctiveness, lexical diversity, length sweet spot, attribution/metadata completeness, dialogue expressiveness, and template/markup artifact penalties.
- **`CallableVoiceQualityScorer(fn)`**: Wraps any 1-, 2-, or 3-argument Python callable `(candidate[, metadata[, context]]) -> float | VoiceScoreResult`.
- **`LLMVoiceQualityScorer(extractor, *, batch_size=15, fallback_scorer=None)`**: Optional batched LLM judge backed by any `JsonCompletionClient` (`OpenAICompatibleExtractor`, `CallableExtractor`, `CachedDeltaExtractor`, or custom client). Evaluates all Gate-1 candidates from a scene in a single JSON call (verifying character voice, register fidelity, and optional `source_text` alignment), with optional `metadata_updates`, fallback scoring, and `last_usage`/`total_usage` tracking.

Configure a default scorer on `CanonArcEngine(..., voice_scorer=...)` or pass `scorer=...` per call to `tx.harvest_voice(...)`.

---

## Voice Bank Retrieval: Policy-Driven Hard-Filter / Soft-Boost Cascade

When retrieving exemplars for a character or agent, the Voice Bank applies a multi-priority ranking cascade driven by `SchemaPolicy` (defaulting to `register` hard-filter and `tone` soft-boost):

1. **Causal Filter**: Only exemplars visible along the lineage with `exemplar.step <= cutoff_step`.
2. **Hard Attribute Filter** (`policy.hard_filter_keys`, default `("register",)`):
   - If the active directed edge specifies a value for a hard-filter key (e.g. `register`: `formal` vs. `informal`, `ti` vs. `Vi`; or `channel`: `slack` vs. `email`), exemplars with a conflicting value are **excluded**.
   - **Sparse Fallback**: If strict filtering leaves zero candidates and `policy.fallback_on_empty_hard_filter=True`, the ranker falls back to mismatched candidates with a configurable penalty (`policy.hard_filter_mismatch_penalty`) rather than returning an empty voice bank.
3. **Composite Scoring**:
   For candidate exemplars:
   $$\text{Score} = \text{CosineSim}(q, e) + \sum_{k \in \text{soft\_boost\_weights}} w_k \cdot \mathbb{I}(\text{match}_k) + w_{\text{addr}} \cdot \mathbb{I}(\text{addressee match}) + w_{\text{qual}} \cdot \left(\frac{\text{quality}}{10}\right) - \text{penalty}$$

---

## Multi-Party Dyadic Voice Resolution

In scenes with multiple co-present characters (e.g., Arthur in a room with both Merlin in `ti`/`furious` and the Queen in `Vi`/`reverent`), collapsing a speaker's outgoing edges to a single arbitrary target would lock the speaker into one register and hard-filter out all other co-present registers.

`CanonArcEngine.retrieve_context` and `VoiceBank.rank_exemplars_for_targets` resolve voice targets **per dyad** instead:

1. **In-Scene Target Prioritization**: Outgoing edges directed to co-present characters in the scene (`in_scene_ids`) are prioritized over off-screen 1-hop neighbors. An active `facade` toward an absent character never overrides an in-scene `stance`.
2. **Per-Pair Layer Merging (`base < stance < facade`)**: Edges are grouped by `(speaker_id, target_id)` and merged from lowest to highest priority (`kinship`/base $\to$ `stance` $\to$ `facade`).
   - Split attributes (e.g., `register` on a `kinship` edge and `tone` on a `stance` edge) combine cleanly for that pair.
   - A `facade` edge overrides outward `register`/`tone` **only for that specific addressee**, without contaminating the speaker's `stance` toward other characters in the room.
   - Masked speakers (`masked:<id>`) are restricted to outward `facade` edges only so hidden `stance` state is never leaked.
3. **Deterministic Dyad Ordering**: Co-present targets are ordered by first mention position in `text`, then edge `valid_from_step` recency, then canonical `target_id`.
4. **Round-Robin Exemplar Allocation**: When multiple dyads are active, `VoiceBank.rank_exemplars_for_targets` scores each `(addressee_id, register, tone)` target independently, tags returned copies with `metadata["for_addressee"]`, and interleaves them via Round-Robin (deduplicated by `exemplar_id`) up to `top_k_voice`. Single-target scenes use `rank_exemplars` directly.

---

## Embedders & Adapters (`canonarc.embedders`)

By default, `canonarc` ships with `OfflineNgramEmbedder`:
- **Zero external dependencies**, 128-dimensional L2-normalized `float32` vectors.
- **Unicode & CJK-aware tokenization** (`_tokenize_units`): Tokenizes Latin/Cyrillic alphanumeric words as well as CJK ideographs, Hiragana/Katakana, and Hangul syllables + adjacent character bigrams so non-space-delimited languages work out of the box.
- **Weinberger signed CRC32 feature hashing**: Uses `zlib.crc32` for both bucket index and a `±1` sign bit to cancel hash-collision bias cleanly across Python processes.
- **Batch Embedding**: Implements `embed_batch(texts)` (with `embed_texts_batch()` helper) so committing batches of voice exemplars or episodes uses a single vectorized call when backed by remote/GPU embedders.
- **Embedder provenance & mismatch detection**: `CausalStore` records `embedder_id` and `embedding_dim` in `store_metadata` on first write (and infers `embedding_dim` for legacy DBs). When reopening a database with a different embedder or dimension, `CanonArcEngine(..., on_embedder_mismatch=...)` supports:
  - `"warn"` *(default)*: Emits a `UserWarning` advising migration.
  - `"raise"`: Raises `EmbedderMismatchError` immediately (catches both dimension mismatches and same-dimension model swaps).
  - `"reindex"`: Automatically runs `engine.reindex_embeddings()` in batches on startup.
  - `"ignore"`: Suppresses the startup check (`VoiceBank` and `DefaultEpisodeRanker` still fall back to on-the-fly re-embedding on shape mismatch).
- **Batched migration**: Call `engine.reindex_embeddings(batch_size=128)` at any time to re-embed all stored episodes and voice exemplars in batches and update `store_metadata`.

### Built-in Adapters
- `CallableEmbedder(fn, batch_fn=..., embedder_id=..., dim=...)`: Wrap any custom Python function or local Sentence-Transformers model.
- `OpenAICompatibleEmbedder(model=..., base_url=..., api_key=...)`: Zero-dependency `urllib.request` client for OpenAI, LM Studio, vLLM, or OpenRouter `/v1/embeddings` endpoints.
- `OllamaEmbedder(model=..., base_url=...)`: Local Ollama `/api/embed` client with batch support.
- Or implement the `Embedder` protocol directly (`embed_text(str) -> np.ndarray`, optionally `embed_batch(Sequence[str]) -> list[np.ndarray]`).
