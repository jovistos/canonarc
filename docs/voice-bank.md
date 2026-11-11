# Tier 5 Voice Bank & Anti-Drift Architecture

## The Problem: Error Compounding & Generation Drift

In sequential LLM generation, feeding past outputs back into the prompt without filtering creates a dangerous feedback loop:
1. In Chapter 1, the LLM generates a slightly awkward or generic dialogue line.
2. In Chapter 4, this awkward line is retrieved as a few-shot exemplar.
3. The LLM mimics the awkward phrasing, reinforcing the error until character voice collapses.

---

## The Solution: Two-Stage Anti-Drift Gate

`narrative-memory` guards the Voice Bank behind two strict gates:

```
Raw Output Prose
       │
       ▼
[Gate 1: Pre-LLM Deterministic Gate]
  - Extract quoted dialogue across styles ("...", “...”, „...“, «...», — ...)
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

---

## Voice Bank Retrieval: Hard-Register / Soft-Tone Cascade

When retrieving exemplars for a character, the Voice Bank applies a multi-priority ranking cascade:

1. **Causal Filter**: Only exemplars visible along the lineage with `exemplar.step < current_step`.
2. **Hard Register Filter**:
   - If the active directed edge specifies a grammatical register (e.g. `formal` vs. `informal`, `ti` vs. `Vi`, `tu` vs. `vous`), exemplars with a conflicting register are **strictly excluded**.
   - *Why:* Injecting an informal exemplar into a formal dialogue scene causes grammatical hallucinations.
3. **Composite Scoring**:
   For candidate exemplars in the valid register:
   $$\text{Score} = \text{CosineSim}(q, e) + 0.15 \cdot \mathbb{I}(\text{tone match}) + 0.10 \cdot \mathbb{I}(\text{addressee match}) + 0.05 \cdot \left(\frac{\text{quality}}{10}\right)$$

---

## Offline Hybrid N-Gram Embedder

By default, `narrative-memory` ships with `OfflineNgramEmbedder`:
- Zero external dependencies.
- 128-dimensional float32 normalized vectors.
- Hashes word tokens and character 3-grams to measure lexical and stylistic similarity.
- Pluggable: Pass any custom embedding class satisfying the `Embedder` protocol (`embed_text(str) -> np.ndarray`) to use OpenAI, LM Studio, or Sentence-Transformers.
