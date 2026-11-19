# Model Extraction Benchmark: The Aethelgard Chronicle (18-Chunk Multi-Chapter 12K Marathon) (`aethelgard_chronicle_12k`)

- **Guidance Mode**: `none`
- **Total Story Chunks**: `18`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | SKIPPED (Cached) | **72.7%** | 63.2% | 63.6% | 63.1% | 91.4% | 82.8% | 40.7% | 89.7% | 81.8% | 1 | 33.3% | 33 | PASS | 0 (0%) | 1 | 145,351 | 128.3s | $0.02036 |
| `qwen/qwen3.8-27b` | RAN | **75.8%** | 72.2% | 72.7% | 72.2% | 89.6% | 97.0% | 61.1% | 96.4% | 90.9% | 1 | 0.0% | 30 | PASS | 0 (0%) | 0 | 129,620 | 237.5s | $0.00000 |
| `qwen/qwen3.5-9b` | SKIPPED (Cached) | **65.2%** | 69.9% | 45.5% | 71.2% | 88.4% | 78.0% | 60.2% | 72.5% | 72.7% | 2 | 0.0% | 24 | PASS | 0 (0%) | 0 | 126,784 | 144.0s | $0.00000 |
| `rule-based-non-llm` | SKIPPED (Cached) | **52.6%** | 38.3% | 45.5% | 37.9% | 28.6% | 20.0% | 53.7% | 87.6% | 90.9% | 8 | 0.0% | 82 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
