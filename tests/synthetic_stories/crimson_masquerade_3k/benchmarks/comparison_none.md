# Model Extraction Benchmark: The Crimson Masquerade (Extended 3K Edition) (`crimson_masquerade_3k`)

- **Guidance Mode**: `none`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | SKIPPED (Cached) | **90.6%** | 86.0% | 100.0% | 83.3% | 100.0% | 100.0% | 70.8% | 100.0% | 100.0% | 0 | 85.7% | 1 | PASS | 0 (0%) | 0 | 43,772 | 48.2s | $0.00605 |
| `qwen/qwen3.8-27b` | RAN | **79.4%** | 94.0% | 75.0% | 97.6% | 100.0% | 100.0% | 95.8% | 93.3% | 100.0% | 2 | 33.3% | 5 | PASS | 0 (0%) | 0 | 39,468 | 79.6s | $0.00000 |
| `qwen/qwen3.5-9b` | SKIPPED (Cached) | **68.9%** | 82.0% | 62.5% | 85.7% | 95.4% | 90.9% | 83.3% | 76.9% | 83.3% | 2 | 33.3% | 5 | PASS | 0 (0%) | 0 | 39,283 | 52.9s | $0.00000 |
| `rule-based-non-llm` | SKIPPED (Cached) | **55.5%** | 44.0% | 50.0% | 42.9% | 38.9% | 22.2% | 54.2% | 87.5% | 100.0% | 4 | 0.0% | 11 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
