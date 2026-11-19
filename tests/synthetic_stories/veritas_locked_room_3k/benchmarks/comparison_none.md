# Model Extraction Benchmark: The Veritas Locked-Room Inquest (5-Character Detective Topology) (`veritas_locked_room_3k`)

- **Guidance Mode**: `none`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | SKIPPED (Cached) | **81.8%** | 69.4% | 75.0% | 68.5% | 100.0% | 100.0% | 46.7% | 100.0% | 100.0% | 0 | 33.3% | 3 | PASS | 0 (0%) | 0 | 44,503 | 48.4s | $0.00667 |
| `qwen/qwen3.8-27b` | RAN | **85.5%** | 72.6% | 75.0% | 72.2% | 100.0% | 100.0% | 53.3% | 96.3% | 83.3% | 0 | 33.3% | 1 | PASS | 0 (0%) | 0 | 39,549 | 77.1s | $0.00000 |
| `qwen/qwen3.5-9b` | SKIPPED (Cached) | **69.5%** | 74.2% | 62.5% | 75.9% | 72.3% | 83.3% | 86.7% | 54.5% | 50.0% | 2 | 33.3% | 6 | PASS | 0 (0%) | 0 | 39,045 | 44.6s | $0.00000 |
| `rule-based-non-llm` | SKIPPED (Cached) | **52.4%** | 38.7% | 50.0% | 37.0% | 43.5% | 33.3% | 43.3% | 82.3% | 100.0% | 6 | 0.0% | 20 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
