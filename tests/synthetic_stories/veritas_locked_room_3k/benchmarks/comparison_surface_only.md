# Model Extraction Benchmark: The Veritas Locked-Room Inquest (5-Character Detective Topology) (`veritas_locked_room_3k`)

- **Guidance Mode**: `surface_only`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | RAN | **80.8%** | 69.4% | 75.0% | 68.5% | 100.0% | 100.0% | 46.7% | 100.0% | 100.0% | 0 | 57.1% | 3 | PASS | 0 (0%) | 0 | 43,481 | 48.8s | $0.00655 |
| `rule-based-non-llm` | RAN | **52.4%** | 38.7% | 50.0% | 37.0% | 43.5% | 33.3% | 43.3% | 82.4% | 100.0% | 6 | 0.0% | 20 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
