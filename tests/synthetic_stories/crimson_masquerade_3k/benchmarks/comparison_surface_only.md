# Model Extraction Benchmark: The Crimson Masquerade (Extended 3K Edition) (`crimson_masquerade_3k`)

- **Guidance Mode**: `surface_only`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | RAN | **84.8%** | 82.0% | 75.0% | 83.3% | 81.5% | 80.0% | 87.5% | 88.0% | 66.7% | 0 | 100.0% | 3 | PASS | 0 (0%) | 0 | 41,239 | 36.8s | $0.00542 |
| `rule-based-non-llm` | RAN | **52.1%** | 44.0% | 50.0% | 42.9% | 38.9% | 22.2% | 54.2% | 87.5% | 100.0% | 4 | 0.0% | 11 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
