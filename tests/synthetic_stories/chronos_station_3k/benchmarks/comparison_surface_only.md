# Model Extraction Benchmark: Echoes of Chronos Station (Extended 3K Edition) (`chronos_station_3k`)

- **Guidance Mode**: `surface_only`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | RAN | **94.3%** | 86.0% | 100.0% | 83.3% | 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 0 | 100.0% | 0 | PASS | 0 (0%) | 0 | 41,724 | 40.8s | $0.00598 |
| `rule-based-non-llm` | RAN | **54.9%** | 56.0% | 62.5% | 54.8% | 41.7% | 25.0% | 79.2% | 84.8% | 100.0% | 5 | 40.0% | 13 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
