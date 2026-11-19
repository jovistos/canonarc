# Model Extraction Benchmark: Echoes of Chronos Station (Extended 3K Edition) (`chronos_station_3k`)

- **Guidance Mode**: `none`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | SKIPPED (Cached) | **95.0%** | 84.0% | 100.0% | 81.0% | 100.0% | 100.0% | 79.2% | 100.0% | 100.0% | 0 | 100.0% | 0 | PASS | 0 (0%) | 0 | 41,769 | 37.6s | $0.00589 |
| `qwen/qwen3.8-27b` | RAN | **82.0%** | 90.0% | 75.0% | 92.9% | 100.0% | 100.0% | 91.7% | 93.3% | 100.0% | 2 | 33.3% | 5 | PASS | 0 (0%) | 0 | 39,571 | 78.8s | $0.00000 |
| `qwen/qwen3.5-9b` | SKIPPED (Cached) | **70.6%** | 82.0% | 62.5% | 85.7% | 93.1% | 90.9% | 83.3% | 60.0% | 66.7% | 0 | 28.6% | 4 | PASS | 0 (0%) | 0 | 38,944 | 48.5s | $0.00000 |
| `rule-based-non-llm` | SKIPPED (Cached) | **57.8%** | 56.0% | 62.5% | 54.8% | 41.7% | 25.0% | 79.2% | 84.9% | 100.0% | 5 | 40.0% | 13 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
