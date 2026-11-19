# Model Extraction Benchmark: The Obsidian Signet Conspiracy (Extended 3K Edition) (`obsidian_signet_3k`)

- **Guidance Mode**: `surface_only`
- **Total Story Chunks**: `6`

| Model | Run Status | Overall Quality | Strict Checks | Step-Delta Checks | Cumulative State | Room Occ. Jaccard | Loc Delta F1 | Mood Soft Acc | Witness F1 | Silent Witness | Epistemic Leaks | Inform F1 | POV Secret Leaks | Flashback Guard | Hallucinated IDs | Retries | Tokens | Latency (s) | Est. Cost (USD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | SKIPPED (Cached) | **88.5%** | 84.0% | 87.5% | 83.3% | 100.0% | 92.3% | 70.8% | 100.0% | 100.0% | 0 | 57.1% | 1 | PASS | 3 (4%) | 0 | 43,057 | 50.5s | $0.00650 |
| `qwen/qwen3.8-27b` | RAN | **82.4%** | 92.0% | 75.0% | 95.2% | 100.0% | 100.0% | 91.7% | 93.3% | 100.0% | 2 | 50.0% | 4 | PASS | 2 (3%) | 0 | 38,287 | 76.1s | $0.00000 |
| `qwen/qwen3.5-9b` | SKIPPED (Cached) | **69.7%** | 78.0% | 62.5% | 81.0% | 89.8% | 44.4% | 79.2% | 75.0% | 83.3% | 1 | 40.0% | 3 | PASS | 9 (11%) | 0 | 38,310 | 53.2s | $0.00000 |
| `rule-based-non-llm` | SKIPPED (Cached) | **54.3%** | 56.0% | 50.0% | 57.1% | 55.6% | 0.0% | 70.8% | 90.3% | 100.0% | 3 | 0.0% | 8 | PASS | 0 (0%) | 0 | 0 | 0.0s | $0.00000 |
