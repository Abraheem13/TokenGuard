# Error stickiness within MMLU-Pro, by size of the answer space

Items pooled over three generation seeds; m = 3. Buckets with fewer than 40 items are omitted. `AGREE delta` is the accuracy change of answer agreement against full generation, in points.

| model | bucket | items | rho_w | q_w | P_spur | lost-correct | AGREE delta |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | |A| <= 6 | 45 | 0.774 | 0.395 | 0.237 | 0.214 | -8.9 |
| Qwen3-4B | |A| = 10 | 517 | 0.610 | 0.327 | 0.121 | 0.173 | -8.9 |
| Qwen3-8B | |A| <= 6 | 47 | 0.682 | 0.375 | 0.174 | 0.133 | -2.1 |
| Qwen3-8B | |A| = 10 | 515 | 0.557 | 0.273 | 0.084 | 0.192 | -9.7 |
