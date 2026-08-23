# Proposition 2 tested within MMLU-Pro (answer-space size varied, all else fixed)

Items pooled over three generation seeds; m = 3. Buckets with fewer than 40 items are reported but not interpreted.

| model | bucket | items | rho_w | q_w | P_spur | lost-correct | AGREE delta |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | |A| <= 6 | 45 | 0.774 | 0.395 | 0.237 | 0.214 | -8.9 |
| Qwen3-4B | |A| = 10 | 517 | 0.610 | 0.327 | 0.121 | 0.173 | -8.9 |
| Qwen3-8B | |A| <= 6 | 47 | 0.682 | 0.375 | 0.174 | 0.133 | -2.1 |
| Qwen3-8B | |A| = 10 | 515 | 0.557 | 0.273 | 0.084 | 0.192 | -9.7 |

## Reading

Stickiness falls monotonically with the answer space in both models, and the small-|A| bucket reproduces the value measured on four-option GPQA-Diamond (0.72-0.77). The downstream deficit in the small bucket rests on ~46 items per model and is not resolvable at that sample size; the claim rests on rho_w and P_spur.
