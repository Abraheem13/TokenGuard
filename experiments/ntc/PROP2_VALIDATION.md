# Proposition 2 — empirical validation
Estimated on undecided items only; m=3.

| benchmark/model | eff. \|A\| | rho | q_max | K | bound | observed AGREE Δ |
|---|---|---|---|---|---|---|
| gsm8k/Qwen3-4B | 2.33 | 0.819 | 0.818 | 9.2 | 1.000 | +4.5 |
| gsm8k/Qwen3-8B | 2.36 | 0.812 | 0.803 | 9.1 | 1.000 | +8.5 |
| math500/Qwen3-1.7B | 2.69 | 0.587 | 0.750 | 9.7 | 1.000 | -2.0 |
| math500/Qwen3-4B | 2.68 | 0.560 | 0.739 | 9.6 | 1.000 | -5.5 |
| math500/Qwen3-8B | 2.58 | 0.575 | 0.763 | 9.8 | 1.000 | +3.0 |
| gpqa_diamond/Qwen3-4B | 1.89 | 0.857 | 0.790 | 9.9 | 1.000 | -4.0 |
| gpqa_diamond/Qwen3-8B | 1.97 | 0.849 | 0.758 | 10.0 | 1.000 | -13.1 |
| aime24/Qwen3-4B | 3.87 | 0.615 | 0.567 | 10.0 | 1.000 | -30.0 |
| aime25/Qwen3-4B | 3.93 | 0.593 | 0.537 | 10.0 | 1.000 | -10.0 |
| math500/DeepSeek-R1-Distill-Qwen-7B | 3.18 | 0.502 | 0.648 | 8.5 | 1.000 | -14.0 |

Spearman(bound, observed Δ) = -0.867
