# Proposition 2 — empirical validation (corrected estimator)
rho_w / q_w estimated over INCORRECT trial answers only; m=3.

| benchmark/model | rho_w | q_w | P_spur | lost-correct risk | AGREE Δ (pts) | n |
|---|---|---|---|---|---|---|
| gsm8k/Qwen3-4B | 0.130 | 0.098 | 0.002 | 0.034 | +4.5 | 200 |
| gsm8k/Qwen3-8B | 0.157 | 0.101 | 0.002 | 0.030 | +8.5 | 200 |
| math500/Qwen3-1.7B | 0.257 | 0.361 | 0.024 | 0.110 | -2.0 | 200 |
| math500/Qwen3-4B | 0.270 | 0.339 | 0.025 | 0.134 | -5.5 | 200 |
| math500/Qwen3-8B | 0.233 | 0.314 | 0.017 | 0.067 | +3.0 | 200 |
| gpqa_diamond/Qwen3-4B | 0.765 | 0.468 | 0.274 | 0.277 | -4.0 | 198 |
| gpqa_diamond/Qwen3-8B | 0.724 | 0.444 | 0.232 | 0.391 | -13.1 | 198 |
| aime24/Qwen3-4B | 0.425 | 0.417 | 0.075 | 0.450 | -30.0 | 30 |
| aime25/Qwen3-4B | 0.375 | 0.347 | 0.049 | 0.286 | -10.0 | 30 |
| math500/DeepSeek-R1-Distill-Qwen-7B | 0.256 | 0.319 | 0.021 | 0.184 | -14.0 | 200 |

n = 10 settings.
Spearman(P_spur, lost-correct risk) = +0.867 (positive supports Prop. 2).
Spearman(P_spur, AGREE delta) = -0.648 (negative supports Prop. 2).
