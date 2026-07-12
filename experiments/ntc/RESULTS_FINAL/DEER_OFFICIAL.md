# DEER official results (authors' code iie-ycx/DEER, default config:
# threshold 0.95, think_ratio 0.8, policy avg2, max 16000, greedy)
| model | dataset | acc | final_tokens | total_tokens_incl_trials |
|---|---|---|---|---|
| Qwen3-4B | math500 | 0.9200 | 2042.7 | 3538.8 |
| Qwen3-4B | gpqa | 0.5455 | 4529.0 | 7535.9 |
| Qwen3-4B | aime24 | 0.6667 | 6291.5 | 10534.8 |
| Qwen3-8B | math500 | 0.9300 | 1649.0 | 2946.2 |
| Qwen3-8B | gpqa | 0.5758 | 5258.4 | 8872.2 |
| Qwen3-8B | aime24 | 0.6667 | 5708.7 | 10011.5 |

Note: total_tokens counts all trial-answer inductions DEER pays online
(~1.7x the final chain on MATH — a 40-73% probe overhead vs our ~3-5%).

## Extended official runs (attack set + temp regimes)
| model | dataset (temp) | acc | final_tokens | total_tokens |
|---|---|---|---|---|
| Qwen3-4B | gsm8k (T=0.0) | 0.9409 | 787.9014404852161 | 1213.393479909022 |
| Qwen3-8B | gsm8k (T=0.0) | 0.9500 | 679.3828658074299 | 1064.120545868082 |
| DeepSeek-R1-Distill-Qwen-7B | math (T=0.0) | 0.8820 | 1329.792 | 2396.964 |
| Qwen3-4B | math (T=0.0) | 0.9200 | 2042.668 | 3538.766 |
| Qwen3-4B | math (T=0.6) | 0.9240 | 2017.008 | 3529.164 |
| Qwen3-8B | math (T=0.0) | 0.9300 | 1649.006 | 2946.16 |
| Qwen3-8B | math (T=0.6) | 0.9340 | 1735.988 | 3043.706 |
