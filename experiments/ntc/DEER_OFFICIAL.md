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
