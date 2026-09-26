# Matched comparison with DEER

Same models and items, 16k thinking budget, greedy decoding, one symbolic grader. Each cell is accuracy @ mean tokens per item, counting every probe or trial answer paid online. `vanilla-BF`: full generation with budget forcing.

| model | benchmark | vanilla-BF acc@tok | DEER official acc@tok | AGREE(m=3) acc@tok | NTC-full† acc@tok | NTC-full pick |
|---|---|---|---|---|---|---|
| Qwen3-4B | math500 (n=500) | 0.862 @ 5087 | 0.920 @ 3539 | 0.752 @ 2754 | 0.850 @ 4792 | NTC-v2{'m': 2, 'theta': 0.95} |
| Qwen3-8B | math500 (n=500) | 0.854 @ 5307 | 0.930 @ 2946 | 0.762 @ 2783 | 0.823 @ 5289 | NTC-v2{'m': 2, 'theta': 0.95} |
| Qwen3-4B | gpqa_diamond (n=198) | 0.500 @ 9798 | 0.545 @ 7536 | 0.354 @ 3263 | 0.479 @ 9783 | NTC-conf{'theta': 0.99} |
| Qwen3-8B | gpqa_diamond (n=198) | 0.571 @ 9889 | 0.576 @ 8872 | 0.389 @ 3121 | 0.513 @ 10353 | NEVER-HALT{} |
| Qwen3-4B | aime24 (n=30) | 0.600 @ 12233 | 0.667 @ 10535 | 0.233 @ 5200 | 0.611 @ 12010 | NTC-v2{'m': 2, 'theta': 0.95} |
| Qwen3-8B | aime24 (n=30) | 0.667 @ 11883 | 0.667 @ 10012 | 0.467 @ 6023 | 0.556 @ 11551 | NTC-v2{'m': 2, 'theta': 0.9} |

† NTC-full (the selection tier) is calibrated on 40% of the items and scored on the other 60%; the other columns are fixed defaults on all items (DEER threshold 0.95, agreement m = 3).
