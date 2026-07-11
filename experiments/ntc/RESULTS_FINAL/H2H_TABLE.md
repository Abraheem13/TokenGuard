# Head-to-head Table 1 — identical conditions
(same models, data, 16k thinking budget, greedy decoding, sympy grader;
token counts are ONLINE cost incl. all probe/trial tokens)

| model | benchmark | vanilla-BF acc@tok | DEER official acc@tok | AGREE(m=3) acc@tok | NTC-full† acc@tok | NTC-full pick |
|---|---|---|---|---|---|---|
| Qwen3-4B | math500 (n=500) | 0.862 @ 5087 | 0.920 @ 3539 | 0.752 @ 2754 | 0.830 @ 5234 | NTC-conf{'theta': 0.99} |
| Qwen3-8B | math500 (n=500) | 0.854 @ 5307 | 0.930 @ 2946 | 0.762 @ 2783 | 0.817 @ 5514 | NTC-conf{'theta': 0.99} |
| Qwen3-4B | gpqa_diamond (n=198) | 0.500 @ 9798 | 0.545 @ 7536 | 0.354 @ 3263 | 0.471 @ 6126 | MUR-mom{'gamma': 0.9} |
| Qwen3-8B | gpqa_diamond (n=198) | 0.571 @ 9889 | 0.576 @ 8872 | 0.389 @ 3121 | 0.487 @ 7300 | MUR-mom{'gamma': 0.9} |
| Qwen3-4B | aime24 (n=30) | 0.600 @ 12233 | 0.667 @ 10535 | 0.233 @ 5200 | 0.611 @ 11793 | DEER{'lam': 0.99} |
| Qwen3-8B | aime24 (n=30) | 0.667 @ 11883 | 0.667 @ 10012 | 0.467 @ 6023 | 0.556 @ 11779 | DEER{'lam': 0.99} |

† NTC-full: (signal, param) calibrated on 40% warm-up, held-out 60% reported; others are fixed-default policies on the full set (DEER's default lambda=0.95, AGREE's default m=3).
