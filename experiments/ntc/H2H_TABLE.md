# Head-to-head Table 1 — identical conditions
(same models, data, 16k thinking budget, greedy decoding, sympy grader;
token counts are ONLINE cost incl. all probe/trial tokens)

| model | benchmark | vanilla acc@tok | DEER official acc@tok | AGREE(m=3) acc@tok | NTC-full† acc@tok | NTC-full pick |
|---|---|---|---|---|---|---|
| Qwen3-4B | math500 (n=500) | 0.834 @ 5192 | 0.920 @ 3539 | 0.746 @ 2638 | 0.783 @ 5353 | NTC-conf{'theta': 0.99} |
| Qwen3-8B | math500 (n=500) | 0.830 @ 5450 | 0.930 @ 2946 | 0.754 @ 2771 | 0.817 @ 5765 | NTC-conf{'theta': 0.99} |
| Qwen3-4B | gpqa_diamond (n=198) | 0.399 @ 10359 | 0.545 @ 7536 | 0.364 @ 3396 | 0.429 @ 5574 | MUR-mom{'gamma': 0.9} |
| Qwen3-8B | gpqa_diamond (n=198) | 0.510 @ 10308 | 0.576 @ 8872 | 0.359 @ 3746 | 0.504 @ 9695 | EAT{'delta': 0.0001} |
| Qwen3-4B | aime24 (n=30) | 0.600 @ 11585 | 0.667 @ 10535 | 0.333 @ 6272 | 0.611 @ 10751 | DEER{'lam': 0.99} |
| Qwen3-8B | aime24 (n=30) | 0.533 @ 12181 | 0.667 @ 10012 | 0.433 @ 4956 | 0.500 @ 6372 | NTC-v2{'m': 3, 'theta': 0.5} |

† NTC-full: (signal, param) calibrated on 40% warm-up, held-out 60% reported; others are fixed-default policies on the full set (DEER's default lambda=0.95, AGREE's default m=3).
