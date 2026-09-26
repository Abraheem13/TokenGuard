# Checkpoint density: what does probing more often buy?

MATH-500, n = 500, greedy decoding, 16k thinking budget, one symbolic grader. Each `acc@tok` cell is accuracy @ mean online tokens per item, counting every probe paid. `overhead` is the cost of probing at every checkpoint without halting, relative to full generation. The selection tier (NTC-Select) is calibrated on 40% of the items with eps = 0.05 and scored on the rest. rho_w is the error stickiness between consecutive probes and P_spur = rho_w^2 * q_w. DEER is the authors' code at its default configuration, with overhead measured on the same convention.

| model | density | probes/item | vanilla acc@tok | overhead | AGREE m=3 acc@tok | NTC-Select acc@tok | rho_w | P_spur | selected rule |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 1x  (every 256 tok, <=10) | 9.3 | 0.848 @ 5086 | +3.6% | 0.752 @ 2754 | 0.813 @ 3848 | 0.249 | 0.021 | NTC-v2{'m': 2, 'theta': 0.9} |
| Qwen3-4B | 2x  (every 128 tok, <=20) | 17.4 | 0.832 @ 5116 | +6.6% | 0.596 @ 1975 | 0.827 @ 4790 | 0.344 | 0.040 | NTC-v2{'m': 3, 'theta': 0.95} |
| Qwen3-4B | 4x  (every  64 tok, <=40) | 30.4 | 0.842 @ 5073 | +11.4% | 0.458 @ 1574 | 0.813 @ 4564 | 0.425 | 0.062 | NTC-v2{'m': 3, 'theta': 0.95} |
| Qwen3-4B | DEER (authors' code) | n/a | n/a | +73.2% | n/a | 0.920 @ 3539 | n/a | n/a | threshold 0.95 |
