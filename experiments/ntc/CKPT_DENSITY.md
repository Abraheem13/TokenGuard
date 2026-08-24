# Checkpoint density: what does probing more often buy?

MATH-500, n=500, greedy decoding, 16k thinking budget, one symbolic grader; token counts are online cost inclusive of every probe purchased. `overhead` is the cost of running the controller and declining to halt, relative to plain generation. DEER is the authors' code at its default configuration, with overhead measured on the same convention. Where the density levels cover different numbers of items, all levels are scored on the items they share.

| model | density | probes/item | vanilla acc@tok | overhead | AGREE m=3 acc@tok | NTC-Select acc@tok | rho_w | P_spur | selected rule |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 1x  (every 256 tok, <=10) | 9.3 | 0.848 @ 5086 | +3.6% | 0.752 @ 2754 | 0.813 @ 3848 | 0.249 | 0.021 | NTC-v2{'m': 2, 'theta': 0.9} |
| Qwen3-4B | 2x  (every 128 tok, <=20) | 17.4 | 0.832 @ 5116 | +6.6% | 0.596 @ 1975 | 0.827 @ 4790 | 0.344 | 0.040 | NTC-v2{'m': 3, 'theta': 0.95} |
| Qwen3-4B | 4x  (every  64 tok, <=40) | 30.4 | 0.842 @ 5073 | +11.4% | 0.458 @ 1574 | 0.813 @ 4564 | 0.425 | 0.062 | NTC-v2{'m': 3, 'theta': 0.95} |
| Qwen3-4B | DEER (authors' code) | — | — | +73.2% | — | 0.920 @ 3539 | — | — | threshold 0.95 |
| Qwen3-8B | 1x  (every 256 tok, <=10) | 9.4 | 0.842 @ 5306 | +2.8% | 0.762 @ 2783 | 0.800 @ 4581 | 0.237 | 0.020 | NTC-v2{'m': 2, 'theta': 0.9} |
| Qwen3-8B | 2x  (every 128 tok, <=20) | — | *not yet generated* | — | — | — | — | — | — |
| Qwen3-8B | 4x  (every  64 tok, <=40) | — | *not yet generated* | — | — | — | — | — | — |
| Qwen3-8B | DEER (authors' code) | — | — | +78.7% | — | 0.930 @ 2946 | — | — | threshold 0.95 |

## Reading

Two things to read off. First, whether the controller's accuracy rises with density: if it does not, the gap to DEER is not granularity and the paper must say so. Second, whether rho_w rises with density: Proposition 2 predicts it must, because adjacent probes give a wrong answer fewer opportunities to change, and that agreement-based halting must therefore degrade as probes are placed closer together.

Missing files (2): `dens2x_math500_Qwen3-8B.json`, `dens4x_math500_Qwen3-8B.json`. Generate with `bash run_queue_density.sh`.
