# Checkpoint density: what does probing more often buy?

MATH-500, n=500, greedy decoding, 16k thinking budget, one symbolic grader; token counts are online cost inclusive of every probe purchased. `overhead` is the cost of running the controller and declining to halt, relative to plain generation. DEER is the authors' code at its default configuration, with overhead measured on the same convention.

| model | density | probes/item | vanilla acc@tok | overhead | AGREE m=3 acc@tok | NTC-Select acc@tok | selected rule |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 1x  (every 256 tok, <=10) | 9.3 | 0.848 @ 5086 | +3.6% | 0.752 @ 2754 | 0.813 @ 3848 | NTC-v2{'m': 2, 'theta': 0.9} |
| Qwen3-4B | 2x  (every 128 tok, <=20) | — | *not yet generated* | — | — | — | — |
| Qwen3-4B | 4x  (every  64 tok, <=40) | — | *not yet generated* | — | — | — | — |
| Qwen3-4B | DEER (authors' code) | — | — | +73.2% | — | 0.920 @ 3539 | threshold 0.95 |
| Qwen3-8B | 1x  (every 256 tok, <=10) | 9.4 | 0.842 @ 5306 | +2.8% | 0.762 @ 2783 | 0.800 @ 4581 | NTC-v2{'m': 2, 'theta': 0.9} |
| Qwen3-8B | 2x  (every 128 tok, <=20) | — | *not yet generated* | — | — | — | — |
| Qwen3-8B | 4x  (every  64 tok, <=40) | — | *not yet generated* | — | — | — | — |
| Qwen3-8B | DEER (authors' code) | — | — | +78.7% | — | 0.930 @ 2946 | threshold 0.95 |

## Reading

If accuracy rises with density and approaches DEER's while overhead rises towards DEER's, the gap of Table 12 is granularity and not signal quality, and the honest statement is that the two systems sit at different points on one density-overhead trade-off. If accuracy does not rise, the gap is the signal, and that should be said plainly.

Missing files (4): `dens2x_math500_Qwen3-4B.json`, `dens4x_math500_Qwen3-4B.json`, `dens2x_math500_Qwen3-8B.json`, `dens4x_math500_Qwen3-8B.json`. Generate with `bash run_queue_density.sh`.
