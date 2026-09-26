# Does the fusion tier's parameter transfer across domains?

12 domains x 5 splits, the same evaluation items as SHIFT_CERTIFICATE.md. Deficit: accuracy change against full generation on the evaluation split, in points. Cut: saving in % of full-generation tokens (KV-fork).

## Fixed parameters (no calibration)

| m | theta | mean deficit | worst cell | within 1 pt | mean cut |
|---|---|---|---|---|---|
| 2 | 0.7 | -5.48 | -22.2 | 33% | 55.1% |
| 2 | 0.9 | +0.38 | -5.9 | 77% | 17.5% |
| 2 | 0.95 | -0.28 | -4.2 | 83% | 5.0% |
| 3 | 0.7 | -0.75 | -16.7 | 43% | 43.6% |
| 3 | 0.9 | +1.45 | -4.2 | 92% | 13.4% |
| 3 | 0.95 | +0.35 | -1.7 | 98% | 3.0% |

## eps = 0.01

| rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |
|---|---|---|---|---|---|
| transferred (tuned on another domain) | -1.69 | -22.2 | 60% | 60% | 32.0% |
| leave-one-domain-out | +0.61 | -4.2 | 90% | 90% | 7.8% |
| library leave-one-domain-out, no bound | -0.01 | -0.8 | 100% | 100% | -2.9% |

LODO picks: `{'m=3,theta=0.9': 25, 'm=3,theta=0.95': 33, 'm=2,theta=0.95': 2}`

library LODO picks: `{'NEVER-HALT{}': 59, "NTC-v2{'m': 3, 'theta': 0.95}": 1}`

Per-target worst transferred cell: GSM8K-4B -2.5, GSM8K-8B +0.8, MATH-1.7B -15.0, MATH-4B -17.5, MATH-8B -7.5, GPQA-4B -10.9, GPQA-8B -14.3, MMLU-Pro-4B -15.0, MMLU-Pro-8B -3.3, AIME-24 -22.2, AIME-25 +0.0, DeepSeek-MATH -8.3

## eps = 0.05

| rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |
|---|---|---|---|---|---|
| transferred (tuned on another domain) | -2.74 | -22.2 | 47% | 62% | 42.3% |
| leave-one-domain-out | +0.53 | -5.9 | 82% | 95% | 14.2% |
| library leave-one-domain-out, no bound | +0.54 | -5.9 | 82% | 95% | 14.5% |

LODO picks: `{'m=2,theta=0.9': 37, 'm=3,theta=0.95': 11, 'm=3,theta=0.9': 12}`

library LODO picks: `{"NTC-v2{'m': 2, 'theta': 0.9}": 37, "NTC-v2{'m': 3, 'theta': 0.9}": 11, "NTC-v2{'m': 3, 'theta': 0.95}": 10, "EAT{'delta': 0.0001}": 1, "EAT{'delta': 0.001}": 1}`

Per-target worst transferred cell: GSM8K-4B -2.5, GSM8K-8B +1.7, MATH-1.7B -15.0, MATH-4B -17.5, MATH-8B -7.5, GPQA-4B -10.9, GPQA-8B -14.3, MMLU-Pro-4B -15.0, MMLU-Pro-8B -3.3, AIME-24 -22.2, AIME-25 +0.0, DeepSeek-MATH -8.3
