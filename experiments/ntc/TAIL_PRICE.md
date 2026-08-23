# Pricing the tail (11 canonical settings)

Utility of deploying method M on setting s is `cut - kappa * max(0, -delta)`, kappa in token-saving points forgone per accuracy point.

| method | mean cut | mean delta | worst delta |
|---|---|---|---|
| NTC-Fuse | 13.3% | +1.12 | -0.50 |
| NTC-Select (eps=0.01) | 8.9% | +0.40 | -1.60 |
| NTC-Select (eps=0.05) | 22.5% | +0.52 | -1.60 |
| Answer agreement | 55.7% | -5.41 | -23.50 |
| Confidence threshold | 4.4% | -2.21 | -7.50 |
| Smoothed confidence | 1.6% | -0.65 | -2.40 |
| Entropy (EAT) | 12.4% | -1.21 | -3.40 |
| Uncertainty momentum | 10.8% | -5.18 | -23.60 |
| Bandit threshold | 33.0% | -7.84 | -20.90 |

## Best method as a function of kappa

| kappa | best by mean utility | best by worst-case utility |
|---|---|---|
| 0 | Answer agreement (55.7) | Answer agreement (44.9) |
| 1 | Answer agreement (48.9) | Answer agreement (21.8) |
| 2 | Answer agreement (42.2) | NTC-Select (eps=0.05) (3.7) |
| 3 | Answer agreement (35.4) | NTC-Select (eps=0.05) (2.1) |
| 4 | Answer agreement (28.6) | NTC-Select (eps=0.05) (0.5) |
| 6 | NTC-Select (eps=0.05) (19.2) | NTC-Fuse (-2.3) |
| 8 | NTC-Select (eps=0.05) (18.1) | NTC-Fuse (-2.9) |
| 10 | NTC-Select (eps=0.05) (17.0) | NTC-Fuse (-3.5) |
| 15 | NTC-Select (eps=0.05) (14.3) | NTC-Fuse (-5.0) |
| 20 | NTC-Select (eps=0.05) (11.6) | NTC-Fuse (-7.4) |
| 30 | NTC-Fuse (10.0) | NTC-Fuse (-12.4) |

## Crossovers against NTC-Fuse

- mean: NTC-Select (eps=0.05) loses to NTC-Fuse beyond kappa = 21.19
- mean: Answer agreement loses to NTC-Fuse beyond kappa = 6.38
- mean: Bandit threshold loses to NTC-Fuse beyond kappa = 2.56
- worst-case: NTC-Select (eps=0.05) loses to NTC-Fuse beyond kappa = 5.69
- worst-case: Answer agreement loses to NTC-Fuse beyond kappa = 1.97
- worst-case: Entropy (EAT) loses to NTC-Fuse beyond kappa = 0.30
- worst-case: Uncertainty momentum loses to NTC-Fuse beyond kappa = 0.83
- worst-case: Bandit threshold loses to NTC-Fuse beyond kappa = 1.86
