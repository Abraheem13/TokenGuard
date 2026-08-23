# Does the risk certificate survive domain shift?

12 target domains x 5 calibration splits; |C| = 19; alpha = 0.1. `worst cell` is the single worst domain-split combination, a harsher statistic than the seed-averaged worst case of SLO_ATTAINMENT.md. `transferred` is averaged over all 11 sources per target.

## eps = 0.01

| certification rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |
|---|---|---|---|---|---|
| in-domain (A1 holds) | +0.10 | -4.2 | 88% | 88% | 5.5% |
| transferred (A1 violated) | -1.53 | -38.9 | 75% | 75% | 9.2% |
| domain-robust (LODO) | +0.00 | +0.0 | 100% | 100% | -3.0% |

in-domain picks: `{'NEVER-HALT{}': 21, "NTC-conf{'theta': 0.99}": 8, "DEER{'lam': 0.95}": 8, "NTC-v2{'m': 2, 'theta': 0.95}": 5, "EAT{'delta': 0.0001}": 4}`

domain-robust picks: `{'NEVER-HALT{}': 60}`

## eps = 0.05

| certification rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |
|---|---|---|---|---|---|
| in-domain (A1 holds) | +0.70 | -5.8 | 80% | 97% | 19.6% |
| transferred (A1 violated) | -3.42 | -50.0 | 53% | 71% | 25.4% |
| domain-robust (LODO) | +0.00 | +0.0 | 100% | 100% | -3.0% |

in-domain picks: `{"NTC-v2{'m': 2, 'theta': 0.9}": 9, "NTC-conf{'theta': 0.85}": 8, "NTC-v2{'m': 3, 'theta': 0.9}": 7, "DEER{'lam': 0.95}": 7, "EAT{'delta': 0.001}": 5}`

domain-robust picks: `{'NEVER-HALT{}': 60}`

