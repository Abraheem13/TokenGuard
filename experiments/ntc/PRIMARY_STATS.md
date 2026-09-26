# Primary comparison with uncertainty and paired tests

Deployable protocol. 5 random calibration/evaluation splits per generation-seed file; AUCC is averaged within each setting, then aggregated across 12 settings. `NTC-full wins` counts settings where the selection tier has the higher AUCC; the sign test is exact and two-sided, and the Wilcoxon statistic is the tie-corrected normal approximation. `*`: p < 0.05.

| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |
|---|---|---|---|---|---|---|
| NTC-full (selection) | 0.677 | 0.153 | 0.402 | n/a | n/a | n/a |
| NTC-v2 (fusion) | 0.658 | 0.162 | 0.341 | 8/12 | 0.3877 | +1.49 |
| Answer agreement | 0.653 | 0.168 | 0.332 | 9/12 | 0.1460 | +1.80 |
| Entropy (EAT) | 0.609 | 0.179 | 0.270 | 11/12 | 0.0063* | +2.75 |
| Smoothed confidence | 0.600 | 0.144 | 0.356 | 12/12 | 0.0005* | +3.06 |
| Confidence (DEER-λ) | 0.577 | 0.146 | 0.300 | 12/12 | 0.0005* | +3.06 |

## OPERATIONAL-REGION AUCC (budgets b <= 0.6, where early exit matters)

AUCC over the full grid includes b = 1.0, where every method may decline to halt, so part of the grid cannot separate methods. The operational region restricts the average to the budgets at which early exit is required.

| method | mean | s.d. | worst setting | NTC-full wins | sign-test p |
|---|---|---|---|---|---|
| NTC-v2 (fusion) | 0.611 | 0.187 | 0.256 | 3/12 | 0.1460 |
| Answer agreement | 0.601 | 0.197 | 0.242 | 6/12 | 1.0000 |
| NTC-full (selection) | 0.585 | 0.232 | 0.132 | n/a | n/a |
| Entropy (EAT) | 0.550 | 0.203 | 0.197 | 7/12 | 0.7744 |
| Smoothed confidence | 0.541 | 0.171 | 0.254 | 10/12 | 0.0386* |
| Confidence (DEER-λ) | 0.505 | 0.169 | 0.203 | 9/12 | 0.1460 |

## Minimax regret (operational region)

For each setting, regret(M) is the best operational-region AUCC in that setting minus that of M. The table reports its maximum and mean over settings; the maximum is the criterion for committing to one method without knowing which workload will arrive.

| method | max regret | mean regret |
|---|---|---|
| NTC-v2 (fusion) | 0.031 | 0.008 |
| Answer agreement | 0.100 | 0.017 |
| Entropy (EAT) | 0.145 | 0.069 |
| Smoothed confidence | 0.210 | 0.077 |
| NTC-full (selection) | 0.220 | 0.034 |
| Confidence (DEER-λ) | 0.262 | 0.114 |
