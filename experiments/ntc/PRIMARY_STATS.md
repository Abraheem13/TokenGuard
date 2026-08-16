# Primary metric with uncertainty and paired significance

Deployable protocol. 5 random calibration/evaluation splits per generation-seed file; AUCC averaged within each setting, then aggregated across 12 settings. `wins` counts settings where NTC-full has the higher AUCC; the sign test is exact (two-sided) and the Wilcoxon statistic is the tie-corrected normal approximation.

| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |
|---|---|---|---|---|---|---|
| NTC-full (ours) | 0.679 | 0.152 | 0.441 | — | — | — |
| NTC-v2 (fusion) | 0.664 | 0.151 | 0.428 | 8/12 | 0.3877 | +1.49 |
| Answer agreement | 0.654 | 0.166 | 0.382 | 9/12 | 0.1460 | +1.80 |
| Entropy (EAT) | 0.612 | 0.177 | 0.289 | 12/12 | 0.0005* | +3.06 |
| Smoothed confidence | 0.607 | 0.132 | 0.451 | 11/12 | 0.0063* | +2.90 |
| Confidence (DEER-λ) | 0.586 | 0.133 | 0.386 | 11/12 | 0.0063* | +2.98 |

## OPERATIONAL-REGION AUCC (budgets b <= 0.6, where early exit matters)

Plain AUCC includes b = 1.0, where every method may simply never halt, so a third of the grid cannot separate methods at all. Restricting to the operational region measures the regime early exit exists for.

| method | mean | s.d. | worst setting | NTC-full wins | sign-test p |
|---|---|---|---|---|---|
| NTC-v2 (fusion) | 0.622 | 0.170 | 0.344 | 4/12 | 0.3877 |
| Answer agreement | 0.605 | 0.195 | 0.239 | 6/12 | 1.0000 |
| NTC-full (ours) | 0.598 | 0.216 | 0.167 | — | — |
| Entropy (EAT) | 0.555 | 0.200 | 0.211 | 9/12 | 0.1460 |
| Smoothed confidence | 0.553 | 0.155 | 0.354 | 10/12 | 0.0386* |
| Confidence (DEER-λ) | 0.513 | 0.159 | 0.263 | 10/12 | 0.0386* |

## Minimax regret (operational region)

For each setting, regret(M) = best AUCC in that setting minus M's AUCC; the table reports the MAXIMUM over settings. This is the decision-theoretic criterion for committing to one method without knowing which workload arrives: it penalises being far from the best on any single workload — exactly the failure mode of a fixed signal.

| method | max regret | mean regret |
|---|---|---|
| NTC-v2 (fusion) | 0.036 | 0.009 |
| Answer agreement | 0.152 | 0.027 |
| Entropy (EAT) | 0.152 | 0.077 |
| NTC-full (ours) | 0.187 | 0.033 |
| Smoothed confidence | 0.201 | 0.078 |
| Confidence (DEER-λ) | 0.258 | 0.118 |
