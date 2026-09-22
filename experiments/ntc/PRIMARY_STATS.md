# Primary metric with uncertainty and paired significance

Deployable protocol. 5 random calibration/evaluation splits per generation-seed file; AUCC averaged within each setting, then aggregated across 12 settings. `wins` counts settings where NTC-full has the higher AUCC; the sign test is exact (two-sided) and the Wilcoxon statistic is the tie-corrected normal approximation.

| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |
|---|---|---|---|---|---|---|
| NTC-full (ours) | 0.660 | 0.176 | 0.388 | — | — | — |
| NTC-v2 (fusion) | 0.641 | 0.182 | 0.341 | 8/12 | 0.3877 | +1.65 |
| Answer agreement | 0.636 | 0.187 | 0.332 | 9/12 | 0.1460 | +1.80 |
| Entropy (EAT) | 0.592 | 0.193 | 0.270 | 10/12 | 0.0386* | +2.75 |
| Smoothed confidence | 0.586 | 0.158 | 0.356 | 12/12 | 0.0005* | +3.06 |
| Confidence (DEER-λ) | 0.565 | 0.157 | 0.300 | 10/12 | 0.0386* | +2.82 |

## OPERATIONAL-REGION AUCC (budgets b <= 0.6, where early exit matters)

Plain AUCC includes b = 1.0, where every method may simply never halt, so a third of the grid cannot separate methods at all. Restricting to the operational region measures the regime early exit exists for.

| method | mean | s.d. | worst setting | NTC-full wins | sign-test p |
|---|---|---|---|---|---|
| NTC-v2 (fusion) | 0.590 | 0.207 | 0.256 | 3/12 | 0.1460 |
| Answer agreement | 0.580 | 0.215 | 0.242 | 6/12 | 1.0000 |
| NTC-full (ours) | 0.564 | 0.248 | 0.132 | — | — |
| Entropy (EAT) | 0.529 | 0.218 | 0.197 | 8/12 | 0.3877 |
| Smoothed confidence | 0.522 | 0.187 | 0.254 | 9/11 | 0.0654 |
| Confidence (DEER-λ) | 0.490 | 0.177 | 0.203 | 8/12 | 0.3877 |

## Minimax regret (operational region)

For each setting, regret(M) = best AUCC in that setting minus M's AUCC; the table reports the MAXIMUM over settings. This is the decision-theoretic criterion for committing to one method without knowing which workload arrives: it penalises being far from the best on any single workload — exactly the failure mode of a fixed signal.

| method | max regret | mean regret |
|---|---|---|
| NTC-v2 (fusion) | 0.031 | 0.009 |
| Answer agreement | 0.100 | 0.019 |
| Entropy (EAT) | 0.145 | 0.070 |
| Smoothed confidence | 0.210 | 0.077 |
| NTC-full (ours) | 0.220 | 0.035 |
| Confidence (DEER-λ) | 0.262 | 0.109 |
