# Primary metric with uncertainty and paired significance

Deployable protocol. 5 random calibration/evaluation splits per generation-seed file; AUCC averaged within each setting, then aggregated across 12 settings. `wins` counts settings where NTC-full has the higher AUCC; the sign test is exact (two-sided) and the Wilcoxon statistic is the tie-corrected normal approximation.

| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |
|---|---|---|---|---|---|---|
| NTC-full (ours) | 0.667 | 0.150 | 0.441 | — | — | — |
| NTC-v2 (fusion) | 0.664 | 0.151 | 0.428 | 6/12 | 1.0000 | +0.00 |
| Answer agreement | 0.654 | 0.166 | 0.382 | 6/12 | 1.0000 | +0.31 |
| Entropy (EAT) | 0.611 | 0.175 | 0.289 | 10/12 | 0.0386* | +2.67 |
| Smoothed confidence | 0.608 | 0.133 | 0.451 | 10/12 | 0.0386* | +2.75 |
| Confidence (DEER-λ) | 0.586 | 0.133 | 0.386 | 11/12 | 0.0063* | +2.90 |
