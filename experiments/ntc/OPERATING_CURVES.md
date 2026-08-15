# Primary evaluation — operating curves under two protocols

`ORACLE` gives each fixed policy its best knob per budget, chosen on the evaluation data (an upper bound on the signal, not deployable). `DEPLOYABLE` makes every method choose its knob from the warm-up split only — what a user can actually ship. Both are scored on the same held-out split with overhead-inclusive costs.

## gsm8k / Qwen3-4B — held-out n=120, vanilla accuracy 0.900

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.950 | 100% | 0.942 | 0.33 |
| Answer agreement | 0.938 | 100% | 0.933 | 0.44 |
| NTC-full (ours) | 0.931 | 100% | 0.933 | 0.33 |
| Smoothed confidence | 0.924 | 100% | 0.925 | 0.29 |
| Entropy (EAT) | 0.908 | 86% | 0.908 | 0.47 |
| Confidence (DEER-λ) | 0.864 | 100% | 0.867 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.931 | 100% | 0.933 | 0.33 |
| Answer agreement | 0.926 | 100% | 0.933 | 0.44 |
| Entropy (EAT) | 0.908 | 86% | 0.908 | 0.47 |
| Smoothed confidence | 0.893 | 100% | 0.892 | 0.29 |
| NTC-v2 (fusion) | 0.886 | 100% | 0.883 | 1.00 |
| Confidence (DEER-λ) | 0.857 | 100% | 0.867 | 1.00 |
## gsm8k / Qwen3-8B — held-out n=120, vanilla accuracy 0.842

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.924 | 100% | 0.908 | 0.31 |
| NTC-v2 (fusion) | 0.923 | 100% | 0.917 | 0.31 |
| NTC-full (ours) | 0.908 | 100% | 0.908 | 0.31 |
| Smoothed confidence | 0.893 | 100% | 0.900 | 0.29 |
| Entropy (EAT) | 0.883 | 86% | 0.883 | 0.44 |
| Confidence (DEER-λ) | 0.838 | 100% | 0.833 | 0.34 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.908 | 100% | 0.908 | 0.31 |
| NTC-v2 (fusion) | 0.908 | 100% | 0.908 | 0.31 |
| NTC-full (ours) | 0.908 | 100% | 0.908 | 0.31 |
| Entropy (EAT) | 0.883 | 86% | 0.883 | 0.44 |
| Confidence (DEER-λ) | 0.835 | 100% | 0.833 | 0.34 |
| Smoothed confidence | 0.806 | 100% | 0.800 | 1.00 |
## math500 / Qwen3-1.7B — held-out n=120, vanilla accuracy 0.767

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.731 | 100% | 0.658 | 0.64 |
| Answer agreement | 0.712 | 100% | 0.617 | 0.64 |
| NTC-full (ours) | 0.696 | 100% | 0.658 | 1.00 |
| Entropy (EAT) | 0.624 | 86% | 0.508 | 1.00 |
| Smoothed confidence | 0.612 | 100% | 0.558 | 1.00 |
| Confidence (DEER-λ) | 0.567 | 100% | 0.508 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.696 | 100% | 0.658 | 1.00 |
| Answer agreement | 0.690 | 100% | 0.617 | 1.00 |
| NTC-v2 (fusion) | 0.690 | 100% | 0.617 | 1.00 |
| Entropy (EAT) | 0.578 | 86% | 0.508 | 1.00 |
| Smoothed confidence | 0.561 | 100% | 0.558 | 1.00 |
| Confidence (DEER-λ) | 0.529 | 100% | 0.508 | 1.00 |
## math500 / Qwen3-4B — held-out n=120, vanilla accuracy 0.783

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.773 | 100% | 0.758 | 0.54 |
| Answer agreement | 0.765 | 100% | 0.758 | 0.61 |
| NTC-full (ours) | 0.694 | 100% | 0.675 | 1.00 |
| Entropy (EAT) | 0.681 | 86% | 0.600 | 1.00 |
| Confidence (DEER-λ) | 0.630 | 100% | 0.567 | 1.00 |
| Smoothed confidence | 0.630 | 100% | 0.567 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.765 | 100% | 0.758 | 0.61 |
| NTC-v2 (fusion) | 0.765 | 100% | 0.758 | 0.61 |
| NTC-full (ours) | 0.694 | 100% | 0.675 | 1.00 |
| Entropy (EAT) | 0.661 | 86% | 0.600 | 1.00 |
| Confidence (DEER-λ) | 0.598 | 100% | 0.492 | 1.00 |
| Smoothed confidence | 0.581 | 100% | 0.567 | 1.00 |
## math500 / Qwen3-8B — held-out n=120, vanilla accuracy 0.758

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.819 | 100% | 0.808 | 0.41 |
| Answer agreement | 0.808 | 100% | 0.808 | 0.48 |
| NTC-full (ours) | 0.799 | 100% | 0.808 | 0.48 |
| Entropy (EAT) | 0.685 | 86% | 0.617 | 1.00 |
| Smoothed confidence | 0.637 | 100% | 0.608 | 1.00 |
| Confidence (DEER-λ) | 0.587 | 100% | 0.492 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.799 | 100% | 0.808 | 0.48 |
| NTC-v2 (fusion) | 0.799 | 100% | 0.808 | 0.41 |
| NTC-full (ours) | 0.799 | 100% | 0.808 | 0.48 |
| Entropy (EAT) | 0.672 | 86% | 0.617 | 1.00 |
| Smoothed confidence | 0.601 | 100% | 0.533 | 1.00 |
| Confidence (DEER-λ) | 0.565 | 100% | 0.417 | 1.00 |
## gpqa_diamond / Qwen3-4B — held-out n=119, vanilla accuracy 0.538

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Confidence (DEER-λ) | 0.511 | 100% | 0.504 | 0.95 |
| Smoothed confidence | 0.503 | 100% | 0.513 | 0.94 |
| NTC-v2 (fusion) | 0.499 | 100% | 0.487 | 1.00 |
| Answer agreement | 0.498 | 100% | 0.487 | 1.00 |
| Entropy (EAT) | 0.475 | 100% | 0.454 | 0.93 |
| NTC-full (ours) | 0.463 | 100% | 0.420 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Confidence (DEER-λ) | 0.478 | 100% | 0.445 | 1.00 |
| NTC-full (ours) | 0.463 | 100% | 0.420 | 1.00 |
| Entropy (EAT) | 0.444 | 100% | 0.429 | 1.00 |
| Answer agreement | 0.437 | 100% | 0.420 | 1.00 |
| NTC-v2 (fusion) | 0.437 | 100% | 0.420 | 1.00 |
| Smoothed confidence | 0.430 | 100% | 0.412 | 1.00 |
## gpqa_diamond / Qwen3-8B — held-out n=119, vanilla accuracy 0.563

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.507 | 100% | 0.479 | 1.00 |
| NTC-v2 (fusion) | 0.503 | 100% | 0.479 | 0.98 |
| Confidence (DEER-λ) | 0.491 | 100% | 0.412 | 0.87 |
| Smoothed confidence | 0.478 | 100% | 0.429 | 0.94 |
| Entropy (EAT) | 0.471 | 100% | 0.395 | 0.96 |
| NTC-full (ours) | 0.450 | 100% | 0.378 | 0.97 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.498 | 100% | 0.479 | 1.00 |
| NTC-v2 (fusion) | 0.489 | 100% | 0.479 | 1.00 |
| Confidence (DEER-λ) | 0.475 | 100% | 0.412 | 0.87 |
| Entropy (EAT) | 0.462 | 100% | 0.395 | 0.96 |
| NTC-full (ours) | 0.450 | 100% | 0.378 | 0.97 |
| Smoothed confidence | 0.425 | 100% | 0.370 | 0.94 |
## mmlu_pro / Qwen3-4B — held-out n=120, vanilla accuracy 0.683

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.665 | 100% | 0.617 | 0.63 |
| NTC-v2 (fusion) | 0.664 | 100% | 0.625 | 0.75 |
| Entropy (EAT) | 0.637 | 100% | 0.583 | 0.76 |
| Smoothed confidence | 0.617 | 100% | 0.592 | 0.98 |
| NTC-full (ours) | 0.608 | 100% | 0.550 | 1.00 |
| Confidence (DEER-λ) | 0.599 | 100% | 0.575 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| Answer agreement | 0.650 | 100% | 0.617 | 1.00 |
| NTC-v2 (fusion) | 0.650 | 100% | 0.617 | 1.00 |
| Entropy (EAT) | 0.630 | 100% | 0.583 | 1.00 |
| NTC-full (ours) | 0.608 | 100% | 0.550 | 1.00 |
| Smoothed confidence | 0.593 | 100% | 0.592 | 1.00 |
| Confidence (DEER-λ) | 0.580 | 100% | 0.575 | 1.00 |
## mmlu_pro / Qwen3-8B — held-out n=120, vanilla accuracy 0.733

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.724 | 100% | 0.700 | 0.62 |
| Answer agreement | 0.708 | 100% | 0.642 | 0.66 |
| Smoothed confidence | 0.686 | 100% | 0.683 | 0.91 |
| Entropy (EAT) | 0.671 | 100% | 0.625 | 0.94 |
| NTC-full (ours) | 0.663 | 100% | 0.642 | 0.91 |
| Confidence (DEER-λ) | 0.632 | 100% | 0.558 | 0.98 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.696 | 100% | 0.700 | 1.00 |
| Answer agreement | 0.693 | 100% | 0.642 | 1.00 |
| Smoothed confidence | 0.674 | 100% | 0.683 | 1.00 |
| NTC-full (ours) | 0.663 | 100% | 0.642 | 0.91 |
| Entropy (EAT) | 0.651 | 100% | 0.625 | 1.00 |
| Confidence (DEER-λ) | 0.629 | 100% | 0.558 | 1.00 |
## aime24 / Qwen3-4B — held-out n=18, vanilla accuracy 0.667

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.611 | 57% | — | 0.90 |
| Smoothed confidence | 0.540 | 100% | 0.444 | 0.97 |
| NTC-v2 (fusion) | 0.508 | 100% | 0.444 | 0.96 |
| Confidence (DEER-λ) | 0.476 | 100% | 0.333 | 0.90 |
| Answer agreement | 0.405 | 100% | 0.222 | 1.00 |
| Entropy (EAT) | 0.389 | 100% | 0.222 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.611 | 57% | — | 0.90 |
| Smoothed confidence | 0.492 | 100% | 0.389 | 0.97 |
| Confidence (DEER-λ) | 0.460 | 100% | 0.333 | 0.90 |
| Entropy (EAT) | 0.444 | 71% | — | 1.00 |
| NTC-v2 (fusion) | 0.413 | 100% | 0.222 | 0.96 |
| Answer agreement | 0.397 | 100% | 0.222 | 1.00 |
## aime25 / Qwen3-4B — held-out n=18, vanilla accuracy 0.500

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.489 | 71% | — | 0.88 |
| Smoothed confidence | 0.478 | 71% | — | 0.72 |
| NTC-v2 (fusion) | 0.437 | 100% | 0.278 | 0.71 |
| Answer agreement | 0.429 | 100% | 0.278 | 0.83 |
| Confidence (DEER-λ) | 0.413 | 100% | 0.278 | 0.75 |
| Entropy (EAT) | 0.389 | 86% | 0.278 | 0.92 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-full (ours) | 0.489 | 71% | — | 0.88 |
| Answer agreement | 0.429 | 100% | 0.278 | 0.83 |
| NTC-v2 (fusion) | 0.429 | 100% | 0.278 | 0.83 |
| Smoothed confidence | 0.422 | 71% | — | 1.00 |
| Entropy (EAT) | 0.370 | 86% | 0.278 | 0.92 |
| Confidence (DEER-λ) | 0.341 | 100% | 0.167 | 1.00 |
## math500 / DeepSeek-R1-Distill-Qwen-7B — held-out n=120, vanilla accuracy 0.892

**ORACLE knob (baselines flattered)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.792 | 86% | 0.642 | 0.89 |
| NTC-full (ours) | 0.769 | 86% | 0.625 | 1.00 |
| Answer agreement | 0.761 | 86% | 0.625 | 1.00 |
| Smoothed confidence | 0.721 | 100% | 0.592 | 1.00 |
| Entropy (EAT) | 0.721 | 86% | 0.592 | 1.00 |
| Confidence (DEER-λ) | 0.690 | 100% | 0.500 | 1.00 |

**DEPLOYABLE knob (what ships)**

| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |
|---|---|---|---|---|
| NTC-v2 (fusion) | 0.781 | 86% | 0.625 | 0.89 |
| NTC-full (ours) | 0.769 | 86% | 0.625 | 1.00 |
| Answer agreement | 0.743 | 86% | 0.625 | 1.00 |
| Entropy (EAT) | 0.703 | 86% | 0.592 | 1.00 |
| Smoothed confidence | 0.690 | 100% | 0.592 | 1.00 |
| Confidence (DEER-λ) | 0.680 | 100% | 0.500 | 1.00 |

## Cross-setting aggregate (deployable protocol)

A deployer chooses one method for a workload mix and cannot know which benchmark arrives next, so the decision-relevant summary is the aggregate over settings, not the per-setting winner.

| method | mean AUCC | min AUCC | worst Δ accuracy (pts) |
|---|---|---|---|
| NTC-full (ours) | 0.674 | 0.450 | -3.3 |
| NTC-v2 (fusion) | 0.662 | 0.413 | -11.8 |
| Answer agreement | 0.661 | 0.397 | -11.8 |
| Entropy (EAT) | 0.617 | 0.370 | -10.9 |
| Smoothed confidence | 0.597 | 0.422 | -12.6 |
| Confidence (DEER-λ) | 0.586 | 0.341 | -16.7 |

