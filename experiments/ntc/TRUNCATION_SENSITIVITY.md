# Sensitivity to the scoring of budget-exhausted traces

Calibration tolerance eps = 0.01; 10 calibration/evaluation splits per generation-seed file. `truncated`: share of traces that reach the generation budget; `no answer`: share that end without an answer. `full`: accuracy of full generation as recorded; `full-BF`: budget-forced (an unanswered trace takes its last trial answer). For each rule, `d` and `d-BF` are its accuracy change in points against the two references as evaluated in SLO_ATTAINMENT.md, with its token cut; `fb d-BF` and `fb cut` are the same rule with the forced fallback, calibrated against the budget-forced reference.

| setting | truncated | no answer | full | full-BF | AGREE d | AGREE d-BF | AGREE cut | AGREE fb d-BF | AGREE fb cut | NTC-v2 d | NTC-v2 d-BF | NTC-v2 cut | NTC-v2 fb d-BF | NTC-v2 fb cut | SELECT d | SELECT d-BF | SELECT cut | SELECT fb d-BF | SELECT fb cut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K-4B | 11.2% | 9.3% | 0.869 | 0.922 | +5.2 | -0.1 | 54.9% | +0.2 | 54.4% | +4.3 | -1.1 | 39.7% | +0.5 | 28.8% | +2.5 | -2.9 | 22.2% | -0.1 | 8.6% |
| GSM8K-8B | 14.0% | 11.2% | 0.835 | 0.914 | +8.2 | +0.3 | 57.6% | +0.7 | 57.1% | +7.6 | -0.3 | 52.7% | +0.4 | 9.7% | +5.8 | -2.1 | 46.1% | -0.0 | 5.9% |
| MATH-500-1.7B | 22.0% | 18.3% | 0.736 | 0.796 | -1.1 | -7.0 | 45.4% | -6.9 | 45.4% | +0.1 | -5.8 | 2.5% | -0.4 | 1.0% | +0.0 | -5.9 | -1.8% | -0.4 | 0.5% |
| MATH-500-4B | 15.5% | 13.0% | 0.794 | 0.842 | -5.5 | -10.2 | 49.2% | -10.2 | 49.2% | +0.5 | -4.2 | 12.8% | -0.3 | 14.9% | -0.1 | -4.9 | 3.5% | -0.3 | 8.8% |
| MATH-500-8B | 20.0% | 16.8% | 0.758 | 0.827 | +1.4 | -5.5 | 48.9% | -5.1 | 48.9% | +0.2 | -6.7 | 2.8% | -0.5 | 1.8% | +0.1 | -6.8 | 2.4% | -0.5 | 1.8% |
| MMLU-Pro-4B | 6.3% | 5.0% | 0.705 | 0.726 | -8.2 | -10.3 | 64.5% | -10.3 | 64.5% | +0.1 | -2.0 | 18.3% | -1.7 | 18.3% | -0.2 | -2.3 | 3.3% | -0.9 | 5.7% |
| MMLU-Pro-8B | 5.7% | 4.8% | 0.741 | 0.763 | -7.6 | -9.8 | 64.7% | -9.2 | 64.2% | +0.0 | -2.2 | 2.0% | -0.2 | 2.3% | -0.4 | -2.6 | 5.0% | -0.8 | 6.2% |
| AIME-24-4B | 31.7% | 30.0% | 0.658 | 0.705 | -23.5 | -28.2 | 45.3% | -26.5 | 45.3% | -0.4 | -5.1 | 2.6% | -1.0 | 3.3% | -1.5 | -6.2 | 4.5% | -2.8 | 6.4% |
| AIME-25-4B | 51.2% | 48.8% | 0.459 | 0.536 | -9.4 | -17.1 | 44.9% | -17.1 | 44.9% | +0.7 | -7.0 | 3.4% | +0.0 | -0.3% | -1.0 | -8.8 | 6.7% | -2.2 | 5.6% |

## Head-to-head track: the selection tier on its own evaluation items

Calibration split 0, eps = 0.01, as in H2H_TABLE.md. `full` and `full-BF` are full generation as recorded and budget-forced on the same evaluation items.

| probe file | evaluation items | selection tier | full | full-BF | selected rule |
|---|---|---|---|---|---|
| `h2h2_math500_Qwen3-4B` | 300 | 0.850 | 0.853 | 0.860 | NTC-v2{'m': 2, 'theta': 0.95} |
| `h2h2_math500_Qwen3-8B` | 300 | 0.823 | 0.833 | 0.850 | NTC-v2{'m': 2, 'theta': 0.95} |
| `h2h2_aime24_Qwen3-4B` | 18 | 0.611 | 0.611 | 0.611 | NTC-v2{'m': 2, 'theta': 0.95} |
| `h2h2_aime24_Qwen3-8B` | 18 | 0.556 | 0.556 | 0.667 | NTC-v2{'m': 2, 'theta': 0.9} |
