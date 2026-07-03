# NTC — Final Results (calibrated, held-out, multi-seed)


## Qwen/Qwen3-4B · math500 (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.689 ± 0.023 | 0.0 ± 0.0 |
| DEER | 0.635 ± 0.056 | 15.0 ± 15.0 |
| EAT | 0.653 ± 0.034 | 25.0 ± 5.5 |
| NTC-conf | 0.613 ± 0.045 | 21.5 ± 15.6 |
| AGREE | 0.679 ± 0.020 | 50.6 ± 2.0 |
| NTC-v2 | 0.648 ± 0.052 | 54.3 ± 4.1 |
| **NTC-full** | 0.648 ± 0.052 | 54.3 ± 4.1 |
| NTC-full incl. probe overhead | — | 52.7 ± 4.2 |

## Qwen/Qwen3-4B · gsm8k (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.877 ± 0.014 | 0.0 ± 0.0 |
| DEER | 0.853 ± 0.023 | 35.0 ± 19.7 |
| EAT | 0.898 ± 0.011 | 42.6 ± 2.0 |
| NTC-conf | 0.907 ± 0.014 | 60.1 ± 1.8 |
| AGREE | 0.896 ± 0.039 | 63.6 ± 6.2 |
| NTC-v2 | 0.880 ± 0.029 | 69.1 ± 0.8 |
| **NTC-full** | 0.880 ± 0.029 | 69.1 ± 0.8 |
| NTC-full incl. probe overhead | — | 67.9 ± 0.9 |

## Qwen/Qwen3-4B · gpqa_diamond (n=198, 10 seeds, eval n=119)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.497 ± 0.028 | 0.0 ± 0.0 |
| DEER | 0.461 ± 0.048 | 26.0 ± 12.2 |
| EAT | 0.454 ± 0.044 | 15.9 ± 13.3 |
| NTC-conf | 0.471 ± 0.022 | 23.9 ± 7.8 |
| AGREE | 0.341 ± 0.032 | 66.8 ± 0.9 |
| NTC-v2 | 0.342 ± 0.026 | 69.3 ± 2.2 |
| **NTC-full** | 0.449 ± 0.053 | 30.9 ± 11.3 |
| NTC-full incl. probe overhead | — | 29.4 ± 11.5 |

## Qwen/Qwen3-1.7B · math500 (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.657 ± 0.035 | 0.0 ± 0.0 |
| DEER | 0.588 ± 0.045 | 13.1 ± 16.8 |
| EAT | 0.613 ± 0.039 | 18.2 ± 6.9 |
| NTC-conf | 0.578 ± 0.040 | 27.9 ± 17.4 |
| AGREE | 0.609 ± 0.054 | 52.5 ± 6.3 |
| NTC-v2 | 0.594 ± 0.056 | 54.4 ± 5.9 |
| **NTC-full** | 0.578 ± 0.042 | 50.3 ± 13.7 |
| NTC-full incl. probe overhead | — | 48.4 ± 14.3 |

---
## Limitations (stated for the paper)

* Generation uses a single sampling seed (temp 0.6, seed 42); the reported ± std is over 10 calibration splits, not generation seeds.
* Token counts follow the total-generated convention (thinking + emitted answer); the overhead-inclusive row adds all trial-answer probe tokens actually paid online.
* GPQA-Diamond n=198; AIME excluded (n=30/set requires avg@16 for meaningful comparison, out of compute scope).
