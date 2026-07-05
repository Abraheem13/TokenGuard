# NTC — Final Results (calibrated, held-out, multi-seed)


## Qwen/Qwen3-4B · math500 (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.689 ± 0.023 | 0.0 ± 0.0 |
| DEER | 0.643 ± 0.038 | 12.6 ± 8.0 |
| EAT | 0.669 ± 0.024 | 16.6 ± 3.4 |
| NTC-conf | 0.659 ± 0.034 | 4.6 ± 8.0 |
| AGREE | 0.679 ± 0.020 | 50.6 ± 2.0 |
| NTC-v2 | 0.679 ± 0.020 | 50.6 ± 2.0 |
| **NTC-full(e=0.01)** | 0.667 ± 0.021 | 25.8 ± 24.0 |
| **NTC-full(e=0.025)** | 0.672 ± 0.024 | 43.7 ± 12.8 |
| NTC-full(e=0.05) incl. probe overhead | — | 41.7 ± 13.2 |

## Qwen/Qwen3-4B · gsm8k (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.877 ± 0.014 | 0.0 ± 0.0 |
| DEER | 0.862 ± 0.018 | 23.2 ± 12.9 |
| EAT | 0.898 ± 0.011 | 42.6 ± 2.0 |
| NTC-conf | 0.907 ± 0.014 | 60.1 ± 1.8 |
| AGREE | 0.908 ± 0.042 | 61.1 ± 5.9 |
| NTC-v2 | 0.888 ± 0.033 | 67.8 ± 3.4 |
| **NTC-full(e=0.01)** | 0.885 ± 0.027 | 68.0 ± 3.1 |
| **NTC-full(e=0.025)** | 0.880 ± 0.029 | 69.1 ± 0.8 |
| NTC-full(e=0.05) incl. probe overhead | — | 67.9 ± 0.9 |

## Qwen/Qwen3-4B · gpqa_diamond (n=198, 10 seeds, eval n=119)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.497 ± 0.028 | 0.0 ± 0.0 |
| DEER | 0.486 ± 0.030 | 8.2 ± 8.8 |
| EAT | 0.472 ± 0.030 | 9.7 ± 3.7 |
| NTC-conf | 0.477 ± 0.025 | 9.6 ± 12.1 |
| AGREE | 0.341 ± 0.032 | 66.8 ± 0.9 |
| NTC-v2 | 0.340 ± 0.026 | 69.3 ± 2.2 |
| **NTC-full(e=0.01)** | 0.476 ± 0.025 | 13.7 ± 12.3 |
| **NTC-full(e=0.025)** | 0.467 ± 0.024 | 20.4 ± 10.8 |
| NTC-full(e=0.05) incl. probe overhead | — | 18.7 ± 11.0 |

## Qwen/Qwen3-1.7B · math500 (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.657 ± 0.035 | 0.0 ± 0.0 |
| DEER | 0.597 ± 0.030 | 10.1 ± 9.7 |
| EAT | 0.637 ± 0.038 | 10.9 ± 2.4 |
| NTC-conf | 0.607 ± 0.031 | 10.4 ± 15.5 |
| AGREE | 0.637 ± 0.033 | 48.5 ± 2.0 |
| NTC-v2 | 0.629 ± 0.037 | 49.5 ± 3.3 |
| **NTC-full(e=0.01)** | 0.605 ± 0.024 | 26.4 ± 23.3 |
| **NTC-full(e=0.025)** | 0.604 ± 0.037 | 36.3 ± 21.0 |
| NTC-full(e=0.05) incl. probe overhead | — | 33.8 ± 21.9 |

---
## Limitations (stated for the paper)

* Generation uses a single sampling seed (temp 0.6, seed 42); the reported ± std is over 10 calibration splits, not generation seeds.
* Token counts follow the total-generated convention (thinking + emitted answer); the overhead-inclusive row adds all trial-answer probe tokens actually paid online.
* GPQA-Diamond n=198; AIME excluded (n=30/set requires avg@16 for meaningful comparison, out of compute scope).
