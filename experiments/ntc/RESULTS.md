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
| **NTC-full(e=0.01)** | 0.667 ± 0.021 | 17.8 ± 21.7 |
| **NTC-full(e=0.05)** | 0.670 ± 0.024 | 33.6 ± 17.3 |
| NTC-full(e=0.05) incl. probe overhead | — | 31.3 ± 17.8 |

## Qwen/Qwen3-4B · gsm8k (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.877 ± 0.014 | 0.0 ± 0.0 |
| DEER | 0.862 ± 0.018 | 23.2 ± 12.9 |
| EAT | 0.892 ± 0.017 | 38.8 ± 7.8 |
| NTC-conf | 0.896 ± 0.021 | 47.2 ± 22.4 |
| AGREE | 0.916 ± 0.039 | 59.8 ± 5.1 |
| NTC-v2 | 0.906 ± 0.036 | 63.1 ± 5.8 |
| **NTC-full(e=0.01)** | 0.897 ± 0.030 | 62.7 ± 7.7 |
| **NTC-full(e=0.05)** | 0.886 ± 0.028 | 67.2 ± 4.0 |
| NTC-full(e=0.05) incl. probe overhead | — | 65.8 ± 4.2 |

## Qwen/Qwen3-4B · gpqa_diamond (n=198, 10 seeds, eval n=119)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.497 ± 0.028 | 0.0 ± 0.0 |
| DEER | 0.487 ± 0.030 | 6.3 ± 7.4 |
| EAT | 0.472 ± 0.030 | 9.7 ± 3.7 |
| NTC-conf | 0.482 ± 0.025 | 6.6 ± 7.3 |
| AGREE | 0.341 ± 0.032 | 66.8 ± 0.9 |
| NTC-v2 | 0.342 ± 0.026 | 69.3 ± 2.2 |
| **NTC-full(e=0.01)** | 0.476 ± 0.025 | 9.3 ± 7.7 |
| **NTC-full(e=0.05)** | 0.468 ± 0.025 | 21.0 ± 10.1 |
| NTC-full(e=0.05) incl. probe overhead | — | 19.4 ± 10.3 |

## Qwen/Qwen3-1.7B · math500 (n=200, 10 seeds, eval n=120)

| method | accuracy (mean±std) | token cut % (mean±std) |
|---|---|---|
| vanilla | 0.657 ± 0.035 | 0.0 ± 0.0 |
| DEER | 0.597 ± 0.030 | 10.1 ± 9.7 |
| EAT | 0.637 ± 0.038 | 10.9 ± 2.4 |
| NTC-conf | 0.618 ± 0.036 | 6.7 ± 12.2 |
| AGREE | 0.637 ± 0.033 | 48.5 ± 2.0 |
| NTC-v2 | 0.637 ± 0.033 | 48.5 ± 2.0 |
| **NTC-full(e=0.01)** | 0.613 ± 0.025 | 25.4 ± 22.1 |
| **NTC-full(e=0.05)** | 0.610 ± 0.036 | 31.6 ± 21.4 |
| NTC-full(e=0.05) incl. probe overhead | — | 28.9 ± 22.3 |

---
## Limitations (stated for the paper)

* Generation uses a single sampling seed (temp 0.6, seed 42); the reported ± std is over 10 calibration splits, not generation seeds.
* Token counts follow the total-generated convention (thinking + emitted answer); the overhead-inclusive row adds all trial-answer probe tokens actually paid online.
* GPQA-Diamond n=198; AIME excluded (n=30/set requires avg@16 for meaningful comparison, out of compute scope).
