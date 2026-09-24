# Corpus by experimental track

Track assignment: `primary` = `w1_*` and `w1sh_*` (the unshuffled `w1_gpqa16k_*` runs are superseded by `w1sh_*` and excluded); `head-to-head` = `h2h*`; `density` = `dens*`. Tokens per trace is the mean of `n_total_tokens`; truncation is the share of traces whose generation did not stop of its own accord.

## primary

| benchmark | files | pool | items | budget | traces | probes | tokens per trace | truncation |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 6 | 1319 | 200 | 4096 | 1200 | 9734 | 2164 | 10.0 to 15.5% |
| MATH-500 | 10 | 500 | 200 | 8192/16384 | 2000 | 18575 | 4278 | 3.0 to 23.0% |
| GPQA-Diamond | 7 | 198 | 198 | 8192/16384 | 1386 | 13528 | 8680 | 4.5 to 56.6% |
| MMLU-Pro | 6 | 12032 | 200 | 16384 | 1200 | 10110 | 5439 | 5.0 to 7.5% |
| AIME-2024 | 8 | 30 | 30 | 16384 | 240 | 2400 | 11549 | 23.3 to 40.0% |
| AIME-2025 | 8 | 30 | 30 | 16384 | 240 | 2400 | 13288 | 46.7 to 60.0% |
| **total** | | | | | **6266** | **56747** | | |

## head-to-head

| benchmark | files | pool | items | budget | traces | probes | tokens per trace | truncation |
|---|---|---|---|---|---|---|---|---|
| MATH-500 | 4 | 500 | 500 | 16384 | 2000 | 18844 | 5259 | 5.4 to 7.2% |
| GPQA-Diamond | 4 | 198 | 198 | 16384 | 792 | 7634 | 10088 | 18.2 to 29.3% |
| AIME-2024 | 4 | 30 | 30 | 16384 | 120 | 1171 | 11967 | 30.0 to 43.3% |
| **total** | | | | | **2912** | **27649** | | |

## density

| benchmark | files | pool | items | budget | traces | probes | tokens per trace | truncation |
|---|---|---|---|---|---|---|---|---|
| MATH-500 | 2 | 500 | 500 | 16384 | 1000 | 23870 | 5095 | 6.4 to 6.4% |
| **total** | | | | | **1000** | **23870** | | |

## All tracks

| quantity | value |
|---|---|
| distinct questions in the primary track | 858 |
| distinct questions in all tracks | 1158 |
| reasoning traces | 10178 |
| graded trial answers | 108266 |

