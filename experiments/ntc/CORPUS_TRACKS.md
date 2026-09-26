# Corpus by experimental track

Tracks: `primary` = `w1_*` and `w1sh_*` (GPQA-Diamond with shuffled options); `head-to-head` = `h2h2_*`; `density` = `dens*`. `items` is the number of questions per file; tokens per trace is the mean of `n_total_tokens`; truncation is the share of a file's traces that reach the thinking budget, as a range over files.

## primary

| benchmark | files | pool | items | budget | traces | probes | tokens per trace | truncation |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 6 | 1319 | 200 | 4096 | 1200 | 9734 | 2164 | 10.0 to 15.5% |
| MATH-500 | 10 | 500 | 200 | 8192/16384 | 2000 | 18575 | 4278 | 3.0 to 23.0% |
| GPQA-Diamond | 6 | 198 | 198 | 16384 | 1188 | 11593 | 8983 | 4.5 to 8.6% |
| MMLU-Pro | 6 | 12032 | 200 | 16384 | 1200 | 10110 | 5439 | 5.0 to 7.5% |
| AIME-2024 | 8 | 30 | 30 | 16384 | 240 | 2400 | 11549 | 23.3 to 40.0% |
| AIME-2025 | 8 | 30 | 30 | 16384 | 240 | 2400 | 13288 | 46.7 to 60.0% |
| **total** | | | | | **6068** | **54812** | | |

## head-to-head

| benchmark | files | pool | items | budget | traces | probes | tokens per trace | truncation |
|---|---|---|---|---|---|---|---|---|
| MATH-500 | 2 | 500 | 500 | 16384 | 1000 | 9357 | 5196 | 5.4 to 6.2% |
| GPQA-Diamond | 2 | 198 | 198 | 16384 | 396 | 3833 | 9842 | 18.2 to 23.7% |
| AIME-2024 | 2 | 30 | 30 | 16384 | 60 | 585 | 12052 | 40.0 to 40.0% |
| **total** | | | | | **1456** | **13775** | | |

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
| reasoning traces | 8524 |
| graded trial answers | 92457 |

