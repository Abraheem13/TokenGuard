# Joint tier: model routing with calibrated halting (math500)

Evaluation n = 120, split seed 0. Small model Qwen/Qwen3-1.7B (rule NTC-v2 {'m': 2, 'theta': 0.7}); large model Qwen/Qwen3-4B (rule AGREE {'m': 3}); router: TF-IDF and logistic regression trained on the 80 calibration items. `tau` is the routing threshold on the predicted probability that the small model is correct; `cost` is mean tokens x model size in billions of parameters; `%small` is the share of queries routed to the small model.

| tau | acc | cost | tokens | %small |
|---|---|---|---|---|
| 0.00 | 0.592 | 2972 | 1748 | 100% |
| 0.05 | 0.592 | 2972 | 1748 | 100% |
| 0.10 | 0.592 | 2972 | 1748 | 100% |
| 0.15 | 0.592 | 2972 | 1748 | 100% |
| 0.20 | 0.592 | 2972 | 1748 | 100% |
| 0.25 | 0.592 | 2972 | 1748 | 100% |
| 0.30 | 0.592 | 2972 | 1748 | 100% |
| 0.35 | 0.592 | 2972 | 1748 | 100% |
| 0.40 | 0.583 | 3071 | 1747 | 98% |
| 0.45 | 0.600 | 3597 | 1775 | 88% |
| 0.50 | 0.633 | 4831 | 1874 | 69% |
| 0.55 | 0.667 | 6348 | 2012 | 45% |
| 0.60 | 0.683 | 6961 | 2019 | 25% |
| 0.65 | 0.700 | 7882 | 2087 | 11% |
| 0.70 | 0.708 | 8369 | 2098 | 1% |
| 0.75 | 0.708 | 8409 | 2102 | 0% |
| 0.80 | 0.708 | 8409 | 2102 | 0% |
| 0.85 | 0.708 | 8409 | 2102 | 0% |
| 0.90 | 0.708 | 8409 | 2102 | 0% |
| 0.95 | 0.708 | 8409 | 2102 | 0% |
| 1.00 | 0.708 | 8409 | 2102 | 0% |

| reference | acc | cost |
|---|---|---|
| large-vanilla | 0.683 | 17511 |
| large-NTC | 0.708 | 8409 |
| small-NTC | 0.592 | 2972 |
