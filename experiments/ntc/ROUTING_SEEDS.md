# Routing tier — multi-seed results (RouterBench test split)

Seeds: [42, 43, 44]. Contrastive config: w_bce=6.0, epochs=120, proj_dim=256, encoder=Qwen3-0.6B.

| router | AIQ (mean ± std) | APGR (mean ± std) |
|---|---|---|
| matrix-factorization | 0.7479 ± 0.0035 | 0.9270 ± 0.0038 |
| knn-embedding | 0.7490 ± 0.0025 | 0.9294 ± 0.0063 |
| cascade | 0.6341 ± 0.0012 | 0.6900 ± 0.0029 |
| contrastive | 0.7404 ± 0.0021 | 0.9114 ± 0.0098 |
