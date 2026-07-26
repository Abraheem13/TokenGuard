# Probe-overhead accounting under three deployment regimes
(prefill charged at w=0.2 decode-token equivalents; savings are % vs full thinking)

| model | benchmark | policy | acc | KV-fork | prefix-cache | black-box |
|---|---|---|---|---|---|---|
| Qwen3-4B | gsm8k | DEER{'lam': 0.95} | 0.845 | +56.6% | +56.2% | +27.3% |
| Qwen3-4B | gsm8k | EAT{'delta': 0.001} | 0.895 | +28.0% | +27.4% | -28.2% |
| Qwen3-4B | gsm8k | NTC-conf{'theta': 0.95} | 0.890 | +17.0% | +16.3% | -51.1% |
| Qwen3-4B | gsm8k | AGREE{'m': 3} | 0.930 | +56.0% | +55.5% | +33.6% |
| Qwen3-4B | gsm8k | NTC-v2{'m': 2, 'theta': 0.7} | 0.905 | +67.2% | +66.9% | +53.2% |
| Qwen3-8B | gsm8k | DEER{'lam': 0.95} | 0.860 | +14.7% | +14.0% | -51.0% |
| Qwen3-8B | gsm8k | EAT{'delta': 0.001} | 0.860 | +29.4% | +28.8% | -24.3% |
| Qwen3-8B | gsm8k | NTC-conf{'theta': 0.95} | 0.855 | -0.7% | -1.5% | -82.3% |
| Qwen3-8B | gsm8k | AGREE{'m': 3} | 0.920 | +56.4% | +56.0% | +34.6% |
| Qwen3-8B | gsm8k | NTC-v2{'m': 2, 'theta': 0.7} | 0.905 | +67.9% | +67.6% | +55.1% |
| Qwen3-4B | math500 | DEER{'lam': 0.95} | 0.655 | +35.4% | +35.1% | -16.8% |
| Qwen3-4B | math500 | EAT{'delta': 0.001} | 0.730 | +17.3% | +17.0% | -52.7% |
| Qwen3-4B | math500 | NTC-conf{'theta': 0.95} | 0.715 | +8.8% | +8.3% | -70.8% |
| Qwen3-4B | math500 | AGREE{'m': 3} | 0.730 | +49.3% | +49.0% | +15.4% |
| Qwen3-4B | math500 | NTC-v2{'m': 2, 'theta': 0.7} | 0.650 | +57.7% | +57.5% | +29.8% |
| Qwen3-8B | math500 | DEER{'lam': 0.95} | 0.650 | +17.8% | +17.4% | -53.3% |
| Qwen3-8B | math500 | EAT{'delta': 0.001} | 0.710 | +15.8% | +15.4% | -57.6% |
| Qwen3-8B | math500 | NTC-conf{'theta': 0.95} | 0.680 | +2.3% | +1.8% | -85.1% |
| Qwen3-8B | math500 | AGREE{'m': 3} | 0.780 | +50.0% | +49.8% | +17.4% |
| Qwen3-8B | math500 | NTC-v2{'m': 2, 'theta': 0.7} | 0.735 | +57.3% | +57.1% | +29.7% |
| Qwen3-4B | gpqa_diamond | DEER{'lam': 0.95} | 0.500 | +28.4% | +28.2% | -37.8% |
| Qwen3-4B | gpqa_diamond | EAT{'delta': 0.001} | 0.500 | +22.8% | +22.6% | -46.3% |
| Qwen3-4B | gpqa_diamond | NTC-conf{'theta': 0.95} | 0.515 | +8.1% | +7.8% | -80.4% |
| Qwen3-4B | gpqa_diamond | AGREE{'m': 3} | 0.475 | +67.5% | +67.4% | +53.2% |
| Qwen3-4B | gpqa_diamond | NTC-v2{'m': 2, 'theta': 0.7} | 0.444 | +72.4% | +72.3% | +57.2% |
| Qwen3-8B | gpqa_diamond | DEER{'lam': 0.95} | 0.561 | +1.8% | +1.6% | -92.0% |
| Qwen3-8B | gpqa_diamond | EAT{'delta': 0.001} | 0.505 | +21.4% | +21.2% | -49.5% |
| Qwen3-8B | gpqa_diamond | NTC-conf{'theta': 0.95} | 0.551 | -0.9% | -1.1% | -97.6% |
| Qwen3-8B | gpqa_diamond | AGREE{'m': 3} | 0.424 | +67.4% | +67.3% | +52.4% |
| Qwen3-8B | gpqa_diamond | NTC-v2{'m': 2, 'theta': 0.7} | 0.449 | +49.5% | +49.4% | +8.4% |

Interpretation: our headline savings assume the KV-fork regime (an engine that forks and resumes). Under prefix-cache the cost is nearly identical because only the short cue is re-prefilled; under a pure black-box API that re-sends the prefix at every checkpoint, probing can erase the savings entirely.
