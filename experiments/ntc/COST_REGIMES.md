# Net token saving under three serving regimes

Prefill is charged at w = 0.2 decode-token equivalents. Each rule runs at the middle value of its parameter grid on all items; savings are relative to full generation, and a negative saving is a net cost.

| model | benchmark | policy | acc | KV-fork | prefix-cache | black-box |
|---|---|---|---|---|---|---|
| Qwen3-4B | gsm8k | NEVER-HALT{} | 0.885 | -6.4% | -7.3% | -95.7% |
| Qwen3-4B | gsm8k | DEER{'lam': 0.95} | 0.845 | +56.6% | +56.2% | +27.3% |
| Qwen3-4B | gsm8k | EAT{'delta': 0.001} | 0.895 | +28.0% | +27.4% | -28.2% |
| Qwen3-4B | gsm8k | NTC-conf{'theta': 0.95} | 0.890 | +17.0% | +16.3% | -51.1% |
| Qwen3-4B | gsm8k | AGREE{'m': 3} | 0.930 | +56.0% | +55.5% | +33.6% |
| Qwen3-4B | gsm8k | NTC-v2{'m': 3, 'theta': 0.7} | 0.935 | +55.6% | +55.2% | +32.8% |
| Qwen3-8B | gsm8k | NEVER-HALT{} | 0.835 | -6.9% | -7.7% | -95.2% |
| Qwen3-8B | gsm8k | DEER{'lam': 0.95} | 0.860 | +14.7% | +14.0% | -51.0% |
| Qwen3-8B | gsm8k | EAT{'delta': 0.001} | 0.860 | +29.4% | +28.8% | -24.3% |
| Qwen3-8B | gsm8k | NTC-conf{'theta': 0.95} | 0.855 | -0.7% | -1.5% | -82.3% |
| Qwen3-8B | gsm8k | AGREE{'m': 3} | 0.920 | +56.4% | +56.0% | +34.6% |
| Qwen3-8B | gsm8k | NTC-v2{'m': 3, 'theta': 0.7} | 0.930 | +55.5% | +55.1% | +32.8% |
| Qwen3-4B | math500 | NEVER-HALT{} | 0.785 | -3.4% | -3.9% | -94.6% |
| Qwen3-4B | math500 | DEER{'lam': 0.95} | 0.655 | +35.4% | +35.1% | -16.8% |
| Qwen3-4B | math500 | EAT{'delta': 0.001} | 0.730 | +17.3% | +17.0% | -52.7% |
| Qwen3-4B | math500 | NTC-conf{'theta': 0.95} | 0.715 | +8.8% | +8.3% | -70.8% |
| Qwen3-4B | math500 | AGREE{'m': 3} | 0.730 | +49.3% | +49.0% | +15.4% |
| Qwen3-4B | math500 | NTC-v2{'m': 3, 'theta': 0.7} | 0.755 | +45.3% | +45.0% | +6.9% |
| Qwen3-8B | math500 | NEVER-HALT{} | 0.750 | -4.4% | -4.9% | -98.2% |
| Qwen3-8B | math500 | DEER{'lam': 0.95} | 0.650 | +17.8% | +17.4% | -53.3% |
| Qwen3-8B | math500 | EAT{'delta': 0.001} | 0.710 | +15.8% | +15.4% | -57.6% |
| Qwen3-8B | math500 | NTC-conf{'theta': 0.95} | 0.680 | +2.3% | +1.8% | -85.1% |
| Qwen3-8B | math500 | AGREE{'m': 3} | 0.780 | +50.0% | +49.8% | +17.4% |
| Qwen3-8B | math500 | NTC-v2{'m': 3, 'theta': 0.7} | 0.790 | +45.1% | +44.8% | +6.9% |
| Qwen3-4B | gpqa_diamond | NEVER-HALT{} | 0.515 | -0.9% | -1.2% | -98.4% |
| Qwen3-4B | gpqa_diamond | DEER{'lam': 0.95} | 0.500 | +28.4% | +28.2% | -37.8% |
| Qwen3-4B | gpqa_diamond | EAT{'delta': 0.001} | 0.500 | +22.8% | +22.6% | -46.3% |
| Qwen3-4B | gpqa_diamond | NTC-conf{'theta': 0.95} | 0.515 | +8.1% | +7.8% | -80.4% |
| Qwen3-4B | gpqa_diamond | AGREE{'m': 3} | 0.475 | +67.5% | +67.4% | +53.2% |
| Qwen3-4B | gpqa_diamond | NTC-v2{'m': 3, 'theta': 0.7} | 0.465 | +57.0% | +56.9% | +29.7% |
| Qwen3-8B | gpqa_diamond | NEVER-HALT{} | 0.556 | -1.3% | -1.5% | -98.5% |
| Qwen3-8B | gpqa_diamond | DEER{'lam': 0.95} | 0.561 | +1.8% | +1.6% | -92.0% |
| Qwen3-8B | gpqa_diamond | EAT{'delta': 0.001} | 0.505 | +21.4% | +21.2% | -49.5% |
| Qwen3-8B | gpqa_diamond | NTC-conf{'theta': 0.95} | 0.551 | -0.9% | -1.1% | -97.6% |
| Qwen3-8B | gpqa_diamond | AGREE{'m': 3} | 0.424 | +67.4% | +67.3% | +52.4% |
| Qwen3-8B | gpqa_diamond | NTC-v2{'m': 3, 'theta': 0.7} | 0.505 | +39.4% | +39.2% | -10.6% |
