# Does the medium tier bind?

NTC-v2 (m=3, theta=0.9) versus AGREE (m=3) on the same traces. `differs` = share of items halting at a different checkpoint; `delayed` = share where the confidence gate postponed the halt; `gate never binds` means the fused policy is exactly its agreement component and should be reported as such.

| setting | items | differs | delayed | identical? |
|---|---|---|---|---|
| math500/Qwen3-4B | 200 | 55.5% | 55.5% | no |
| gsm8k/Qwen3-8B | 200 | 73.0% | 73.0% | no |
| gpqa_diamond/Qwen3-8B | 198 | 100.0% | 100.0% | no |
| mmlu_pro/Qwen3-4B | 200 | 52.5% | 52.5% | no |
