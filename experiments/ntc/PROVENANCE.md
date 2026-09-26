# Provenance of the probe files

Settings read from the arguments stored in each probe file. `T` is the sampling temperature of the thinking pass (probes are always greedy); `instruction` records whether the answer-format instruction was appended to each question; `options` gives the GPQA-Diamond option order. All files were generated with vLLM on single NVIDIA L40S GPUs.

| probe file | model | benchmark | items | budget | checkpoints | seed | T | instruction | options |
|---|---|---|---|---|---|---|---|---|---|
| `dens2x_math500_Qwen3-4B.json` | Qwen3-4B | math500 | 500 | 16384 | every 128 tok, at most 20 | 42 | 0 | yes | n/a |
| `dens4x_math500_Qwen3-4B.json` | Qwen3-4B | math500 | 500 | 16384 | every 64 tok, at most 40 | 42 | 0 | yes | n/a |
| `h2h2_aime24_Qwen3-4B.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | n/a |
| `h2h2_aime24_Qwen3-8B.json` | Qwen3-8B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | n/a |
| `h2h2_gpqa_Qwen3-4B.json` | Qwen3-4B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | source order |
| `h2h2_gpqa_Qwen3-8B.json` | Qwen3-8B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | source order |
| `h2h2_math500_Qwen3-4B.json` | Qwen3-4B | math500 | 500 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | n/a |
| `h2h2_math500_Qwen3-8B.json` | Qwen3-8B | math500 | 500 | 16384 | every 256 tok, at most 10 | 42 | 0 | yes | n/a |
| `w1_aime24_Qwen3-4B_s100.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 100 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s101.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 101 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s102.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 102 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s103.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 103 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s104.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 104 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s105.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 105 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s106.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 106 | 0.6 | no | n/a |
| `w1_aime24_Qwen3-4B_s107.json` | Qwen3-4B | aime24 | 30 | 16384 | every 256 tok, at most 10 | 107 | 0.6 | no | n/a |
| `w1_aime25_Qwen3-4B_s100.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 100 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s101.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 101 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s102.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 102 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s103.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 103 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s104.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 104 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s105.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 105 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s106.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 106 | 0.6 | yes | n/a |
| `w1_aime25_Qwen3-4B_s107.json` | Qwen3-4B | aime25 | 30 | 16384 | every 256 tok, at most 10 | 107 | 0.6 | yes | n/a |
| `w1_gsm8k_Qwen3-4B.json` | Qwen3-4B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_gsm8k_Qwen3-4B_s43.json` | Qwen3-4B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 43 | 0.6 | no | n/a |
| `w1_gsm8k_Qwen3-4B_s44.json` | Qwen3-4B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 44 | 0.6 | no | n/a |
| `w1_gsm8k_Qwen3-8B.json` | Qwen3-8B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_gsm8k_Qwen3-8B_s43.json` | Qwen3-8B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 43 | 0.6 | no | n/a |
| `w1_gsm8k_Qwen3-8B_s44.json` | Qwen3-8B | gsm8k | 200 | 4096 | every 256 tok, at most 10 | 44 | 0.6 | no | n/a |
| `w1_math500_DeepSeek-R1-Distill-Qwen-7B.json` | DeepSeek-R1-Distill-Qwen-7B | math500 | 200 | 16384 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_math500_Qwen3-1.7B.json` | Qwen3-1.7B | math500 | 200 | 8192 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_math500_Qwen3-1.7B_s43.json` | Qwen3-1.7B | math500 | 200 | 8192 | every 256 tok, at most 10 | 43 | 0.6 | no | n/a |
| `w1_math500_Qwen3-1.7B_s44.json` | Qwen3-1.7B | math500 | 200 | 8192 | every 256 tok, at most 10 | 44 | 0.6 | no | n/a |
| `w1_math500_Qwen3-4B.json` | Qwen3-4B | math500 | 200 | 8192 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_math500_Qwen3-4B_s43.json` | Qwen3-4B | math500 | 200 | 8192 | every 256 tok, at most 10 | 43 | 0.6 | no | n/a |
| `w1_math500_Qwen3-4B_s44.json` | Qwen3-4B | math500 | 200 | 8192 | every 256 tok, at most 10 | 44 | 0.6 | no | n/a |
| `w1_math500_Qwen3-8B.json` | Qwen3-8B | math500 | 200 | 8192 | every 256 tok, at most 10 | 42 | 0.6 | no | n/a |
| `w1_math500_Qwen3-8B_s43.json` | Qwen3-8B | math500 | 200 | 8192 | every 256 tok, at most 10 | 43 | 0.6 | no | n/a |
| `w1_math500_Qwen3-8B_s44.json` | Qwen3-8B | math500 | 200 | 8192 | every 256 tok, at most 10 | 44 | 0.6 | no | n/a |
| `w1_mmlupro_Qwen3-4B_s42.json` | Qwen3-4B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 42 | 0.6 | yes | n/a |
| `w1_mmlupro_Qwen3-4B_s43.json` | Qwen3-4B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 43 | 0.6 | yes | n/a |
| `w1_mmlupro_Qwen3-4B_s44.json` | Qwen3-4B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 44 | 0.6 | yes | n/a |
| `w1_mmlupro_Qwen3-8B_s42.json` | Qwen3-8B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 42 | 0.6 | yes | n/a |
| `w1_mmlupro_Qwen3-8B_s43.json` | Qwen3-8B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 43 | 0.6 | yes | n/a |
| `w1_mmlupro_Qwen3-8B_s44.json` | Qwen3-8B | mmlu_pro | 200 | 16384 | every 256 tok, at most 10 | 44 | 0.6 | yes | n/a |
| `w1sh_gpqa_Qwen3-4B.json` | Qwen3-4B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 42 | 0.6 | yes | shuffled |
| `w1sh_gpqa_Qwen3-4B_s43.json` | Qwen3-4B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 43 | 0.6 | yes | shuffled |
| `w1sh_gpqa_Qwen3-4B_s44.json` | Qwen3-4B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 44 | 0.6 | yes | shuffled |
| `w1sh_gpqa_Qwen3-8B.json` | Qwen3-8B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 42 | 0.6 | yes | shuffled |
| `w1sh_gpqa_Qwen3-8B_s43.json` | Qwen3-8B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 43 | 0.6 | yes | shuffled |
| `w1sh_gpqa_Qwen3-8B_s44.json` | Qwen3-8B | gpqa_diamond | 198 | 16384 | every 256 tok, at most 10 | 44 | 0.6 | yes | shuffled |

## Regeneration commands

Each command reproduces one file's generation settings; it needs a GPU and vLLM (see README).

```bash
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 500 --max-think 16384 --probe-every 128 --max-probes 20 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/dens2x_math500_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 500 --max-think 16384 --probe-every 64 --max-probes 40 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/dens4x_math500_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/h2h2_aime24_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/h2h2_aime24_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --source-order --out experiments/ntc/h2h2_gpqa_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --source-order --out experiments/ntc/h2h2_gpqa_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 500 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/h2h2_math500_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark math500 --limit 500 --max-think 16384 --max-model-len 24576 --seed 42 --temperature 0.0 --out experiments/ntc/h2h2_math500_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 100 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s100.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 101 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s101.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 102 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s102.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 103 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s103.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 104 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s104.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 105 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s105.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 106 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s106.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 --limit 30 --max-think 16384 --max-model-len 24576 --seed 107 --no-instruction --out experiments/ntc/w1_aime24_Qwen3-4B_s107.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 100 --out experiments/ntc/w1_aime25_Qwen3-4B_s100.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 101 --out experiments/ntc/w1_aime25_Qwen3-4B_s101.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 102 --out experiments/ntc/w1_aime25_Qwen3-4B_s102.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 103 --out experiments/ntc/w1_aime25_Qwen3-4B_s103.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 104 --out experiments/ntc/w1_aime25_Qwen3-4B_s104.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 105 --out experiments/ntc/w1_aime25_Qwen3-4B_s105.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 106 --out experiments/ntc/w1_aime25_Qwen3-4B_s106.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --max-model-len 24576 --seed 107 --out experiments/ntc/w1_aime25_Qwen3-4B_s107.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 42 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 43 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-4B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 44 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-4B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 42 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 43 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-8B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gsm8k --limit 200 --max-think 4096 --max-model-len 12288 --seed 44 --no-instruction --out experiments/ntc/w1_gsm8k_Qwen3-8B_s44.json
python scripts/ntc_w1_thinking.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --benchmark math500 --limit 200 --max-think 16384 --max-model-len 24576 --seed 42 --no-instruction --out experiments/ntc/w1_math500_DeepSeek-R1-Distill-Qwen-7B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-1.7B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 42 --no-instruction --out experiments/ntc/w1_math500_Qwen3-1.7B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-1.7B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 43 --no-instruction --out experiments/ntc/w1_math500_Qwen3-1.7B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-1.7B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 44 --no-instruction --out experiments/ntc/w1_math500_Qwen3-1.7B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 42 --no-instruction --out experiments/ntc/w1_math500_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 43 --no-instruction --out experiments/ntc/w1_math500_Qwen3-4B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 44 --no-instruction --out experiments/ntc/w1_math500_Qwen3-4B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 42 --no-instruction --out experiments/ntc/w1_math500_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 43 --no-instruction --out experiments/ntc/w1_math500_Qwen3-8B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark math500 --limit 200 --max-think 8192 --max-model-len 16384 --seed 44 --no-instruction --out experiments/ntc/w1_math500_Qwen3-8B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 42 --out experiments/ntc/w1_mmlupro_Qwen3-4B_s42.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 43 --out experiments/ntc/w1_mmlupro_Qwen3-4B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 44 --out experiments/ntc/w1_mmlupro_Qwen3-4B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 42 --out experiments/ntc/w1_mmlupro_Qwen3-8B_s42.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 43 --out experiments/ntc/w1_mmlupro_Qwen3-8B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark mmlu_pro --limit 200 --max-think 16384 --max-model-len 24576 --seed 44 --out experiments/ntc/w1_mmlupro_Qwen3-8B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 42 --out experiments/ntc/w1sh_gpqa_Qwen3-4B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 43 --out experiments/ntc/w1sh_gpqa_Qwen3-4B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 44 --out experiments/ntc/w1sh_gpqa_Qwen3-4B_s44.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 42 --out experiments/ntc/w1sh_gpqa_Qwen3-8B.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 43 --out experiments/ntc/w1sh_gpqa_Qwen3-8B_s43.json
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-8B --benchmark gpqa_diamond --limit 198 --max-think 16384 --max-model-len 24576 --seed 44 --out experiments/ntc/w1sh_gpqa_Qwen3-8B_s44.json
```
