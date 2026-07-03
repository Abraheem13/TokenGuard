#!/usr/bin/env bash
# Q1 GPU queue: generation-seed replicates + DeepSeek cross-family run.
# Submits independent sbatch jobs, each with a HARD time limit (billing-safe).
# Usage:  bash run_queue.sh          (submit all)
#         bash run_queue.sh --dry    (print only)
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"

submit () {  # name, hours, command
  local name="$1" hours="$2" cmd="$3"
  local f="/tmp/job_${name}.sbatch"
  cat > "$f" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PART}
#SBATCH --time=${hours}:00:00
#SBATCH --output=experiments/ntc/slurm-%x-%j.log
cd ~/TokenGuard
source .venv/bin/activate
${cmd}
echo "JOB DONE"
EOF
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f";
  else sbatch "$f"; fi
}

T="python scripts/ntc_w1_thinking.py"

# generation-seed replicates (seeds 43, 44) — same configs as seed-42 canon
submit m500s43  4 "$T --model Qwen/Qwen3-4B  --benchmark math500      --limit 200 --max-think 8192  --batch 16 --tp-size 1 --max-model-len 16384 --seed 43"
submit m500s44  4 "$T --model Qwen/Qwen3-4B  --benchmark math500      --limit 200 --max-think 8192  --batch 16 --tp-size 1 --max-model-len 16384 --seed 44"
submit gsms43   3 "$T --model Qwen/Qwen3-4B  --benchmark gsm8k        --limit 200 --max-think 4096  --batch 16 --tp-size 1 --seed 43"
submit gsms44   3 "$T --model Qwen/Qwen3-4B  --benchmark gsm8k        --limit 200 --max-think 4096  --batch 16 --tp-size 1 --seed 44"
submit gpqs43   6 "$T --model Qwen/Qwen3-4B  --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 16 --tp-size 1 --max-model-len 24576 --seed 43 --out experiments/ntc/w1_gpqa16k_Qwen3-4B_s43.json"
submit gpqs44   6 "$T --model Qwen/Qwen3-4B  --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 16 --tp-size 1 --max-model-len 24576 --seed 44 --out experiments/ntc/w1_gpqa16k_Qwen3-4B_s44.json"
submit m17s43   4 "$T --model Qwen/Qwen3-1.7B --benchmark math500     --limit 200 --max-think 8192  --batch 16 --tp-size 1 --max-model-len 16384 --seed 43"
submit m17s44   4 "$T --model Qwen/Qwen3-1.7B --benchmark math500     --limit 200 --max-think 8192  --batch 16 --tp-size 1 --max-model-len 16384 --seed 44"
# cross-family: DeepSeek-R1-Distill-Qwen-7B (thinking format compatible)
submit r1d7b    6 "$T --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --benchmark math500 --limit 200 --max-think 16384 --batch 8 --tp-size 1 --max-model-len 24576 --seed 42"

echo ""
echo "Submitted. Monitor: squeue -u \$USER   |   logs: experiments/ntc/slurm-*.log"
