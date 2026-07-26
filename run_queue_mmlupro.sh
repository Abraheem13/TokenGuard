#!/usr/bin/env bash
# MMLU-Pro (|A|=10) — falsifiable test of Prop. 2, 3 gen-seeds x {4B, 8B}.
set -e
PART="gpu-2c-l40s-1g"; DRY="${1:-}"
T="python scripts/ntc_w1_thinking.py"
submit () {
  local name="$1" hours="$2" cmd="$3"; local f="/tmp/job_${name}.sbatch"
  cat > "$f" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PART}
#SBATCH --time=${hours}:00:00
#SBATCH --output=experiments/ntc/slurm-%x-%j.log
cd ~/TokenGuard
source .venv/bin/activate
${cmd}
echo "JOB DONE ${name}"
EOF
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f"; echo; else sbatch "$f"; fi
}
for M in 4B 8B; do for S in 42 43 44; do
  submit mmlu_${M}_s${S} 7 "$T --model Qwen/Qwen3-${M} --benchmark mmlu_pro --limit 200 --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --seed ${S} --out experiments/ntc/w1_mmlupro_Qwen3-${M}_s${S}.json"
done; done
echo "MMLU-Pro queued (6 jobs)."
