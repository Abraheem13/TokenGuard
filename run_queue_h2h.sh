#!/usr/bin/env bash
# HEAD-TO-HEAD queue: our harness under DEER-MATCHED conditions
# (n=500 MATH, max-think 16384, greedy temp=0.0, DEER-grader scoring).
# Run from ~/TokenGuard
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"
T="python scripts/ntc_w1_thinking.py"

submit () {
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
echo "JOB DONE ${name}"
EOF
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f"; echo;
  else sbatch "$f"; fi
}

for M in 4B 8B; do
  submit h2h_m500_${M} 8 "$T --model Qwen/Qwen3-${M} --benchmark math500 --limit 500 --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --temperature 0.0 --out experiments/ntc/h2h_math500_Qwen3-${M}.json"
  submit h2h_gpqa_${M} 7 "$T --model Qwen/Qwen3-${M} --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --temperature 0.0 --out experiments/ntc/h2h_gpqa_Qwen3-${M}.json"
  submit h2h_aime_${M} 4 "$T --model Qwen/Qwen3-${M} --benchmark aime24 --limit 30 --max-think 16384 --batch 8 --tp-size 1 --max-model-len 24576 --temperature 0.0 --out experiments/ntc/h2h_aime24_Qwen3-${M}.json"
done
echo ""
echo "H2H queued (6 jobs, DEER-matched). Monitor: squeue -u \$USER"
