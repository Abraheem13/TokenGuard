#!/usr/bin/env bash
# Day-5 queue: GPQA shuffled validation (2 jobs) + AIME25 avg@8 (8 jobs).
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

# GPQA shuffled (temp 0.6, seed 42) — position-bias validation, both models
for M in 4B 8B; do
  submit gpqash_${M} 7 "$T --model Qwen/Qwen3-${M} --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --seed 42 --out experiments/ntc/w1sh_gpqa_Qwen3-${M}.json"
done
# AIME25 avg@8 (4B)
for seed in 100 101 102 103 104 105 106 107; do
  submit a25_s${seed} 3 "$T --model Qwen/Qwen3-4B --benchmark aime25 --limit 30 --max-think 16384 --batch 8 --tp-size 1 --max-model-len 24576 --seed ${seed}"
done
echo "Day-5 queued (10 jobs)."
