#!/usr/bin/env bash
# DEER OFFICIAL baseline queue — authors' code (iie-ycx/DEER), default config,
# on our models x datasets. Billing-safe sbatch with hard time limits.
# Run from ~/TokenGuard/external/DEER
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"
DEER_DIR="$HOME/TokenGuard/external/DEER"

submit () {  # name hours model dataset
  local name="$1" hours="$2" model="$3" ds="$4"
  local f="/tmp/job_${name}.sbatch"
  cat > "$f" << EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PART}
#SBATCH --time=${hours}:00:00
#SBATCH --output=${DEER_DIR}/logs/slurm-%x-%j.log
cd ${DEER_DIR}
source ~/TokenGuard/.venv/bin/activate
python vllm-deer-qwen3.py \
  --model_name_or_path ${model} \
  --dataset ${ds} \
  --threshold 0.95 --think_ratio 0.8 --policy avg2 \
  --max_generated_tokens 16000
echo "DEER JOB DONE ${name}"
EOF
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f"; echo;
  else sbatch "$f"; fi
}

mkdir -p "${DEER_DIR}/logs"
# Qwen3-4B (matches our NTC runs) x 3 datasets
submit deer_4b_math  5 Qwen/Qwen3-4B math
submit deer_4b_gpqa  6 Qwen/Qwen3-4B gpqa
submit deer_4b_aime  6 Qwen/Qwen3-4B aime
# Qwen3-8B (the new family-scale) x 3 datasets
submit deer_8b_math  6 Qwen/Qwen3-8B math
submit deer_8b_gpqa  7 Qwen/Qwen3-8B gpqa
submit deer_8b_aime  7 Qwen/Qwen3-8B aime

echo ""
echo "DEER official queued. Monitor: squeue -u \$USER"
echo "Outputs: ${DEER_DIR}/outputs/<model>/<dataset>/*.jsonl"
