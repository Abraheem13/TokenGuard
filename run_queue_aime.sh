#!/usr/bin/env bash
# AIME24 avg@8: 8 sampling seeds x 30 problems. Aggregate with ntc_genseed_agg.
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"
for seed in 100 101 102 103 104 105 106 107; do
  f="/tmp/job_aime_s${seed}.sbatch"
  cat > "$f" << EOF
#!/bin/bash
#SBATCH --job-name=aime${seed}
#SBATCH --partition=${PART}
#SBATCH --time=3:00:00
#SBATCH --output=experiments/ntc/slurm-%x-%j.log
cd ~/TokenGuard
source .venv/bin/activate
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark aime24 \
  --limit 30 --max-think 16384 --batch 8 --tp-size 1 --max-model-len 24576 \
  --seed ${seed}
echo "JOB DONE"
EOF
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f"; else sbatch "$f"; fi
done
echo "AIME avg@8 queued. Aggregate later with:"
echo "  python scripts/ntc_genseed_agg.py \$(for s in 100 101 102 103 104 105 106 107; do echo --probes experiments/ntc/w1_aime24_Qwen3-4B_s\$s.json; done) --tag aime24_avg8"
