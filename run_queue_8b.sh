 #!/usr/bin/env bash
# Qwen3-8B NTC runs (the new model scale) — our thinking+probing harness.
# MATH-500 + GSM8K + GPQA x 3 generation seeds. Billing-safe time limits.
# Run from ~/TokenGuard
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"
T="python scripts/ntc_w1_thinking.py"

submit () {  # name hours args
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

# Qwen3-8B needs more VRAM/time; L40S 48GB handles it. 3 seeds each.
for seed in 42 43 44; do
  submit m500_8b_s${seed}  6 "$T --model Qwen/Qwen3-8B --benchmark math500 --limit 200 --max-think 8192  --batch 12 --tp-size 1 --max-model-len 16384 --seed ${seed}"
  submit gsm_8b_s${seed}   5 "$T --model Qwen/Qwen3-8B --benchmark gsm8k   --limit 200 --max-think 4096  --batch 12 --tp-size 1 --seed ${seed}"
  submit gpqa_8b_s${seed}  8 "$T --model Qwen/Qwen3-8B --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 8 --tp-size 1 --max-model-len 24576 --seed ${seed} --out experiments/ntc/w1_gpqa16k_Qwen3-8B_s${seed}.json"
done

echo ""
echo "Qwen3-8B NTC queued (9 jobs). Monitor: squeue -u \$USER"