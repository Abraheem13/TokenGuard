#!/usr/bin/env bash
# Checkpoint-density sweep — the one open question that needs a GPU.
#
# Hypothesis (Section 6.6 of the paper): our accuracy gap to DEER on MATH-500 is
# explained by checkpoint granularity, not by the halting signal.  Our harness
# probes every 256 thinking tokens and caps at 10 probes; DEER interrupts at
# every reasoning transition and pays 66-79% of the chain in trial answers.
# This queue re-runs the head-to-head configuration at 1x, 2x and 4x density so
# the trade can be measured instead of asserted.
#
# Conditions are otherwise IDENTICAL to run_queue_h2h2.sh (greedy, 16k budget,
# n=500, standard boxed prompt), so the new files drop straight into
# scripts/ntc_ckpt_density.py alongside the existing h2h2_* files.
#
#   bash run_queue_density.sh --dry   # inspect
#   bash run_queue_density.sh         # submit 4 jobs
set -e
PART="gpu-2c-l40s-1g"
DRY="${1:-}"
T="python scripts/ntc_w1_thinking.py"

submit () {
  local name="$1" hours="$2" cmd="$3"
  local f="/tmp/job_${name}.sbatch"
  cat > "$f" << SBATCH
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=${PART}
#SBATCH --time=${hours}:00:00
#SBATCH --output=experiments/ntc/slurm-%x-%j.log
cd ~/TokenGuard
source .venv/bin/activate
${cmd}
echo "JOB DONE ${name}"
SBATCH
  if [ "$DRY" = "--dry" ]; then echo "--- $f ---"; cat "$f"; echo;
  else sbatch "$f"; fi
}

for M in 4B 8B; do
  # 2x density: a probe every 128 thinking tokens, up to 20 probes
  submit dens2x_${M} 9 "$T --model Qwen/Qwen3-${M} --benchmark math500 --limit 500 \
    --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --temperature 0.0 \
    --probe-every 128 --max-probes 20 \
    --out experiments/ntc/dens2x_math500_Qwen3-${M}.json"
  # 4x density: a probe every 64 thinking tokens, up to 40 probes
  submit dens4x_${M} 10 "$T --model Qwen/Qwen3-${M} --benchmark math500 --limit 500 \
    --max-think 16384 --batch 10 --tp-size 1 --max-model-len 24576 --temperature 0.0 \
    --probe-every 64 --max-probes 40 \
    --out experiments/ntc/dens4x_math500_Qwen3-${M}.json"
done

echo ""
echo "Density sweep queued (4 jobs). The 1x baseline already exists as"
echo "  experiments/ntc/h2h2_math500_Qwen3-{4B,8B}.json"
echo "When the jobs finish:  python scripts/ntc_ckpt_density.py"

# ---------------------------------------------------------------------------
# Optional second job set: a genuinely different backbone family.
# DeepSeek-R1-Distill-Qwen-7B (already run) is a Qwen backbone, so "two model
# families" is thin.  DeepSeek-R1-Distill-Llama-8B is a reasoning model on a
# Llama backbone and uses the same <think> protocol, so it drops into the same
# harness.  Enable with:  CROSS_FAMILY=1 bash run_queue_density.sh
if [ -n "${CROSS_FAMILY:-}" ]; then
  submit xfam_llama8b 8 "$T --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --benchmark math500 --limit 200 --max-think 16384 --batch 8 --tp-size 1 \
    --max-model-len 24576 --seed 42 \
    --out experiments/ntc/w1_math500_DeepSeek-R1-Distill-Llama-8B.json"
  submit xfam_llama8b_g 7 "$T --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --benchmark gpqa_diamond --limit 198 --max-think 16384 --batch 8 --tp-size 1 \
    --max-model-len 24576 --seed 42 \
    --out experiments/ntc/w1sh_gpqa_DeepSeek-R1-Distill-Llama-8B.json"
  echo "Cross-family jobs queued (2 more)."
fi
