#!/usr/bin/env bash
# One-time GPU environment for generating probe streams, followed by a short
# smoke test (24 GSM8K questions with Qwen3-4B, a few minutes).
#
#   bash scripts/gpu_setup.sh
#
# Needs Linux or WSL2 with an NVIDIA driver, about 12 GB of free VRAM and about
# 15 GB of disk for the model. It creates .venv-gpu/ in the repository and
# writes experiments/v2/smoke.log. Paste the block printed at the end back.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system =="
uname -sr
if grep -qi microsoft /proc/version 2>/dev/null; then echo "WSL2: yes"; else echo "WSL2: no"; fi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader

echo "== python environment (.venv-gpu) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d .venv-gpu ]; then uv venv --python 3.12 --seed .venv-gpu; fi
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install transformers datasets -r requirements.txt
uv pip install -e .

echo "== versions =="
python - <<'PY'
import torch, vllm, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("vllm", vllm.__version__, "| transformers", transformers.__version__)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "== smoke test: 24 GSM8K questions, Qwen3-4B =="
mkdir -p experiments/v2
start=$(date +%s)
python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark gsm8k \
    --limit 24 --max-think 4096 --max-model-len 8192 --batch 24 --gpu-mem 0.85 \
    --out experiments/v2/smoke_gsm8k_Qwen3-4B.json > experiments/v2/smoke.log 2>&1 \
  || { echo "SMOKE TEST FAILED. Last lines of experiments/v2/smoke.log:"; tail -n 40 experiments/v2/smoke.log; exit 1; }
end=$(date +%s)

echo
echo "==================== PASTE THIS BLOCK BACK ===================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"
tail -n 8 experiments/v2/smoke.log
echo "elapsed: $((end - start)) s"
echo "SMOKE TEST PASSED"
echo "==============================================================="
