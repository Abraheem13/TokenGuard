# One-time GPU environment on native Windows (no administrator rights needed),
# followed by a short smoke test: 24 GSM8K questions with Qwen3-4B on the
# PyTorch engine (--engine hf). vLLM does not run on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\win_gpu_setup.ps1
#
# Needs: an NVIDIA GPU with about 12 GB of free memory, uv (https://astral.sh/uv),
# about 20 GB of disk. Creates .venv-gpu\ in the repository and writes
# experiments\v2\smoke.log. Paste the block printed at the end back.

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Check($what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit code $LASTEXITCODE)"; exit 1 }
}

Write-Host "== GPU =="
$gpu = nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits
Write-Host $gpu
$used = [int]($gpu.Split(",")[2].Trim())
if ($used -gt 5000) {
    Write-Host "The GPU already has $used MiB in use. Wait until the other job has finished, then run this again."
    exit 1
}

Write-Host "== Python environment (.venv-gpu) =="
if (-not (Test-Path ".venv-gpu")) { uv venv --python 3.12 .venv-gpu; Check "uv venv" }
uv pip install --python .venv-gpu torch --torch-backend=auto; Check "install torch"
uv pip install --python .venv-gpu transformers datasets -r requirements.txt; Check "install packages"
uv pip install --python .venv-gpu -e .; Check "install tokenguard"
$py = ".venv-gpu\Scripts\python.exe"

Write-Host "== versions =="
& $py -c "import torch, transformers; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| transformers', transformers.__version__); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
Check "import torch"

Write-Host "== smoke test: 24 GSM8K questions, Qwen3-4B (first run downloads about 8 GB) =="
New-Item -ItemType Directory -Force -Path "experiments\v2" | Out-Null
$start = Get-Date
cmd /c "$py scripts\ntc_w1_thinking.py --engine hf --model Qwen/Qwen3-4B --benchmark gsm8k --limit 24 --batch 12 --max-think 4096 --probe-batch 8 --out experiments\v2\smoke_gsm8k_Qwen3-4B.json > experiments\v2\smoke.log 2>&1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "SMOKE TEST FAILED. Last lines of experiments\v2\smoke.log:"
    Get-Content "experiments\v2\smoke.log" -Tail 40
    exit 1
}
$secs = [int]((Get-Date) - $start).TotalSeconds

Write-Host ""
Write-Host "==================== PASTE THIS BLOCK BACK ===================="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
& $py -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
Get-Content "experiments\v2\smoke.log" -Tail 10
Write-Host "elapsed: $secs s"
Write-Host "SMOKE TEST PASSED"
Write-Host "==============================================================="
