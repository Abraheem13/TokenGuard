#!/usr/bin/env bash
# =============================================================================
# TokenGuard — Day 1 environment setup
# Usage:  bash setup_env.sh [--venv-dir .venv]
# Creates a virtual environment, installs pinned dependencies, installs the
# package in editable mode, and runs the environment check.
# =============================================================================
set -euo pipefail

VENV_DIR="${2:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> [1/5] Checking Python version (need >= 3.10)"
"$PYTHON_BIN" - <<'EOF'
import sys
assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version}"
print(f"    OK: {sys.version.split()[0]}")
EOF

echo "==> [2/5] Creating virtual environment at ${VENV_DIR}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
# Upgrade the build toolchain FIRST. On Python 3.13 an old setuptools lacks
# pkg_resources, which breaks source builds of some wheels (e.g. pyarrow).
python -m pip install --upgrade pip setuptools wheel --quiet

echo "==> [3/5] Installing dependencies (this can take a few minutes)"
pip install -r requirements.txt --quiet

echo "==> [4/5] Installing tokenguard in editable mode"
pip install -e . --quiet

echo "==> [5/5] Running environment check"
python scripts/day1_setup_check.py

echo ""
echo "Setup complete. Activate with:  source ${VENV_DIR}/bin/activate"
echo "Next:                           python scripts/day1_download_data.py"