#!/usr/bin/env python
"""Day 1 — environment verification.

Prints a pass/fail table for every dependency tier and the GPU, and exits
non-zero if anything *required for Day 1* is broken. Later-day dependencies
(torch/transformers/fastapi) are reported but only warned about, so the data
work can start even while heavy installs finish.

Run:  python scripts/day1_setup_check.py
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED_DAY1 = ["numpy", "pandas", "yaml", "pyarrow", "huggingface_hub", "tqdm"]
REQUIRED_LATER = ["torch", "transformers", "peft", "sklearn", "matplotlib",
                  "fastapi", "uvicorn", "streamlit", "datasets"]


def check(modname: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(modname)
        return True, getattr(mod, "__version__", "?")
    except Exception as exc:  # noqa: BLE001 — report any import failure
        return False, str(exc).splitlines()[0][:60]


def main() -> int:
    print(f"\nTokenGuard environment check")
    print(f"{'=' * 62}")
    print(f"Python   : {sys.version.split()[0]}  ({platform.platform()})")

    failures = 0
    print(f"\n{'package':<18}{'tier':<10}{'status':<8}version / error")
    print("-" * 62)
    for name in REQUIRED_DAY1:
        ok, info = check(name)
        failures += (not ok)
        print(f"{name:<18}{'day-1':<10}{'OK' if ok else 'FAIL':<8}{info}")
    for name in REQUIRED_LATER:
        ok, info = check(name)
        print(f"{name:<18}{'later':<10}{'OK' if ok else 'warn':<8}{info}")

    # GPU report (informational on Day 1; required from Day 4)
    print("-" * 62)
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"GPU      : {dev.name}  ({dev.total_memory / 2**30:.1f} GB, "
                  f"CUDA {torch.version.cuda})")
        else:
            print("GPU      : torch installed, CUDA NOT available "
                  "(fine for Days 1-3; required from Day 4)")
    except ImportError:
        print("GPU      : torch not installed yet (fine for Days 1-3)")

    print("=" * 62)
    if failures:
        print(f"RESULT   : FAIL — {failures} day-1 package(s) missing. "
              f"Run: pip install -r requirements.txt")
        return 1
    print("RESULT   : PASS — Day 1 environment ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
