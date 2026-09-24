#!/usr/bin/env python
"""Is the answer grader deterministic?  (no GPU)

Every accuracy in this project is produced by one symbolic grader, whose verdicts
are cached on disk so that the numbers are reproducible. This script recomputes
each cached verdict from scratch and reports any disagreement; a clean run means
the stored verdicts are exactly what the grader produces today.

Writes experiments/ntc/GRADER_CHECK.md.

    python scripts/ntc_grader_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokenguard.reasoning import datasets as D  # noqa: E402

CACHE = ROOT / "experiments" / "ntc" / "grader_cache.json"
OUT = ROOT / "experiments" / "ntc" / "GRADER_CHECK.md"


def main() -> int:
    cache = json.loads(CACHE.read_text())
    disagree, checked = [], 0
    for key, stored in cache.items():
        pred, _, gold = key.partition("\x1f")
        again = D._graded_equal_compute(pred, gold)
        checked += 1
        if bool(again) != bool(stored):
            disagree.append((pred, gold, stored, again))
        if checked % 500 == 0:
            print(f"  {checked}/{len(cache)} rechecked", flush=True)
    md = ["# Grader determinism", "",
          "Recomputed every cached verdict from scratch with the symbolic grader. "
          "Disagreements, if any, are listed below; the analyses read the committed "
          "cache, so they do not change any reported number.", "",
          "| quantity | value |", "|---|---|",
          f"| stored verdicts | {len(cache)} |",
          f"| recomputed | {checked} |",
          f"| disagreements | {len(disagree)} |", ""]
    for p, g, s, a in disagree[:20]:
        md.append(f"- `{p}` vs `{g}`: stored {s}, recomputed {a}")
    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0                                  # a report, not a gate


if __name__ == "__main__":
    raise SystemExit(main())
