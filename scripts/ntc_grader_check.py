#!/usr/bin/env python
"""Reproducibility of the answer grader's cached verdicts.

Every accuracy in this repository comes from one symbolic grader (the
equivalence checker distributed with DEER), whose verdicts are cached in
experiments/ntc/grader_cache.json and read by every analysis. This script
recomputes each cached verdict from scratch, in one process and in cache order.
The grader is not stateless across calls, so a long sequential run can disagree
with a verdict computed in isolation; every disagreement is therefore
recomputed again in a fresh interpreter, and the report states how many of them
the fresh computation resolves in favour of the stored verdict.

Writes experiments/ntc/GRADER_CHECK.md.

    python scripts/ntc_grader_check.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokenguard.reasoning import datasets as D

CACHE = ROOT / "experiments" / "ntc" / "grader_cache.json"
OUT = ROOT / "experiments" / "ntc" / "GRADER_CHECK.md"
FRESH = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
         "from tokenguard.reasoning import datasets as D; "
         "print(json.dumps(bool(D._graded_equal_compute(sys.argv[2], sys.argv[3]))))")


def fresh_verdict(pred: str, gold: str) -> bool:
    """The grader's verdict computed in a new interpreter."""
    r = subprocess.run([sys.executable, "-c", FRESH, str(ROOT / "src"), pred, gold],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def cell(text: str) -> str:
    """Escape a LaTeX answer for a markdown table cell."""
    return text.replace("|", "\\|")


def main() -> int:
    cache = json.loads(CACHE.read_text())
    disagree = []
    for i, (key, stored) in enumerate(cache.items(), start=1):
        pred, _, gold = key.partition("\x1f")
        again = D._graded_equal_compute(pred, gold)
        if bool(again) != bool(stored):
            disagree.append((pred, gold, bool(stored), bool(again)))
        if i % 1000 == 0:
            print(f"  {i}/{len(cache)} recomputed", flush=True)
    resolved = [(p, g, s, a, fresh_verdict(p, g)) for p, g, s, a in disagree]
    n_ok = sum(1 for p, g, s, a, f in resolved if f == s)

    md = ["# Grader reproducibility", "",
          "Every cached verdict recomputed from scratch in one process, in cache order. "
          "Each disagreement is recomputed again in a fresh interpreter; `fresh` is that "
          "verdict. The analyses read the committed cache, so no reported number depends "
          "on this recomputation.", "",
          "| quantity | value |", "|---|---|",
          f"| stored verdicts | {len(cache)} |",
          f"| recomputed | {len(cache)} |",
          f"| disagreements | {len(disagree)} |",
          f"| disagreements matching the stored verdict when recomputed in a fresh "
          f"interpreter | {n_ok} |", ""]
    if resolved:
        md += ["| prediction | reference | stored | sequential | fresh |",
               "|---|---|---|---|---|"]
        md += [f"| `{cell(p)}` | `{cell(g)}` | {s} | {a} | {f} |" for p, g, s, a, f in resolved]
    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
