#!/usr/bin/env python
"""Diagnostic — does the medium tier (NTC-v2) actually do anything?

Several canonical tables show NTC-v2 and answer agreement with identical
accuracy AND identical token cut, which means the confidence gate never binds
and the fused policy silently reduces to its agreement component. A reviewer
will notice two "different" methods with byte-identical rows, so we quantify
it: for every setting we report the fraction of items where NTC-v2 halts at a
different checkpoint from AGREE at the same m, and the fraction where the gate
delays the halt at all.

    python scripts/ntc_v2_diagnostic.py --probes experiments/ntc/w1_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("w1s", _here / "ntc_w1_stats.py")
S = importlib.util.module_from_spec(spec)
sys.modules["w1s"] = S
spec.loader.exec_module(S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--out", default="experiments/ntc/NTC_V2_DIAGNOSTIC.md")
    a = ap.parse_args()

    md = ["# Does the medium tier bind?", "",
          f"NTC-v2 (m={a.m}, theta={a.theta}) versus AGREE (m={a.m}) on the same "
          "traces. `differs` = share of items halting at a different checkpoint; "
          "`delayed` = share where the confidence gate postponed the halt; "
          "`gate never binds` means the fused policy is exactly its agreement "
          "component and should be reported as such.", "",
          "| setting | items | differs | delayed | identical? |",
          "|---|---|---|---|---|"]
    print(f"{'setting':30s} {'items':>6s} {'differs':>9s} {'delayed':>9s}")
    for pf in a.probes:
        d = json.loads(Path(pf).read_text())
        traces, bench = d["traces"], d["benchmark"]
        tag = f"{bench}/{d['model'].split('/')[-1]}"
        for t in traces:
            t["natural_correct"] = bool(
                S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
        S.enrich_probes_with_nll(traces)
        diff = delayed = 0
        for t in traces:
            pr = t["probes"]
            if not pr:
                continue
            ka = S.agree_policy(pr, m=a.m, bm=bench)
            kv = S.ntc_v2_policy(pr, m=a.m, theta=a.theta, bm=bench)
            if ka != kv:
                diff += 1
                if (kv is None and ka is not None) or \
                   (kv is not None and ka is not None and kv > ka):
                    delayed += 1
        n = len(traces)
        pd_, pl = 100 * diff / n, 100 * delayed / n
        print(f"{tag:30s} {n:6d} {pd_:8.1f}% {pl:8.1f}%")
        md.append(f"| {tag} | {n} | {pd_:.1f}% | {pl:.1f}% | "
                  f"{'**yes — gate never binds**' if diff == 0 else 'no'} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
