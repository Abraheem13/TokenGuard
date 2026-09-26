#!/usr/bin/env python
"""Does the fusion tier's confidence gate change the halting decision?

The fusion tier halts when m consecutive trial answers agree and the smoothed
confidence clears theta. If the gate never binds, the tier reduces to answer
agreement. For each probe file this reports the share of items on which the
fusion tier (m, theta) halts at a different checkpoint from agreement (m), and
the share on which the gate delays the halt.

ntc_dissertation_numbers.py uses gate_rows() for its gate diagnostic; run as a
script, the table is printed and written to --out.

    python scripts/ntc_v2_diagnostic.py --theta 0.9 --probes experiments/ntc/w1_math500_Qwen3-4B.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ntc_w1_stats as S


def gate_rows(files, m=3, theta=0.5):
    """Markdown table rows `| setting | items | differs | delayed | identical? |`."""
    rows = []
    for pf in files:
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
            ka = S.agree_policy(pr, m=m, bm=bench)
            kv = S.ntc_v2_policy(pr, m=m, theta=theta, bm=bench)
            if ka != kv:
                diff += 1
                if (kv is None and ka is not None) or \
                   (kv is not None and ka is not None and kv > ka):
                    delayed += 1
        n = len(traces)
        rows.append(f"| {tag} | {n} | {100 * diff / n:.1f}% | {100 * delayed / n:.1f}% | "
                    f"{'yes, the gate never binds' if diff == 0 else 'no'} |")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="optional markdown output file")
    a = ap.parse_args()

    md = ["# Confidence-gate diagnostic", "",
          f"Fusion tier (m={a.m}, theta={a.theta}) against answer agreement (m={a.m}) on "
          "the same traces. `differs`: share of items halting at a different checkpoint; "
          "`delayed`: share on which the gate postponed the halt.", "",
          "| setting | items | differs | delayed | identical? |",
          "|---|---|---|---|---|"] + gate_rows(a.probes, a.m, a.theta)
    print("\n".join(md))
    if a.out:
        Path(a.out).write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
