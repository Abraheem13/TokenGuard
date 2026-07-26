#!/usr/bin/env python
"""P0 FIX #5 — overhead accounting under three deployment regimes.

Our headline "3-5% probe overhead" implicitly assumes an engine that can fork
the KV cache and resume. Reviewers (correctly) ask what happens without it.
This script recomputes every policy's cost under:

  kv-fork      : probes cost DECODE tokens only (prefix reused, resume free)
  prefix-cache : probes additionally re-prefill the answer cue (cached prefix)
  black-box    : every probe re-sends the WHOLE prefix (no reuse at all)

Prefill tokens are charged at weight w (default 0.2 decode-token equivalents,
since prefill is compute-bound and cheaper per token); w is a CLI flag so the
sensitivity is explicit rather than hidden.

    python scripts/ntc_cost_regimes.py --probes experiments/ntc/w1_*.json
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

CUE_TOKENS = 12  # length of the forced-answer cue we inject


def costs(traces, bench, fn, kw, w):
    """Return (acc, kv, pc, bb) mean per-item costs in decode-token equivalents."""
    ok, kv, pc, bb = [], [], [], []
    for t in traces:
        pr = t["probes"]
        k = fn(pr, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames
                      else kw)) if pr else None
        used = pr if k is None else pr[:k + 1]
        think = t["n_total_tokens"] if k is None else pr[k]["ckpt_tokens"]
        dec = sum(p["n_probe_tokens"] for p in used)
        pre_cue = w * CUE_TOKENS * len(used)
        pre_full = w * sum(p["ckpt_tokens"] + CUE_TOKENS for p in used)
        if k is None:
            ok.append(bool(t["natural_correct"]))
        else:
            ok.append(S.is_correct(pr[k]["answer"], t["gold"], bench))
        kv.append(think + dec)
        pc.append(think + dec + pre_cue)
        bb.append(think + dec + pre_full)
    return (float(np.mean(ok)), float(np.mean(kv)), float(np.mean(pc)),
            float(np.mean(bb)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--prefill-weight", type=float, default=0.2)
    ap.add_argument("--out", default="experiments/ntc/COST_REGIMES.md")
    a = ap.parse_args()

    md = ["# Probe-overhead accounting under three deployment regimes",
          f"(prefill charged at w={a.prefill_weight} decode-token equivalents; "
          "savings are % vs full thinking)", "",
          "| model | benchmark | policy | acc | KV-fork | prefix-cache | black-box |",
          "|---|---|---|---|---|---|---|"]
    for pf in a.probes:
        d = json.loads(Path(pf).read_text())
        traces, bench = d["traces"], d["benchmark"]
        model = d["model"].split("/")[-1]
        for t in traces:
            t["natural_correct"] = bool(
                S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
        van = float(np.mean([t["n_total_tokens"] for t in traces]))
        S.enrich_probes_with_nll(traces)
        rows = []
        for fam, (fn, grid) in S.FAMILIES.items():
            kw = grid[len(grid) // 2]
            acc, kv, pc, bb = costs(traces, bench, fn, kw, a.prefill_weight)
            rows.append((fam, kw, acc, kv, pc, bb))
        for fam, kw, acc, kv, pc, bb in rows:
            sv = lambda c: 100 * (1 - c / van)
            md.append(f"| {model} | {bench} | {fam}{kw} | {acc:.3f} "
                      f"| {sv(kv):+.1f}% | {sv(pc):+.1f}% | {sv(bb):+.1f}% |")
            print(f"{model:10s} {bench:14s} {fam:16s} acc={acc:.3f}  "
                  f"kv={sv(kv):+6.1f}%  pc={sv(pc):+6.1f}%  bb={sv(bb):+7.1f}%")
        print()
    md.append("")
    md.append("Interpretation: our headline savings assume the KV-fork regime "
              "(an engine that forks and resumes). Under prefix-cache the cost "
              "is nearly identical because only the short cue is re-prefilled; "
              "under a pure black-box API that re-sends the prefix at every "
              "checkpoint, probing can erase the savings entirely.")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
