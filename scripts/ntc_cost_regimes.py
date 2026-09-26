#!/usr/bin/env python
"""Net token saving of each halting rule under three serving regimes.

What a probe costs depends on the serving engine:

  KV-fork       the probe branches from the cached prefix: it costs only the
                tokens it decodes
  prefix-cache  as KV-fork, and the answer cue is prefilled again
  black-box     every probe resends the whole prefix, which is prefilled again

Prefill tokens are charged at a weight w decode-token equivalents (default 0.2,
set with --prefill-weight). Each rule is evaluated on all items at the middle
value of its parameter grid; savings are relative to full generation.

Writes experiments/ntc/COST_REGIMES.md.

    python scripts/ntc_cost_regimes.py --probes experiments/ntc/w1_gsm8k_Qwen3-4B.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ntc_w1_stats as S

CUE_TOKENS = 12  # length of the answer cue appended at each probe


def costs(traces, bench, fn, kw, w):
    """(accuracy, KV-fork, prefix-cache, black-box) mean per-item cost."""
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

    md = ["# Net token saving under three serving regimes", "",
          f"Prefill is charged at w = {a.prefill_weight} decode-token equivalents. Each rule "
          "runs at the middle value of its parameter grid on all items; savings are "
          "relative to full generation, and a negative saving is a net cost.", "",
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
        for fam, (fn, grid) in S.FAMILIES.items():
            kw = grid[len(grid) // 2]
            acc, kv, pc, bb = costs(traces, bench, fn, kw, a.prefill_weight)
            sv = [100 * (1 - c / van) for c in (kv, pc, bb)]
            md.append(f"| {model} | {bench} | {fam}{kw} | {acc:.3f} "
                      f"| {sv[0]:+.1f}% | {sv[1]:+.1f}% | {sv[2]:+.1f}% |")
            print(f"{model:10s} {bench:14s} {fam:16s} acc={acc:.3f}  "
                  f"kv={sv[0]:+6.1f}%  pc={sv[1]:+6.1f}%  bb={sv[2]:+7.1f}%")
        print()
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
