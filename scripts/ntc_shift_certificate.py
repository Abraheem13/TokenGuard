#!/usr/bin/env python
"""Does the risk certificate survive domain shift?  (no GPU)

Theorem 1 assumes (A1): the warm-up items are drawn from the deployment
distribution.  Every other experiment in this repository calibrates and deploys
inside one domain, so (A1) holds by construction and the guarantee is never
challenged.  This script challenges it, comparing three certification rules on
the SAME held-out evaluation splits:

  IN-DOMAIN      certify on the target domain's own warm-up split      (A1 holds)
  TRANSFERRED    certify on ANOTHER domain's warm-up split             (A1 violated)
  DOMAIN-ROBUST  leave-one-domain-out: certify only candidates whose lower
                 bound clears -eps in EVERY source domain, Bonferroni over
                 |C| x |sources|

Writes experiments/ntc/SHIFT_CERTIFICATE.md.

    python scripts/ntc_shift_certificate.py
"""
from __future__ import annotations

import argparse
import json
import sys
import types
import typing
from collections import Counter
from pathlib import Path

# Python >= 3.13 removed typing.io, which an old antlr4 runtime still imports.
if "typing.io" not in sys.modules:  # pragma: no cover - environment shim
    _m = types.ModuleType("typing.io")
    _m.TextIO, _m.IO, _m.BinaryIO = typing.TextIO, typing.IO, typing.BinaryIO
    sys.modules["typing.io"] = _m

import numpy as np

_here = Path(__file__).resolve().parent
ROOT = _here.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib.util

spec = importlib.util.spec_from_file_location("oc", _here / "ntc_operating_curves.py")
OC = importlib.util.module_from_spec(spec)
sys.modules["oc"] = OC
spec.loader.exec_module(OC)
S = OC.S

NTC = ROOT / "experiments" / "ntc"

DOMAINS = [
    ("GSM8K-4B", "w1_gsm8k_Qwen3-4B.json"),
    ("GSM8K-8B", "w1_gsm8k_Qwen3-8B.json"),
    ("MATH-1.7B", "w1_math500_Qwen3-1.7B.json"),
    ("MATH-4B", "w1_math500_Qwen3-4B.json"),
    ("MATH-8B", "w1_math500_Qwen3-8B.json"),
    ("GPQA-4B", "w1sh_gpqa_Qwen3-4B.json"),
    ("GPQA-8B", "w1sh_gpqa_Qwen3-8B.json"),
    ("MMLU-Pro-4B", "w1_mmlupro_Qwen3-4B_s42.json"),
    ("MMLU-Pro-8B", "w1_mmlupro_Qwen3-8B_s42.json"),
    ("AIME-24", "w1_aime24_Qwen3-4B_s100.json"),
    ("AIME-25", "w1_aime25_Qwen3-4B_s100.json"),
    ("DeepSeek-MATH", "w1_math500_DeepSeek-R1-Distill-Qwen-7B.json"),
]


def candidate_stats(items, bench, cands, reps=3, k_folds=5):
    """Per-candidate (repeated paired K-fold mean delta, per-item SE, mean cost)."""
    n = len(items)
    k_folds = max(2, min(k_folds, n))
    van = np.array([t["natural_correct"] for t in items], dtype=float)
    out = []
    for fam, kw in cands:
        ok, tok = S.per_item(items, bench, S.FAMILIES[fam][0], kw)
        ok = ok.astype(float)
        ds = []
        for r in range(reps):
            idx = np.random.default_rng(1000 + r).permutation(n)
            for i in range(k_folds):
                f = idx[i::k_folds]
                if len(f):
                    ds.append(float(ok[f].mean() - van[f].mean()))
        d = ok - van
        se = float(np.std(d, ddof=1) / np.sqrt(max(1, len(d)))) if len(d) > 1 else 1.0
        out.append({"md": float(np.mean(ds)), "se": se, "tok": float(tok.mean())})
    return out


def lcb(c, n, m_tests, alpha):
    return c["md"] - S._t_quantile(1.0 - alpha / max(1, m_tests), max(2, n - 1)) * c["se"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out", default=str(NTC / "SHIFT_CERTIFICATE.md"))
    a = ap.parse_args()

    cands = [(fam, kw) for fam, (_, grid) in S.FAMILIES.items() for kw in grid]
    NC = len(cands)
    print(f"|C| = {NC} candidates, {len(DOMAINS)} domains, {a.splits} splits, alpha={a.alpha}")

    rows = {e: {"in": [], "transfer": [], "robust": []} for e in a.eps}
    picks = {e: {"in": [], "robust": []} for e in a.eps}
    for s in range(a.splits):
        cache = {}
        for name, fn in DOMAINS:
            d = json.loads((NTC / fn).read_text())
            traces, bench = d["traces"], d["benchmark"]
            for t in traces:
                t["natural_correct"] = bool(
                    S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            warm, ev = OC.split(traces, seed=s)
            cache[name] = dict(
                bench=bench, warm=warm, ev=ev,
                van_tok=float(np.mean([t["n_total_tokens"] for t in ev])),
                van_acc=float(np.mean([t["natural_correct"] for t in ev])),
                st=candidate_stats(warm, bench, cands))
            del d, traces

        def deploy(T, fam, kw):
            ok, tok = S.per_item(T["ev"], T["bench"], S.FAMILIES[fam][0], kw)
            return (100 * (float(ok.mean()) - T["van_acc"]),
                    100 * (1 - tok.mean() / T["van_tok"]))

        for tgt, _ in DOMAINS:
            T = cache[tgt]
            srcs = [x for x, _ in DOMAINS if x != tgt]
            for e in a.eps:
                n = len(T["warm"])
                feas = [i for i, c in enumerate(T["st"]) if lcb(c, n, NC, a.alpha) >= -e]
                if feas:
                    i = min(feas, key=lambda i: T["st"][i]["tok"])
                    dfc, cut = deploy(T, *cands[i]); pk = f"{cands[i][0]}{cands[i][1]}"
                else:
                    dfc, cut, pk = 0.0, 0.0, "pi0"
                rows[e]["in"].append((tgt, dfc, cut)); picks[e]["in"].append(pk)

                for src in srcs:
                    A = cache[src]; na = len(A["warm"])
                    fe = [i for i, c in enumerate(A["st"]) if lcb(c, na, NC, a.alpha) >= -e]
                    if not fe:
                        rows[e]["transfer"].append((tgt, 0.0, 0.0)); continue
                    i = min(fe, key=lambda i: A["st"][i]["tok"])
                    rows[e]["transfer"].append((tgt, *deploy(T, *cands[i])))

                M = NC * len(srcs)
                ok_c = []
                for i in range(NC):
                    worst = min(lcb(cache[x]["st"][i], len(cache[x]["warm"]), M, a.alpha)
                                for x in srcs)
                    if worst >= -e:
                        ok_c.append((i, float(np.mean([cache[x]["st"][i]["tok"] for x in srcs]))))
                if ok_c:
                    i = min(ok_c, key=lambda t: t[1])[0]
                    dfc, cut = deploy(T, *cands[i]); pk = f"{cands[i][0]}{cands[i][1]}"
                else:
                    dfc, cut, pk = 0.0, 0.0, "pi0"
                rows[e]["robust"].append((tgt, dfc, cut)); picks[e]["robust"].append(pk)
        print(f"  split {s} done", flush=True)

    md = ["# Does the risk certificate survive domain shift?", "",
          f"{len(DOMAINS)} target domains x {a.splits} calibration splits; "
          f"|C| = {NC}; alpha = {a.alpha}. `worst cell` is the single worst "
          "domain-split combination, a harsher statistic than the seed-averaged "
          "worst case of SLO_ATTAINMENT.md. `transferred` is averaged over all "
          f"{len(DOMAINS)-1} sources per target.", ""]
    for e in a.eps:
        print(f"\n=== eps = {e} ===")
        print(f"{'rule':14s} {'mean':>8s} {'worst':>8s} {'w/in 1pt':>9s} {'w/in eps':>9s} {'cut':>8s}")
        md += [f"## eps = {e}", "",
               "| certification rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |",
               "|---|---|---|---|---|---|"]
        for rule, lab in (("in", "in-domain (A1 holds)"),
                          ("transfer", "transferred (A1 violated)"),
                          ("robust", "domain-robust (LODO)")):
            v = rows[e][rule]
            d = np.array([x[1] for x in v]); c = np.array([x[2] for x in v])
            w1 = float(np.mean(d >= -1.0)); we = float(np.mean(d >= -100 * e))
            print(f"{rule:14s} {d.mean():+8.2f} {d.min():+8.2f} {w1:8.0%} {we:8.0%} {c.mean():7.1f}%")
            md.append(f"| {lab} | {d.mean():+.2f} | {d.min():+.1f} | {w1:.0%} | {we:.0%} | {c.mean():.1f}% |")
        md += ["", f"in-domain picks: `{dict(Counter(picks[e]['in']).most_common(5))}`",
               "", f"domain-robust picks: `{dict(Counter(picks[e]['robust']).most_common(5))}`", ""]
        print("  in-domain picks:", dict(Counter(picks[e]["in"]).most_common(4)))
        print("  robust picks   :", dict(Counter(picks[e]["robust"]).most_common(4)))
    Path(a.out).write_text("\n".join(md) + "\n")
    side = Path(a.out).with_suffix(".json")
    side.write_text(json.dumps(
        {"alpha": a.alpha, "splits": a.splits,
         "cells": {str(e): {r: [[x[0], x[1], x[2]] for x in rows[e][r]] for r in rows[e]}
                   for e in a.eps},
         "cut": {str(e): {r: float(np.mean([x[2] for x in rows[e][r]])) for r in rows[e]}
                 for e in a.eps}}, indent=1))
    print(f"\ntable: {a.out}\nraw  : {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
