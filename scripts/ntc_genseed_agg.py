#!/usr/bin/env python
"""Aggregate results ACROSS GENERATION SEEDS (and across avg@k samples).

Give it the probe files of the same (model, benchmark) config generated with
different sampling seeds. For each file it runs the calibrated split analysis
(reusing ntc_w1_stats), then reports the across-generation-seed mean ± std for
every method — the variance source a Q1 reviewer asks about first.

Also serves AIME avg@k: pass the k per-seed AIME files; the per-file accuracy
is one sample, the aggregate is avg@k with std.

Usage:
    python scripts/ntc_genseed_agg.py \
        --probes experiments/ntc/w1_math500_Qwen3-4B.json \
        --probes experiments/ntc/w1_math500_Qwen3-4B_s43.json \
        --probes experiments/ntc/w1_math500_Qwen3-4B_s44.json \
        --tag math500_Qwen3-4B
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "src"))
spec = importlib.util.spec_from_file_location("w1stats", _here / "ntc_w1_stats.py")
S = importlib.util.module_from_spec(spec)
sys.modules["w1stats"] = S
spec.loader.exec_module(S)


def analyse_file(path, warmup_frac, n_splits, eps_list):
    d = json.loads(Path(path).read_text())
    traces, bench = d["traces"], d["benchmark"]
    fams = dict(S.FAMILIES)  # copy: MUR registration is per-file
    if S.enrich_probes_with_nll(traces):
        fams["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
    n = len(traces)
    n_warm = int(n * warmup_frac)
    out = {}
    names = list(fams) + [f"NTC-full(e={e})" for e in eps_list] + \
            ["vanilla", "REFRAIN-SWUCB"]
    acc = {k: [] for k in names}
    cut = {k: [] for k in names}
    saved_FAM = S.FAMILIES
    S.FAMILIES = fams
    try:
        for seed in range(n_splits):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            warm = [traces[i] for i in idx[:n_warm]]
            ev = [traces[i] for i in idx[n_warm:]]
            vt = float(np.mean([t["n_total_tokens"] for t in ev]))
            acc["vanilla"].append(float(np.mean([t["natural_correct"] for t in ev])))
            cut["vanilla"].append(0.0)
            picks, _ = S.calibrate(warm, bench, eps=eps_list[0])
            for fam, kw in picks.items():
                ok, tok = S.per_item(ev, bench, fams[fam][0], kw)
                acc[fam].append(float(ok.mean()))
                cut[fam].append(100 * (1 - tok.mean() / vt))
            for e in eps_list:
                _, (gf, gk) = S.calibrate(warm, bench, eps=e)
                ok, tok = S.per_item(ev, bench, fams[gf][0], gk)
                acc[f"NTC-full(e={e})"].append(float(ok.mean()))
                cut[f"NTC-full(e={e})"].append(100 * (1 - tok.mean() / vt))
            rok, rtok = S.refrain_swucb_stream(ev, bench)
            acc["REFRAIN-SWUCB"].append(float(rok.mean()))
            cut["REFRAIN-SWUCB"].append(100 * (1 - rtok.mean() / vt))
    finally:
        S.FAMILIES = saved_FAM
    for k in names:
        out[k] = (float(np.mean(acc[k])) if acc[k] else float("nan"),
                  float(np.mean(cut[k])) if cut[k] else float("nan"))
    return d["model"], bench, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--cv-eps", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_file = []
    for pf in args.probes:
        model, bench, res = analyse_file(pf, args.warmup_frac,
                                         args.n_splits, args.cv_eps)
        per_file.append((pf, res))
        print(f"[done] {pf}")

    keys = sorted({k for _, r in per_file for k in r},
                  key=lambda k: (k != "vanilla", k))
    print(f"\n=== ACROSS {len(per_file)} GENERATION SEEDS — {model} / {bench} ===")
    print(f"{'method':<18}{'acc mean±std':>18}{'cut% mean±std':>18}")
    lines = [f"# Generation-seed aggregate — {args.tag} "
             f"({len(per_file)} seeds x {args.n_splits} splits)\n",
             "| method | accuracy (mean ± std over gen seeds) | cut % |",
             "|---|---|---|"]
    for k in keys:
        a = np.array([r[k][0] for _, r in per_file if k in r])
        c = np.array([r[k][1] for _, r in per_file if k in r])
        print(f"{k:<18}{a.mean():>10.3f} ±{a.std():>5.3f}"
              f"{c.mean():>12.1f} ±{c.std():>4.1f}")
        lines.append(f"| {k} | {a.mean():.3f} ± {a.std():.3f} "
                     f"| {c.mean():.1f} ± {c.std():.1f} |")
    out = args.out or f"experiments/ntc/GENSEEDS_{args.tag}.md"
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"\ntable: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
