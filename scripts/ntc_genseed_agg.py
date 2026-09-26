#!/usr/bin/env python
"""Per-setting results aggregated over generation seeds.

Takes the probe files of one benchmark and model generated with different
sampling seeds. For each file the calibrated analysis is repeated over random
calibration/evaluation splits (every method chooses its parameter on the
calibration items and is scored on the evaluation items), and the table reports
the mean and standard deviation over seed files of each method's accuracy and
token cut. For the AIME sets each seed file is one sample, so the aggregate is
avg@k; --pool-calib adds the same calibration questions from the sibling seed
files, so that no evaluation question enters calibration in any generation.

Writes experiments/ntc/GENSEEDS_<tag>.md.

    python scripts/ntc_genseed_agg.py --tag math500_Qwen3-4B \
        --probes experiments/ntc/w1_math500_Qwen3-4B.json \
        --probes experiments/ntc/w1_math500_Qwen3-4B_s43.json \
        --probes experiments/ntc/w1_math500_Qwen3-4B_s44.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ntc_w1_stats as S


def analyse_file(path, warmup_frac, n_splits, eps_list, siblings=None):
    d = json.loads(Path(path).read_text())
    traces, bench = d["traces"], d["benchmark"]
    for t in traces:
        t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
    fams = dict(S.FAMILIES)
    if S.enrich_probes_with_nll(traces):
        fams["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
    n = len(traces)
    n_warm = int(n * warmup_frac)
    names = list(fams) + [f"NTC-full(e={e})" for e in eps_list] + ["vanilla", "REFRAIN-SWUCB"]
    acc = {k: [] for k in names}
    cut = {k: [] for k in names}
    saved = S.FAMILIES
    S.FAMILIES = fams
    try:
        for seed in range(n_splits):
            idx = np.random.default_rng(seed).permutation(n)
            widx = idx[:n_warm]
            warm = [traces[i] for i in widx]
            for sib in (siblings or []):
                warm += [sib[i] for i in widx if i < len(sib)]
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
        S.FAMILIES = saved
    out = {k: (float(np.mean(acc[k])) if acc[k] else float("nan"),
               float(np.mean(cut[k])) if cut[k] else float("nan")) for k in names}
    return d["model"], bench, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--cv-eps", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--pool-calib", action="store_true",
                    help="pool calibration items across seed files (avg@k benchmarks)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_file = []
    for pf in args.probes:
        sibs = None
        if args.pool_calib:
            sibs = []
            for other in args.probes:
                if other == pf:
                    continue
                od = json.loads(Path(other).read_text())
                ot = od["traces"]
                for t in ot:
                    t["natural_correct"] = bool(S.is_correct(
                        t.get("natural_answer", ""), t["gold"], od["benchmark"]))
                S.enrich_probes_with_nll(ot)
                sibs.append(ot)
        model, bench, res = analyse_file(pf, args.warmup_frac, args.n_splits, args.cv_eps,
                                         siblings=sibs)
        per_file.append((pf, res))
        print(f"done: {pf}")

    keys = sorted({k for _, r in per_file for k in r}, key=lambda k: (k != "vanilla", k))
    print(f"\n=== {len(per_file)} generation seeds: {model} / {bench} ===")
    print(f"{'method':<18}{'acc mean±sd':>16}{'cut% mean±sd':>16}")
    lines = [f"# Generation-seed aggregate: {args.tag} "
             f"({len(per_file)} seeds x {args.n_splits} splits)\n",
             "| method | accuracy (mean ± std over gen seeds) | cut % |",
             "|---|---|---|"]
    for k in keys:
        a = np.array([r[k][0] for _, r in per_file if k in r])
        c = np.array([r[k][1] for _, r in per_file if k in r])
        print(f"{k:<18}{a.mean():>10.3f} ±{a.std():>5.3f}{c.mean():>10.1f} ±{c.std():>4.1f}")
        lines.append(f"| {k} | {a.mean():.3f} ± {a.std():.3f} | {c.mean():.1f} ± {c.std():.1f} |")
    out = args.out or f"experiments/ntc/GENSEEDS_{args.tag}.md"
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"\ntable: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
