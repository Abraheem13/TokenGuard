#!/usr/bin/env python
"""Primary comparison: AUCC with uncertainty, paired tests and minimax regret.

Every method chooses its parameter from calibration items only (deployable
protocol) and is scored on the disjoint evaluation items with
overhead-inclusive costs. For each generation-seed file the analysis is
repeated over R random calibration/evaluation splits; seed files of the same
benchmark and model form one setting.

Reported across settings:

  * AUCC over budgets {0.4, ..., 1.0}: mean, s.d. and minimum per method, with an
    exact two-sided sign test and a tie-corrected Wilcoxon signed-rank statistic
    against the selection tier (NTC-full);
  * operational-region AUCC over budgets {0.4, 0.5, 0.6}, where early exit
    matters, with the same paired tests;
  * minimax regret on the operational region: for each setting, the gap to the
    best method in that setting, maximised over settings.

Writes experiments/ntc/PRIMARY_STATS.md.

    python scripts/ntc_primary_stats.py --probes experiments/ntc/w1_gsm8k_Qwen3-4B.json ...
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import ntc_operating_curves as OC
import ntc_w1_stats as S

REF = "NTC-full (selection)"


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Bin(n, p), computed exactly."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(diffs):
    """Exact two-sided sign test on the non-zero paired differences."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return float("nan"), 0, 0
    pos = sum(1 for x in d if x > 0)
    k = max(pos, n - pos)
    return min(1.0, 2 * binom_sf(k, n)), pos, n


def wilcoxon_z(diffs):
    """Wilcoxon signed-rank statistic, normal approximation, average ranks for ties."""
    d = np.array([x for x in diffs if abs(x) > 1e-12], dtype=float)
    n = len(d)
    if n < 5:
        return float("nan")
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n, dtype=float)
    a = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w = float(np.sum(ranks[d > 0]))
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return (w - mu) / sd if sd > 0 else float("nan")


def setting_key(path, d):
    """Seed files of the same benchmark and model form one setting."""
    return f"{d['benchmark']}/{d['model'].split('/')[-1]}"


def aucc_for_file(traces, bench, split_seed):
    """AUCC and operational-region AUCC of every method on one split."""
    warm, ev = OC.split(traces, seed=split_seed)
    van_tok = float(np.mean([t["n_total_tokens"] for t in ev]))
    van_acc = float(np.mean([t["natural_correct"] for t in ev]))
    out, out_op = {}, {}

    def op_region(pts):
        vals = []
        for b in (0.4, 0.5, 0.6):
            f = [acc for (c, acc, _) in pts if c <= b + 1e-9]
            vals.append(max(f) if f else np.nan)
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else 0.0

    for name, (fn, grid) in OC.SWEEPS.items():
        pts = OC.curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc)
        out[name] = OC.metrics(pts, van_acc)["aucc"]
        out_op[name] = op_region(pts)
    nf = OC.curve_ntc_full(warm, ev, bench, van_tok, van_acc)
    out[REF] = OC.metrics(nf, van_acc)["aucc"]
    out_op[REF] = op_region(nf)
    return out, out_op


def fmt_test(w, p):
    if w is None:
        return "n/a", "n/a"
    return w, f"{p:.4f}" + ("*" if p < 0.05 else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--splits", type=int, default=5,
                    help="random calibration/evaluation splits per file")
    ap.add_argument("--out", default="experiments/ntc/PRIMARY_STATS.md")
    a = ap.parse_args()

    groups = defaultdict(list)
    for pf in a.probes:
        d = json.loads(Path(pf).read_text())
        groups[setting_key(pf, d)].append((pf, d))

    per_setting, per_setting_op = {}, {}
    for key, files in sorted(groups.items()):
        reps, reps_op = defaultdict(list), defaultdict(list)
        bench = files[0][1]["benchmark"]
        for _, d in files:
            traces = d["traces"]
            for t in traces:
                t["natural_correct"] = bool(
                    S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            S.enrich_probes_with_nll(traces)
            for sd_ in range(a.splits):
                full, op = aucc_for_file(traces, bench, sd_)
                for m, v in full.items():
                    reps[m].append(v)
                for m, v in op.items():
                    reps_op[m].append(v)
        per_setting[key] = {m: (float(np.mean(v)), float(np.std(v)), len(v))
                            for m, v in reps.items()}
        per_setting_op[key] = {m: float(np.mean(v)) for m, v in reps_op.items()}
        n_rep = max(len(v) for v in reps.values())
        print(f"\n=== {key}  ({len(files)} seed file(s) x {a.splits} splits "
              f"= {n_rep} replicates) ===")
        for m, (mu, sd, _) in sorted(per_setting[key].items(), key=lambda kv: -kv[1][0]):
            print(f"  {m:24s} AUCC {mu:.3f} ± {sd:.3f}")

    methods = sorted({m for v in per_setting.values() for m in v})
    print(f"\nAggregate across {len(per_setting)} settings (paired tests against {REF})")
    print(f"{'method':24s} {'mean AUCC':>11s} {'sd':>7s} {'min':>7s} "
          f"{'wins':>7s} {'sign p':>9s} {'Wilcoxon z':>11s}")

    md = ["# Primary comparison with uncertainty and paired tests", "",
          f"Deployable protocol. {a.splits} random calibration/evaluation splits per "
          "generation-seed file; AUCC is averaged within each setting, then aggregated "
          f"across {len(per_setting)} settings. `NTC-full wins` counts settings where the "
          "selection tier has the higher AUCC; the sign test is exact and two-sided, and "
          "the Wilcoxon statistic is the tie-corrected normal approximation. "
          "`*`: p < 0.05.", "",
          "| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |",
          "|---|---|---|---|---|---|---|"]
    rows = []
    for m in methods:
        vals = [per_setting[k][m][0] for k in per_setting if m in per_setting[k]]
        if not vals:
            continue
        if m == REF:
            rows.append((m, float(np.mean(vals)), float(np.std(vals)),
                         float(np.min(vals)), None, float("nan"), float("nan")))
            continue
        diffs = [per_setting[k][REF][0] - per_setting[k][m][0]
                 for k in per_setting if m in per_setting[k] and REF in per_setting[k]]
        p, pos, n = sign_test(diffs)
        rows.append((m, float(np.mean(vals)), float(np.std(vals)),
                     float(np.min(vals)), f"{pos}/{n}", p, wilcoxon_z(diffs)))
    for m, mu, sd, mn, w, p, z in sorted(rows, key=lambda r: -r[1]):
        ws, ps = fmt_test(w, p)
        zs = "n/a" if w is None else f"{z:+.2f}"
        print(f"{m:24s} {mu:11.3f} {sd:7.3f} {mn:7.3f} {ws:>7s} {ps:>9s} {zs:>11s}")
        md.append(f"| {m} | {mu:.3f} | {sd:.3f} | {mn:.3f} | {ws} | {ps} | {zs} |")

    title = "OPERATIONAL-REGION AUCC (budgets b <= 0.6, where early exit matters)"
    print(f"\n{title}")
    print(f"{'method':24s} {'mean':>9s} {'sd':>7s} {'worst':>8s} {'wins':>7s} {'sign p':>9s}")
    md += ["", f"## {title}", "",
           "AUCC over the full grid includes b = 1.0, where every method may decline to "
           "halt, so part of the grid cannot separate methods. The operational region "
           "restricts the average to the budgets at which early exit is required.", "",
           "| method | mean | s.d. | worst setting | NTC-full wins | sign-test p |",
           "|---|---|---|---|---|---|"]
    rr = []
    for m in methods:
        vals = [per_setting_op[k][m] for k in per_setting_op if m in per_setting_op[k]]
        if not vals:
            continue
        if m == REF:
            rr.append((m, float(np.mean(vals)), float(np.std(vals)),
                       float(np.min(vals)), None, float("nan")))
            continue
        dd = [per_setting_op[k][REF] - per_setting_op[k][m] for k in per_setting_op
              if m in per_setting_op[k] and REF in per_setting_op[k]]
        pp, pos, nn = sign_test(dd)
        rr.append((m, float(np.mean(vals)), float(np.std(vals)),
                   float(np.min(vals)), f"{pos}/{nn}", pp))
    for m, mu, sd, mn, w, pp in sorted(rr, key=lambda r: -r[1]):
        ws, ps = fmt_test(w, pp)
        print(f"{m:24s} {mu:9.3f} {sd:7.3f} {mn:8.3f} {ws:>7s} {ps:>9s}")
        md.append(f"| {m} | {mu:.3f} | {sd:.3f} | {mn:.3f} | {ws} | {ps} |")

    regret = {}
    for k, tab in per_setting_op.items():
        best = max(tab.values())
        regret[k] = {m: best - v for m, v in tab.items()}
    print("\nMinimax regret over settings (operational region; lower is better)")
    print(f"{'method':24s} {'max regret':>11s} {'mean regret':>12s}")
    md += ["", "## Minimax regret (operational region)", "",
           "For each setting, regret(M) is the best operational-region AUCC in that "
           "setting minus that of M. The table reports its maximum and mean over "
           "settings; the maximum is the criterion for committing to one method "
           "without knowing which workload will arrive.",
           "", "| method | max regret | mean regret |", "|---|---|---|"]
    for m, mx, mn_ in sorted(
            [(m, max(r[m] for r in regret.values() if m in r),
              float(np.mean([r[m] for r in regret.values() if m in r])))
             for m in methods], key=lambda t: t[1]):
        print(f"{m:24s} {mx:11.3f} {mn_:12.3f}")
        md.append(f"| {m} | {mx:.3f} | {mn_:.3f} |")

    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
