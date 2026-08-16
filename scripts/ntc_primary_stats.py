#!/usr/bin/env python
"""PRIMARY METRIC WITH UNCERTAINTY — the table that carries the headline claim.

Two gaps in the earlier aggregate are closed here:

  (1) variability. AUCC was computed from a single calibration/evaluation split
      and, for multi-seed benchmarks, a single generation seed. We now repeat
      over R random splits and over every available generation seed, and report
      mean +/- s.d. of AUCC per method per setting, then across settings.

  (2) significance. "Highest mean AUCC" over 12 settings is a claim about a
      paired comparison, so we test it: for each baseline we take the
      per-setting AUCC difference (ours minus theirs) and run an exact sign
      test (binomial) plus a Wilcoxon signed-rank statistic. With 12 settings
      the exact sign test is the honest instrument; both are reported.

Everything uses the DEPLOYABLE protocol: every method, ours included, chooses
its knob from warm-up data only, and all methods are scored on the same
held-out split with overhead-inclusive costs.

    python scripts/ntc_primary_stats.py --probes experiments/ntc/w1_*.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("oc", _here / "ntc_operating_curves.py")
OC = importlib.util.module_from_spec(spec)
sys.modules["oc"] = OC
spec.loader.exec_module(OC)
S = OC.S


def binom_sf(k, n, p=0.5):
    """P(X >= k) for X ~ Bin(n, p) — exact, no SciPy."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(diffs):
    """Two-sided exact sign test on non-zero paired differences."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return float("nan"), 0, 0
    pos = sum(1 for x in d if x > 0)
    k = max(pos, n - pos)
    return min(1.0, 2 * binom_sf(k, n)), pos, n


def wilcoxon_z(diffs):
    """Signed-rank statistic, normal approximation with tie-corrected ranks."""
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
    """Group generation seeds of the same benchmark x model into one setting."""
    return f"{d['benchmark']}/{d['model'].split('/')[-1]}"


def aucc_for_file(traces, bench, split_seed):
    warm, ev = OC.split(traces, seed=split_seed)
    van_tok = float(np.mean([t["n_total_tokens"] for t in ev]))
    van_acc = float(np.mean([t["natural_correct"] for t in ev]))
    out = {}
    for name, (fn, grid) in OC.SWEEPS.items():
        pts = OC.curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc)
        out[name] = OC.metrics(pts, van_acc)["aucc"]
    out["NTC-full (ours)"] = OC.metrics(
        OC.curve_ntc_full(warm, ev, bench, van_tok, van_acc), van_acc)["aucc"]
    return out


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

    per_setting = {}          # setting -> method -> (mean, sd, n_reps)
    for key, files in sorted(groups.items()):
        reps = defaultdict(list)
        bench = files[0][1]["benchmark"]
        for pf, d in files:
            traces = d["traces"]
            for t in traces:
                t["natural_correct"] = bool(
                    S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            S.enrich_probes_with_nll(traces)
            for sd_ in range(a.splits):
                for m, v in aucc_for_file(traces, bench, sd_).items():
                    reps[m].append(v)
        per_setting[key] = {m: (float(np.mean(v)), float(np.std(v)), len(v))
                            for m, v in reps.items()}
        n_rep = max(len(v) for v in reps.values())
        print(f"\n=== {key}  ({len(files)} seed file(s) x {a.splits} splits "
              f"= {n_rep} replicates) ===")
        for m, (mu, sd, _) in sorted(per_setting[key].items(), key=lambda kv: -kv[1][0]):
            print(f"  {m:24s} AUCC {mu:.3f} ± {sd:.3f}")

    methods = sorted({m for v in per_setting.values() for m in v})
    OURS = "NTC-full (ours)"
    print("\n" + "=" * 78)
    print(f"AGGREGATE ACROSS {len(per_setting)} SETTINGS  (paired tests vs {OURS})")
    print("=" * 78)
    print(f"{'method':24s} {'mean AUCC':>11s} {'sd':>7s} {'min':>7s} "
          f"{'wins':>7s} {'sign p':>9s} {'Wilcoxon z':>11s}")

    md = ["# Primary metric with uncertainty and paired significance", "",
          f"Deployable protocol. {a.splits} random calibration/evaluation splits per "
          "generation-seed file; AUCC averaged within each setting, then aggregated "
          f"across {len(per_setting)} settings. `wins` counts settings where "
          "NTC-full has the higher AUCC; the sign test is exact (two-sided) and the "
          "Wilcoxon statistic is the tie-corrected normal approximation.", "",
          "| method | mean AUCC | s.d. across settings | min | NTC-full wins | sign-test p | Wilcoxon z |",
          "|---|---|---|---|---|---|---|"]
    rows = []
    for m in methods:
        vals = [per_setting[k][m][0] for k in per_setting if m in per_setting[k]]
        if not vals:
            continue
        if m == OURS:
            rows.append((m, float(np.mean(vals)), float(np.std(vals)),
                         float(np.min(vals)), None, float("nan"), float("nan")))
            continue
        diffs = [per_setting[k][OURS][0] - per_setting[k][m][0]
                 for k in per_setting if m in per_setting[k] and OURS in per_setting[k]]
        p, pos, n = sign_test(diffs)
        z = wilcoxon_z(diffs)
        rows.append((m, float(np.mean(vals)), float(np.std(vals)),
                     float(np.min(vals)), f"{pos}/{n}", p, z))
    for m, mu, sd, mn, w, p, z in sorted(rows, key=lambda r: -r[1]):
        ws = "—" if w is None else w
        ps = "—" if w is None else (f"{p:.4f}" + ("*" if p < 0.05 else ""))
        zs = "—" if w is None else f"{z:+.2f}"
        print(f"{m:24s} {mu:11.3f} {sd:7.3f} {mn:7.3f} {ws:>7s} {ps:>9s} {zs:>11s}")
        md.append(f"| {m} | {mu:.3f} | {sd:.3f} | {mn:.3f} | {ws} | {ps} | {zs} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
