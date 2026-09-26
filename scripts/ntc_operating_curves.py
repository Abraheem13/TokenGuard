#!/usr/bin/env python
"""Operating curves of every halting rule under two tuning protocols.

Each setting is split once into calibration (40%) and evaluation (60%) items,
and every rule is scored on the same evaluation items with overhead-inclusive
costs. The protocols differ only in where a rule's parameter is chosen:

  oracle      each rule gets its best parameter per budget, chosen on the
              evaluation items (an upper bound; not realisable in deployment)
  deployable  each rule chooses its parameter from the calibration items: the
              cheapest parameter whose calibration accuracy change is within
              eps of full generation, for each eps in EPS_GRID

Metrics, identical under both protocols:

  A(b)      best accuracy attainable at cost <= b x full-generation cost
  AUCC      mean of A(b) over b in {0.4, ..., 1.0}; coverage = share of b served
  A(0.5)    accuracy at half the full-generation cost
  B*(1 pt)  cheapest budget within one accuracy point of full generation

The module also provides the curve and metric functions used by
ntc_primary_stats.py and ntc_shift_certificate.py. Writes
experiments/ntc/OPERATING_CURVES.md.

    python scripts/ntc_operating_curves.py --probes experiments/ntc/w1_gsm8k_Qwen3-4B.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ntc_w1_stats as S

SWEEPS = {
    "Confidence (DEER-λ)": (S.deer_policy,
        [{"lam": v} for v in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999)]),
    "Entropy (EAT)": (S.eat_policy,
        [{"delta": v} for v in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5)]),
    "Smoothed confidence": (S.ntc_conf_policy,
        [{"theta": v} for v in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)]),
    "Answer agreement": (S.agree_policy, [{"m": v} for v in (2, 3, 4, 5)]),
    "NTC-v2 (fusion)": (S.ntc_v2_policy,
        [{"m": m, "theta": t} for m in (2, 3, 4) for t in (0.3, 0.5, 0.7, 0.9)]),
}
EPS_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
BUDGETS = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def split(traces, warm_frac=0.4, seed=0):
    """Random calibration/evaluation split of a list of traces."""
    idx = np.random.default_rng(seed).permutation(len(traces))
    c = int(len(traces) * warm_frac)
    return [traces[i] for i in idx[:c]], [traces[i] for i in idx[c:]]


def evaluate(items, bench, fn, kw):
    ok, tok = S.per_item(items, bench, fn, kw)
    return float(ok.mean()), float(tok.mean())


def curve_oracle(ev, bench, fn, grid, van_tok, van_acc):
    """One (relative cost, accuracy, label) point per parameter, scored on `ev`."""
    pts = [(1.0, van_acc, "never-halt")]
    for kw in grid:
        a, c = evaluate(ev, bench, fn, kw)
        pts.append((c / van_tok, a, str(kw)))
    return pts


def curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc):
    """One point per eps: parameter chosen on `warm`, scored on `ev`."""
    wv_acc = float(np.mean([t["natural_correct"] for t in warm]))
    cand = []
    for kw in grid:
        a, c = evaluate(warm, bench, fn, kw)
        cand.append((kw, a - wv_acc, c))
    pts = [(1.0, van_acc, "never-halt")]
    for eps in EPS_GRID:
        feas = [(kw, d, c) for kw, d, c in cand if d >= -eps]
        if not feas:
            continue
        kw = min(feas, key=lambda x: x[2])[0]
        a, c = evaluate(ev, bench, fn, kw)
        pts.append((c / van_tok, a, f"eps={eps}->{kw}"))
    return pts


def curve_ntc_full(warm, ev, bench, van_tok, van_acc):
    """One point per eps for the selection tier (library-wide calibration)."""
    pts = [(1.0, van_acc, "never-halt")]
    for eps in EPS_GRID:
        try:
            _, (fam, kw) = S.calibrate(warm, bench, eps=eps)
        except Exception:
            continue
        a, c = evaluate(ev, bench, S.FAMILIES[fam][0], kw)
        pts.append((c / van_tok, a, f"eps={eps}->{fam}{kw}"))
    return pts


def metrics(pts, van_acc, eps=0.01):
    A = []
    for b in BUDGETS:
        f = [a for (c, a, _) in pts if c <= b + 1e-9]
        A.append(max(f) if f else np.nan)
    A = np.array(A, dtype=float)
    ok = [c for (c, a, _) in pts if a >= van_acc - eps]
    return {"A": A, "aucc": float(np.nanmean(A)),
            "cover": float(np.mean(~np.isnan(A))),
            "at50": A[list(BUDGETS).index(0.5)],
            "bstar": min(ok) if ok else np.nan}


def show(title, table, md):
    print(f"\n  {title}")
    print(f"  {'method':24s} {'AUCC':>7s} {'cover':>7s} {'A(0.5)':>8s} {'B*(1pt)':>9s}")
    md += ["", f"**{title}**", "",
           "| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |", "|---|---|---|---|---|"]
    for name, m in sorted(table.items(), key=lambda kv: -kv[1]["aucc"]):
        bs = "n/a" if np.isnan(m["bstar"]) else f"{m['bstar']:.2f}"
        a5 = "n/a" if np.isnan(m["at50"]) else f"{m['at50']:.3f}"
        print(f"  {name:24s} {m['aucc']:7.3f} {m['cover']:7.0%} {a5:>8s} {bs:>9s}")
        md.append(f"| {name} | {m['aucc']:.3f} | {m['cover']:.0%} | {a5} | {bs} |")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--out", default="experiments/ntc/OPERATING_CURVES.md")
    a = ap.parse_args()

    md = ["# Operating curves under the oracle and deployable protocols", "",
          "One calibration/evaluation split (seed 0) per setting. `oracle`: each rule's "
          "parameter is chosen per budget on the evaluation items (an upper bound, not "
          "realisable in deployment). `deployable`: each rule chooses its parameter "
          "from the calibration items. Both protocols are scored on the same evaluation "
          "items with overhead-inclusive costs. `n/a`: no parameter reaches that budget.", ""]
    agg = {}
    for pf in a.probes:
        d = json.loads(Path(pf).read_text())
        traces, bench = d["traces"], d["benchmark"]
        model = d["model"].split("/")[-1]
        for t in traces:
            t["natural_correct"] = bool(
                S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
        S.enrich_probes_with_nll(traces)
        warm, ev = split(traces)
        van_tok = float(np.mean([t["n_total_tokens"] for t in ev]))
        van_acc = float(np.mean([t["natural_correct"] for t in ev]))

        orc, dep = {}, {}
        for name, (fn, grid) in SWEEPS.items():
            orc[name] = metrics(curve_oracle(ev, bench, fn, grid, van_tok, van_acc), van_acc)
            pts = curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc)
            dep[name] = metrics(pts, van_acc)
            at1 = [acc for _, acc, lb in pts if lb.startswith("eps=0.01-")]
            dep[name]["slo_dev"] = 100.0 * (at1[0] - van_acc) if at1 else 0.0
        nf = curve_ntc_full(warm, ev, bench, van_tok, van_acc)
        orc["NTC-full (selection)"] = dep["NTC-full (selection)"] = metrics(nf, van_acc)
        at1 = [acc for _, acc, lb in nf if lb.startswith("eps=0.01-")]
        dep["NTC-full (selection)"]["slo_dev"] = 100.0 * (at1[0] - van_acc) if at1 else 0.0

        tag = f"{bench} / {model}"
        print(f"\n=== {tag}   (evaluation n={len(ev)}, full-generation accuracy {van_acc:.3f}) ===")
        md.append(f"## {tag}: evaluation n={len(ev)}, full-generation accuracy {van_acc:.3f}")
        md = show("Oracle protocol (parameter chosen on the evaluation items)", orc, md)
        md = show("Deployable protocol (parameter chosen on the calibration items)", dep, md)
        for nm, m in dep.items():
            agg.setdefault(nm, []).append((m["aucc"], m.get("slo_dev", float("nan"))))

    if agg:
        n_set = len(a.probes)
        print(f"\nCross-setting aggregate, deployable protocol, {n_set} settings")
        print(f"{'method':24s} {'mean AUCC':>10s} {'min AUCC':>10s} {'worst Δacc':>11s}")
        md += ["", "## Cross-setting aggregate (deployable protocol)", "",
               "`worst Δ accuracy` is the lowest accuracy change, over settings, of the "
               "point each rule deploys at a one-point tolerance (eps = 0.01).", "",
               "| method | mean AUCC | min AUCC | worst Δ accuracy (pts) |",
               "|---|---|---|---|"]
        rank = sorted(agg.items(), key=lambda kv: -sum(x for x, _ in kv[1]) / len(kv[1]))
        for nm, vals in rank:
            mu = sum(x for x, _ in vals) / len(vals)
            mn = min(x for x, _ in vals)
            wd = min(dv for _, dv in vals)
            print(f"{nm:24s} {mu:10.3f} {mn:10.3f} {wd:+11.1f}")
            md.append(f"| {nm} | {mu:.3f} | {mn:.3f} | {wd:+.1f} |")
        md.append("")

    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
