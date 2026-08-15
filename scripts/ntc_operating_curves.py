#!/usr/bin/env python
"""PRIMARY EVALUATION v2 — two comparison protocols, one held-out split.

Fixes two fairness defects of v1:

  * v1 let every fixed policy pick its best knob **on the evaluation data**,
    which no deployment can do, while our controller had to choose from
    warm-up data only. v2 reports BOTH protocols explicitly.
  * v1 scored fixed policies on the full set and the controller on the
    held-out split, so the numbers had different denominators. v2 scores
    everything on the same held-out split.

Protocols
---------
ORACLE      each fixed policy is given its best knob per budget, chosen on the
            evaluation data. An upper bound on what the SIGNAL could do, not
            what a user can ship. Baselines are flattered here by design.
DEPLOYABLE  every method, ours included, must choose its knob from the warm-up
            split alone (cheapest knob whose warm-up accuracy deficit is within
            eps). This is what a practitioner actually gets.

Metrics (identical in both protocols)
  A(b)      best accuracy attainable at cost <= b x full-thinking cost
  AUCC      mean of A(b) over b in {0.4 ... 1.0};  coverage = fraction served
  A(0.5)    matched-budget accuracy at half the tokens
  B*(eps)   cheapest budget still within eps accuracy points of full thinking

All costs are overhead-inclusive. 

    python scripts/ntc_operating_curves.py --probes experiments/ntc/w1_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("w1s", _here / "ntc_w1_stats.py")
S = importlib.util.module_from_spec(spec)
sys.modules["w1s"] = S
spec.loader.exec_module(S)

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
    idx = np.random.default_rng(seed).permutation(len(traces))
    c = int(len(traces) * warm_frac)
    return [traces[i] for i in idx[:c]], [traces[i] for i in idx[c:]]


def evaluate(items, bench, fn, kw):
    ok, tok = S.per_item(items, bench, fn, kw)
    return float(ok.mean()), float(tok.mean())


def curve_oracle(ev, bench, fn, grid, van_tok, van_acc):
    pts = [(1.0, van_acc, "never-halt")]
    for kw in grid:
        a, c = evaluate(ev, bench, fn, kw)
        pts.append((c / van_tok, a, str(kw)))
    return pts


def curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc):
    """One point per eps: knob chosen on WARM-UP, scored on eval."""
    wv_acc = float(np.mean([t["natural_correct"] for t in warm]))
    cand = []
    for kw in grid:
        a, c = evaluate(warm, bench, fn, kw)
        cand.append((kw, a - wv_acc, c))
    pts = [(1.0, van_acc, "never-halt")]
    for eps in EPS_GRID:
        feas = [(kw, d, c) for kw, d, c in cand if d >= -eps]
        if not feas:
            continue                      # nothing safe at this eps -> never halt
        kw = min(feas, key=lambda x: x[2])[0]
        a, c = evaluate(ev, bench, fn, kw)
        pts.append((c / van_tok, a, f"eps={eps}->{kw}"))
    return pts


def curve_ntc_full(warm, ev, bench, van_tok, van_acc):
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
    md += [f"", f"**{title}**", "",
           "| method | AUCC | coverage | A(b=0.5) | B*(1 pt) |", "|---|---|---|---|---|"]
    for name, m in sorted(table.items(), key=lambda kv: -kv[1]["aucc"]):
        bs = "—" if np.isnan(m["bstar"]) else f"{m['bstar']:.2f}"
        a5 = "—" if np.isnan(m["at50"]) else f"{m['at50']:.3f}"
        print(f"  {name:24s} {m['aucc']:7.3f} {m['cover']:7.0%} {a5:>8s} {bs:>9s}")
        md.append(f"| {name} | {m['aucc']:.3f} | {m['cover']:.0%} | {a5} | {bs} |")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--out", default="experiments/ntc/OPERATING_CURVES.md")
    a = ap.parse_args()

    md = ["# Primary evaluation — operating curves under two protocols", "",
          "`ORACLE` gives each fixed policy its best knob per budget, chosen on the "
          "evaluation data (an upper bound on the signal, not deployable). "
          "`DEPLOYABLE` makes every method choose its knob from the warm-up split "
          "only — what a user can actually ship. Both are scored on the same "
          "held-out split with overhead-inclusive costs.", ""]
    figdata = []
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
            dep[name] = metrics(curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc), van_acc)
        nf = curve_ntc_full(warm, ev, bench, van_tok, van_acc)
        orc["NTC-full (ours)"] = dep["NTC-full (ours)"] = metrics(nf, van_acc)

        tag = f"{bench} / {model}"
        print(f"\n=== {tag}   (held-out n={len(ev)}, vanilla acc {van_acc:.3f}) ===")
        md.append(f"## {tag} — held-out n={len(ev)}, vanilla accuracy {van_acc:.3f}")
        md = show("ORACLE knob (baselines flattered)", orc, md)
        md = show("DEPLOYABLE knob (what ships)", dep, md)
        for nm, m in dep.items():
            agg.setdefault(nm, []).append(
                (m["aucc"], 100.0 * (m["A"][-1] - van_acc)))
        figdata.append((tag, ev, bench, van_tok, van_acc, warm))

    # ---- CROSS-SETTING AGGREGATE (the decision-relevant summary) ----
    if agg:
        print("\n" + "=" * 74)
        print("CROSS-SETTING AGGREGATE — deployable protocol, "
              f"{len(figdata)} settings")
        print("=" * 74)
        print(f"{'method':24s} {'mean AUCC':>10s} {'min AUCC':>10s} "
              f"{'worst Δacc':>11s}")
        md += ["", "## Cross-setting aggregate (deployable protocol)", "",
               "A deployer chooses one method for a workload mix and cannot know "
               "which benchmark arrives next, so the decision-relevant summary is "
               "the aggregate over settings, not the per-setting winner.", "",
               "| method | mean AUCC | min AUCC | worst Δ accuracy (pts) |",
               "|---|---|---|---|"]
        rank = sorted(agg.items(), key=lambda kv: -sum(a for a, _ in kv[1]) / len(kv[1]))
        for nm, vals in rank:
            mu = sum(a for a, _ in vals) / len(vals)
            mn = min(a for a, _ in vals)
            wd = min(d for _, d in vals)
            print(f"{nm:24s} {mu:10.3f} {mn:10.3f} {wd:+11.1f}")
            md.append(f"| {nm} | {mu:.3f} | {mn:.3f} | {wd:+.1f} |")
        md.append("")

    k = min(4, len(figdata))
    if k:
        fig, axes = plt.subplots(1, k, figsize=(4.3 * k, 3.9), squeeze=False)
        for j in range(k):
            tag, ev, bench, van_tok, van_acc, warm = figdata[j]
            ax = axes[0][j]
            for name, (fn, grid) in SWEEPS.items():
                pts = sorted(curve_deployable(warm, ev, bench, fn, grid, van_tok, van_acc))
                ax.plot([c for c, _, _ in pts], [a for _, a, _ in pts],
                        marker="o", ms=3.2, lw=1.3, alpha=.85, label=name)
            pts = sorted(curve_ntc_full(warm, ev, bench, van_tok, van_acc))
            ax.plot([c for c, _, _ in pts], [a for _, a, _ in pts],
                    marker="s", ms=4.5, lw=2.2, color="#0b6e4f", label="NTC-full (ours)")
            ax.axhline(van_acc, color="#555", ls="--", lw=.9)
            ax.set_xlabel("cost / full thinking")
            if j == 0:
                ax.set_ylabel("accuracy")
            ax.set_title(tag, fontsize=9)
        axes[0][0].legend(fontsize=6.5, loc="lower right")
        fig.suptitle("Deployable operating curves — knob chosen on warm-up only",
                     fontsize=11)
        fig.tight_layout()
        Path("paper_figures").mkdir(exist_ok=True)
        for e in ("png", "pdf"):
            fig.savefig(f"paper_figures/fig3_operating_curves.{e}", dpi=300)
        print("\nfigure: paper_figures/fig3_operating_curves.png (+pdf)")

    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
