#!/usr/bin/env python
"""P1 FIX #6 (v2) — empirical validation of Proposition 2, done correctly.

v1 was wrong in two ways and is superseded:
  * the bound (K-m+1)*rho^(m-1)*q_max exceeded 1 in every setting, so after
    clipping it carried no information (and a rank correlation over identical
    values silently reduced to input-file order -- a spurious result);
  * rho and q_max were estimated over ALL trial answers, so a model that
    converges CORRECTLY (GSM8K) scored as highly "spurious", which is the
    opposite of what the proposition is about.

Corrected formulation. Spurious agreement is agreement on a WRONG answer, so
we estimate the stickiness of incorrect trial answers:

    rho_w  = P(a_{k+1} = a_k | a_k incorrect)        (persistence of an error)
    q_w    = modal mass of incorrect answers among all probes of an item
    P_spur = rho_w^(m-1) * q_w                        in [0, 1], no clipping

and compare it against two observables, both estimated from the same traces:

    lost-correct risk  = P(vanilla correct AND AGREE(m) halts on a wrong answer)
    AGREE delta        = acc(AGREE) - acc(vanilla)   (points)

Proposition 2 predicts P_spur to be monotone in the lost-correct risk and
anti-monotone in the AGREE delta. Rank correlations use average ranks for
ties, and we report n so the reader can judge the evidence.

    python scripts/ntc_prop2_validate.py --probes experiments/ntc/w1_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def rankdata(a):
    """Ranks with average ties (avoids the degenerate all-equal case)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(traces, bench, m=3):
    rho_w, q_w, lost = [], [], []
    for t in traces:
        pr = [p for p in t["probes"] if p.get("answer")]
        if len(pr) < 2:
            continue
        ans = [p["answer"] for p in pr]
        wrong = [not S.is_correct(a, t["gold"], bench) for a in ans]
        # persistence of an incorrect answer
        pairs = [1.0 if S.is_correct(ans[i + 1], ans[i], bench) else 0.0
                 for i in range(len(ans) - 1) if wrong[i]]
        if pairs:
            rho_w.append(float(np.mean(pairs)))
        bad = [a for a, w in zip(ans, wrong) if w]
        q_w.append(Counter(bad).most_common(1)[0][1] / len(ans) if bad else 0.0)
        # observed lost-correct risk of AGREE(m) on this item
        k = S.agree_policy(t["probes"], m=m, bm=bench)
        if t["natural_correct"]:
            halted_wrong = (k is not None and
                            not S.is_correct(t["probes"][k]["answer"],
                                             t["gold"], bench))
            lost.append(1.0 if halted_wrong else 0.0)
    if not rho_w:
        return None
    r, q = float(np.mean(rho_w)), float(np.mean(q_w))
    return {"rho_w": r, "q_w": q, "p_spur": (r ** (m - 1)) * q,
            "lost": float(np.mean(lost)) if lost else float("nan"),
            "n": len(traces)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", default="experiments/ntc/PROP2_VALIDATION.md")
    a = ap.parse_args()

    rows = []
    for pf in a.probes:
        d = json.loads(Path(pf).read_text())
        traces, bench = d["traces"], d["benchmark"]
        model = d["model"].split("/")[-1]
        for t in traces:
            t["natural_correct"] = bool(
                S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
        st = analyse(traces, bench, a.m)
        if st is None:
            continue
        van = float(np.mean([t["natural_correct"] for t in traces]))
        ok, _ = S.per_item(traces, bench, S.agree_policy, {"m": a.m})
        st.update(tag=f"{bench}/{model}",
                  delta=100 * (float(np.mean(ok)) - van))
        rows.append(st)
        print(f"{bench:14s} {model:12s} rho_w={st['rho_w']:.3f} "
              f"q_w={st['q_w']:.3f} P_spur={st['p_spur']:.3f} | "
              f"lost-correct risk={st['lost']:.3f}  AGREE Δ={st['delta']:+.1f}")

    s_risk = spearman([r["p_spur"] for r in rows], [r["lost"] for r in rows])
    s_delta = spearman([r["p_spur"] for r in rows], [r["delta"] for r in rows])
    print(f"\nn={len(rows)} settings")
    print(f"Spearman(P_spur, lost-correct risk) = {s_risk:+.3f}  "
          f"(positive supports Prop. 2)")
    print(f"Spearman(P_spur, AGREE Δ)           = {s_delta:+.3f}  "
          f"(negative supports Prop. 2)")

    if len(rows) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
        for ax, key, lbl, ttl in (
                (axes[0], "lost", "observed lost-correct risk",
                 f"Spearman {s_risk:+.2f}"),
                (axes[1], "delta", "observed AGREE Δ accuracy (pts)",
                 f"Spearman {s_delta:+.2f}")):
            xs = [r["p_spur"] for r in rows]
            ys = [r[key] for r in rows]
            ax.scatter(xs, ys, s=44, c="#b5342b")
            for r in rows:
                ax.annotate(r["tag"].split("/")[0], (r["p_spur"], r[key]),
                            fontsize=7, xytext=(4, 4),
                            textcoords="offset points")
            ax.set_xlabel(f"Prop. 2 spurious-agreement probability (m={a.m})")
            ax.set_ylabel(lbl)
            ax.set_title(ttl, fontsize=10)
            if key == "delta":
                ax.axhline(0, color="#888", lw=.8)
        fig.suptitle("Error stickiness predicts agreement collapse", fontsize=11)
        fig.tight_layout()
        Path("paper_figures").mkdir(exist_ok=True)
        for e in ("png", "pdf"):
            fig.savefig(f"paper_figures/fig2_prop2_validation.{e}", dpi=300)
        print("figure: paper_figures/fig2_prop2_validation.png (+pdf)")

    md = ["# Proposition 2 — empirical validation (corrected estimator)",
          f"rho_w / q_w estimated over INCORRECT trial answers only; m={a.m}.",
          "", "| benchmark/model | rho_w | q_w | P_spur | lost-correct risk "
          "| AGREE Δ (pts) | n |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['rho_w']:.3f} | {r['q_w']:.3f} "
                  f"| {r['p_spur']:.3f} | {r['lost']:.3f} | {r['delta']:+.1f} "
                  f"| {r['n']} |")
    md += ["", f"n = {len(rows)} settings.",
           f"Spearman(P_spur, lost-correct risk) = {s_risk:+.3f} "
           "(positive supports Prop. 2).",
           f"Spearman(P_spur, AGREE delta) = {s_delta:+.3f} "
           "(negative supports Prop. 2)."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
