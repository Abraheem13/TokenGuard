#!/usr/bin/env python
"""P1 FIX #6 — empirical validation of Proposition 2 (spurious agreement).

Prop. 2 models trial answers on an undecided item as a stationary process over
an effective answer set A with self-transition (persistence) rho and modal mass
q_max, giving a spurious m-run probability bounded below by

    P_spur(m, K) >= (K - m + 1) * rho^(m-1) * q_max

Here we ESTIMATE rho, q_max and |A| directly from the recorded probe streams of
every benchmark, compute the predicted bound, and correlate it with the
OBSERVED agreement collapse (AGREE accuracy minus vanilla accuracy). A
theory that predicts the ordering of collapse severity is far stronger than a
post-hoc explanation.

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


def stats_for(traces, bench, m=3):
    """Estimate |A|, rho, q_max and the Prop-2 bound on UNDECIDED items."""
    rhos, qs, sizes, Ks = [], [], [], []
    for t in traces:
        pr = [p for p in t["probes"] if p.get("answer")]
        if len(pr) < 2:
            continue
        ans = [p["answer"] for p in pr]
        # "undecided" = the trace does not settle on the gold answer throughout
        settled = all(S.is_correct(a, t["gold"], bench) for a in ans)
        if settled:
            continue
        same = [1.0 if S.is_correct(ans[i], ans[i - 1], bench) else 0.0
                for i in range(1, len(ans))]
        cnt = Counter(ans)
        rhos.append(float(np.mean(same)))
        qs.append(cnt.most_common(1)[0][1] / len(ans))
        sizes.append(len(cnt))
        Ks.append(len(ans))
    if not rhos:
        return None
    rho, q, sz, K = (float(np.mean(rhos)), float(np.mean(qs)),
                     float(np.mean(sizes)), float(np.mean(Ks)))
    bound = max(0.0, (K - m + 1)) * (rho ** (m - 1)) * q
    return {"rho": rho, "q_max": q, "eff_A": sz, "K": K,
            "bound": min(1.0, bound), "n_undecided": len(rhos)}


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
        st = stats_for(traces, bench, a.m)
        if st is None:
            continue
        van = float(np.mean([t["natural_correct"] for t in traces]))
        ok, _ = S.per_item(traces, bench, S.agree_policy, {"m": a.m})
        delta = 100 * (float(np.mean(ok)) - van)
        rows.append({"tag": f"{bench}/{model}", **st, "delta": delta})
        print(f"{bench:14s} {model:12s} |A|~{st['eff_A']:.2f} rho={st['rho']:.3f} "
              f"q={st['q_max']:.3f} bound={st['bound']:.3f}  observed AGREE "
              f"delta={delta:+.1f} pts  (n_undec={st['n_undecided']})")

    if len(rows) >= 3:
        x = np.array([r["bound"] for r in rows])
        y = np.array([r["delta"] for r in rows])
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        rho_s = float(np.corrcoef(rx, ry)[0, 1])
        print(f"\nSpearman(predicted bound, observed delta) = {rho_s:+.3f} "
              f"(negative = theory predicts collapse ordering)")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(x, y, s=46, c="#c0392b")
        for r in rows:
            ax.annotate(r["tag"], (r["bound"], r["delta"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="#888", lw=.8)
        ax.set_xlabel(f"Prop. 2 spurious-agreement bound (m={a.m})")
        ax.set_ylabel("observed AGREE Δ accuracy (points)")
        ax.set_title(f"Theory predicts collapse severity (Spearman {rho_s:+.2f})",
                     fontsize=10)
        fig.tight_layout()
        Path("paper_figures").mkdir(exist_ok=True)
        for e in ("png", "pdf"):
            fig.savefig(f"paper_figures/fig2_prop2_validation.{e}", dpi=300)
        print("figure: paper_figures/fig2_prop2_validation.png (+pdf)")
    else:
        rho_s = float("nan")

    md = ["# Proposition 2 — empirical validation",
          f"Estimated on undecided items only; m={a.m}.", "",
          "| benchmark/model | eff. \\|A\\| | rho | q_max | K | bound | observed AGREE Δ |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['eff_A']:.2f} | {r['rho']:.3f} "
                  f"| {r['q_max']:.3f} | {r['K']:.1f} | {r['bound']:.3f} "
                  f"| {r['delta']:+.1f} |")
    md += ["", f"Spearman(bound, observed Δ) = {rho_s:+.3f}"]
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
