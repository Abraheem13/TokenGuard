#!/usr/bin/env python
"""Error stickiness against the observed failures of answer agreement.

Spurious agreement is agreement on a wrong answer. Estimated from incorrect
trial answers only:

    rho_w  = P(a_{k+1} = a_k | a_k incorrect)        persistence of an error
    q_w    = modal share of incorrect answers among an item's probes
    P_spur = rho_w^(m-1) * q_w                        in [0, 1]

and compared, across settings, with two observables of the same traces:

    lost-correct risk  = P(full generation correct and agreement(m) halts on a wrong answer)
    agreement delta    = acc(agreement) - acc(full generation), in points

Proposition 2 predicts P_spur to rise with the lost-correct risk and to fall
with the agreement delta. Spearman correlations use average ranks for ties.

Writes experiments/ntc/PROP2_VALIDATION.md.

    python scripts/ntc_prop2_validate.py --probes experiments/ntc/w1_gsm8k_Qwen3-4B.json ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import ntc_w1_stats as S


def rankdata(a):
    """Ranks with ties given their average rank."""
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
        pairs = [1.0 if S.is_correct(ans[i + 1], ans[i], bench) else 0.0
                 for i in range(len(ans) - 1) if wrong[i]]
        if pairs:
            rho_w.append(float(np.mean(pairs)))
        bad = [a for a, w in zip(ans, wrong) if w]
        q_w.append(Counter(bad).most_common(1)[0][1] / len(ans) if bad else 0.0)
        k = S.agree_policy(t["probes"], m=m, bm=bench)
        if t["natural_correct"]:
            halted_wrong = (k is not None and
                            not S.is_correct(t["probes"][k]["answer"], t["gold"], bench))
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
        st.update(tag=f"{bench}/{model}", delta=100 * (float(np.mean(ok)) - van))
        rows.append(st)
        print(f"{bench:14s} {model:12s} rho_w={st['rho_w']:.3f} q_w={st['q_w']:.3f} "
              f"P_spur={st['p_spur']:.3f} | lost-correct risk={st['lost']:.3f}  "
              f"AGREE Δ={st['delta']:+.1f}")

    s_risk = spearman([r["p_spur"] for r in rows], [r["lost"] for r in rows])
    s_delta = spearman([r["p_spur"] for r in rows], [r["delta"] for r in rows])
    print(f"\nn={len(rows)} settings")
    print(f"Spearman(P_spur, lost-correct risk) = {s_risk:+.3f}")
    print(f"Spearman(P_spur, AGREE Δ)           = {s_delta:+.3f}")

    md = ["# Error stickiness and the failures of answer agreement", "",
          f"rho_w and q_w are estimated over incorrect trial answers only; m = {a.m}. "
          "`AGREE Δ` is the accuracy change of answer agreement against full generation.",
          "", "| benchmark/model | rho_w | q_w | P_spur | lost-correct risk "
          "| AGREE Δ (pts) | n |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['rho_w']:.3f} | {r['q_w']:.3f} "
                  f"| {r['p_spur']:.3f} | {r['lost']:.3f} | {r['delta']:+.1f} | {r['n']} |")
    md += ["", f"n = {len(rows)} settings.",
           f"Spearman(P_spur, lost-correct risk) = {s_risk:+.3f} "
           "(Proposition 2 predicts a positive value).",
           f"Spearman(P_spur, AGREE delta) = {s_delta:+.3f} "
           "(Proposition 2 predicts a negative value)."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
