#!/usr/bin/env python
"""JOINT router+budget controller — the NTC unification result. No GPU.

Uses the two saved probe files for the SAME benchmark questions under two
model scales (e.g. Qwen3-1.7B and Qwen3-4B on MATH-500). The slow tier picks
each model's halting policy by paired-CV calibration on the warm-up; a
difficulty router (TF-IDF + logistic regression on warm-up labels: "is the
SMALL model correct under its calibrated halting?") then routes each held-out
query to the small or large model. Sweeping the routing threshold traces the
joint (model, budget) cost-accuracy frontier.

Cost model: relative compute per generated token proportional to parameter
count (FLOPs/token ~ 2N). Reported as normalized cost units AND raw tokens.

Baselines on the same held-out split: large-vanilla, large-NTC, small-NTC,
random model mix (no router). Claim tested: the JOINT controller reaches
operating points (accuracy at total-compute) that no single-model halting
baseline attains — the structural advantage of unifying routing with
budget control.

Usage:
    python scripts/ntc_joint_router.py \
        --small experiments/ntc/w1_math500_Qwen3-1.7B.json \
        --large experiments/ntc/w1_math500_Qwen3-4B.json \
        --small-params 1.7 --large-params 4.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tokenguard.reasoning.datasets import is_correct


# ---- policy families (identical to ntc_w1_stats) ----
def deer_policy(probes, lam=0.95):
    for k, p in enumerate(probes):
        if p["confidence"] >= lam:
            return k
    return None


def eat_policy(probes, delta=1e-3, alpha=0.2, warmup=3):
    ema = emv = None
    for k, p in enumerate(probes):
        h = p["first_entropy"]
        if ema is None:
            ema, emv = h, 0.0
        else:
            d = h - ema
            ema += alpha * d
            emv = (1 - alpha) * (emv + alpha * d * d)
        if k + 1 >= warmup and emv < delta:
            return k
    return None


def ntc_conf_policy(probes, theta=0.9, eta=0.6, patience=2):
    S, above = None, 0
    for k, p in enumerate(probes):
        c = p["confidence"]
        S = c if S is None else eta * S + (1 - eta) * c
        above = above + 1 if S >= theta else 0
        if above >= patience:
            return k
    return None


def agree_policy(probes, m=2, bm="math500"):
    run = 1
    for k in range(1, len(probes)):
        same = (probes[k]["answer"] and
                is_correct(probes[k]["answer"], probes[k - 1]["answer"], bm))
        run = run + 1 if same else 1
        if run >= m:
            return k
    return None


def ntc_v2_policy(probes, m=2, theta=0.5, eta=0.6, bm="math500"):
    S, run = None, 1
    for k in range(len(probes)):
        c = probes[k]["confidence"]
        S = c if S is None else eta * S + (1 - eta) * c
        if k >= 1:
            same = (probes[k]["answer"] and
                    is_correct(probes[k]["answer"], probes[k - 1]["answer"], bm))
            run = run + 1 if same else 1
            if run >= m and S >= theta:
                return k
    return None


FAMILIES = {
    "DEER":     (deer_policy,     [{"lam": v} for v in (0.90, 0.95, 0.99)]),
    "EAT":      (eat_policy,      [{"delta": v} for v in (1e-2, 1e-3, 1e-4)]),
    "NTC-conf": (ntc_conf_policy, [{"theta": v} for v in (0.85, 0.90, 0.95, 0.99)]),
    "AGREE":    (agree_policy,    [{"m": v} for v in (2, 3)]),
    "NTC-v2":   (ntc_v2_policy,   [{"m": m, "theta": t}
                                   for m, t in ((2, .3), (2, .5), (2, .7), (3, .3), (3, .5))]),
}


def per_item(traces, bench, fn, kw):
    ok, tok = [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
             if probes else None
        if kk is None:
            ok.append(bool(t["natural_correct"])); tok.append(t["n_total_tokens"])
        else:
            p = probes[kk]
            ok.append(is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"] + p["n_probe_tokens"])
    return np.array(ok), np.array(tok, dtype=float)


def calibrate_global(warm, bench, k_folds=5, eps=0.03):
    """Paired K-fold CV pick of (family, kw) — same rule as ntc_w1_stats."""
    n = len(warm)
    k_folds = max(2, min(k_folds, n))
    idx = np.arange(n)
    folds = [f for f in (idx[i::k_folds] for i in range(k_folds)) if len(f)]
    van_all = np.array([t["natural_correct"] for t in warm], dtype=float)
    van_by_fold = [van_all[f].mean() for f in folds]
    gcands = []
    for fam, (fn, grid) in FAMILIES.items():
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            ok = ok.astype(float)
            d = np.array([ok[f].mean() - van_by_fold[j]
                          for j, f in enumerate(folds)])
            md = float(d.mean())
            se = float(d.std(ddof=1) / math.sqrt(len(folds)))
            gcands.append({"fam": fam, "kw": kw, "md": md,
                           "tok": float(tok.mean()), "lcb": md - se})
    feas = [c for c in gcands if c["lcb"] >= -eps]
    g = (min(feas, key=lambda c: c["tok"]) if feas
         else max(gcands, key=lambda c: c["md"]))
    return g["fam"], g["kw"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--small-params", type=float, default=1.7,
                    help="relative cost/token of the small model (B params)")
    ap.add_argument("--large-params", type=float, default=4.0)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-eps", type=float, default=0.05,
                    help="accuracy-SLO tolerance for per-model policy calibration")
    ap.add_argument("--figdir", default="experiments/ntc/figures")
    ap.add_argument("--out", default="experiments/ntc/JOINT.md")
    args = ap.parse_args()

    ds = json.loads(Path(args.small).read_text())
    dl = json.loads(Path(args.large).read_text())
    assert ds["benchmark"] == dl["benchmark"], "benchmark mismatch"
    bench = ds["benchmark"]
    S = {t["qid"]: t for t in ds["traces"]}
    L = {t["qid"]: t for t in dl["traces"]}
    qids = [q for q in S if q in L]
    assert len(qids) >= 100, f"aligned questions too few: {len(qids)}"
    print(f"aligned questions: {len(qids)} ({ds['model']} vs {dl['model']})")

    rng = np.random.default_rng(args.seed)
    qids = [qids[i] for i in rng.permutation(len(qids))]
    n_warm = int(len(qids) * args.warmup_frac)
    warm_q, eval_q = qids[:n_warm], qids[n_warm:]

    warm_S = [S[q] for q in warm_q]; warm_L = [L[q] for q in warm_q]
    ev_S = [S[q] for q in eval_q];   ev_L = [L[q] for q in eval_q]

    # 1) slow tier: calibrated halting policy per model (warm-up only)
    famS, kwS = calibrate_global(warm_S, bench, eps=args.cv_eps)
    famL, kwL = calibrate_global(warm_L, bench, eps=args.cv_eps)
    print(f"small policy: {famS} {kwS} | large policy: {famL} {kwL}")

    okS_w, tokS_w = per_item(warm_S, bench, FAMILIES[famS][0], kwS)
    okS_e, tokS_e = per_item(ev_S, bench, FAMILIES[famS][0], kwS)
    okL_e, tokL_e = per_item(ev_L, bench, FAMILIES[famL][0], kwL)
    van_okL_e = np.array([t["natural_correct"] for t in ev_L])
    van_tokL_e = np.array([t["n_total_tokens"] for t in ev_L], dtype=float)

    cS, cL = args.small_params, args.large_params
    costS_e = tokS_e * cS
    costL_e = tokL_e * cL
    van_costL_e = van_tokL_e * cL

    # 2) medium tier: difficulty router — predict "small model suffices"
    #    from warm-up labels; TF-IDF + logistic regression (no GPU).
    texts_w = [t["question"] for t in warm_S]
    texts_e = [t["question"] for t in ev_S]
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    Xw = vec.fit_transform(texts_w)
    Xe = vec.transform(texts_e)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xw, okS_w.astype(int))
    p_small = clf.predict_proba(Xe)[:, 1]

    # 3) sweep routing threshold -> joint frontier on held-out
    pts = []
    for tau in np.linspace(0.0, 1.0, 21):
        use_small = p_small >= tau
        acc = np.where(use_small, okS_e, okL_e).mean()
        cost = np.where(use_small, costS_e, costL_e).mean()
        toks = np.where(use_small, tokS_e, tokL_e).mean()
        pts.append({"tau": float(tau), "acc": float(acc),
                    "cost": float(cost), "tokens": float(toks),
                    "frac_small": float(use_small.mean())})

    refs = {
        "large-vanilla": {"acc": float(van_okL_e.mean()),
                          "cost": float(van_costL_e.mean())},
        "large-NTC":     {"acc": float(okL_e.mean()), "cost": float(costL_e.mean())},
        "small-NTC":     {"acc": float(okS_e.mean()), "cost": float(costS_e.mean())},
    }
    # random-mix line (no router): convex combination of small-NTC and large-NTC
    mix = [{"acc": a * refs["small-NTC"]["acc"] + (1 - a) * refs["large-NTC"]["acc"],
            "cost": a * refs["small-NTC"]["cost"] + (1 - a) * refs["large-NTC"]["cost"]}
           for a in np.linspace(0, 1, 11)]

    # 4) report + figure
    print(f"\n=== JOINT frontier — {bench} held-out n={len(eval_q)} "
          f"(cost = tokens x params-B) ===")
    print(f"{'tau':>5}{'acc':>8}{'cost':>10}{'tokens':>9}{'%small':>8}")
    for p in pts[::2]:
        print(f"{p['tau']:>5.2f}{p['acc']:>8.3f}{p['cost']:>10.0f}"
              f"{p['tokens']:>9.0f}{100*p['frac_small']:>7.0f}%")
    for k, v in refs.items():
        print(f"{k:<14} acc={v['acc']:.3f} cost={v['cost']:.0f}")

    # headline: joint point matching large-NTC accuracy at min cost
    tgt = refs["large-NTC"]["acc"] - 0.01
    ok_pts = [p for p in pts if p["acc"] >= tgt]
    if ok_pts:
        best = min(ok_pts, key=lambda p: p["cost"])
        save = 100 * (1 - best["cost"] / refs["large-NTC"]["cost"])
        save_v = 100 * (1 - best["cost"] / refs["large-vanilla"]["cost"])
        print(f"\nHEADLINE: joint router+budget matches large-NTC accuracy "
              f"({best['acc']:.3f}) at {save:.0f}% less compute than large-NTC "
              f"and {save_v:.0f}% less than large-vanilla "
              f"(tau={best['tau']:.2f}, {100*best['frac_small']:.0f}% routed small)")

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot([p["cost"] for p in pts], [p["acc"] for p in pts], "-o", ms=3.5,
            color="crimson", label="NTC joint (router+budget)", zorder=5)
    ax.plot([m["cost"] for m in mix], [m["acc"] for m in mix], "--", lw=1.2,
            color="gray", label="random model mix (no router)")
    marks = {"large-vanilla": ("tab:orange", "*"), "large-NTC": ("tab:blue", "s"),
             "small-NTC": ("tab:green", "^")}
    for k, v in refs.items():
        c, mk = marks[k]
        ax.scatter([v["cost"]], [v["acc"]], marker=mk, s=110, color=c,
                   edgecolor="k", linewidth=0.4, zorder=6, label=k)
    ax.set_xlabel("Mean compute per problem (tokens × params-B, relative)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Joint model-routing + budget control · {bench}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.figdir}/pareto_joint_{bench}.{ext}", dpi=300)
    plt.close(fig)
    print(f"figure: {args.figdir}/pareto_joint_{bench}.png")

    md = [f"# Joint router+budget — {bench} (held-out n={len(eval_q)}, seed {args.seed})",
          f"small={ds['model']} (policy {famS} {kwS}), large={dl['model']} "
          f"(policy {famL} {kwL}); router=TF-IDF+LogReg on warm-up n={n_warm}.",
          "", "| tau | acc | cost | tokens | %small |", "|---|---|---|---|---|"]
    md += [f"| {p['tau']:.2f} | {p['acc']:.3f} | {p['cost']:.0f} "
           f"| {p['tokens']:.0f} | {100*p['frac_small']:.0f}% |" for p in pts]
    md += ["", "| reference | acc | cost |", "|---|---|---|"]
    md += [f"| {k} | {v['acc']:.3f} | {v['cost']:.0f} |" for k, v in refs.items()]
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
