#!/usr/bin/env python
"""Joint tier: route each query to a small or a large model, each with calibrated halting.

Uses probe files for the same questions under two model sizes (Qwen3-1.7B and
Qwen3-4B on MATH-500). On the calibration split each model's halting rule is
chosen by repeated paired cross-validation; a router (TF-IDF features and
logistic regression, trained on calibration labels "the small model is correct
under its calibrated rule") then sends each evaluation query to the small or
the large model. Sweeping the routing threshold tau traces the joint
cost-accuracy frontier.

Cost is tokens multiplied by the model's size in billions of parameters
(compute per token is proportional to parameter count), reported with raw
tokens. The references, on the same evaluation split, are the large model with
full generation, and each model alone with its calibrated rule.

Writes experiments/ntc/JOINT.md.

    python scripts/ntc_joint_router.py --small experiments/ntc/w1_math500_Qwen3-1.7B.json \
        --large experiments/ntc/w1_math500_Qwen3-4B.json --small-params 1.7 --large-params 4.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import ntc_w1_stats as S

# Candidate grid of this experiment (a subset of the selection tier's library).
FAMILIES = {
    "DEER": (S.deer_policy, [{"lam": v} for v in (0.90, 0.95, 0.99)]),
    "EAT": (S.eat_policy, [{"delta": v} for v in (1e-2, 1e-3, 1e-4)]),
    "NTC-conf": (S.ntc_conf_policy, [{"theta": v} for v in (0.85, 0.90, 0.95, 0.99)]),
    "AGREE": (S.agree_policy, [{"m": v} for v in (2, 3)]),
    "NTC-v2": (S.ntc_v2_policy,
               [{"m": m, "theta": t} for m, t in ((2, .3), (2, .5), (2, .7), (3, .3), (3, .5))]),
}


def per_item(traces, bench, fn, kw):
    """Per-item (correct, tokens); tokens are thinking to the halt plus the halting probe."""
    ok, tok = [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
            if probes else None
        if kk is None:
            ok.append(bool(t["natural_correct"]))
            tok.append(t["n_total_tokens"])
        else:
            p = probes[kk]
            ok.append(S.is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"] + p["n_probe_tokens"])
    return np.array(ok), np.array(tok, dtype=float)


def calibrate_global(warm, bench, k_folds=5, eps=0.025, reps=3):
    """Cheapest (family, kw) whose repeated paired K-fold accuracy change is >= -eps."""
    n = len(warm)
    k_folds = max(2, min(k_folds, n))
    van_all = np.array([t["natural_correct"] for t in warm], dtype=float)
    gcands = []
    for fam, (fn, grid) in FAMILIES.items():
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            ok = ok.astype(float)
            ds = []
            for r in range(reps):
                idx = np.random.default_rng(1000 + r).permutation(n)
                folds = [f for f in (idx[i::k_folds] for i in range(k_folds)) if len(f)]
                ds += [float(ok[f].mean() - van_all[f].mean()) for f in folds]
            gcands.append({"fam": fam, "kw": kw, "md": float(np.mean(ds)),
                           "tok": float(tok.mean())})
    feas = [c for c in gcands if c["md"] >= -eps]
    g = (min(feas, key=lambda c: c["tok"]) if feas
         else max(gcands, key=lambda c: c["md"]))
    return g["fam"], g["kw"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--small-params", type=float, default=1.7,
                    help="size of the small model in billions of parameters")
    ap.add_argument("--large-params", type=float, default=4.0,
                    help="size of the large model in billions of parameters")
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-eps", type=float, default=0.025,
                    help="accuracy tolerance of each model's halting calibration")
    ap.add_argument("--out", default="experiments/ntc/JOINT.md")
    args = ap.parse_args()

    ds = json.loads(Path(args.small).read_text())
    dl = json.loads(Path(args.large).read_text())
    if ds["benchmark"] != dl["benchmark"]:
        raise SystemExit("the two probe files must share a benchmark")
    bench = ds["benchmark"]
    small = {t["qid"]: t for t in ds["traces"]}
    large = {t["qid"]: t for t in dl["traces"]}
    qids = [q for q in small if q in large]
    print(f"aligned questions: {len(qids)} ({ds['model']} and {dl['model']})")

    rng = np.random.default_rng(args.seed)
    qids = [qids[i] for i in rng.permutation(len(qids))]
    n_warm = int(len(qids) * args.warmup_frac)
    warm_q, eval_q = qids[:n_warm], qids[n_warm:]
    warm_s = [small[q] for q in warm_q]
    warm_l = [large[q] for q in warm_q]
    ev_s = [small[q] for q in eval_q]
    ev_l = [large[q] for q in eval_q]

    fam_s, kw_s = calibrate_global(warm_s, bench, eps=args.cv_eps)
    fam_l, kw_l = calibrate_global(warm_l, bench, eps=args.cv_eps)
    print(f"small-model rule: {fam_s} {kw_s} | large-model rule: {fam_l} {kw_l}")

    ok_s_w, _ = per_item(warm_s, bench, FAMILIES[fam_s][0], kw_s)
    ok_s, tok_s = per_item(ev_s, bench, FAMILIES[fam_s][0], kw_s)
    ok_l, tok_l = per_item(ev_l, bench, FAMILIES[fam_l][0], kw_l)
    van_ok_l = np.array([t["natural_correct"] for t in ev_l])
    van_tok_l = np.array([t["n_total_tokens"] for t in ev_l], dtype=float)
    cost_s = tok_s * args.small_params
    cost_l = tok_l * args.large_params

    vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    x_warm = vec.fit_transform([t["question"] for t in warm_s])
    x_eval = vec.transform([t["question"] for t in ev_s])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_warm, ok_s_w.astype(int))
    p_small = clf.predict_proba(x_eval)[:, 1]

    pts = []
    for tau in np.linspace(0.0, 1.0, 21):
        use_small = p_small >= tau
        pts.append({"tau": float(tau),
                    "acc": float(np.where(use_small, ok_s, ok_l).mean()),
                    "cost": float(np.where(use_small, cost_s, cost_l).mean()),
                    "tokens": float(np.where(use_small, tok_s, tok_l).mean()),
                    "frac_small": float(use_small.mean())})
    refs = {
        "large-vanilla": {"acc": float(van_ok_l.mean()),
                          "cost": float((van_tok_l * args.large_params).mean())},
        "large-NTC": {"acc": float(ok_l.mean()), "cost": float(cost_l.mean())},
        "small-NTC": {"acc": float(ok_s.mean()), "cost": float(cost_s.mean())},
    }

    print(f"\njoint frontier: {bench}, evaluation n={len(eval_q)} (cost = tokens x params-B)")
    print(f"{'tau':>5}{'acc':>8}{'cost':>10}{'tokens':>9}{'%small':>8}")
    for p in pts[::2]:
        print(f"{p['tau']:>5.2f}{p['acc']:>8.3f}{p['cost']:>10.0f}"
              f"{p['tokens']:>9.0f}{100 * p['frac_small']:>7.0f}%")
    for k, v in refs.items():
        print(f"{k:<14} acc={v['acc']:.3f} cost={v['cost']:.0f}")

    md = [f"# Joint tier: model routing with calibrated halting ({bench})", "",
          f"Evaluation n = {len(eval_q)}, split seed {args.seed}. Small model {ds['model']} "
          f"(rule {fam_s} {kw_s}); large model {dl['model']} (rule {fam_l} {kw_l}); router: "
          f"TF-IDF and logistic regression trained on the {n_warm} calibration items. "
          "`tau` is the routing threshold on the predicted probability that the small "
          "model is correct; `cost` is mean tokens x model size in billions of parameters; "
          "`%small` is the share of queries routed to the small model.",
          "", "| tau | acc | cost | tokens | %small |", "|---|---|---|---|---|"]
    md += [f"| {p['tau']:.2f} | {p['acc']:.3f} | {p['cost']:.0f} "
           f"| {p['tokens']:.0f} | {100 * p['frac_small']:.0f}% |" for p in pts]
    md += ["", "| reference | acc | cost |", "|---|---|---|"]
    md += [f"| {k} | {v['acc']:.3f} | {v['cost']:.0f} |" for k, v in refs.items()]
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
