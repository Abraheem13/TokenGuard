#!/usr/bin/env python
"""Halting rules, replay evaluation and risk-controlled calibration.

This module is the shared library of the analysis scripts. It defines every
halting rule as a causal function of one probe stream, the overhead-inclusive
replay that scores a rule on a set of traces, and the selection tier's
calibration: a Bonferroni-corrected lower confidence bound over the whole
candidate library, with a null action that never halts.

Run as a script, it reports for one probe file the calibrated results over
repeated calibration/evaluation splits, exact McNemar tests on the first split
and a paired bootstrap interval for the token saving:

    python scripts/ntc_w1_stats.py --probes experiments/ntc/w1_math500_Qwen3-4B.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenguard.reasoning.datasets import is_correct


# ------------------------------------------------------------ halting rules
def deer_policy(probes, lam=0.95):
    """Confidence threshold: halt at the first probe whose confidence >= lam."""
    for k, p in enumerate(probes):
        if p["confidence"] >= lam:
            return k
    return None


def eat_policy(probes, delta=1e-3, alpha=0.2, warmup=3):
    """Entropy stability: halt once the EMA variance of first-token entropy < delta."""
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
    """Smoothed confidence: halt after `patience` probes with EMA confidence >= theta."""
    S, above = None, 0
    for k, p in enumerate(probes):
        c = p["confidence"]
        S = c if S is None else eta * S + (1 - eta) * c
        above = above + 1 if S >= theta else 0
        if above >= patience:
            return k
    return None


def agree_policy(probes, m=2, bm="math500"):
    """Answer agreement: halt when m consecutive trial answers are equivalent."""
    run = 1
    for k in range(1, len(probes)):
        same = (probes[k]["answer"] and
                is_correct(probes[k]["answer"], probes[k - 1]["answer"], bm))
        run = run + 1 if same else 1
        if run >= m:
            return k
    return None


def ntc_v2_policy(probes, m=2, theta=0.5, eta=0.6, bm="math500"):
    """Fusion tier: halt when m answers agree and EMA confidence >= theta."""
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


def enrich_probes_with_nll(traces):
    """Attach to each probe the mean token NLL of the thinking segment it closes.

    Segment boundaries are the probes' `ckpt_tokens` offsets into the per-token
    NLL of the thinking pass. Returns True if any trace carries token NLL.
    """
    any_nll = False
    for t in traces:
        nll = t.get("token_nll") or []
        if not nll:
            continue
        any_nll = True
        prev = 0
        for p in t["probes"]:
            end = max(prev + 1, min(len(nll), int(p["ckpt_tokens"])))
            seg = nll[prev:end]
            p["seg_nll"] = float(np.mean(seg)) if seg else 0.0
            prev = end
    return any_nll


def mur_policy(probes, gamma=0.9, beta=0.9):
    """Uncertainty momentum (MUR, arXiv:2507.14958) as a halting rule.

    Tracks momentum M_k = beta * M + (1 - beta) * m_k of the segment NLL m_k and
    halts at the first checkpoint whose momentary uncertainty m_k <= gamma * M.
    Requires `seg_nll` (see enrich_probes_with_nll).
    """
    M = None
    for k, p in enumerate(probes):
        m = p.get("seg_nll")
        if m is None:
            return None
        if M is None:
            M = m
            continue
        if m <= gamma * M:
            return k
        M = beta * M + (1 - beta) * m
    return None


def refrain_swucb_stream(traces, bench, arms=(0.85, 0.90, 0.95, 0.99),
                         window=50, ucb_c=0.5, mu=0.2):
    """Bandit threshold (REFRAIN, arXiv:2510.10103) over a query stream.

    A sliding-window UCB bandit chooses the confidence threshold for each query
    in stream order; the reward is 1[correct] - mu * tokens / mean full tokens.
    Returns per-item (correct, tokens) arrays.
    """
    van_tok = float(np.mean([t["n_total_tokens"] for t in traces])) or 1.0
    hist = deque(maxlen=window)
    ok_arr, tok_arr = [], []
    for step, t in enumerate(traces, start=1):
        counts = [1e-9] * len(arms)
        sums = [0.0] * len(arms)
        for a, r in hist:
            counts[a] += 1
            sums[a] += r
        ucb = [sums[a] / counts[a] + ucb_c * math.sqrt(
                   math.log(max(2, min(step, window))) / counts[a])
               for a in range(len(arms))]
        a = int(np.argmax(ucb))
        k = deer_policy(t["probes"], lam=arms[a]) if t["probes"] else None
        if k is None:
            ok = bool(t["natural_correct"])
            tok = t["n_total_tokens"]
        else:
            p = t["probes"][k]
            ok = is_correct(p["answer"], t["gold"], bench)
            tok = p["ckpt_tokens"] + p["n_probe_tokens"]
        hist.append((a, float(ok) - mu * tok / van_tok))
        ok_arr.append(ok)
        tok_arr.append(tok)
    return np.array(ok_arr), np.array(tok_arr, dtype=float)


def never_halt_policy(probes, **kw):
    """Null action: never halt early."""
    return None


# The candidate library of the selection tier (|C| = 19; 22 with MUR-mom,
# which is added per file when the probe stream carries token NLL).
FAMILIES = {
    "NEVER-HALT": (never_halt_policy, [{}]),
    "DEER": (deer_policy, [{"lam": v} for v in (0.90, 0.95, 0.99)]),
    "EAT": (eat_policy, [{"delta": v} for v in (1e-2, 1e-3, 1e-4)]),
    "NTC-conf": (ntc_conf_policy, [{"theta": v} for v in (0.85, 0.90, 0.95, 0.99)]),
    "AGREE": (agree_policy, [{"m": v} for v in (2, 3)]),
    "NTC-v2": (ntc_v2_policy,
               [{"m": _m, "theta": _t} for _m in (2, 3) for _t in (0.7, 0.9, 0.95)]),
}


# -------------------------------------------------------------- evaluation
def per_item(traces, bench, fn, kw):
    """Per-item (correct, tokens) of a rule; tokens include every probe paid."""
    ok, tok = [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
            if probes else None
        if kk is None:
            ok.append(bool(t["natural_correct"]))
            tok.append(t["n_total_tokens"]
                       + sum(q.get("n_probe_tokens", 0) for q in probes))
        else:
            p = probes[kk]
            ok.append(is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"]
                       + sum(q.get("n_probe_tokens", 0) for q in probes[:kk + 1]))
    return np.array(ok), np.array(tok, dtype=float)


def _norm_quantile(q):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if q <= 0.0:
        return -8.0
    if q >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        x = math.sqrt(-2 * math.log(q))
        return (((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
               ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
    if q > ph:
        x = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
                ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
    x = q - 0.5
    r = x * x
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*x / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _t_quantile(q, df):
    """Student-t quantile by the Cornish-Fisher expansion of the normal quantile."""
    z = _norm_quantile(q)
    if df is None or df <= 2:
        return z * 2.0
    g1 = (z ** 3 + z) / (4.0 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df * df)
    return z + g1 + g2


def calibrate(warm, bench, k_folds=5, eps=0.025, reps=3):
    """Choose a rule on calibration items under an accuracy tolerance eps.

    For every candidate (family, parameter) the accuracy change against full
    generation is estimated by repeated paired K-fold cross-validation (3 x 5
    folds), and its standard error from the per-item paired differences. A
    candidate is admissible when its one-sided lower confidence bound,
    Bonferroni-corrected over the whole library, is >= -eps; the cheapest
    admissible candidate is chosen, or the candidate with the highest bound
    if none is admissible. Set TG_SELECT=point to rank by the point estimate
    instead of the bound, and TG_DELTA to change the risk level (default 0.1).

    Returns the per-family choices and the library-wide choice (family, kw).
    """
    n = len(warm)
    k_folds = max(2, min(k_folds, n))
    van_all = np.array([t["natural_correct"] for t in warm], dtype=float)

    picks, gcands = {}, []
    for fam, (fn, grid) in FAMILIES.items():
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            ok = ok.astype(float)
            ds = []
            for r in range(reps):
                rng = np.random.default_rng(1000 + r)
                idx = rng.permutation(n)
                folds = [f for f in (idx[i::k_folds] for i in range(k_folds))
                         if len(f)]
                ds += [float(ok[f].mean() - van_all[f].mean()) for f in folds]
            dlt = ok - van_all
            se = float(np.std(dlt, ddof=1) / np.sqrt(max(1, len(dlt)))) \
                if len(dlt) > 1 else 1.0
            gcands.append({"fam": fam, "kw": kw, "md": float(np.mean(ds)), "se": se,
                           "tok": float(tok.mean())})
    mode = os.environ.get("TG_SELECT", "lcb").lower()
    delta = float(os.environ.get("TG_DELTA", "0.1"))
    z = _t_quantile(1.0 - delta / max(1, len(gcands)), max(2, n - 1))
    for c in gcands:
        c["lcb"] = c["md"] - z * c["se"] if mode == "lcb" else c["md"]
    for fam in FAMILIES:
        cands = [c for c in gcands if c["fam"] == fam]
        feas = [c for c in cands if c["lcb"] >= -eps]
        picks[fam] = (min(feas, key=lambda c: c["tok"])["kw"] if feas
                      else max(cands, key=lambda c: c["lcb"])["kw"])
    gfeas = [c for c in gcands if c["lcb"] >= -eps]
    g = (min(gfeas, key=lambda c: c["tok"]) if gfeas
         else max(gcands, key=lambda c: c["lcb"]))
    return picks, (g["fam"], g["kw"])


def mcnemar_exact(a_ok, b_ok):
    """Exact two-sided McNemar test on paired correctness arrays."""
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** (n - 1))
    return b, c, min(1.0, p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cv-eps", type=float, nargs="+", default=[0.01, 0.025],
                    help="accuracy tolerances of the selection tier")
    args = ap.parse_args()

    d = json.loads(Path(args.probes).read_text())
    traces, bench = d["traces"], d["benchmark"]
    n = len(traces)
    n_warm = int(n * args.warmup_frac)

    if enrich_probes_with_nll(traces):
        FAMILIES["MUR-mom"] = (mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])

    full_names = [f"NTC-full(e={e})" for e in args.cv_eps]
    agg = {fam: {"acc": [], "cut": []}
           for fam in list(FAMILIES) + full_names + ["vanilla", "REFRAIN-SWUCB"]}
    ntc_full_picks = {nm: [] for nm in full_names}
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        warm = [traces[i] for i in idx[:n_warm]]
        ev = [traces[i] for i in idx[n_warm:]]
        van_ok = np.array([t["natural_correct"] for t in ev])
        van_tok = np.array([t["n_total_tokens"] for t in ev], dtype=float)
        agg["vanilla"]["acc"].append(van_ok.mean())
        agg["vanilla"]["cut"].append(0.0)
        picks, _ = calibrate(warm, bench, eps=args.cv_eps[0])
        for fam, kw in picks.items():
            ok, tok = per_item(ev, bench, FAMILIES[fam][0], kw)
            agg[fam]["acc"].append(ok.mean())
            agg[fam]["cut"].append(100 * (1 - tok.mean() / van_tok.mean()))
        rok, rtok = refrain_swucb_stream(ev, bench)
        agg["REFRAIN-SWUCB"]["acc"].append(rok.mean())
        agg["REFRAIN-SWUCB"]["cut"].append(100 * (1 - rtok.mean() / van_tok.mean()))
        for e, nm in zip(args.cv_eps, full_names):
            _, (gfam, gkw) = calibrate(warm, bench, eps=e)
            ntc_full_picks[nm].append(f"{gfam}{gkw}")
            gok, gtok = per_item(ev, bench, FAMILIES[gfam][0], gkw)
            agg[nm]["acc"].append(gok.mean())
            agg[nm]["cut"].append(100 * (1 - gtok.mean() / van_tok.mean()))

    print(f"=== calibrated results: {d['model']} / {bench} "
          f"({args.n_seeds} splits, evaluation n={n - n_warm}) ===")
    print(f"{'method':<16}{'acc mean±sd':>14}{'cut% mean±sd':>16}")
    order = ["vanilla", "DEER", "EAT", "NTC-conf", "AGREE", "NTC-v2"]
    if "MUR-mom" in FAMILIES:
        order.append("MUR-mom")
    order.append("REFRAIN-SWUCB")
    for k in order + full_names:
        a, c = np.array(agg[k]["acc"]), np.array(agg[k]["cut"])
        print(f"{k:<16}{a.mean():>8.3f} ±{a.std():>5.3f}{c.mean():>11.1f} ±{c.std():>5.1f}")
    for nm in full_names:
        print(f"{nm} choices: {dict(Counter(ntc_full_picks[nm]))}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    warm = [traces[i] for i in idx[:n_warm]]
    ev = [traces[i] for i in idx[n_warm:]]
    picks, (gfam, gkw) = calibrate(warm, bench, eps=args.cv_eps[-1])
    gok, gtok = per_item(ev, bench, FAMILIES[gfam][0], gkw)
    van_ok = np.array([t["natural_correct"] for t in ev])
    van_tok = np.array([t["n_total_tokens"] for t in ev], dtype=float)

    print(f"\n=== seed-0 significance (eval n={len(ev)}; NTC-full={gfam}{gkw}) ===")
    for base in ["DEER", "EAT", "NTC-conf"]:
        bok, _ = per_item(ev, bench, FAMILIES[base][0], picks[base])
        b, c, pv = mcnemar_exact(gok, bok)
        print(f"McNemar NTC-full vs {base:<9}: NTC+only={b:>3} {base}+only={c:>3} "
              f"p={pv:.4f}{' *' if pv < 0.05 else ''}")
    b, c, pv = mcnemar_exact(gok, van_ok)
    print(f"McNemar NTC-full vs vanilla   : NTC+only={b:>3} van+only={c:>3} p={pv:.4f}")

    diffs = van_tok - gtok
    boots = np.array([diffs[np.random.default_rng(s).integers(0, len(diffs), len(diffs))].mean()
                      for s in range(args.n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nBootstrap token saving/item (NTC-full vs vanilla): "
          f"{diffs.mean():.0f} tokens [95% CI {lo:.0f}, {hi:.0f}] "
          f"= {100*diffs.mean()/van_tok.mean():.0f}% cut")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
