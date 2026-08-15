#!/usr/bin/env python
"""Statistical rigor for the NTC results — no GPU needed.

For a saved probe file:
  1. MULTI-SEED calibration splits (default 10): re-draw the warm-up/eval split,
     re-calibrate every method (incl. NTC-full adaptive) on warm-up, evaluate
     held-out. Report mean ± std of accuracy / tokens / cut% across seeds.
  2. McNEMAR exact test (seed-0 split): NTC-full vs each calibrated baseline on
     paired per-item correctness.
  3. PAIRED BOOTSTRAP (10k) on per-item token usage: NTC-full vs vanilla — 95%
     CI on the mean token saving.

Usage:
    python scripts/ntc_w1_stats.py --probes experiments/ntc/w1_math500_Qwen3-4B.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os as _os
import numpy as np

from tokenguard.reasoning.datasets import is_correct


# ---------------- policies (self-contained copies) ----------------
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




# ---------------- v2 additions: faithful-signal baselines ----------------
def enrich_probes_with_nll(traces):
    """Attach per-checkpoint segment mean-NLL (from the thinking pass) to each
    probe. ckpt_tokens indexes are computed via re-encoding the text prefix
    with the same tokenizer; alignment is exact up to +/- a few join tokens,
    negligible over ~256-token segments. Returns True if any trace has NLL."""
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
    """MUR's momentum-uncertainty signal (arXiv:2507.14958) adapted to the
    halting setting under identical answer-forcing: momentary step NLL m_k vs
    momentum M_k = beta*M + (1-beta)*m. MUR itself SCALES per-step compute
    when m_k exceeds gamma-scaled momentum; as a halting criterion we stop at
    the first checkpoint whose momentary uncertainty falls to <= gamma * M
    (reasoning has stabilised). Requires seg_nll (v2 traces)."""
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
    """REFRAIN's online mechanism (arXiv:2510.10103), faithful-in-spirit:
    a sliding-window UCB bandit adapts the confidence-halting threshold over
    the query STREAM (arms = lambda grid over DEER confidence); reward =
    1[correct] - mu * (tokens / stream-mean vanilla tokens). Evaluated in
    stream order on the held-out split. Returns (ok, tok) arrays."""
    import collections
    van_tok = float(np.mean([t["n_total_tokens"] for t in traces])) or 1.0
    hist = collections.deque(maxlen=window)  # (arm_idx, reward)
    ok_arr, tok_arr = [], []
    for step, t in enumerate(traces, start=1):
        # SW-UCB arm choice
        counts = [1e-9] * len(arms)
        sums = [0.0] * len(arms)
        for a, r in hist:
            counts[a] += 1
            sums[a] += r
        ucb = [sums[a] / counts[a] + ucb_c * math.sqrt(
                   math.log(max(2, min(step, window))) / counts[a])
               for a in range(len(arms))]
        a = int(np.argmax(ucb))
        lam = arms[a]
        k = deer_policy(t["probes"], lam=lam) if t["probes"] else None
        if k is None:
            ok = bool(t["natural_correct"]); tok = t["n_total_tokens"]
        else:
            p = t["probes"][k]
            ok = is_correct(p["answer"], t["gold"], bench)
            tok = p["ckpt_tokens"] + p["n_probe_tokens"]
        hist.append((a, float(ok) - mu * tok / van_tok))
        ok_arr.append(ok); tok_arr.append(tok)
    return np.array(ok_arr), np.array(tok_arr, dtype=float)

def never_halt_policy(probes, **kw):
    """Null action: never halt (spend the full thinking budget)."""
    return None


FAMILIES = {
    "NEVER-HALT": (never_halt_policy, [{}]),
    "DEER":     (deer_policy,     [{"lam": v} for v in (0.90, 0.95, 0.99)]),
    "EAT":      (eat_policy,      [{"delta": v} for v in (1e-2, 1e-3, 1e-4)]),
    "NTC-conf": (ntc_conf_policy, [{"theta": v} for v in (0.85, 0.90, 0.95, 0.99)]),
    "AGREE":    (agree_policy,    [{"m": v} for v in (2, 3)]),
    "NTC-v2":   (ntc_v2_policy,   [{"m": m, "theta": t}
                                   for m, t in ((2, .3), (2, .5), (2, .7), (3, .3), (3, .5))]),
}


# ---------------- evaluation ----------------
def per_item(traces, bench, fn, kw):
    """Return per-item (correct, tokens) arrays for a policy."""
    ok, tok = [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
             if probes else None
        # OVERHEAD_INCLUSIVE: all probes paid up to (and including) the halt
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
    """Inverse standard-normal CDF (Acklam rational approximation)."""
    import math
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
    """Student-t quantile via Cornish-Fisher expansion of the normal quantile."""
    z = _norm_quantile(q)
    if df is None or df <= 2:
        return z * 2.0
    g1 = (z ** 3 + z) / (4.0 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96.0 * df * df)
    return z + g1 + g2


def calibrate(warm, bench, k_folds=5, eps=0.025, reps=3):
    """Slow-tier selection via REPEATED paired K-fold CV (evidence-tested on
    the real MATH-500/GPQA probe data).

    For each candidate (family, param): pool the paired per-fold accuracy
    differences d = acc_candidate(fold) - acc_vanilla(fold) over `reps`
    shuffled fold assignments (3 x 5-fold = 15 paired diffs), cancelling
    shared item-difficulty variance and fold-assignment noise. Feasibility:
        mean(d) >= -eps      (eps = accuracy SLO, default 2.5 points)
    Among feasible candidates take minimum mean tokens; if none feasible,
    the maximum mean(d) candidate. The same rule yields per-family picks and
    the global NTC-full pick.
    """
    n = len(warm)
    k_folds = max(2, min(k_folds, n))
    van_all = np.array([t["natural_correct"] for t in warm], dtype=float)

    picks, gcands = {}, []
    for fam, (fn, grid) in FAMILIES.items():
        cands = []
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
            md = float(np.mean(ds))
            # paired per-item difference -> honest finite-sample SE
            _dlt = ok - van_all
            _se = float(np.std(_dlt, ddof=1) / np.sqrt(max(1, len(_dlt)))) \
                if len(_dlt) > 1 else 1.0
            cands.append({"kw": kw, "md": md, "se": _se,
                          "tok": float(tok.mean())})
            gcands.append({"fam": fam, **cands[-1]})
        picks[fam] = None  # filled after the global Bonferroni correction
    # LCB_SELECT: Bonferroni-corrected one-sided lower confidence bound over
    # the whole candidate library (see module docstring).
    _mode = _os.environ.get("TG_SELECT", "lcb").lower()
    _delta = float(_os.environ.get("TG_DELTA", "0.1"))
    _m = max(1, len(gcands))
    _z = _t_quantile(1.0 - _delta / _m, max(2, n - 1))
    for c in gcands:
        c["lcb"] = c["md"] - _z * c["se"] if _mode == "lcb" else c["md"]
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
    """Exact two-sided McNemar on paired correctness arrays."""
    b = int(np.sum(a_ok & ~b_ok))   # NTC right, baseline wrong
    c = int(np.sum(~a_ok & b_ok))   # NTC wrong, baseline right
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
                    help="accuracy-SLO tolerances for NTC-full (strict, relaxed)")
    args = ap.parse_args()

    d = json.loads(Path(args.probes).read_text())
    traces, bench = d["traces"], d["benchmark"]
    n = len(traces)
    n_warm = int(n * args.warmup_frac)

    if enrich_probes_with_nll(traces):
        FAMILIES["MUR-mom"] = (mur_policy,
                               [{"gamma": g} for g in (0.7, 0.8, 0.9)])
        print("[v2] token NLL found -> faithful MUR-momentum family enabled")
    else:
        print("[v2] no token NLL in this probe file -> MUR-mom skipped "
              "(regenerate traces with harness v2 to enable)")

    # ---------- 1) multi-seed splits ----------
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
        agg["vanilla"]["acc"].append(van_ok.mean()); agg["vanilla"]["cut"].append(0.0)
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

    print(f"=== multi-seed calibrated results — {d['model']} / {bench} "
          f"({args.n_seeds} splits, eval n={n - n_warm}) ===")
    print(f"{'method':<12}{'acc mean±std':>16}{'cut% mean±std':>18}")
    fam_order = ["vanilla", "DEER", "EAT", "NTC-conf", "AGREE", "NTC-v2"]
    if "MUR-mom" in FAMILIES:
        fam_order.append("MUR-mom")
    fam_order.append("REFRAIN-SWUCB")
    for k in fam_order + full_names:
        a, c = np.array(agg[k]["acc"]), np.array(agg[k]["cut"])
        print(f"{k:<16}{a.mean():>8.3f} ±{a.std():>5.3f}{c.mean():>11.1f} ±{c.std():>5.1f}")
    from collections import Counter
    for nm in full_names:
        print(f"{nm} picks: {dict(Counter(ntc_full_picks[nm]))}")

    # ---------- 2) McNemar + 3) bootstrap on seed-0 split ----------
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
