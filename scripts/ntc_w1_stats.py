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

FAMILIES = {
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
        if kk is None:
            ok.append(bool(t["natural_correct"])); tok.append(t["n_total_tokens"])
        else:
            p = probes[kk]
            ok.append(is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"] + p["n_probe_tokens"])
    return np.array(ok), np.array(tok, dtype=float)


def calibrate(warm, bench, k_folds=5, eps=0.03):
    """Slow-tier selection via PAIRED K-fold cross-validation on the warm-up.

    For each candidate (family, param) we compute, per fold f, the paired
    accuracy difference d_f = acc_candidate(f) - acc_vanilla(f). Pairing on
    the same fold cancels shared item-difficulty variance, giving a far
    tighter standard error than comparing point estimates. Feasibility:
        mean(d) - SE(d) >= -eps        (lower confidence bound rule)
    Among feasible candidates we take minimum mean tokens; if none is
    feasible, the maximum mean(d) candidate (accuracy-safest). The same rule
    yields per-family picks and the global NTC-full pick, so on the warm-up
    objective NTC-full is >= each fixed component by construction.
    """
    n = len(warm)
    k_folds = max(2, min(k_folds, n))
    idx = np.arange(n)
    folds = [f for f in (idx[i::k_folds] for i in range(k_folds)) if len(f)]
    van_all = np.array([t["natural_correct"] for t in warm], dtype=float)
    van_by_fold = [van_all[f].mean() for f in folds]

    picks, gcands = {}, []
    for fam, (fn, grid) in FAMILIES.items():
        cands = []
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            ok = ok.astype(float)
            d = np.array([ok[f].mean() - van_by_fold[j]
                          for j, f in enumerate(folds)])
            md = float(d.mean())
            se = float(d.std(ddof=1) / math.sqrt(k_folds))
            cands.append({"kw": kw, "md": md, "se": se,
                          "tok": float(tok.mean()), "lcb": md - se})
            gcands.append({"fam": fam, **cands[-1]})
        feas = [c for c in cands if c["lcb"] >= -eps]
        picks[fam] = (min(feas, key=lambda c: c["tok"])["kw"] if feas
                      else max(cands, key=lambda c: c["md"])["kw"])
    gfeas = [c for c in gcands if c["lcb"] >= -eps]
    if gfeas:
        g = min(gfeas, key=lambda c: c["tok"])
    else:
        g = max(gcands, key=lambda c: c["md"])
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
    ap.add_argument("--cv-eps", type=float, nargs="+", default=[0.01, 0.05],
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
