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


def calibrate(warm, bench, eps=None):
    """Per-family constrained pick + NTC-full global pick on warm-up.

    eps defaults to the ONE-STANDARD-ERROR rule: the accuracy constraint is
    'within one binomial standard error of the warm-up vanilla estimate' —
    a statistically principled tolerance that stabilises the pick on small
    warm-up sets while still excluding genuinely-collapsing policies."""
    van = float(np.mean([t["natural_correct"] for t in warm]))
    if eps is None:
        eps = max(0.01, math.sqrt(van * (1 - van) / max(1, len(warm))))
    picks, gfeas, gall = {}, [], []
    for fam, (fn, grid) in FAMILIES.items():
        feas, allp = [], []
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            r = (float(ok.mean()), float(tok.mean()))
            allp.append((kw, r))
            if r[0] >= van - eps:
                feas.append((kw, r))
            gall.append((fam, kw, r))
            if r[0] >= van - eps:
                gfeas.append((fam, kw, r))
        picks[fam] = (min(feas, key=lambda x: x[1][1])[0] if feas
                      else max(allp, key=lambda x: x[1][0])[0])
    gpick = (min(gfeas, key=lambda x: x[2][1])[:2] if gfeas
             else max(gall, key=lambda x: x[2][0])[:2])
    return picks, gpick  # per-family kws, (family, kw) for NTC-full


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
    args = ap.parse_args()

    d = json.loads(Path(args.probes).read_text())
    traces, bench = d["traces"], d["benchmark"]
    n = len(traces)
    n_warm = int(n * args.warmup_frac)

    # ---------- 1) multi-seed splits ----------
    agg = {fam: {"acc": [], "cut": []} for fam in list(FAMILIES) + ["NTC-full", "vanilla"]}
    ntc_full_picks = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        warm = [traces[i] for i in idx[:n_warm]]
        ev = [traces[i] for i in idx[n_warm:]]
        van_ok = np.array([t["natural_correct"] for t in ev])
        van_tok = np.array([t["n_total_tokens"] for t in ev], dtype=float)
        agg["vanilla"]["acc"].append(van_ok.mean()); agg["vanilla"]["cut"].append(0.0)
        picks, (gfam, gkw) = calibrate(warm, bench)
        ntc_full_picks.append(f"{gfam}{gkw}")
        for fam, kw in picks.items():
            ok, tok = per_item(ev, bench, FAMILIES[fam][0], kw)
            agg[fam]["acc"].append(ok.mean())
            agg[fam]["cut"].append(100 * (1 - tok.mean() / van_tok.mean()))
        gok, gtok = per_item(ev, bench, FAMILIES[gfam][0], gkw)
        agg["NTC-full"]["acc"].append(gok.mean())
        agg["NTC-full"]["cut"].append(100 * (1 - gtok.mean() / van_tok.mean()))

    print(f"=== multi-seed calibrated results — {d['model']} / {bench} "
          f"({args.n_seeds} splits, eval n={n - n_warm}) ===")
    print(f"{'method':<12}{'acc mean±std':>16}{'cut% mean±std':>18}")
    for k in ["vanilla", "DEER", "EAT", "NTC-conf", "AGREE", "NTC-v2", "NTC-full"]:
        a, c = np.array(agg[k]["acc"]), np.array(agg[k]["cut"])
        print(f"{k:<12}{a.mean():>8.3f} ±{a.std():>5.3f}{c.mean():>11.1f} ±{c.std():>5.1f}")
    from collections import Counter
    print(f"NTC-full picks across seeds: {dict(Counter(ntc_full_picks))}")

    # ---------- 2) McNemar + 3) bootstrap on seed-0 split ----------
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    warm = [traces[i] for i in idx[:n_warm]]
    ev = [traces[i] for i in idx[n_warm:]]
    picks, (gfam, gkw) = calibrate(warm, bench)
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
