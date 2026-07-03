#!/usr/bin/env python
"""Publication figures + final results tables from saved probe files. No GPU.

For each --probes file:
  * Pareto figure (PNG+PDF): accuracy vs mean tokens for every (policy, param)
    sweep point, vanilla line, oracle point, and the calibrated NTC-full star
    (seed-0 held-out).
  * Appends a row block to experiments/ntc/RESULTS.md with the 10-seed
    calibrated mean±std table (paper-ready).

Usage:
    python scripts/ntc_w1_figures.py \
        --probes experiments/ntc/w1_math500_Qwen3-4B.json \
        --probes experiments/ntc/w1_gsm8k_Qwen3-4B.json \
        --probes experiments/ntc/w1_gpqa16k_Qwen3-4B.json \
        --probes experiments/ntc/w1_math500_Qwen3-1.7B.json
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

from tokenguard.reasoning.datasets import is_correct

# ---- policies (identical to stats script) ----
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
STYLE = {"DEER": ("tab:red", "s"), "EAT": ("tab:orange", "^"),
         "NTC-conf": ("tab:purple", "v"), "AGREE": ("tab:green", "o"),
         "NTC-v2": ("tab:blue", "D")}


def per_item(traces, bench, fn, kw, with_ovh=False):
    ok, tok, ovh = [], [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
             if probes else None
        if kk is None:
            ok.append(bool(t["natural_correct"])); tok.append(t["n_total_tokens"])
            ovh.append(t["n_total_tokens"] + sum(q["n_probe_tokens"] for q in probes))
        else:
            p = probes[kk]
            ok.append(is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"] + p["n_probe_tokens"])
            ovh.append(p["ckpt_tokens"] + sum(q["n_probe_tokens"] for q in probes[:kk + 1]))
    if with_ovh:
        return np.array(ok), np.array(tok, dtype=float), np.array(ovh, dtype=float)
    return np.array(ok), np.array(tok, dtype=float)


def calibrate(warm, bench, k_folds=5, eps=0.03):
    """Paired K-fold CV slow-tier selection (see ntc_w1_stats.calibrate).
    Returns (per_family_picks, (global_family, global_kw))."""
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
            se = float(d.std(ddof=1) / math.sqrt(len(folds)))
            cands.append({"kw": kw, "md": md, "se": se,
                          "tok": float(tok.mean()), "lcb": md - se})
            gcands.append({"fam": fam, **cands[-1]})
        feas = [c for c in cands if c["lcb"] >= -eps]
        picks[fam] = (min(feas, key=lambda c: c["tok"])["kw"] if feas
                      else max(cands, key=lambda c: c["md"])["kw"])
    gfeas = [c for c in gcands if c["lcb"] >= -eps]
    g = (min(gfeas, key=lambda c: c["tok"]) if gfeas
         else max(gcands, key=lambda c: c["md"]))
    return picks, (g["fam"], g["kw"])


def oracle(traces, bench):
    accs, toks = [], []
    for t in traces:
        k = None
        for i, p in enumerate(t["probes"]):
            if is_correct(p["answer"], t["gold"], bench):
                k = i; break
        if k is None:
            accs.append(t["natural_correct"]); toks.append(t["n_total_tokens"])
        else:
            p = t["probes"][k]
            accs.append(True); toks.append(p["ckpt_tokens"] + p["n_probe_tokens"])
    return float(np.mean(accs)), float(np.mean(toks))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", action="append", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--figdir", default="experiments/ntc/figures")
    ap.add_argument("--results-md", default="experiments/ntc/RESULTS.md")
    args = ap.parse_args()

    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    md = ["# NTC — Final Results (calibrated, held-out, multi-seed)\n"]

    for pf in args.probes:
        d = json.loads(Path(pf).read_text())
        traces, bench, model = d["traces"], d["benchmark"], d["model"]
        n = len(traces); n_warm = int(n * args.warmup_frac)
        tag = f"{bench}_{model.split('/')[-1]}"

        # ---- figure: full-data sweep Pareto + calibrated star ----
        van_acc = float(np.mean([t["natural_correct"] for t in traces]))
        van_tok = float(np.mean([t["n_total_tokens"] for t in traces]))
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for fam, (fn, grid) in FAMILIES.items():
            xs, ys = [], []
            for kw in grid:
                ok, tok = per_item(traces, bench, fn, kw)
                xs.append(tok.mean()); ys.append(ok.mean())
            c, mk = STYLE[fam]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker=mk, ms=5,
                    lw=1.0, color=c, alpha=0.85, label=fam)
        oa, ot = oracle(traces, bench)
        ax.scatter([ot], [oa], marker="*", s=140, color="black", zorder=5,
                   label="Oracle")
        # calibrated NTC-full on seed-0 held-out
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        warm = [traces[i] for i in idx[:n_warm]]
        ev = [traces[i] for i in idx[n_warm:]]
        _, (gfam, gkw) = calibrate(warm, bench, eps=0.05)
        gok, gtok = per_item(ev, bench, FAMILIES[gfam][0], gkw)
        ax.scatter([gtok.mean()], [gok.mean()], marker="*", s=220,
                   color="crimson", edgecolor="k", zorder=6,
                   label="NTC-full (calibrated)")
        ax.axhline(van_acc, color="gray", ls="--", lw=1, label="vanilla acc")
        ax.axvline(van_tok, color="gray", ls=":", lw=1)
        ax.set_xlabel("Mean tokens per problem")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model.split('/')[-1]} · {bench} (n={n})", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{args.figdir}/pareto_{tag}.{ext}", dpi=200)
        plt.close(fig)
        print(f"figure: {args.figdir}/pareto_{tag}.png")

        # ---- multi-seed table for RESULTS.md ----
        agg = {k: {"acc": [], "cut": []} for k in list(FAMILIES) + ["vanilla"]}
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            warm = [traces[i] for i in idx[:n_warm]]
            ev = [traces[i] for i in idx[n_warm:]]
            vt = np.array([t["n_total_tokens"] for t in ev], dtype=float).mean()
            agg["vanilla"]["acc"].append(np.mean([t["natural_correct"] for t in ev]))
            agg["vanilla"]["cut"].append(0.0)
            picks, _ = calibrate(warm, bench, eps=0.01)
            for fam, kwp in picks.items():
                fn = FAMILIES[fam][0]
                ok, tok = per_item(ev, bench, fn, kwp)
                agg[fam]["acc"].append(ok.mean())
                agg[fam]["cut"].append(100 * (1 - tok.mean() / vt))
            for e, nm in [(0.01, "NTC-full(e=0.01)"), (0.05, "NTC-full(e=0.05)")]:
                _, (gfam, gkw) = calibrate(warm, bench, eps=e)
                ok, tok, ovh = per_item(ev, bench, FAMILIES[gfam][0], gkw, with_ovh=True)
                a = agg.setdefault(nm, {"acc": [], "cut": []})
                a["acc"].append(ok.mean())
                a["cut"].append(100 * (1 - tok.mean() / vt))
                a.setdefault("cut_ovh", []).append(100 * (1 - ovh.mean() / vt))

        md.append(f"\n## {model} · {bench} (n={n}, {args.n_seeds} seeds, "
                  f"eval n={n - n_warm})\n")
        md.append("| method | accuracy (mean±std) | token cut % (mean±std) |")
        md.append("|---|---|---|")
        for k in ["vanilla", "DEER", "EAT", "NTC-conf", "AGREE", "NTC-v2",
                  "NTC-full(e=0.01)", "NTC-full(e=0.05)"]:
            a = np.array(agg[k]["acc"]); c = np.array(agg[k]["cut"])
            bold = "**" if k.startswith("NTC-full") else ""
            md.append(f"| {bold}{k}{bold} | {a.mean():.3f} ± {a.std():.3f} "
                      f"| {c.mean():.1f} ± {c.std():.1f} |")
        co = np.array(agg["NTC-full(e=0.05)"].get("cut_ovh", [0.0]))
        md.append(f"| NTC-full(e=0.05) incl. probe overhead | — "
                  f"| {co.mean():.1f} ± {co.std():.1f} |")

    md.append("\n---\n## Limitations (stated for the paper)\n")
    md.append("* Generation uses a single sampling seed (temp 0.6, seed 42); "
              "the reported ± std is over 10 calibration splits, not "
              "generation seeds.")
    md.append("* Token counts follow the total-generated convention "
              "(thinking + emitted answer); the overhead-inclusive row adds "
              "all trial-answer probe tokens actually paid online.")
    md.append("* GPQA-Diamond n=198; AIME excluded (n=30/set requires avg@16 "
              "for meaningful comparison, out of compute scope).")
    Path(args.results_md).write_text("\n".join(md) + "\n")
    print(f"\nresults table: {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
