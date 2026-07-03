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


def calibrate(warm, bench):
    van = float(np.mean([t["natural_correct"] for t in warm]))
    eps = max(0.01, math.sqrt(van * (1 - van) / max(1, len(warm))))
    gfeas, gall = [], []
    for fam, (fn, grid) in FAMILIES.items():
        for kw in grid:
            ok, tok = per_item(warm, bench, fn, kw)
            r = (float(ok.mean()), float(tok.mean()))
            gall.append((fam, kw, r))
            if r[0] >= van - eps:
                gfeas.append((fam, kw, r))
    return (min(gfeas, key=lambda x: x[2][1])[:2] if gfeas
            else max(gall, key=lambda x: x[2][0])[:2])


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
        gfam, gkw = calibrate(warm, bench)
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
        agg = {k: {"acc": [], "cut": []} for k in list(FAMILIES) + ["NTC-full", "vanilla"]}
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            warm = [traces[i] for i in idx[:n_warm]]
            ev = [traces[i] for i in idx[n_warm:]]
            vt = np.array([t["n_total_tokens"] for t in ev], dtype=float).mean()
            agg["vanilla"]["acc"].append(np.mean([t["natural_correct"] for t in ev]))
            agg["vanilla"]["cut"].append(0.0)
            # per-family constrained pick (reuse global calibrate per family)
            van = float(np.mean([t["natural_correct"] for t in warm]))
            eps = max(0.01, math.sqrt(van * (1 - van) / max(1, len(warm))))
            for fam, (fn, grid) in FAMILIES.items():
                feas, allp = [], []
                for kw in grid:
                    ok, tok = per_item(warm, bench, fn, kw)
                    r = (float(ok.mean()), float(tok.mean()))
                    allp.append((kw, r))
                    if r[0] >= van - eps:
                        feas.append((kw, r))
                kwp = (min(feas, key=lambda x: x[1][1])[0] if feas
                       else max(allp, key=lambda x: x[1][0])[0])
                ok, tok = per_item(ev, bench, fn, kwp)
                agg[fam]["acc"].append(ok.mean())
                agg[fam]["cut"].append(100 * (1 - tok.mean() / vt))
            gfam, gkw = calibrate(warm, bench)
            ok, tok = per_item(ev, bench, FAMILIES[gfam][0], gkw)
            agg["NTC-full"]["acc"].append(ok.mean())
            agg["NTC-full"]["cut"].append(100 * (1 - tok.mean() / vt))

        md.append(f"\n## {model} · {bench} (n={n}, {args.n_seeds} seeds, "
                  f"eval n={n - n_warm})\n")
        md.append("| method | accuracy (mean±std) | token cut % (mean±std) |")
        md.append("|---|---|---|")
        for k in ["vanilla", "DEER", "EAT", "NTC-conf", "AGREE", "NTC-v2", "NTC-full"]:
            a = np.array(agg[k]["acc"]); c = np.array(agg[k]["cut"])
            bold = "**" if k == "NTC-full" else ""
            md.append(f"| {bold}{k}{bold} | {a.mean():.3f} ± {a.std():.3f} "
                      f"| {c.mean():.1f} ± {c.std():.1f} |")

    Path(args.results_md).write_text("\n".join(md) + "\n")
    print(f"\nresults table: {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
