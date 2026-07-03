#!/usr/bin/env python
"""Evaluate halting policies (DEER / EAT / NTC-momentum / oracle) from the saved
probe file — NO GPU needed. Honest halt-then-emit accounting throughout.

Per problem, a policy walks the probe sequence (each probe = a checkpoint with a
FORCED answer + confidence + entropy) and picks a halt probe, or runs the full
chain. Scoring uses the probe's OWN emitted answer (honest). Token accounting:
  tokens        = thinking tokens at halt + emitted-answer tokens
  tokens+ovh    = ... + probe overhead actually paid online (all probes tried)

Policies:
  vanilla   full chain, natural answer
  DEER      halt at first probe with confidence > lambda           (sweep)
  EAT       halt when EMA-variance of probe first-entropy < delta  (sweep)
  NTC-fast  momentum-smoothed confidence S=eta*S+(1-eta)*C, halt when
            S > theta for `patience` consecutive probes            (sweep)
  oracle    earliest CORRECT probe (upper bound on savings)

Usage:
    python scripts/ntc_w1_policies.py --probes experiments/ntc/w1_math500_Qwen3-4B.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tokenguard.reasoning.datasets import is_correct


# ---------------------------------------------------------------- policies --
def deer_policy(probes, lam=0.95):
    for k, p in enumerate(probes):
        if p["confidence"] >= lam:
            return k
    return None


def eat_policy(probes, delta=1e-3, alpha=0.2, warmup=3):
    ema, emv = None, None
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


def ntc_policy(probes, theta=0.9, eta=0.6, patience=2):
    S, above = None, 0
    for k, p in enumerate(probes):
        c = p["confidence"]
        S = c if S is None else eta * S + (1 - eta) * c
        above = above + 1 if S >= theta else 0
        if above >= patience:
            return k
    return None


def agree_policy(probes, m=2, bm="math500"):
    """Halt when the last m probe ANSWERS agree (answer-stability signal)."""
    if m < 2:
        m = 2
    run = 1
    for k in range(1, len(probes)):
        same = (probes[k]["answer"] and
                is_correct(probes[k]["answer"], probes[k - 1]["answer"], bm))
        run = run + 1 if same else 1
        if run >= m:
            return k
    return None


def ntc_v2_policy(probes, m=2, theta=0.5, eta=0.6, bm="math500"):
    """NTC fusion: answer-stability (fast) gated by momentum confidence (slow).

    Halt at probe k when the last m answers agree AND the EMA-smoothed
    confidence S_k >= theta. Agreement gives precision; the (soft) confidence
    momentum blocks premature stops on unstable low-confidence streaks."""
    if m < 2:
        m = 2
    S = None
    run = 1
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


def oracle_policy(probes, gold, bench):
    for k, p in enumerate(probes):
        if is_correct(p["answer"], gold, bench):
            return k
    return None


# ---------------------------------------------------------------- evaluate --
def evaluate(traces, bench, pick, **kw):
    accs, toks, toks_ovh, halted, rescued = [], [], [], 0, 0
    for t in traces:
        probes = t["probes"]
        k = pick(probes, **kw) if probes else None
        if k is None:  # never halted -> full chain, natural answer
            ok = t["natural_correct"]
            tok = t["n_total_tokens"]
            ovh = tok + sum(p["n_probe_tokens"] for p in probes)
        else:
            p = probes[k]
            ok = is_correct(p["answer"], t["gold"], bench)
            tok = p["ckpt_tokens"] + p["n_probe_tokens"]
            ovh = p["ckpt_tokens"] + sum(q["n_probe_tokens"] for q in probes[:k + 1])
            halted += 1
            if ok and not t["natural_correct"]:
                rescued += 1
        accs.append(ok); toks.append(tok); toks_ovh.append(ovh)
    return {"acc": float(np.mean(accs)), "tokens": float(np.mean(toks)),
            "tokens_ovh": float(np.mean(toks_ovh)),
            "halt_rate": halted / max(1, len(traces)),
            "rescued": rescued}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help=">0 enables calibrated mode: choose each policy's param "
                         "on the warm-up split, report held-out numbers only")
    ap.add_argument("--mu", type=float, default=0.0002,
                    help="token penalty in calibration reward = acc - mu*tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.loads(Path(args.probes).read_text())
    traces, bench = d["traces"], d["benchmark"]
    n = len(traces)

    warm = []
    if args.warmup_frac > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(n)
        n_warm = int(n * args.warmup_frac)
        warm = [traces[i] for i in idx[:n_warm]]
        traces = [traces[i] for i in idx[n_warm:]]
        print(f"[calibrated mode] warm-up n={len(warm)}, eval n={len(traces)}")

    vanilla_acc = float(np.mean([t["natural_correct"] for t in traces]))
    vanilla_tok = float(np.mean([t["n_total_tokens"] for t in traces]))
    print(f"=== policies on {d['model']} / {bench} (n={n}) ===")
    print(f"vanilla: acc={vanilla_acc:.3f} tokens={vanilla_tok:.0f}\n")

    # oracle (upper bound) — needs gold per trace, computed directly
    accs, toks = [], []
    for t in traces:
        k = oracle_policy(t["probes"], t["gold"], bench) if t["probes"] else None
        if k is None:
            accs.append(t["natural_correct"]); toks.append(t["n_total_tokens"])
        else:
            p = t["probes"][k]
            accs.append(True); toks.append(p["ckpt_tokens"] + p["n_probe_tokens"])
    print(f"ORACLE (earliest-correct-probe): acc={np.mean(accs):.3f} "
          f"tokens={np.mean(toks):.0f} (cut {100*(1-np.mean(toks)/vanilla_tok):.0f}%) "
          f"← ceiling\n")

    header = f"{'policy':<22}{'param':>8}{'acc':>8}{'tokens':>9}{'cut%':>7}{'ovh_tok':>9}{'halt%':>7}{'resc':>6}"
    print(header)
    rows = []
    for lam in [0.90, 0.95, 0.99]:
        r = evaluate(traces, bench, deer_policy, lam=lam)
        rows.append(("DEER", lam, r))
    for delta in [1e-2, 1e-3, 1e-4]:
        r = evaluate(traces, bench, eat_policy, delta=delta)
        rows.append(("EAT", delta, r))
    for theta in [0.85, 0.90, 0.95, 0.99]:
        r = evaluate(traces, bench, ntc_policy, theta=theta)
        rows.append(("NTC-conf (momentum)", theta, r))
    for m in [2, 3]:
        r = evaluate(traces, bench, agree_policy, m=m, bm=bench)
        rows.append((f"AGREE", m, r))
    for m, theta in [(2, 0.3), (2, 0.5), (2, 0.7), (3, 0.3), (3, 0.5)]:
        r = evaluate(traces, bench, ntc_v2_policy, m=m, theta=theta, bm=bench)
        rows.append((f"NTC-v2 (agree+conf)", f"{m}/{theta}", r))

    for name, param, r in rows:
        cut = 100 * (1 - r["tokens"] / vanilla_tok)
        print(f"{name:<22}{str(param):>8}{r['acc']:>8.3f}{r['tokens']:>9.0f}"
              f"{cut:>6.0f}%{r['tokens_ovh']:>9.0f}{100*r['halt_rate']:>6.0f}%"
              f"{r.get('rescued', 0):>6}")

    # calibrated mode: choose each family's param on WARM-UP by reward, then
    # print a held-out table with ONLY the calibrated pick per family
    if warm:
        fams = {}
        for name, param, _ in rows:
            fams.setdefault(name, []).append(param)
        picks = {}
        policy_fn = {"DEER": (deer_policy, lambda p: {"lam": p}),
                     "EAT": (eat_policy, lambda p: {"delta": p}),
                     "NTC-conf (momentum)": (ntc_policy, lambda p: {"theta": p}),
                     "AGREE": (agree_policy, lambda p: {"m": p, "bm": bench}),
                     "NTC-v2 (agree+conf)": (ntc_v2_policy,
                         lambda p: {"m": int(str(p).split("/")[0]),
                                    "theta": float(str(p).split("/")[1]),
                                    "bm": bench})}
        print("\n=== CALIBRATED (param chosen on warm-up, reported on held-out) ===")
        print(f"{'policy':<22}{'param':>8}{'acc':>8}{'tokens':>9}{'cut%':>7}{'resc':>6}")
        # warm-up vanilla accuracy = the constraint anchor
        warm_van_acc = float(np.mean([t["natural_correct"] for t in warm]))
        for fam, params in fams.items():
            fn, kwf = policy_fn[fam]
            # constrained pick: max token-cut s.t. warm acc >= vanilla_warm - eps;
            # if nothing satisfies the constraint, fall back to highest warm acc.
            # one-standard-error rule (statistically principled tolerance)
            import math as _m
            eps = max(0.01, _m.sqrt(warm_van_acc * (1 - warm_van_acc) / max(1, len(warm))))
            feasible, all_pts = [], []
            for p_ in params:
                rw = evaluate(warm, bench, fn, **kwf(p_))
                all_pts.append((p_, rw))
                if rw["acc"] >= warm_van_acc - eps:
                    feasible.append((p_, rw))
            if feasible:
                best_p = min(feasible, key=lambda x: x[1]["tokens"])[0]
            else:
                best_p = max(all_pts, key=lambda x: x[1]["acc"])[0]
            re_ = evaluate(traces, bench, fn, **kwf(best_p))
            picks[fam] = (best_p, re_)
            cut = 100 * (1 - re_["tokens"] / vanilla_tok)
            print(f"{fam:<22}{str(best_p):>8}{re_['acc']:>8.3f}{re_['tokens']:>9.0f}"
                  f"{cut:>6.0f}%{re_.get('rescued', 0):>6}")

        # ---- NTC-full (adaptive): slow tier picks (signal family, param) jointly
        # on warm-up under the accuracy constraint; held-out reported.
        gfeas, gall = [], []
        for fam, params in fams.items():
            fn, kwf = policy_fn[fam]
            for p_ in params:
                rw = evaluate(warm, bench, fn, **kwf(p_))
                gall.append((fam, p_, rw))
                if rw["acc"] >= warm_van_acc - eps:
                    gfeas.append((fam, p_, rw))
        if gfeas:
            gfam, gp, _ = min(gfeas, key=lambda x: x[2]["tokens"])
        else:
            gfam, gp, _ = max(gall, key=lambda x: x[2]["acc"])
        gfn, gkwf = policy_fn[gfam]
        gre = evaluate(traces, bench, gfn, **gkwf(gp))
        gcut = 100 * (1 - gre["tokens"] / vanilla_tok)
        print("-" * 62)
        print(f"{'NTC-full (adaptive)':<22}{gfam[:5]+'/'+str(gp):>8}{gre['acc']:>8.3f}"
              f"{gre['tokens']:>9.0f}{gcut:>6.0f}%{gre.get('rescued', 0):>6}")
        print(f"  slow tier selected: signal={gfam}, param={gp} (on warm-up)")

    # headline: best policy point with acc >= vanilla - 0.01
    ok_pts = [(n_, p_, r_) for n_, p_, r_ in rows if r_["acc"] >= vanilla_acc - 0.01]
    if ok_pts:
        best = min(ok_pts, key=lambda x: x[2]["tokens"])
        cut = 100 * (1 - best[2]["tokens"] / vanilla_tok)
        print(f"\nHEADLINE: {best[0]} (param={best[1]}) keeps accuracy "
              f"({best[2]['acc']:.3f} vs {vanilla_acc:.3f}) at {cut:.0f}% fewer tokens")
    else:
        best_acc = max(rows, key=lambda x: x[2]["acc"])
        print(f"\nNo policy holds accuracy within 1pt yet. Best acc: "
              f"{best_acc[0]} param={best_acc[1]} acc={best_acc[2]['acc']:.3f}")

    out = args.out or args.probes.replace(".json", "_policies.json")
    Path(out).write_text(json.dumps(
        {"vanilla": {"acc": vanilla_acc, "tokens": vanilla_tok},
         "rows": [{"policy": n_, "param": p_, **r_} for n_, p_, r_ in rows]},
        indent=1))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
