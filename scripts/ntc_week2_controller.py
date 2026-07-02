#!/usr/bin/env python
"""Week 2 — HONEST halting + auto-calibrated NTC controller + Pareto.

Key fixes over week 1:
  * HONEST scoring: re-extract the answer from the TRUNCATED chain at each halt
    point (early halts that cut the answer now correctly score wrong).
  * Momentum halting: Qwen3's per-step uncertainty is spiky with early 0.0s, so
    absolute-threshold halting fires immediately. NTC instead halts on a
    *smoothed* (EMA / momentum) signal sustained for `patience` steps — the
    Nested-Learning surprise-momentum idea — which only triggers once reasoning
    has genuinely converged.
  * SLOW prior (auto-calibration): tau is chosen on a warm-up split to maximise
    reward = accuracy - mu * tokens, instead of being guessed. This is the slow
    timescale of the nested controller.

Run:
    python scripts/ntc_week2_controller.py --model Qwen/Qwen3-1.7B \
        --benchmark gsm8k --limit 150 --max-tokens 1024 --tp-size 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tokenguard.baselines import prompts
from tokenguard.llm.generate import LLMRunner
from tokenguard.reasoning.datasets import load_benchmark, is_correct


def momentum_halt(unc, eta=0.6, tau=0.08, patience=4, min_steps=5):
    """Halt when the EMA-smoothed uncertainty stays <= tau for `patience` steps."""
    ema = None
    below = 0
    for t, u in enumerate(unc, start=1):
        ema = u if ema is None else eta * ema + (1 - eta) * u
        below = below + 1 if ema <= tau else 0
        if t >= min_steps and below >= patience:
            return t
    return len(unc)


def refrain_halt(unc, tau=0.15, patience=2, min_steps=1):
    below = 0
    for t, u in enumerate(unc, start=1):
        below = below + 1 if u <= tau else 0
        if t >= min_steps and below >= patience:
            return t
    return len(unc)


def mur_halt(unc, momentum=0.9, tau=0.15, min_steps=1):
    ema = None
    for t, u in enumerate(unc, start=1):
        ema = u if ema is None else momentum * ema + (1 - momentum) * u
        if t >= min_steps and ema <= tau:
            return t
    return len(unc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--benchmark", default="gsm8k",
                    choices=["gsm8k", "math500", "gpqa_diamond"])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--mu", type=float, default=0.0005,
                    help="token penalty in reward = acc - mu*tokens")
    ap.add_argument("--out", default="experiments/ntc/week2.json")
    args = ap.parse_args()

    data = load_benchmark(args.benchmark, limit=args.limit)
    runner = LLMRunner(model_name=args.model, tensor_parallel_size=args.tp_size)
    bench = args.benchmark

    # Generate full chains once; cache per-step text + uncertainty.
    gens = []
    for ex in data:
        g = runner.generate(ex["question"], max_tokens=args.max_tokens,
                            cot_prefix=prompts.COT)
        gens.append((ex, g))

    # warm-up / eval split (slow-prior calibration on warm-up only)
    n_warm = int(len(gens) * args.warmup_frac)
    warm, evalset = gens[:n_warm], gens[n_warm:]

    def score_at_halt(g, stop_step):
        if stop_step >= len(g.step_texts) or not g.step_texts:
            return g.text, g.n_gen_tokens
        return g.text_upto_step(stop_step), g.tokens_upto_step(stop_step)

    def eval_on(subset, halt_fn, **kw):
        accs, toks = [], []
        for ex, g in subset:
            stop = halt_fn(g.step_uncertainty, **kw) if g.step_uncertainty else len(g.step_uncertainty)
            tr, t = score_at_halt(g, stop)
            accs.append(is_correct(tr, ex["answer"], bench))
            toks.append(t)
        return float(np.mean(accs)), float(np.mean(toks))

    # full-chain reference (honest)
    full_acc, full_tok = eval_on(evalset, lambda u: len(u))

    # SLOW PRIOR: calibrate each method's tau on the warm-up split by reward.
    def calibrate(halt_fn, taus, extra):
        best, best_r = None, -1e18
        for tau in taus:
            a, t = eval_on(warm, halt_fn, tau=tau, **extra)
            r = a - args.mu * t
            if r > best_r:
                best_r, best = r, tau
        return best

    refrain_taus = [0.05, 0.1, 0.15, 0.2, 0.3]
    mur_taus = [0.05, 0.1, 0.15, 0.2, 0.3]
    ntc_taus = [0.03, 0.05, 0.08, 0.1, 0.12]

    refrain_tau = calibrate(refrain_halt, refrain_taus, {"patience": 2})
    mur_tau = calibrate(mur_halt, mur_taus, {"momentum": 0.9})
    ntc_tau = calibrate(momentum_halt, ntc_taus, {"eta": 0.6, "patience": 4, "min_steps": 5})

    # EVAL on held-out set with the calibrated tau (apples-to-apples)
    results = {}
    for name, fn, tau, extra in [
        ("REFRAIN", refrain_halt, refrain_tau, {"patience": 2}),
        ("MUR", mur_halt, mur_tau, {"momentum": 0.9}),
        ("NTC", momentum_halt, ntc_tau, {"eta": 0.6, "patience": 4, "min_steps": 5}),
    ]:
        a, t = eval_on(evalset, fn, tau=tau, **extra)
        results[name] = {"tau": tau, "acc": a, "tokens": t,
                         "cut": 100 * (1 - t / full_tok),
                         "acc_drop": full_acc - a}

    # also dump full Pareto curves for plotting
    def curve(fn, taus, extra):
        return [{"tau": tau, **dict(zip(("acc", "tokens"), eval_on(evalset, fn, tau=tau, **extra)))}
                for tau in taus]

    print(f"\n=== Week 2 (HONEST, calibrated) — {args.model} on {bench} ===")
    print(f"eval n={len(evalset)} (warm-up n={len(warm)})")
    print(f"full-chain: acc={full_acc:.3f}  tokens={full_tok:.1f}\n")
    print(f"{'method':<10}{'tau':>6}{'acc':>8}{'tokens':>9}{'cut%':>7}{'acc_drop':>10}")
    for name in ["REFRAIN", "MUR", "NTC"]:
        r = results[name]
        print(f"{name:<10}{r['tau']:>6}{r['acc']:>8.3f}{r['tokens']:>9.1f}"
              f"{r['cut']:>6.1f}%{r['acc_drop']:>9.3f}")

    # winner: highest accuracy among those cutting >=30% tokens (honest efficiency)
    eligible = {k: v for k, v in results.items() if v["cut"] >= 30}
    if eligible:
        win = max(eligible, key=lambda k: eligible[k]["acc"])
        print(f"\nBest efficient method (>=30% cut, highest acc): {win} "
              f"(acc={results[win]['acc']:.3f}, {results[win]['cut']:.0f}% fewer tokens)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "benchmark": bench,
        "full": {"acc": full_acc, "tokens": full_tok},
        "calibrated": results,
        "refrain_curve": curve(refrain_halt, refrain_taus, {"patience": 2}),
        "mur_curve": curve(mur_halt, mur_taus, {"momentum": 0.9}),
        "ntc_curve": curve(momentum_halt, ntc_taus, {"eta": 0.6, "patience": 4, "min_steps": 5}),
    }, indent=2))
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
