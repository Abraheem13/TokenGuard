#!/usr/bin/env python
"""Week 1 — real-LLM harness + reasoning benchmarks + reproduce baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tokenguard.baselines import prompts
from tokenguard.baselines.halting import refrain_halt, mur_halt, vanilla_halt
from tokenguard.llm.generate import LLMRunner
from tokenguard.reasoning.datasets import load_benchmark, is_correct


def _step_token_counts(text: str, n_gen: int, n_steps: int) -> list[int]:
    if n_steps == 0:
        return [n_gen]
    base = n_gen // n_steps
    counts = [base] * n_steps
    counts[-1] += n_gen - base * n_steps
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--benchmark", default="gsm8k",
                    choices=["gsm8k", "math500", "gpqa_diamond"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--mode", choices=["cot", "cod"], default="cot")
    ap.add_argument("--out", default="experiments/ntc/week1.json")
    args = ap.parse_args()

    data = load_benchmark(args.benchmark, limit=args.limit)
    runner = LLMRunner(model_name=args.model, tensor_parallel_size=args.tp_size)
    cot_prefix = prompts.COT if args.mode == "cot" else prompts.CHAIN_OF_DRAFT

    rows = []
    for ex in data:
        g = runner.generate(ex["question"], max_tokens=args.max_tokens,
                            cot_prefix=cot_prefix)
        steps = g.step_uncertainty
        stoks = _step_token_counts(g.text, g.n_gen_tokens, len(steps))
        correct = is_correct(g.text, ex["answer"], args.benchmark)
        rows.append({
            "id": ex["id"], "correct": bool(correct),
            "n_gen": g.n_gen_tokens, "n_steps": len(steps),
            "step_unc": steps, "step_toks": stoks,
        })

    def eval_baseline(halt_fn, **kw):
        accs, toks = [], []
        for r in rows:
            stop = halt_fn(r["step_unc"], **kw) if r["step_unc"] else r["n_steps"]
            toks.append(sum(r["step_toks"][:stop]) if r["step_toks"] else r["n_gen"])
            accs.append(r["correct"])
        return float(np.mean(accs)), float(np.mean(toks))

    vanilla_acc, vanilla_tok = eval_baseline(vanilla_halt)
    refrain_acc, refrain_tok = eval_baseline(refrain_halt, tau=args.tau, patience=2)
    mur_acc, mur_tok = eval_baseline(mur_halt, tau=args.tau, momentum=0.9)

    print(f"\n=== Week 1 — {args.model} on {args.benchmark} (n={len(rows)}, mode={args.mode}) ===")
    print(f"{'method':<12}{'accuracy':>10}{'mean_tokens':>14}{'token_cut':>12}")
    for name, a, t in [("vanilla", vanilla_acc, vanilla_tok),
                       ("REFRAIN", refrain_acc, refrain_tok),
                       ("MUR", mur_acc, mur_tok)]:
        cut = 100 * (1 - t / vanilla_tok) if vanilla_tok else 0
        print(f"{name:<12}{a:>10.3f}{t:>14.1f}{cut:>11.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "benchmark": args.benchmark, "rows": rows,
         "summary": {"vanilla": [vanilla_acc, vanilla_tok],
                     "refrain": [refrain_acc, refrain_tok],
                     "mur": [mur_acc, mur_tok]}}, indent=2))
    print(f"\nSaved {args.out}")
    print("GATE W1: check REFRAIN/MUR cut tokens at ~equal accuracy (vs papers).")
    print("Next: scripts/ntc_week2_controller.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
