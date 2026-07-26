#!/usr/bin/env python
"""Week 1 rebuild — thinking-mode generation + answer probing (the foundation).

Runs a benchmark with Qwen3 in THINKING mode, then probes a forced answer at
checkpoints inside each thinking trace (halt-then-emit). Saves everything to a
rich JSON so ALL halting policies (DEER / EAT / NTC) are evaluated later
WITHOUT re-running the GPU.

Reports:
  * vanilla thinking accuracy (natural answer after </think>) — the ceiling
  * thinking-token distribution — proves the overthinking headroom exists
  * probe sanity — % of problems where an EARLY probe already answers correctly
    (that percentage IS the token-saving opportunity)

Run (batched, fast):
    python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B \
        --benchmark math500 --limit 100 --max-think 6144 --batch 16

GATE W1: vanilla accuracy ~ published Qwen3 numbers; median thinking tokens
large (>1k on MATH-500); early-probe-correct fraction >= 40%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tokenguard.llm.thinking import ThinkingRunner, build_checkpoints, Probe, ThinkTrace
from tokenguard.reasoning.datasets import load_benchmark, is_correct


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--benchmark", default="math500",
                    choices=["gsm8k", "math500", "gpqa_diamond", "aime24", "aime25", "mmlu_pro"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-think", type=int, default=6144)
    ap.add_argument("--probe-every", type=int, default=256)
    ap.add_argument("--max-probes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="0.0 = greedy (DEER-matched); 0.6 = Qwen3 default")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suffix = "" if args.seed == 42 else f"_s{args.seed}"
    out_path = args.out or (f"experiments/ntc/w1_{args.benchmark}_"
                            f"{args.model.split('/')[-1]}{suffix}.json")
    PROMPT_INSTR = {
        "math500": "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "aime25":  "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "aime24":  "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "gsm8k":   "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "gpqa_diamond": "\n\nPlease reason step by step, then answer with only the letter of the correct option within \\boxed{}.",
        "mmlu_pro": "\n\nPlease reason step by step, then answer with only the letter of the correct option within \\boxed{}.",
    }
    data = load_benchmark(args.benchmark, limit=args.limit)
    instr = PROMPT_INSTR.get(args.benchmark, "")
    for ex in data:
        ex["question"] = ex["question"] + instr
    runner = ThinkingRunner(model_name=args.model,
                            tensor_parallel_size=args.tp_size,
                            max_model_len=args.max_model_len, seed=args.seed)
    bench = args.benchmark

    traces: list[dict] = []
    for i in range(0, len(data), args.batch):
        chunk = data[i:i + args.batch]
        gens = runner.generate_thinking([ex["question"] for ex in chunk],
                                        max_tokens=args.max_think,
                                        temperature=args.temperature)
        # build ALL probe jobs for this chunk, run in one vLLM call
        jobs, owners = [], []
        ckpts_per_q = []
        for ex, g in zip(chunk, gens):
            cks = build_checkpoints(g["think_text"], runner.tok,
                                    probe_every=args.probe_every,
                                    max_probes=args.max_probes)
            ckpts_per_q.append(cks)
            for (_, prefix) in cks:
                jobs.append((ex["question"], prefix))
                owners.append(ex["id"])
        probes_flat = runner.probe_batch(jobs) if jobs else []

        # assemble traces
        pi = 0
        for ex, g, cks in zip(chunk, gens, ckpts_per_q):
            my_probes = probes_flat[pi:pi + len(cks)]
            pi += len(cks)
            nat_ans = g["answer_text"]
            nat_ok = bool(nat_ans) and is_correct(nat_ans, ex["answer"], bench)
            tr = ThinkTrace(
                qid=ex["id"], question=ex["question"], gold=ex["answer"],
                think_text=g["think_text"], n_think_tokens=g["n_think"],
                natural_answer=nat_ans, natural_correct=nat_ok,
                n_total_tokens=g["n_total"], finish_reason=g["finish"],
                token_nll=g.get("token_nll", []),
                probes=my_probes)
            traces.append(tr.to_dict())
        done = min(i + args.batch, len(data))
        print(f"[{done}/{len(data)}] generated + probed")

    # ---- report ----
    nat_acc = float(np.mean([t["natural_correct"] for t in traces]))
    think_toks = [t["n_think_tokens"] for t in traces]
    total_toks = [t["n_total_tokens"] for t in traces]
    n_len = sum(1 for t in traces if t["finish_reason"] == "length")

    # probe sanity: earliest probe whose forced answer is already correct
    early_correct, early_frac_tokens = 0, []
    for t in traces:
        for p in t["probes"]:
            if is_correct(p["answer"], t["gold"], bench):
                early_correct += 1
                early_frac_tokens.append(p["ckpt_tokens"] / max(1, t["n_think_tokens"]))
                break

    print(f"\n=== W1 thinking-mode report — {args.model} on {bench} (n={len(traces)}) ===")
    print(f"vanilla (natural answer) accuracy : {nat_acc:.3f}")
    print(f"thinking tokens  mean/median/p90  : {np.mean(think_toks):.0f} / "
          f"{np.median(think_toks):.0f} / {np.percentile(think_toks, 90):.0f}")
    print(f"total tokens mean                 : {np.mean(total_toks):.0f}")
    print(f"hit max-think cap (overthinkers)  : {n_len}/{len(traces)}")
    print(f"early-probe already-correct       : {early_correct}/{len(traces)} "
          f"({100*early_correct/len(traces):.0f}%)")
    if early_frac_tokens:
        print(f"  ...at median {100*float(np.median(early_frac_tokens)):.0f}% of the "
              f"thinking trace  ← the token-saving headroom")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"model": args.model, "benchmark": bench, "args": vars(args),
         "traces": traces}, indent=1))
    print(f"\nSaved {out_path}")
    print("Next: scripts/ntc_w1_policies.py evaluates DEER/EAT/NTC from this file (no GPU).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
