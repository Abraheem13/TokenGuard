#!/usr/bin/env python
"""Generate a probe stream: thinking-mode generation with answer probing (GPU).

Runs a benchmark with a reasoning model in thinking mode, then probes a forced
answer at checkpoints inside each thinking trace: the thinking prefix up to the
checkpoint is followed by the cue `</think>\n\nThe final answer is \boxed{` and
at most 24 greedy tokens are read. Each probe records the trial answer, its
confidence (geometric mean of the chosen-token probabilities), the entropy of
the first answer token and the tokens it cost; each trace also records the
per-token NLL of the thinking pass. Every halting rule is later scored by
replaying this file, with no further GPU work.

    python scripts/ntc_w1_thinking.py --model Qwen/Qwen3-4B --benchmark math500 \
        --limit 200 --max-think 8192 --batch 16 --seed 42

The settings of every committed probe file, and the command that regenerates
it, are listed in experiments/ntc/PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenguard.llm.thinking import ThinkingRunner, ThinkTrace, build_checkpoints
from tokenguard.reasoning.datasets import is_correct, load_benchmark

BOXED = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
LETTER = ("\n\nPlease reason step by step, then answer with only the letter of the "
          "correct option within \\boxed{}.")
INSTRUCTION = {"math500": BOXED, "aime24": BOXED, "aime25": BOXED, "gsm8k": BOXED,
               "gpqa_diamond": LETTER, "mmlu_pro": LETTER}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--benchmark", default="math500",
                    choices=["gsm8k", "math500", "gpqa_diamond", "aime24", "aime25", "mmlu_pro"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-think", type=int, default=6144, help="thinking-token budget")
    ap.add_argument("--probe-every", type=int, default=256)
    ap.add_argument("--max-probes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="0.0 for greedy decoding; 0.6 is the Qwen3 default")
    ap.add_argument("--no-instruction", action="store_true",
                    help="omit the answer-format instruction appended to each question")
    ap.add_argument("--source-order", action="store_true",
                    help="keep GPQA-Diamond options in source order (correct option first)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suffix = "" if args.seed == 42 else f"_s{args.seed}"
    out_path = args.out or (f"experiments/ntc/w1_{args.benchmark}_"
                            f"{args.model.split('/')[-1]}{suffix}.json")
    data = load_benchmark(args.benchmark, limit=args.limit,
                          shuffle_options=not args.source_order)
    instr = "" if args.no_instruction else INSTRUCTION.get(args.benchmark, "")
    for ex in data:
        ex["question"] = ex["question"] + instr
    runner = ThinkingRunner(model_name=args.model, tensor_parallel_size=args.tp_size,
                            max_model_len=args.max_model_len, seed=args.seed)
    bench = args.benchmark

    traces: list[dict] = []
    for i in range(0, len(data), args.batch):
        chunk = data[i:i + args.batch]
        gens = runner.generate_thinking([ex["question"] for ex in chunk],
                                        max_tokens=args.max_think,
                                        temperature=args.temperature)
        jobs, ckpts_per_q = [], []
        for ex, g in zip(chunk, gens):
            cks = build_checkpoints(g["think_text"], runner.tok,
                                    probe_every=args.probe_every,
                                    max_probes=args.max_probes)
            ckpts_per_q.append(cks)
            jobs += [(ex["question"], prefix) for _, prefix in cks]
        probes_flat = runner.probe_batch(jobs) if jobs else []

        pi = 0
        for ex, g, cks in zip(chunk, gens, ckpts_per_q):
            my_probes = probes_flat[pi:pi + len(cks)]
            pi += len(cks)
            nat_ans = g["answer_text"]
            tr = ThinkTrace(
                qid=ex["id"], question=ex["question"], gold=ex["answer"],
                think_text=g["think_text"], n_think_tokens=g["n_think"],
                natural_answer=nat_ans,
                natural_correct=bool(nat_ans) and is_correct(nat_ans, ex["answer"], bench),
                n_total_tokens=g["n_total"], finish_reason=g["finish"],
                token_nll=g.get("token_nll", []), probes=my_probes)
            traces.append(tr.to_dict())
        print(f"[{min(i + args.batch, len(data))}/{len(data)}] generated and probed")

    nat_acc = float(np.mean([t["natural_correct"] for t in traces]))
    think_toks = [t["n_think_tokens"] for t in traces]
    n_len = sum(1 for t in traces if t["finish_reason"] == "length")
    early = 0
    for t in traces:
        if any(is_correct(p["answer"], t["gold"], bench) for p in t["probes"]):
            early += 1
    print(f"\n{args.model} on {bench} (n={len(traces)})")
    print(f"full-generation accuracy           : {nat_acc:.3f}")
    print(f"thinking tokens mean/median/p90    : {np.mean(think_toks):.0f} / "
          f"{np.median(think_toks):.0f} / {np.percentile(think_toks, 90):.0f}")
    print(f"traces reaching the budget         : {n_len}/{len(traces)}")
    print(f"traces with a correct trial answer : {early}/{len(traces)}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"model": args.model, "benchmark": bench, "args": vars(args), "traces": traces},
        indent=1))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
