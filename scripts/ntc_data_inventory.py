#!/usr/bin/env python
"""Inventory of every probe file, and of what the analyses fit.

For each probe file: benchmark, model, items, size of the official test split,
calibration and evaluation split sizes, checkpoints per item, mean thinking
tokens and truncation rate; then totals per benchmark, a statement of what is
fitted and on which items, and how items were sampled.

Writes experiments/ntc/DATA_INVENTORY.md.

    python scripts/ntc_data_inventory.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
WARM = 0.4
FULL_SIZES = {"gsm8k": 1319, "math500": 500, "gpqa_diamond": 198,
              "aime24": 30, "aime25": 30, "mmlu_pro": 12032}


def main() -> int:
    rows, tot_items, tot_probes = [], 0, 0
    for f in sorted(NTC.glob("*.json")):
        d = json.loads(f.read_text())
        if "traces" not in d:
            continue
        traces, bench = d["traces"], d["benchmark"]
        n = len(traces)
        pr = sum(len(t.get("probes", [])) for t in traces)
        think = float(np.mean([t.get("n_think_tokens", t["n_total_tokens"]) for t in traces]))
        capped = sum(1 for t in traces if t.get("finish_reason") != "stop")
        rows.append({"file": f.name, "bench": bench, "model": d["model"].split("/")[-1],
                     "n": n, "probes": pr, "ckpt_per_item": pr / max(1, n), "think": think,
                     "cap": 100 * capped / max(1, n), "warm": int(n * WARM),
                     "eval": n - int(n * WARM), "pool": FULL_SIZES[bench]})
        tot_items += n
        tot_probes += pr

    md = ["# Probe files and protocol inventory", "",
          "| probe file | benchmark | model | items | test pool | calib | eval "
          "| checkpoints/item | mean thinking tok | truncated |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    print(f"{'benchmark':15s} {'model':12s} {'items':>6s} {'pool':>7s} "
          f"{'calib':>6s} {'eval':>6s} {'ckpt/it':>8s} {'think':>8s} {'trunc':>7s}")
    for r in rows:
        print(f"{r['bench']:15s} {r['model']:12s} {r['n']:6d} {r['pool']:>7d} "
              f"{r['warm']:6d} {r['eval']:6d} {r['ckpt_per_item']:8.1f} "
              f"{r['think']:8.0f} {r['cap']:6.1f}%")
        md.append(f"| `{r['file']}` | {r['bench']} | {r['model']} | {r['n']} "
                  f"| {r['pool']} | {r['warm']} | {r['eval']} "
                  f"| {r['ckpt_per_item']:.1f} | {r['think']:.0f} | {r['cap']:.1f}% |")

    by_b = {}
    for r in rows:
        by_b.setdefault(r["bench"], [0, 0])
        by_b[r["bench"]][0] += r["n"]
        by_b[r["bench"]][1] += r["probes"]
    md += ["", "## Totals", "",
           "| benchmark | reasoning traces | graded trial answers |", "|---|---|---|"]
    for b, (n, pr) in sorted(by_b.items()):
        md.append(f"| {b} | {n} | {pr} |")
    md.append(f"| **all** | **{tot_items}** | **{tot_probes}** |")
    print(f"\n{'all files':15s} {tot_items:6d} traces  {tot_probes:8d} trial answers")

    md += ["", "## What is fitted, and on which items", "",
           "* No model weights are trained or updated; the language models are used as "
           "released.",
           "* The selection tier chooses one (rule, parameter) pair per setting by "
           "repeated paired cross-validation on the calibration split only (40% of "
           "items). The evaluation split (60%) is never read during selection. For the "
           "AIME sets, calibration pools the same questions from every generation seed, "
           "so no evaluation question enters calibration in any generation.",
           "* The joint tier's router is a TF-IDF and logistic-regression classifier "
           "fitted on calibration items only, with the label 'the small model is correct "
           "under its calibrated rule'.",
           "* Every halting rule is otherwise a deterministic function of the probe stream.",
           "",
           "## Sampling", "",
           "Where `items` is smaller than `test pool`, the items are the first items of "
           "the official test split (GSM8K, MATH-500) or a seeded random sample (MMLU-Pro). "
           "GPQA-Diamond and both AIME sets are used in full. GPQA-Diamond lists the "
           "correct option first, so the primary track (`w1sh_*`) shuffles the four "
           "options deterministically per item; the matched comparison with DEER "
           "(`h2h2_*`) keeps the source order, which is also the order of DEER's own "
           "data file, so that both systems see identical items."]
    (NTC / "DATA_INVENTORY.md").write_text("\n".join(md) + "\n")
    print(f"table: {NTC / 'DATA_INVENTORY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
