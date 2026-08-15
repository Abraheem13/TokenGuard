#!/usr/bin/env python
"""Dataset and protocol inventory — the exact accounting an examiner asks for.

Walks every probe file and reports: benchmark, model, how many items were
evaluated, how many checkpoints/probes were generated, the mean thinking
length, the truncation rate, and the calibration/evaluation split sizes.
Also totals the corpus so the dissertation can state exactly how much data
the study rests on, and states precisely what is fitted (nothing is trained
by gradient descent; only the slow-tier selection and the routing classifier
are fitted, both on warm-up items only).

    python scripts/ntc_data_inventory.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NTC = Path("experiments/ntc")
WARM = 0.4

FULL_SIZES = {  # official size of the underlying test set, for the sampling note
    "gsm8k": 1319, "math500": 500, "gpqa_diamond": 198,
    "aime24": 30, "aime25": 30, "mmlu_pro": 12032,
}


def main() -> int:
    rows, tot_items, tot_probes = [], 0, 0
    for f in sorted(NTC.glob("*.json")):
        if f.name in ("grader_cache.json",):
            continue
        try:
            d = json.loads(f.read_text())
            traces = d["traces"]
            bench = d["benchmark"]
        except Exception:
            continue
        n = len(traces)
        pr = sum(len(t.get("probes", [])) for t in traces)
        think = float(np.mean([t.get("n_think_tokens", t["n_total_tokens"])
                               for t in traces]))
        capped = sum(1 for t in traces if t.get("finish_reason") != "stop")
        rows.append({
            "file": f.name, "bench": bench,
            "model": d["model"].split("/")[-1], "n": n, "probes": pr,
            "ckpt_per_item": pr / max(1, n), "think": think,
            "cap": 100 * capped / max(1, n),
            "warm": int(n * WARM), "eval": n - int(n * WARM),
            "pool": FULL_SIZES.get(bench, "—"),
        })
        tot_items += n
        tot_probes += pr

    md = ["# Dataset and protocol inventory", "",
          "| probe file | benchmark | model | items | test pool | calib | eval "
          "| checkpoints/item | mean thinking tok | truncated |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    print(f"{'benchmark':15s} {'model':12s} {'items':>6s} {'pool':>7s} "
          f"{'calib':>6s} {'eval':>6s} {'ckpt/it':>8s} {'think':>8s} {'trunc':>7s}")
    for r in rows:
        print(f"{r['bench']:15s} {r['model']:12s} {r['n']:6d} {str(r['pool']):>7s} "
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
           "| benchmark | reasoning traces | forced trial answers |", "|---|---|---|"]
    print("\nTOTALS")
    for b, (n, pr) in sorted(by_b.items()):
        print(f"  {b:15s} {n:6d} traces  {pr:8d} probes")
        md.append(f"| {b} | {n} | {pr} |")
    print(f"  {'ALL':15s} {tot_items:6d} traces  {tot_probes:8d} probes")
    md.append(f"| **all** | **{tot_items}** | **{tot_probes}** |")

    md += ["", "## What is fitted, and on what", "",
           "* **Nothing is trained by gradient descent.** The language models are "
           "used off the shelf; no weights are updated at any point.",
           "* **Slow-tier signal selection** chooses one (signal, parameter) pair "
           "per domain by repeated paired cross-validation on the warm-up split "
           "only (40% of items). The evaluation split (60%) is never seen during "
           "selection, and for avg@k benchmarks the pooled warm-up draws the same "
           "question indices from every generation seed, so no evaluation question "
           "enters calibration in any generation.",
           "* **Routing classifier** (joint tier) is a TF-IDF + logistic model "
           "fitted on warm-up items only, with labels 'is the small model correct "
           "under its calibrated halting policy?'.",
           "* **Everything else is parameter-free at test time**: the halting "
           "signals are deterministic functions of the probe stream.", "",
           "## Sampling",
           "Where `items` is smaller than `test pool`, items are a deterministic "
           "prefix (MATH-500, GSM8K) or a seeded stratified sample (MMLU-Pro) of "
           "the official test split; GPQA-Diamond, AIME-24 and AIME-25 are used in "
           "full. Options in GPQA-Diamond are deterministically shuffled per item "
           "to remove the gold-position artifact present in the public loader.",
           ]
    Path(NTC / "DATA_INVENTORY.md").write_text("\n".join(md) + "\n")
    print("\ntable: experiments/ntc/DATA_INVENTORY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
