#!/usr/bin/env python
"""Corpus accounting by experimental track.

Every probe file belongs to exactly one track:

  primary       generation-seed replicates behind the per-setting and aggregate
                results (`w1_*`, and `w1sh_*` for GPQA-Diamond with shuffled options)
  head-to-head  the matched comparison with DEER (`h2h2_*`)
  density       the checkpoint-density sweep (`dens*`); its 1x level is the
                head-to-head MATH-500 run and is counted there

For each track and benchmark it reports the size of the official test split,
the items sampled, the thinking budget, the reasoning traces, the graded trial
answers, the mean tokens per trace and the range of truncation rates over files.

Writes experiments/ntc/CORPUS_TRACKS.md.

    python scripts/ntc_corpus_tracks.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
POOL = {"gsm8k": 1319, "math500": 500, "gpqa_diamond": 198,
        "aime24": 30, "aime25": 30, "mmlu_pro": 12032}
ORDER = ["gsm8k", "math500", "gpqa_diamond", "mmlu_pro", "aime24", "aime25"]
LABEL = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-Diamond",
         "mmlu_pro": "MMLU-Pro", "aime24": "AIME-2024", "aime25": "AIME-2025"}
TRACKS = (("primary", ("w1_", "w1sh_")), ("head-to-head", ("h2h2_",)),
          ("density", ("dens",)))


def track(name: str) -> str | None:
    for label, prefixes in TRACKS:
        if name.startswith(prefixes):
            return label
    return None


def main() -> int:
    per = defaultdict(lambda: defaultdict(lambda: {
        "files": 0, "items": set(), "traces": 0, "probes": 0,
        "tokens": 0.0, "trunc": [], "budget": set(), "qids": set()}))
    for f in sorted(NTC.glob("*.json")):
        tr = track(f.name)
        if tr is None:
            continue
        d = json.loads(f.read_text())
        traces, bench = d["traces"], d["benchmark"]
        s = per[tr][bench]
        s["files"] += 1
        s["items"].add(len(traces))
        s["traces"] += len(traces)
        s["probes"] += sum(len(t.get("probes", [])) for t in traces)
        s["tokens"] += float(np.sum([t["n_total_tokens"] for t in traces]))
        capped = sum(1 for t in traces if t.get("finish_reason") != "stop")
        s["trunc"].append(100 * capped / max(1, len(traces)))
        s["budget"].add(int(d.get("args", {}).get("max_think", 0)))
        s["qids"].update(f"{bench}:{t.get('qid', i)}" for i, t in enumerate(traces))

    md = ["# Corpus by experimental track", "",
          "Tracks: `primary` = `w1_*` and `w1sh_*` (GPQA-Diamond with shuffled options); "
          "`head-to-head` = `h2h2_*`; `density` = `dens*`. `items` is the number of "
          "questions per file; tokens per trace is the mean of `n_total_tokens`; "
          "truncation is the share of a file's traces that reach the thinking budget, as "
          "a range over files.", ""]
    for tr, _ in TRACKS:
        if tr not in per:
            continue
        md += [f"## {tr}", "",
               "| benchmark | files | pool | items | budget | traces | probes "
               "| tokens per trace | truncation |", "|---|---|---|---|---|---|---|---|---|"]
        tot_tr = tot_pr = 0
        for b in [x for x in ORDER if x in per[tr]]:
            s = per[tr][b]
            budgets = "/".join(str(x) for x in sorted(s["budget"]))
            items = "/".join(str(x) for x in sorted(s["items"]))
            md.append(f"| {LABEL[b]} | {s['files']} | {POOL[b]} | {items} | {budgets} "
                      f"| {s['traces']} | {s['probes']} "
                      f"| {s['tokens'] / s['traces']:.0f} "
                      f"| {min(s['trunc']):.1f} to {max(s['trunc']):.1f}% |")
            tot_tr += s["traces"]
            tot_pr += s["probes"]
        md += [f"| **total** | | | | | **{tot_tr}** | **{tot_pr}** | | |", ""]

    all_tr = sum(s["traces"] for t in per.values() for s in t.values())
    all_pr = sum(s["probes"] for t in per.values() for s in t.values())
    qs = len(set().union(*[s["qids"] for s in per["primary"].values()]))
    qs_all = len(set().union(*[s["qids"] for t in per.values() for s in t.values()]))
    md += ["## All tracks", "",
           "| quantity | value |", "|---|---|",
           f"| distinct questions in the primary track | {qs} |",
           f"| distinct questions in all tracks | {qs_all} |",
           f"| reasoning traces | {all_tr} |",
           f"| graded trial answers | {all_pr} |", ""]
    out = NTC / "CORPUS_TRACKS.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"table: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
