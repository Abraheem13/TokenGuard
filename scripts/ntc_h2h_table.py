#!/usr/bin/env python
"""Assemble the paper's head-to-head Table 1: DEER (authors' official code,
default config) vs our methods, under IDENTICAL conditions (same models, same
data, 16k thinking budget, greedy decoding, same sympy grader).

Fairness protocol:
  * DEER ran with its fixed default threshold (0.95) on the FULL set.
  * We therefore report AGREE with its fixed default (m=3) on the FULL set
    (apples-to-apples: fixed defaults vs fixed defaults), plus NTC-full whose
    (signal, param) is calibrated on a 40% warm-up and reported on the 60%
    held-out split (marked with a dagger in the table).
  * Token accounting: ours = thinking-at-halt + emitted answer + ALL probe
    tokens paid online (overhead-inclusive); DEER's token_num likewise
    includes its trial-answer inductions. Both are the honest online cost.

Usage:
    python scripts/ntc_h2h_table.py \
        --h2h experiments/ntc/h2h_math500_Qwen3-4B.json ... (all six)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parents[0] / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("w1stats", _here / "ntc_w1_stats.py")
S = importlib.util.module_from_spec(spec)
sys.modules["w1stats"] = S
spec.loader.exec_module(S)
import tokenguard.reasoning.datasets as _ds
import importlib as _il
_il.reload(_ds)
S.is_correct = _ds.is_correct

DEER_OFFICIAL = {  # from experiments/ntc/DEER_OFFICIAL.md (authors' code)
    ("Qwen3-4B", "math500"): (0.9200, 3538.8),
    ("Qwen3-4B", "gpqa_diamond"): (0.5455, 7535.9),
    ("Qwen3-4B", "aime24"): (0.6667, 10534.8),
    ("Qwen3-8B", "math500"): (0.9300, 2946.2),
    ("Qwen3-8B", "gpqa_diamond"): (0.5758, 8872.2),
    ("Qwen3-8B", "aime24"): (0.6667, 10011.5),
}


def full_set_policy(traces, bench, fn, kw):
    """(acc, overhead-inclusive tokens) for a FIXED policy on the full set."""
    ok, tok, ovh = [], [], []
    for t in traces:
        probes = t["probes"]
        kk = fn(probes, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) \
             if probes else None
        if kk is None:
            ok.append(bool(t["natural_correct"]))
            ovh.append(t["n_total_tokens"] + sum(p["n_probe_tokens"] for p in probes))
        else:
            p = probes[kk]
            ok.append(S.is_correct(p["answer"], t["gold"], bench))
            ovh.append(p["ckpt_tokens"] + sum(q["n_probe_tokens"] for q in probes[:kk + 1]))
    return float(np.mean(ok)), float(np.mean(ovh))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h2h", action="append", required=True)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--out", default="experiments/ntc/H2H_TABLE.md")
    args = ap.parse_args()

    rows = []
    for pf in args.h2h:
        d = json.loads(Path(pf).read_text())
        traces, bench, model = d["traces"], d["benchmark"], d["model"].split("/")[-1]
        # rescore natural_correct with current grader (retroactive fairness)
        for t in traces:
            t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""),
                                                     t["gold"], bench))
        van_acc = float(np.mean([t["natural_correct"] for t in traces]))
        van_tok = float(np.mean([t["n_total_tokens"] for t in traces]))

        # AGREE m=3 fixed default on FULL set (mirror of DEER's fixed 0.95)
        ag_acc, ag_tok = full_set_policy(traces, bench, S.agree_policy,
                                         {"m": 3})
        # NTC-full calibrated (warm-up 40%), held-out reported
        n = len(traces)
        rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        warm = [traces[i] for i in idx[:int(n * args.warmup_frac)]]
        ev = [traces[i] for i in idx[int(n * args.warmup_frac):]]
        if S.enrich_probes_with_nll(traces):
            S.FAMILIES["MUR-mom"] = (S.mur_policy,
                                     [{"gamma": g} for g in (0.7, 0.8, 0.9)])
        picks, (gfam, gkw) = S.calibrate(warm, bench, eps=0.01)
        nf_acc, nf_tok = full_set_policy(ev, bench, S.FAMILIES[gfam][0], gkw)
        ev_van_tok = float(np.mean([t["n_total_tokens"] for t in ev]))

        deer = DEER_OFFICIAL.get((model, bench), (float("nan"), float("nan")))
        rows.append({
            "model": model, "bench": bench, "n": n,
            "van_acc": van_acc, "van_tok": van_tok,
            "deer_acc": deer[0], "deer_tok": deer[1],
            "ag_acc": ag_acc, "ag_tok": ag_tok,
            "nf_acc": nf_acc, "nf_tok": nf_tok, "nf_pick": f"{gfam}{gkw}",
            "ev_van_tok": ev_van_tok,
        })
        print(f"[done] {model} {bench}: vanilla {van_acc:.3f}@{van_tok:.0f} | "
              f"AGREE {ag_acc:.3f}@{ag_tok:.0f} | DEER {deer[0]:.3f}@{deer[1]:.0f} | "
              f"NTC-full† {nf_acc:.3f}@{nf_tok:.0f} ({gfam})")

    md = ["# Head-to-head Table 1 — identical conditions",
          "(same models, data, 16k thinking budget, greedy decoding, sympy grader;",
          "token counts are ONLINE cost incl. all probe/trial tokens)",
          "",
          "| model | benchmark | vanilla acc@tok | DEER official acc@tok "
          "| AGREE(m=3) acc@tok | NTC-full† acc@tok | NTC-full pick |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['bench']} (n={r['n']}) "
                  f"| {r['van_acc']:.3f} @ {r['van_tok']:.0f} "
                  f"| {r['deer_acc']:.3f} @ {r['deer_tok']:.0f} "
                  f"| {r['ag_acc']:.3f} @ {r['ag_tok']:.0f} "
                  f"| {r['nf_acc']:.3f} @ {r['nf_tok']:.0f} "
                  f"| {r['nf_pick']} |")
    md += ["", "† NTC-full: (signal, param) calibrated on 40% warm-up, held-out "
           "60% reported; others are fixed-default policies on the full set "
           "(DEER's default lambda=0.95, AGREE's default m=3)."]
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
