#!/usr/bin/env python
"""Matched comparison with DEER.

DEER was run from its authors' code at its default configuration (threshold
0.95) on the same models and items, with a 16k thinking budget, greedy decoding
and the same grader; its results are recorded in experiments/ntc/DEER_OFFICIAL.md.
Under identical conditions this script reports, per setting:

  * full generation, with the last probe's answer used for traces that reach the
    budget without an answer (budget forcing, as DEER does);
  * answer agreement at its default m = 3, on all items;
  * the selection tier, calibrated on 40% of the items and scored on the other 60%.

Token counts are the online cost: thinking up to the halt plus every probe paid,
matching DEER's count, which includes its trial answers.

Writes experiments/ntc/H2H_TABLE.md.

    python scripts/ntc_h2h_table.py --h2h experiments/ntc/h2h2_math500_Qwen3-4B.json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ntc_w1_stats as S

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
BENCH = {"math500": "math500", "gpqa": "gpqa_diamond", "aime24": "aime24"}


def deer_official() -> dict:
    """(model, benchmark) -> (accuracy, final-chain tokens, total tokens incl. trials)."""
    out = {}
    for ln in (NTC / "DEER_OFFICIAL.md").read_text().split("\n## ")[0].splitlines():
        c = [x.strip() for x in ln.strip("|").split("|")]
        if ln.startswith("|") and len(c) == 5 and c[1] in BENCH:
            out[(c[0], BENCH[c[1]])] = (float(c[2]), float(c[3]), float(c[4]))
    return out


def full_set_policy(traces, bench, fn, kw):
    """(accuracy, overhead-inclusive tokens) of a fixed rule on the given items."""
    ok, ovh = [], []
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

    deer_all = deer_official()
    rows = []
    for pf in args.h2h:
        d = json.loads(Path(pf).read_text())
        traces, bench, model = d["traces"], d["benchmark"], d["model"].split("/")[-1]
        for t in traces:
            t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""),
                                                     t["gold"], bench))
        bf_ok, bf_tok = [], []
        for t in traces:
            if t["natural_correct"] or not t["probes"]:
                bf_ok.append(t["natural_correct"])
                bf_tok.append(t["n_total_tokens"])
            elif not t.get("natural_answer", "").strip():
                lp = t["probes"][-1]
                bf_ok.append(S.is_correct(lp["answer"], t["gold"], bench))
                bf_tok.append(t["n_total_tokens"] + lp["n_probe_tokens"])
            else:
                bf_ok.append(t["natural_correct"])
                bf_tok.append(t["n_total_tokens"])
        vbf_acc, vbf_tok = float(np.mean(bf_ok)), float(np.mean(bf_tok))

        ag_acc, ag_tok = full_set_policy(traces, bench, S.agree_policy, {"m": 3})

        n = len(traces)
        idx = np.random.default_rng(0).permutation(n)
        warm = [traces[i] for i in idx[:int(n * args.warmup_frac)]]
        ev = [traces[i] for i in idx[int(n * args.warmup_frac):]]
        if S.enrich_probes_with_nll(traces):
            S.FAMILIES["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
        _, (gfam, gkw) = S.calibrate(warm, bench, eps=0.01)
        nf_acc, nf_tok = full_set_policy(ev, bench, S.FAMILIES[gfam][0], gkw)

        deer = deer_all.get((model, bench), (float("nan"),) * 3)
        rows.append({"model": model, "bench": bench, "n": n,
                     "vbf_acc": vbf_acc, "vbf_tok": vbf_tok,
                     "deer_acc": deer[0], "deer_tok": deer[2],
                     "ag_acc": ag_acc, "ag_tok": ag_tok,
                     "nf_acc": nf_acc, "nf_tok": nf_tok, "nf_pick": f"{gfam}{gkw}"})
        print(f"{model} {bench}: full generation {vbf_acc:.3f} | AGREE {ag_acc:.3f}@{ag_tok:.0f} | "
              f"DEER {deer[0]:.3f}@{deer[2]:.0f} | NTC-full {nf_acc:.3f}@{nf_tok:.0f} ({gfam})")

    md = ["# Matched comparison with DEER", "",
          "Same models and items, 16k thinking budget, greedy decoding, one symbolic "
          "grader. Each cell is accuracy @ mean tokens per item, counting every probe or "
          "trial answer paid online. `vanilla-BF`: full generation with budget forcing.", "",
          "| model | benchmark | vanilla-BF acc@tok | DEER official acc@tok "
          "| AGREE(m=3) acc@tok | NTC-full† acc@tok | NTC-full pick |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['bench']} (n={r['n']}) "
                  f"| {r['vbf_acc']:.3f} @ {r['vbf_tok']:.0f} "
                  f"| {r['deer_acc']:.3f} @ {r['deer_tok']:.0f} "
                  f"| {r['ag_acc']:.3f} @ {r['ag_tok']:.0f} "
                  f"| {r['nf_acc']:.3f} @ {r['nf_tok']:.0f} "
                  f"| {r['nf_pick']} |")
    md += ["", "† NTC-full (the selection tier) is calibrated on 40% of the items and scored "
           "on the other 60%; the other columns are fixed defaults on all items (DEER "
           "threshold 0.95, agreement m = 3)."]
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
