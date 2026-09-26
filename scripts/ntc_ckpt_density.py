#!/usr/bin/env python
"""Checkpoint density: does probing more often close the gap to DEER?

The controller probes every 256 thinking tokens, at most ten times; DEER
induces a trial answer at every reasoning transition. This sweep repeats the
matched MATH-500 run (greedy, 16k budget, n = 500, same grader) at two and
four times the checkpoint density and reports, per level: probes per item,
full-generation accuracy and cost, the overhead of probing without halting,
answer agreement (m = 3), the selection tier (calibrated on 40% of the items,
eps = 0.05, scored on the other 60%), and the error stickiness between
consecutive probes, rho_w, with P_spur = rho_w^(m-1) * q_w.

All levels of a model are scored on the items they share. Models without
density files are skipped. Writes experiments/ntc/CKPT_DENSITY.md.

    python scripts/ntc_ckpt_density.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import ntc_w1_stats as S
from ntc_h2h_table import deer_official

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
LEVELS = [("1x  (every 256 tok, <=10)", "h2h2_math500_Qwen3-{m}.json"),
          ("2x  (every 128 tok, <=20)", "dens2x_math500_Qwen3-{m}.json"),
          ("4x  (every  64 tok, <=40)", "dens4x_math500_Qwen3-{m}.json")]


def stickiness(traces, bench, m=3):
    """(rho_w, q_w, P_spur) between consecutive probes."""
    rho, qw = [], []
    for t in traces:
        pr = [p for p in t["probes"] if p.get("answer")]
        if len(pr) < 2:
            continue
        ans = [p["answer"] for p in pr]
        wrong = [not S.is_correct(x, t["gold"], bench) for x in ans]
        pairs = [1.0 if S.is_correct(ans[i + 1], ans[i], bench) else 0.0
                 for i in range(len(ans) - 1) if wrong[i]]
        if pairs:
            rho.append(float(np.mean(pairs)))
        bad = [x for x, w in zip(ans, wrong) if w]
        qw.append(Counter(bad).most_common(1)[0][1] / len(ans) if bad else 0.0)
    if not rho:
        return float("nan"), float("nan"), float("nan")
    r, q = float(np.mean(rho)), float(np.mean(qw))
    return r, q, (r ** (m - 1)) * q


def full_set(traces, bench, fn, kw):
    """(accuracy, overhead-inclusive online cost) of a fixed rule."""
    ok, tok = [], []
    for t in traces:
        pr = t["probes"]
        k = fn(pr, **({**kw, "bm": bench} if "bm" in fn.__code__.co_varnames else kw)) if pr else None
        if k is None:
            ok.append(bool(t["natural_correct"]))
            tok.append(t["n_total_tokens"] + sum(p["n_probe_tokens"] for p in pr))
        else:
            p = pr[k]
            ok.append(S.is_correct(p["answer"], t["gold"], bench))
            tok.append(p["ckpt_tokens"] + sum(q["n_probe_tokens"] for q in pr[:k + 1]))
    return float(np.mean(ok)), float(np.mean(tok))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--out", default=str(NTC / "CKPT_DENSITY.md"))
    a = ap.parse_args()

    deer = deer_official()
    models = []
    common = {}
    for model in ("Qwen3-4B", "Qwen3-8B"):
        files = [NTC / pat.format(m=model.split("-")[1]) for _, pat in LEVELS]
        if not all(f.exists() for f in files):
            print(f"{model}: density files not present; skipped")
            continue
        models.append(model)
        sets = [{t["qid"] for t in json.loads(f.read_text())["traces"]} for f in files]
        common[model] = set.intersection(*sets)

    md = ["# Checkpoint density: what does probing more often buy?", "",
          "MATH-500, n = 500, greedy decoding, 16k thinking budget, one symbolic grader. "
          "Each `acc@tok` cell is accuracy @ mean online tokens per item, counting every "
          "probe paid. `overhead` is the cost of probing at every checkpoint without "
          "halting, relative to full generation. The selection tier (NTC-Select) is "
          f"calibrated on {a.warmup_frac:.0%} of the items with eps = {a.eps} and scored on "
          "the rest. rho_w is the error stickiness between consecutive probes and "
          "P_spur = rho_w^2 * q_w. DEER is the authors' code at its default "
          "configuration, with overhead measured on the same convention.", "",
          "| model | density | probes/item | vanilla acc@tok | overhead | AGREE m=3 acc@tok "
          "| NTC-Select acc@tok | rho_w | P_spur | selected rule |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        for label, pat in LEVELS:
            d = json.loads((NTC / pat.format(m=model.split("-")[1])).read_text())
            bench = d["benchmark"]
            traces = [t for t in d["traces"] if t["qid"] in common[model]]
            for t in traces:
                t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            fams = dict(S.FAMILIES)
            if S.enrich_probes_with_nll(traces):
                fams["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
            ppi = float(np.mean([len(t["probes"]) for t in traces]))
            van_acc = float(np.mean([t["natural_correct"] for t in traces]))
            van_tok = float(np.mean([t["n_total_tokens"] for t in traces]))
            nh_tok = float(np.mean([t["n_total_tokens"] + sum(p["n_probe_tokens"] for p in t["probes"])
                                    for t in traces]))
            ovh = 100 * (nh_tok / van_tok - 1)
            ag_acc, ag_tok = full_set(traces, bench, S.agree_policy, {"m": 3})
            n = len(traces)
            idx = np.random.default_rng(0).permutation(n)
            warm = [traces[i] for i in idx[:int(n * a.warmup_frac)]]
            ev = [traces[i] for i in idx[int(n * a.warmup_frac):]]
            saved = S.FAMILIES
            S.FAMILIES = fams
            try:
                _, (gfam, gkw) = S.calibrate(warm, bench, eps=a.eps)
            finally:
                S.FAMILIES = saved
            nf_acc, nf_tok = full_set(ev, bench, fams[gfam][0], gkw)
            rho, _, psp = stickiness(traces, bench)
            md.append(f"| {model} | {label} | {ppi:.1f} | {van_acc:.3f} @ {van_tok:.0f} "
                      f"| {ovh:+.1f}% | {ag_acc:.3f} @ {ag_tok:.0f} "
                      f"| {nf_acc:.3f} @ {nf_tok:.0f} | {rho:.3f} | {psp:.3f} | {gfam}{gkw} |")
            print(f"{model} {label}: probes/item {ppi:.1f}  full {van_acc:.3f}@{van_tok:.0f}  "
                  f"overhead {ovh:+.1f}%  AGREE {ag_acc:.3f}@{ag_tok:.0f}  "
                  f"NTC-Select {nf_acc:.3f}@{nf_tok:.0f}  rho_w {rho:.3f}  P_spur {psp:.3f} "
                  f"({gfam}{gkw})", flush=True)
        acc, fin, tot = deer[(model, "math500")]
        md.append(f"| {model} | DEER (authors' code) | n/a | n/a | {100*(tot-fin)/fin:+.1f}% | n/a "
                  f"| {acc:.3f} @ {tot:.0f} | n/a | n/a | threshold 0.95 |")
        print(f"{model} DEER: {acc:.3f}@{tot:.0f}  overhead {100*(tot-fin)/fin:+.1f}%")
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
