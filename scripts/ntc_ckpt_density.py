#!/usr/bin/env python
"""Does checkpoint density explain the accuracy gap to DEER?

Section 6.6 attributes our residual accuracy gap on MATH-500 to granularity: we
probe every 256 thinking tokens and cap at 10 probes, while DEER interrupts at
every reasoning transition and pays 66-79% of the chain in trial answers.  This
script measures the trade directly across density levels produced by
run_queue_density.sh, under conditions otherwise identical to the head-to-head
track (greedy, 16k budget, n=500, same grader).

Runs on whatever density files exist and says which are missing, so it can be
used before and after the GPU sweep.  Writes experiments/ntc/CKPT_DENSITY.md.

    python scripts/ntc_ckpt_density.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import typing
from pathlib import Path

if "typing.io" not in sys.modules:  # pragma: no cover - environment shim
    _m = types.ModuleType("typing.io")
    _m.TextIO, _m.IO, _m.BinaryIO = typing.TextIO, typing.IO, typing.BinaryIO
    sys.modules["typing.io"] = _m

import numpy as np

_here = Path(__file__).resolve().parent
ROOT = _here.parent
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("w1s", _here / "ntc_w1_stats.py")
S = importlib.util.module_from_spec(spec)
sys.modules["w1s"] = S
spec.loader.exec_module(S)

NTC = ROOT / "experiments" / "ntc"

# DEER run from the authors' code at its default configuration (DEER_OFFICIAL.md)
DEER = {"Qwen3-4B": (0.9200, 2042.7, 3538.8), "Qwen3-8B": (0.9300, 1649.0, 2946.2)}

LEVELS = [("1x  (every 256 tok, <=10)", "h2h2_math500_Qwen3-{m}.json"),
          ("2x  (every 128 tok, <=20)", "dens2x_math500_Qwen3-{m}.json"),
          ("4x  (every  64 tok, <=40)", "dens4x_math500_Qwen3-{m}.json")]


def stickiness(traces, bench, m=3):
    """Error stickiness measured between CONSECUTIVE probes (Proposition 2).

    Proposition 2 predicts that rho_w rises as checkpoints are placed closer
    together, because adjacent probes give a wrong answer fewer opportunities to
    change, and that agreement-based halting must therefore degrade with
    density.  This measures it.
    """
    from collections import Counter
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
    """(accuracy, overhead-inclusive online cost) for a fixed policy."""
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

    # Restrict every density level of a model to the items they share, so a
    # sweep run with a smaller --limit still compares like with like.
    common = {}
    for model in ("Qwen3-4B", "Qwen3-8B"):
        sets = []
        for _, pat in LEVELS:
            f = NTC / pat.format(m=model.split("-")[1])
            if f.exists():
                sets.append({t["qid"] for t in json.loads(f.read_text())["traces"]})
        common[model] = set.intersection(*sets) if sets else set()
        if sets and len(common[model]) < max(len(x) for x in sets):
            print(f"[{model}] density levels share {len(common[model])} of "
                  f"{max(len(x) for x in sets)} items; comparing on the shared subset")

    md = ["# Checkpoint density: what does probing more often buy?", "",
          "MATH-500, n=500, greedy decoding, 16k thinking budget, one symbolic grader; "
          "token counts are online cost inclusive of every probe purchased. "
          "`overhead` is the cost of running the controller and declining to halt, "
          "relative to plain generation. DEER is the authors' code at its default "
          "configuration, with overhead measured on the same convention. Where the "
          "density levels cover different numbers of items, all levels are scored on "
          "the items they share.", "",
          "| model | density | probes/item | vanilla acc@tok | overhead | AGREE m=3 acc@tok "
          "| NTC-Select acc@tok | rho_w | P_spur | selected rule |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    missing = []
    for model in ("Qwen3-4B", "Qwen3-8B"):
        for label, pat in LEVELS:
            f = NTC / pat.format(m=model.split("-")[1])
            if not f.exists():
                missing.append(f.name)
                md.append(f"| {model} | {label} | — | *not yet generated* | — | — | — | — | — | — |")
                continue
            d = json.loads(f.read_text())
            traces, bench = d["traces"], d["benchmark"]
            if common[model]:
                traces = [t for t in traces if t["qid"] in common[model]]
            for t in traces:
                t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            has_nll = S.enrich_probes_with_nll(traces)
            fams = dict(S.FAMILIES)
            if has_nll:
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
            rho, qw, psp = stickiness(traces, bench)
            md.append(f"| {model} | {label} | {ppi:.1f} | {van_acc:.3f} @ {van_tok:.0f} "
                      f"| {ovh:+.1f}% | {ag_acc:.3f} @ {ag_tok:.0f} "
                      f"| {nf_acc:.3f} @ {nf_tok:.0f} | {rho:.3f} | {psp:.3f} | {gfam}{gkw} |")
            print(f"{model} {label}: probes/item {ppi:.1f}  vanilla {van_acc:.3f}@{van_tok:.0f}  "
                  f"overhead {ovh:+.1f}%  AGREE {ag_acc:.3f}@{ag_tok:.0f}  "
                  f"NTC-Select {nf_acc:.3f}@{nf_tok:.0f}  rho_w {rho:.3f}  P_spur {psp:.3f} "
                  f"({gfam}{gkw})", flush=True)
        acc, fin, tot = DEER[model]
        md.append(f"| {model} | DEER (authors' code) | — | — | {100*(tot-fin)/fin:+.1f}% | — "
                  f"| {acc:.3f} @ {tot:.0f} | — | — | threshold 0.95 |")
        print(f"{model} DEER official: {acc:.3f}@{tot:.0f}  overhead {100*(tot-fin)/fin:+.1f}%")
    md += ["", "## Reading", "",
           "Two things to read off. First, whether the controller's accuracy rises with "
           "density: if it does not, the gap to DEER is not granularity and the paper must "
           "say so. Second, whether rho_w rises with density: Proposition 2 predicts it "
           "must, because adjacent probes give a wrong answer fewer opportunities to "
           "change, and that agreement-based halting must therefore degrade as probes are "
           "placed closer together."]
    if missing:
        md += ["", f"Missing files ({len(missing)}): " + ", ".join(f"`{m}`" for m in missing) +
               ". Generate with `bash run_queue_density.sh`."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    if missing:
        print(f"missing {len(missing)} density files; run: bash run_queue_density.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
