#!/usr/bin/env python
"""Does the fusion tier's parameter transfer across domains?  (no GPU)

SHIFT_CERTIFICATE.md tests the selection tier's certificate under transfer. The
dissertation's deployment rule recommends the fusion tier where per-domain
calibration is unavailable, which requires its two parameters (m, theta) to be
set without data from the target domain. This script tests that directly, on the
same 12 domains, the same five calibration/evaluation splits and the same
evaluation items as ntc_shift_certificate.py, over the fusion grid of the
candidate library (m in {2, 3}, theta in {0.7, 0.9, 0.95}):

  FIXED          each grid point deployed everywhere with no calibration at all
  TRANSFERRED    the cheapest grid point whose calibration accuracy change is
                 >= -eps on ANOTHER domain's warm-up split, deployed on the target
                 (the way a fixed signal would be tuned once and reused)
  LODO           the cheapest grid point whose calibration accuracy change is
                 >= -eps on EVERY other domain's warm-up split
  LIBRARY LODO   the same leave-one-domain-out rule over the whole candidate
                 library |C| = 19 (no confidence bound), for comparison with the
                 Bonferroni-corrected domain-robust rule of SHIFT_CERTIFICATE.md

Writes experiments/ntc/FUSION_TRANSFER.md.

    python scripts/ntc_fusion_transfer.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shift", _here / "ntc_shift_certificate.py")
SH = importlib.util.module_from_spec(spec)
sys.modules["shift"] = SH
spec.loader.exec_module(SH)
OC, S, NTC, DOMAINS = SH.OC, SH.S, SH.NTC, SH.DOMAINS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out", default=str(NTC / "FUSION_TRANSFER.md"))
    a = ap.parse_args()

    cands = [("NTC-v2", kw) for kw in S.FAMILIES["NTC-v2"][1]]
    lib = [(fam, kw) for fam, (_, grid) in S.FAMILIES.items() for kw in grid]
    fixed = {i: [] for i in range(len(cands))}
    rows = {e: {"transfer": [], "lodo": [], "liblodo": []} for e in a.eps}
    libpk = {e: [] for e in a.eps}
    for s in range(a.splits):
        cache = {}
        for name, fn in DOMAINS:
            d = json.loads((NTC / fn).read_text())
            traces, bench = d["traces"], d["benchmark"]
            for t in traces:
                t["natural_correct"] = bool(
                    S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
            warm, ev = OC.split(traces, seed=s)
            van_tok = float(np.mean([t["n_total_tokens"] for t in ev]))
            van_acc = float(np.mean([t["natural_correct"] for t in ev]))
            dep = []
            for fam, kw in cands:
                ok, tok = S.per_item(ev, bench, S.FAMILIES[fam][0], kw)
                dep.append((100 * (float(ok.mean()) - van_acc),
                            100 * (1 - float(tok.mean()) / van_tok)))
            libst = SH.candidate_stats(warm, bench, lib)
            cache[name] = dict(st=SH.candidate_stats(warm, bench, cands), dep=dep,
                               libst=libst, ev=ev, bench=bench,
                               van_tok=van_tok, van_acc=van_acc)
            del d, traces
        for tgt, _ in DOMAINS:
            T = cache[tgt]
            for i in range(len(cands)):
                fixed[i].append((tgt, *T["dep"][i]))
            srcs = [x for x, _ in DOMAINS if x != tgt]
            for e in a.eps:
                for src in srcs:
                    st = cache[src]["st"]
                    fe = [i for i, c in enumerate(st) if c["md"] >= -e]
                    i = (min(fe, key=lambda i: st[i]["tok"]) if fe
                         else max(range(len(st)), key=lambda i: st[i]["md"]))
                    rows[e]["transfer"].append((tgt, *T["dep"][i]))
                ok_c = [i for i in range(len(cands))
                        if all(cache[x]["st"][i]["md"] >= -e for x in srcs)]
                if ok_c:
                    i = min(ok_c, key=lambda i: np.mean([cache[x]["st"][i]["tok"] for x in srcs]))
                else:
                    i = max(range(len(cands)),
                            key=lambda i: min(cache[x]["st"][i]["md"] for x in srcs))
                rows[e]["lodo"].append((tgt, *T["dep"][i], i))
                okl = [j for j in range(len(lib))
                       if all(cache[x]["libst"][j]["md"] >= -e for x in srcs)]
                j = min(okl, key=lambda j: np.mean([cache[x]["libst"][j]["tok"] for x in srcs]))
                fam, kw = lib[j]
                ok, tok = S.per_item(T["ev"], T["bench"], S.FAMILIES[fam][0], kw)
                rows[e]["liblodo"].append((tgt, 100 * (float(ok.mean()) - T["van_acc"]),
                                           100 * (1 - float(tok.mean()) / T["van_tok"])))
                libpk[e].append(f"{fam}{kw}")
        print(f"  split {s} done", flush=True)

    def summ(v):
        d = np.array([x[1] for x in v]); c = np.array([x[2] for x in v])
        return d.mean(), d.min(), float(np.mean(d >= -1.0)), c.mean(), d, c

    md = ["# Does the fusion tier's parameter transfer across domains?", "",
          f"{len(DOMAINS)} domains x {a.splits} splits, same evaluation items as "
          "SHIFT_CERTIFICATE.md. Deficit in accuracy points against full generation on "
          "the evaluation split; cut in % of full-generation tokens (KV-fork).", "",
          "## Fixed parameters (no calibration)", "",
          "| m | theta | mean deficit | worst cell | within 1 pt | mean cut |",
          "|---|---|---|---|---|---|"]
    for i, (_, kw) in enumerate(cands):
        mu, wo, w1, cu, _, _ = summ(fixed[i])
        md.append(f"| {kw['m']} | {kw['theta']} | {mu:+.2f} | {wo:+.1f} | {w1:.0%} | {cu:.1f}% |")
    for e in a.eps:
        md += ["", f"## eps = {e}", "",
               "| rule | mean deficit | worst cell | within 1 pt | within eps | mean cut |",
               "|---|---|---|---|---|---|"]
        for r, lab in (("transfer", "transferred (tuned on another domain)"),
                       ("lodo", "leave-one-domain-out"),
                       ("liblodo", "library leave-one-domain-out, no bound")):
            mu, wo, w1, cu, d, _ = summ(rows[e][r])
            we = float(np.mean(d >= -100 * e))
            md.append(f"| {lab} | {mu:+.2f} | {wo:+.1f} | {w1:.0%} | {we:.0%} | {cu:.1f}% |")
        from collections import Counter
        pk = Counter(f"m={cands[x[3]][1]['m']},theta={cands[x[3]][1]['theta']}"
                     for x in rows[e]["lodo"])
        md += ["", f"LODO picks: `{dict(pk)}`",
               "", f"library LODO picks: `{dict(Counter(libpk[e]).most_common(5))}`"]
        # per-target worst for the transferred rule
        md += ["", "Per-target worst transferred cell: " + ", ".join(
            f"{t} {min(x[1] for x in rows[e]['transfer'] if x[0] == t):+.1f}"
            for t, _ in DOMAINS)]
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
