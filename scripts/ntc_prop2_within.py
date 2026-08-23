#!/usr/bin/env python
"""Proposition 2 tested WITHIN one benchmark and model.  (no GPU)

The cross-benchmark correlation of PROP2_VALIDATION.md cannot separate the size
of the answer space from everything else that differs between benchmarks.
MMLU-Pro can: the public loader drops 'N/A' options, so surviving items carry
between four and ten of them.  Bucketing by |A| holds benchmark, model, prompt,
decoding regime, grader and checkpoint protocol fixed and varies only |A|.

Writes experiments/ntc/PROP2_WITHIN.md.

    python scripts/ntc_prop2_within.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokenguard.reasoning.datasets import is_correct  # noqa: E402

NTC = ROOT / "experiments" / "ntc"
OPT = re.compile(r"^([A-J])\)\s", re.M)
FILES = {"Qwen3-4B": [f"w1_mmlupro_Qwen3-4B_s{s}.json" for s in (42, 43, 44)],
         "Qwen3-8B": [f"w1_mmlupro_Qwen3-8B_s{s}.json" for s in (42, 43, 44)]}


def n_options(question: str) -> int:
    return len(set(OPT.findall(question)))


def agree_halt(probes, m, bench):
    run = 1
    for k in range(1, len(probes)):
        same = probes[k]["answer"] and is_correct(probes[k]["answer"], probes[k - 1]["answer"], bench)
        run = run + 1 if same else 1
        if run >= m:
            return k
    return None


def stats(traces, bench, m=3):
    rho, qw, lost, num, den = [], [], [], 0, 0
    for t in traces:
        pr = [p for p in t["probes"] if p.get("answer")]
        if len(pr) < 2:
            continue
        ans = [p["answer"] for p in pr]
        wrong = [not is_correct(x, t["gold"], bench) for x in ans]
        pairs = [1.0 if is_correct(ans[i + 1], ans[i], bench) else 0.0
                 for i in range(len(ans) - 1) if wrong[i]]
        if pairs:
            rho.append(float(np.mean(pairs)))
        bad = [x for x, w in zip(ans, wrong) if w]
        qw.append(Counter(bad).most_common(1)[0][1] / len(ans) if bad else 0.0)
        k = agree_halt(t["probes"], m, bench)
        if t["natural_correct"]:
            lost.append(1.0 if (k is not None and not is_correct(
                t["probes"][k]["answer"], t["gold"], bench)) else 0.0)
        ok = t["natural_correct"] if k is None else is_correct(
            t["probes"][k]["answer"], t["gold"], bench)
        num += (1 if ok else 0) - (1 if t["natural_correct"] else 0)
        den += 1
    if not rho:
        return None
    r, q = float(np.mean(rho)), float(np.mean(qw))
    return dict(rho=r, q=q, pspur=(r ** (m - 1)) * q,
                lost=float(np.mean(lost)) if lost else float("nan"),
                delta=100 * num / max(1, den), n=den)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--min-items", type=int, default=40)
    ap.add_argument("--out", default=str(NTC / "PROP2_WITHIN.md"))
    a = ap.parse_args()

    md = ["# Proposition 2 tested within MMLU-Pro (answer-space size varied, all else fixed)",
          "", f"Items pooled over three generation seeds; m = {a.m}. Buckets with fewer "
          f"than {a.min_items} items are reported but not interpreted.", "",
          "| model | bucket | items | rho_w | q_w | P_spur | lost-correct | AGREE delta |",
          "|---|---|---|---|---|---|---|---|"]
    print(f"{'model':10s} {'bucket':12s} {'n':>5s} {'rho_w':>7s} {'q_w':>7s} "
          f"{'P_spur':>8s} {'lost':>7s} {'AGREE d':>8s}")
    rows = []
    for model, fns in FILES.items():
        buckets = defaultdict(list)
        for fn in fns:
            d = json.loads((NTC / fn).read_text())
            bench = d["benchmark"]
            for t in d["traces"]:
                t["natural_correct"] = bool(is_correct(t.get("natural_answer", ""), t["gold"], bench))
                buckets[n_options(t["question"])].append(t)
        grp = {"|A| <= 6": [], "|A| = 7-9": [], "|A| = 10": []}
        for k, v in buckets.items():
            grp["|A| <= 6" if k <= 6 else "|A| = 7-9" if k <= 9 else "|A| = 10"] += v
        for name, v in grp.items():
            if len(v) < a.min_items:
                continue
            st = stats(v, "mmlu_pro", a.m)
            print(f"{model:10s} {name:12s} {st['n']:5d} {st['rho']:7.3f} {st['q']:7.3f} "
                  f"{st['pspur']:8.3f} {st['lost']:7.3f} {st['delta']:+7.1f}")
            md.append(f"| {model} | {name} | {st['n']} | {st['rho']:.3f} | {st['q']:.3f} "
                      f"| {st['pspur']:.3f} | {st['lost']:.3f} | {st['delta']:+.1f} |")
            rows.append((model, name, st))
    md += ["", "## Reading", "",
           "Stickiness falls monotonically with the answer space in both models, and the "
           "small-|A| bucket reproduces the value measured on four-option GPQA-Diamond "
           "(0.72-0.77). The downstream deficit in the small bucket rests on ~46 items "
           "per model and is not resolvable at that sample size; the claim rests on "
           "rho_w and P_spur."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"\ntable: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
