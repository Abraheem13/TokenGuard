#!/usr/bin/env python
"""Sensitivity of the results to how budget-exhausted traces are scored.

In the primary track, full generation is scored as recorded: a trace that
reaches the generation budget before it emits an answer counts as incorrect.
This script repeats the calibrated analysis of ntc_genseed_agg.py under a
second reference, budget-forced full generation, in which such a trace is
answered with its last trial answer (the rule of ntc_h2h_table.py). It reports,
per setting and averaged over generation seeds and calibration splits:

  * the share of traces that reach the budget, and that end without an answer;
  * the accuracy of full generation as recorded and budget-forced;
  * answer agreement, the fusion tier and the selection tier as evaluated in
    SLO_ATTAINMENT.md, against both references;
  * the same three rules when the controller also falls back to the last trial
    answer on an unanswered trace and is calibrated against the budget-forced
    reference ("forced fallback").

The forced fallback costs nothing extra: the last probe has already been paid
for by a rule that did not halt. Settings whose probe files are absent (the
GPQA-Diamond files, before scripts/unpack_gpqa.sh has been run) are skipped.

It also reports, for the head-to-head probe files that are present, the selection
tier of ntc_h2h_table.py (calibration split 0, eps = 0.01) beside full generation
as recorded and budget-forced on the same evaluation items.

Writes experiments/ntc/TRUNCATION_SENSITIVITY.md.

    python scripts/ntc_truncation_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ntc_w1_stats as S

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
SETTINGS = [
    ("GSM8K-4B", ["w1_gsm8k_Qwen3-4B", "w1_gsm8k_Qwen3-4B_s43", "w1_gsm8k_Qwen3-4B_s44"], False),
    ("GSM8K-8B", ["w1_gsm8k_Qwen3-8B", "w1_gsm8k_Qwen3-8B_s43", "w1_gsm8k_Qwen3-8B_s44"], False),
    ("MATH-500-1.7B", ["w1_math500_Qwen3-1.7B", "w1_math500_Qwen3-1.7B_s43",
                       "w1_math500_Qwen3-1.7B_s44"], False),
    ("MATH-500-4B", ["w1_math500_Qwen3-4B", "w1_math500_Qwen3-4B_s43",
                     "w1_math500_Qwen3-4B_s44"], False),
    ("MATH-500-8B", ["w1_math500_Qwen3-8B", "w1_math500_Qwen3-8B_s43",
                     "w1_math500_Qwen3-8B_s44"], False),
    ("MMLU-Pro-4B", [f"w1_mmlupro_Qwen3-4B_s{s}" for s in (42, 43, 44)], False),
    ("MMLU-Pro-8B", [f"w1_mmlupro_Qwen3-8B_s{s}" for s in (42, 43, 44)], False),
    ("GPQA-D-4B", ["w1sh_gpqa_Qwen3-4B", "w1sh_gpqa_Qwen3-4B_s43", "w1sh_gpqa_Qwen3-4B_s44"], False),
    ("GPQA-D-8B", ["w1sh_gpqa_Qwen3-8B", "w1sh_gpqa_Qwen3-8B_s43", "w1sh_gpqa_Qwen3-8B_s44"], False),
    ("AIME-24-4B", [f"w1_aime24_Qwen3-4B_s10{i}" for i in range(8)], True),
    ("AIME-25-4B", [f"w1_aime25_Qwen3-4B_s10{i}" for i in range(8)], True),
]
RULES = ("AGREE", "NTC-v2", "SELECT")


def load(stem):
    """Traces of one probe file, graded as recorded and budget-forced."""
    d = json.loads((NTC / f"{stem}.json").read_text())
    bench = d["benchmark"]
    for t in d["traces"]:
        ans = t.get("natural_answer", "")
        t["natural_correct"] = bool(S.is_correct(ans, t["gold"], bench))
        if t["natural_correct"] or ans.strip() or not t["probes"]:
            t["bf_correct"] = t["natural_correct"]
        else:
            t["bf_correct"] = bool(S.is_correct(t["probes"][-1]["answer"], t["gold"], bench))
    return d["traces"], bench


def forced(traces):
    """The same traces with the forced fallback as their full-generation outcome."""
    return [dict(t, natural_correct=t["bf_correct"]) for t in traces]


def analyse(stems, pool, n_splits, eps, warm_frac=0.4):
    files = [load(s) for s in stems]
    bench = files[0][1]
    res = {k: [] for k in ("trunc", "noans", "van", "bf")}
    for r in RULES:
        for k in ("d", "dbf", "cut", "fd", "fcut"):
            res[f"{r}:{k}"] = []
    saved = S.FAMILIES
    try:
        for fi, (traces, _) in enumerate(files):
            fams = dict(saved)
            if S.enrich_probes_with_nll(traces):
                fams["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
            S.FAMILIES = fams
            ftraces = forced(traces)
            n = len(traces)
            n_warm = int(n * warm_frac)
            res["trunc"].append(np.mean([t["finish_reason"] != "stop" for t in traces]))
            res["noans"].append(np.mean([not t.get("natural_answer", "").strip() for t in traces]))
            for seed in range(n_splits):
                idx = np.random.default_rng(seed).permutation(n)
                w, e = idx[:n_warm], idx[n_warm:]
                warm, fwarm = [traces[i] for i in w], [ftraces[i] for i in w]
                if pool:
                    for fj, (other, _) in enumerate(files):
                        if fj != fi:
                            S.enrich_probes_with_nll(other)
                            warm += [other[i] for i in w if i < len(other)]
                            fwarm += [dict(other[i], natural_correct=other[i]["bf_correct"])
                                      for i in w if i < len(other)]
                ev, fev = [traces[i] for i in e], [ftraces[i] for i in e]
                van = np.mean([t["natural_correct"] for t in ev])
                bf = np.mean([t["bf_correct"] for t in ev])
                vt = np.mean([t["n_total_tokens"] for t in ev])
                res["van"].append(van)
                res["bf"].append(bf)
                for calib, items, tag in ((warm, ev, ""), (fwarm, fev, "f")):
                    picks, (gf, gk) = S.calibrate(calib, bench, eps=eps)
                    chosen = {"AGREE": ("AGREE", picks["AGREE"]),
                              "NTC-v2": ("NTC-v2", picks["NTC-v2"]), "SELECT": (gf, gk)}
                    for r, (fam, kw) in chosen.items():
                        ok, tok = S.per_item(items, bench, fams[fam][0], kw)
                        cut = 100 * (1 - tok.mean() / vt)
                        if tag == "":
                            res[f"{r}:d"].append(100 * (ok.mean() - van))
                            res[f"{r}:dbf"].append(100 * (ok.mean() - bf))
                            res[f"{r}:cut"].append(cut)
                        else:
                            res[f"{r}:fd"].append(100 * (ok.mean() - bf))
                            res[f"{r}:fcut"].append(cut)
    finally:
        S.FAMILIES = saved
    return {k: float(np.mean(v)) for k, v in res.items()}


H2H = ["h2h2_math500_Qwen3-4B", "h2h2_math500_Qwen3-8B", "h2h2_gpqa_Qwen3-4B",
       "h2h2_gpqa_Qwen3-8B", "h2h2_aime24_Qwen3-4B", "h2h2_aime24_Qwen3-8B"]


def h2h_paired(stem, eps=0.01, warm_frac=0.4):
    """Selection tier against both references on its own evaluation items."""
    traces, bench = load(stem)
    saved = S.FAMILIES
    fams = dict(saved)
    if S.enrich_probes_with_nll(traces):
        fams["MUR-mom"] = (S.mur_policy, [{"gamma": g} for g in (0.7, 0.8, 0.9)])
    S.FAMILIES = fams
    try:
        n = len(traces)
        idx = np.random.default_rng(0).permutation(n)
        warm = [traces[i] for i in idx[:int(n * warm_frac)]]
        ev = [traces[i] for i in idx[int(n * warm_frac):]]
        _, (gf, gk) = S.calibrate(warm, bench, eps=eps)
        ok, _ = S.per_item(ev, bench, fams[gf][0], gk)
    finally:
        S.FAMILIES = saved
    return (len(ev), float(ok.mean()), float(np.mean([t["natural_correct"] for t in ev])),
            float(np.mean([t["bf_correct"] for t in ev])), f"{gf}{gk}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--out", default=str(NTC / "TRUNCATION_SENSITIVITY.md"))
    a = ap.parse_args()

    md = ["# Sensitivity to the scoring of budget-exhausted traces", "",
          f"Calibration tolerance eps = {a.eps}; {a.n_splits} calibration/evaluation splits per "
          "generation-seed file. `truncated`: share of traces that reach the generation budget; "
          "`no answer`: share that end without an answer. `full`: accuracy of full generation "
          "as recorded; `full-BF`: budget-forced (an unanswered trace takes its last trial "
          "answer). For each rule, `d` and `d-BF` are its accuracy change in points against "
          "the two references as evaluated in SLO_ATTAINMENT.md, with its token cut; `fb d-BF` "
          "and `fb cut` are the same rule with the forced fallback, calibrated against the "
          "budget-forced reference.", "",
          "| setting | truncated | no answer | full | full-BF | "
          + " | ".join(f"{r} d | {r} d-BF | {r} cut | {r} fb d-BF | {r} fb cut" for r in RULES)
          + " |", "|" + "---|" * (5 + 5 * len(RULES))]
    for name, stems, pool in SETTINGS:
        if not all((NTC / f"{s}.json").exists() for s in stems):
            print(f"{name}: probe files not present; skipped")
            continue
        r = analyse(stems, pool, a.n_splits, a.eps)
        cells = [f"{100 * r['trunc']:.1f}%", f"{100 * r['noans']:.1f}%",
                 f"{r['van']:.3f}", f"{r['bf']:.3f}"]
        for k in RULES:
            cells += [f"{r[k + ':d']:+.1f}", f"{r[k + ':dbf']:+.1f}", f"{r[k + ':cut']:.1f}%",
                      f"{r[k + ':fd']:+.1f}", f"{r[k + ':fcut']:.1f}%"]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
        print(md[-1], flush=True)
    md += ["", "## Head-to-head track: the selection tier on its own evaluation items", "",
           "Calibration split 0, eps = 0.01, as in H2H_TABLE.md. `full` and `full-BF` are "
           "full generation as recorded and budget-forced on the same evaluation items.", "",
           "| probe file | evaluation items | selection tier | full | full-BF | selected rule |",
           "|---|---|---|---|---|---|"]
    for stem in H2H:
        if not (NTC / f"{stem}.json").exists():
            print(f"{stem}: probe file not present; skipped")
            continue
        n, acc, full, bf, pick = h2h_paired(stem)
        md.append(f"| `{stem}` | {n} | {acc:.3f} | {full:.3f} | {bf:.3f} | {pick} |")
        print(md[-1], flush=True)
    Path(a.out).write_text("\n".join(md) + "\n")
    print(f"table: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
