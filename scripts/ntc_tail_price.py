#!/usr/bin/env python
"""What is the tail worth?  (no GPU)

A deployer choosing one halting policy trades token savings against accuracy.
Score a method on a setting by

    U = cut% - kappa * max(0, -delta_pts)

where kappa is the number of token-saving percentage points the deployer would
forgo to avoid one accuracy point, and report the mean and the worst case over
the canonical settings.  The crossovers say which tier is rational at which
exchange rate.  Writes experiments/ntc/TAIL_PRICE.md.

    python scripts/ntc_tail_price.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
SKIP = ("_POINT", "_LCB", "gpqa16k")
NAME = {"NTC-v2": "NTC-Fuse", "NTC-full(e=0.01)": "NTC-Select (eps=0.01)",
        "NTC-full(e=0.05)": "NTC-Select (eps=0.05)", "AGREE": "Answer agreement",
        "DEER": "Confidence threshold", "NTC-conf": "Smoothed confidence",
        "EAT": "Entropy (EAT)", "MUR-mom": "Uncertainty momentum",
        "REFRAIN-SWUCB": "Bandit threshold"}


def parse(md: Path):
    out = {}
    for ln in md.read_text().splitlines():
        if not ln.startswith("|"):
            continue
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) < 3:
            continue
        m = re.match(r"([\d.]+)", c[1])
        k = re.match(r"(-?[\d.]+)", c[2])
        if m:
            out[c[0].strip("*")] = (float(m.group(1)), float(k.group(1)) if k else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(NTC / "TAIL_PRICE.md"))
    a = ap.parse_args()

    D = {}
    for f in sorted(NTC.glob("GENSEEDS_*.md")):
        if any(s in f.stem for s in SKIP):
            continue
        d = parse(f)
        if "vanilla" not in d:
            continue
        van = d["vanilla"][0]
        D[f.stem.replace("GENSEEDS_", "")] = {
            k: (100 * (v[0] - van), v[1]) for k, v in d.items() if k in NAME}
    S = sorted(D)
    M = [m for m in NAME if all(m in D[s] for s in S)]

    def U(m, kap, worst=False):
        v = [D[s][m][1] - kap * max(0.0, -D[s][m][0]) for s in S]
        return min(v) if worst else float(np.mean(v))

    def crossover(m, ref, worst):
        lo, hi = 0.0, 60.0
        if U(m, lo, worst) <= U(ref, lo, worst) or U(m, hi, worst) > U(ref, hi, worst):
            return None
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if U(m, mid, worst) > U(ref, mid, worst) else (lo, mid)
        return (lo + hi) / 2

    md = [f"# Pricing the tail ({len(S)} canonical settings)", "",
          "Utility of deploying method M on setting s is `cut - kappa * max(0, -delta)`, "
          "kappa in token-saving points forgone per accuracy point.", "",
          "| method | mean cut | mean delta | worst delta |", "|---|---|---|---|"]
    print(f"{'method':24s} {'mean cut':>9s} {'mean dfc':>9s} {'worst':>8s}")
    for m in M:
        cuts = [D[s][m][1] for s in S]; dfc = [D[s][m][0] for s in S]
        print(f"{NAME[m]:24s} {np.mean(cuts):8.1f}% {np.mean(dfc):+9.2f} {min(dfc):+8.2f}")
        md.append(f"| {NAME[m]} | {np.mean(cuts):.1f}% | {np.mean(dfc):+.2f} | {min(dfc):+.2f} |")
    md += ["", "## Best method as a function of kappa", "",
           "| kappa | best by mean utility | best by worst-case utility |", "|---|---|---|"]
    print(f"\n{'kappa':>6s} | {'best (mean)':28s} | best (worst case)")
    for kap in [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30]:
        bm = max(M, key=lambda m: U(m, kap)); bw = max(M, key=lambda m: U(m, kap, True))
        print(f"{kap:6.1f} | {NAME[bm]:28s} | {NAME[bw]}")
        md.append(f"| {kap:g} | {NAME[bm]} ({U(bm,kap):.1f}) | {NAME[bw]} ({U(bw,kap,True):.1f}) |")
    md += ["", "## Crossovers against NTC-Fuse", ""]
    for worst in (False, True):
        lbl = "worst-case" if worst else "mean"
        for m in M:
            if m == "NTC-v2":
                continue
            c = crossover(m, "NTC-v2", worst)
            if c is not None:
                line = f"- {lbl}: {NAME[m]} loses to NTC-Fuse beyond kappa = {c:.2f}"
                print(line); md.append(line)
    Path(a.out).write_text("\n".join(md) + "\n")
    side = Path(a.out).with_suffix(".json")
    side.write_text(json.dumps({"settings": S, "data": {m: {s: D[s][m] for s in S} for m in M},
                                "names": NAME}, indent=1))
    print(f"\ntable: {a.out}\nraw  : {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
