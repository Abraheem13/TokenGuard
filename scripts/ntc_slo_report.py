#!/usr/bin/env python
"""Accuracy-target attainment per setting and across settings.

Reads the GENSEEDS_*.md tables and reports, for every setting, each method's
accuracy change against full generation on the evaluation items (points) with
its token cut, then the share of settings in which each method stays within
1.0, 2.5 and 5.0 points of full generation, and its worst setting.

Writes experiments/ntc/SLO_ATTAINMENT.md.

    python scripts/ntc_slo_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

NTC = Path(__file__).resolve().parents[1] / "experiments" / "ntc"
POLICIES = ["NTC-full(e=0.01)", "NTC-full(e=0.05)", "AGREE", "DEER",
            "NTC-conf", "EAT", "MUR-mom", "NTC-v2", "REFRAIN-SWUCB"]


def parse(md):
    """method -> (mean accuracy, mean cut) from a GENSEEDS table."""
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
    rows = []
    for f in sorted(NTC.glob("GENSEEDS_*.md")):
        d = parse(f)
        if "vanilla" not in d:
            continue
        van = d["vanilla"][0]
        rows.append((f.stem.replace("GENSEEDS_", ""), van,
                     {p: (100 * (d[p][0] - van), d[p][1]) for p in POLICIES if p in d}))

    short = {p: p.replace("NTC-full(e=", "NTCf").replace(")", "")
             .replace("REFRAIN-SWUCB", "REFR") for p in POLICIES}
    print(f"{'setting':24s} {'van':>6s} " + " ".join(f"{short[p]:>14s}" for p in POLICIES))
    md = ["# Accuracy-target attainment: accuracy change (points) and token cut (%)", "",
          "Accuracy change against full generation on the evaluation items, averaged over "
          "generation seeds and calibration splits, with the token cut in brackets. "
          "`vanilla` is the accuracy of full generation. `n/a`: the method does not apply "
          "to that setting.", "",
          "| setting | vanilla | " + " | ".join(POLICIES) + " |",
          "|" + "---|" * (len(POLICIES) + 2)]
    for tag, van, pol in rows:
        cells = [(f"{pol[p][0]:+.1f} ({pol[p][1]:.0f}%)" if p in pol else "n/a")
                 for p in POLICIES]
        print(f"{tag:24s} {van:6.3f} " + " ".join(f"{c:>14s}" for c in cells))
        md.append(f"| {tag} | {van:.3f} | " + " | ".join(cells) + " |")

    print()
    md += ["", "## Attainment rates (fraction of settings within the bound)", ""]
    for p in POLICIES:
        ds = [pol[p][0] for _, _, pol in rows if p in pol]
        if not ds:
            continue
        r1 = sum(1 for d in ds if d >= -1.0) / len(ds)
        r25 = sum(1 for d in ds if d >= -2.5) / len(ds)
        r5 = sum(1 for d in ds if d >= -5.0) / len(ds)
        line = (f"within 1.0pt: {r1:5.0%}   within 2.5pt: {r25:5.0%}   "
                f"within 5.0pt: {r5:5.0%}   worst: {min(ds):+.1f}pt")
        print(f"{p:18s} {line}")
        md.append(f"- `{p}`: {line}")
    out = NTC / "SLO_ATTAINMENT.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\ntable: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
