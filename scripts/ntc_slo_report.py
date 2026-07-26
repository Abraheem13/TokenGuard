#!/usr/bin/env python
"""SLO attainment report — the headline evidence that the risk-controlled slow
tier respects its accuracy budget on HELD-OUT data.

Reads every canonical GENSEEDS_*.md table and reports, per setting, each
adaptive policy's held-out accuracy deficit against full thinking (with its
token cut), then the fraction of settings within 1.0 / 2.5 / 5.0 points. This
is the table a reviewer looks for whenever a method advertises a user-facing
accuracy guarantee.

    python scripts/ntc_slo_report.py
"""
from __future__ import annotations

import re
from pathlib import Path

NTC = Path("experiments/ntc")
POLICIES = ["NTC-full(e=0.01)", "NTC-full(e=0.05)", "AGREE", "DEER",
            "NTC-conf", "EAT", "MUR-mom", "NTC-v2", "REFRAIN-SWUCB"]
SKIP = ("_POINT", "_LCB")   # A/B diagnostics, not canonical settings


def parse(md):
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
            out[c[0].strip("*")] = (float(m.group(1)),
                                    float(k.group(1)) if k else 0.0)
    return out


def main() -> int:
    files = [f for f in sorted(NTC.glob("GENSEEDS_*.md"))
             if not any(s in f.stem for s in SKIP)]
    rows = []
    for f in files:
        d = parse(f)
        if "vanilla" not in d:
            continue
        van = d["vanilla"][0]
        rows.append((f.stem.replace("GENSEEDS_", ""), van,
                     {p: (100 * (d[p][0] - van), d[p][1])
                      for p in POLICIES if p in d}))

    short = {p: p.replace("NTC-full(e=", "NTCf").replace(")", "")
             .replace("REFRAIN-SWUCB", "REFR") for p in POLICIES}
    print(f"{'setting':24s} {'van':>6s} " +
          " ".join(f"{short[p]:>14s}" for p in POLICIES))
    md = ["# SLO attainment — held-out accuracy deficit (points) and token cut (%)",
          "", "| setting | vanilla | " + " | ".join(POLICIES) + " |",
          "|" + "---|" * (len(POLICIES) + 2)]
    for tag, van, pol in rows:
        cells = [(f"{pol[p][0]:+.1f} ({pol[p][1]:.0f}%)" if p in pol else "—")
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
        md.append(f"- `{p}` — {line}")
    Path(NTC / "SLO_ATTAINMENT.md").write_text("\n".join(md) + "\n")
    print("\ntable: experiments/ntc/SLO_ATTAINMENT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
