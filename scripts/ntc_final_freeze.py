#!/usr/bin/env python
"""FINAL FREEZE: assemble every result artifact into RESULTS_FINAL/ and build
MASTER.md — the single source of truth for paper writing.

Collects: all GENSEEDS_*.md tables, H2H_TABLE.md, DEER_OFFICIAL.md, JOINT.md,
ROUTING_SEEDS.md, RESULTS.md, plus a machine-extracted headline summary
(vanilla / AGREE / NTC-full rows per benchmark) printed to stdout.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

NTC = Path("experiments/ntc")
OUT = NTC / "RESULTS_FINAL"


def extract_rows(md_path: Path, wanted=("vanilla", "AGREE", "NTC-conf",
                                        "NTC-full(e=0.01)", "REFRAIN-SWUCB",
                                        "MUR-mom", "DEER")):
    rows = []
    for line in md_path.read_text().splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and any(cells[0] == w or cells[0] == f"**{w}**"
                             for w in wanted):
                rows.append(f"  {cells[0]:<20} {'  '.join(cells[1:])}")
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# MASTER — TokenGuard/NTC final results inventory\n"]

    # 1) copy every artifact
    artifacts = sorted(NTC.glob("GENSEEDS_*.md")) + [
        NTC / n for n in ("H2H_TABLE.md", "DEER_OFFICIAL.md", "JOINT.md",
                          "ROUTING_SEEDS.md", "RESULTS.md")
        if (NTC / n).exists()]
    for a in artifacts:
        shutil.copy(a, OUT / a.name)

    # 2) headline extraction per table
    md.append("## Headline rows per benchmark (method | acc | cut)\n")
    print("=" * 72)
    print("MASTER SUMMARY — headline rows")
    print("=" * 72)
    for a in artifacts:
        if not a.name.startswith(("GENSEEDS", "H2H", "DEER_OFF", "JOINT",
                                  "ROUTING")):
            continue
        rows = extract_rows(a)
        header = f"\n### {a.name}"
        md.append(header)
        print(header)
        for r in rows:
            md.append(r)
            print(r)

    # 3) figures inventory
    md.append("\n## Figures frozen (paper_figures/)")
    figs = sorted(Path("paper_figures").glob("*.png")) if Path("paper_figures").exists() else []
    for f in figs:
        md.append(f"  {f.name}")
    md.append(f"\nTotal figures: {len(figs)}")

    (OUT / "MASTER.md").write_text("\n".join(md) + "\n")
    print("\n" + "=" * 72)
    print(f"FROZEN: {len(artifacts)} tables + {len(figs)} figures -> {OUT}/")
    print(f"master: {OUT}/MASTER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
