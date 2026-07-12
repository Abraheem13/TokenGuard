#!/usr/bin/env python
"""Figure 1 — the collapse map. Parses GENSEEDS_*.md tables directly (no
manual transcription) and renders a signal x benchmark heatmap of
delta-accuracy vs vanilla, annotated with token-cut %.

    python scripts/paper_fig1_collapse_map.py
"""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NTC = Path("experiments/ntc")
COLS = [  # (file tag, pretty column name)
    ("GENSEEDS_gsm8k_Qwen3-4B.md",    "GSM8K\n4B"),
    ("GENSEEDS_gsm8k_Qwen3-8B.md",    "GSM8K\n8B"),
    ("GENSEEDS_math500_Qwen3-1.7B.md","MATH\n1.7B"),
    ("GENSEEDS_math500_Qwen3-4B.md",  "MATH\n4B"),
    ("GENSEEDS_math500_Qwen3-8B.md",  "MATH\n8B"),
    ("GENSEEDS_gpqash_Qwen3-4B.md",   "GPQA*\n4B"),
    ("GENSEEDS_gpqash_Qwen3-8B.md",   "GPQA*\n8B"),
    ("GENSEEDS_aime24_avg8.md",       "AIME24\n4B"),
    ("GENSEEDS_aime25_avg8.md",       "AIME25\n4B"),
]
ROWS = ["AGREE", "DEER", "EAT", "MUR-mom", "NTC-conf",
        "REFRAIN-SWUCB", "NTC-full(e=0.01)"]
PRETTY = {"DEER": "Confidence (DEER-λ)", "EAT": "Entropy (EAT)",
          "MUR-mom": "NLL-momentum (MUR)", "AGREE": "Answer agreement",
          "NTC-conf": "Smoothed confidence", "REFRAIN-SWUCB": "Bandit-λ (REFRAIN)",
          "NTC-full(e=0.01)": "NTC-full (ours, adaptive)"}


def parse(md):
    out = {}
    for ln in md.read_text().splitlines():
        if not ln.startswith("|"):
            continue
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) >= 3:
            m = re.match(r"([\d.]+)", c[1])
            k = re.match(r"(-?[\d.]+)", c[2])
            if m:
                out[c[0].strip("*")] = (float(m.group(1)),
                                        float(k.group(1)) if k else 0.0)
    return out


def main():
    data = {}
    for f, name in COLS:
        p = NTC / f
        if p.exists():
            data[name] = parse(p)
        else:
            print(f"[warn] missing {f} — column skipped")
    cols = list(data.keys())
    D = np.full((len(ROWS), len(cols)), np.nan)
    CUT = np.full_like(D, np.nan)
    for j, cn in enumerate(cols):
        van = data[cn].get("vanilla", (np.nan, 0))[0]
        for i, r in enumerate(ROWS):
            if r in data[cn]:
                D[i, j] = 100 * (data[cn][r][0] - van)
                CUT[i, j] = data[cn][r][1]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    im = ax.imshow(D, cmap="RdYlGn", vmin=-20, vmax=8, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(ROWS)), [PRETTY[r] for r in ROWS], fontsize=8)
    for i in range(len(ROWS)):
        for j in range(len(cols)):
            if not np.isnan(D[i, j]):
                ax.text(j, i, f"{D[i,j]:+.1f}\n({CUT[i,j]:.0f}%)",
                        ha="center", va="center", fontsize=6.4,
                        color="black")
    ax.set_title("No fixed halting signal generalizes — Δ accuracy vs full "
                 "thinking (token cut %)\nGPQA* = option-shuffled; 3-8 "
                 "generation seeds per cell", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Δ accuracy (points)", fontsize=8)
    fig.tight_layout()
    Path("paper_figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"paper_figures/fig1_collapse_map.{ext}", dpi=300)
    print("figure: paper_figures/fig1_collapse_map.png (+pdf)")


if __name__ == "__main__":
    main()
