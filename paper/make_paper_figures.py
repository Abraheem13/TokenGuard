#!/usr/bin/env python
"""Figures 2-5 of the paper, drawn from the result files of the TokenGuard repository.

    python make_paper_figures.py --repo /path/to/TokenGuard --out .

Figures 2, 3 and 5 read committed result files only (SLO_ATTAINMENT.md,
PROP2_VALIDATION.md, SHIFT_CERTIFICATE.json, TAIL_PRICE.json). Figure 4 replays
the probe streams of four settings with the repository's own curve functions
(scripts/ntc_operating_curves.py), on calibration split 0 of the first
generation seed, as in OPERATING_CURVES.md.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "#333333", "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

C = {"agree": "#b5651d", "conf": "#1f4e79", "ent": "#6a3d9a", "smooth": "#4d4d4d",
     "fuse": "#00695c", "sel": "#2e8b57", "mom": "#a23b3b", "bandit": "#8c6d1f"}


def md_rows(text):
    return [[c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in text.splitlines() if ln.startswith("|") and "---" not in ln]


def fig_collapse(ntc, out):
    rows = md_rows((ntc / "SLO_ATTAINMENT.md").read_text())
    hdr = rows[0]
    body = {r[0]: r for r in rows[1:]}
    sets = [("GSM8K\n4B", "gsm8k_Qwen3-4B"), ("GSM8K\n8B", "gsm8k_Qwen3-8B"),
            ("MATH\n1.7B", "math500_Qwen3-1.7B"), ("MATH\n4B", "math500_Qwen3-4B"),
            ("MATH\n8B", "math500_Qwen3-8B"), ("MMLU-P\n4B", "mmlupro_Qwen3-4B"),
            ("MMLU-P\n8B", "mmlupro_Qwen3-8B"), ("GPQA$^{\\ast}$\n4B", "gpqash_Qwen3-4B"),
            ("GPQA$^{\\ast}$\n8B", "gpqash_Qwen3-8B"), ("AIME-24\n4B", "aime24_avg8"),
            ("AIME-25\n4B", "aime25_avg8")]
    sigs = [("Answer agreement", "AGREE"), ("Confidence threshold", "DEER"),
            ("Entropy stabilisation", "EAT"), ("Uncertainty momentum", "MUR-mom"),
            ("Smoothed confidence", "NTC-conf"), ("Bandit threshold", "REFRAIN-SWUCB")]
    D = np.zeros((len(sigs), len(sets)))
    T = [[None] * len(sets) for _ in sigs]
    for i, (_, m) in enumerate(sigs):
        for j, (_, s) in enumerate(sets):
            cell = body[s][hdr.index(m)]
            d, cut = re.match(r"([-+]?[\d.]+) \((-?\d+)%\)", cell).groups()
            D[i, j] = float(d)
            T[i][j] = (float(d), int(cut))
    fig, ax = plt.subplots(figsize=(7.1, 2.3))
    norm = TwoSlopeNorm(vmin=-24, vcenter=0, vmax=9)
    mesh = ax.pcolormesh(D, cmap="RdYlGn", norm=norm, edgecolors="white", linewidth=0.8)
    for i in range(len(sigs)):
        for j in range(len(sets)):
            d, cut = T[i][j]
            dark = d <= -14 or d >= 7
            ax.text(j + 0.5, i + 0.5, f"{d:+.1f}\n({cut}%)", ha="center", va="center",
                    fontsize=6.3, linespacing=0.95, color="white" if dark else "black")
    ax.set_xticks(np.arange(len(sets)) + 0.5, [s for s, _ in sets], fontsize=6.8)
    ax.set_yticks(np.arange(len(sigs)) + 0.5, [s for s, _ in sigs], fontsize=7.5)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, pad=0.012, aspect=18, ticks=[-20, -15, -10, -5, 0, 5])
    cb.set_label("accuracy change (points)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)
    fig.savefig(out / "fig_collapse.pdf")
    plt.close(fig)


def fig_mechanism(ntc, out):
    text = (ntc / "PROP2_VALIDATION.md").read_text()
    rows = [r for r in md_rows(text) if "/" in r[0] and not r[0].startswith("benchmark")]
    kind = {"gsm8k": "open", "math500": "open", "aime24": "hard", "aime25": "hard",
            "mmlu_pro": "mc10", "gpqa_diamond": "mc4"}
    col = {"open": C["conf"], "hard": C["agree"], "mc10": C["ent"], "mc4": C["mom"]}
    mk = {"open": "o", "hard": "s", "mc10": "^", "mc4": "D"}
    lab = {"open": "open answer", "hard": "open answer, competition level",
           "mc10": "multiple choice, up to 10 options", "mc4": "multiple choice, 4 options"}
    P = [(r[0], kind[r[0].split("/")[0]], float(r[1]), float(r[3]), float(r[4]), float(r[5]))
         for r in rows]
    rs = float(re.search(r"lost-correct risk\) = ([-+][\d.]+)", text).group(1))
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85))
    for name, k, rho, ps, lost, delta in P:
        ax[0].scatter(rho, delta, s=13, marker=mk[k], color=col[k], edgecolor="white",
                      linewidth=0.4, zorder=3)
        ax[1].scatter(ps, lost, s=13, marker=mk[k], color=col[k], edgecolor="white",
                      linewidth=0.4, zorder=3)
    ax[0].axhline(0, color="#888888", lw=0.5)
    ax[0].set_xlabel(r"error stickiness $\rho_w$", fontsize=7)
    ax[0].set_ylabel("agreement, acc. change (pts)", fontsize=7)
    ax[0].set_title("(a)", fontsize=7, loc="left", pad=2)
    ax[0].set_xlim(0.08, 0.82)
    ax[0].set_ylim(-34, 13)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.0013, 0.45)
    ax[1].set_ylim(-0.02, 0.5)
    ax[1].set_xlabel(r"$P_{\mathrm{spur}}$, $m=3$ (log scale)", fontsize=7)
    ax[1].set_ylabel("lost-correct risk", fontsize=7)
    ax[1].set_title("(b)", fontsize=7, loc="left", pad=2)
    ax[1].text(0.50, 0.05, rf"$r_s={rs:+.2f}$, $n={len(P)}$",
               transform=ax[1].transAxes, fontsize=6.5)
    note = {"gsm8k/Qwen3-8B": ("GSM8K-8B", (4, -2)), "aime24/Qwen3-4B": ("AIME-24", (4, -2)),
            "gpqa_diamond/Qwen3-8B": ("GPQA-8B", (-27, -8)),
            "math500/DeepSeek-R1-Distill-Qwen-7B": ("DeepSeek", (4, -3))}
    for name, k, rho, ps, lost, delta in P:
        if name in note:
            ax[0].annotate(note[name][0], (rho, delta), xytext=note[name][1],
                           textcoords="offset points", fontsize=5.5, color="#555555")
    note1 = {"aime24/Qwen3-4B": ("AIME-24", (-27, -2)), "gpqa_diamond/Qwen3-8B": ("GPQA-8B", (-30, -2)),
             "gpqa_diamond/Qwen3-4B": ("GPQA-4B", (-22, -8)), "gsm8k/Qwen3-8B": ("GSM8K", (4, 1))}
    for name, k, rho, ps, lost, delta in P:
        if name in note1:
            ax[1].annotate(note1[name][0], (ps, lost), xytext=note1[name][1],
                           textcoords="offset points", fontsize=5.5, color="#555555")
    for a in ax:
        a.grid(True)
        a.set_axisbelow(True)
        a.tick_params(labelsize=6)
    h = [Line2D([], [], marker=mk[k], ls="", color=col[k], label=lab[k], markersize=3.5) for k in col]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, -0.16),
               columnspacing=1.0, handletextpad=0.2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "fig_mechanism.pdf")
    plt.close(fig)


def fig_curves(repo, out):
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    import ntc_operating_curves as OC
    import ntc_w1_stats as S
    ntc = repo / "experiments" / "ntc"
    panels = [("GSM8K, Qwen3-4B", "w1_gsm8k_Qwen3-4B.json"), ("GSM8K, Qwen3-8B", "w1_gsm8k_Qwen3-8B.json"),
              ("MATH-500, Qwen3-1.7B", "w1_math500_Qwen3-1.7B.json"),
              ("MATH-500, Qwen3-4B", "w1_math500_Qwen3-4B.json")]
    style = {"Confidence (DEER-λ)": ("Confidence", C["conf"], "o", "-", 0.9, 1),
             "Entropy (EAT)": ("Entropy", C["ent"], "s", "-", 0.9, 1),
             "Smoothed confidence": ("Smoothed confidence", C["smooth"], "^", "-", 0.9, 1),
             "Answer agreement": ("Agreement", C["agree"], "D", "-", 3.2, 0.45),
             "NTC-v2 (fusion)": ("Fusion (swept)", C["fuse"], "*", "-", 1.2, 1),
             "NTC-full (selection)": ("NTC-Select", C["sel"], "P", "--", 1.1, 1)}
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.7))
    for ax, (title, fn) in zip(axes, panels):
        d = json.loads((ntc / fn).read_text())
        traces, bench = d["traces"], d["benchmark"]
        for t in traces:
            t["natural_correct"] = bool(S.is_correct(t.get("natural_answer", ""), t["gold"], bench))
        S.enrich_probes_with_nll(traces)
        warm, ev = OC.split(traces)
        vt = float(np.mean([t["n_total_tokens"] for t in ev]))
        va = float(np.mean([t["natural_correct"] for t in ev]))
        curves = {n: OC.curve_deployable(warm, ev, bench, f, g, vt, va)
                  for n, (f, g) in OC.SWEEPS.items()}
        curves["NTC-full (selection)"] = OC.curve_ntc_full(warm, ev, bench, vt, va)
        ax.axhline(va, color="#555555", lw=0.6, ls=":")
        for name, pts in curves.items():
            lab, col, m, ls, lw, al = style[name]
            p = sorted({(round(c, 4), round(a, 4)) for c, a, _ in pts})
            ax.plot([c for c, _ in p], [a for _, a in p], ls=ls, lw=lw, color=col, alpha=al,
                    marker=m, markersize=3.2 if m != "*" else 5, markeredgewidth=0, zorder=3)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("cost / full thinking", fontsize=7.5)
        ax.set_xlim(0.05, 1.09)
        ax.tick_params(labelsize=6.5)
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy", fontsize=7.5)
    h = [Line2D([], [], color=c, marker=m, ls=ls, lw=min(lw, 1.6), alpha=al, label=lab,
                markersize=3.5 if m != "*" else 5.5) for lab, c, m, ls, lw, al in style.values()]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, -0.1),
               columnspacing=1.3, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_curves.pdf")
    plt.close(fig)


def fig_shift(ntc, out):
    sh = json.loads((ntc / "SHIFT_CERTIFICATE.json").read_text())["cells"]["0.05"]
    tp = json.loads((ntc / "TAIL_PRICE.json").read_text())
    S_, D = tp["settings"], tp["data"]
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw={"width_ratios": [1, 1.15]})
    rng = np.random.default_rng(0)
    for x, (key, col) in enumerate([("in", C["sel"]), ("transfer", C["mom"]), ("robust", C["conf"])]):
        d = np.array([c[1] for c in sh[key]])
        ax[0].scatter(x + rng.uniform(-0.17, 0.17, len(d)), d, s=2, color=col, alpha=0.45,
                      linewidth=0, zorder=2)
        ax[0].plot([x - 0.27, x + 0.27], [d.mean(), d.mean()], color="black", lw=1.3, zorder=4)
        ax[0].text(x, 22, f"{d.mean():+.1f}", fontsize=6, va="center", ha="center")
    ax[0].axhline(-5, color="#555555", lw=0.6, ls="--")
    ax[0].text(2.47, -6, r"$-\varepsilon$", fontsize=6, va="top", ha="right", color="#555555")
    ax[0].set_xticks([0, 1, 2], ["in-\ndomain", "trans-\nferred", "domain-\nrobust"], fontsize=6)
    ax[0].set_xlim(-0.5, 2.5)
    ax[0].set_ylabel("held-out acc. change (pts)", fontsize=7)
    ax[0].set_ylim(-55, 27)
    ax[0].tick_params(axis="y", labelsize=6)
    ax[0].set_title("(a)", fontsize=7, loc="left", pad=2)
    ax[0].grid(True, axis="y")
    ax[0].set_axisbelow(True)

    def U(m, k):
        return float(np.mean([D[m][s][1] - k * max(0.0, -D[m][s][0]) for s in S_]))
    ks = np.linspace(0, 25, 101)
    lines = [("NTC-v2", "NTC-Fuse", C["fuse"], "-", 1.3), ("NTC-full(e=0.05)", r"NTC-Select ($\varepsilon{=}0.05$)", C["sel"], "--", 1.3),
             ("AGREE", "Agreement", C["agree"], "-", 0.9), ("EAT", "Entropy", C["ent"], "-.", 0.9),
             ("REFRAIN-SWUCB", "Bandit", C["bandit"], ":", 1.1),
             ("MUR-mom", "Momentum", C["mom"], (0, (5, 1, 1, 1, 1, 1)), 0.9)]
    for key, lab, col, ls, lw in lines:
        ax[1].plot(ks, [U(key, k) for k in ks], color=col, ls=ls, lw=lw, label=lab)
    for k in (5.3, 21.2):
        ax[1].axvline(k, color="#777777", lw=0.5, ls=":")
    ax[1].set_ylim(-15, 60)
    ax[1].set_xlim(0, 25)
    ax[1].set_xlabel(r"exchange rate $\kappa$", fontsize=7)
    ax[1].set_ylabel(r"mean utility $U_\kappa$", fontsize=7)
    ax[1].tick_params(labelsize=6)
    ax[1].set_title("(b)", fontsize=7, loc="left", pad=2)
    ax[1].grid(True)
    ax[1].set_axisbelow(True)
    ax[1].legend(fontsize=5.2, loc="upper right", handlelength=2.2, labelspacing=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_shift.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="path to a checkout of the TokenGuard repository")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    ntc = repo / "experiments" / "ntc"
    out.mkdir(parents=True, exist_ok=True)
    fig_collapse(ntc, out)
    fig_mechanism(ntc, out)
    fig_shift(ntc, out)
    fig_curves(repo, out)
    print("wrote fig_collapse.pdf, fig_mechanism.pdf, fig_shift.pdf, fig_curves.pdf to", out)


if __name__ == "__main__":
    main()
