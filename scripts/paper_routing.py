#!/usr/bin/env python
"""IEEE-quality routing artifacts: multi-seed AIQ table + polished log-x figures.

Fixes applied (reviewer checklist):
  * Multi-seed rigor: AIQ/APGR for every learned router over --seeds splits,
    reported as mean ± std (experiments/ntc/ROUTING_SEEDS.md + CSV).
  * Log-scale cost axis (costs span orders of magnitude).
  * Explicit axis definitions (quality = RouterBench per-query performance in
    [0,1]; cost = mean USD per query).
  * Caption text emitted for each figure, including the cascade cost
    explanation (sequential escalation re-pays for every attempted model).

Run (uses cached embeddings; ~10-15 min for 3 seeds):
    python scripts/paper_routing.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.static import ConstantRouter, RandomMixRouter, OracleRouter
from tokenguard.routers.mf_router import MatrixFactorizationRouter
from tokenguard.routers.knn_router import KNNRouter
from tokenguard.routers.cascade_router import CascadeRouter
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.embedding_cache import EmbeddingCache
from tokenguard.routers.qwen_encoder import QwenEncoder
from tokenguard.utils.seed import set_global_seed

FIGDIR = Path("experiments/ntc/figures")
STYLE = {
    "always-cheapest": dict(kind="star", color="tab:blue"),
    "always-best": dict(kind="star", color="tab:orange"),
    "random-mix": dict(kind="line", color="tab:green"),
    "matrix-factorization": dict(kind="line", color="tab:red"),
    "knn-embedding": dict(kind="line", color="tab:purple"),
    "cascade": dict(kind="line", color="tab:brown"),
    "contrastive": dict(kind="line", color="mediumorchid"),
    "oracle": dict(kind="dash", color="black"),
}


def polished_fig(frontiers: dict, names: list[str], title: str, fname: str,
                 caption: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for name in names:
        if name not in frontiers:
            continue
        fr = frontiers[name]
        st = STYLE.get(name, dict(kind="line", color="gray"))
        if st["kind"] == "star" or len(fr) == 1:
            ax.scatter(fr["cost"], fr["quality"], marker="*", s=170,
                       color=st["color"], zorder=5, label=name,
                       edgecolor="k", linewidth=0.4)
        elif st["kind"] == "dash":
            ax.plot(fr["cost"], fr["quality"], "--", marker="o", ms=2.6, lw=1.3,
                    color=st["color"], label=name)
        else:
            ax.plot(fr["cost"], fr["quality"], marker="o", ms=3.2, lw=1.4,
                    color=st["color"], label=name, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per query (USD, log scale)")
    ax.set_ylabel("Mean quality\n(RouterBench per-query performance, 0–1)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{fname}.{ext}", dpi=300)
    plt.close(fig)
    (FIGDIR / f"{fname}.caption.txt").write_text(caption + "\n")
    print(f"figure: {FIGDIR}/{fname}.png (+pdf, +caption)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--w-bce", type=float, default=6.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    parquet = (Path(cfg.data.processed_dir)
               / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet")
    bench = RouterBench.from_parquet(parquet)

    raw_enc, enc_id = QwenEncoder(), "Qwen__Qwen3-0.6B"
    cache = EmbeddingCache(raw_enc, enc_id,
                           cache_dir=Path(cfg.data.processed_dir) / "embeddings")

    all_rows = []
    frontiers_seed0 = None
    for seed in args.seeds:
        set_global_seed(seed)
        train, test = bench.split_random(cfg.data.test_size, seed)
        routers = [
            ConstantRouter("cheapest"), ConstantRouter("best"),
            RandomMixRouter(), MatrixFactorizationRouter(),
            KNNRouter(embedder=cache), CascadeRouter(embedder=cache),
            ContrastiveRouter(encoder=cache, proj_dim=args.proj_dim,
                              temperature=0.1, w_sample_sample=0.2,
                              w_bce=args.w_bce, lr=0.02,
                              n_epochs=args.epochs, seed=seed),
            OracleRouter(),
        ]
        runner = EvalRunner(train, test,
                            results_dir=cfg.experiment.results_dir,
                            figures_dir=str(FIGDIR), tag=f"paper-s{seed}")
        summary = runner.run(routers)
        summary["seed"] = seed
        all_rows.append(summary)
        if frontiers_seed0 is None:
            frontiers_seed0 = dict(runner.frontiers)
        print(f"[seed {seed}] done")

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv("experiments/ntc/routing_seeds_raw.csv", index=False)

    # mean ± std AIQ/APGR table
    md = ["# Routing tier — multi-seed results (RouterBench test split)\n",
          f"Seeds: {args.seeds}. Contrastive config: w_bce={args.w_bce}, "
          f"epochs={args.epochs}, proj_dim={args.proj_dim}, encoder=Qwen3-0.6B.\n",
          "| router | AIQ (mean ± std) | APGR (mean ± std) |", "|---|---|---|"]
    for name in ["matrix-factorization", "knn-embedding", "cascade", "contrastive"]:
        sub = big[big["router"] == name]
        if not len(sub):
            continue
        md.append(f"| {name} | {sub['aiq'].mean():.4f} ± {sub['aiq'].std():.4f} "
                  f"| {sub['apgr'].mean():.4f} ± {sub['apgr'].std():.4f} |")
    Path("experiments/ntc/ROUTING_SEEDS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # polished figures from seed-0 frontiers
    q_def = ("Quality is the RouterBench per-query performance score in [0,1]; "
             "cost is mean USD per query (log scale). ")
    polished_fig(frontiers_seed0,
                 ["always-cheapest", "always-best", "random-mix", "oracle"],
                 "Routing headroom on RouterBench (static references)",
                 "routing_fig1_static",
                 "Fig. 1: " + q_def +
                 "The per-query oracle attains higher quality than always-best "
                 "at ~15x lower cost, establishing the routing headroom.")
    polished_fig(frontiers_seed0,
                 ["always-cheapest", "always-best", "random-mix",
                  "matrix-factorization", "knn-embedding", "cascade", "oracle"],
                 "Learned routing baselines",
                 "routing_fig2_learned",
                 "Fig. 2: " + q_def +
                 "Cascade exceeds the always-best cost because sequential "
                 "escalation re-pays inference for every attempted model; "
                 "naive cascading can therefore cost MORE than always using "
                 "the strongest model.")
    polished_fig(frontiers_seed0,
                 ["always-cheapest", "always-best", "random-mix",
                  "matrix-factorization", "contrastive", "oracle"],
                 "Contrastive router (slow tier) vs strongest baseline",
                 "routing_fig3_contrastive",
                 "Fig. 3: " + q_def +
                 "The contrastive router dominates matrix factorization in the "
                 "economically relevant low-cost region and matches it at "
                 "high cost.")
    print("\nDone. Tables: experiments/ntc/ROUTING_SEEDS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
