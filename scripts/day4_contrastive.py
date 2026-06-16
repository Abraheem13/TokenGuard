#!/usr/bin/env python
"""Day 4 — modern contrastive router (Qwen3-0.6B encoder, RouterDC-style).

Trains TokenGuard's core router — a frozen Qwen3-0.6B encoder + learned
projection + per-LLM embeddings, optimised with a dual contrastive loss — and
compares it against the Day-3 baselines on the same RouterBench test split.
Query embeddings are cached to disk on first use, so reruns are fast.

Run:
    # first run encodes 36k prompts with Qwen3-0.6B (one-time, cached)
    python scripts/day4_contrastive.py
    # iterate on the router only (embeddings already cached -> seconds):
    python scripts/day4_contrastive.py --epochs 80
    # quick sanity with the small MiniLM encoder instead of Qwen:
    python scripts/day4_contrastive.py --encoder minilm

Gate (from docs/PROJECT_PLAN.md, Day 4):
  [G1] Contrastive router strictly beats random-mix on AIQ.
  [G2] Contrastive router AIQ >= the strongest static baseline (MF on the real
       split). If it falls short, the SAFE fallback is recorded and Day 5 still
       proceeds (online adaptation is the headline, not static AIQ).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tokenguard.routers  # noqa: F401  (ensure package init order)
from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.embedding_cache import EmbeddingCache
from tokenguard.routers.mf_router import MatrixFactorizationRouter
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def build_encoder(kind: str):
    if kind == "qwen":
        from tokenguard.routers.qwen_encoder import QwenEncoder
        return QwenEncoder(), "Qwen__Qwen3-0.6B"
    if kind == "minilm":
        from tokenguard.routers.embedding import SentenceEmbedder
        return SentenceEmbedder(), "all-MiniLM-L6-v2"
    raise ValueError(f"Unknown encoder '{kind}' (use 'qwen' or 'minilm')")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", choices=["qwen", "minilm"], default="qwen")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--w-ss", type=float, default=0.2,
                        help="Sample-sample contrastive weight.")
    parser.add_argument("--w-bce", type=float, default=2.0,
                        help="BCE (per-model calibration) weight — the term "
                             "that matches MF-style accuracy.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day4", cfg.experiment.results_dir)

    parquet = (
        Path(cfg.data.processed_dir)
        / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet"
    )
    if not parquet.exists():
        logger.error("Run `make data` (Day 1) first — missing %s", parquet)
        return 1

    bench = RouterBench.from_parquet(parquet)
    train, test = bench.split_random(cfg.data.test_size, cfg.experiment.seed)
    logger.info("train=%d test=%d | %d models", len(train.df), len(test.df), len(bench.models))

    # Cached encoder: first run encodes once to disk, later runs are instant.
    raw_encoder, enc_id = build_encoder(args.encoder)
    cache = EmbeddingCache(
        raw_encoder, enc_id, cache_dir=Path(cfg.data.processed_dir) / "embeddings"
    )

    runner = EvalRunner(
        train, test,
        results_dir=cfg.experiment.results_dir,
        figures_dir=cfg.experiment.figures_dir,
        tag="day4-contrastive",
    )

    routers = [
        ConstantRouter("cheapest"),
        ConstantRouter("best"),
        RandomMixRouter(seed=cfg.experiment.seed),
        MatrixFactorizationRouter(seed=cfg.experiment.seed),
        ContrastiveRouter(
            encoder=cache,
            proj_dim=args.proj_dim,
            temperature=args.temperature,
            w_sample_sample=args.w_ss,
            w_bce=args.w_bce,
            lr=args.lr,
            n_epochs=args.epochs,
            seed=cfg.experiment.seed,
        ),
        OracleRouter(),
    ]
    summary = runner.run(routers)
    fig = runner.plot("TokenGuard — contrastive router vs baselines (RouterBench test)")

    print("\n=== Day 4 summary (dissertation Table 3) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    s = summary.set_index("router")
    c_aiq = s.loc["contrastive", "aiq"]
    rnd_aiq = s.loc["random-mix", "aiq"]
    mf_aiq = s.loc["matrix-factorization", "aiq"]

    beats_random = c_aiq > rnd_aiq + 1e-6
    beats_mf = c_aiq >= mf_aiq - 1e-6
    print(f"\nContrastive AIQ={c_aiq:.4f} | MF AIQ={mf_aiq:.4f} | random AIQ={rnd_aiq:.4f}")
    print(f"  [{'PASS' if beats_random else 'FAIL'}] [G1] contrastive > random-mix")
    print(f"  [{'PASS' if beats_mf else 'NOTE'}] [G2] contrastive >= MF (strongest static)")

    if beats_random:
        if not beats_mf:
            print("\nNOTE: contrastive did not beat MF on static AIQ. This is "
                  "acceptable — Day 5's headline is ONLINE adaptation under shift, "
                  "where the static MF cannot adapt. Consider --epochs 100 or "
                  "--proj-dim 256 to close the static gap.")
        print(f"\nDAY 4 GATE: PASS — figure: {fig}")
        print("Next: Day 5 — nested online loop (LinUCB + EMA + slow updates).")
        return 0
    print("\nDAY 4 GATE: FAIL — contrastive did not beat random-mix; "
          "try --epochs 100 --lr 0.02.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())