#!/usr/bin/env python
"""Week 1 — NESTOR foundations.

Establishes the two things every later NESTOR result is measured against:
  1. the encoder swap to Qwen3-Embedding-0.6B (modern, BERT-free), re-using the
     Day-4 contrastive pipeline and reporting AIQ, and
  2. the single-timescale LinUCB baseline (the strong online comparison point,
     PILOT/BARP/MixLLM family) on shuffled + shift streams.

Run (offline, cached embeddings):
    python scripts/nestor_week1_baselines.py --encoder qwen3emb --w-bce 6.0 --epochs 120

Gate:
  [G1] contrastive base AIQ >= 0.752 with the chosen encoder
  [G2] single-timescale LinUCB beats static under shift (sanity; expected from Day 5)

See docs/ROADMAP.md and docs/NL_MAPPING.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.runner import EvalRunner
from tokenguard.online.shift import shift_stream, shuffled_stream
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.embedding_cache import EmbeddingCache
from tokenguard.routers.single_timescale import SingleTimescaleLinUCB
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def build_encoder(kind: str):
    """Return (raw_encoder, cache_id). 'qwen3emb' is the Week-1 upgrade."""
    if kind == "qwen3emb":
        from tokenguard.encoders.qwen3_embedding import Qwen3Embedding
        return Qwen3Embedding(), "Qwen__Qwen3-Embedding-0.6B"
    if kind == "qwen":
        from tokenguard.routers.qwen_encoder import QwenEncoder
        return QwenEncoder(), "Qwen__Qwen3-0.6B"
    from tokenguard.routers.embedding import SentenceEmbedder
    return SentenceEmbedder(), "all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", choices=["qwen3emb", "qwen", "minilm"],
                        default="qwen3emb")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--w-ss", type=float, default=0.2)
    parser.add_argument("--w-bce", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-cost", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("nestor-w1", cfg.experiment.results_dir)

    parquet = (Path(cfg.data.processed_dir)
               / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet")
    bench = RouterBench.from_parquet(parquet)
    train, test = bench.split_random(cfg.data.test_size, cfg.experiment.seed)

    raw_encoder, enc_id = build_encoder(args.encoder)
    cache = EmbeddingCache(raw_encoder, enc_id,
                           cache_dir=Path(cfg.data.processed_dir) / "embeddings")

    logger.info("Training contrastive base (encoder=%s)...", args.encoder)
    base = ContrastiveRouter(
        encoder=cache, proj_dim=args.proj_dim, temperature=args.temperature,
        w_sample_sample=args.w_ss, w_bce=args.w_bce, lr=args.lr,
        n_epochs=args.epochs, seed=cfg.experiment.seed,
    )
    base.fit(train)

    # ---- [G1] AIQ of the contrastive base with this encoder ----
    runner = EvalRunner(
        train, test,
        results_dir=cfg.experiment.results_dir,
        figures_dir=getattr(cfg.experiment, "figures_dir",
                            Path(cfg.experiment.results_dir).parent / "figures"),
        tag=f"nestor-w1-{args.encoder}",
    )
    summary = runner.run([base])
    base_row = summary[summary["router"].str.contains("ontrastive", case=False)]
    aiq_val = float(base_row["aiq"].iloc[0]) if len(base_row) else float("nan")
    logger.info("Contrastive AIQ (encoder=%s) = %.4f", args.encoder, aiq_val)

    # ---- [G2] single-timescale LinUCB baseline on streams ----
    print("\n=== Week 1 — single-timescale LinUCB baseline ===")
    for regime, stream in {
        "shuffled": shuffled_stream(test, seed=cfg.experiment.seed),
        "shift":    shift_stream(test, seed=cfg.experiment.seed, granularity="task"),
    }.items():
        r = SingleTimescaleLinUCB(base, lambda_cost=args.lambda_cost,
                                  alpha=args.alpha, seed=cfg.experiment.seed)
        r.warm_start(train)
        out = r.run_stream(stream)
        print(f"  [{regime:8s}] reward={out['mean_reward']:.4f} "
              f"quality={out['mean_quality']:.4f} cost=${out['mean_cost']:.6f}")

    print(f"\nContrastive AIQ (encoder={args.encoder}) = {aiq_val:.4f}")
    print("Week 1 complete. Next: scripts/nestor_week2_midmemory.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())