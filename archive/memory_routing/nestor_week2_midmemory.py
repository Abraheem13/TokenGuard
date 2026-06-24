#!/usr/bin/env python
"""Week 2 — L2 Titans neural memory: the core NESTOR experiment.

Tests the central claim head-to-head against the single-timescale LinUCB
baseline across three stream regimes:

  * shuffled     — i.i.d. control (memory expected to be neutral/slightly down).
  * recurrence   — recurring topic bursts.
  * model-drift  — the per-cluster best model rotates over time (the regime where
                   a recency memory MUST beat a static contextual bandit).

The headline result is the model-drift delta: NESTOR (fast + Titans mid memory)
vs single-timescale LinUCB. Run with the strongest encoder (qwen, AIQ 0.7537):

    python scripts/nestor_week2_midmemory.py --encoder qwen --w-bce 6.0 --epochs 120

Gate:
  [G1] NESTOR beats single-timescale on the model-drift regime.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.online.shift import shuffled_stream
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.embedding_cache import EmbeddingCache
from tokenguard.routers.nestor_router import NestorRouter
from tokenguard.routers.single_timescale import SingleTimescaleLinUCB
from tokenguard.streams.drift import drift_stream, model_drift_stream
from tokenguard.streams.recurrence import recurrence_stream
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def build_encoder(kind: str):
    if kind == "qwen3emb":
        from tokenguard.encoders.qwen3_embedding import Qwen3Embedding
        return Qwen3Embedding(), "Qwen__Qwen3-Embedding-0.6B"
    if kind == "qwen":
        from tokenguard.routers.qwen_encoder import QwenEncoder
        return QwenEncoder(), "Qwen__Qwen3-0.6B"
    from tokenguard.routers.embedding import SentenceEmbedder
    return SentenceEmbedder(), "all-MiniLM-L6-v2"


def _clone(base):
    c = copy.copy(base)
    c.P_, c.E_ = base.P_.copy(), base.E_.copy()
    c.a_, c.b_ = base.a_.copy(), base.b_.copy()
    return c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", choices=["qwen3emb", "qwen", "minilm"], default="qwen")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--w-ss", type=float, default=0.2)
    parser.add_argument("--w-bce", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-cost", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--mid-weight", type=float, default=0.5)
    parser.add_argument("--mem-lr", type=float, default=0.2)
    parser.add_argument("--surprise-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(args.seed)
    logger = get_logger("nestor-w2", cfg.experiment.results_dir)

    parquet = (Path(cfg.data.processed_dir)
               / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet")
    bench = RouterBench.from_parquet(parquet)
    train, test = bench.split_random(cfg.data.test_size, args.seed)

    raw_encoder, enc_id = build_encoder(args.encoder)
    cache = EmbeddingCache(raw_encoder, enc_id,
                           cache_dir=Path(cfg.data.processed_dir) / "embeddings")

    logger.info("Training contrastive base (encoder=%s)...", args.encoder)
    base = ContrastiveRouter(
        encoder=cache, proj_dim=args.proj_dim, temperature=args.temperature,
        w_sample_sample=args.w_ss, w_bce=args.w_bce, lr=args.lr,
        n_epochs=args.epochs, seed=args.seed,
    )
    base.fit(train)

    regimes = {
        "shuffled":    shuffled_stream(test, seed=args.seed),
        "recurrence":  recurrence_stream(test, n_clusters=15, burst_size=40,
                                         revisits=6, seed=args.seed),
        "drift":       drift_stream(test, n_phases=5, seed=args.seed),
        "model-drift": model_drift_stream(test, phase_len=3000, seed=args.seed),
    }

    rows = []
    for regime, stream in regimes.items():
        single = SingleTimescaleLinUCB(_clone(base), lambda_cost=args.lambda_cost,
                                       alpha=args.alpha, seed=args.seed).warm_start(train)
        os_ = single.run_stream(stream)
        nestor = NestorRouter(_clone(base), lambda_cost=args.lambda_cost,
                              alpha=args.alpha, mid_weight=args.mid_weight,
                              lr=args.mem_lr, surprise_scale=args.surprise_scale,
                              seed=args.seed).warm_start(train)
        on_ = nestor.run_stream(stream)
        delta = on_["mean_reward"] - os_["mean_reward"]
        rows.append({"regime": regime,
                     "single_reward": os_["mean_reward"],
                     "nestor_reward": on_["mean_reward"],
                     "delta": delta,
                     "single_quality": os_["mean_quality"],
                     "nestor_quality": on_["mean_quality"]})
        logger.info("[%s] single=%.4f nestor=%.4f delta=%+.4f",
                    regime, os_["mean_reward"], on_["mean_reward"], delta)

    summary = pd.DataFrame(rows)
    out_csv = Path(cfg.experiment.results_dir) / "nestor-w2-midmemory.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== Week 2 — NESTOR (fast+mid) vs single-timescale LinUCB ===")
    print(summary.to_string(index=False,
          columns=["regime", "single_reward", "nestor_reward", "delta"]))

    md = summary[summary.regime == "model-drift"].iloc[0]
    ok = md["delta"] > 0
    print(f"\nModel-drift headline — single={md['single_reward']:.4f}, "
          f"nestor={md['nestor_reward']:.4f}, delta={md['delta']:+.4f}")
    print(f"  [{'PASS' if ok else 'FAIL'}] [G1] NESTOR mid memory beats single-timescale under model drift")
    if ok:
        print("\nWEEK 2 GATE: PASS. Next: scripts/nestor_week3_slow.py")
        return 0
    print("\nWEEK 2 GATE: FAIL — try --mid-weight 0.4 --mem-lr 0.15, or pivot to "
          "the characterization framing (see docs/ROADMAP.md).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())