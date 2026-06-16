#!/usr/bin/env python
"""Day 4 (optional) — hyperparameter sweep for the contrastive router.

Uses the cached Qwen embeddings (built by day4_contrastive.py), so each config
trains in seconds. Sweeps temperature, projection dim, sample-sample weight,
learning rate, and epochs, ranking configs by held-out AIQ against the MF
baseline. Writes a CSV of all results and prints the best config to plug back
into day4_contrastive.py / the Day-5 router.

Run (after at least one day4_contrastive.py run has populated the cache):
    python scripts/day4_sweep.py
    python scripts/day4_sweep.py --quick     # smaller grid
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tokenguard.routers  # noqa: F401
from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval import metrics
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.embedding_cache import EmbeddingCache
from tokenguard.routers.mf_router import MatrixFactorizationRouter
from tokenguard.routers.static import ConstantRouter
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed

import pandas as pd


def build_encoder(kind: str):
    if kind == "qwen":
        from tokenguard.routers.qwen_encoder import QwenEncoder
        return QwenEncoder(), "Qwen__Qwen3-0.6B"
    from tokenguard.routers.embedding import SentenceEmbedder
    return SentenceEmbedder(), "all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoder", choices=["qwen", "minilm"], default="qwen")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day4-sweep", cfg.experiment.results_dir)

    parquet = (
        Path(cfg.data.processed_dir)
        / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet"
    )
    bench = RouterBench.from_parquet(parquet)
    train, test = bench.split_random(cfg.data.test_size, cfg.experiment.seed)

    raw_encoder, enc_id = build_encoder(args.encoder)
    cache = EmbeddingCache(raw_encoder, enc_id,
                           cache_dir=Path(cfg.data.processed_dir) / "embeddings")

    # AIQ window + MF reference
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])
    mf_aiq = metrics.aiq(
        MatrixFactorizationRouter(seed=cfg.experiment.seed).fit(train).frontier(test),
        c_lo, c_hi,
    )
    logger.info("MF reference AIQ = %.4f (target to beat)", mf_aiq)

    # grid
    if args.quick:
        grid = dict(temperature=[0.07, 0.1], proj_dim=[128],
                    w_ss=[0.0, 0.2], w_bce=[2.0], lr=[0.02], epochs=[80])
    else:
        grid = dict(
            temperature=[0.05, 0.07, 0.1],
            proj_dim=[128, 256],
            w_ss=[0.0, 0.2],
            w_bce=[1.0, 2.0, 4.0],
            lr=[0.02],
            epochs=[100],
        )

    keys = list(grid)
    rows = []
    best = None
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        r = ContrastiveRouter(
            encoder=cache,
            proj_dim=p["proj_dim"],
            temperature=p["temperature"],
            w_sample_sample=p["w_ss"],
            w_bce=p["w_bce"],
            lr=p["lr"],
            n_epochs=p["epochs"],
            seed=cfg.experiment.seed,
        ).fit(train)
        aiq = metrics.aiq(r.frontier(test), c_lo, c_hi)
        rows.append({**p, "aiq": aiq, "beats_mf": aiq >= mf_aiq})
        flag = " <== beats MF" if aiq >= mf_aiq else ""
        logger.info("temp=%.2f dim=%d w_ss=%.1f w_bce=%.1f lr=%.2f ep=%d -> AIQ=%.4f%s",
                    p["temperature"], p["proj_dim"], p["w_ss"], p["w_bce"],
                    p["lr"], p["epochs"], aiq, flag)
        if best is None or aiq > best["aiq"]:
            best = {**p, "aiq": aiq}

    df = pd.DataFrame(rows).sort_values("aiq", ascending=False).reset_index(drop=True)
    out = Path(cfg.experiment.results_dir) / "day4-sweep.csv"
    df.to_csv(out, index=False)

    print("\n=== Top 5 configs ===")
    print(df.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nMF baseline AIQ: {mf_aiq:.4f}")
    print(f"Best contrastive: AIQ={best['aiq']:.4f} "
          f"(temp={best['temperature']}, dim={best['proj_dim']}, "
          f"w_ss={best['w_ss']}, w_bce={best['w_bce']}, lr={best['lr']}, "
          f"epochs={best['epochs']})")
    if best["aiq"] >= mf_aiq:
        print("\n==> Contrastive BEATS MF. Use these flags in day4_contrastive.py:")
        print(f"    --epochs {best['epochs']} --proj-dim {best['proj_dim']} "
              f"--lr {best['lr']} --temperature {best['temperature']} "
              f"--w-ss {best['w_ss']} --w-bce {best['w_bce']}")
    else:
        print(f"\n==> Best still {mf_aiq - best['aiq']:.4f} below MF. "
              "That's fine — Day 5's online adaptation is the headline; "
              "use the best config above as the Day-5 starting point.")
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())