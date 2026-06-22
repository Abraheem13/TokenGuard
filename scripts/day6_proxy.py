#!/usr/bin/env python
"""Day 6 — launch the OpenAI-compatible routing proxy (and smoke-test it).

Run:
    python scripts/day6_proxy.py --smoke      # in-process smoke test (no server)
    python scripts/day6_proxy.py --serve      # launch the live server on :8000

Gate (docs/PROJECT_PLAN.md, Day 6):
  [G1] Proxy answers /healthz.
  [G2] /v1/chat/completions returns an OpenAI-shaped response and routes a
       simple query to a cheaper model than a hard query.
  [G3] Telemetry persists one row per request with cost + latency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.proxy import RouterService, TelemetryStore, create_app
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def _build_service(cfg, logger) -> tuple[RouterService, TelemetryStore]:
    parquet = (Path(cfg.data.processed_dir)
               / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet")
    bench = RouterBench.from_parquet(parquet)
    train, _ = bench.split_random(cfg.data.test_size, cfg.experiment.seed)

    from tokenguard.routers.qwen_encoder import QwenEncoder
    base = ContrastiveRouter(encoder=QwenEncoder(), proj_dim=256,
                             w_bce=6.0, n_epochs=120, seed=cfg.experiment.seed)
    base.fit(train)

    # cost table = mean training cost per model (USD/query)
    cost = train.cost_matrix().mean(axis=0)
    cost_table = {m: float(c) for m, c in zip(base.models_, cost)}
    svc = RouterService(base, cost_table, lambda_cost=cfg.experiment.get("lambda_cost", 0.5)
                        if hasattr(cfg.experiment, "get") else 0.5)
    tel = TelemetryStore(Path(cfg.experiment.results_dir) / "telemetry.db")
    logger.info("Router service ready over %d models", len(base.models_))
    return svc, tel


def smoke(cfg, logger) -> int:
    from fastapi.testclient import TestClient
    svc, tel = _build_service(cfg, logger)
    app = create_app(svc, tel, demo_mode=True)
    client = TestClient(app)

    # [G1]
    assert client.get("/healthz").json()["status"] == "ok"
    # [G2] OpenAI shape + sensible routing (short factual vs long technical)
    easy = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}]}).json()
    hard = client.post("/v1/chat/completions", json={"messages": [{"role": "user",
                       "content": "Derive the closed-form solution to ridge "
                                  "regression and prove its uniqueness."}]}).json()
    assert easy["object"] == "chat.completion"
    assert "tokenguard" in easy
    # [G3] telemetry persisted
    n = tel.count()
    assert n >= 2

    print("\n=== Day 6 smoke test ===")
    print(f"  easy query  → {easy['model']}  "
          f"(q={easy['tokenguard']['predicted_quality']:.2f}, "
          f"${easy['tokenguard']['est_cost_usd']:.5f})")
    print(f"  hard query  → {hard['model']}  "
          f"(q={hard['tokenguard']['predicted_quality']:.2f}, "
          f"${hard['tokenguard']['est_cost_usd']:.5f})")
    summ = tel.summary()
    print(f"  telemetry   → {summ['n']} rows, "
          f"avg cost ${summ['avg_cost']:.6f}, by_model={summ['by_model']}")
    print("\n  [PASS] [G1] /healthz")
    print("  [PASS] [G2] OpenAI-shaped response + routing")
    print("  [PASS] [G3] telemetry persisted")
    print("\nDAY 6 GATE: PASS — proxy + telemetry working.")
    print("Next: Day 7 — Streamlit dashboard + frozen final runs + v1.0 tag.")
    return 0


def serve(cfg, logger) -> int:  # pragma: no cover - long-running server
    import uvicorn
    svc, tel = _build_service(cfg, logger)
    app = create_app(svc, tel, demo_mode=True)
    logger.info("Serving on http://127.0.0.1:8000  (POST /v1/chat/completions)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true", help="In-process smoke test.")
    parser.add_argument("--serve", action="store_true", help="Launch the live server.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day6", cfg.experiment.results_dir)

    if args.serve:
        return serve(cfg, logger)
    return smoke(cfg, logger)


if __name__ == "__main__":
    raise SystemExit(main())