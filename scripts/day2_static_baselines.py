#!/usr/bin/env python
"""Day 2 — evaluation harness + static baselines.

Run:  python scripts/day2_static_baselines.py [--config configs/default.yaml]

Gate (from docs/PROJECT_PLAN.md, Day 2):
  [G1] Frontier figure with 4 static baselines saved to experiments/figures/
  [G2] Summary table (AIQ / APGR) saved to experiments/results/
  [G3] Sanity ordering holds:
         oracle AIQ >= random-mix AIQ,
         random-mix APGR ~ 0.5,
         always-best quality >= always-cheapest quality.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day2", cfg.experiment.results_dir)

    parquet = (
        Path(cfg.data.processed_dir)
        / f"{Path(cfg.data.hf_filename).stem}.canonical.parquet"
    )
    if not parquet.exists():
        logger.error("Processed data missing: %s — run `make data` (Day 1) first.", parquet)
        return 1

    bench = RouterBench.from_parquet(parquet)
    train, test = bench.split_random(cfg.data.test_size, cfg.experiment.seed)
    logger.info("Loaded %d rows | train=%d test=%d | %d models",
                len(bench.df), len(train.df), len(test.df), len(bench.models))

    runner = EvalRunner(
        train, test,
        results_dir=cfg.experiment.results_dir,
        figures_dir=cfg.experiment.figures_dir,
        tag="day2-static",
    )
    routers = [
        ConstantRouter("cheapest"),
        ConstantRouter("best"),
        RandomMixRouter(seed=cfg.experiment.seed),
        OracleRouter(),
    ]
    summary = runner.run(routers)
    fig_path = runner.plot("TokenGuard — static baselines (RouterBench test split)")

    print("\n=== Day 2 summary (dissertation Table 2, static rows) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # [G3] sanity ordering
    s = summary.set_index("router")
    oracle_aiq = s.loc["oracle", "aiq"]
    random_aiq = s.loc["random-mix", "aiq"]
    random_apgr = s.loc["random-mix", "apgr"]
    q_weak = s.loc["always-cheapest", "max_quality"]
    q_strong = s.loc["always-best", "max_quality"]

    checks = {
        "oracle_aiq >= random_aiq": oracle_aiq >= random_aiq - 1e-9,
        "0.3 <= random_apgr <= 0.7": 0.3 <= random_apgr <= 0.7,
        "always-best q >= always-cheapest q": q_strong >= q_weak - 1e-9,
    }
    for desc, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    if all(checks.values()):
        print(f"\nDAY 2 GATE: PASS — figure: {fig_path}")
        print("Next: Day 3 — learned baselines (matrix factorisation + BERT).")
        return 0
    print("\nDAY 2 GATE: FAIL — inspect the checks above before moving on.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
