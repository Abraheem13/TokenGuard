#!/usr/bin/env python
"""Day 1 — download RouterBench, canonicalise, and print the gate report.

Run:  python scripts/day1_download_data.py [--config configs/default.yaml]
                                           [--force]

Gate (from docs/PROJECT_PLAN.md, Day 1):
  [G1] Raw pickle downloaded from the Hugging Face Hub
  [G2] Schema detected (>= 2 models with both perf and cost columns)
  [G3] Canonical parquet written with zero NaNs
  [G4] Dataset statistics table printed and saved (dissertation Table 1)
  [G5] Train/test split sizes verified
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to base config YAML")
    parser.add_argument("--force", action="store_true", help="Re-download / re-process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day1", cfg.experiment.results_dir)

    # [G1-G3] download + canonicalise (cached after first run)
    bench = RouterBench.download(
        repo_id=cfg.data.hf_repo_id,
        filename=cfg.data.hf_filename,
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        force=args.force,
    )

    # [G4] statistics
    logger.info("Rows: %d | Models: %d | Tasks: %d",
                len(bench.df), len(bench.models), len(bench.tasks))
    logger.info("Tasks: %s", ", ".join(bench.tasks))

    summary = bench.summary()
    oracle = bench.oracle_stats()
    print("\n=== Per-model summary (dissertation Table 1) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nOracle upper bound: perf={oracle['oracle_perf']:.4f} "
          f"at cost=${oracle['oracle_cost']:.6f}/query")

    out_csv = Path(cfg.experiment.results_dir) / f"{cfg.experiment.tag}-dataset-summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    logger.info("Saved summary table: %s", out_csv)

    # [G5] split check
    train, test = bench.split_random(cfg.data.test_size, cfg.experiment.seed)
    logger.info("Split: train=%d test=%d (test_size=%.2f, stratified by task)",
                len(train.df), len(test.df), cfg.data.test_size)
    assert set(train.models) == set(test.models), "model sets diverged across split"
    assert len(test.df) > 0 and len(train.df) > 0

    print("\nDAY 1 GATE: PASS — [G1] download  [G2] schema  [G3] parquet  "
          "[G4] stats table  [G5] split verified")
    print("Next: Day 2 — evaluation harness + static baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
