#!/usr/bin/env python
"""Day 3 — learned baselines (matrix factorisation, kNN, cascade, BERT).

Builds the full baseline suite the dissertation compares against:

  * matrix-factorization  — RouteLLM-style low-rank logistic router (2024)
  * knn-embedding         — non-parametric embedding router (2024)
  * cascade               — FrugalGPT / cascade-routing escalation (2023-25)
  * bert-classifier       — the supervisor-rejected reference (built fairly)

All share the same cost-aware decision rule and the same evaluation harness,
so the comparison is apples-to-apples. The narrative is NOT "BERT loses on
quality" (a fairly trained BERT is competitive on static quality); it is that
*every* baseline here is static — trained once, unable to adapt online — which
is precisely the gap the Day-5 nested router fills.

Run:
    python scripts/day3_learned_baselines.py                  # all four
    python scripts/day3_learned_baselines.py --skip-bert      # skip slow BERT
    python scripts/day3_learned_baselines.py --skip-bert --skip-embedding
    python scripts/day3_learned_baselines.py --bert-epochs 1  # quick BERT

Gate (from docs/PROJECT_PLAN.md, Day 3):
  [G1] Every learned baseline run strictly beats random-mix on AIQ.
  [G2] All learned baselines appear on the frontier figure + summary table.
  [G3] (informational) full AIQ ranking of learned baselines is reported, so
       Day-5 improvements are measured against the *strongest* static baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.cascade_router import CascadeRouter
from tokenguard.routers.knn_router import KNNRouter
from tokenguard.routers.mf_router import MatrixFactorizationRouter
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter
from tokenguard.utils.logging import get_logger
from tokenguard.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip kNN + cascade (they need sentence-transformers).")
    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.experiment.seed)
    logger = get_logger("day3", cfg.experiment.results_dir)

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

    runner = EvalRunner(
        train, test,
        results_dir=cfg.experiment.results_dir,
        figures_dir=cfg.experiment.figures_dir,
        tag="day3-learned",
    )

    routers = [
        ConstantRouter("cheapest"),
        ConstantRouter("best"),
        RandomMixRouter(seed=cfg.experiment.seed),
        MatrixFactorizationRouter(seed=cfg.experiment.seed),
    ]
    learned_names = ["matrix-factorization"]

    if not args.skip_embedding:
        routers.append(KNNRouter(k=args.knn_k))
        routers.append(CascadeRouter(k=args.knn_k))
        learned_names += ["knn-embedding", "cascade"]

    if not args.skip_bert:
        from tokenguard.routers.bert_router import BertClassifierRouter
        routers.append(BertClassifierRouter(epochs=args.bert_epochs, seed=cfg.experiment.seed))
        learned_names.append("bert-classifier")

    routers.append(OracleRouter())

    summary = runner.run(routers)
    fig = runner.plot("TokenGuard — static + learned baselines (RouterBench test)")

    print("\n=== Day 3 summary (dissertation Table 2) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    s = summary.set_index("router")
    rnd_aiq = s.loc["random-mix", "aiq"]

    print("\n--- Learned-baseline AIQ ranking (Day-5 target = the top one) ---")
    ranked = (
        summary[summary["router"].isin(learned_names)]
        .sort_values("aiq", ascending=False)[["router", "aiq", "apgr"]]
    )
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nStrongest static baseline: {ranked.iloc[0]['router']} "
          f"(AIQ {ranked.iloc[0]['aiq']:.4f})")
    print("Day-5 nested router must beat THIS on the adaptation/shift experiments.")

    all_beat_random = True
    for name in learned_names:
        ok = s.loc[name, "aiq"] > rnd_aiq + 1e-6
        all_beat_random &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} AIQ > random-mix AIQ")

    if all_beat_random:
        print(f"\nDAY 3 GATE: PASS — figure: {fig}")
        print("Next: Day 4 — modern contrastive router (Qwen3-0.6B encoder).")
        return 0
    print("\nDAY 3 GATE: FAIL — a learned baseline did not beat random-mix.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())