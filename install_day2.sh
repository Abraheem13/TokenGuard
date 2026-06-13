#!/usr/bin/env bash
# =============================================================================
# TokenGuard — Day 2 file installer
# Run from the repo root (the folder that contains src/, scripts/, tests/):
#     bash install_day2.sh
# Writes all six Day-2 source files + updates two package __init__.py files.
# Safe to re-run (overwrites in place). Does NOT touch Day-1 files.
# =============================================================================
set -euo pipefail

# Guard: must be in the repo root.
if [[ ! -d src/tokenguard/eval || ! -d scripts || ! -d tests ]]; then
  echo "ERROR: run this from the tokenguard repo root (where src/, scripts/, tests/ live)." >&2
  exit 1
fi
echo "==> Writing Day-2 files..."

mkdir -p "$(dirname src/tokenguard/eval/__init__.py)"
cat > src/tokenguard/eval/__init__.py << 'TG_EOF'
"""Evaluation: cost-quality metrics and the experiment runner (Day 2)."""

from tokenguard.eval.metrics import (
    DEFAULT_LAMBDAS,
    aiq,
    apgr,
    evaluate_choices,
    pareto_front,
    quality_at_cost,
    summarise_router,
)
from tokenguard.eval.runner import EvalRunner

__all__ = [
    "DEFAULT_LAMBDAS", "aiq", "apgr", "evaluate_choices", "pareto_front",
    "quality_at_cost", "summarise_router", "EvalRunner",
]
TG_EOF
echo "   wrote src/tokenguard/eval/__init__.py"

mkdir -p "$(dirname src/tokenguard/eval/metrics.py)"
cat > src/tokenguard/eval/metrics.py << 'TG_EOF'
"""Cost-quality evaluation metrics for LLM routing.

This module is the measurement core of the dissertation. Conventions:

* **Quality** = mean RouterBench performance of the chosen model per query
  (values in [0, 1]).
* **Cost** = mean *true* per-query cost (USD) of the chosen model. Routers
  never see true test-time cost when *deciding* — they use per-model cost
  estimates learned on the training split (see ``routers.base``) — but
  accounting always uses true cost. This separation prevents label leakage.
* **Operating point** = one (cost, quality) pair produced by a router at one
  trade-off setting.
* **Frontier** = the Pareto-optimal set of a router's operating points,
  traced by sweeping the trade-off parameter λ in
  ``score = predicted_quality − λ · normalised_cost``.
* **AIQ** (Average Improvement in Quality, after RouterBench, arXiv:2403.12031)
  = the mean quality of the *linearly interpolated* frontier over a shared
  cost window [c_lo, c_hi]. Linear interpolation is justified because any two
  operating points can be mixed stochastically, achieving the line segment
  between them.
* **APGR** (Average Performance Gap Recovered, after RouteLLM,
  arXiv:2406.18665) = the fraction of the quality gap between a weak and a
  strong reference endpoint that the router recovers, averaged over matched
  cost budgets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default λ sweep: 0 (quality-only) plus a log grid spanning five decades.
# Costs are normalised by their mean inside the decision rule, so this grid
# is scale-free and works for any model pool.
DEFAULT_LAMBDAS: np.ndarray = np.concatenate(
    [[0.0], np.geomspace(1e-3, 1e3, num=49)]
)


# --------------------------------------------------------------------------- #
# Point evaluation                                                            #
# --------------------------------------------------------------------------- #
def evaluate_choices(
    perf: np.ndarray, cost: np.ndarray, choices: np.ndarray
) -> tuple[float, float]:
    """Mean (quality, cost) of per-sample model ``choices``.

    Parameters
    ----------
    perf, cost : (n_samples, n_models) ground-truth matrices.
    choices    : (n_samples,) integer model indices.
    """
    rows = np.arange(len(choices))
    return float(perf[rows, choices].mean()), float(cost[rows, choices].mean())


# --------------------------------------------------------------------------- #
# Pareto frontier                                                             #
# --------------------------------------------------------------------------- #
def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated subset of operating points, sorted by ascending cost.

    A point dominates another if it has (cost ≤, quality ≥) with at least one
    strict inequality. Input/output columns: ``cost``, ``quality`` (extra
    columns are preserved).
    """
    pts = points.sort_values(["cost", "quality"], ascending=[True, False])
    keep, best_q = [], -np.inf
    for idx, row in pts.iterrows():
        if row["quality"] > best_q + 1e-12:
            keep.append(idx)
            best_q = row["quality"]
    return pts.loc[keep].sort_values("cost").reset_index(drop=True)


def quality_at_cost(frontier: pd.DataFrame, budget: float) -> float:
    """Quality achievable at a cost ``budget`` by linear interpolation.

    Below the cheapest point the router cannot operate: returns the cheapest
    point's quality only at/above its cost, else −inf is avoided by clamping
    to the cheapest quality (documented choice: a router charged less than its
    minimum cost simply isn't run; for AIQ windows we always start at a
    feasible cost). Above the most expensive point, quality saturates.
    """
    c = frontier["cost"].to_numpy()
    q = frontier["quality"].to_numpy()
    if len(c) == 1:
        return float(q[0])
    return float(np.interp(budget, c, q))


# --------------------------------------------------------------------------- #
# Aggregate metrics                                                           #
# --------------------------------------------------------------------------- #
def aiq(frontier: pd.DataFrame, c_lo: float, c_hi: float, num: int = 256) -> float:
    """Average Interpolated Quality over the shared window [c_lo, c_hi]."""
    if c_hi <= c_lo:
        raise ValueError(f"Invalid AIQ window: [{c_lo}, {c_hi}]")
    grid = np.linspace(c_lo, c_hi, num)
    vals = np.array([quality_at_cost(frontier, b) for b in grid])
    return float(vals.mean())


def apgr(
    frontier: pd.DataFrame,
    weak_point: tuple[float, float],
    strong_point: tuple[float, float],
    num: int = 64,
) -> float:
    """Average Performance Gap Recovered vs. weak/strong reference endpoints.

    ``weak_point``/``strong_point`` are (cost, quality) of the always-weak and
    always-strong baselines. At each budget in [weak_cost, strong_cost], the
    reference is the *random-mix line* between the endpoints; APGR is the mean
    of (router − weak) / (strong − weak) quality, so the random mix scores
    ~the mean mixing ratio and the strong endpoint scores 1.0 at full budget.
    """
    (cw, qw), (cs, qs) = weak_point, strong_point
    if qs <= qw:
        raise ValueError("strong endpoint must outperform weak endpoint")
    grid = np.linspace(cw, cs, num)
    rec = [(quality_at_cost(frontier, b) - qw) / (qs - qw) for b in grid]
    return float(np.mean(rec))


def summarise_router(
    name: str,
    frontier: pd.DataFrame,
    c_lo: float,
    c_hi: float,
    weak_point: tuple[float, float],
    strong_point: tuple[float, float],
) -> dict:
    """One results-table row per router (dissertation Table 2)."""
    return {
        "router": name,
        "n_points": len(frontier),
        "min_cost": frontier["cost"].min(),
        "max_quality": frontier["quality"].max(),
        "aiq": aiq(frontier, c_lo, c_hi),
        "apgr": apgr(frontier, weak_point, strong_point),
    }
TG_EOF
echo "   wrote src/tokenguard/eval/metrics.py"

mkdir -p "$(dirname src/tokenguard/eval/runner.py)"
cat > src/tokenguard/eval/runner.py << 'TG_EOF'
"""Evaluation runner: fit routers, trace frontiers, save tables and the
frontier figure. Every later day re-uses this runner unchanged, so all
dissertation figures share one code path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (cluster, CI)
import matplotlib.pyplot as plt
import pandas as pd

from tokenguard.data.routerbench import RouterBench
from tokenguard.eval import metrics
from tokenguard.routers.base import Router
from tokenguard.routers.static import ConstantRouter
from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.eval")


class EvalRunner:
    """Fit on ``train``, evaluate frontiers on ``test``, persist artifacts."""

    def __init__(
        self,
        train: RouterBench,
        test: RouterBench,
        results_dir: str | Path,
        figures_dir: str | Path,
        tag: str,
    ):
        self.train, self.test = train, test
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.tag = tag
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.frontiers: dict[str, pd.DataFrame] = {}

        # Shared references for AIQ window and APGR endpoints: the constant
        # weak/strong routers evaluated on THIS test split.
        weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
        strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
        self.weak_point = (float(weak["cost"]), float(weak["quality"]))
        self.strong_point = (float(strong["cost"]), float(strong["quality"]))
        self.c_lo, self.c_hi = self.weak_point[0], self.strong_point[0]
        logger.info(
            "References — weak: q=%.4f @ $%.6f | strong: q=%.4f @ $%.6f",
            self.weak_point[1], self.weak_point[0],
            self.strong_point[1], self.strong_point[0],
        )

    # ------------------------------------------------------------------ #
    def run(self, routers: list[Router]) -> pd.DataFrame:
        """Fit + frontier every router; return the summary table."""
        rows = []
        for router in routers:
            router.fit(self.train)
            frontier = router.frontier(self.test)
            self.frontiers[router.name] = frontier
            frontier.to_csv(
                self.results_dir / f"{self.tag}-frontier-{router.name}.csv",
                index=False,
            )
            if len(frontier) > 1:  # frontier-capable router
                rows.append(
                    metrics.summarise_router(
                        router.name, frontier, self.c_lo, self.c_hi,
                        self.weak_point, self.strong_point,
                    )
                )
            else:  # single-point router: report the point, AIQ/APGR n/a
                pt = frontier.iloc[0]
                rows.append(
                    {
                        "router": router.name,
                        "n_points": 1,
                        "min_cost": pt["cost"],
                        "max_quality": pt["quality"],
                        "aiq": float("nan"),
                        "apgr": float("nan"),
                    }
                )
            logger.info("Evaluated %-18s (%d frontier points)", router.name, len(frontier))

        summary = pd.DataFrame(rows).sort_values("aiq", ascending=False, na_position="last")
        summary.to_csv(self.results_dir / f"{self.tag}-summary.csv", index=False)
        return summary.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def plot(self, title: str) -> Path:
        """Cost-quality frontier figure (dissertation Figure 2)."""
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
        palette = plt.cm.tab10.colors

        for i, (name, frontier) in enumerate(self.frontiers.items()):
            color = palette[i % len(palette)]
            if len(frontier) == 1:
                ax.scatter(
                    frontier["cost"], frontier["quality"],
                    marker="*", s=180, color=color, zorder=5, label=name,
                )
            else:
                style = dict(linestyle="--", linewidth=1.4) if name == "oracle" else {}
                ax.plot(
                    frontier["cost"], frontier["quality"],
                    marker="o", markersize=3.5, color=color, label=name, **style,
                )

        ax.set_xlabel("Mean cost per query (USD)")
        ax.set_ylabel("Mean quality (RouterBench performance)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        out = self.figures_dir / f"{self.tag}-frontier.png"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved frontier figure: %s", out)
        return out
TG_EOF
echo "   wrote src/tokenguard/eval/runner.py"

mkdir -p "$(dirname src/tokenguard/routers/__init__.py)"
cat > src/tokenguard/routers/__init__.py << 'TG_EOF'
"""Routers: base interface (Day 2), static baselines (Day 2), learned
baselines (Day 3), contrastive router (Day 4), nested online router (Day 5)."""

from tokenguard.routers.base import Router
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter

__all__ = ["Router", "ConstantRouter", "OracleRouter", "RandomMixRouter"]
TG_EOF
echo "   wrote src/tokenguard/routers/__init__.py"

mkdir -p "$(dirname src/tokenguard/routers/base.py)"
cat > src/tokenguard/routers/base.py << 'TG_EOF'
"""Router interface.

Every router in the project — static baselines (Day 2), learned baselines
(Day 3), the contrastive router (Day 4), and the nested online router
(Day 5) — implements this interface, so the evaluation runner and the proxy
treat them identically.

Decision rule (shared by default):

    choice(x) = argmax_j  q̂_j(x) − λ · ĉ_j / mean(ĉ)

where ``q̂_j(x)`` is the router's predicted quality of model *j* on query *x*
and ``ĉ_j`` is the per-model **cost estimate learned on the training split**
(mean observed cost). Normalising by ``mean(ĉ)`` makes λ scale-free, so one
λ grid serves any model pool.

Leakage rule: routers must never read test-time ``perf::*`` or per-sample
test ``cost::*`` when deciding. The single sanctioned exception is
``OracleRouter`` (an explicit upper bound, see ``routers.static``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.metrics import DEFAULT_LAMBDAS, evaluate_choices, pareto_front


class Router(ABC):
    """Abstract router. Subclasses set ``name`` and implement the API below."""

    name: str = "base"

    def __init__(self) -> None:
        self.models_: tuple[str, ...] | None = None
        self.cost_estimates_: np.ndarray | None = None  # (n_models,)

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def fit(self, train: RouterBench) -> "Router":
        """Learn cost estimates (and, in subclasses, quality predictors)."""
        self.models_ = train.models
        self.cost_estimates_ = train.cost_matrix().mean(axis=0)
        return self

    def _check_fitted(self, bench: RouterBench) -> None:
        if self.models_ is None or self.cost_estimates_ is None:
            raise RuntimeError(f"{self.name}: call fit() before routing")
        if bench.models != self.models_:
            raise ValueError(
                f"{self.name}: model set mismatch.\n"
                f"fit:   {self.models_}\nroute: {bench.models}"
            )

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        """Predicted quality matrix, shape (n_samples, n_models)."""

    # ------------------------------------------------------------------ #
    # Decision + frontier                                                #
    # ------------------------------------------------------------------ #
    def route(self, bench: RouterBench, lambda_cost: float) -> np.ndarray:
        """Per-sample model choices at one trade-off setting λ."""
        self._check_fitted(bench)
        q_hat = self.predict_quality(bench)
        c_norm = self.cost_estimates_ / max(self.cost_estimates_.mean(), 1e-12)
        scores = q_hat - lambda_cost * c_norm[None, :]
        return scores.argmax(axis=1)

    def frontier(
        self, bench: RouterBench, lambdas: np.ndarray = DEFAULT_LAMBDAS
    ) -> pd.DataFrame:
        """Pareto frontier of (cost, quality) operating points over a λ sweep.

        Accounting uses ground truth; decisions never do (see module note).
        Subclasses with a different natural sweep (e.g. the random-mix
        baseline) override this method.
        """
        perf, cost = bench.perf_matrix(), bench.cost_matrix()
        rows = []
        for lam in lambdas:
            choices = self.route(bench, lam)
            quality, mean_cost = evaluate_choices(perf, cost, choices)
            rows.append({"lambda": lam, "cost": mean_cost, "quality": quality})
        return pareto_front(pd.DataFrame(rows))
TG_EOF
echo "   wrote src/tokenguard/routers/base.py"

mkdir -p "$(dirname src/tokenguard/routers/static.py)"
cat > src/tokenguard/routers/static.py << 'TG_EOF'
"""Static baseline routers (Day 2).

These four baselines anchor every frontier plot in the dissertation:

* ``ConstantRouter("cheapest")``  — *always-small*: one operating point.
* ``ConstantRouter("best")``      — *always-large*: one operating point.
* ``RandomMixRouter``             — routes each query to the strong model
  with probability *p*, else the weak model; sweeping p ∈ [0, 1] traces the
  straight line between the two endpoints. This is the "random" reference
  used by RouteLLM-style APGR: any learned router must bend *above* this line.
* ``OracleRouter``                — per-sample argmax over **true**
  performance/cost (the only router allowed to peek at labels). Its λ-sweep
  is the unbeatable upper frontier; reported as the ceiling, never as a
  competitor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.metrics import evaluate_choices, pareto_front
from tokenguard.routers.base import Router


# --------------------------------------------------------------------------- #
# Constant (single-model) routers                                             #
# --------------------------------------------------------------------------- #
class ConstantRouter(Router):
    """Always route to one model, selected on the *training* split.

    strategy:
      * ``"cheapest"`` — lowest mean training cost (always-small),
      * ``"best"``     — highest mean training performance (always-large),
      * any model name — pin that model explicitly.
    """

    def __init__(self, strategy: str):
        super().__init__()
        self.strategy = strategy
        self.target_idx_: int | None = None
        self.name = f"always-{strategy}"

    def fit(self, train: RouterBench) -> "ConstantRouter":
        super().fit(train)
        if self.strategy == "cheapest":
            self.target_idx_ = int(np.argmin(self.cost_estimates_))
        elif self.strategy == "best":
            mean_perf = train.perf_matrix().mean(axis=0)
            self.target_idx_ = int(np.argmax(mean_perf))
        elif self.strategy in train.models:
            self.target_idx_ = train.models.index(self.strategy)
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Use 'cheapest', 'best', or one of {train.models}"
            )
        self.name = f"always-{train.models[self.target_idx_]}" if self.strategy not in (
            "cheapest", "best"
        ) else f"always-{self.strategy}"
        return self

    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        """One-hot quality: the pinned model always wins the argmax."""
        q = np.zeros((len(bench.df), len(self.models_)))
        q[:, self.target_idx_] = 1.0
        return q

    def frontier(self, bench: RouterBench, lambdas=None) -> pd.DataFrame:
        """A constant router has exactly one operating point."""
        self._check_fitted(bench)
        perf, cost = bench.perf_matrix(), bench.cost_matrix()
        choices = np.full(len(bench.df), self.target_idx_, dtype=int)
        quality, mean_cost = evaluate_choices(perf, cost, choices)
        return pd.DataFrame([{"lambda": np.nan, "cost": mean_cost, "quality": quality}])


# --------------------------------------------------------------------------- #
# Random-mix baseline                                                         #
# --------------------------------------------------------------------------- #
class RandomMixRouter(Router):
    """Stochastic mix of the weak (cheapest) and strong (best) models.

    The natural sweep parameter is the mixing probability p, not λ, so this
    class overrides ``frontier``. The expected frontier is the straight
    segment between the two endpoints; with finite samples it wobbles by
    sampling noise, which the Pareto filter cleans up.
    """

    name = "random-mix"

    def __init__(self, seed: int = 42, num_p: int = 21):
        super().__init__()
        self.seed = seed
        self.num_p = num_p
        self.weak_idx_: int | None = None
        self.strong_idx_: int | None = None

    def fit(self, train: RouterBench) -> "RandomMixRouter":
        super().fit(train)
        self.weak_idx_ = int(np.argmin(self.cost_estimates_))
        self.strong_idx_ = int(np.argmax(train.perf_matrix().mean(axis=0)))
        return self

    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        """Random scores → λ-free uniform choice between weak and strong."""
        rng = np.random.default_rng(self.seed)
        q = np.zeros((len(bench.df), len(self.models_)))
        pick_strong = rng.random(len(bench.df)) < 0.5
        q[pick_strong, self.strong_idx_] = 1.0
        q[~pick_strong, self.weak_idx_] = 1.0
        return q

    def frontier(self, bench: RouterBench, lambdas=None) -> pd.DataFrame:
        self._check_fitted(bench)
        perf, cost = bench.perf_matrix(), bench.cost_matrix()
        rng = np.random.default_rng(self.seed)
        rows = []
        for p in np.linspace(0.0, 1.0, self.num_p):
            choices = np.where(
                rng.random(len(bench.df)) < p, self.strong_idx_, self.weak_idx_
            )
            quality, mean_cost = evaluate_choices(perf, cost, choices)
            rows.append({"lambda": p, "cost": mean_cost, "quality": quality})
        return pareto_front(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# Oracle                                                                      #
# --------------------------------------------------------------------------- #
class OracleRouter(Router):
    """Per-sample optimum using ground-truth labels (explicit upper bound).

    LEAKAGE NOTE: this router reads test-time performance and per-sample
    cost by design. It exists to plot the achievable ceiling; it is never a
    competitor and is excluded from win/loss claims.
    """

    name = "oracle"

    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        return bench.perf_matrix()

    def route(self, bench: RouterBench, lambda_cost: float) -> np.ndarray:
        """Argmax over true perf − λ · true per-sample normalised cost."""
        self._check_fitted(bench)
        perf, cost = bench.perf_matrix(), bench.cost_matrix()
        c_norm = cost / max(cost.mean(), 1e-12)
        return (perf - lambda_cost * c_norm).argmax(axis=1)
TG_EOF
echo "   wrote src/tokenguard/routers/static.py"

mkdir -p "$(dirname scripts/day2_static_baselines.py)"
cat > scripts/day2_static_baselines.py << 'TG_EOF'
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
TG_EOF
echo "   wrote scripts/day2_static_baselines.py"

mkdir -p "$(dirname tests/test_day2.py)"
cat > tests/test_day2.py << 'TG_EOF'
"""Day 2 tests: metrics correctness, static-router behaviour, runner E2E.

The synthetic pool is constructed so that bigger models are strictly better
AND strictly more expensive on average — making the expected orderings
(oracle ≥ random-mix ≥ endpoints-line) provable, not accidental.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.eval import metrics
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter

MODELS = ["small-1b", "mid-8b", "large-70b"]
TASKS = ["mmlu", "gsm8k", "hellaswag", "mbpp"]


@pytest.fixture()
def bench() -> RouterBench:
    rng = np.random.default_rng(1)
    n = 800
    data = {
        "sample_id": [f"s{i:05d}" for i in range(n)],
        "prompt": [f"Question {i}?" for i in range(n)],
        "eval_name": [TASKS[i % len(TASKS)] for i in range(n)],
    }
    for j, m in enumerate(MODELS):
        acc = 0.45 + 0.20 * j  # 0.45 / 0.65 / 0.85
        data[f"{m}|performance"] = rng.binomial(1, acc, size=n).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.8, 1.2, size=n) * 1e-4 * (10.0 ** j)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


@pytest.fixture()
def split(bench):
    return bench.split_random(test_size=0.25, seed=42)


# ------------------------------- metrics ----------------------------------- #
def test_pareto_front_removes_dominated_points() -> None:
    pts = pd.DataFrame(
        {"cost": [1.0, 2.0, 3.0, 4.0], "quality": [0.5, 0.4, 0.7, 0.7]}
    )
    front = metrics.pareto_front(pts)
    # (2.0, 0.4) dominated by (1.0, 0.5); (4.0, 0.7) dominated by (3.0, 0.7)
    assert list(front["cost"]) == [1.0, 3.0]
    assert (front["quality"].diff().dropna() > 0).all()


def test_quality_at_cost_interpolates_and_saturates() -> None:
    front = pd.DataFrame({"cost": [1.0, 3.0], "quality": [0.4, 0.8]})
    assert metrics.quality_at_cost(front, 2.0) == pytest.approx(0.6)
    assert metrics.quality_at_cost(front, 100.0) == pytest.approx(0.8)  # saturates
    assert metrics.quality_at_cost(front, 0.0) == pytest.approx(0.4)  # clamps


def test_aiq_of_flat_frontier_is_its_quality() -> None:
    front = pd.DataFrame({"cost": [1.0, 2.0], "quality": [0.6, 0.6]})
    assert metrics.aiq(front, 1.0, 2.0) == pytest.approx(0.6)


def test_aiq_rejects_degenerate_window() -> None:
    front = pd.DataFrame({"cost": [1.0], "quality": [0.6]})
    with pytest.raises(ValueError):
        metrics.aiq(front, 2.0, 2.0)


def test_apgr_of_straight_line_is_half() -> None:
    # A frontier exactly on the weak->strong segment recovers half the gap
    # on average (uniform budgets): integral of t over [0,1] = 0.5.
    weak, strong = (1.0, 0.4), (3.0, 0.8)
    line = pd.DataFrame({"cost": [1.0, 3.0], "quality": [0.4, 0.8]})
    assert metrics.apgr(line, weak, strong) == pytest.approx(0.5, abs=0.02)


# ---------------------------- static routers ------------------------------- #
def test_constant_routers_pick_expected_models(split) -> None:
    train, test = split
    cheap = ConstantRouter("cheapest").fit(train)
    best = ConstantRouter("best").fit(train)
    assert train.models[cheap.target_idx_] == "small-1b"
    assert train.models[best.target_idx_] == "large-70b"
    assert len(cheap.frontier(test)) == 1
    assert len(best.frontier(test)) == 1


def test_constant_router_rejects_unknown_strategy(split) -> None:
    train, _ = split
    with pytest.raises(ValueError):
        ConstantRouter("does-not-exist").fit(train)


def test_router_refuses_mismatched_model_set(split) -> None:
    train, test = split
    router = ConstantRouter("best").fit(train)
    shrunk = RouterBench(
        test.df.drop(columns=["perf::small-1b", "cost::small-1b"])
    )
    with pytest.raises(ValueError):
        router.route(shrunk, 0.0)


def test_random_mix_endpoints_match_constant_routers(split) -> None:
    train, test = split
    mix = RandomMixRouter(seed=0).fit(train)
    front = mix.frontier(test)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    # p=0 endpoint == always-cheapest point; p=1 endpoint == always-best point
    assert front.iloc[0]["cost"] == pytest.approx(weak["cost"], rel=1e-9)
    assert front.iloc[0]["quality"] == pytest.approx(weak["quality"], rel=1e-9)
    assert front.iloc[-1]["cost"] == pytest.approx(strong["cost"], rel=1e-9)
    assert front.iloc[-1]["quality"] == pytest.approx(strong["quality"], rel=1e-9)


def test_oracle_dominates_random_mix_on_aiq(split) -> None:
    train, test = split
    oracle_front = OracleRouter().fit(train).frontier(test)
    mix_front = RandomMixRouter(seed=0).fit(train).frontier(test)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])
    assert metrics.aiq(oracle_front, c_lo, c_hi) >= metrics.aiq(
        mix_front, c_lo, c_hi
    ) - 1e-9


def test_oracle_lambda_zero_equals_per_sample_max(split) -> None:
    train, test = split
    oracle = OracleRouter().fit(train)
    choices = oracle.route(test, lambda_cost=0.0)
    perf = test.perf_matrix()
    assert perf[np.arange(len(choices)), choices].mean() == pytest.approx(
        perf.max(axis=1).mean()
    )


# ------------------------------- runner ------------------------------------ #
def test_runner_end_to_end_writes_artifacts(split, tmp_path) -> None:
    train, test = split
    runner = EvalRunner(
        train, test,
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        tag="t",
    )
    summary = runner.run(
        [ConstantRouter("cheapest"), ConstantRouter("best"),
         RandomMixRouter(seed=0), OracleRouter()]
    )
    fig = runner.plot("test")

    assert fig.exists()
    assert (tmp_path / "results" / "t-summary.csv").exists()
    assert len(summary) == 4
    s = summary.set_index("router")
    assert s.loc["oracle", "aiq"] >= s.loc["random-mix", "aiq"] - 1e-9
    assert 0.3 <= s.loc["random-mix", "apgr"] <= 0.7
TG_EOF
echo "   wrote tests/test_day2.py"

echo ""
echo "==> Day-2 files installed. Now run:  make test   (expect 25 passed)"