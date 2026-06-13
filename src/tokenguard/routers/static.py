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
