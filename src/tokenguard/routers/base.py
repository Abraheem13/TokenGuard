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
