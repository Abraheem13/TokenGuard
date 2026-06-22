"""Day 6 — router service: the bridge between the proxy and the trained router.

Loads a fitted contrastive (+ optional online) router and exposes a single
``route(query)`` call returning the chosen model, the predicted quality, and an
estimated cost. The proxy stays thin: it does HTTP, this does the ML.

For the live demo the heavy candidate models are served separately; this service
only decides *which* model to call, which is the contribution under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RouteDecision:
    chosen_model: str
    predicted_quality: float
    est_cost_usd: float
    scores: dict[str, float]


class RouterService:
    """Wrap a fitted router with a query→decision interface.

    Parameters
    ----------
    base
        A fitted ``ContrastiveRouter`` (has ``encoder``, ``P_``, ``E_``,
        ``a_``, ``b_``, ``models_``).
    cost_table
        Mapping model-name → estimated USD cost per query (from training data).
    lambda_cost
        Cost–quality trade-off used by the decision rule.
    """

    def __init__(self, base, cost_table: dict[str, float], lambda_cost: float = 0.5):
        if base.P_ is None:
            raise RuntimeError("router must be fitted before serving")
        self.base = base
        self.models = list(base.models_)
        self.cost_table = cost_table
        self.lambda_cost = lambda_cost
        self._mean_cost = float(np.mean(list(cost_table.values()))) or 1e-9

    # ------------------------------------------------------------------ #
    def _quality(self, query: str) -> np.ndarray:
        emb = self.base.encoder.encode([query]).astype(np.float32)
        proj = emb @ self.base.P_
        proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        E = self.base.E_ / (np.linalg.norm(self.base.E_, axis=1, keepdims=True) + 1e-12)
        sim = (proj @ E.T)[0]
        return 1.0 / (1.0 + np.exp(-(self.base.a_ * sim + self.base.b_)))

    def route(self, query: str) -> RouteDecision:
        q = self._quality(query)
        costs = np.array([self.cost_table.get(m, self._mean_cost) for m in self.models])
        c_norm = costs / self._mean_cost
        score = q - self.lambda_cost * c_norm
        arm = int(np.argmax(score))
        return RouteDecision(
            chosen_model=self.models[arm],
            predicted_quality=float(q[arm]),
            est_cost_usd=float(costs[arm]),
            scores={m: float(s) for m, s in zip(self.models, score)},
        )