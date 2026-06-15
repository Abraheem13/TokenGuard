"""Cascade router (FrugalGPT / Cascade-Routing style baseline, 2023-2025).

Idea: order the models from cheapest to most expensive. For each query, run
the cheapest model whose *predicted success probability* clears a confidence
threshold τ; if no cheap model is confident enough, escalate up the chain,
falling back to the most expensive model. Sweeping τ from 0 -> 1 traces a
cost-quality frontier: low τ keeps almost everything on cheap models (cheap,
lower quality); high τ escalates aggressively (expensive, higher quality).

This is the dominant *deployed* routing pattern in industry (FrugalGPT, Chen
et al. 2023; unified cascade-routing, ICML 2025). It is a strong, intuitive
baseline and a useful foil for the learned router: cascades make a *sequence*
of local accept/reject decisions using a fixed per-model confidence model,
whereas a learned router makes one global decision. Crucially, the cascade's
confidence model here is the same kNN success predictor, so any frontier gap
reflects the *decision policy*, not the underlying quality estimates.

Cost model: escalation means you pay for *every* model you tried. We account
for this honestly — the per-query cost is the sum of the costs of all models
run for that query, not just the final one. This is what makes cascades less
trivially dominant than they first appear, and is the realistic accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench
from tokenguard.eval.metrics import pareto_front
from tokenguard.routers.base import Router
from tokenguard.routers.knn_router import KNNRouter


class CascadeRouter(Router):
    """Cheapest-first escalation with a confidence threshold sweep."""

    name = "cascade"

    def __init__(self, k: int = 50, embedder=None, n_thresholds: int = 21):
        super().__init__()
        # Reuse the kNN predictor as the per-model confidence estimator.
        self._confidence = KNNRouter(k=k, embedder=embedder)
        self.n_thresholds = n_thresholds
        self.cost_order_: np.ndarray | None = None  # model indices, cheap->dear

    def fit(self, train: RouterBench) -> "CascadeRouter":
        super().fit(train)
        self._confidence.fit(train)
        self.cost_order_ = np.argsort(self.cost_estimates_)  # cheapest first
        return self

    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        """Expose the underlying confidence model (used by the base contract)."""
        return self._confidence.predict_quality(bench)

    # The cascade has its own decision + accounting, so it overrides frontier.
    def frontier(self, bench: RouterBench, lambdas=None) -> pd.DataFrame:
        self._check_fitted(bench)
        perf = bench.perf_matrix()
        cost = bench.cost_matrix()
        conf = self._confidence.predict_quality(bench)  # (Nq, M) success probs
        order = self.cost_order_

        rows = []
        for tau in np.linspace(0.0, 1.0, self.n_thresholds):
            chosen, paid = self._run_cascade(conf, cost, order, tau)
            r = np.arange(len(chosen))
            quality = float(perf[r, chosen].mean())
            mean_cost = float(paid.mean())
            rows.append({"lambda": tau, "cost": mean_cost, "quality": quality})
        return pareto_front(pd.DataFrame(rows))

    @staticmethod
    def _run_cascade(conf, cost, order, tau):
        """Return (chosen_model_idx, total_cost_paid) per query for threshold τ."""
        nq = conf.shape[0]
        chosen = np.full(nq, order[-1], dtype=int)  # default: most expensive
        paid = np.zeros(nq)
        decided = np.zeros(nq, dtype=bool)

        for pos, j in enumerate(order):
            # Pay for trying model j on every query not yet decided.
            trying = ~decided
            paid[trying] += cost[trying, j]
            # Accept where confidence clears τ (last model always accepts).
            is_last = pos == len(order) - 1
            accept = trying & ((conf[:, j] >= tau) | is_last)
            chosen[accept] = j
            decided[accept] = True
            if decided.all():
                break
        return chosen, paid