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
