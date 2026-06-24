"""Quality-token Pareto + matched-budget / matched-accuracy + significance.

The three honest ways to show 'same quality, fewer tokens':
  1. pareto_area(points)                     area under accuracy-vs-tokens curve
  2. accuracy_at_budget(points, budget)      interpolate accuracy at common tokens
  3. tokens_at_accuracy(points, target_acc)  tokens to reach an accuracy
Plus paired bootstrap CIs for deltas.

TODO(week3): finalise interpolation + bootstrap/Wilcoxon helpers.
"""
from __future__ import annotations
import numpy as np


def pareto_area(tokens, accuracy) -> float:
    order = np.argsort(tokens)
    t, a = np.asarray(tokens)[order], np.asarray(accuracy)[order]
    return float(np.trapezoid(a, t) / (t.max() - t.min() + 1e-9))


def accuracy_at_budget(tokens, accuracy, budget) -> float:
    return float(np.interp(budget, np.sort(tokens),
                           np.asarray(accuracy)[np.argsort(tokens)]))


def tokens_at_accuracy(tokens, accuracy, target_acc) -> float:
    order = np.argsort(accuracy)
    return float(np.interp(target_acc, np.asarray(accuracy)[order],
                           np.asarray(tokens)[order]))


def bootstrap_delta(a, b, n=10000, seed=0):
    """Paired bootstrap 95% CI for mean(a)-mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    deltas = a[idx].mean(1) - b[idx].mean(1)
    return float(deltas.mean()), (float(np.percentile(deltas, 2.5)),
                                  float(np.percentile(deltas, 97.5)))
