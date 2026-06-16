"""LinUCB contextual-bandit head — the FAST level of the nested router.

Implements LinUCB (Li et al., WWW 2010): one ridge-regression model per arm
(LLM). The context is the query's (projected) embedding; each arm predicts the
reward of routing that query to that model, plus an upper-confidence bonus that
drives exploration of under-observed (query-region, model) pairs.

Update rule per request (arm a played, reward r, context x):
    A_a <- A_a + x x^T
    b_a <- b_a + r x
    theta_a = A_a^{-1} b_a
    UCB_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

This is the C = 1 (every-request) level in the Nested Learning hierarchy: it
adapts the routing policy online from observed rewards without any gradient
backprop, so it is cheap enough to run inside the proxy on every call.

We maintain A^{-1} incrementally via the Sherman-Morrison update, so each step
is O(d^2) with no matrix inversion. Rewards use the shared definition
r = quality - lambda * cost (computed by the caller), and the head exposes a
``recommend`` that returns the cost-aware arm directly.
"""

from __future__ import annotations

import numpy as np


class LinUCBHead:
    """Per-arm LinUCB with incremental inverse covariance (Sherman-Morrison)."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, l2: float = 1.0):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        # A_a = l2 * I  (ridge prior); store A^{-1} directly
        self.A_inv = np.stack([np.eye(dim, dtype=np.float64) / l2 for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim), dtype=np.float64)
        self.theta = np.zeros((n_arms, dim), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def scores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, ucb_bonus) per arm for a single context ``x``."""
        x = x.astype(np.float64)
        mean = self.theta @ x                                  # (n_arms,)
        # bonus_a = alpha * sqrt(x^T A_a^{-1} x)
        bonus = np.array([
            self.alpha * np.sqrt(max(x @ self.A_inv[a] @ x, 0.0))
            for a in range(self.n_arms)
        ])
        return mean, bonus

    def ucb(self, x: np.ndarray) -> np.ndarray:
        mean, bonus = self.scores(x)
        return mean + bonus

    def recommend(self, x: np.ndarray, cost: np.ndarray, lambda_cost: float) -> int:
        """Cost-aware arm: argmax UCB(quality) - lambda * normalised cost."""
        u = self.ucb(x)
        c_norm = cost / max(cost.mean(), 1e-12)
        return int(np.argmax(u - lambda_cost * c_norm))

    # ------------------------------------------------------------------ #
    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """Sherman-Morrison rank-1 update of arm ``arm`` with (x, reward)."""
        x = x.astype(np.float64)
        Ainv = self.A_inv[arm]
        Ax = Ainv @ x
        denom = 1.0 + x @ Ax
        self.A_inv[arm] = Ainv - np.outer(Ax, Ax) / denom      # rank-1 downdate
        self.b[arm] += reward * x
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]

    # ------------------------------------------------------------------ #
    def warm_start(self, X: np.ndarray, R: np.ndarray) -> None:
        """Batch-initialise all arms from offline (context, reward) data.

        ``R`` is (n, n_arms) of per-arm rewards. Used to seed the fast head
        from the contrastive router's training split so online learning starts
        from a sensible policy rather than cold.
        """
        for a in range(self.n_arms):
            for x, r in zip(X, R[:, a]):
                self.update(a, x, float(r))