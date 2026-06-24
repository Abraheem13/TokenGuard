"""Single-timescale LinUCB router — the honest comparison point for NESTOR.

This is the strong, established online baseline that NESTOR must beat to justify
its extra machinery. It is a pure contextual bandit (LinUCB) over the query
embedding with a cost-aware decision rule — the same family as PILOT
(arXiv:2508.21141), BARP (arXiv:2510.07429) and MixLLM (NAACL 2025). It is the
``fast``-only configuration of the nested router, promoted to a first-class,
clearly-named baseline so every NESTOR result is reported as a delta against it
(not merely against a static router).

Why this matters for the dissertation
-------------------------------------
Your honest Day-5 finding was that the fast level already captures the
adaptable signal on a well-calibrated base. The scientific question NESTOR asks
is therefore: *does adding multi-timescale associative memory beat this strong
single-timescale bandit, and under which non-stationary regimes?* This class is
that yardstick.
"""

from __future__ import annotations

import numpy as np

from tokenguard.online.linucb import LinUCBHead


class SingleTimescaleLinUCB:
    """Pure LinUCB contextual-bandit router (one timescale, no memory hierarchy).

    Parameters
    ----------
    base
        A fitted contrastive/MF base used only to (a) provide the per-query
        projected context and (b) warm-start the bandit from training data, so
        the comparison with NESTOR is apples-to-apples.
    lambda_cost
        Cost–quality trade-off in the decision rule argmax(q - λ·cost).
    alpha
        LinUCB exploration coefficient.
    seed
        RNG seed.
    """

    def __init__(self, base, lambda_cost: float = 0.5, alpha: float = 0.5,
                 seed: int = 42):
        if base.P_ is None:
            raise RuntimeError("base must be fitted before constructing baseline")
        self.base = base
        self.lambda_cost = lambda_cost
        self.alpha = alpha
        self.seed = seed
        self.n_models = len(base.models_)
        self.ctx_dim = base.P_.shape[1]
        self.fast = LinUCBHead(self.n_models, self.ctx_dim, alpha=alpha)

    # ------------------------------------------------------------------ #
    def _project(self, stream) -> np.ndarray:
        emb = self.base.encoder.encode(list(stream.df["prompt"]))
        proj = emb @ self.base.P_
        proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        return proj.astype(np.float64)

    def warm_start(self, train, n: int = 4000) -> "SingleTimescaleLinUCB":
        """Initialise the bandit from a sample of training contexts/rewards."""
        proj = self._project(train)
        perf = train.perf_matrix()
        cost = train.cost_matrix()
        mean_cost = max(cost.mean(), 1e-12)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(proj), size=min(n, len(proj)), replace=False)
        R = perf[idx] - self.lambda_cost * (cost[idx] / mean_cost)
        self.fast.warm_start(proj[idx], R)
        return self

    def run_stream(self, stream, record_every: int = 200) -> dict:
        """Process a query stream online; same return contract as the nested
        router so the two are directly comparable."""
        proj = self._project(stream)
        perf = stream.perf_matrix()
        cost = stream.cost_matrix()
        mean_cost = max(cost.mean(), 1e-12)
        c_norm = cost / mean_cost

        rewards, qualities, costs = [], [], []
        trace_steps, trace_reward, cum = [], [], 0.0
        for t in range(len(stream.df)):
            ctx = proj[t]
            arm = self.fast.recommend(ctx, c_norm[t], self.lambda_cost)
            quality = float(perf[t, arm])
            spend = float(cost[t, arm])
            reward = quality - self.lambda_cost * (spend / mean_cost)
            self.fast.update(arm, ctx, reward)

            rewards.append(reward); qualities.append(quality); costs.append(spend)
            cum += reward
            if t % record_every == 0:
                trace_steps.append(t); trace_reward.append(cum)

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_quality": float(np.mean(qualities)),
            "mean_cost": float(np.mean(costs)),
            "cum_reward": float(cum),
            "trace_steps": trace_steps,
            "trace_reward": trace_reward,
        }