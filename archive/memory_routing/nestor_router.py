"""NESTOR top-level router — fast LinUCB (L1) + Titans mid memory (L2) via CMS.

Same run_stream contract as SingleTimescaleLinUCB so the two are directly
comparable: NESTOR adds the L2 neural memory on top of the identical fast level.
The scientific test is whether L2 helps on streams with recurring structure.
"""

from __future__ import annotations

import numpy as np

from tokenguard.memory.cms import ContinuumMemoryRouter
from tokenguard.online.linucb import LinUCBHead


class NestorRouter:
    def __init__(self, base, lambda_cost: float = 0.5, alpha: float = 0.5,
                 c2: int = 1, mid_weight: float = 0.5, lr: float = 0.2,
                 surprise_scale: float = 3.0, use_mid: bool = True,
                 seed: int = 42):
        if base.P_ is None:
            raise RuntimeError("base must be fitted before constructing NESTOR")
        self.base = base
        self.lambda_cost = lambda_cost
        self.alpha = alpha
        self.use_mid = use_mid
        self.seed = seed
        self.n_models = len(base.models_)
        self.ctx_dim = base.P_.shape[1]
        self.fast = LinUCBHead(self.n_models, self.ctx_dim, alpha=alpha)
        self.cms = ContinuumMemoryRouter(
            key_dim=self.ctx_dim, n_models=self.n_models, c2=c2,
            mid_weight=mid_weight, lr=lr, surprise_scale=surprise_scale, seed=seed,
        )

    # ------------------------------------------------------------------ #
    def _project(self, stream) -> np.ndarray:
        emb = self.base.encoder.encode(list(stream.df["prompt"]))
        proj = emb @ self.base.P_
        proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        return proj.astype(np.float64)

    def warm_start(self, train, n: int = 4000) -> "NestorRouter":
        proj = self._project(train)
        perf = train.perf_matrix()
        cost = train.cost_matrix()
        mc = max(cost.mean(), 1e-12)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(proj), size=min(n, len(proj)), replace=False)
        R = perf[idx] - self.lambda_cost * (cost[idx] / mc)
        self.fast.warm_start(proj[idx], R)
        return self

    def run_stream(self, stream, record_every: int = 200) -> dict:
        proj = self._project(stream)
        perf = stream.perf_matrix()
        cost = stream.cost_matrix()
        mc = max(cost.mean(), 1e-12)
        c_norm = cost / mc

        rewards, qualities, costs = [], [], []
        trace_steps, trace_reward, cum = [], [], 0.0
        for t in range(len(stream.df)):
            ctx = proj[t]
            # L1 fast UCB scores
            mean, bonus = self.fast.scores(ctx)
            fast_pred = mean + bonus
            # blend with L2 mid memory (CMS)
            score = self.cms.predict(ctx, fast_pred) if self.use_mid else fast_pred
            arm = int(np.argmax(score - self.lambda_cost * c_norm[t]))

            quality = float(perf[t, arm])
            spend = float(cost[t, arm])
            reward = quality - self.lambda_cost * (spend / mc)

            # update both levels
            self.fast.update(arm, ctx, reward)
            if self.use_mid:
                self.cms.observe(ctx, arm, reward, recurrent=True)

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