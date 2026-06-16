"""Nested online router — the three-timescale core (TokenGuard's contribution).

This composes the Day-4 contrastive router with online adaptation on three
update frequencies, instantiating the Nested Learning principle (Behrouz et
al., NeurIPS 2025) that *update frequency defines the optimisation level*:

  L1 · FAST  (every request) — a LinUCB bandit head over the contrastive
             embedding adapts the routing policy from each observed reward.
  L2 · MID   (decay)         — an EMA of the fast head's theta stabilises the
             policy, damping the noise of single-request updates.
  L3 · SLOW  (every N reqs)  — a periodic refit of the per-model calibration
             (a, b) over a replay-buffer sample consolidates knowledge and
             tracks slow drift in model quality.

A conventional static router is the degenerate single-frequency case (train
once, never update). Under distribution shift, the multi-timescale router
keeps the cost-quality frontier where static routers decay — the dissertation's
central experiment (see online/shift.py).

The router consumes a *stream* of queries. For each query it:
  1. embeds it (via the frozen encoder / cache),
  2. forms a context = [contrastive projection ; calibrated quality estimate],
  3. picks an arm with the FAST head under the cost-aware rule,
  4. observes the true per-model success (from RouterBench — no model calls),
  5. computes reward r = quality - lambda * cost and updates FAST + MID,
  6. periodically runs the SLOW refit from the replay buffer.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench
from tokenguard.online.linucb import LinUCBHead
from tokenguard.online.replay_buffer import ReplayBuffer
from tokenguard.routers.contrastive_router import ContrastiveRouter, _l2norm, _sigmoid
from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.online.nested")


class NestedOnlineRouter:
    """Contrastive base + LinUCB(FAST) + EMA(MID) + replay refit(SLOW)."""

    def __init__(
        self,
        base: ContrastiveRouter,
        lambda_cost: float = 0.5,
        alpha: float = 0.5,
        ema_beta: float = 0.99,
        slow_update_every: int = 500,
        slow_sample_size: int = 2000,
        replay_capacity: int = 20000,
        enable_fast: bool = True,
        enable_mid: bool = True,
        enable_slow: bool = True,
        seed: int = 42,
    ):
        if base.P_ is None:
            raise RuntimeError("base contrastive router must be fitted first")
        self.base = base
        self.lambda_cost = lambda_cost
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.slow_update_every = slow_update_every
        self.slow_sample_size = slow_sample_size
        self.enable_fast = enable_fast
        self.enable_mid = enable_mid
        self.enable_slow = enable_slow
        self.seed = seed

        self.models = base.models_
        self.n_models = len(self.models)
        # context = contrastive projection (proj_dim) + calibrated q (n_models)
        self.ctx_dim = base.P_.shape[1] + self.n_models
        self.fast = LinUCBHead(self.n_arms_, self.ctx_dim, alpha=alpha)
        self.ema_theta = self.fast.theta.copy()
        self.replay = ReplayBuffer(replay_capacity, base.P_.shape[1], self.n_models, seed=seed)
        self._step = 0

    @property
    def n_arms_(self) -> int:
        return len(self.base.models_)

    # ------------------------------------------------------------------ #
    def _embed(self, prompts: list[str]) -> np.ndarray:
        return self.base.encoder.encode(prompts).astype(np.float32)

    def _project(self, emb: np.ndarray) -> np.ndarray:
        return _l2norm(emb @ self.base.P_)

    def _calibrated_quality(self, proj: np.ndarray) -> np.ndarray:
        E = _l2norm(self.base.E_)
        sim = proj @ E.T
        return _sigmoid(self.base.a_ * sim + self.base.b_)

    def _context(self, proj_row: np.ndarray, q_row: np.ndarray) -> np.ndarray:
        return np.concatenate([proj_row, q_row]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def warm_start(self, train: RouterBench) -> "NestedOnlineRouter":
        """Seed FAST from the base router's training split (warm, not cold).

        Uses the base contrastive predictions as the initial reward estimate so
        the bandit starts near the offline policy and then adapts online.
        """
        emb = self._embed(train.df["prompt"].tolist())
        proj = self._project(emb)
        q = self._calibrated_quality(proj)
        cost = train.cost_matrix()
        c_norm = cost / max(cost.mean(), 1e-12)
        R = q - self.lambda_cost * c_norm                       # (n, m) reward
        # build contexts and batch-seed each arm (subsample for speed)
        rng = np.random.default_rng(self.seed)
        n = len(emb)
        take = min(n, 4000)
        idx = rng.choice(n, size=take, replace=False)
        X = np.stack([self._context(proj[i], q[i]) for i in idx])
        self.fast.warm_start(X, R[idx])
        self.ema_theta = self.fast.theta.copy()
        logger.info("Warm-started FAST head from %d training contexts", take)
        return self

    # ------------------------------------------------------------------ #
    def run_stream(self, stream: RouterBench, record_every: int = 200) -> dict:
        """Process a query stream online; return per-step metrics + summary.

        Returns a dict with cumulative reward, a running quality/cost trace, and
        the mean reward — the inputs to the Day-5 adaptation curves.
        """
        emb = self._embed(stream.df["prompt"].tolist())
        proj = self._project(emb)
        perf = stream.perf_matrix()
        cost = stream.cost_matrix()
        c_norm_all = cost / max(cost.mean(), 1e-12)

        rewards, qualities, costs = [], [], []
        trace_steps, trace_reward = [], []
        cum_reward = 0.0

        for t in range(len(stream.df)):
            q_row = self._calibrated_quality(proj[t : t + 1])[0]
            ctx = self._context(proj[t], q_row)

            if self.enable_fast:
                u = self.fast.ucb(ctx) if not self.enable_mid else (
                    self.ema_theta @ ctx
                    + np.array([
                        self.alpha * np.sqrt(max(ctx @ self.fast.A_inv[a] @ ctx, 0.0))
                        for a in range(self.n_models)
                    ])
                )
                arm = int(np.argmax(u - self.lambda_cost * c_norm_all[t]))
            else:
                # static base policy (no online adaptation) — ablation baseline
                arm = int(np.argmax(q_row - self.lambda_cost * c_norm_all[t]))

            # observe true outcome (table lookup; no model inference)
            quality = float(perf[t, arm])
            spend = float(cost[t, arm])
            reward = quality - self.lambda_cost * (spend / max(cost.mean(), 1e-12))

            # FAST + MID updates
            if self.enable_fast:
                self.fast.update(arm, ctx, reward)
                if self.enable_mid:
                    self.ema_theta = (
                        self.ema_beta * self.ema_theta
                        + (1 - self.ema_beta) * self.fast.theta
                    )
            self.replay.add(proj[t], perf[t])

            # SLOW update
            self._step += 1
            if (self.enable_slow and self._step % self.slow_update_every == 0
                    and len(self.replay) >= 100):
                self._slow_refit()

            cum_reward += reward
            rewards.append(reward); qualities.append(quality); costs.append(spend)
            if (t + 1) % record_every == 0:
                trace_steps.append(t + 1)
                trace_reward.append(np.mean(rewards[-record_every:]))

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_quality": float(np.mean(qualities)),
            "mean_cost": float(np.mean(costs)),
            "cum_reward": float(cum_reward),
            "trace_steps": trace_steps,
            "trace_reward": trace_reward,
        }

    # ------------------------------------------------------------------ #
    def _slow_refit(self) -> None:
        """Refit per-model calibration (a, b) on a replay sample (SLOW level)."""
        proj, perf = self.replay.sample(self.slow_sample_size, recency_weighted=True)
        E = _l2norm(self.base.E_)
        S = _l2norm(proj) @ E.T                                 # similarities
        a, b = self.base.a_.copy(), self.base.b_.copy()
        for _ in range(100):
            P = _sigmoid(a * S + b)
            G = (P - perf) / len(perf)
            a -= 0.5 * (G * S).sum(axis=0)
            b -= 0.5 * G.sum(axis=0)
        # write back the consolidated calibration
        self.base.a_, self.base.b_ = a.astype(np.float32), b.astype(np.float32)