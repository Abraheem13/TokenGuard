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
        surprise_gate: bool = True,
        surprise_scale: float = 3.0,
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
        self.surprise_gate = surprise_gate
        self.surprise_scale = surprise_scale
        self.seed = seed

        self.models = base.models_
        self.n_models = len(self.models)
        # context = contrastive projection (proj_dim) + calibrated q (n_models)
        self.ctx_dim = base.P_.shape[1] + self.n_models
        self.fast = LinUCBHead(self.n_arms_, self.ctx_dim, alpha=alpha)
        self.ema_theta = self.fast.theta.copy()
        self.replay = ReplayBuffer(replay_capacity, base.P_.shape[1], self.n_models, seed=seed)
        self._step = 0
        # B2 coupling state (Fast→Slow surprise-triggered consolidation)
        self._surprise_ema = 0.0
        self._last_slow = 0
        self._min_slow_gap = 150          # don't consolidate too often
        self._surprise_trigger = 0.35     # surprise EMA threshold to fire slow
        self._anchor_gain = 0.05          # Slow→Fast anchoring (very weak)
        self._slow_blend = 0.3            # how much refit replaces calibration
        self._slow_min_calerr = 0.30      # skip slow if base already well-calibrated

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
    def run_stream(self, stream: RouterBench, record_every: int = 200,
                   arrival_step: int | None = None, arrival_arm: int | None = None) -> dict:
        """Process a query stream online; return per-step metrics + summary.

        If ``arrival_step`` and ``arrival_arm`` are given, that arm is masked out
        (unavailable) before ``arrival_step`` and becomes selectable afterwards —
        the new-model-arrival experiment. A static policy that was warm-started
        without the arm keeps a stale preference and rarely tries it; the online
        head can discover it via exploration once it is available.

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

            # new-model-arrival mask: the arriving arm is unavailable until its
            # arrival step. We implement masking by a large penalty on that arm.
            avail_penalty = np.zeros(self.n_models)
            if arrival_arm is not None and arrival_step is not None and t < arrival_step:
                avail_penalty[arrival_arm] = 1e9

            if self.enable_fast:
                u = self.fast.ucb(ctx) if not self.enable_mid else (
                    self.ema_theta @ ctx
                    + np.array([
                        self.alpha * np.sqrt(max(ctx @ self.fast.A_inv[a] @ ctx, 0.0))
                        for a in range(self.n_models)
                    ])
                )
                arm = int(np.argmax(u - self.lambda_cost * c_norm_all[t] - avail_penalty))
            else:
                # static base policy (no online adaptation) — ablation baseline
                arm = int(np.argmax(q_row - self.lambda_cost * c_norm_all[t] - avail_penalty))

            # observe true outcome (table lookup; no model inference)
            quality = float(perf[t, arm])
            spend = float(cost[t, arm])
            reward = quality - self.lambda_cost * (spend / max(cost.mean(), 1e-12))

            # FAST + MID updates, gated by surprise (Nested-Learning mechanism 1)
            if self.enable_fast:
                # surprise = |observed − predicted reward| for the chosen arm.
                # The predicted reward uses the fast head's current estimate.
                pred = float(self.fast.theta[arm] @ ctx)
                surprise = abs(reward - pred)
                if self.surprise_gate:
                    # map surprise in [0, ~2] to a gain ≥ ~0.3: familiar outcomes
                    # update gently, surprising ones strongly. Bounded for stability.
                    gain = 1.0 + self.surprise_scale * min(surprise, 2.0)
                else:
                    gain = 1.0
                self.fast.update(arm, ctx, reward, gain=gain)
                self._surprise_ema = (
                    0.99 * getattr(self, "_surprise_ema", surprise) + 0.01 * surprise
                )
                if self.enable_mid:
                    self.ema_theta = (
                        self.ema_beta * self.ema_theta
                        + (1 - self.ema_beta) * self.fast.theta
                    )
            self.replay.add(proj[t], perf[t])

            # SLOW update (Nested-Learning mechanism 2: Fast→Slow context flow).
            # Consolidation fires either on the periodic schedule OR when
            # accumulated surprise crosses a threshold — i.e. when the world has
            # changed enough to warrant it (surprise-triggered consolidation,
            # mirroring the Continuum Memory System).
            self._step += 1
            surprise_triggered = (
                self.enable_slow and self.surprise_gate
                and getattr(self, "_surprise_ema", 0.0) > self._surprise_trigger
                and (self._step - self._last_slow) >= self._min_slow_gap
                and len(self.replay) >= 100
            )
            scheduled = (
                self.enable_slow and self._step % self.slow_update_every == 0
                and len(self.replay) >= 100
            )
            if surprise_triggered or scheduled:
                self._slow_refit()
                self._last_slow = self._step
                # reset the surprise accumulator after consolidating
                self._surprise_ema = 0.0

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
            "post_arrival_quality": (
                float(np.mean(qualities[arrival_step:]))
                if arrival_step is not None else float(np.mean(qualities))
            ),
            "post_arrival_reward": (
                float(np.mean(rewards[arrival_step:]))
                if arrival_step is not None else float(np.mean(rewards))
            ),
        }

    # ------------------------------------------------------------------ #
    def _slow_refit(self) -> None:
        """Consolidate (SLOW level) and anchor the fast head (Slow→Fast flow).

        Designed to *refine, never replace*. On a well-calibrated base (e.g. the
        contrastive router after BCE training) an aggressive refit can only move
        calibration away from its optimum, so we (i) blend the refit calibration
        conservatively with the existing one and (ii) anchor the fast head only
        weakly toward high-confidence consolidated estimates. Both effects are
        deliberately small: the slow level should track genuine long-horizon
        drift, not overwrite a good prior.
        """
        proj, perf = self.replay.sample(self.slow_sample_size, recency_weighted=True)
        E = _l2norm(self.base.E_)
        S = _l2norm(proj) @ E.T                                 # similarities
        a0, b0 = self.base.a_.copy(), self.base.b_.copy()
        # Safety guard: only consolidate if the current calibration is actually
        # miscalibrated on recent replay (mean |p − y| above a floor). On a
        # well-calibrated base the refit can only add noise, so we skip it —
        # this is what keeps `full` from regressing below `fast` when the base
        # is already strong (the real-RouterBench regime).
        P0 = _sigmoid(a0 * S + b0)
        cal_err = float(np.abs(P0 - perf).mean())
        if cal_err < self._slow_min_calerr:
            return
        a, b = a0.copy(), b0.copy()
        for _ in range(60):
            P = _sigmoid(a * S + b)
            G = (P - perf) / len(perf)
            a -= 0.3 * (G * S).sum(axis=0)
            b -= 0.3 * G.sum(axis=0)
        # (i) conservative blend — keep most of the trained calibration
        beta = self._slow_blend
        self.base.a_ = ((1 - beta) * a0 + beta * a).astype(np.float32)
        self.base.b_ = ((1 - beta) * b0 + beta * b).astype(np.float32)

        # (ii) weak Slow→Fast anchoring, only where the consolidated estimate is
        # confident (quality clearly high), so we never drag the fast head toward
        # an uncertain arm.
        if not self.enable_fast or self._anchor_gain <= 0.0:
            return
        k = min(128, proj.shape[0])
        q_cons = _sigmoid(self.base.a_ * S[:k] + self.base.b_)
        for i in range(k):
            best = int(np.argmax(q_cons[i]))
            if q_cons[i, best] < 0.6:        # only anchor on confident estimates
                continue
            ctx = self._context(proj[i], q_cons[i])
            self.fast.update(best, ctx, float(q_cons[i, best]), gain=self._anchor_gain)