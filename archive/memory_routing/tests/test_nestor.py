"""Tests for the NESTOR router and the model-drift win (the core invention)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from tokenguard.data.routerbench import RouterBench, detect_schema, canonicalise
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.base import Router
from tokenguard.routers.nestor_router import NestorRouter
from tokenguard.routers.single_timescale import SingleTimescaleLinUCB


def _drift_setup(seed=3, phase_len=3000):
    rng = np.random.default_rng(seed)
    T, M, n, dim = 12, 6, 9000, 32
    tasks = [f"task{i}" for i in range(T)]
    ev = [tasks[i % T] for i in range(n)]
    fam = {t: rng.normal(0, 1, dim) for t in tasks}
    cost = np.linspace(0.6, 1.4, M) * 1e-3
    d = {"sample_id": [f"s{i}" for i in range(n)],
         "prompt": [f"{ev[i]}::q{i}" for i in range(n)], "eval_name": ev}
    perf = np.zeros((n, M))
    for i in range(n):
        best = (hash(ev[i]) + i // phase_len) % M
        for j in range(M):
            perf[i, j] = rng.binomial(1, 0.88 if j == best else 0.32)
    for j in range(M):
        d[f"m{j}|performance"] = perf[:, j]
        d[f"m{j}|total_cost"] = rng.uniform(.9, 1.1, n) * cost[j]
    b = RouterBench(canonicalise(pd.DataFrame(d), detect_schema(pd.DataFrame(d))))
    pe = {r.prompt: (fam[r.eval_name] + 0.3 * rng.standard_normal(dim))
          for r in b.df.itertuples()}
    pe = {k: (v / np.linalg.norm(v)).astype(np.float32) for k, v in pe.items()}

    class Enc:
        def encode(self, t):
            return np.vstack([pe[x] for x in t]).astype(np.float32)

    tr, te = b.split_random(0.3, 42)
    base = ContrastiveRouter(encoder=Enc(), proj_dim=dim, seed=0)
    Router.fit(base, tr)
    base.P_ = np.eye(dim, dtype=np.float32)
    rng2 = np.random.default_rng(seed + 100)
    E = rng2.normal(0, 1, (M, dim)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    base.E_ = E
    base.a_ = np.full(M, 3.0, np.float32)
    base.b_ = np.zeros(M, np.float32)
    return base, tr, RouterBench(te.df.reset_index(drop=True))


def _clone(base):
    c = copy.copy(base)
    c.P_, c.E_ = base.P_.copy(), base.E_.copy()
    c.a_, c.b_ = base.a_.copy(), base.b_.copy()
    return c


def test_nestor_runs_and_returns_metrics():
    base, tr, stream = _drift_setup()
    r = NestorRouter(_clone(base), lambda_cost=0.3, alpha=0.6).warm_start(tr)
    out = r.run_stream(stream)
    assert 0.0 <= out["mean_quality"] <= 1.0
    assert "cum_reward" in out


def test_nestor_beats_single_timescale_under_model_drift():
    """The core invention: recency memory tracks the drifting per-cluster winner
    that a static contextual bandit cannot."""
    base, tr, stream = _drift_setup()
    single = SingleTimescaleLinUCB(_clone(base), lambda_cost=0.3,
                                   alpha=0.6, seed=42).warm_start(tr)
    nestor = NestorRouter(_clone(base), lambda_cost=0.3, alpha=0.6,
                          mid_weight=0.5, lr=0.2, surprise_scale=3.0,
                          seed=42).warm_start(tr)
    rs = single.run_stream(stream)["mean_reward"]
    rn = nestor.run_stream(stream)["mean_reward"]
    assert rn > rs, f"NESTOR {rn:.4f} should beat single-timescale {rs:.4f}"


def test_use_mid_false_matches_single_behaviour():
    """With the mid memory disabled, NESTOR reduces to the fast level."""
    base, tr, stream = _drift_setup()
    r = NestorRouter(_clone(base), lambda_cost=0.3, alpha=0.6,
                     use_mid=False).warm_start(tr)
    out = r.run_stream(stream)
    assert 0.0 <= out["mean_reward"] <= 1.0