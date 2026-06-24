"""Tests for the single-timescale LinUCB baseline router."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, detect_schema, canonicalise
from tokenguard.routers.single_timescale import SingleTimescaleLinUCB
from tokenguard.routers.contrastive_router import ContrastiveRouter
from tokenguard.routers.base import Router


def _toy_bench(n=1500, M=4, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    tasks = [f"t{i}" for i in range(8)]
    ev = [tasks[i % len(tasks)] for i in range(n)]
    d = {"sample_id": [f"s{i}" for i in range(n)],
         "prompt": [f"{ev[i]}::q{i}" for i in range(n)], "eval_name": ev}
    acc = np.linspace(0.7, 0.4, M)
    cost = np.linspace(0.5, 2.0, M) * 1e-3
    for j in range(M):
        d[f"m{j}|performance"] = rng.binomial(1, acc[j], n).astype(float)
        d[f"m{j}|total_cost"] = rng.uniform(.9, 1.1, n) * cost[j]
    df = pd.DataFrame(d)
    return RouterBench(canonicalise(df, detect_schema(df))), ev, dim


def _fitted_base(bench, ev, dim, seed=0):
    rng = np.random.default_rng(seed)
    famvec = {t: rng.normal(0, 1, dim) for t in set(ev)}
    pe = {}
    for row in bench.df.itertuples():
        v = famvec[row.eval_name] + 0.3 * rng.standard_normal(dim)
        pe[row.prompt] = (v / np.linalg.norm(v)).astype(np.float32)

    class Enc:
        def encode(self, t):
            return np.vstack([pe[x] for x in t]).astype(np.float32)

    base = ContrastiveRouter(encoder=Enc(), proj_dim=dim, seed=seed)
    Router.fit(base, bench)
    base.P_ = np.eye(dim, dtype=np.float32)
    M = len(base.models_)
    E = rng.normal(0, 1, (M, dim)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    base.E_ = E
    base.a_ = np.full(M, 4.0, np.float32)
    base.b_ = np.zeros(M, np.float32)
    return base


def test_baseline_runs_and_returns_metrics():
    bench, ev, dim = _toy_bench()
    train, test = bench.split_random(0.3, 42)
    base = _fitted_base(bench, ev, dim)
    r = SingleTimescaleLinUCB(base, lambda_cost=0.3, alpha=0.5).warm_start(train)
    out = r.run_stream(test)
    assert 0.0 <= out["mean_quality"] <= 1.0
    assert "cum_reward" in out and "trace_reward" in out


def test_baseline_requires_fitted_base():
    class Unfitted:
        P_ = None
        models_ = []
    with pytest.raises(RuntimeError):
        SingleTimescaleLinUCB(Unfitted())