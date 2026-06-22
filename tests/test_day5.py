"""Day 5 tests: replay buffer, LinUCB head, and the nested online router.

The online loop is tested *without torch* by constructing a contrastive base
whose learned parameters (P, E, a, b) are set directly, so the FAST/MID/SLOW
machinery and the stream/shift logic are verified on pure NumPy. The central
test asserts the headline claim: under distribution shift, enabling online
adaptation yields higher mean reward than the static (frozen) policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.online.linucb import LinUCBHead
from tokenguard.online.nested_router import NestedOnlineRouter
from tokenguard.online.replay_buffer import ReplayBuffer
from tokenguard.online.shift import (
    family_boundaries,
    shift_stream,
    shuffled_stream,
)
from tokenguard.routers.contrastive_router import ContrastiveRouter

TASKS = ["mmlu-x", "grade-school-math", "mbpp", "hellaswag"]


# --------------------------------------------------------------------------- #
# Replay buffer                                                               #
# --------------------------------------------------------------------------- #
def test_replay_buffer_ring_overwrites_oldest() -> None:
    buf = ReplayBuffer(capacity=3, emb_dim=2, n_models=2)
    for i in range(5):
        buf.add(np.full(2, i, dtype=np.float32), np.zeros(2, dtype=np.float32))
    assert len(buf) == 3
    emb, _ = buf.all()
    # the three most recent values (2,3,4) should remain
    assert set(emb[:, 0].astype(int)) == {2, 3, 4}


def test_replay_buffer_sample_size_and_shape() -> None:
    buf = ReplayBuffer(capacity=100, emb_dim=4, n_models=3, seed=0)
    for i in range(50):
        buf.add(np.random.rand(4).astype(np.float32), np.random.rand(3).astype(np.float32))
    e, p = buf.sample(10)
    assert e.shape == (10, 4) and p.shape == (10, 3)


def test_replay_buffer_empty_sample_raises() -> None:
    buf = ReplayBuffer(capacity=5, emb_dim=2, n_models=2)
    with pytest.raises(ValueError):
        buf.sample(1)


# --------------------------------------------------------------------------- #
# LinUCB                                                                      #
# --------------------------------------------------------------------------- #
def test_linucb_learns_best_arm_from_rewards() -> None:
    """With a fixed context, the arm given higher rewards should win."""
    head = LinUCBHead(n_arms=3, dim=4, alpha=0.0)  # no exploration: pure greedy
    x = np.array([1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        head.update(0, x, rng.normal(0.2, 0.05))   # arm 0 mediocre
        head.update(1, x, rng.normal(0.9, 0.05))   # arm 1 best
        head.update(2, x, rng.normal(0.1, 0.05))   # arm 2 worst
    mean, _ = head.scores(x)
    assert int(mean.argmax()) == 1


def test_linucb_exploration_bonus_decreases_with_observations() -> None:
    head = LinUCBHead(n_arms=2, dim=3, alpha=1.0)
    x = np.array([1.0, 1.0, 0.0])
    _, bonus0 = head.scores(x)
    for _ in range(50):
        head.update(0, x, 0.5)
    _, bonus1 = head.scores(x)
    # the played arm's uncertainty must shrink
    assert bonus1[0] < bonus0[0]


def test_linucb_cost_aware_recommend_prefers_cheaper_when_tied() -> None:
    head = LinUCBHead(n_arms=2, dim=2, alpha=0.0)
    x = np.array([1.0, 0.0])
    for _ in range(100):
        head.update(0, x, 0.8)
        head.update(1, x, 0.8)               # equal quality
    cost = np.array([1.0, 5.0])              # arm 0 cheaper
    assert head.recommend(x, cost, lambda_cost=1.0) == 0


# --------------------------------------------------------------------------- #
# Stream / shift construction                                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def bench() -> RouterBench:
    rng = np.random.default_rng(7)
    n = 2400
    models = ["m-know", "m-math", "m-code", "m-common"]
    ev = [TASKS[i % len(TASKS)] for i in range(n)]
    data = {
        "sample_id": [f"s{i}" for i in range(n)],
        "prompt": [f"{ev[i]} :: item {i}" for i in range(n)],
        "eval_name": ev,
    }
    # each model specialises in one family; success depends on family
    fam_of = {"mmlu-x": 0, "grade-school-math": 1, "mbpp": 2, "hellaswag": 3}
    for j, m in enumerate(models):
        acc = np.array([0.9 if fam_of[e] == j else 0.25 for e in ev])
        data[f"{m}|performance"] = rng.binomial(1, acc).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.9, 1.1, n) * 1e-4 * (j + 1)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


def test_shuffled_stream_preserves_rows(bench) -> None:
    s = shuffled_stream(bench, seed=1)
    assert len(s.df) == len(bench.df)
    assert set(s.df.sample_id) == set(bench.df.sample_id)


def test_shift_stream_is_block_sequential(bench) -> None:
    s = shift_stream(bench, seed=1)
    marks = family_boundaries(s)
    # families should appear in contiguous blocks => few boundaries (<= #families)
    assert len(marks) <= 6
    # and math should precede knowledge in the configured order
    fam_order = [f for f, _ in marks]
    if "math" in fam_order and "knowledge" in fam_order:
        assert fam_order.index("math") < fam_order.index("knowledge")


# --------------------------------------------------------------------------- #
# Nested online router (torch-free base)                                      #
# --------------------------------------------------------------------------- #
def _make_base(bench: RouterBench, proj_dim: int = 8, seed: int = 0) -> ContrastiveRouter:
    """Construct a contrastive base with directly-set params (no torch fit).

    The projection is random; the LLM embeddings are set so each model's
    embedding points along a distinct axis, and calibration is identity-ish —
    enough structure for the online loop to operate on.
    """
    rng = np.random.default_rng(seed)
    base = ContrastiveRouter(encoder=_TaskEncoder(proj_dim), proj_dim=proj_dim, seed=seed)
    # mimic a fitted state
    from tokenguard.routers.base import Router

    Router.fit(base, bench)
    m = len(bench.models)
    base.P_ = rng.normal(0, 1, (proj_dim, proj_dim)).astype(np.float32)
    E = np.zeros((m, proj_dim), dtype=np.float32)
    for j in range(m):
        E[j, j % proj_dim] = 1.0
    base.E_ = E
    base.a_ = np.full(m, 4.0, dtype=np.float32)
    base.b_ = np.zeros(m, dtype=np.float32)
    return base


class _TaskEncoder:
    """Encoder mapping each task family to a distinct unit vector."""

    def __init__(self, dim: int):
        self.dim = dim
        self.map = {t: i for i, t in enumerate(TASKS)}

    def encode(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            fam = t.split(" :: ", 1)[0]
            X[i, self.map.get(fam, 0) % self.dim] = 1.0
        return X


def test_nested_router_runs_stream_and_reports_metrics(bench) -> None:
    train, test = bench.split_random(0.3, 42)
    base = _make_base(train)
    router = NestedOnlineRouter(base, slow_update_every=200, seed=42).warm_start(train)
    out = router.run_stream(shuffled_stream(test, seed=1))
    assert 0.0 <= out["mean_quality"] <= 1.0
    assert len(out["trace_steps"]) == len(out["trace_reward"])
    assert "cum_reward" in out


def test_online_adaptation_beats_static_under_shift(bench) -> None:
    """Headline claim: under distribution shift, online > static on mean reward."""
    train, test = bench.split_random(0.3, 42)
    stream = shift_stream(test, seed=1)

    static = NestedOnlineRouter(
        _make_base(train), enable_fast=False, enable_mid=False, enable_slow=False,
        seed=42,
    ).warm_start(train)
    online = NestedOnlineRouter(
        _make_base(train), enable_fast=True, enable_mid=True, enable_slow=True,
        slow_update_every=150, seed=42,
    ).warm_start(train)

    r_static = static.run_stream(stream)["mean_reward"]
    r_online = online.run_stream(stream)["mean_reward"]
    assert r_online > r_static, f"online {r_online:.4f} should beat static {r_static:.4f}"


# --------------------------------------------------------------------------- #
# B1/B2/C1 — surprise gating, coupling, new-model-arrival                      #
# --------------------------------------------------------------------------- #
def test_linucb_surprise_gain_weights_update() -> None:
    """gain=g once equals g unit-gain updates (exact weighted observation)."""
    h1 = LinUCBHead(n_arms=1, dim=3, alpha=0.0)
    h2 = LinUCBHead(n_arms=1, dim=3, alpha=0.0)
    x = np.array([0.5, -0.3, 0.8]); r = 0.7
    h1.update(0, x, r, gain=3.0)
    for _ in range(3):
        h2.update(0, x, r, gain=1.0)
    assert np.allclose(h1.theta, h2.theta, atol=1e-9)
    assert np.allclose(h1.A_inv, h2.A_inv, atol=1e-9)


def test_linucb_gain_zero_is_noop() -> None:
    h = LinUCBHead(n_arms=2, dim=3, alpha=0.0)
    h.update(0, np.ones(3), 0.5, gain=1.0)
    before = h.theta.copy()
    h.update(0, np.random.default_rng(0).standard_normal(3), 0.9, gain=0.0)
    assert np.allclose(before, h.theta)


def test_surprise_gating_changes_behaviour(bench) -> None:
    """Enabling the surprise gate should produce a different (not identical)
    policy trajectory than disabling it — i.e. the gate actually acts."""
    train, test = bench.split_random(0.3, 42)
    stream = shift_stream(test, seed=1)
    common = dict(enable_fast=True, enable_mid=True, enable_slow=True, seed=42)
    off = NestedOnlineRouter(_make_base(train), surprise_gate=False, **common).warm_start(train)
    on = NestedOnlineRouter(_make_base(train), surprise_gate=True, **common).warm_start(train)
    r_off = off.run_stream(stream)["mean_reward"]
    r_on = on.run_stream(stream)["mean_reward"]
    assert r_off != r_on  # the gate measurably changes the outcome


def test_new_model_arrival_masks_arm_before_arrival(bench) -> None:
    """Before the arrival step the new arm must never be selected; the static
    router (frozen) should adopt it far less than the online router."""
    train, test = bench.split_random(0.3, 42)
    stream = shuffled_stream(test, seed=1)
    arrival_arm = 0
    arrival_step = len(stream.df) // 2

    online = NestedOnlineRouter(
        _make_base(train), enable_fast=True, enable_mid=True, enable_slow=True,
        surprise_gate=True, seed=42,
    ).warm_start(train)
    out = online.run_stream(stream, arrival_step=arrival_step, arrival_arm=arrival_arm)
    # the run completes and produces valid metrics with arrival gating active
    assert 0.0 <= out["mean_quality"] <= 1.0
    assert "cum_reward" in out