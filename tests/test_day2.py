"""Day 2 tests: metrics correctness, static-router behaviour, runner E2E.

The synthetic pool is constructed so that bigger models are strictly better
AND strictly more expensive on average — making the expected orderings
(oracle ≥ random-mix ≥ endpoints-line) provable, not accidental.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.eval import metrics
from tokenguard.eval.runner import EvalRunner
from tokenguard.routers.static import ConstantRouter, OracleRouter, RandomMixRouter

MODELS = ["small-1b", "mid-8b", "large-70b"]
TASKS = ["mmlu", "gsm8k", "hellaswag", "mbpp"]


@pytest.fixture()
def bench() -> RouterBench:
    rng = np.random.default_rng(1)
    n = 800
    data = {
        "sample_id": [f"s{i:05d}" for i in range(n)],
        "prompt": [f"Question {i}?" for i in range(n)],
        "eval_name": [TASKS[i % len(TASKS)] for i in range(n)],
    }
    for j, m in enumerate(MODELS):
        acc = 0.45 + 0.20 * j  # 0.45 / 0.65 / 0.85
        data[f"{m}|performance"] = rng.binomial(1, acc, size=n).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.8, 1.2, size=n) * 1e-4 * (10.0 ** j)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


@pytest.fixture()
def split(bench):
    return bench.split_random(test_size=0.25, seed=42)


# ------------------------------- metrics ----------------------------------- #
def test_pareto_front_removes_dominated_points() -> None:
    pts = pd.DataFrame(
        {"cost": [1.0, 2.0, 3.0, 4.0], "quality": [0.5, 0.4, 0.7, 0.7]}
    )
    front = metrics.pareto_front(pts)
    # (2.0, 0.4) dominated by (1.0, 0.5); (4.0, 0.7) dominated by (3.0, 0.7)
    assert list(front["cost"]) == [1.0, 3.0]
    assert (front["quality"].diff().dropna() > 0).all()


def test_quality_at_cost_interpolates_and_saturates() -> None:
    front = pd.DataFrame({"cost": [1.0, 3.0], "quality": [0.4, 0.8]})
    assert metrics.quality_at_cost(front, 2.0) == pytest.approx(0.6)
    assert metrics.quality_at_cost(front, 100.0) == pytest.approx(0.8)  # saturates
    assert metrics.quality_at_cost(front, 0.0) == pytest.approx(0.4)  # clamps


def test_aiq_of_flat_frontier_is_its_quality() -> None:
    front = pd.DataFrame({"cost": [1.0, 2.0], "quality": [0.6, 0.6]})
    assert metrics.aiq(front, 1.0, 2.0) == pytest.approx(0.6)


def test_aiq_rejects_degenerate_window() -> None:
    front = pd.DataFrame({"cost": [1.0], "quality": [0.6]})
    with pytest.raises(ValueError):
        metrics.aiq(front, 2.0, 2.0)


def test_apgr_of_straight_line_is_half() -> None:
    # A frontier exactly on the weak->strong segment recovers half the gap
    # on average (uniform budgets): integral of t over [0,1] = 0.5.
    weak, strong = (1.0, 0.4), (3.0, 0.8)
    line = pd.DataFrame({"cost": [1.0, 3.0], "quality": [0.4, 0.8]})
    assert metrics.apgr(line, weak, strong) == pytest.approx(0.5, abs=0.02)


# ---------------------------- static routers ------------------------------- #
def test_constant_routers_pick_expected_models(split) -> None:
    train, test = split
    cheap = ConstantRouter("cheapest").fit(train)
    best = ConstantRouter("best").fit(train)
    assert train.models[cheap.target_idx_] == "small-1b"
    assert train.models[best.target_idx_] == "large-70b"
    assert len(cheap.frontier(test)) == 1
    assert len(best.frontier(test)) == 1


def test_constant_router_rejects_unknown_strategy(split) -> None:
    train, _ = split
    with pytest.raises(ValueError):
        ConstantRouter("does-not-exist").fit(train)


def test_router_refuses_mismatched_model_set(split) -> None:
    train, test = split
    router = ConstantRouter("best").fit(train)
    shrunk = RouterBench(
        test.df.drop(columns=["perf::small-1b", "cost::small-1b"])
    )
    with pytest.raises(ValueError):
        router.route(shrunk, 0.0)


def test_random_mix_endpoints_match_constant_routers(split) -> None:
    train, test = split
    mix = RandomMixRouter(seed=0).fit(train)
    front = mix.frontier(test)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    # p=0 endpoint == always-cheapest point; p=1 endpoint == always-best point
    assert front.iloc[0]["cost"] == pytest.approx(weak["cost"], rel=1e-9)
    assert front.iloc[0]["quality"] == pytest.approx(weak["quality"], rel=1e-9)
    assert front.iloc[-1]["cost"] == pytest.approx(strong["cost"], rel=1e-9)
    assert front.iloc[-1]["quality"] == pytest.approx(strong["quality"], rel=1e-9)


def test_oracle_dominates_random_mix_on_aiq(split) -> None:
    train, test = split
    oracle_front = OracleRouter().fit(train).frontier(test)
    mix_front = RandomMixRouter(seed=0).fit(train).frontier(test)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])
    assert metrics.aiq(oracle_front, c_lo, c_hi) >= metrics.aiq(
        mix_front, c_lo, c_hi
    ) - 1e-9


def test_oracle_lambda_zero_equals_per_sample_max(split) -> None:
    train, test = split
    oracle = OracleRouter().fit(train)
    choices = oracle.route(test, lambda_cost=0.0)
    perf = test.perf_matrix()
    assert perf[np.arange(len(choices)), choices].mean() == pytest.approx(
        perf.max(axis=1).mean()
    )


# ------------------------------- runner ------------------------------------ #
def test_runner_end_to_end_writes_artifacts(split, tmp_path) -> None:
    train, test = split
    runner = EvalRunner(
        train, test,
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        tag="t",
    )
    summary = runner.run(
        [ConstantRouter("cheapest"), ConstantRouter("best"),
         RandomMixRouter(seed=0), OracleRouter()]
    )
    fig = runner.plot("test")

    assert fig.exists()
    assert (tmp_path / "results" / "t-summary.csv").exists()
    assert len(summary) == 4
    s = summary.set_index("router")
    assert s.loc["oracle", "aiq"] >= s.loc["random-mix", "aiq"] - 1e-9
    assert 0.3 <= s.loc["random-mix", "apgr"] <= 0.7
