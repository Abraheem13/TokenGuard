"""Day 1 tests.

The loader is tested against a synthetic fixture that mimics the published
RouterBench wire format (``model|metric`` columns), so the parsing logic is
verified *before* the real download — and continues to be tested in CI
without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.config import load_config
from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.utils.seed import set_global_seed

MODELS = ["gpt-4-1106-preview", "claude-instant-v1", "mistral-8x7b-chat"]
TASKS = ["mmlu", "gsm8k", "hellaswag"]


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """Synthetic raw frame in RouterBench wire format (240 rows)."""
    rng = np.random.default_rng(0)
    n = 240
    data = {
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "prompt": [f"Question number {i}?" for i in range(n)],
        "eval_name": [TASKS[i % len(TASKS)] for i in range(n)],
    }
    for j, m in enumerate(MODELS):
        # Bigger models: higher performance, higher cost (realistic ordering).
        data[f"{m}|performance"] = rng.binomial(1, 0.55 + 0.15 * j, size=n).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.0001, 0.001, size=n) * (j + 1) ** 2
    return pd.DataFrame(data)


@pytest.fixture()
def bench(raw_frame: pd.DataFrame) -> RouterBench:
    schema = detect_schema(raw_frame)
    return RouterBench(canonicalise(raw_frame, schema))


# --------------------------- schema detection ------------------------------ #
def test_detect_schema_finds_models_and_columns(raw_frame: pd.DataFrame) -> None:
    schema = detect_schema(raw_frame)
    assert set(schema.models) == set(MODELS)
    assert schema.perf_suffix == "performance"
    assert schema.cost_suffix == "total_cost"
    assert schema.prompt_col == "prompt"
    assert schema.eval_col == "eval_name"


def test_detect_schema_fails_loudly_without_metric_columns() -> None:
    bad = pd.DataFrame({"prompt": ["x"], "eval_name": ["mmlu"]})
    with pytest.raises(ValueError):
        detect_schema(bad)


def test_metadata_columns_are_not_mistaken_for_models(raw_frame: pd.DataFrame) -> None:
    # A column with the separator but only one metric must not create a model.
    raw_frame["oracle|performance"] = 1.0  # no matching oracle|total_cost
    schema = detect_schema(raw_frame)
    assert "oracle" not in schema.models


# --------------------------- canonicalisation ------------------------------ #
def test_canonical_frame_has_no_nans_and_correct_shape(bench: RouterBench) -> None:
    assert not bench.df.isna().any().any()
    assert bench.perf_matrix().shape == (len(bench.df), len(MODELS))
    assert bench.cost_matrix().shape == (len(bench.df), len(MODELS))


def test_rows_with_missing_values_are_dropped(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, f"{MODELS[0]}|performance"] = np.nan
    schema = detect_schema(raw_frame)
    canonical = canonicalise(raw_frame, schema)
    assert len(canonical) == len(raw_frame) - 1


# --------------------------------- splits ---------------------------------- #
def test_random_split_is_stratified_and_disjoint(bench: RouterBench) -> None:
    train, test = bench.split_random(test_size=0.2, seed=42)
    assert len(train.df) + len(test.df) == len(bench.df)
    assert set(train.df.sample_id).isdisjoint(set(test.df.sample_id))
    # every task appears on both sides (stratification)
    assert set(train.tasks) == set(TASKS) and set(test.tasks) == set(TASKS)


def test_random_split_is_reproducible(bench: RouterBench) -> None:
    _, t1 = bench.split_random(0.2, seed=42)
    _, t2 = bench.split_random(0.2, seed=42)
    assert list(t1.df.sample_id) == list(t2.df.sample_id)


def test_leave_one_task_out_split(bench: RouterBench) -> None:
    train, test = bench.split_leave_one_task_out("gsm8k")
    assert set(test.df.eval_name) == {"gsm8k"}
    assert "gsm8k" not in set(train.df.eval_name)


def test_leave_one_task_out_rejects_unknown_task(bench: RouterBench) -> None:
    with pytest.raises(ValueError):
        bench.split_leave_one_task_out("not-a-task")


# ------------------------------ summaries ---------------------------------- #
def test_oracle_dominates_every_single_model(bench: RouterBench) -> None:
    oracle_perf = bench.oracle_stats()["oracle_perf"]
    assert oracle_perf >= bench.summary()["mean_perf"].max() - 1e-12


# ------------------------------- config ------------------------------------ #
def test_load_default_config_from_repo_root() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.experiment.seed == 42
    assert cfg.data.hf_repo_id == "withmartian/routerbench"
    assert 0.0 < cfg.data.test_size < 1.0
    assert cfg.router.slow_update_every > 0


def test_config_rejects_unknown_keys(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment:\n  sedd: 1\n")  # typo must fail loudly
    with pytest.raises(KeyError):
        load_config("configs/default.yaml", override_path=bad)


# -------------------------------- seeding ---------------------------------- #
def test_global_seed_makes_numpy_deterministic() -> None:
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)
