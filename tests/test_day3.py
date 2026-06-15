"""Day 3 tests: matrix-factorisation router (full) and BERT router glue.

The MF router is pure NumPy, so it is tested end-to-end here. The BERT router
depends on torch/transformers; rather than download a model in CI, we test its
*routing/decision glue* by subclassing it with a deterministic stub
``predict_quality`` — this verifies the Router contract (fit-state, decision
rule, frontier shape) without the heavy dependency. The real encoder path is
exercised on the user's machine via ``scripts/day3_learned_baselines.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.eval import metrics
from tokenguard.routers.bert_router import BertClassifierRouter
from tokenguard.routers.mf_router import HashingFeaturizer, MatrixFactorizationRouter
from tokenguard.routers.static import ConstantRouter, RandomMixRouter

TASKS = ["mmlu", "gsm8k", "code", "qa"]


@pytest.fixture()
def bench() -> RouterBench:
    """Synthetic pool with *learnable structure*: each model is best on the
    task whose name it carries, so a content-aware router can beat random."""
    rng = np.random.default_rng(3)
    n = 1200
    models = ["m-mmlu", "m-gsm8k", "m-code", "m-qa"]
    prompts, evals = [], []
    for i in range(n):
        t = TASKS[i % len(TASKS)]
        evals.append(t)
        prompts.append(f"this is a {t} question about topic {i % 7}")
    data = {"sample_id": [f"s{i}" for i in range(n)], "prompt": prompts, "eval_name": evals}
    for j, m in enumerate(models):
        specialty = TASKS[j]
        # high accuracy on its specialty task, low elsewhere
        acc = np.array([0.9 if e == specialty else 0.3 for e in evals])
        data[f"{m}|performance"] = rng.binomial(1, acc).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.8, 1.2, n) * 1e-4 * (j + 1)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


# ----------------------------- featuriser ---------------------------------- #
def test_hashing_featurizer_is_deterministic() -> None:
    f = HashingFeaturizer(dim=128)
    a = f.transform(["hello world", "foo bar"])
    b = f.transform(["hello world", "foo bar"])
    assert np.array_equal(a, b)
    assert a.shape == (2, 128)


def test_hashing_featurizer_rows_are_unit_norm() -> None:
    f = HashingFeaturizer(dim=128)
    X = f.transform(["some longer piece of text here", "x"])
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# --------------------------- MF router ------------------------------------- #
def test_mf_router_fits_and_predicts_valid_probabilities(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    mf = MatrixFactorizationRouter(n_steps=200, seed=42).fit(train)
    q = mf.predict_quality(test)
    assert q.shape == (len(test.df), len(test.models))
    assert (q >= 0).all() and (q <= 1).all()


def test_mf_router_is_reproducible(bench) -> None:
    train, _ = bench.split_random(0.25, 42)
    q1 = MatrixFactorizationRouter(n_steps=100, seed=7).fit(train).predict_quality(train)
    q2 = MatrixFactorizationRouter(n_steps=100, seed=7).fit(train).predict_quality(train)
    assert np.allclose(q1, q2)


def test_mf_router_beats_random_mix_on_structured_data(bench) -> None:
    """The core Day-3 claim: a learned router exploits query->model structure."""
    train, test = bench.split_random(0.25, 42)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])

    mf_front = MatrixFactorizationRouter(n_steps=300, seed=42).fit(train).frontier(test)
    rnd_front = RandomMixRouter(seed=42).fit(train).frontier(test)

    mf_aiq = metrics.aiq(mf_front, c_lo, c_hi)
    rnd_aiq = metrics.aiq(rnd_front, c_lo, c_hi)
    assert mf_aiq > rnd_aiq, f"MF AIQ {mf_aiq:.4f} should exceed random {rnd_aiq:.4f}"


# --------------------------- BERT router glue ------------------------------ #
class _StubBert(BertClassifierRouter):
    """BERT router with the heavy encoder replaced by a deterministic stub,
    so the Router-contract glue is testable without torch/transformers."""

    name = "bert-stub"

    def fit(self, train):  # noqa: D401 — bypass real training
        # Reuse the base fit() cost-estimate logic without building the encoder.
        from tokenguard.routers.base import Router

        Router.fit(self, train)
        rng = np.random.default_rng(self.seed)
        self._proto = rng.random((len(train.models), 8)).astype(np.float32)
        self._feat = HashingFeaturizer(dim=8)
        return self

    def predict_quality(self, bench):
        X = self._feat.transform(bench.df["prompt"].tolist())
        from tokenguard.routers.mf_router import _sigmoid

        return _sigmoid(X @ self._proto.T)


def test_bert_router_glue_obeys_router_contract(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    router = _StubBert().fit(train)
    q = router.predict_quality(test)
    assert q.shape == (len(test.df), len(test.models))
    choices = router.route(test, lambda_cost=0.5)
    assert choices.shape == (len(test.df),)
    assert choices.min() >= 0 and choices.max() < len(test.models)
    front = router.frontier(test)
    assert {"cost", "quality"}.issubset(front.columns)


def test_bert_router_requires_fit_before_predict(bench) -> None:
    with pytest.raises(RuntimeError):
        BertClassifierRouter().predict_quality(bench)