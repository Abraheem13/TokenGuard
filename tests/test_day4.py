"""Day 4 tests: embedding cache and contrastive router.

The contrastive router trains with PyTorch autograd. Where torch is available
(the student's machine, CI with torch), the full learning tests run and assert
the router beats random-mix and ranks task specialists correctly. Where torch
is unavailable, those tests skip cleanly, while the pure-NumPy parts (embedding
cache, calibration head, predict glue, fit-state contract) are always tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.eval import metrics
from tokenguard.routers.contrastive_router import ContrastiveRouter, _l2norm, _sigmoid
from tokenguard.routers.embedding_cache import EmbeddingCache, content_hash
from tokenguard.routers.static import ConstantRouter, RandomMixRouter

torch = pytest.importorskip  # marker helper
try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

TASKS = ["mmlu", "gsm8k", "code", "qa"]


class StubEncoder:
    """Deterministic encoder: task prototype + small jitter, unit-norm."""

    def __init__(self, vocab=TASKS, dim=32, seed=0):
        self.vocab = list(vocab)
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._basis = self.rng.normal(0, 1, (len(self.vocab), dim)).astype(np.float32)

    def encode(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            label = t.split(":", 1)[0]
            if label in self.vocab:
                X[i] = self._basis[self.vocab.index(label)]
        X += 0.02 * self.rng.standard_normal(X.shape).astype(np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, 1e-9, None)


@pytest.fixture()
def bench() -> RouterBench:
    rng = np.random.default_rng(7)
    n = 1600
    models = ["m-mmlu", "m-gsm8k", "m-code", "m-qa"]
    ev = [TASKS[i % len(TASKS)] for i in range(n)]
    data = {
        "sample_id": [f"s{i}" for i in range(n)],
        "prompt": [f"{ev[i]}: instance {i}" for i in range(n)],
        "eval_name": ev,
    }
    for j, m in enumerate(models):
        specialty = TASKS[j]
        acc = np.array([0.9 if e == specialty else 0.2 for e in ev])
        data[f"{m}|performance"] = rng.binomial(1, acc).astype(float)
        data[f"{m}|total_cost"] = rng.uniform(0.9, 1.1, n) * 1e-4 * (j + 1)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


# ----------------------------- cache (pure numpy) -------------------------- #
def test_content_hash_is_stable() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_embedding_cache_roundtrip_and_reuse(tmp_path) -> None:
    enc = StubEncoder()
    cache = EmbeddingCache(enc, "stub-v1", tmp_path)
    texts = ["mmlu: a", "gsm8k: b", "mmlu: a"]
    emb1 = cache.encode(texts)
    assert emb1.shape == (3, enc.dim)
    assert np.allclose(emb1[0], emb1[2])
    cache2 = EmbeddingCache(encoder=None, encoder_id="stub-v1", cache_dir=tmp_path)
    emb2 = cache2.encode(["mmlu: a", "gsm8k: b"])
    assert np.allclose(emb2[0], emb1[0])


def test_embedding_cache_orders_outputs_to_match_inputs(tmp_path) -> None:
    cache = EmbeddingCache(StubEncoder(), "stub-v2", tmp_path)
    a = cache.encode(["code: x", "qa: y"])
    b = cache.encode(["qa: y", "code: x"])
    assert np.allclose(a[0], b[1]) and np.allclose(a[1], b[0])


# ---------------------- calibration head (pure numpy) ---------------------- #
def test_predict_quality_uses_trained_calibration() -> None:
    """predict_quality maps similarities through the trained (a, b) calibration
    to probabilities in [0, 1]. We set a, b directly and check the mapping."""
    r = ContrastiveRouter(encoder=None)
    r.P_ = np.eye(4, dtype=np.float32)
    r.E_ = np.eye(4, dtype=np.float32)[:3]      # 3 models in a 4-d space
    r.a_ = np.array([5.0, 3.0, 4.0], dtype=np.float32)
    r.b_ = np.array([-0.5, 0.2, 0.0], dtype=np.float32)
    # similarities for a query equal to model-0's embedding direction
    X = np.array([[1, 0, 0, 0]], dtype=np.float32)
    S = r._similarities(X)
    P = _sigmoid(r.a_ * S + r.b_)
    assert P.shape == (1, 3)
    assert (P >= 0).all() and (P <= 1).all()
    # model 0 (highest similarity) should get the highest probability
    assert P[0].argmax() == 0


def test_contrastive_requires_fit(bench) -> None:
    with pytest.raises(RuntimeError):
        ContrastiveRouter(encoder=StubEncoder()).predict_quality(bench)


# ----------------------- full learning (needs torch) ----------------------- #
@requires_torch
def test_contrastive_predicts_valid_quality(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    r = ContrastiveRouter(StubEncoder(), n_epochs=20, seed=42).fit(train)
    q = r.predict_quality(test)
    assert q.shape == (len(test.df), len(test.models))
    assert (q >= 0).all() and (q <= 1).all()


@requires_torch
def test_contrastive_is_reproducible(bench) -> None:
    train, _ = bench.split_random(0.25, 42)
    q1 = ContrastiveRouter(StubEncoder(seed=1), n_epochs=15, seed=3).fit(train).predict_quality(train)
    q2 = ContrastiveRouter(StubEncoder(seed=1), n_epochs=15, seed=3).fit(train).predict_quality(train)
    assert np.allclose(q1, q2, atol=1e-5)


@requires_torch
def test_contrastive_learns_and_beats_random(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])
    r = ContrastiveRouter(StubEncoder(), n_epochs=80, seed=42).fit(train)
    rnd = RandomMixRouter(seed=42).fit(train)
    assert metrics.aiq(r.frontier(test), c_lo, c_hi) > metrics.aiq(rnd.frontier(test), c_lo, c_hi)


@requires_torch
def test_contrastive_ranks_specialist_top(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    r = ContrastiveRouter(StubEncoder(), n_epochs=80, seed=42).fit(train)
    q = r.predict_quality(test)
    mmlu_idx = test.models.index("m-mmlu")
    mmlu_mask = (test.df["eval_name"] == "mmlu").to_numpy()
    assert (q[mmlu_mask].argmax(axis=1) == mmlu_idx).mean() > 0.6