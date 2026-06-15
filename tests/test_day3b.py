"""Tests for the modern baselines (kNN-embedding and cascade routers).

A deterministic *stub embedder* replaces Sentence-Transformers so the suite is
fast and network-free. The stub embeds each prompt by its task label, so
queries from the same task are identical in embedding space — this gives the
kNN router perfect local structure to exploit, making the expected orderings
(kNN > random-mix; cascade frontier well-formed) provable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tokenguard.data.routerbench import RouterBench, canonicalise, detect_schema
from tokenguard.eval import metrics
from tokenguard.routers.cascade_router import CascadeRouter
from tokenguard.routers.knn_router import KNNRouter
from tokenguard.routers.static import ConstantRouter, RandomMixRouter

TASKS = ["mmlu", "gsm8k", "code", "qa"]


class StubEmbedder:
    """Deterministic embedder: one-hot over a fixed task vocabulary.

    Embeds by the leading token of the prompt ("<task>:"), so same-task queries
    collapse to the same point. Unit-norm, like the real encoder.
    """

    def __init__(self, vocab=TASKS):
        self.vocab = list(vocab)

    def encode(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            label = t.split(":", 1)[0]
            if label in self.vocab:
                X[i, self.vocab.index(label)] = 1.0
        return X


@pytest.fixture()
def bench() -> RouterBench:
    rng = np.random.default_rng(5)
    n = 1200
    models = ["m-mmlu", "m-gsm8k", "m-code", "m-qa"]
    prompts, evals = [], []
    for i in range(n):
        t = TASKS[i % len(TASKS)]
        evals.append(t)
        prompts.append(f"{t}: question instance {i}")
    data = {"sample_id": [f"s{i}" for i in range(n)], "prompt": prompts, "eval_name": evals}
    for j, m in enumerate(models):
        specialty = TASKS[j]
        acc = np.array([0.9 if e == specialty else 0.25 for e in evals])
        data[f"{m}|performance"] = rng.binomial(1, acc).astype(float)
        # cheapest model is the mmlu specialist, dearest is the qa specialist
        data[f"{m}|total_cost"] = rng.uniform(0.9, 1.1, n) * 1e-4 * (j + 1)
    raw = pd.DataFrame(data)
    return RouterBench(canonicalise(raw, detect_schema(raw)))


# -------------------------------- kNN -------------------------------------- #
def test_knn_predicts_valid_probabilities(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    knn = KNNRouter(k=20, embedder=StubEmbedder()).fit(train)
    q = knn.predict_quality(test)
    assert q.shape == (len(test.df), len(test.models))
    assert (q >= 0).all() and (q <= 1).all()


def test_knn_recovers_specialist_structure(bench) -> None:
    """With task-pure embeddings, kNN should rank each task's specialist top."""
    train, test = bench.split_random(0.25, 42)
    knn = KNNRouter(k=30, embedder=StubEmbedder()).fit(train)
    q = knn.predict_quality(test)
    # models are stored alphabetically; resolve the mmlu specialist by name
    mmlu_idx = test.models.index("m-mmlu")
    mmlu_mask = (test.df["eval_name"] == "mmlu").to_numpy()
    # for mmlu queries, the mmlu specialist should be the top-predicted model
    assert (q[mmlu_mask].argmax(axis=1) == mmlu_idx).mean() > 0.95


def test_knn_beats_random_mix_on_aiq(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    weak = ConstantRouter("cheapest").fit(train).frontier(test).iloc[0]
    strong = ConstantRouter("best").fit(train).frontier(test).iloc[0]
    c_lo, c_hi = float(weak["cost"]), float(strong["cost"])
    knn_front = KNNRouter(k=30, embedder=StubEmbedder()).fit(train).frontier(test)
    rnd_front = RandomMixRouter(seed=42).fit(train).frontier(test)
    assert metrics.aiq(knn_front, c_lo, c_hi) > metrics.aiq(rnd_front, c_lo, c_hi)


# ------------------------------ cascade ------------------------------------ #
def test_cascade_frontier_is_monotone_in_cost(bench) -> None:
    train, test = bench.split_random(0.25, 42)
    casc = CascadeRouter(k=30, embedder=StubEmbedder()).fit(train)
    front = casc.frontier(test)
    assert (front["cost"].diff().dropna() >= -1e-9).all()
    assert (front["quality"].diff().dropna() >= -1e-9).all()


def test_cascade_charges_for_every_model_tried(bench) -> None:
    """At τ=1 every query escalates to the last model, paying for all tried."""
    train, test = bench.split_random(0.25, 42)
    casc = CascadeRouter(k=30, embedder=StubEmbedder()).fit(train)
    conf = casc._confidence.predict_quality(test)
    cost = test.cost_matrix()
    chosen, paid = casc._run_cascade(conf, cost, casc.cost_order_, tau=1.0)
    # everyone ends on the dearest model, having paid the full chain
    assert (chosen == casc.cost_order_[-1]).all()
    full_chain = cost[:, casc.cost_order_].sum(axis=1)
    assert np.allclose(paid, full_chain)


def test_cascade_low_threshold_is_cheap(bench) -> None:
    """At τ=0 nothing escalates: everyone stays on the cheapest model."""
    train, test = bench.split_random(0.25, 42)
    casc = CascadeRouter(k=30, embedder=StubEmbedder()).fit(train)
    conf = casc._confidence.predict_quality(test)
    cost = test.cost_matrix()
    chosen, paid = casc._run_cascade(conf, cost, casc.cost_order_, tau=0.0)
    assert (chosen == casc.cost_order_[0]).all()
    assert np.allclose(paid, cost[:, casc.cost_order_[0]])