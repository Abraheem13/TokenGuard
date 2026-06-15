"""kNN-embedding router (non-parametric modern baseline, 2024-style).

Idea (as used in RouterBench, Hu et al. 2024, and many embedding-router
systems): represent each query by a semantic embedding; at inference, find the
k nearest training queries and predict each model's success as the
(distance-weighted) mean of that model's correctness over those neighbours.
Route under the shared cost-aware decision rule.

Why it is a strong baseline: it is non-parametric, so it captures local
query->model structure without the capacity/data-hunger issues of a fine-tuned
encoder, and it needs no training beyond building the index. Its weaknesses —
memory footprint (stores the whole training set) and the inability to *adapt
online without recomputing neighbours efficiently* — are exactly the axes on
which the Day-5 nested router differentiates.

Implementation: exact cosine-kNN via a single normalised matrix product
(embeddings are unit-norm, so cosine = dot product). For 30k training rows and
384-dim embeddings this is a ~30k x 384 matrix; brute-force search over a 7k
test split is a couple of seconds and avoids an FAISS dependency. A pluggable
``embedder`` lets tests inject a deterministic stub.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench
from tokenguard.routers.base import Router
from tokenguard.routers.embedding import SentenceEmbedder


class KNNRouter(Router):
    """Distance-weighted k-nearest-neighbour success predictor."""

    name = "knn-embedding"

    def __init__(self, k: int = 50, temperature: float = 0.1, embedder=None):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.embedder = embedder or SentenceEmbedder()
        self.train_emb_: np.ndarray | None = None   # (N_train, D)
        self.train_perf_: np.ndarray | None = None   # (N_train, M)

    def fit(self, train: RouterBench) -> "KNNRouter":
        super().fit(train)
        self.train_emb_ = self.embedder.encode(train.df["prompt"].tolist())
        self.train_perf_ = train.perf_matrix()
        return self

    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        if self.train_emb_ is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        q_emb = self.embedder.encode(bench.df["prompt"].tolist())   # (Nq, D)
        sims = q_emb @ self.train_emb_.T                             # (Nq, N_train) cosine
        k = min(self.k, sims.shape[1])

        # top-k neighbours per query
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]   # (Nq, k)
        row = np.arange(sims.shape[0])[:, None]
        top_sims = sims[row, top_idx]                               # (Nq, k)

        # distance(similarity)-weighted softmax over neighbours
        w = np.exp(top_sims / max(self.temperature, 1e-6))
        w /= w.sum(axis=1, keepdims=True)                          # (Nq, k)

        neigh_perf = self.train_perf_[top_idx]                     # (Nq, k, M)
        q_hat = np.einsum("nk,nkm->nm", w, neigh_perf)             # (Nq, M)
        return q_hat