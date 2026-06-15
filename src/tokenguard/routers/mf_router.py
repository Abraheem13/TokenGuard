"""Matrix-factorisation router (RouteLLM-style learned baseline).

This is the *honest strong* learned baseline of the dissertation. It follows
the matrix-factorisation idea from RouteLLM (Ong et al., arXiv:2406.18665):
learn a low-rank model of the (query, model) -> quality relationship, then
route by predicted quality under the shared cost-aware decision rule.

Design (kept deliberately simple and fully reproducible, pure NumPy):

* Each **query** is represented by a fixed feature vector. We use a hashed
  bag-of-words / character n-gram embedding (the "hashing trick"), so there is
  no external tokuteniser dependency and the featuriser is deterministic and
  fast. Crucially this is a *content* feature, so the router generalises to
  unseen queries (unlike a per-sample lookup, which would be label leakage).
* Each **model** j has a learned latent vector ``v_j`` and bias ``b_j``.
* Predicted quality:  q̂_j(x) = sigmoid( (W x) · v_j + b_j ),
  where ``W`` projects the query features into the latent space.
* Training target: the binary/continuous RouterBench performance label for
  each (query, model) pair, optimised with full-batch gradient descent on the
  logistic loss. With K≈16 latent dims and a few hundred steps this fits in
  seconds on CPU over the whole training split.

Why this beats random and approaches the oracle's *shape*: it learns which
*families of queries* each model handles well (e.g. cheap models on easy MMLU,
GPT-4 on hard maths), which is exactly the structure a good router exploits.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench
from tokenguard.routers.base import Router


# --------------------------------------------------------------------------- #
# Deterministic query featuriser (hashing trick)                              #
# --------------------------------------------------------------------------- #
class HashingFeaturizer:
    """Map a string to a fixed-dim sparse-ish vector via feature hashing.

    Uses word unigrams + character 3-grams. Deterministic (seeded by Python's
    stable ``hash`` replacement: we hash with a fixed salt via ``blake2b``),
    so runs are reproducible across machines and Python invocations.
    """

    def __init__(self, dim: int = 512, salt: str = "tokenguard"):
        self.dim = dim
        self.salt = salt.encode()

    def _hash(self, token: str) -> int:
        import hashlib

        h = hashlib.blake2b(token.encode("utf-8"), salt=self.salt[:16], digest_size=8)
        return int.from_bytes(h.digest(), "little") % self.dim

    def transform(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text = text.lower()
            tokens = text.split()
            for w in tokens:
                X[i, self._hash("w:" + w)] += 1.0
            for k in range(len(text) - 2):  # char 3-grams
                X[i, self._hash("c:" + text[k : k + 3])] += 1.0
        # L2-normalise rows so long prompts don't dominate.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        np.divide(X, norms, out=X, where=norms > 0)
        return X


# --------------------------------------------------------------------------- #
# Matrix-factorisation router                                                 #
# --------------------------------------------------------------------------- #
class MatrixFactorizationRouter(Router):
    """Low-rank logistic predictor of per-model quality."""

    name = "matrix-factorization"

    def __init__(
        self,
        feat_dim: int = 512,
        latent_dim: int = 16,
        lr: float = 0.5,
        n_steps: int = 400,
        l2: float = 1e-4,
        seed: int = 42,
    ):
        super().__init__()
        self.featurizer = HashingFeaturizer(dim=feat_dim)
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_steps = n_steps
        self.l2 = l2
        self.seed = seed
        self.W_: np.ndarray | None = None   # (feat_dim, latent_dim)
        self.V_: np.ndarray | None = None   # (n_models, latent_dim)
        self.b_: np.ndarray | None = None   # (n_models,)

    # ------------------------------------------------------------------ #
    def fit(self, train: RouterBench) -> "MatrixFactorizationRouter":
        super().fit(train)
        rng = np.random.default_rng(self.seed)

        X = self.featurizer.transform(train.df["prompt"].tolist())  # (N, F)
        Y = train.perf_matrix()                                     # (N, M)
        n, f = X.shape
        m = Y.shape[1]
        k = self.latent_dim

        # Xavier-ish init.
        self.W_ = rng.normal(0, 1 / np.sqrt(f), size=(f, k)).astype(np.float32)
        self.V_ = rng.normal(0, 1 / np.sqrt(k), size=(m, k)).astype(np.float32)
        self.b_ = np.zeros(m, dtype=np.float32)

        for step in range(self.n_steps):
            Z = X @ self.W_                       # (N, K) query latents
            logits = Z @ self.V_.T + self.b_      # (N, M)
            P = _sigmoid(logits)                  # (N, M) predicted quality
            G = (P - Y) / n                       # (N, M) loss grad wrt logits

            grad_V = G.T @ Z + self.l2 * self.V_  # (M, K)
            grad_b = G.sum(axis=0)                # (M,)
            grad_W = X.T @ (G @ self.V_) + self.l2 * self.W_  # (F, K)

            self.V_ -= self.lr * grad_V
            self.b_ -= self.lr * grad_b
            self.W_ -= self.lr * grad_W

        return self

    # ------------------------------------------------------------------ #
    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        if self.W_ is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        X = self.featurizer.transform(bench.df["prompt"].tolist())
        return _sigmoid(X @ self.W_ @ self.V_.T + self.b_)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))