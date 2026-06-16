"""Contrastive router (RouterDC-style) — TokenGuard's core modern router.

This is the model the dissertation's online/nested extensions (Day 5) build on,
and the replacement for the rejected BERT classifier. It follows RouterDC
(Chen et al., NeurIPS 2024): represent each query with a frozen encoder
(Qwen3-0.6B, mean-pooled), project it into a routing space, and learn one
embedding per candidate LLM. A query is routed by similarity between its
projected embedding and each LLM embedding, under the shared cost-aware rule.

Training objective — *dual contrastive loss*:

1. **Sample-LLM contrastive (InfoNCE).** For each query, the LLMs that answered
   it correctly are positives, the rest negatives. We maximise the softmax
   probability mass on positives over a temperature-scaled similarity, pulling
   the query toward the models that suit it and away from those that do not.

2. **Sample-sample contrastive.** Queries with similar LLM-success patterns are
   pulled together in routing space and dissimilar ones pushed apart, using the
   cosine of their binary success vectors as the in-batch target similarity.
   This regularises the geometry and improves generalisation to unseen queries.

Only a projection ``P`` and the per-LLM embedding table ``E`` are trained (the
encoder is frozen), so training is fast and CPU/MPS/GPU-friendly. We use
PyTorch autograd for the training so the (otherwise error-prone) gradients are
exact; a calibration head then maps similarities to **calibrated success
probabilities**, which the cost-aware decision rule (q̂ − λ·c) requires.

The encoder is injected (any object with ``encode(list[str]) -> ndarray``), so
tests use a deterministic stub and the real run uses a cached Qwen encoder.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench
from tokenguard.routers.base import Router
from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.routers.contrastive")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ContrastiveRouter(Router):
    """Frozen encoder + learned projection + per-LLM embeddings (RouterDC)."""

    name = "contrastive"

    def __init__(
        self,
        encoder,
        proj_dim: int = 128,
        temperature: float = 0.1,
        lr: float = 0.01,
        n_epochs: int = 60,
        batch_size: int = 512,
        w_sample_llm: float = 1.0,
        w_sample_sample: float = 0.2,
        w_bce: float = 2.0,
        weight_decay: float = 1e-5,
        seed: int = 42,
    ):
        super().__init__()
        self.encoder = encoder
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.w_sample_llm = w_sample_llm
        self.w_sample_sample = w_sample_sample
        self.w_bce = w_bce
        self.weight_decay = weight_decay
        self.seed = seed
        self.P_: np.ndarray | None = None   # (enc_dim, proj_dim)
        self.E_: np.ndarray | None = None   # (n_models, proj_dim)
        self.a_: np.ndarray | None = None   # (n_models,) calibration scale
        self.b_: np.ndarray | None = None   # (n_models,) calibration bias

    # ------------------------------------------------------------------ #
    def fit(self, train: RouterBench) -> "ContrastiveRouter":
        super().fit(train)
        X = self.encoder.encode(train.df["prompt"].tolist()).astype(np.float32)
        Y = train.perf_matrix().astype(np.float32)
        self._train_torch(X, Y)
        return self

    # ------------------------------------------------------------------ #
    def _train_torch(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train P, E, and calibration (a, b) with exact autograd gradients.

        The objective combines three terms:
          * BCE — per-model binary prediction P(correct)=sigmoid(a·sim+b) vs
            the true success labels. This is the term that matches matrix
            factorisation's accuracy: it learns each model's exact success
            probability, not merely *which* models are correct.
          * sample-LLM InfoNCE — sharpens the routing geometry toward the
            models that suit each query.
          * sample-sample — clusters queries with similar success patterns.
        Training a, b jointly (rather than as a post-hoc fit) lets the geometry
        and the calibrated probabilities co-adapt, which closes the gap to MF
        while keeping the contrastive structure the Day-5 online router needs.
        """
        import torch

        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        n, enc_dim = X.shape
        m = Y.shape[1]

        Xt = torch.from_numpy(X)
        Yt = torch.from_numpy(Y)
        P = torch.nn.Parameter(torch.randn(enc_dim, self.proj_dim, generator=g) / enc_dim**0.5)
        E = torch.nn.Parameter(torch.randn(m, self.proj_dim, generator=g) / self.proj_dim**0.5)
        a = torch.nn.Parameter(torch.full((m,), 4.0))   # calibration scale
        b = torch.nn.Parameter(torch.zeros(m))          # calibration bias
        opt = torch.optim.Adam([P, E, a, b], lr=self.lr, weight_decay=self.weight_decay)

        idx_all = torch.arange(n)
        for epoch in range(self.n_epochs):
            perm = idx_all[torch.randperm(n, generator=g)]
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = Xt[idx], Yt[idx]
                opt.zero_grad()
                loss = self._loss_torch(torch, xb, yb, P, E, a, b)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach()) * len(idx)
            if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                logger.info("Contrastive epoch %d/%d — loss %.4f",
                            epoch + 1, self.n_epochs, epoch_loss / n)

        self.P_ = P.detach().cpu().numpy().astype(np.float32)
        self.E_ = E.detach().cpu().numpy().astype(np.float32)
        self.a_ = a.detach().cpu().numpy().astype(np.float32)
        self.b_ = b.detach().cpu().numpy().astype(np.float32)

    def _loss_torch(self, torch, xb, yb, P, E, a, b):
        Q = torch.nn.functional.normalize(xb @ P, dim=1)     # (b, d)
        En = torch.nn.functional.normalize(E, dim=1)         # (m, d)
        sim = Q @ En.T                                       # (b, m) cosine
        S = sim / self.temperature                           # temperature-scaled

        # (0) BCE — calibrated per-model success prediction (MF-style accuracy)
        logits = a * sim + b                                 # (b, m)
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)

        # (1) sample-LLM InfoNCE: target = uniform over correct models
        pos = (yb > 0.5).float()
        has_pos = pos.sum(dim=1) > 0
        logp = torch.log_softmax(S, dim=1)
        target = pos / pos.sum(dim=1, keepdim=True).clamp(min=1.0)
        loss_llm = -(target * logp).sum(dim=1)
        loss_llm = (loss_llm * has_pos).sum() / has_pos.sum().clamp(min=1)

        # (2) sample-sample: match query cosine to success-pattern cosine
        Yn = torch.nn.functional.normalize(yb, dim=1)
        T = Yn @ Yn.T
        Qsim = Q @ Q.T
        loss_ss = ((Qsim - T) ** 2).mean()

        return (
            self.w_bce * loss_bce
            + self.w_sample_llm * loss_llm
            + self.w_sample_sample * loss_ss
        )

    # ------------------------------------------------------------------ #
    def _similarities(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarities between projected queries and LLM embeddings."""
        Q = _l2norm(X @ self.P_)
        E = _l2norm(self.E_)
        return Q @ E.T

    # ------------------------------------------------------------------ #
    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        X = self.encoder.encode(bench.df["prompt"].tolist()).astype(np.float32)
        S = self._similarities(X)
        return _sigmoid(self.a_ * S + self.b_)


def _l2norm(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.clip(n, 1e-12, None)