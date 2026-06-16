"""Replay buffer for the online/nested router.

Stores observed (embedding, chosen-arm, reward, full-reward-vector) tuples as
the request stream is processed. It serves two roles in the Nested Learning
design:

* the FAST level reads the most recent reward to update the bandit head every
  request;
* the SLOW level samples from the whole buffer to periodically refit the
  calibration / encoder head, which is what lets the router consolidate
  knowledge and resist forgetting under distribution shift.

The buffer is a fixed-capacity ring (oldest entries overwritten), with a
recency-weighted sampler so slow updates emphasise recent data without wholly
forgetting the past — mirroring the Continuum-Memory intuition that different
levels retain information over different horizons.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-capacity ring buffer of online routing observations."""

    def __init__(self, capacity: int, emb_dim: int, n_models: int, seed: int = 0):
        self.capacity = capacity
        self.emb_dim = emb_dim
        self.n_models = n_models
        self.rng = np.random.default_rng(seed)
        self._emb = np.zeros((capacity, emb_dim), dtype=np.float32)
        self._perf = np.zeros((capacity, n_models), dtype=np.float32)
        self._size = 0
        self._ptr = 0

    def add(self, emb: np.ndarray, perf_vec: np.ndarray) -> None:
        """Add one observation: query embedding + its full per-model success."""
        self._emb[self._ptr] = emb
        self._perf[self._ptr] = perf_vec
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, n: int, recency_weighted: bool = True):
        """Sample (emb, perf) of up to ``n`` past observations.

        With ``recency_weighted``, more recent entries are likelier to be
        drawn (geometric-ish weighting), so slow updates track drift while
        still revisiting old data for consolidation.
        """
        if self._size == 0:
            raise ValueError("replay buffer is empty")
        n = min(n, self._size)
        if recency_weighted:
            # age 0 = most recent; weight decays with age
            ages = (self._ptr - 1 - np.arange(self._size)) % self.capacity
            w = 0.999 ** ages
            w = w / w.sum()
            idx = self.rng.choice(self._size, size=n, replace=False, p=w)
        else:
            idx = self.rng.choice(self._size, size=n, replace=False)
        return self._emb[idx].copy(), self._perf[idx].copy()

    def all(self):
        """Return all stored (emb, perf) without copying semantics guarantees."""
        return self._emb[: self._size], self._perf[: self._size]