"""L2 Titans-style neural memory, test-time surprise updates (Eqs. 13-14)."""
from __future__ import annotations
import numpy as np


class NeuralRoutingMemory:
    def __init__(self, key_dim, n_models, lr=0.05, momentum=0.9, forget=0.01, seed=42):
        self.key_dim, self.n_models = key_dim, n_models
        self.lr, self.momentum, self.forget = lr, momentum, forget
        self.rng = np.random.default_rng(seed)
        self.M = np.zeros((n_models, key_dim))
        self.S = np.zeros_like(self.M)

    def predict(self, key):
        return self.M @ key

    def update(self, key, model, reward):
        pred = float(self.M[model] @ key)
        surprise = reward - pred
        grad = -surprise * key
        self.S[model] = self.momentum * self.S[model] - self.lr * grad
        self.M[model] = (1.0 - self.forget) * self.M[model] + self.S[model]
        return abs(surprise)
