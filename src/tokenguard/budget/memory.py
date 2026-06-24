"""MEDIUM tier — associative budget memory (Titans surprise update).

Maps a query key (router embedding) -> a correction on the initial budget prior
b0, learned online from realized (key, optimal-budget) pairs. Titans update with
momentum + data-dependent forgetting. Corrects b0 -> b_t.

TODO(week2): integrate with the controller; tune lr/forget on a held-out split.
"""
from __future__ import annotations
import numpy as np


class BudgetMemory:
    def __init__(self, key_dim: int, lr: float = 0.1, momentum: float = 0.9,
                 forget: float = 0.01, seed: int = 42):
        self.w = np.zeros(key_dim)          # linear key -> budget correction
        self.S = np.zeros(key_dim)
        self.lr, self.momentum, self.forget = lr, momentum, forget

    def predict(self, key: np.ndarray) -> float:
        return float(self.w @ key)

    def update(self, key: np.ndarray, target_correction: float) -> float:
        pred = float(self.w @ key)
        surprise = target_correction - pred
        grad = -surprise * key
        self.S = self.momentum * self.S - self.lr * grad
        self.w = (1 - self.forget) * self.w + self.S
        return abs(surprise)
