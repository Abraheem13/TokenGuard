"""Sliding-window UCB adapter for the halting threshold tau (REFRAIN-style).

Picks tau from a small grid to maximise reward = accuracy - mu * tokens, over a
sliding window so it tracks drift. Reward feeds the MEDIUM tier, closing the
nested loop.

TODO(week2): wire reward from realized (correct?, tokens) per query.
"""
from __future__ import annotations
import numpy as np


class SWUCBThreshold:
    def __init__(self, taus=(0.05, 0.1, 0.15, 0.2, 0.3), window: int = 200,
                 c: float = 1.0, seed: int = 42):
        self.taus = list(taus)
        self.window = window
        self.c = c
        self.hist = {i: [] for i in range(len(self.taus))}
        self._t = 0

    def select(self) -> tuple[int, float]:
        self._t += 1
        best, best_score = 0, -1e9
        for i in range(len(self.taus)):
            h = self.hist[i][-self.window:]
            if not h:
                return i, self.taus[i]            # try each arm once
            mean = float(np.mean(h))
            bonus = self.c * np.sqrt(np.log(self._t + 1) / len(h))
            score = mean + bonus
            if score > best_score:
                best, best_score = i, score
        return best, self.taus[best]

    def update(self, arm: int, reward: float) -> None:
        self.hist[arm].append(reward)
