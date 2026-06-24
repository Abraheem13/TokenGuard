"""FAST tier — per-step surprise / uncertainty signal driving halting.

surprise_t = predictive uncertainty of reasoning step t (entropy or negative
answer-log-likelihood), smoothed by a momentum EMA (Titans Eqs. 13-14 analogue:
S_t = eta * S_{t-1} + (1-eta) * u_t). Halt when the smoothed signal has
converged below threshold tau for `patience` steps.

TODO(week1): compute u_t from the model's step logprobs/entropy in llm/.
"""
from __future__ import annotations
import numpy as np


class SurpriseHalter:
    def __init__(self, tau: float = 0.15, eta: float = 0.7, patience: int = 2,
                 min_steps: int = 1, max_steps: int = 64):
        self.tau, self.eta, self.patience = tau, eta, patience
        self.min_steps, self.max_steps = min_steps, max_steps
        self.reset()

    def reset(self) -> None:
        self._ema = None
        self._below = 0
        self._t = 0

    def observe(self, uncertainty: float) -> bool:
        """Feed step uncertainty; return True if generation should HALT now."""
        self._t += 1
        self._ema = uncertainty if self._ema is None else (
            self.eta * self._ema + (1 - self.eta) * uncertainty)
        self._below = self._below + 1 if self._ema <= self.tau else 0
        if self._t < self.min_steps:
            return False
        if self._t >= self.max_steps:
            return True
        return self._below >= self.patience

    @property
    def smoothed(self) -> float:
        return float(self._ema if self._ema is not None else 0.0)
