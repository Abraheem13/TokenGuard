"""Continuum Memory System orchestrator — wires L1 + L2 (+ L3) (CMS Eq. 70-71).

Owns the mid neural memory (L2) and the surprise gate, and blends the fast
prediction (L1, owned by the router) with the mid memory's prediction. The blend
weight on L2 grows with how much evidence the memory has accumulated for the
current key neighbourhood, so the memory only influences routing once it has
learned something — preventing an untrained memory from hurting early decisions.
"""

from __future__ import annotations

import numpy as np

from tokenguard.memory.neural_memory import NeuralRoutingMemory
from tokenguard.memory.surprise import SurpriseGate
from tokenguard.memory.consolidation import SlowConsolidator


class ContinuumMemoryRouter:
    def __init__(self, key_dim: int, n_models: int, c2: int = 1, c3: int = 4000,
                 mid_weight: float = 0.5, lr: float = 0.05,
                 surprise_scale: float = 2.0, seed: int = 42):
        self.mid = NeuralRoutingMemory(key_dim, n_models, lr=lr,
                                       surprise_scale=surprise_scale, seed=seed)
        self.gate = SurpriseGate()
        self.slow = SlowConsolidator(every=c3)
        self.c2 = c2
        self.mid_weight = mid_weight
        self._step = 0
        self._writes = 0

    # ------------------------------------------------------------------ #
    def predict(self, key: np.ndarray, fast_pred: np.ndarray) -> np.ndarray:
        """Blend L1 (fast) with L2 (mid). The mid weight ramps with the number
        of writes so an untrained memory does not perturb early routing."""
        if self._writes == 0:
            return fast_pred
        ramp = min(1.0, self._writes / 500.0)          # warm-up ramp
        w = self.mid_weight * ramp
        return (1.0 - w) * fast_pred + w * self.mid.predict(key)

    def observe(self, key: np.ndarray, model: int, reward: float,
                recurrent: bool = True) -> float:
        """Route the write to L2 when the gate fires; return surprise."""
        self._step += 1
        pred = self.mid.predict_one(key, model)
        surprise = abs(reward - pred)
        dec = self.gate.decide(surprise, recurrent)
        if dec.to_mid and self._step % self.c2 == 0:
            self.mid.update(key, model, reward)
            self._writes += 1
        return surprise