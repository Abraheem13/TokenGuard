"""Continuum Memory System orchestrator (L1+L2+L3)."""
from __future__ import annotations
import numpy as np
from tokenguard.memory.neural_memory import NeuralRoutingMemory
from tokenguard.memory.surprise import SurpriseGate
from tokenguard.memory.consolidation import SlowConsolidator


class ContinuumMemoryRouter:
    def __init__(self, key_dim, n_models, c2=1, c3=4000, seed=42):
        self.mid = NeuralRoutingMemory(key_dim, n_models, seed=seed)
        self.gate = SurpriseGate()
        self.slow = SlowConsolidator(every=c3)
        self.c2, self._step = c2, 0

    def predict(self, key, fast_pred):
        return 0.5 * fast_pred + 0.5 * self.mid.predict(key)

    def observe(self, key, model, reward, recurrent=False):
        self._step += 1
        pred = float(self.mid.predict(key)[model])
        surprise = abs(reward - pred)
        dec = self.gate.decide(surprise, recurrent)
        if dec.to_mid and self._step % self.c2 == 0:
            self.mid.update(key, model, reward)
        return surprise
