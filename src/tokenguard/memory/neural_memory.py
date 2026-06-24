"""L2 — Titans-style neural memory updated at TEST TIME by gradient-of-surprise.

This is the core novelty of NESTOR. The memory M is a linear associative map
key -> per-model reward, trained ONLINE (at test time) with the Titans update:

    associative loss : ell(M; k, v) = || M(k) - v ||^2            (Titans Eq. 12)
    surprise momentum: S_t = eta_t * S_{t-1} - theta_t * grad      (Titans Eq. 14)
    memory update    : M_t = (1 - alpha_t) * M_{t-1} + S_t          (Titans Eq. 13)

The faithful part (vs. the Day-5 heuristic) is that alpha_t (forget gate) and
eta_t (surprise decay) are DATA-DEPENDENT — functions of the momentary surprise,
not constants. Large surprise -> retain more past surprise (higher eta_t) and
write harder; small surprise -> let the memory coast and gently forget. This is
exactly what makes Titans memorise the *surprising, recurring* structure and
forget the rest, and it is the mechanism the Day-5 constant gate lacked.

Keyed on query-cluster / user embeddings, it remembers which model wins for
recurring query neighbourhoods and adapts the ROUTING POLICY (not the answer).
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class NeuralRoutingMemory:
    """Test-time-updated associative memory: query key -> per-model reward.

    Parameters
    ----------
    key_dim, n_models
        Dimensions of the key (projected query) and the value (reward vector).
    lr : float
        Base write strength theta_0 (scaled up by surprise).
    momentum : float
        Base momentum for past surprise (the eta_0 floor).
    forget : float
        Base forget rate alpha_0 (scaled by surprise so confident, low-surprise
        steps forget less).
    surprise_scale : float
        How strongly the data-dependent gates react to surprise.
    """

    def __init__(self, key_dim: int, n_models: int,
                 lr: float = 0.05, momentum: float = 0.9,
                 forget: float = 0.01, surprise_scale: float = 2.0,
                 seed: int = 42):
        self.key_dim = key_dim
        self.n_models = n_models
        self.lr = lr
        self.momentum = momentum
        self.forget = forget
        self.surprise_scale = surprise_scale
        self.rng = np.random.default_rng(seed)
        # linear associative map M: (n_models, key_dim); surprise momentum S
        self.M = np.zeros((n_models, key_dim), dtype=np.float64)
        self.S = np.zeros_like(self.M)

    # ------------------------------------------------------------------ #
    def predict(self, key: np.ndarray) -> np.ndarray:
        """Per-model reward estimate for a query key, shape (n_models,)."""
        return self.M @ key

    def predict_one(self, key: np.ndarray, model: int) -> float:
        return float(self.M[model] @ key)

    # ------------------------------------------------------------------ #
    def update(self, key: np.ndarray, model: int, reward: float) -> float:
        """Test-time Titans update for the observed (key, model, reward).

        Returns the momentary surprise |reward - prediction|.

        Data-dependent gates (the genuine Titans mechanism):
          surprise s   = |reward - pred|
          eta_t        = momentum * sigmoid(scale * s)   (retain more past
                         surprise when the world is surprising)
          theta_t      = lr * (1 + scale * s)            (write harder on
                         surprise)
          alpha_t      = forget * (1 - sigmoid(scale*s)) (forget less when
                         surprised; coast/forget when confident)
        """
        key = key.astype(np.float64)
        pred = float(self.M[model] @ key)
        residual = reward - pred                 # signed surprise
        s = abs(residual)

        # data-dependent gates, all bounded in [0,1)-ish ranges for stability
        sg = _sigmoid(self.surprise_scale * s)
        eta_t = self.momentum * sg                                    # past-surprise retention (<momentum)
        theta_t = self.lr * (0.5 + sg)                                # write strength, bounded in [0.5lr, 1.5lr]
        alpha_t = self.forget * (1.0 - sg)                            # forgetting

        grad = -residual * key                   # d/dM of 0.5*(pred-reward)^2 for row `model`
        # surprise momentum (Titans Eq. 14): S_t = eta_t S_{t-1} - theta_t grad
        self.S[model] = eta_t * self.S[model] - theta_t * grad
        # clip momentum to keep the test-time update stable on long streams
        np.clip(self.S[model], -10.0, 10.0, out=self.S[model])
        # memory update (Titans Eq. 13): M_t = (1-alpha_t) M_{t-1} + S_t
        self.M[model] = (1.0 - alpha_t) * self.M[model] + self.S[model]
        np.clip(self.M[model], -10.0, 10.0, out=self.M[model])
        return s

    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        return {"M": self.M.copy(), "S": self.S.copy()}

    def load_state(self, st: dict) -> None:
        self.M = st["M"].copy()
        self.S = st["S"].copy()