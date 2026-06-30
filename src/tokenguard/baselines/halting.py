"""REFRAIN / MUR-style uncertainty-halting baselines (for true reproduction)."""

from __future__ import annotations


def vanilla_halt(step_uncertainties) -> int:
    return len(step_uncertainties)


def refrain_halt(step_uncertainties, tau: float = 0.15, patience: int = 2,
                 min_steps: int = 1) -> int:
    below = 0
    for t, u in enumerate(step_uncertainties, start=1):
        below = below + 1 if u <= tau else 0
        if t >= min_steps and below >= patience:
            return t
    return len(step_uncertainties)


def mur_halt(step_uncertainties, momentum: float = 0.9, tau: float = 0.15,
             min_steps: int = 1) -> int:
    ema = None
    for t, u in enumerate(step_uncertainties, start=1):
        ema = u if ema is None else momentum * ema + (1 - momentum) * u
        if t >= min_steps and ema <= tau:
            return t
    return len(step_uncertainties)


def tokens_after_halt(step_token_counts, stop_step: int) -> int:
    return int(sum(step_token_counts[:stop_step]))
