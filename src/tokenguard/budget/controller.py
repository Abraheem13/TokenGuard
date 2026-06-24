"""NTC controller — wires SLOW (router prior) + MEDIUM (memory) + FAST (halting).

decide(query_key) -> (model, budget_prior, halter, tau)
observe(query_key, correct, tokens) -> updates bandit + memory.

TODO(week2): integrate the contrastive router for the SLOW prior; connect to the
llm/ harness so halting actually stops generation.
"""
from __future__ import annotations
import numpy as np

from tokenguard.budget.surprise import SurpriseHalter
from tokenguard.budget.memory import BudgetMemory
from tokenguard.budget.bandit import SWUCBThreshold


class NestedBudgetController:
    def __init__(self, key_dim: int, base_budget: int = 32, mu: float = 0.01,
                 seed: int = 42):
        self.memory = BudgetMemory(key_dim, seed=seed)
        self.bandit = SWUCBThreshold(seed=seed)
        self.base_budget = base_budget
        self.mu = mu

    def decide(self, key: np.ndarray):
        b0 = self.base_budget
        b_t = max(1, int(round(b0 + self.memory.predict(key))))
        arm, tau = self.bandit.select()
        halter = SurpriseHalter(tau=tau, max_steps=b_t)
        return {"budget": b_t, "tau": tau, "arm": arm, "halter": halter}

    def observe(self, key: np.ndarray, arm: int, correct: bool, tokens: int,
                target_budget: int) -> None:
        reward = (1.0 if correct else 0.0) - self.mu * tokens
        self.bandit.update(arm, reward)
        self.memory.update(key, target_budget - self.base_budget)
