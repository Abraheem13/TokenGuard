import numpy as np
from tokenguard.budget.memory import BudgetMemory


def test_memory_learns_correction():
    m = BudgetMemory(key_dim=4, lr=0.2, forget=0.0)
    k = np.ones(4) / 2
    for _ in range(50):
        m.update(k, target_correction=5.0)
    assert m.predict(k) > 1.0
