import numpy as np
from tokenguard.memory.neural_memory import NeuralRoutingMemory


def test_memory_predict_shape():
    m = NeuralRoutingMemory(key_dim=8, n_models=3)
    assert m.predict(np.ones(8)).shape == (3,)


def test_memory_update_reduces_error():
    m = NeuralRoutingMemory(key_dim=8, n_models=3, lr=0.1, forget=0.0)
    key = np.ones(8) / np.sqrt(8)
    for _ in range(50):
        m.update(key, model=1, reward=1.0)
    assert m.predict(key)[1] > m.predict(key)[0]
