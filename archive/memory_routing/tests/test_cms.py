import numpy as np
from tokenguard.memory.cms import ContinuumMemoryRouter


def test_cms_observe_returns_surprise():
    cms = ContinuumMemoryRouter(key_dim=8, n_models=3)
    s = cms.observe(np.ones(8) / np.sqrt(8), model=0, reward=1.0, recurrent=True)
    assert s >= 0.0


def test_cms_predict_blend_shape():
    cms = ContinuumMemoryRouter(key_dim=8, n_models=3)
    out = cms.predict(np.ones(8), fast_pred=np.zeros(3))
    assert out.shape == (3,)
