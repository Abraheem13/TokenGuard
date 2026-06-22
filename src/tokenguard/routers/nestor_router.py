"""NESTOR top-level router wiring CMS to the decision rule."""
from __future__ import annotations
import numpy as np
from tokenguard.memory.cms import ContinuumMemoryRouter


class NestorRouter:
    def __init__(self, base, key_dim, lambda_cost=0.5, c2=1, c3=4000, seed=42):
        self.base, self.lambda_cost = base, lambda_cost
        self.cms = ContinuumMemoryRouter(key_dim, len(base.models_), c2=c2, c3=c3, seed=seed)

    def run_stream(self, stream, **kw):
        raise NotImplementedError("wire CMS into run_stream in week 3")
