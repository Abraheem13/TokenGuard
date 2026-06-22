"""Qwen3-Embedding-0.6B wrapper (modern BERT replacement)."""
from __future__ import annotations
import numpy as np


class Qwen3Embedding:
    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self, device="mps", cache_path=None):
        self.device, self.cache_path, self._model = device, cache_path, None

    def encode(self, texts):
        raise NotImplementedError("implement Qwen3-Embedding encode in week 1")
