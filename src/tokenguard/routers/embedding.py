"""Shared semantic embedding featuriser for the modern baselines.

The kNN and cascade routers both need a *semantic* query representation (not
just the hashing-trick bag-of-words used by the MF baseline). We use a small
Sentence-Transformers encoder (``all-MiniLM-L6-v2``, 22M params, 384-dim),
which:

* runs comfortably on CPU/MPS with no API budget,
* is a widely used 2021-2024 retrieval encoder, so the kNN/cascade baselines
  are faithful to how such routers are built in the literature, and
* is cached and reused across both routers and across runs.

To keep the test-suite network-free and fast, the encoder is loaded lazily;
unit tests inject a deterministic stub encoder instead (see tests/test_day3b).
"""

from __future__ import annotations

import numpy as np

from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.routers.embed")

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """Lazy wrapper around a Sentence-Transformers encoder with row caching."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence encoder: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""
        model = self._lazy()
        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)