"""Disk-backed embedding cache.

Day 3 showed that re-encoding the 36k RouterBench prompts on every run costs
~90 minutes on Apple MPS. Every Day-4/5 experiment needs those embeddings
repeatedly (the contrastive router, the kNN/cascade baselines, and the online
stream simulation that replays the test split thousands of times). Encoding
once and caching to disk turns those minutes into milliseconds.

Design:

* Embeddings are keyed by ``(encoder_id, content_hash)`` where ``content_hash``
  is a BLAKE2b digest of the exact prompt string. Identical prompts across
  splits share a cached vector; changing the encoder invalidates cleanly.
* The cache is a single parquet (``sample_id``, ``hash``, ``dim_0..dim_{D-1}``)
  plus a small JSON manifest recording the encoder id and dimension. Parquet
  keeps load fast and the file portable.
* ``EmbeddingCache.encode`` returns embeddings in the *exact order* of the
  input prompts, computing only the cache misses and persisting them.

This module deliberately knows nothing about which encoder is used — it takes
any object exposing ``encode(list[str]) -> np.ndarray`` (Qwen3, MiniLM, or a
test stub), so the same caching path serves every router.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.cache")


def content_hash(text: str) -> str:
    """Stable 16-hex-char digest of a prompt string."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


class EmbeddingCache:
    """Cache embeddings for an arbitrary encoder, keyed by content hash."""

    def __init__(self, encoder, encoder_id: str, cache_dir: str | Path):
        """``encoder`` must expose ``encode(list[str]) -> (n, dim) ndarray``."""
        self.encoder = encoder
        self.encoder_id = encoder_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_id = encoder_id.replace("/", "__")
        self.parquet_path = self.cache_dir / f"emb__{safe_id}.parquet"
        self.manifest_path = self.cache_dir / f"emb__{safe_id}.manifest.json"
        self._table: dict[str, np.ndarray] = {}
        self._dim: int | None = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.parquet_path.exists():
            return
        df = pd.read_parquet(self.parquet_path)
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        self._dim = len(dim_cols)
        mat = df[dim_cols].to_numpy(dtype=np.float32)
        for h, vec in zip(df["hash"].to_numpy(), mat):
            self._table[h] = vec
        logger.info("Loaded %d cached embeddings from %s",
                    len(self._table), self.parquet_path)

    def _persist(self) -> None:
        if not self._table:
            return
        hashes = list(self._table.keys())
        mat = np.vstack([self._table[h] for h in hashes])
        df = pd.DataFrame(mat, columns=[f"dim_{i}" for i in range(mat.shape[1])])
        df.insert(0, "hash", hashes)
        df.to_parquet(self.parquet_path, index=False)
        with open(self.manifest_path, "w") as fh:
            json.dump(
                {"encoder_id": self.encoder_id, "dim": int(mat.shape[1]),
                 "n_cached": len(hashes)},
                fh, indent=2,
            )

    # ------------------------------------------------------------------ #
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for ``texts`` in order, computing only misses."""
        hashes = [content_hash(t) for t in texts]
        missing = [(t, h) for t, h in zip(texts, hashes) if h not in self._table]

        if missing:
            uniq: dict[str, str] = {}
            for t, h in missing:
                uniq.setdefault(h, t)
            logger.info("Encoding %d new prompts (%d cached)...",
                        len(uniq), len(self._table))
            new_vecs = self.encoder.encode(list(uniq.values()))
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            if self._dim is None:
                self._dim = new_vecs.shape[1]
            for h, vec in zip(uniq.keys(), new_vecs):
                self._table[h] = vec
            self._persist()

        return np.vstack([self._table[h] for h in hashes])

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Cache empty; encode something first.")
        return self._dim