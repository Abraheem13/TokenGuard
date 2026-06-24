"""Tests for the modern Qwen3-Embedding encoder wrapper (network-free)."""

from __future__ import annotations

import numpy as np

from tokenguard.encoders.qwen3_embedding import Qwen3Embedding


class _StubST:
    """Minimal stand-in for SentenceTransformer used to test the wrapper logic
    without downloading the real model."""

    def __init__(self, dim=8):
        self.dim = dim
        self.last_prompt = "UNSET"

    def encode(self, texts, prompt=None, batch_size=64, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        self.last_prompt = prompt
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        if normalize_embeddings:
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb


def test_encode_shape_and_norm():
    enc = Qwen3Embedding()
    enc._model = _StubST(dim=8)              # inject stub
    out = enc.encode(["hello", "world", "foo"])
    assert out.shape == (3, 8)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)   # L2-normalised contract


def test_instruction_is_passed_when_enabled():
    enc = Qwen3Embedding(use_instruction=True)
    stub = _StubST()
    enc._model = stub
    enc.encode(["q"])
    assert stub.last_prompt is not None and "Query:" in stub.last_prompt


def test_instruction_skipped_when_disabled():
    enc = Qwen3Embedding(use_instruction=False)
    stub = _StubST()
    enc._model = stub
    enc.encode(["q"])
    assert stub.last_prompt is None