"""Qwen3-Embedding-0.6B encoder — the modern, MTEB-leading replacement for BERT.

Unlike ``QwenEncoder`` (which mean-pools the *base* Qwen3-0.6B language model),
this wraps **Qwen3-Embedding-0.6B**, a dedicated embedding model released by the
Qwen team in June 2025. The 8B sibling tops the MTEB multilingual leaderboard;
the 0.6B variant is the smallest of the family (0.6B / 4B / 8B) and is a strong,
Apache-2.0, 32K-context, instruction-aware encoder — a defensible, current
choice (and explicitly *not* BERT, per the supervisor's requirement).

Design notes
------------
* Drop-in: exposes ``encode(list[str]) -> np.ndarray`` returning L2-normalised
  float32 rows, identical to ``SentenceEmbedder`` / ``QwenEncoder``, so it slots
  straight into the contrastive router and the embedding cache.
* Query-side prompt: Qwen3-Embedding supports an instruction/prompt for the
  query side, applied via ``prompt=`` when the installed sentence-transformers
  version supports it, with a graceful fallback otherwise.
* Lazy load + injectable: the model is only constructed on first ``encode`` so
  the test-suite stays network-free; unit tests inject a deterministic stub.
"""

from __future__ import annotations

import numpy as np

from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.encoders.qwen3_embedding")

_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Recommended retrieval instruction for the query side (Qwen3-Embedding card).
_QUERY_INSTRUCTION = (
    "Instruct: Given a user query, represent it for selecting the best "
    "language model to answer it.\nQuery: "
)


class Qwen3Embedding:
    """Lazy wrapper around Qwen3-Embedding-0.6B with L2-normalised output."""

    MODEL_ID = _DEFAULT_MODEL

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 8,
        device: str | None = None,
        use_instruction: bool = True,
        truncate_dim: int | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self.use_instruction = use_instruction
        self.truncate_dim = truncate_dim
        self._model = None

    # ------------------------------------------------------------------ #
    def _lazy(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        if self._device is None:
            try:
                import torch

                self._device = (
                    "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"
                )
            except Exception:  # pragma: no cover
                self._device = "cpu"

        logger.info("Loading Qwen3-Embedding encoder %s on %s",
                    self.model_name, self._device)
        kwargs = {"device": self._device}
        if self.truncate_dim is not None:
            kwargs["truncate_dim"] = self.truncate_dim
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    # ------------------------------------------------------------------ #
    def _empty_cache(self) -> None:
        """Free accelerator memory between chunks (MPS/CUDA)."""
        try:
            import torch
            if self._device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def encode(self, texts: list[str], chunk_size: int = 512) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim).

        Encodes in chunks and clears the accelerator cache between them so a
        large corpus does not exhaust MPS/GPU memory (Qwen3-Embedding-0.6B is a
        1.2 GB model; the default batch_size is intentionally small).
        """
        model = self._lazy()
        encode_kwargs = dict(
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        prompt_kw = {}
        if self.use_instruction:
            prompt_kw = {"prompt": _QUERY_INSTRUCTION}

        out = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            try:
                emb = model.encode(chunk, **prompt_kw, **encode_kwargs)
            except TypeError:
                emb = model.encode(chunk, **encode_kwargs)
            out.append(np.asarray(emb, dtype=np.float32))
            self._empty_cache()
        return np.vstack(out).astype(np.float32)