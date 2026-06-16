"""Qwen3-0.6B query encoder (the modern, 2025 replacement for BERT).

Produces a fixed-length semantic embedding per query by mean-pooling the last
hidden states of Qwen3-0.6B. This encoder is *frozen* — the contrastive router
(see ``contrastive_router``) learns a small projection and per-LLM embeddings
on top, RouterDC-style, which is far cheaper than fine-tuning the backbone and
matches the dissertation's "small, fast router" claim.

The class is loaded lazily and is normally wrapped by ``EmbeddingCache`` so the
36k prompts are encoded only once. Tests inject a stub encoder instead.
"""

from __future__ import annotations

import numpy as np

from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.routers.qwen")


class QwenEncoder:
    """Mean-pooled last-hidden-state encoder over Qwen3-0.6B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self._tok = None
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        logger.info("Loading Qwen encoder %s on %s", self.model_name, self._device)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        self._lazy()
        torch = self._torch
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self._tok(
                    batch, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / counts                       # mean-pool
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out)