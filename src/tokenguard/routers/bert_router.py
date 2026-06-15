"""BERT-classifier router (the supervisor-rejected baseline, built *fairly*).

We include this baseline not as a strawman but to **earn** the claim that a
static BERT classifier is the wrong tool for this regime. It is implemented
the way a competent practitioner would: a pre-trained transformer encoder
(DistilBERT by default) with a multi-label classification head predicting,
for each model, whether it answers the query correctly. Routing then uses the
predicted per-model success probabilities under the shared cost-aware rule —
identical to every other router, so the comparison is apples-to-apples.

Why it tends to lag matrix factorisation here (and in RouteLLM's own results,
where the BERT router scored APGR 0.391, below random, on Arena data): with
86 tasks, 11 models, and a few tens of thousands of binary labels, a
high-capacity encoder is data-hungry and prone to collapsing toward the
majority-accuracy model, whereas the low-rank factoriser is well matched to
the low-rank structure of the (query, model) success matrix.

This module depends on torch + transformers. To keep the test-suite fast and
network-free, training/inference live behind methods that are only exercised
when those libraries (and a model cache) are available; a pure-NumPy unit test
covers the routing/decision glue via a tiny stub.
"""

from __future__ import annotations

import numpy as np

from tokenguard.data.routerbench import RouterBench
from tokenguard.routers.base import Router
from tokenguard.utils.logging import get_logger

logger = get_logger("tokenguard.routers.bert")


class BertClassifierRouter(Router):
    """DistilBERT encoder + multi-label success-probability head."""

    name = "bert-classifier"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        device: str | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self._device = device
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------ #
    def _lazy_init(self, n_models: int) -> None:
        """Import torch/transformers and build the encoder + head on demand."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(self.seed)
        self._torch = torch
        if self._device is None:
            self._device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        logger.info("BERT router device: %s", self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoder = AutoModel.from_pretrained(self.model_name)

        class _Net(torch.nn.Module):
            def __init__(self, enc, hidden, n_out):
                super().__init__()
                self.enc = enc
                self.head = torch.nn.Linear(hidden, n_out)

            def forward(self, input_ids, attention_mask):
                out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0]  # [CLS] pooling
                return self.head(cls)

        hidden = encoder.config.hidden_size
        self._model = _Net(encoder, hidden, n_models).to(self._device)

    # ------------------------------------------------------------------ #
    def fit(self, train: RouterBench) -> "BertClassifierRouter":
        super().fit(train)
        self._lazy_init(len(train.models))
        torch = self._torch

        prompts = train.df["prompt"].tolist()
        Y = torch.tensor(train.perf_matrix(), dtype=torch.float32)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        n = len(prompts)
        rng = np.random.default_rng(self.seed)

        self._model.train()
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_prompts = [prompts[i] for i in idx]
                enc = self._tokenizer(
                    batch_prompts, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                targets = Y[idx].to(self._device)

                opt.zero_grad()
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                loss = loss_fn(logits, targets)
                loss.backward()
                opt.step()
                epoch_loss += float(loss) * len(idx)
            logger.info("BERT epoch %d/%d — loss %.4f",
                        epoch + 1, self.epochs, epoch_loss / n)
        return self

    # ------------------------------------------------------------------ #
    def predict_quality(self, bench: RouterBench) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        torch = self._torch
        prompts = bench.df["prompt"].tolist()
        self._model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                enc = self._tokenizer(
                    batch, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                logits = self._model(enc["input_ids"], enc["attention_mask"])
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs)