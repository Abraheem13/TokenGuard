"""Real-LLM generation with token counting + step uncertainty.

Wraps vLLM (preferred) or transformers to: generate a reasoning chain, expose
per-step logprobs/entropy (the FAST-tier surprise signal), count tokens, and
support early halting. Keep models small (Qwen3-1.7B/4B/8B) for GPU budget.

TODO(week1): implement generate(prompt, max_tokens, halter=None) -> dict with
keys {text, n_tokens, step_uncertainty[list], answer}.
"""
from __future__ import annotations


class LLMRunner:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", device: str = "cuda",
                 backend: str = "vllm"):
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self._llm = None

    def _lazy(self):
        raise NotImplementedError("implement vLLM/transformers load in week 1")

    def generate(self, prompt: str, max_tokens: int = 512, halter=None) -> dict:
        raise NotImplementedError("implement generation + token counting in week 1")
