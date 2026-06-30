"""Real-LLM generation with token counting + per-step uncertainty (vLLM backend)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field


@dataclass
class GenResult:
    text: str
    answer_text: str
    n_prompt_tokens: int
    n_gen_tokens: int
    step_uncertainty: list[float] = field(default_factory=list)
    token_entropy: list[float] = field(default_factory=list)
    mean_logprob: float = 0.0
    finish_reason: str = ""


def _entropy_from_logprobs(logprob_dict) -> float:
    if not logprob_dict:
        return 0.0
    lps = [lp.logprob for lp in logprob_dict.values()]
    ps = [math.exp(x) for x in lps]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


class LLMRunner:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B",
                 tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 8192,
                 top_logprobs: int = 20,
                 seed: int = 42):
        self.model_name = model_name
        self.tp_size = tensor_parallel_size or self._auto_tp()
        self.dtype = dtype
        self.gpu_mem = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.top_logprobs = top_logprobs
        self.seed = seed
        self._llm = None
        self._tok = None

    @staticmethod
    def _auto_tp() -> int:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            return max(1, len([x for x in vis.split(",") if x.strip() != ""]))
        try:
            import torch
            return max(1, torch.cuda.device_count())
        except Exception:
            return 1

    def _lazy(self):
        if self._llm is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_mem,
            max_model_len=self.max_model_len,
            seed=self.seed,
            trust_remote_code=True,
        )

    def _build_prompt(self, question: str, system: str | None, cot_prefix: str) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": question + "\n" + cot_prefix})
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, max_tokens: int = 512,
                 temperature: float = 0.0, system: str | None = None,
                 cot_prefix: str = "Let's think step by step.",
                 stop: list[str] | None = None) -> GenResult:
        self._lazy()
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=temperature, max_tokens=max_tokens,
            logprobs=self.top_logprobs, stop=stop, seed=self.seed,
        )
        prompt = self._build_prompt(question, system, cot_prefix)
        out = self._llm.generate([prompt], sp)[0]
        comp = out.outputs[0]

        token_entropy: list[float] = []
        chosen_logprobs: list[float] = []
        for i, tok_id in enumerate(comp.token_ids):
            lp_dict = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else None
            token_entropy.append(_entropy_from_logprobs(lp_dict))
            if lp_dict and tok_id in lp_dict:
                chosen_logprobs.append(lp_dict[tok_id].logprob)

        text = comp.text
        step_uncertainty = self._steps_uncertainty(text, comp.token_ids, token_entropy)
        mean_lp = float(sum(chosen_logprobs) / len(chosen_logprobs)) if chosen_logprobs else 0.0
        return GenResult(
            text=text, answer_text=text,
            n_prompt_tokens=len(out.prompt_token_ids),
            n_gen_tokens=len(comp.token_ids),
            step_uncertainty=step_uncertainty, token_entropy=token_entropy,
            mean_logprob=mean_lp, finish_reason=comp.finish_reason or "",
        )

    def _steps_uncertainty(self, text, token_ids, token_entropy) -> list[float]:
        if not token_entropy:
            return []
        steps, cur = [], []
        for i, tid in enumerate(token_ids):
            cur.append(token_entropy[i])
            piece = self._tok.decode([tid])
            if "\n" in piece:
                if cur:
                    steps.append(sum(cur) / len(cur))
                    cur = []
        if cur:
            steps.append(sum(cur) / len(cur))
        return steps

    def count_tokens(self, text: str) -> int:
        self._lazy()
        return len(self._tok.encode(text))
