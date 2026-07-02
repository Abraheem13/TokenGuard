"""Real-LLM generation with token counting + per-step uncertainty (vLLM backend).

Returns per-step text + token boundaries so a halting policy can truncate the
chain at any step and the answer can be HONESTLY re-extracted from the truncated
text (not assumed from the full chain).
"""

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
    step_texts: list[str] = field(default_factory=list)
    step_token_counts: list[int] = field(default_factory=list)
    token_entropy: list[float] = field(default_factory=list)
    mean_logprob: float = 0.0
    finish_reason: str = ""

    def text_upto_step(self, k: int) -> str:
        return "".join(self.step_texts[:k])

    def tokens_upto_step(self, k: int) -> int:
        return int(sum(self.step_token_counts[:k]))


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

    def generate(self, question: str, max_tokens: int = 1024,
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

        token_entropy = []
        chosen_logprobs = []
        for i, tok_id in enumerate(comp.token_ids):
            lp_dict = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else None
            token_entropy.append(_entropy_from_logprobs(lp_dict))
            if lp_dict and tok_id in lp_dict:
                chosen_logprobs.append(lp_dict[tok_id].logprob)

        steps_u, steps_txt, steps_tok = self._split_steps(comp.token_ids, token_entropy)
        mean_lp = float(sum(chosen_logprobs) / len(chosen_logprobs)) if chosen_logprobs else 0.0
        return GenResult(
            text=comp.text, answer_text=comp.text,
            n_prompt_tokens=len(out.prompt_token_ids),
            n_gen_tokens=len(comp.token_ids),
            step_uncertainty=steps_u, step_texts=steps_txt,
            step_token_counts=steps_tok, token_entropy=token_entropy,
            mean_logprob=mean_lp, finish_reason=comp.finish_reason or "",
        )

    def _split_steps(self, token_ids, token_entropy):
        steps_u, steps_txt, steps_tok = [], [], []
        cur_u, cur_ids = [], []
        for i, tid in enumerate(token_ids):
            cur_u.append(token_entropy[i])
            cur_ids.append(tid)
            piece = self._tok.decode([tid])
            if "\n" in piece:
                steps_u.append(sum(cur_u) / len(cur_u))
                steps_txt.append(self._tok.decode(cur_ids))
                steps_tok.append(len(cur_ids))
                cur_u, cur_ids = [], []
        if cur_ids:
            steps_u.append(sum(cur_u) / len(cur_u))
            steps_txt.append(self._tok.decode(cur_ids))
            steps_tok.append(len(cur_ids))
        return steps_u, steps_txt, steps_tok

    def count_tokens(self, text: str) -> int:
        self._lazy()
        return len(self._tok.encode(text))
