"""Thinking-mode generation with answer probing.

Each question's <think>...</think> trace is generated once. At checkpoints
inside the trace an answer is forced by appending
"</think>\n\nThe final answer is \\boxed{" to the thinking prefix and letting
the model complete it greedily. Each probe records the trial answer, its
confidence (geometric mean of the chosen-token probabilities, as in DEER), the
entropy of the first answer token (as in EAT) and its token cost, so that any
halting rule can be scored afterwards by replaying the saved probes.

All probes of a batch are generated in one vLLM call.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field

# Openings of a paragraph that marks a reflective transition in the reasoning.
ATP_WORDS = ("Wait", "Alternatively", "Hmm", "But wait", "Let me double-check",
             "Actually", "Let me reconsider")

ANSWER_CUE = "\n</think>\n\nThe final answer is \\boxed{"


@dataclass
class Probe:
    ckpt_tokens: int          # thinking tokens consumed at this checkpoint
    answer: str               # trial answer (the content of \boxed{...})
    confidence: float         # geometric mean of the chosen-token probabilities
    first_entropy: float      # entropy of the first answer token
    n_probe_tokens: int       # tokens the probe decoded


@dataclass
class ThinkTrace:
    qid: str
    question: str
    gold: str
    think_text: str
    n_think_tokens: int
    natural_answer: str       # answer after the model's own </think> ("" if none)
    natural_correct: bool
    n_total_tokens: int       # thinking + natural answer tokens
    finish_reason: str        # "stop", or "length" if the budget was reached
    token_nll: list[float] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def read_boxed(text: str) -> str:
    """Read a forced \\boxed{...} answer: `text` starts inside the braces, so read
    up to the matching closing brace."""
    depth = 1
    out = []
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def split_think(text: str) -> tuple[str, str, bool]:
    """Split a generation into (think_text, answer_text, had_close_tag)."""
    if "</think>" in text:
        think, _, rest = text.partition("</think>")
        think = think.replace("<think>", "", 1)
        return think.strip("\n"), rest.strip(), True
    return text.replace("<think>", "", 1).strip("\n"), "", False


def build_checkpoints(think_text: str, tok, probe_every: int = 256,
                      max_probes: int = 10) -> list[tuple[int, str]]:
    """Paragraph-aligned checkpoints as [(cumulative tokens, thinking prefix)].

    The trace is split on blank lines. A checkpoint is placed at each paragraph
    boundary that crosses a multiple of `probe_every` tokens or precedes a
    reflective transition (a paragraph opening with one of ATP_WORDS). At most
    `max_probes` checkpoints are kept, thinned evenly.
    """
    paras = [p for p in re.split(r"\n\s*\n", think_text) if p.strip()]
    if not paras:
        return []
    cks: list[tuple[int, str]] = []
    cum_tokens = 0
    prefix_parts: list[str] = []
    next_mark = probe_every
    for i, p in enumerate(paras):
        prefix_parts.append(p)
        cum_tokens += len(tok.encode(p, add_special_tokens=False)) + 2
        is_atp_next = (i + 1 < len(paras) and
                       any(paras[i + 1].lstrip().startswith(w) for w in ATP_WORDS))
        crossed = cum_tokens >= next_mark
        if crossed or is_atp_next:
            cks.append((cum_tokens, "\n\n".join(prefix_parts)))
            while next_mark <= cum_tokens:
                next_mark += probe_every
    # A checkpoint at the end of the trace would repeat full generation.
    if cks and cks[-1][0] >= cum_tokens:
        cks = cks[:-1]
    if len(cks) > max_probes:  # thin evenly, keeping order
        idx = [round(j * (len(cks) - 1) / (max_probes - 1)) for j in range(max_probes)]
        cks = [cks[j] for j in sorted(set(idx))]
    return cks


def _entropy_from_top(logprob_dict) -> float:
    if not logprob_dict:
        return 0.0
    lps = [lp.logprob for lp in logprob_dict.values()]
    ps = [math.exp(x) for x in lps]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


def deer_confidence(chosen_logprobs: list[float]) -> float:
    """Geometric mean of the chosen-token probabilities (DEER's answer confidence)."""
    if not chosen_logprobs:
        return 0.0
    return float(math.exp(sum(chosen_logprobs) / len(chosen_logprobs)))


class ThinkingRunner:
    """vLLM wrapper for batched thinking generation and batched answer probing."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B",
                 tensor_parallel_size: int | None = None,
                 dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: int = 12288,
                 seed: int = 42):
        self.model_name = model_name
        self.tp_size = tensor_parallel_size or self._auto_tp()
        self.dtype = dtype
        self.gpu_mem = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self._llm = None
        self._tok = None

    @staticmethod
    def _auto_tp() -> int:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            return max(1, len([x for x in vis.split(",") if x.strip()]))
        try:
            import torch
            return max(1, torch.cuda.device_count())
        except Exception:
            return 1

    @property
    def tok(self):
        self._lazy()
        return self._tok

    def _lazy(self):
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._llm = LLM(model=self.model_name, tensor_parallel_size=self.tp_size,
                        dtype=self.dtype, gpu_memory_utilization=self.gpu_mem,
                        max_model_len=self.max_model_len, seed=self.seed,
                        trust_remote_code=True)

    def _chat_prompt(self, question: str) -> str:
        msgs = [{"role": "user", "content": question}]
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except TypeError:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def generate_thinking(self, questions: list[str], max_tokens: int = 6144,
                          temperature: float = 0.6, top_p: float = 0.95,
                          top_k: int = 20) -> list[dict]:
        """Batched thinking-mode generation. Returns per-question dicts:
        {text, think_text, answer_text, had_close, n_think, n_total, finish}."""
        self._lazy()
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k,
                            max_tokens=max_tokens, seed=self.seed, logprobs=1)
        prompts = [self._chat_prompt(q) for q in questions]
        outs = self._llm.generate(prompts, sp)
        res = []
        for out in outs:
            comp = out.outputs[0]
            think, answer, closed = split_think(comp.text)
            n_think = len(self._tok.encode(think, add_special_tokens=False))
            # Per-token NLL of the chosen tokens, used by the momentum baseline.
            nll = []
            for i, tid in enumerate(comp.token_ids):
                lp = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else None
                nll.append(round(-lp[tid].logprob, 4) if lp and tid in lp else 0.0)
            res.append({
                "text": comp.text, "think_text": think, "answer_text": answer,
                "had_close": closed, "n_think": n_think,
                "n_total": len(comp.token_ids),
                "finish": comp.finish_reason or "",
                "token_nll": nll,
            })
        return res

    def probe_batch(self, jobs: list[tuple[str, str]], max_answer_tokens: int = 24
                    ) -> list[Probe]:
        """Probe every (question, think_prefix) job in one vLLM call.

        Greedy decoding; the top 20 log-probabilities give the first-token entropy.
        """
        self._lazy()
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens,
                            logprobs=20, seed=self.seed)
        prompts = [self._chat_prompt(q) + "<think>\n" + prefix + ANSWER_CUE
                   for q, prefix in jobs]
        outs = self._llm.generate(prompts, sp)
        probes: list[Probe] = []
        for (_q, prefix), out in zip(jobs, outs):
            comp = out.outputs[0]
            ans = read_boxed(comp.text)
            chosen = []
            first_H = 0.0
            for i, tid in enumerate(comp.token_ids):
                lp = comp.logprobs[i] if comp.logprobs and i < len(comp.logprobs) else None
                if i == 0:
                    first_H = _entropy_from_top(lp)
                if lp and tid in lp:
                    chosen.append(lp[tid].logprob)
                # Stop once the boxed answer has closed.
                if "}" in self._tok.decode([tid]):
                    break
            probes.append(Probe(
                ckpt_tokens=len(self._tok.encode(prefix, add_special_tokens=False)),
                answer=ans, confidence=deer_confidence(chosen),
                first_entropy=first_H, n_probe_tokens=len(comp.token_ids)))
        return probes
