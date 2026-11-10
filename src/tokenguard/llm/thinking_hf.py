"""Thinking-mode generation and answer probing on plain PyTorch.

A second backend with the interface of `ThinkingRunner` (thinking.py), for
machines on which vLLM does not run (native Windows). It uses the Hugging Face
`transformers` generation loop and records the same quantities: the reasoning
trace, the log-probability of every sampled token under the unmodified model
distribution, and for every probe the trial answer, its confidence, the
entropy of the first answer token over the twenty most probable tokens and
the number of tokens decoded.

It is slower than vLLM: a batch decodes until its longest member has
finished, and probes do not share the cache of the trace.
"""

from __future__ import annotations

import math

from tokenguard.llm.thinking import ANSWER_CUE, Probe, deer_confidence, read_boxed, split_think


def entropy_of_top(logprobs: list[float]) -> float:
    """Entropy of the renormalised distribution over the given top log-probabilities
    (the computation of thinking._entropy_from_top)."""
    if not logprobs:
        return 0.0
    ps = [math.exp(x) for x in logprobs]
    z = sum(ps) or 1.0
    ps = [p / z for p in ps]
    return float(-sum(p * math.log(p + 1e-12) for p in ps))


class HFThinkingRunner:
    """transformers wrapper for batched thinking generation and answer probing."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B", dtype: str = "bfloat16",
                 max_model_len: int = 12288, seed: int = 42, probe_batch_size: int = 8,
                 device: str | None = None, model=None, tokenizer=None):
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.seed = seed
        self.probe_batch_size = probe_batch_size
        self.device = device
        self._model = model
        self._tok = tokenizer
        self._calls = 0

    # -- loading ---------------------------------------------------------
    @property
    def tok(self):
        self._lazy()
        return self._tok

    def _lazy(self):
        import torch
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            dt = getattr(torch, self.dtype)
            try:   # the keyword was renamed from torch_dtype to dtype
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dt)
            except TypeError:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                   torch_dtype=dt)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
        if self.device is None:
            self.device = str(next(self._model.parameters()).device)
        self._model.eval()
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

    def _chat_prompt(self, question: str) -> str:
        msgs = [{"role": "user", "content": question}]
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _eos_ids(self) -> set[int]:
        ids = set()
        for src in (getattr(self._model, "generation_config", None), self._tok):
            eos = getattr(src, "eos_token_id", None)
            if eos is None:
                continue
            ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        return ids

    def _encode(self, prompts: list[str]):
        enc = self._tok(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    # -- thinking generation ---------------------------------------------
    def generate_thinking(self, questions: list[str], max_tokens: int = 6144,
                          temperature: float = 0.6, top_p: float = 0.95,
                          top_k: int = 20) -> list[dict]:
        """Batched thinking-mode generation. Returns per-question dicts:
        {text, think_text, answer_text, had_close, n_think, n_total, finish, token_nll}."""
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        self._lazy()
        ids, mask = self._encode([self._chat_prompt(q) for q in questions])
        n_prompt = ids.shape[1]

        class ChosenLogProb(LogitsProcessor):
            """Records log p(token) of each sampled token under the unmodified
            distribution: it runs before temperature, top-k and top-p."""

            def __init__(self):
                self.prev = None
                self.cols = []

            def __call__(self, input_ids, scores):
                if self.prev is not None:
                    self.cols.append(self.prev.gather(1, input_ids[:, -1:]).squeeze(1).cpu())
                self.prev = torch.log_softmax(scores.float(), dim=-1)
                return scores

        rec = ChosenLogProb()
        sample = temperature > 0
        kwargs = dict(max_new_tokens=max_tokens, do_sample=sample,
                      logits_processor=LogitsProcessorList([rec]),
                      pad_token_id=self._tok.pad_token_id)
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        else:  # greedy: silence the sampling defaults of the model's generation config
            kwargs.update(temperature=None, top_p=None, top_k=None)
        torch.manual_seed(self.seed + self._calls)
        self._calls += 1
        with torch.inference_mode():
            out = self._model.generate(input_ids=ids, attention_mask=mask, **kwargs)
        gen = out[:, n_prompt:]
        if rec.prev is not None and gen.shape[1] > len(rec.cols):
            rec.cols.append(rec.prev.gather(1, gen[:, len(rec.cols)].unsqueeze(1)).squeeze(1).cpu())
        logp = torch.stack(rec.cols, dim=1) if rec.cols else torch.zeros((len(questions), 0))
        gen = gen.cpu()

        eos = self._eos_ids()
        res = []
        for b in range(len(questions)):
            row = gen[b].tolist()
            end = next((i for i, t in enumerate(row) if t in eos), None)
            finish = "stop" if end is not None else "length"
            n_total = (end + 1) if end is not None else len(row)
            body = row[:end] if end is not None else row
            text = self._tok.decode(body, skip_special_tokens=True)
            think, answer, closed = split_think(text)
            nll = [round(-float(x), 4) for x in logp[b, :n_total].tolist()]
            res.append({
                "text": text, "think_text": think, "answer_text": answer,
                "had_close": closed,
                "n_think": len(self._tok.encode(think, add_special_tokens=False)),
                "n_total": n_total, "finish": finish, "token_nll": nll,
            })
        return res

    # -- probing ---------------------------------------------------------
    def probe_batch(self, jobs: list[tuple[str, str]], max_answer_tokens: int = 24
                    ) -> list[Probe]:
        """Probe every (question, think_prefix) job. Greedy decoding; the twenty
        largest log-probabilities of the first step give the first-token entropy."""
        import torch

        self._lazy()
        prompts = [self._chat_prompt(q) + "<think>\n" + prefix + ANSWER_CUE
                   for q, prefix in jobs]
        lengths = [len(self._tok.encode(p, add_special_tokens=False)) for p in prompts]
        order = sorted(range(len(jobs)), key=lambda i: lengths[i])
        eos = self._eos_ids()
        probes: list[Probe | None] = [None] * len(jobs)
        for s in range(0, len(order), self.probe_batch_size):
            sel = order[s:s + self.probe_batch_size]
            ids, mask = self._encode([prompts[i] for i in sel])
            with torch.inference_mode():
                out = self._model.generate(
                    input_ids=ids, attention_mask=mask, max_new_tokens=max_answer_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=self._tok.pad_token_id)
            gen = out.sequences[:, ids.shape[1]:].cpu()
            logps = [torch.log_softmax(sc.float(), dim=-1).cpu() for sc in out.scores]
            for b, i in enumerate(sel):
                row = gen[b].tolist()
                end = next((j for j, t in enumerate(row) if t in eos), None)
                n_tok = (end + 1) if end is not None else len(row)
                body = row[:end] if end is not None else row
                top = torch.topk(logps[0][b], k=min(20, logps[0].shape[1])).values.tolist()
                chosen = []
                for j, tid in enumerate(row[:n_tok]):
                    chosen.append(float(logps[j][b, tid]))
                    if "}" in self._tok.decode([tid]):
                        break
                probes[i] = Probe(
                    ckpt_tokens=len(self._tok.encode(jobs[i][1], add_special_tokens=False)),
                    answer=read_boxed(self._tok.decode(body, skip_special_tokens=True)),
                    confidence=deer_confidence(chosen),
                    first_entropy=entropy_of_top(top), n_probe_tokens=n_tok)
        return probes  # type: ignore[return-value]
