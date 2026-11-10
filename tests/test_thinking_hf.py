"""The PyTorch backend on a tiny random model: recorded log-probabilities are
those of the unmodified distribution, and probing does not depend on batching."""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokenguard.llm.thinking import ANSWER_CUE  # noqa: E402
from tokenguard.llm.thinking_hf import HFThinkingRunner, entropy_of_top  # noqa: E402


def _tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    words = ["<pad>", "<eos>", "<unk>", "<think>", "</think>", "{", "}", "\\boxed", "The",
             "final", "answer", "is", "user", "assistant", ":", "Wait", "so", "x", "=",
             "1", "2", "3", "4", "A", "B", "C", "D", "what", "?", "\n"]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}assistant : ")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(words), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    model = GPT2LMHeadModel(cfg).eval()
    return model, tok


def _teacher_forced_logprobs(model, tok, prompt_ids, gen_ids):
    ids = torch.tensor([prompt_ids + gen_ids])
    with torch.inference_mode():
        logits = model(ids).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    n = len(prompt_ids)
    return [float(lp[n - 1 + j, t]) for j, t in enumerate(gen_ids)]


def test_recorded_nll_is_the_unmodified_log_probability():
    model, tok = _tiny()
    r = HFThinkingRunner(model=model, tokenizer=tok, device="cpu", seed=7)
    qs = ["what is x ?", "what is x = 1 2 3 ?"]
    out = r.generate_thinking(qs, max_tokens=12, temperature=0.6, top_p=0.95, top_k=20)
    assert len(out) == 2
    for o in out:
        assert o["finish"] in ("stop", "length")
        assert len(o["token_nll"]) == o["n_total"] <= 12
        if o["finish"] == "length":
            assert o["n_total"] == 12
    # Re-generate greedily to obtain the token ids, then compare with a plain forward pass.
    g = r.generate_thinking(qs[:1], max_tokens=10, temperature=0.0)[0]
    prompt_ids = tok.encode(r._chat_prompt(qs[0]), add_special_tokens=False)
    with torch.inference_mode():
        seq = model.generate(torch.tensor([prompt_ids]), max_new_tokens=10, do_sample=False,
                             pad_token_id=0)[0].tolist()[len(prompt_ids):]
    ref = _teacher_forced_logprobs(model, tok, prompt_ids, seq[:g["n_total"]])
    assert g["token_nll"] == pytest.approx([-x for x in ref], abs=2e-3)


def test_sampled_tokens_are_scored_before_temperature():
    model, tok = _tiny()
    r = HFThinkingRunner(model=model, tokenizer=tok, device="cpu", seed=3)
    q = "what is x ?"
    prompt_ids = tok.encode(r._chat_prompt(q), add_special_tokens=False)
    captured = {}
    orig = model.generate

    def spy(*a, **k):
        out = orig(*a, **k)
        captured["ids"] = out[0].tolist()[len(prompt_ids):]
        return out

    model.generate = spy
    o = r.generate_thinking([q], max_tokens=10, temperature=0.6)[0]
    model.generate = orig
    ref = _teacher_forced_logprobs(model, tok, prompt_ids, captured["ids"][:o["n_total"]])
    assert o["token_nll"] == pytest.approx([-x for x in ref], abs=2e-3)


def test_probe_batch_is_independent_of_batching_and_matches_a_forward_pass():
    model, tok = _tiny()
    jobs = [("what is x ?", "so x = 1"), ("what is x = 1 2 3 ?", "Wait so x = 2 so x = 3"),
            ("what ?", "so")]
    a = HFThinkingRunner(model=model, tokenizer=tok, device="cpu", probe_batch_size=1)
    b = HFThinkingRunner(model=model, tokenizer=tok, device="cpu", probe_batch_size=3)
    pa, pb = a.probe_batch(jobs, max_answer_tokens=6), b.probe_batch(jobs, max_answer_tokens=6)
    for x, y, (_q, prefix) in zip(pa, pb, jobs):
        assert x.answer == y.answer and x.n_probe_tokens == y.n_probe_tokens
        assert x.confidence == pytest.approx(y.confidence, abs=1e-4)
        assert x.first_entropy == pytest.approx(y.first_entropy, abs=1e-4)
        assert x.ckpt_tokens == len(tok.encode(prefix, add_special_tokens=False))
        assert 0.0 < x.confidence <= 1.0
    # first-token entropy against a plain forward pass
    q, prefix = jobs[0]
    ids = tok.encode(a._chat_prompt(q) + "<think>\n" + prefix + ANSWER_CUE,
                     add_special_tokens=False)
    with torch.inference_mode():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, -1].float(), dim=-1)
    top = torch.topk(lp, 20).values.tolist()
    assert pa[0].first_entropy == pytest.approx(entropy_of_top(top), abs=1e-4)
    assert entropy_of_top([math.log(0.5), math.log(0.5)]) == pytest.approx(math.log(2), abs=1e-6)
