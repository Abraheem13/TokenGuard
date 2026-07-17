#!/usr/bin/env python
"""TokenGuard Product Server v2 — LIVE inference + replay + how-it-works.

Modes:
  GPU node:     python demo/server.py --model Qwen/Qwen3-4B     (real live)
  Anywhere:     TG_FAKE=1 python demo/server.py                  (simulated live
                 from recorded traces — identical UX, zero GPU, demo-day fallback)
Replay + dashboard endpoints always available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokenguard.reasoning.datasets import is_correct  # noqa: E402

NTC = ROOT / "experiments" / "ntc"
app = FastAPI(title="TokenGuard")
ANSWER_CUE = "\n</think>\n\nThe final answer is \\boxed{"

RUNS = {
    "GSM8K · Qwen3-4B": ("w1_gsm8k_Qwen3-4B.json", "gsm8k"),
    "GSM8K · Qwen3-8B": ("w1_gsm8k_Qwen3-8B.json", "gsm8k"),
    "MATH-500 · Qwen3-4B": ("w1_math500_Qwen3-4B.json", "math500"),
    "MATH-500 · Qwen3-8B": ("w1_math500_Qwen3-8B.json", "math500"),
    "GPQA-Diamond · Qwen3-4B": ("w1sh_gpqa_Qwen3-4B.json", "gpqa_diamond"),
    "AIME-24 · Qwen3-4B": ("w1_aime24_Qwen3-4B_s100.json", "aime24"),
    "AIME-25 · Qwen3-4B": ("w1_aime25_Qwen3-4B_s100.json", "aime25"),
    "MATH-500 · DeepSeek-R1-7B": ("w1_math500_DeepSeek-R1-Distill-Qwen-7B.json", "math500"),
}
_cache: dict = {}
LIVE = {"llm": None, "tok": None, "model": None, "fake": bool(os.environ.get("TG_FAKE"))}


def load_run(name):
    if name not in RUNS:
        raise HTTPException(404, "unknown run")
    if name not in _cache:
        f, bench = RUNS[name]
        p = NTC / f
        if not p.exists():
            raise HTTPException(404, f"missing {f}")
        d = json.loads(p.read_text())
        for t in d["traces"]:
            t["natural_correct"] = bool(is_correct(t.get("natural_answer", ""), t["gold"], bench))
        _cache[name] = (d, bench)
    return _cache[name]


# ---------- policies ----------
def p_deer(pr, lam=0.95, **k):
    for i, p in enumerate(pr):
        if p["confidence"] >= lam:
            return i
    return None


def p_eat(pr, delta=1e-3, alpha=0.2, warm=3, **k):
    ema = emv = None
    for i, p in enumerate(pr):
        h = p["entropy"]
        if ema is None:
            ema, emv = h, 0.0
        else:
            d = h - ema
            ema += alpha * d
            emv = (1 - alpha) * (emv + alpha * d * d)
        if i + 1 >= warm and emv < delta:
            return i
    return None


def p_conf(pr, theta=0.9, eta=0.6, pat=2, **k):
    S, ab = None, 0
    for i, p in enumerate(pr):
        S = p["confidence"] if S is None else eta * S + (1 - eta) * p["confidence"]
        ab = ab + 1 if S >= theta else 0
        if ab >= pat:
            return i
    return None


def p_agree(pr, m=3, bm="math500", **k):
    run = 1
    for i in range(1, len(pr)):
        same = pr[i]["answer"] and is_correct(pr[i]["answer"], pr[i - 1]["answer"], bm)
        run = run + 1 if same else 1
        if run >= m:
            return i
    return None


POL = {"Confidence (DEER-λ)": p_deer, "Entropy (EAT)": p_eat,
       "Smoothed confidence": p_conf, "Answer agreement": p_agree}
TG_PICK = {"gsm8k": ("Answer agreement", p_agree, {"m": 3}),
           "math500": ("Answer agreement", p_agree, {"m": 3}),
           "gpqa_diamond": ("Smoothed confidence", p_conf, {"theta": 0.9}),
           "aime24": ("Smoothed confidence", p_conf, {"theta": 0.9}),
           "aime25": ("Smoothed confidence", p_conf, {"theta": 0.9})}


# ---------- replay endpoints (unchanged behavior) ----------
@app.get("/api/runs")
def api_runs():
    return [{"name": n, "benchmark": b} for n, (f, b) in RUNS.items() if (NTC / f).exists()]


@app.get("/api/questions")
def api_questions(run: str):
    d, bench = load_run(run)
    return [{"qid": t["qid"], "preview": t["question"][:110],
             "n_tokens": t["n_total_tokens"], "vanilla_correct": t["natural_correct"]}
            for t in d["traces"]]


@app.get("/api/trace")
def api_trace(run: str, qid: str):
    d, bench = load_run(run)
    t = next((x for x in d["traces"] if x["qid"] == qid), None)
    if not t:
        raise HTTPException(404, "qid")
    van = t["n_total_tokens"]
    fam, fn, kw = TG_PICK.get(bench, TG_PICK["math500"])
    allp = dict(POL)
    allp[f"★ TokenGuard → {fam}"] = lambda pr, **k: fn(pr, **kw, bm=bench)
    dec = {}
    prs = [{"ckpt_tokens": p["ckpt_tokens"], "answer": p["answer"],
            "confidence": p["confidence"], "entropy": p["first_entropy"]}
           for p in t["probes"]]
    for name, f in allp.items():
        i = f(prs, bm=bench) if prs else None
        if i is None:
            dec[name] = {"halt_ckpt": None, "tokens": van,
                         "answer": t.get("natural_answer", "")[-60:],
                         "correct": t["natural_correct"], "saved_pct": 0.0}
        else:
            p, tok = t["probes"][i], t["probes"][i]["ckpt_tokens"] + t["probes"][i]["n_probe_tokens"]
            dec[name] = {"halt_ckpt": i, "tokens": tok, "answer": p["answer"],
                         "correct": bool(is_correct(p["answer"], t["gold"], bench)),
                         "saved_pct": round(100 * (1 - tok / van), 1)}
    return {"qid": t["qid"], "question": t["question"], "gold": t["gold"],
            "think_text": t["think_text"], "n_total_tokens": van,
            "natural_answer": t.get("natural_answer", ""),
            "vanilla_correct": t["natural_correct"], "benchmark": bench,
            "probes": prs, "decisions": dec}


@app.get("/api/summary")
def api_summary():
    rows = []
    for f in sorted(NTC.glob("GENSEEDS_*.md")):
        van = tg = None
        for ln in f.read_text().splitlines():
            c = [x.strip() for x in ln.strip("|").split("|")] if ln.startswith("|") else []
            if len(c) >= 3 and re.match(r"[\d.]+", c[1] or ""):
                v = (float(re.match(r"([\d.]+)", c[1]).group(1)),
                     float(re.match(r"(-?[\d.]+)", c[2]).group(1)) if re.match(r"(-?[\d.]+)", c[2] or "") else 0)
                if c[0] == "vanilla":
                    van = v
                elif c[0].startswith("NTC-full(e=0.01)"):
                    tg = v
        if van and tg:
            rows.append({"benchmark": f.stem.replace("GENSEEDS_", ""),
                         "vanilla_acc": van[0], "tg_acc": tg[0], "tg_cut": tg[1]})
    return rows


@app.get("/api/fig1")
def api_fig1():
    p = ROOT / "paper_figures" / "fig1_collapse_map.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ---------- LIVE ----------
class LiveReq(BaseModel):
    prompt: str
    max_think: int = 4096


def _ensure_llm():
    if LIVE["llm"] is None:
        from vllm import LLM
        from transformers import AutoTokenizer
        m = LIVE["model"] or "Qwen/Qwen3-4B"
        print(f"[live] loading {m} ...")
        LIVE["tok"] = AutoTokenizer.from_pretrained(m)
        LIVE["llm"] = LLM(model=m, max_model_len=8192, gpu_memory_utilization=0.85,
                          dtype="bfloat16")
        print("[live] ready")


def _gen(prompt, max_tokens, stop=None, logprobs=None, temp=0.6):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=max_tokens,
                        stop=stop, logprobs=logprobs, seed=42)
    out = LIVE["llm"].generate([prompt], sp, use_tqdm=False)[0].outputs[0]
    return out


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def live_stream(req: LiveReq):
    bench = "math500"
    fam, fn, kw = TG_PICK[bench]
    if LIVE["fake"]:
        # simulated live from a recorded trace — identical event stream
        d, _ = load_run("MATH-500 · Qwen3-4B")
        t = d["traces"][int(time.time()) % len(d["traces"])]
        yield sse({"type": "status", "msg": f"(simulated) streaming recorded Qwen3-4B trace",
                   "question": t["question"]})
        probes, text, last = [], t["think_text"], 0
        for p in t["probes"]:
            seg_end = min(len(text), int(len(text) * p["ckpt_tokens"] / max(1, t["n_total_tokens"])))
            for i in range(last, seg_end, 60):
                yield sse({"type": "chunk", "text": text[i:i + 60]})
                time.sleep(0.04)
            last = seg_end
            probes.append({"ckpt_tokens": p["ckpt_tokens"], "answer": p["answer"],
                           "confidence": p["confidence"], "entropy": p["first_entropy"]})
            run = 1
            for j in range(len(probes) - 1, 0, -1):
                if probes[j]["answer"] and probes[j]["answer"] == probes[j - 1]["answer"]:
                    run += 1
                else:
                    break
            yield sse({"type": "probe", "ckpt": len(probes) - 1, **probes[-1], "agree_run": run})
            i = fn(probes, **kw, bm=bench)
            if i is not None:
                tok = probes[i]["ckpt_tokens"]
                yield sse({"type": "halt", "policy": f"★ TokenGuard → {fam}",
                           "tokens": tok, "answer": probes[i]["answer"],
                           "vanilla_tokens": t["n_total_tokens"],
                           "saved_pct": round(100 * (1 - tok / t["n_total_tokens"]), 1)})
                yield sse({"type": "final", "vanilla_tokens": t["n_total_tokens"],
                           "vanilla_answer": t.get("natural_answer", "")[-80:],
                           "match": True})
                return
        yield sse({"type": "final", "vanilla_tokens": t["n_total_tokens"],
                   "vanilla_answer": t.get("natural_answer", "")[-80:], "match": True})
        return

    # ---- REAL live path ----
    _ensure_llm()
    tok = LIVE["tok"]
    msgs = [{"role": "user",
             "content": req.prompt + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    think, probes, total = "", [], 0
    yield sse({"type": "status", "msg": f"live · {LIVE['model']}", "question": req.prompt})
    halted = None
    while total < req.max_think:
        seg = _gen(base + think, max_tokens=300, stop=["\n\n"], temp=0.6)
        stext = seg.text
        think += stext + ("\n\n" if seg.finish_reason == "stop" and "</think>" not in stext else "")
        total += len(seg.token_ids)
        for i in range(0, len(stext), 80):
            yield sse({"type": "chunk", "text": stext[i:i + 80]})
        if "</think>" in stext:
            break
        # probe: force a trial answer
        pr = _gen(base + think.rstrip() + ANSWER_CUE, max_tokens=24, logprobs=3, temp=0.0)
        ans = pr.text.split("}")[0].strip()
        lps = []
        ent = 0.0
        if pr.logprobs:
            for j, tid in enumerate(pr.token_ids[:12]):
                lp = pr.logprobs[j]
                if tid in lp:
                    lps.append(lp[tid].logprob)
                if j == 0:
                    ps = [math.exp(v.logprob) for v in lp.values()]
                    s = sum(ps)
                    ent = -sum((x / s) * math.log(max(1e-9, x / s)) for x in ps)
        conf = math.exp(sum(lps) / max(1, len(lps))) if lps else 0.0
        probes.append({"ckpt_tokens": total, "answer": ans, "confidence": conf, "entropy": ent})
        run = 1
        for j in range(len(probes) - 1, 0, -1):
            if probes[j]["answer"] and is_correct(probes[j]["answer"], probes[j - 1]["answer"], bench):
                run += 1
            else:
                break
        yield sse({"type": "probe", "ckpt": len(probes) - 1, **probes[-1], "agree_run": run})
        if halted is None:
            i = fn(probes, **kw, bm=bench)
            if i is not None:
                halted = (probes[i]["ckpt_tokens"], probes[i]["answer"])
                yield sse({"type": "halt", "policy": f"★ TokenGuard → {fam}",
                           "tokens": halted[0], "answer": halted[1]})
    # finish vanilla for comparison
    fin = _gen(base + think + "\n</think>\n\n", max_tokens=400, temp=0.6)
    total += len(fin.token_ids)
    van_ans_m = re.findall(r"\\boxed\{([^}]*)\}", fin.text)
    van_ans = van_ans_m[-1] if van_ans_m else fin.text[-60:]
    ev = {"type": "final", "vanilla_tokens": total, "vanilla_answer": van_ans}
    if halted:
        ev["saved_pct"] = round(100 * (1 - halted[0] / total), 1)
        ev["match"] = bool(is_correct(halted[1], van_ans, bench))
    yield sse(ev)


@app.post("/api/live")
def api_live(req: LiveReq):
    return StreamingResponse(live_stream(req), media_type="text/event-stream")


@app.get("/api/live_status")
def api_live_status():
    return {"fake": LIVE["fake"], "model": LIVE["model"] or "Qwen/Qwen3-4B",
            "loaded": LIVE["llm"] is not None}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True))

if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    LIVE["model"] = a.model
    mode = "FAKE (simulated)" if LIVE["fake"] else f"LIVE ({a.model or 'Qwen/Qwen3-4B'} lazy-load)"
    print(f"TokenGuard v2 → http://localhost:{a.port}   mode: {mode}")
    uvicorn.run(app, host="0.0.0.0", port=a.port)
