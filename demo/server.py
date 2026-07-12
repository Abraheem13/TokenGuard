#!/usr/bin/env python
"""TokenGuard Demo Server (Mode 1: replay). Serves recorded reasoning traces
with per-checkpoint signals and per-policy halting decisions.

Run from repo root:   python demo/server.py   (opens on port 8000)
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tokenguard.reasoning.datasets import is_correct  # noqa: E402

NTC = ROOT / "experiments" / "ntc"
app = FastAPI(title="TokenGuard Demo")

RUNS = {  # display name -> (file, benchmark)
    "GSM8K · Qwen3-4B":   ("w1_gsm8k_Qwen3-4B.json", "gsm8k"),
    "GSM8K · Qwen3-8B":   ("w1_gsm8k_Qwen3-8B.json", "gsm8k"),
    "MATH-500 · Qwen3-4B": ("w1_math500_Qwen3-4B.json", "math500"),
    "MATH-500 · Qwen3-8B": ("w1_math500_Qwen3-8B.json", "math500"),
    "GPQA-Diamond · Qwen3-4B": ("w1sh_gpqa_Qwen3-4B.json", "gpqa_diamond"),
    "AIME-24 · Qwen3-4B": ("w1_aime24_Qwen3-4B_s100.json", "aime24"),
    "AIME-25 · Qwen3-4B": ("w1_aime25_Qwen3-4B_s100.json", "aime25"),
    "MATH-500 · DeepSeek-R1-7B": ("w1_math500_DeepSeek-R1-Distill-Qwen-7B.json", "math500"),
}
# TokenGuard slow-tier calibrated pick per benchmark (from canon tables)
TG_PICK = {"gsm8k": ("AGREE", {"m": 3}), "math500": ("AGREE", {"m": 3}),
           "gpqa_diamond": ("NTC-conf", {"theta": 0.9}),
           "aime24": ("NTC-conf", {"theta": 0.9}),
           "aime25": ("NTC-conf", {"theta": 0.9})}

_cache: dict = {}


def load_run(name: str):
    if name not in RUNS:
        raise HTTPException(404, f"unknown run {name}")
    if name not in _cache:
        f, bench = RUNS[name]
        path = NTC / f
        if not path.exists():
            raise HTTPException(404, f"file missing: {f}")
        d = json.loads(path.read_text())
        for t in d["traces"]:
            t["natural_correct"] = bool(
                is_correct(t.get("natural_answer", ""), t["gold"], bench))
        _cache[name] = (d, bench)
    return _cache[name]


# ---- policies (identical to analysis scripts) ----
def deer(probes, lam=0.95, **kw):
    for k, p in enumerate(probes):
        if p["confidence"] >= lam:
            return k
    return None


def eat(probes, delta=1e-3, alpha=0.2, warmup=3, **kw):
    ema = emv = None
    for k, p in enumerate(probes):
        h = p["first_entropy"]
        if ema is None:
            ema, emv = h, 0.0
        else:
            d = h - ema
            ema += alpha * d
            emv = (1 - alpha) * (emv + alpha * d * d)
        if k + 1 >= warmup and emv < delta:
            return k
    return None


def conf(probes, theta=0.9, eta=0.6, patience=2, **kw):
    S, above = None, 0
    for k, p in enumerate(probes):
        c = p["confidence"]
        S = c if S is None else eta * S + (1 - eta) * c
        above = above + 1 if S >= theta else 0
        if above >= patience:
            return k
    return None


def agree(probes, m=3, bm="math500", **kw):
    run = 1
    for k in range(1, len(probes)):
        same = (probes[k]["answer"] and
                is_correct(probes[k]["answer"], probes[k - 1]["answer"], bm))
        run = run + 1 if same else 1
        if run >= m:
            return k
    return None


POLICIES = {"Confidence (DEER-λ=0.95)": deer, "Entropy (EAT)": eat,
            "Smoothed confidence": conf, "Answer agreement (m=3)": agree}


@app.get("/api/runs")
def api_runs():
    out = []
    for name, (f, bench) in RUNS.items():
        if (NTC / f).exists():
            out.append({"name": name, "benchmark": bench})
    return out


@app.get("/api/questions")
def api_questions(run: str):
    d, bench = load_run(run)
    return [{"qid": t["qid"], "preview": t["question"][:110],
             "n_tokens": t["n_total_tokens"],
             "vanilla_correct": t["natural_correct"]}
            for t in d["traces"]]


@app.get("/api/trace")
def api_trace(run: str, qid: str):
    d, bench = load_run(run)
    t = next((x for x in d["traces"] if x["qid"] == qid), None)
    if t is None:
        raise HTTPException(404, "qid not found")
    # per-policy decisions
    decisions = {}
    fam, kw = TG_PICK.get(bench, ("AGREE", {"m": 3}))
    tg_fn = agree if fam == "AGREE" else conf
    all_p = dict(POLICIES)
    all_p[f"★ TokenGuard (adaptive → {fam})"] = lambda pr, **k: tg_fn(pr, **kw, bm=bench)
    van_tok = t["n_total_tokens"]
    for name, fn in all_p.items():
        k = fn(t["probes"], bm=bench) if t["probes"] else None
        if k is None:
            decisions[name] = {"halt_ckpt": None, "tokens": van_tok,
                               "answer": t.get("natural_answer", "")[-60:],
                               "correct": t["natural_correct"], "saved_pct": 0.0}
        else:
            p = t["probes"][k]
            tok = p["ckpt_tokens"] + p["n_probe_tokens"]
            decisions[name] = {
                "halt_ckpt": k, "tokens": tok, "answer": p["answer"],
                "correct": bool(is_correct(p["answer"], t["gold"], bench)),
                "saved_pct": round(100 * (1 - tok / van_tok), 1)}
    probes = [{"ckpt_tokens": p["ckpt_tokens"], "answer": p["answer"],
               "confidence": p["confidence"], "entropy": p["first_entropy"]}
              for p in t["probes"]]
    return {"qid": t["qid"], "question": t["question"], "gold": t["gold"],
            "think_text": t["think_text"], "n_total_tokens": van_tok,
            "natural_answer": t.get("natural_answer", ""),
            "vanilla_correct": t["natural_correct"],
            "benchmark": bench, "probes": probes, "decisions": decisions}


@app.get("/api/summary")
def api_summary():
    rows = []
    for f in sorted(NTC.glob("GENSEEDS_*.md")):
        tag = f.stem.replace("GENSEEDS_", "")
        van = tg = ag = None
        for ln in f.read_text().splitlines():
            if not ln.startswith("|"):
                continue
            c = [x.strip() for x in ln.strip("|").split("|")]
            if len(c) < 3:
                continue
            acc = re.match(r"([\d.]+)", c[1])
            cut = re.match(r"(-?[\d.]+)", c[2])
            if not acc:
                continue
            v = (float(acc.group(1)), float(cut.group(1)) if cut else 0.0)
            if c[0] == "vanilla":
                van = v
            elif c[0].startswith("NTC-full(e=0.01)"):
                tg = v
            elif c[0] == "AGREE":
                ag = v
        if van and tg:
            rows.append({"benchmark": tag, "vanilla_acc": van[0],
                         "tg_acc": tg[0], "tg_cut": tg[1],
                         "agree_acc": ag[0] if ag else None,
                         "agree_cut": ag[1] if ag else None})
    return rows


@app.get("/api/fig1")
def api_fig1():
    p = ROOT / "paper_figures" / "fig1_collapse_map.png"
    if not p.exists():
        raise HTTPException(404, "fig1 missing")
    return FileResponse(p)


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"),
                           html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("TokenGuard demo → http://localhost:8000  (VS Code: PORTS tab)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
