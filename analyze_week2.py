#!/usr/bin/env python
"""Analyse the saved week2 Pareto curves honestly: which method gives the best
accuracy at each token level (the real comparison)."""
import json, sys
from pathlib import Path

p = "experiments/ntc/week2.json"
d = json.load(open(p))
full = d["full"]
print(f"full-chain: acc={full['acc']:.3f} tokens={full['tokens']:.0f}\n")

def show(name, curve):
    print(f"=== {name} Pareto ===")
    print(f"{'tau':>6}{'acc':>8}{'tokens':>9}{'cut%':>7}")
    for pt in sorted(curve, key=lambda x: x['tokens']):
        cut = 100*(1 - pt['tokens']/full['tokens'])
        print(f"{pt['tau']:>6}{pt['acc']:>8.3f}{pt['tokens']:>9.1f}{cut:>6.0f}%")
    print()

for key, name in [("refrain_curve","REFRAIN"),("mur_curve","MUR"),("ntc_curve","NTC")]:
    if key in d: show(name, d[key])

# HONEST head-to-head: at matched token budgets, who has higher accuracy?
print("=== HEAD-TO-HEAD: accuracy at matched token levels ===")
import numpy as np
def interp_acc(curve, target_tok):
    pts = sorted(curve, key=lambda x: x['tokens'])
    toks = [p['tokens'] for p in pts]; accs = [p['acc'] for p in pts]
    return float(np.interp(target_tok, toks, accs))

if all(k in d for k in ["refrain_curve","ntc_curve"]):
    levels = [100, 200, 300, 400, 500]
    print(f"{'tokens':>8}{'REFRAIN':>10}{'NTC':>8}{'winner':>10}")
    for lv in levels:
        ra = interp_acc(d["refrain_curve"], lv)
        na = interp_acc(d["ntc_curve"], lv)
        w = "NTC" if na > ra else "REFRAIN" if ra > na else "tie"
        print(f"{lv:>8}{ra:>10.3f}{na:>8.3f}{w:>10}")