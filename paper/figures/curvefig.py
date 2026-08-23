"""Deployable operating curves for four representative settings, regenerated from
the released analysis code and the frozen probe files."""
import sys, types, typing, json, warnings
warnings.filterwarnings("ignore")
_m=types.ModuleType("typing.io"); _m.TextIO=typing.TextIO; _m.IO=typing.IO; _m.BinaryIO=typing.BinaryIO
sys.modules["typing.io"]=_m
from pathlib import Path
import importlib.util, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8,
    "pdf.fonttype":42,"axes.linewidth":0.7,"axes.edgecolor":"#40506A","axes.grid":True,
    "grid.color":"#E8EDF3","grid.linewidth":0.6,"axes.axisbelow":True,
    "xtick.labelsize":7,"ytick.labelsize":7,"axes.labelsize":8,"axes.titlesize":8.4})
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("oc",ROOT/"scripts"/"ntc_operating_curves.py")
OC=importlib.util.module_from_spec(spec); sys.modules["oc"]=OC; spec.loader.exec_module(OC); S=OC.S
NTC=ROOT/"experiments"/"ntc"
PANELS=[("w1_gsm8k_Qwen3-4B.json","GSM8K / Qwen3-4B"),
        ("w1_gsm8k_Qwen3-8B.json","GSM8K / Qwen3-8B"),
        ("w1_math500_Qwen3-1.7B.json","MATH-500 / Qwen3-1.7B"),
        ("w1_math500_Qwen3-4B.json","MATH-500 / Qwen3-4B")]
STYLE={"Confidence (DEER-λ)":("#1F4E79","o"),"Entropy (EAT)":("#C2681E","s"),
       "Smoothed confidence":("#7D3C98","^"),"Answer agreement":("#B03A2E","D"),
       "NTC-v2 (fusion)":("#6E7B8B","v")}
LAB={"NTC-v2 (fusion)":"NTC-Fuse (ours)"}
fig,axes=plt.subplots(1,4,figsize=(9.6,2.5),squeeze=False)
for j,(fn,tag) in enumerate(PANELS):
    d=json.loads((NTC/fn).read_text()); traces,bench=d["traces"],d["benchmark"]
    for t in traces: t["natural_correct"]=bool(S.is_correct(t.get("natural_answer",""),t["gold"],bench))
    S.enrich_probes_with_nll(traces)
    warm,ev=OC.split(traces,seed=0)
    vt=float(np.mean([t["n_total_tokens"] for t in ev])); va=float(np.mean([t["natural_correct"] for t in ev]))
    ax=axes[0][j]
    for name,(f,grid) in OC.SWEEPS.items():
        pts=sorted(OC.curve_deployable(warm,ev,bench,f,grid,vt,va))
        c,mk=STYLE[name]
        if name=="NTC-v2 (fusion)":
            ax.plot([p[0] for p in pts],[p[1] for p in pts],marker="*",ms=7.5,lw=2.1,
                    color="#0B6E4F",alpha=.95,zorder=5,label="NTC-Fuse (ours)")
        elif name=="Answer agreement":
            ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=mk,ms=3.4,lw=3.0,
                    color=c,alpha=.42,zorder=3,label=name)
        else:
            ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=mk,ms=3.0,lw=1.1,
                    color=c,alpha=.85,label=name)
    pts=sorted(OC.curve_ntc_full(warm,ev,bench,vt,va))
    ax.plot([p[0] for p in pts],[p[1] for p in pts],marker="P",ms=4.2,lw=1.5,ls="--",
            color="#1F8A70",alpha=.95,zorder=4,label="NTC-Select (ours)")
    ax.axhline(va,color="#555",ls="--",lw=.9)
    ax.set_xlabel("cost / full thinking")
    if j==0: ax.set_ylabel("accuracy")
    ax.set_title(tag,fontsize=8.2)
h,l=axes[0][0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.5,-0.13),ncol=6,frameon=False,fontsize=7)
fig.tight_layout()
fig.savefig("fig_curves.pdf",bbox_inches="tight",pad_inches=0.02)
fig.savefig("fig_curves.png",bbox_inches="tight",pad_inches=0.02,dpi=170)  # PNGDUP
print("wrote fig_curves.pdf")
