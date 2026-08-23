"""Figure: (a) the certificate under domain shift, (b) the price of the tail."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8.2,
 "axes.linewidth":0.7,"axes.edgecolor":"#40506A","pdf.fonttype":42,"xtick.labelsize":7.2,
 "ytick.labelsize":7.2,"axes.titlesize":8.5,"axes.labelsize":8.0,"legend.fontsize":6.6,
 "axes.grid":True,"grid.color":"#E4EAF0","grid.linewidth":0.6,"axes.axisbelow":True})
AUD=Path(__file__).resolve().parents[2]/"experiments"/"ntc"
OURS_A="#0B6E4F"; OURS_B="#1F8A70"; C1="#1F4E79"; C4="#B03A2E"; GREY="#5B6B80"
sc=json.load(open(AUD/"SHIFT_CERTIFICATE.json"))
fig,axes=plt.subplots(1,2,figsize=(7.1,2.85),gridspec_kw={"wspace":0.32,"width_ratios":[1,1.05]})

# ---------- (a) per-cell deficit distribution, eps = 0.05 ----------
ax=axes[0]; e="0.05"
rules=[("in","in-domain\n(A1 holds)",OURS_A),("transfer","transferred\n(A1 violated)",C4),
       ("robust","domain-robust\n(min. over sources)",C1)]
rng=np.random.default_rng(0)
for i,(r,lab,col) in enumerate(rules):
    v=np.array([x[1] for x in sc["cells"][e][r]])
    ax.scatter(np.full(len(v),i)+rng.uniform(-.16,.16,len(v)),v,s=7,color=col,alpha=.45,
               edgecolor="none",zorder=3)
    ax.plot([i-.28,i+.28],[v.mean()]*2,color="#1A1A1A",lw=1.6,zorder=4)
    ax.text(i,7.5,f"mean {v.mean():+.1f}\ncut {sc['cut'][e][r]:.0f}%",ha="center",fontsize=6.3,color=GREY)
    ax.text(i,v.min()-3.2,f"{v.min():+.0f}",ha="center",fontsize=6.3,color=col)
ax.axhline(0,color="#40506A",lw=0.8)
ax.axhline(-5,color=OURS_A,ls="--",lw=0.9)
ax.text(-0.48,-8.0,r"$\varepsilon=5$ pt budget",fontsize=6.2,color=OURS_A,ha="left")
ax.set_xticks(range(3)); ax.set_xticklabels([l for _,l,_ in rules],fontsize=6.5)
ax.set_ylabel("held-out accuracy deficit (points)")
ax.set_title(r"(a) The certificate under shift ($\varepsilon=0.05$)")
ax.set_ylim(-58,16); ax.set_xlim(-.55,2.55)

# ---------- (b) price of the tail ----------
tp=json.load(open(AUD/"TAIL_PRICE.json")); S=tp["settings"]; D=tp["data"]
NAME=[("NTC-v2","NTC-Fuse (ours)",OURS_A,2.0),
      ("NTC-full(e=0.05)",r"NTC-Select $\varepsilon{=}0.05$ (ours)",OURS_B,2.0),
      ("AGREE","Answer agreement",C1,1.1),("EAT","Entropy (EAT)","#C2681E",1.1),
      ("REFRAIN-SWUCB","Bandit threshold","#7D3C98",1.1),
      ("MUR-mom","Uncertainty momentum",C4,1.1)]
ax=axes[1]; kaps=np.linspace(0,20,200)
for m,nm,col,lw in NAME:
    u=[np.mean([D[m][s][1]-k*max(0.0,-D[m][s][0]) for s in S]) for k in kaps]
    ax.plot(kaps,u,lw=lw,color=col,alpha=.95 if lw>1.5 else .8,label=nm)
ax.axvline(6.38,color=GREY,ls=":",lw=1.0)
ax.annotate(r"agreement $\rightarrow$ ours at $\kappa=6.4$",xy=(6.38,-8),xytext=(8.4,-8),
            fontsize=6.4,color=GREY,va="center",
            arrowprops=dict(arrowstyle="->",color=GREY,lw=0.7))
ax.set_xlabel(r"$\kappa$: token-saving points forgone per accuracy point")
ax.set_ylabel("mean utility over 11 settings")
ax.set_title("(b) The price of the tail"); ax.set_ylim(-16,60); ax.set_xlim(0,20)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.27),ncol=2,frameon=False,
          handletextpad=0.4,columnspacing=1.0)
fig.savefig("fig_shift.pdf",bbox_inches="tight",pad_inches=0.02)
fig.savefig("fig_shift.png",bbox_inches="tight",pad_inches=0.02,dpi=170)
print("wrote fig_shift.pdf")
