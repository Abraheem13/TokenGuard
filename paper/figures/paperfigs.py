"""All result figures for the TIST manuscript, rebuilt from the frozen result set.
Every number here is traceable to experiments/ntc/{SLO_ATTAINMENT,GENSEEDS_*,
COST_REGIMES,PROP2_VALIDATION,JOINT}.md or to the recomputed primary statistics
(audit/per_setting.json). Run:  python3 paperfigs.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({
    "font.family":"serif","font.serif":["DejaVu Serif"],"font.size":8.2,
    "axes.linewidth":0.7,"axes.edgecolor":"#40506A","pdf.fonttype":42,
    "xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.labelsize":7.4,
    "ytick.labelsize":7.4,"axes.titlesize":8.6,"axes.labelsize":8.2,
    "legend.fontsize":7.2,"axes.grid":True,"grid.color":"#E4EAF0",
    "grid.linewidth":0.6,"axes.axisbelow":True})
OURS_A="#0B6E4F"; OURS_B="#1F8A70"; C1="#1F4E79"; C2="#C2681E"; C3="#7D3C98"
C4="#B03A2E"; C5="#6E7B8B"; GREY="#5B6B80"
def save(fig,name):
    fig.savefig(name,bbox_inches="tight",pad_inches=0.02)
    fig.savefig(name.replace(".pdf",".png"),bbox_inches="tight",pad_inches=0.02,dpi=170)  # PNGDUP
    plt.close(fig); print("wrote",name)

# ---------------------------------------------------------------- Fig. collapse
cols=["GSM8K\n4B","GSM8K\n8B","MATH\n1.7B","MATH\n4B","MATH\n8B","MMLU-P\n4B",
      "MMLU-P\n8B","GPQA*\n4B","GPQA*\n8B","AIME-24\n4B","AIME-25\n4B"]
rows=["Answer agreement","Confidence (DEER-$\\lambda$)","Entropy (EAT)",
      "NLL momentum (MUR)","Smoothed confidence","Bandit-$\\lambda$ (REFRAIN)"]
D=np.array([
 [ 5.2,  8.3, -1.1, -5.5,  1.4, -8.2, -7.6, -7.3,-11.8,-23.5, -9.4],
 [-0.4,  0.1, -7.5, -5.6, -4.6, -4.6, -0.5, -0.9, -0.3, -0.5,  0.5],
 [ 0.2,  2.4, -3.0, -1.8, -3.4, -1.0, -0.7, -0.3, -1.3, -2.6, -1.8],
 [-0.6, -0.1, -9.7, -9.4, -8.2,  0.6,  0.4, -0.7, -0.9,-23.6, -4.7],
 [ 0.3,  1.9, -2.4, -2.2, -1.7, -1.5, -0.3,  0.0, -0.3, -1.3,  0.4],
 [-5.5, -1.0,-20.9,-16.2,-14.0,-10.8, -2.3, -2.9, -0.9, -8.9, -2.8]])
C=np.array([
 [55, 58, 45, 49, 49, 64, 65, 69, 68, 45, 45],
 [12, -4,  2,  8,  0, 22,  2,  4,  1,  1,  0],
 [23, 32,  7, 12, 14, 14, 12,  8,  8,  0,  5],
 [ 5,  8, 17, 14, 15,  6, 10,  5,  8, 19, 13],
 [ 6, 12, -3, -2, -4,  3,  2,  0,  0,  2,  0],
 [54, 37, 38, 41, 33, 55, 15, 43,  8, 20, 19]])
fig,ax=plt.subplots(figsize=(8.2,2.95))
im=ax.imshow(D,cmap="RdYlGn",vmin=-20,vmax=8,aspect="auto")
ax.set_xticks(range(len(cols)),cols,fontsize=6.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=7.4)
for i in range(len(rows)):
    for j in range(len(cols)):
        ax.text(j,i,f"{D[i,j]:+.1f}\n({C[i,j]:.0f}%)",ha="center",va="center",
                fontsize=5.9,color="white" if D[i,j]<-14 else "black")
ax.set_xticks(np.arange(-.5,len(cols),1),minor=True)
ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
ax.grid(which="minor",color="white",linewidth=0.8); ax.tick_params(which="minor",length=0)
for k in ("top","right","bottom","left"): ax.spines[k].set_visible(False)
cb=fig.colorbar(im,ax=ax,shrink=0.88,pad=0.012)
cb.set_label("change in accuracy against full thinking (points)",fontsize=7.2)
cb.ax.tick_params(labelsize=6.8); fig.tight_layout(pad=0.25)
save(fig,"fig_collapse.pdf")

# ---------------------------------------------------------------- Fig. SLO
methods=["NTC-Fuse\n(ours)","NTC-Select\n(ours)","Smoothed\nconfidence","Entropy\n(EAT)",
         "Confidence\n(DEER-$\\lambda$)","Bandit-$\\lambda$\n(REFRAIN)","Answer\nagreement",
         "Uncertainty\nmomentum"]
w1=np.array([100,82,55,36,64,9,27,55]); w25=np.array([100,100,100,73,64,27,36,55])
worst=np.array([-0.5,-1.6,-2.4,-3.4,-7.5,-20.9,-23.5,-23.6])
cols_=[OURS_A,OURS_B]+[C5]*6
fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),gridspec_kw={"width_ratios":[1.22,1],"wspace":0.42})
x=np.arange(len(methods))
axes[0].bar(x-0.19,w1,0.36,color=cols_,label="within 1.0 point")
axes[0].bar(x+0.19,w25,0.36,color=cols_,alpha=0.45,label="within 2.5 points")
axes[0].set_xticks(x); axes[0].set_xticklabels([m.replace("\n"," ") for m in methods],
    fontsize=6.3,rotation=38,ha="right",rotation_mode="anchor")
axes[0].set_ylabel("settings satisfying the budget (%)"); axes[0].set_ylim(0,112)
axes[0].set_title("(a) SLO attainment across 11 settings")
axes[0].legend(loc="lower center",bbox_to_anchor=(0.5,-0.62),ncol=2,frameon=False)
bars=axes[1].barh(x,worst,color=cols_,height=0.62)
axes[1].set_yticks(x); axes[1].set_yticklabels([m.replace("\n"," ") for m in methods],fontsize=6.4)
axes[1].invert_yaxis(); axes[1].set_xlabel("worst held-out deficit (accuracy points)")
axes[1].set_title("(b) Tail risk: worst case over all settings"); axes[1].set_xlim(-28.5,2.2)
for b,v in zip(bars,worst):
    axes[1].text(v-0.8,b.get_y()+b.get_height()/2,f"{v:+.1f}",va="center",ha="right",
                 fontsize=6.6,color="#1A1A1A")
axes[1].axvline(-1.0,color=OURS_A,ls="--",lw=0.9)
axes[1].text(-1.3,-0.62,"1-point budget",fontsize=6.3,color=OURS_A,ha="center",va="bottom")
save(fig,"fig_slo.pdf")

# ------------------------------------------------------- Fig. efficiency/regret
names=["NTC-Fuse (ours)","Answer agreement","NTC-Select (ours)","Entropy (EAT)",
       "Smoothed confidence","Confidence (DEER-$\\lambda$)"]
aucc=[0.611,0.601,0.585,0.550,0.541,0.505]
mxreg=[0.031,0.100,0.220,0.145,0.210,0.262]
mkr=["*","o","P","s","^","D"]; cl=[OURS_A,C1,OURS_B,C2,C3,C4]
fig,ax=plt.subplots(figsize=(3.5,2.75))
for n,a,r,m,c in zip(names,aucc,mxreg,mkr,cl):
    ax.scatter(r,a,marker=m,s=95 if m=="*" else 46,color=c,zorder=3,
               edgecolor="white",linewidth=0.5,label=n)
ax.set_xlabel("maximum regret over settings  (lower is better)")
ax.set_ylabel("op-AUCC  (higher is better)")
ax.set_title("Efficiency against worst-workload regret")
ax.set_xlim(0.0,0.29); ax.set_ylim(0.48,0.65)
ax.annotate("preferred",xy=(0.042,0.626),xytext=(0.132,0.626),fontsize=6.8,color=GREY,
            va="center",arrowprops=dict(arrowstyle="->",color=GREY,lw=0.8))
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.30),ncol=2,frameon=False,
          handletextpad=0.3,columnspacing=0.9)
save(fig,"fig_regret.pdf")

# ------------------------------------------------------------ Fig. stickiness
bench=["GSM8K-4B","GSM8K-8B","MATH-1.7B","MATH-4B","MATH-8B","MMLU-Pro-4B",
       "MMLU-Pro-8B","AIME-24","AIME-25","GPQA-4B","GPQA-8B","DeepSeek MATH"]
rho=[0.130,0.157,0.257,0.270,0.233,0.601,0.549,0.425,0.375,0.765,0.724,0.256]
dlt=[4.5,8.5,-2.0,-5.5,3.0,-5.5,-6.5,-30.0,-10.0,-4.0,-13.1,-14.0]
fam=["open","open","open","open","open","MCQ (10)","MCQ (10)","open, hard",
     "open, hard","MCQ (4)","MCQ (4)","open"]
palette={"open":C1,"open, hard":C2,"MCQ (10)":C3,"MCQ (4)":C4}
fig,ax=plt.subplots(figsize=(3.5,2.75))
for f in ["open","open, hard","MCQ (10)","MCQ (4)"]:
    idx=[i for i,g in enumerate(fam) if g==f]
    ax.scatter([rho[i] for i in idx],[dlt[i] for i in idx],s=42,color=palette[f],
               edgecolor="white",linewidth=0.5,zorder=3,label=f)
z=np.polyfit(rho,dlt,1); xs=np.linspace(0.10,0.80,50)
ax.plot(xs,np.polyval(z,xs),color=GREY,lw=1.0,ls="--",zorder=2)
ax.axhline(0,color="#9AA7B5",lw=0.7)
ax.set_xlabel(r"error stickiness $\rho_w$")
ax.set_ylabel("agreement $\\Delta$ accuracy (points)")
ax.set_title("Sticky errors predict agreement collapse"); ax.set_ylim(-34,13)
for i in (7,10,0):
    ax.annotate(bench[i],(rho[i],dlt[i]),fontsize=6.2,color=GREY,
                xytext=(5,-8 if i==7 else 5),textcoords="offset points")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.30),ncol=2,frameon=False,
          handletextpad=0.3,columnspacing=0.9)
save(fig,"fig_stickiness.pdf")

# -------------------------------------------------------------- Fig. regimes
setl=["Agreement\nGSM8K-8B","Agreement\nMATH-8B","Sm. confidence\nGSM8K-8B",
      "DEER-$\\lambda$\nGPQA-8B"]
kv=[56.4,50.0,-0.7,1.8]; pc=[56.0,49.8,-1.5,1.6]; bb=[34.6,17.4,-82.3,-92.0]
fig,ax=plt.subplots(figsize=(7.0,2.5)); x=np.arange(len(setl)); wd=0.26
ax.bar(x-wd,kv,wd,color=C1,label="KV-fork (cache reused)")
ax.bar(x,pc,wd,color=C2,label="prefix-cache (cue re-prefilled)")
ax.bar(x+wd,bb,wd,color=C4,label="black-box API (prefix re-sent)")
ax.axhline(0,color="#40506A",lw=0.8); ax.set_xticks(x)
ax.set_xticklabels(setl,fontsize=7.0)
ax.set_ylabel("token saving against full thinking (%)")
ax.set_title("The serving regime changes which signal is preferable"); ax.set_ylim(-105,72)
for xi,v in zip(np.concatenate([x-wd,x,x+wd]),kv+pc+bb):
    ax.text(xi,v+(2.6 if v>=0 else -7.5),f"{v:+.0f}",ha="center",fontsize=6.2,color="#1A1A1A")
ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.40),ncol=3,frameon=False)
save(fig,"fig_regimes.pdf")

# ------------------------------------------------- Fig. overhead + joint tier
fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),gridspec_kw={"width_ratios":[1.05,1],"wspace":0.46})
bm=["GPQA-D","AIME","MMLU-Pro","MATH-500","GSM8K"]
lo=[0.9,1.4,1.5,3.4,6.5]; hi=[1.2,1.5,1.6,4.3,7.1]
mid=[(a+b)/2 for a,b in zip(lo,hi)]
err=[[m-a for m,a in zip(mid,lo)],[b-m for b,m in zip(hi,mid)]]
axes[0].bar(bm,mid,0.55,yerr=err,capsize=2.6,color=OURS_B,error_kw=dict(lw=0.8,ecolor="#40506A"))
axes[0].axhline(3.1,color=GREY,ls="--",lw=0.9)
axes[0].text(4.45,3.35,"mean 3.1%",fontsize=6.4,color=GREY,ha="right")
axes[0].set_ylabel("probing overhead (% of full thinking)")
axes[0].set_title("(a) Measured overhead of the controller")
axes[0].set_ylim(0,8.4); axes[0].tick_params(axis="x",labelsize=6.8)
axes[0].text(-0.42,7.75,"DEER, authors' code: 66-79%",fontsize=6.4,color=C4)
lab=["Full thinking\n(large only)","Calibrated halting\n(large only)","Joint routing\n+ halting"]
cost=[17511,8409,6961]; acc=[0.683,0.708,0.683]
bars=axes[1].bar(lab,cost,0.55,color=[C5,OURS_B,OURS_A])
axes[1].set_ylabel("compute cost (tokens $\\times$ parameters)")
axes[1].set_title("(b) Joint tier at matched-or-better accuracy")
axes[1].set_ylim(0,21500); axes[1].tick_params(axis="x",labelsize=6.8)
for b,c,a in zip(bars,cost,acc):
    axes[1].text(b.get_x()+b.get_width()/2,c+700,f"{c:,}\nacc. {a:.3f}",ha="center",
                 fontsize=6.3,linespacing=1.2)
save(fig,"fig_overhead_joint.pdf")

# ---------------- Fig. Proposition 2 out-of-sample ----------------
pspur=[0.002,0.002,0.024,0.025,0.017,0.117,0.083,0.075,0.049,0.274,0.232,0.021]
lost=[0.034,0.030,0.110,0.134,0.067,0.129,0.168,0.450,0.286,0.277,0.391,0.184]
off={0:(0,-12),1:(0,8),4:(9,-4),2:(-9,-4),3:(9,1),11:(0,8),8:(0,-12),7:(0,8),
     6:(-9,3),5:(0,-12),10:(-9,3),9:(0,-12)}
ha={0:"center",1:"center",4:"left",2:"right",3:"left",11:"center",8:"center",
    7:"center",6:"right",5:"center",10:"right",9:"center"}
fig,ax=plt.subplots(figsize=(4.9,3.3))
for f in ["open","open, hard","MCQ (10)","MCQ (4)"]:
    idx=[i for i,g in enumerate(fam) if g==f]
    ax.scatter([pspur[i] for i in idx],[lost[i] for i in idx],s=44,color=palette[f],
               edgecolor="white",linewidth=0.6,zorder=3,label=f)
for i in range(len(bench)):
    ax.annotate(bench[i],(pspur[i],lost[i]),fontsize=6.0,color="#44526A",
                xytext=off[i],textcoords="offset points",ha=ha[i],zorder=4)
z=np.polyfit(np.log10(pspur),lost,1); xs=np.logspace(np.log10(0.0016),np.log10(0.34),60)
ax.plot(xs,np.polyval(z,np.log10(xs)),color=GREY,lw=1.0,ls="--",zorder=2)
ax.set_xscale("log"); ax.set_xlim(0.0013,0.46); ax.set_ylim(-0.03,0.55)
ax.set_xlabel(r"predicted spurious agreement $P_{\mathrm{spur}}$  ($m=3$, log scale)")
ax.set_ylabel("observed lost-correct risk")
ax.set_title("Predicted risk tracks measured risk across settings")
ax.text(0.0016,0.505,r"Spearman $\rho = +0.71$   ($n=12$)",fontsize=7.2,color="#1A1A1A")
ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.34),ncol=4,frameon=False,
          handletextpad=0.3,columnspacing=1.0)
save(fig,"fig_prop2.pdf")
print("result figures rebuilt")

# ---------------- Fig. mechanism: stickiness + out-of-sample prediction -------
fig,axes=plt.subplots(1,2,figsize=(7.1,2.85),gridspec_kw={"wspace":0.30})
ax=axes[0]
for f in ["open","open, hard","MCQ (10)","MCQ (4)"]:
    idx=[i for i,g in enumerate(fam) if g==f]
    ax.scatter([rho[i] for i in idx],[dlt[i] for i in idx],s=40,color=palette[f],
               edgecolor="white",linewidth=0.5,zorder=3,label=f)
z=np.polyfit(rho,dlt,1); xs=np.linspace(0.10,0.80,50)
ax.plot(xs,np.polyval(z,xs),color=GREY,lw=1.0,ls="--",zorder=2)
ax.axhline(0,color="#9AA7B5",lw=0.7)
ax.set_xlabel(r"error stickiness $\rho_w$")
ax.set_ylabel("agreement $\\Delta$ accuracy (points)")
ax.set_title("(a) Sticky errors predict collapse"); ax.set_ylim(-34,13)
for i in (7,10,0):
    ax.annotate(bench[i],(rho[i],dlt[i]),fontsize=6.0,color=GREY,
                xytext=(5,-8 if i==7 else 5),textcoords="offset points")
ax=axes[1]
for f in ["open","open, hard","MCQ (10)","MCQ (4)"]:
    idx=[i for i,g in enumerate(fam) if g==f]
    ax.scatter([pspur[i] for i in idx],[lost[i] for i in idx],s=40,color=palette[f],
               edgecolor="white",linewidth=0.5,zorder=3,label=f)
z=np.polyfit(np.log10(pspur),lost,1); xs=np.logspace(np.log10(0.0016),np.log10(0.34),60)
ax.plot(xs,np.polyval(z,np.log10(xs)),color=GREY,lw=1.0,ls="--",zorder=2)
ax.set_xscale("log"); ax.set_xlim(0.0013,0.46); ax.set_ylim(-0.03,0.55)
ax.set_xlabel(r"predicted $P_{\mathrm{spur}}$  ($m=3$, log scale)")
ax.set_ylabel("observed lost-correct risk")
ax.set_title("(b) Prediction tracks measurement")
ax.text(0.0016,0.505,r"Spearman $\rho=+0.71$  ($n=12$)",fontsize=7.0,color="#1A1A1A")
for i in (7,9,0):
    ax.annotate(bench[i],(pspur[i],lost[i]),fontsize=6.0,color=GREY,
                xytext=(6,-3),textcoords="offset points")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.5,-0.11),ncol=4,frameon=False,fontsize=7)
save(fig,"fig_mechanism.pdf")
