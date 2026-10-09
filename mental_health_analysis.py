import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

TENK = "annotated_fanar_10kposts_Mentalhealth.csv"
GOLD = "Mental Health.xlsx - Sheet1.csv"
FIGDIR = "figures"

import os
os.makedirs(FIGDIR, exist_ok=True)
matplotlib.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                            "axes.spines.top": False, "axes.spines.right": False})

ORDER = ["Personal Experience & Help-Seeking", "Peer Support & Encouragement",
         "Informational/Educational", "Promotional/Service",
         "Stigmatization/Dismissal"]
ALL6 = ORDER + ["Other/Unclear"]
SHORT = {"Personal Experience & Help-Seeking": "Personal Exp.\n& Help-Seeking",
         "Peer Support & Encouragement": "Peer Support",
         "Informational/Educational": "Informational",
         "Promotional/Service": "Promotional",
         "Stigmatization/Dismissal": "Stigmatization"}
BARCOL = ["#2c6e8f", "#3d8bb0", "#5aa9c9", "#e08a3c", "#c0504d"]


def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False),
                         errors="coerce")


# ---- RQ2: orientation distribution and engagement (10k sample) ----
df = pd.read_csv(TENK)
df["TI"] = to_numeric(df["Total Interactions"])
for c in ["Comments", "Shares", "Love", "Wow", "Haha", "Sad", "Angry",
          "Care", "Followers at Posting"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
N = len(df)

print("Sample size:", N)

vc = df["llm_label"].value_counts()
print("\nOrientation distribution:")
for lab in ALL6:
    print(f"  {lab:38s} {100*vc.get(lab, 0)/N:5.1f}%")
print("  multi-label rows:", df["llm_label"].astype(str).str.contains(",").sum())

med = df.groupby("llm_label")[["TI", "Comments", "Shares"]].median()
print("\nMedian engagement by orientation:")
for lab in ORDER:
    print(f"  {lab:34s} interactions={med.loc[lab,'TI']:.0f} "
          f"comments={med.loc[lab,'Comments']:.0f} shares={med.loc[lab,'Shares']:.0f}")

H, p = kruskal(*[df[df.llm_label == l]["TI"].dropna() for l in ORDER])
print(f"\nKruskal-Wallis on interactions: H={H:.1f}, p={p:.2e}")

sub = df[(df["Followers at Posting"] > 0) & df["TI"].notna()].copy()
sub["rate"] = 1000 * sub["TI"] / sub["Followers at Posting"]
print(f"\nFollower-normalised (coverage {100*len(sub)/N:.1f}%), interactions per 1,000 followers:")
rmed = sub.groupby("llm_label")["rate"].median()
for lab in ORDER:
    print(f"  {lab:38s} {rmed[lab]:.2f}")
Hn, pn = kruskal(*[sub[sub.llm_label == l]["rate"].dropna() for l in ORDER])
print(f"  Kruskal-Wallis on rate: H={Hn:.1f}, p={pn:.2e}")

pairs = list(itertools.combinations(ORDER, 2))
nc = len(pairs)
print(f"\nPairwise Mann-Whitney (Bonferroni over {nc} pairs), with rank-biserial:")
for a, b in pairs:
    x = df[df.llm_label == a]["TI"].dropna()
    y = df[df.llm_label == b]["TI"].dropna()
    U, pr = mannwhitneyu(x, y, alternative="two-sided")
    rbc = 1 - 2*U/(len(x)*len(y))
    flag = "sig" if pr*nc < 0.05 else "n.s."
    print(f"  {a[:14]:14s} vs {b[:14]:14s}  p_bonf={min(1, pr*nc):.1e}  rbc={rbc:+.3f}  {flag}")

react = ["Love", "Wow", "Haha", "Sad", "Angry", "Care"]
comp = df.groupby("llm_label")[react].sum()
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
print("\nReaction mix:")
print(f"  Promotional Love %   : {comp_pct.loc['Promotional/Service','Love']:.0f}")
print(f"  Peer Support Love %  : {comp_pct.loc['Peer Support & Encouragement','Love']:.0f}")
print(f"  Stigmatization Angry %: {comp_pct.loc['Stigmatization/Dismissal','Angry']:.1f}")
print(f"  Personal Exp. Haha % : {comp_pct.loc['Personal Experience & Help-Seeking','Haha']:.1f}")

# Figure: median total interactions by orientation
mint = df.groupby("llm_label")["TI"].median().reindex(ORDER)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bars = ax.bar([SHORT[l] for l in ORDER], mint.values, color=BARCOL)
for b, v in zip(bars, mint.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("Median total interactions"); ax.margins(y=.15); plt.xticks(fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq2_engagement.png", dpi=200, bbox_inches="tight"); plt.close()

# Figure: reaction-mix composition by orientation
rcolors = {"Love": "#e15759", "Care": "#f2a65a", "Sad": "#4e79a7",
           "Angry": "#8c564b", "Haha": "#59a14f", "Wow": "#b07aa1"}
fig, ax = plt.subplots(figsize=(7.6, 3.8))
bottom = np.zeros(len(ORDER))
for r in ["Love", "Care", "Sad", "Angry", "Haha", "Wow"]:
    vals = comp_pct.reindex(ORDER)[r].values
    ax.bar([SHORT[l] for l in ORDER], vals, bottom=bottom, label=r, color=rcolors[r])
    bottom += vals
ax.set_ylabel("Share of reactions (%)"); ax.set_ylim(0, 100)
ax.legend(ncol=6, frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
plt.xticks(fontsize=8.5); plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_rq2_reactions.png", dpi=200, bbox_inches="tight"); plt.close()

# Figure: median comments per post by orientation
mcom = df.groupby("llm_label")["Comments"].median().reindex(ORDER)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
bars = ax.bar([SHORT[l] for l in ORDER], mcom.values, color=BARCOL)
for b, v in zip(bars, mcom.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("Median comments per post"); ax.margins(y=.15); plt.xticks(fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq2_comments.png", dpi=200, bbox_inches="tight"); plt.close()


# ---- RQ3: reliability against the 1k human gold standard ----
g = pd.read_csv(GOLD)
h = g["Manual Annotation"].str.strip()
m = g["llm_label"].str.strip()
print("\nGold-standard size:", len(g))

print(f"Overall agreement: {100*(h==m).mean():.1f}%")
print(f"Overall kappa    : {cohen_kappa_score(h, m):.3f}")

print(f"\nHuman used Other/Unclear: {(h=='Other/Unclear').sum()} posts")
print(f"Fanar used Other/Unclear: {(m=='Other/Unclear').sum()} posts")
oth = m[h == "Other/Unclear"]
print(f"  human-Other -> Fanar Promotional  : {(oth=='Promotional/Service').sum()}")
print(f"  human-Other -> Fanar Informational: {(oth=='Informational/Educational').sum()}")
print(f"Total disagreements: {(h!=m).sum()}")

mask = h != "Other/Unclear"
print(f"\nIn-domain subset: {mask.sum()} posts")
print(f"  agreement: {100*(h[mask]==m[mask]).mean():.1f}%")
print(f"  kappa    : {cohen_kappa_score(h[mask], m[mask]):.2f}")

P, R, F, S = precision_recall_fscore_support(h, m, labels=ALL6, zero_division=0)
print("\nPer-category precision / recall:")
for lab, pr, rc, sp in zip(ALL6, P, R, S):
    print(f"  {lab:38s} precision={pr:.2f} recall={rc:.2f} support={int(sp)}")

for lab in ["Informational/Educational", "Promotional/Service"]:
    s = h == lab
    print(f"When human={lab:26s}: Fanar agrees {100*(m[s]==lab).mean():.1f}%")

print("\nOther/Unclear inflation rates:")
print(f"  P(human Other | Fanar Informational): {100*(h[m=='Informational/Educational']=='Other/Unclear').mean():.0f}%")
print(f"  P(human Other | Fanar Promotional)  : {100*(h[m=='Promotional/Service']=='Other/Unclear').mean():.0f}%")

cm = confusion_matrix(h, m, labels=ALL6)
print("\nConfusion matrix (rows=human, cols=Fanar):")
print(pd.DataFrame(cm, index=ALL6, columns=ALL6).to_string())

SHORT6 = ["Personal\nExp.", "Peer\nSupport", "Informational", "Promotional",
          "Stigmat.", "Other/\nUnclear"]

# Figure: confusion matrix (row-normalised)
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(SHORT6, fontsize=8.5); ax.set_yticklabels(SHORT6, fontsize=8.5)
ax.set_xlabel("Fanar label"); ax.set_ylabel("Human label")
for i in range(6):
    for j in range(6):
        if cm[i, j] > 0:
            ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if cmn[i, j] > 0.5 else "#333")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Row-normalised share")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq3_confusion.png", dpi=200, bbox_inches="tight"); plt.close()

# Figure: precision and recall on the five content categories
fig, ax = plt.subplots(figsize=(7, 3.3))
x = np.arange(5)
ax.bar(x-0.2, P[:5], 0.4, label="Precision", color="#5aa9c9")
ax.bar(x+0.2, R[:5], 0.4, label="Recall", color="#2c6e8f")
ax.set_xticks(x); ax.set_xticklabels([SHORT6[i].replace("\n", " ") for i in range(5)], fontsize=8.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq3_prf.png", dpi=200, bbox_inches="tight"); plt.close()

print("\nFigures written to ./figures/: fig_rq2_engagement, fig_rq2_reactions, "
      "fig_rq2_comments, fig_rq3_confusion, fig_rq3_prf")