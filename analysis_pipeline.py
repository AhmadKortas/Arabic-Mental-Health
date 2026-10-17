#!/usr/bin/env python3
# =============================================================================
#  Help-Seeking Orientation in Arabic Mental Health Discourse on Facebook
#  ---------------------------------------------------------------------------
#  Analysis pipeline. This script computes every statistic reported in the
#  paper and regenerates every figure. Running it end-to-end reproduces:
#
#     RQ1  corpus characterisation      (raw corpus, 297,697 posts)
#     RQ2  orientation & engagement     (10,000 Fanar-annotated posts)
#     RQ3  annotation reliability       (1,000 human-annotated gold posts)
#
#  Inputs (place this file in the same directory as the data):
#     D1 : 06-2014_to_06-2024_الصحة_النفسية_OR_الرعاية_النفسية_OR_الاضطرابات_النفسية.csv
#     D2 : العلاج_النفسي_OR_الصحة_العقلية_OR_الخدمات_النفسية.csv
#     D3 : _MentalHealth_OR__SelfCare_OR__Mindfulness_OR__الصحة_النفسية_OR__العناية_بالنفس__1_.csv
#     10k: annotated_fanar_10kposts_Mentalhealth.csv
#     1k : Mental_Health_xlsx_-_Sheet1.csv
#
#  Outputs: all statistics printed to stdout, and ten PNG figures written to
#           ./figures/ (the same figures used in the paper).
#
#  Requires: pandas, numpy, scipy, scikit-learn, matplotlib
# =============================================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
FILES = {
    "D1": "06-2014 to 06-2024 الصحة النفسية OR الرعاية النفسية OR الاضطرابات النفسية.csv",
    "D2": "العلاج النفسي OR الصحة العقلية OR الخدمات النفسية.csv",
    "D3": "#MentalHealth OR #SelfCare OR #Mindfulness OR #الصحة_النفسية OR #العناية_بالنفس.csv",
}
TENK = "annotated_fanar_10kposts_Mentalhealth.csv"
GOLD = "Mental Health.xlsx - Sheet1.csv"
FIGDIR = "figures"

# The five substantive orientation labels, in the paper's reporting order.
ORDER = ["Personal Experience & Help-Seeking", "Peer Support & Encouragement",
         "Informational/Educational", "Promotional/Service",
         "Stigmatization/Dismissal"]
ALL6 = ORDER + ["Other/Unclear"]
SHORT = {"Personal Experience & Help-Seeking": "Personal Exp.\n& Help-Seeking",
         "Peer Support & Encouragement": "Peer Support",
         "Informational/Educational": "Informational",
         "Promotional/Service": "Promotional",
         "Stigmatization/Dismissal": "Stigmatization"}

os.makedirs(FIGDIR, exist_ok=True)
matplotlib.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                            "axes.spines.top": False, "axes.spines.right": False})


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------
def to_numeric(series):
    """Parse an integer column that may use thousands separators (e.g. '5,043').

    'Total Interactions' in the CrowdTangle export is comma-formatted; a plain
    pandas.to_numeric would coerce those cells to NaN and silently drop the most
    engaged posts. Stripping the commas first is required for correct medians.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False),
                         errors="coerce")


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def account_type(cat):
    """Map a raw CrowdTangle page category to one of nine account types.

    Rules are applied in order; the first match wins. This reproduces Table 1
    exactly (note COMMUNITY_ORGANIZATION is grouped with Non-profit / NGO).
    """
    c = str(cat)
    if any(k in c for k in ["NEWS", "MEDIA", "TV", "BROADCAST", "MAGAZINE",
                            "NEWSPAPER", "JOURNAL"]):
        return "Media / News"
    if any(k in c for k in ["COUNSEL", "HOSPITAL", "MEDICAL", "HEALTH", "CLINIC",
                            "DOCTOR", "PSYCH", "THERAP", "DENTIST", "PHARMA"]):
        return "Health / Clinical"
    if any(k in c for k in ["EDUCATION", "SCHOOL", "UNIVERSITY", "TEACHER",
                            "TRAINING", "COURSE"]):
        return "Education"
    if c in ("PERSON", "PUBLIC_FIGURE", "AUTHOR", "ARTIST", "WRITER"):
        return "Individual / Public figure"
    if any(k in c for k in ["NON_PROFIT", "NGO", "CHARITY", "COMMUNITY_ORGANIZATION"]):
        return "Non-profit / NGO"
    if any(k in c for k in ["GOVERNMENT", "PUBLIC_SERVICE"]):
        return "Government"
    if any(k in c for k in ["COMMUNITY", "SOCIETY", "CLUB"]):
        return "Community group"
    if any(k in c for k in ["COMPANY", "BRAND", "BUSINESS", "RETAIL", "LOCAL",
                            "SHOP", "PRODUCT", "SERVICE", "BEAUTY", "ENTREPRENEUR"]):
        return "Business / Brand"
    return "Other / Unspecified"


# =============================================================================
#  SECTION 1 — Load the raw corpus (the three collection streams)
# =============================================================================
def load_corpus():
    usecols = ["Page Name", "Page Category", "Page Admin Top Country",
               "Post Created", "Type", "Total Interactions", "Likes", "Comments",
               "Shares", "Love", "Wow", "Haha", "Sad", "Angry", "Care",
               "Message", "Followers at Posting"]
    frames = []
    for stream, path in FILES.items():
        d = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
        d["dataset"] = stream
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
#  SECTION 2 — RQ1 statistics (Data Collection + Corpus Characterisation)
# =============================================================================
def compute_rq1(d):
    banner("RQ1  —  CORPUS CHARACTERISATION")
    N = len(d)
    print(f"Total posts (D1+D2+D3)            : {N:,}")
    vc = d["dataset"].value_counts()
    for s in ["D1", "D2", "D3"]:
        print(f"  {s} posts                        : {vc[s]:,}  ({100*vc[s]/N:.1f}%)")

    # -- cross-stream overlap / duplication -----------------------------------
    key = ["Page Name", "Post Created", "Message"]
    in_dupgroup = d.duplicated(subset=key, keep=False).sum()      # both members of each pair
    redundant   = d.duplicated(subset=key, keep="first").sum()    # removable copies only
    cross       = int((d.groupby(key)["dataset"].nunique() > 1).sum())
    print(f"\nRecords in an exact-duplicate group  : {in_dupgroup:,}  ({100*in_dupgroup/N:.1f}%)")
    print(f"  of which redundant (removable)     : {redundant:,}  ({100*redundant/N:.1f}%)")
    print(f"Distinct posts spanning >1 stream    : {cross:,}")

    # -- text / language ------------------------------------------------------
    print(f"\nNon-empty message field              : {100*d['Message'].notna().mean():.1f}%")
    sample = d["Message"].dropna().sample(min(20000, d["Message"].notna().sum()),
                                          random_state=1)
    arabic = sample.str.contains(r"[\u0600-\u06FF]", regex=True).mean()
    print(f"Arabic script (random 20k sample)    : {100*arabic:.1f}%")

    # -- temporal -------------------------------------------------------------
    d["dt"] = pd.to_datetime(d["Post Created"].str.replace(" AST", "", regex=False),
                             errors="coerce")
    d["year"] = d["dt"].dt.year
    yr = d.groupby("year").size()
    print(f"\nDate range                           : {d['dt'].min().date()} to {d['dt'].max().date()}")
    print("Annual post counts:")
    for y in sorted(yr.index):
        print(f"  {int(y)}                             : {yr[y]:,}")

    # -- accounts -------------------------------------------------------------
    pages = d["Page Name"].value_counts()
    top1 = pages.head(max(1, pages.size // 100)).sum()
    print(f"\nDistinct pages                       : {pages.size:,}")
    print(f"Top-1% pages' share of posts         : {100*top1/N:.1f}%")
    print(f"Distinct raw page categories         : {d['Page Category'].nunique():,}")

    d["acct"] = d["Page Category"].apply(account_type)
    acct = d["acct"].value_counts()
    print(f"Account types after grouping         : {acct.size}")
    print("Account-type distribution (Table 1):")
    for name in ["Other / Unspecified", "Health / Clinical", "Media / News",
                 "Business / Brand", "Education", "Non-profit / NGO",
                 "Community group", "Individual / Public figure", "Government"]:
        print(f"  {name:28s} : {acct[name]:>7,}  ({100*acct[name]/N:.1f}%)")

    # -- geography ------------------------------------------------------------
    geo = d["Page Admin Top Country"].value_counts()
    print(f"\nDistinct admin countries             : {d['Page Admin Top Country'].nunique()}")
    print(f"Top-10 countries' share              : {100*geo.head(10).sum()/N:.0f}%")
    print(f"Egypt share                          : {100*geo.get('EG',0)/N:.1f}%")

    # -- post format ----------------------------------------------------------
    pt = d["Type"].value_counts()
    print("\nPost format:")
    for t in ["Photo", "Link", "Status", "Native Video"]:
        print(f"  {t:12s}                     : {100*pt.get(t,0)/N:.1f}%")

    # -- engagement -----------------------------------------------------------
    d["TI"] = to_numeric(d["Total Interactions"])
    print(f"\nInteractions  median / mean          : {d['TI'].median():.0f} / {d['TI'].mean():.1f}")
    print(f"Zero-interaction posts               : {100*(d['TI']==0).mean():.1f}%")
    med_ds = d.groupby("dataset")["TI"].median()
    print(f"Median interactions  D1 / D2 / D3    : {med_ds['D1']:.0f} / {med_ds['D2']:.0f} / {med_ds['D3']:.0f}")
    for c in ["Love", "Sad", "Care"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    print(f"Reaction totals  Love / Sad / Care   : "
          f"{d['Love'].sum()/1e6:.2f}M / {d['Sad'].sum()/1e6:.2f}M / {d['Care'].sum()/1e6:.2f}M")
    return d


# =============================================================================
#  SECTION 3 — RQ1 figures
# =============================================================================
def figures_rq1(d):
    NAME = {"D1": "D1 clinical (health/care/disorders)",
            "D2": "D2 clinical (therapy/services)",
            "D3": "D3 hashtag (#MentalHealth)"}
    COL = {"D1": "#2c6e8f", "D2": "#5aa9c9", "D3": "#e08a3c"}

    # Figure 1 — monthly posting volume, stacked by stream
    d["ym"] = d["dt"].dt.to_period("M").astype(str)
    mon = d.dropna(subset=["dt"]).groupby(["ym", "dataset"]).size().unstack(fill_value=0)
    mon.index = pd.to_datetime(mon.index); mon = mon.sort_index()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.stackplot(mon.index, [mon[c] for c in ["D1", "D2", "D3"]],
                 labels=[NAME[c] for c in ["D1", "D2", "D3"]],
                 colors=[COL[c] for c in ["D1", "D2", "D3"]], alpha=.9)
    ax.set_ylabel("Posts per month"); ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_temporal.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 2 — account types (horizontal bar)
    acct = d["acct"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(acct.index, acct.values, color="#2c6e8f")
    for i, (k, v) in enumerate(acct.items()):
        ax.text(v + max(acct.values)*.01, i, f"{100*v/len(d):.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Number of posts"); ax.margins(x=.12); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_accounts.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 3 — top-10 admin countries
    geo = d["Page Admin Top Country"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(geo.index, geo.values, color="#5aa9c9")
    for i, (k, v) in enumerate(geo.items()):
        ax.text(i, v + max(geo.values)*.01, f"{100*v/len(d):.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("Posts"); ax.set_xlabel("Page admin top country"); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_geo.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 4 — post format
    pt = d["Type"].value_counts()
    photo, link, status, nv = (pt.get("Photo", 0), pt.get("Link", 0),
                               pt.get("Status", 0), pt.get("Native Video", 0))
    fmt = pd.Series({"Photo": photo, "Link": link, "Status": status,
                     "Native\nVideo": nv, "Other": len(d)-(photo+link+status+nv)})
    fmt_pct = 100*fmt/len(d)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(fmt_pct.index, fmt_pct.values,
                  color=["#2c6e8f", "#3d8bb0", "#5aa9c9", "#7bc0d8", "#cdd7db"])
    for b, v in zip(bars, fmt_pct.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Share of posts (%)"); ax.set_ylim(0, 72); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_post_format.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 5 — reaction repertoire
    for c in ["Wow", "Haha", "Angry"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    rr = d[["Love", "Sad", "Haha", "Care", "Wow", "Angry"]].sum()/1e6
    colors = {"Love": "#e15759", "Sad": "#4e79a7", "Haha": "#59a14f",
              "Care": "#f2a65a", "Wow": "#b07aa1", "Angry": "#8c564b"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(rr.index, rr.values, color=[colors[k] for k in rr.index])
    for b, v in zip(bars, rr.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.06, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_ylabel("Total reactions (millions)"); ax.set_ylim(0, 4.2); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_reactions_repertoire.png", dpi=200, bbox_inches="tight"); plt.close()
    print("\n[figures] wrote fig_temporal, fig_accounts, fig_geo, fig_post_format, fig_reactions_repertoire")


# =============================================================================
#  SECTION 4 — RQ2 statistics (orientation distribution + engagement)
# =============================================================================
def compute_rq2():
    banner("RQ2  —  HELP-SEEKING ORIENTATION AND ENGAGEMENT (10k sample)")
    df = pd.read_csv(TENK)
    df["TI"] = to_numeric(df["Total Interactions"])
    for c in ["Comments", "Shares", "Love", "Wow", "Haha", "Sad", "Angry",
              "Care", "Followers at Posting"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    N = len(df)
    print(f"Annotated sample size                : {N:,}")

    # -- distribution ---------------------------------------------------------
    vc = df["llm_label"].value_counts()
    print("Orientation distribution:")
    for lab in ORDER + ["Other/Unclear"]:
        print(f"  {lab:38s} : {100*vc.get(lab,0)/N:5.1f}%")
    print(f"  multi-label rows                     : "
          f"{df['llm_label'].astype(str).str.contains(',').sum()}")

    # -- median engagement (Table 2 + Figure 6) -------------------------------
    med = df.groupby("llm_label")[["TI", "Comments", "Shares"]].median()
    print("\nMedian engagement by orientation (Table 2):")
    print(f"  {'orientation':32s} {'inter.':>7} {'comm.':>6} {'shares':>7}")
    for lab in ORDER:
        print(f"  {lab:32s} {med.loc[lab,'TI']:7.0f} "
              f"{med.loc[lab,'Comments']:6.0f} {med.loc[lab,'Shares']:7.0f}")

    # -- omnibus test ---------------------------------------------------------
    H, p = kruskal(*[df[df.llm_label == l]["TI"].dropna() for l in ORDER])
    print(f"\nKruskal-Wallis (interactions)        : H = {H:.1f}, p = {p:.2e}")

    # -- follower-normalised robustness check ---------------------------------
    sub = df[(df["Followers at Posting"] > 0) & df["TI"].notna()].copy()
    sub["rate"] = 1000 * sub["TI"] / sub["Followers at Posting"]
    print(f"\nFollower-normalised (coverage {100*len(sub)/N:.1f}%): interactions per 1,000 followers")
    rmed = sub.groupby("llm_label")["rate"].median()
    for lab in ORDER:
        print(f"  {lab:38s} : {rmed[lab]:.2f}")
    Hn, pn = kruskal(*[sub[sub.llm_label == l]["rate"].dropna() for l in ORDER])
    print(f"  Kruskal-Wallis (rate)                : H = {Hn:.1f}, p = {pn:.2e}")

    # -- post-hoc pairwise (Bonferroni) + rank-biserial effect size -----------
    pairs = list(itertools.combinations(ORDER, 2)); nc = len(pairs)
    inst = {"Informational/Educational", "Promotional/Service"}
    print(f"\nPost-hoc Mann-Whitney (Bonferroni, {nc} pairs) with rank-biserial:")
    for a, b in pairs:
        x = df[df.llm_label == a]["TI"].dropna()
        y = df[df.llm_label == b]["TI"].dropna()
        U, p = mannwhitneyu(x, y, alternative="two-sided")
        rbc = 1 - 2*U/(len(x)*len(y))
        sig = "sig " if p*nc < 0.05 else "n.s."
        print(f"  {SHORT[a].split(chr(10))[0]:14s} vs {SHORT[b].split(chr(10))[0]:14s} "
              f": p_bonf={min(1,p*nc):.1e}  rbc={rbc:+.3f}  [{sig}]")

    # -- reaction-mix composition (Figure 7) ----------------------------------
    react = ["Love", "Wow", "Haha", "Sad", "Angry", "Care"]
    comp = df.groupby("llm_label")[react].sum()
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    print("\nReaction-mix highlights:")
    print(f"  Promotional Love %                   : {comp_pct.loc['Promotional/Service','Love']:.0f}")
    print(f"  Peer Support Love %                  : {comp_pct.loc['Peer Support & Encouragement','Love']:.0f}")
    print(f"  Stigmatization Angry %               : {comp_pct.loc['Stigmatization/Dismissal','Angry']:.1f}")
    print(f"  Personal Exp. Haha %                 : {comp_pct.loc['Personal Experience & Help-Seeking','Haha']:.1f}")
    return df, med, comp_pct


# =============================================================================
#  SECTION 5 — RQ2 figures
# =============================================================================
def figures_rq2(df, comp_pct):
    order5 = [l for l in ORDER]

    # Figure 6 — median total interactions
    med = df.groupby("llm_label")["TI"].median().reindex(order5)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar([SHORT[l] for l in order5], med.values,
                  color=["#2c6e8f", "#3d8bb0", "#5aa9c9", "#e08a3c", "#c0504d"])
    for b, v in zip(bars, med.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Median total interactions"); ax.margins(y=.15); plt.xticks(fontsize=8.5)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq2_engagement.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 7 — reaction-mix stacked bars
    colors = {"Love": "#e15759", "Care": "#f2a65a", "Sad": "#4e79a7",
              "Angry": "#8c564b", "Haha": "#59a14f", "Wow": "#b07aa1"}
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bottom = np.zeros(len(order5))
    for r in ["Love", "Care", "Sad", "Angry", "Haha", "Wow"]:
        ax.bar([SHORT[l] for l in order5], comp_pct.reindex(order5)[r], bottom=bottom,
               label=r, color=colors[r])
        bottom += comp_pct.reindex(order5)[r].values
    ax.set_ylabel("Share of reactions (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=6, frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    plt.xticks(fontsize=8.5); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_rq2_reactions.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 8 — median comments per post
    medc = df.groupby("llm_label")["Comments"].median().reindex(order5)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.bar([SHORT[l] for l in order5], medc.values,
                  color=["#2c6e8f", "#3d8bb0", "#5aa9c9", "#e08a3c", "#c0504d"])
    for b, v in zip(bars, medc.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("Median comments per post"); ax.margins(y=.15); plt.xticks(fontsize=8.5)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq2_comments.png", dpi=200, bbox_inches="tight"); plt.close()
    print("\n[figures] wrote fig_rq2_engagement, fig_rq2_reactions, fig_rq2_comments")


# =============================================================================
#  SECTION 6 — RQ3 statistics (reliability vs. human gold standard)
# =============================================================================
def compute_rq3():
    banner("RQ3  —  ANNOTATION RELIABILITY (1k human gold standard)")
    g = pd.read_csv(GOLD)
    h = g["Manual Annotation"].str.strip()
    m = g["llm_label"].str.strip()
    print(f"Gold-standard size                   : {len(g):,}")

    print(f"\nOverall agreement                    : {100*(h==m).mean():.1f}%")
    print(f"Overall Cohen's kappa                : {cohen_kappa_score(h,m):.3f}")

    print(f"\nHuman used Other/Unclear on          : {(h=='Other/Unclear').sum()} posts")
    print(f"Fanar used Other/Unclear on          : {(m=='Other/Unclear').sum()} posts")
    oth = m[h == "Other/Unclear"]
    print(f"  human-Other -> Fanar Promotional   : {(oth=='Promotional/Service').sum()}")
    print(f"  human-Other -> Fanar Informational : {(oth=='Informational/Educational').sum()}")
    print(f"Total disagreements                  : {(h!=m).sum()}")

    mask = h != "Other/Unclear"
    print(f"\nIn-domain subset                     : {mask.sum()} posts")
    print(f"  in-domain agreement                : {100*(h[mask]==m[mask]).mean():.1f}%")
    print(f"  in-domain kappa                    : {cohen_kappa_score(h[mask],m[mask]):.2f}")

    P, R, F, S = precision_recall_fscore_support(h, m, labels=ALL6, zero_division=0)
    print("\nPer-category precision / recall (human = reference):")
    print(f"  {'label':38s} {'prec':>5} {'rec':>5} {'support':>8}")
    for lab, p, r, s in zip(ALL6, P, R, S):
        print(f"  {lab:38s} {p:5.2f} {r:5.2f} {int(s):8d}")

    for lab in ["Informational/Educational", "Promotional/Service"]:
        sub = h == lab
        print(f"When human = {lab:26s}: Fanar agrees {100*(m[sub]==lab).mean():.1f}%")

    print("\nOther/Unclear inflation rates (quoted in RQ2):")
    print(f"  P(human Other | Fanar Informational): {100*(h[m=='Informational/Educational']=='Other/Unclear').mean():.0f}%")
    print(f"  P(human Other | Fanar Promotional)  : {100*(h[m=='Promotional/Service']=='Other/Unclear').mean():.0f}%")
    return g, h, m


# =============================================================================
#  SECTION 7 — RQ3 figures
# =============================================================================
def figures_rq3(h, m):
    short = ["Personal\nExp.", "Peer\nSupport", "Informational", "Promotional",
             "Stigmat.", "Other/\nUnclear"]

    # Figure 9 — confusion matrix (row-normalised)
    cm = confusion_matrix(h, m, labels=ALL6).astype(float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(short, fontsize=8.5); ax.set_yticklabels(short, fontsize=8.5)
    ax.set_xlabel("Fanar label"); ax.set_ylabel("Human label")
    for i in range(6):
        for j in range(6):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white" if cmn[i, j] > 0.5 else "#333")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Row-normalised share")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq3_confusion.png", dpi=200, bbox_inches="tight"); plt.close()

    # Figure 10 — precision & recall on the five content categories
    P, R, F, S = precision_recall_fscore_support(h, m, labels=ALL6, zero_division=0)
    idx = list(range(5))
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(5)
    ax.bar(x-0.2, [P[i] for i in idx], 0.4, label="Precision", color="#5aa9c9")
    ax.bar(x+0.2, [R[i] for i in idx], 0.4, label="Recall", color="#2c6e8f")
    ax.set_xticks(x); ax.set_xticklabels([short[i].replace("\n", " ") for i in idx], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig_rq3_prf.png", dpi=200, bbox_inches="tight"); plt.close()
    print("\n[figures] wrote fig_rq3_confusion, fig_rq3_prf")


# =============================================================================
#  Main
# =============================================================================
if __name__ == "__main__":
    corpus = load_corpus()
    corpus = compute_rq1(corpus)
    figures_rq1(corpus)

    df10k, med2, comp_pct = compute_rq2()
    figures_rq2(df10k, comp_pct)

    gold, h, m = compute_rq3()
    figures_rq3(h, m)

    banner("DONE — all statistics computed; 10 figures written to ./figures/")
