from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

# ============================================================
# PATHS
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

ROOT = BASE / "11_manuscript" / \
    "HGV_V3_final_manuscript_figures_v3_20260926"

SRC = ROOT / "00_source_snapshot" / "source_data"

OUT = BASE / "18_figures" / \
    "HGV_V3_FINAL_MANUSCRIPT_FIGURES_v5_20260926"

MAIN = OUT / "main"
SUPP = OUT / "supplementary"
AUDIT = OUT / "audit"

for d in [MAIN, SUPP, AUDIT]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIXED CANONICAL SOURCES
# ============================================================

P_INTERNAL = SRC / "main_internal.tsv"
P_EXTERNAL = SRC / "external_raw.tsv"
P_E6MODES = SRC / "e6_modes.tsv"
P_E6CONTRAST = SRC / "e6_contrasts.tsv"
P_E7 = SRC / "e7.tsv"
P_E8 = SRC / "e8.tsv"
P_E9 = SRC / "e9_ig.tsv"
P_E10CORR = SRC / "e10_corr.tsv"
P_E10TAX = SRC / "e10_tax.tsv"
P_E10CONF = SRC / "e10_confound.tsv"
P_E10TREE = SRC / "e10_tree_pairs.tsv"
P_E10TREESUM = SRC / "e10_tree_summary.tsv"
P_E11 = SRC / "e11.tsv"
P_E12 = SRC / "e12.tsv"

required = [
    P_INTERNAL, P_EXTERNAL,
    P_E6MODES, P_E6CONTRAST,
    P_E7, P_E8, P_E9,
    P_E10CORR, P_E10TAX, P_E10CONF,
    P_E10TREE, P_E10TREESUM,
    P_E11, P_E12
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# READ
# ============================================================

internal = pd.read_csv(P_INTERNAL, sep="\t")
external = pd.read_csv(P_EXTERNAL, sep="\t")
e6m = pd.read_csv(P_E6MODES, sep="\t")
e6c = pd.read_csv(P_E6CONTRAST, sep="\t")
e7 = pd.read_csv(P_E7, sep="\t")
e8 = pd.read_csv(P_E8, sep="\t")
e9 = pd.read_csv(P_E9, sep="\t")
e10corr = pd.read_csv(P_E10CORR, sep="\t")
e10tax = pd.read_csv(P_E10TAX, sep="\t")
e10conf = pd.read_csv(P_E10CONF, sep="\t")
e10tree = pd.read_csv(P_E10TREE, sep="\t")
e10treesum = pd.read_csv(P_E10TREESUM, sep="\t")
e11 = pd.read_csv(P_E11, sep="\t")
e12 = pd.read_csv(P_E12, sep="\t")

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "HGV_Dashing2_tau0p98": "#0072B2",
    "PCA64_Euclidean": "#E69F00",
    "PCA64_L2": "#F0E442",
    "RawK6_BrayCurtis": "#009E73",
    "RawK6_Cosine": "#56B4E9",
    "RawK6_Euclidean": "#CC79A7",
}

SHORT = {
    "HGV_Dashing2_tau0p98": "HGV",
    "PCA64_Euclidean": "PCA64-Euclidean",
    "PCA64_L2": "PCA64-L2",
    "RawK6_BrayCurtis": "Raw-k6 Bray-Curtis",
    "RawK6_Cosine": "Raw-k6 Cosine",
    "RawK6_Euclidean": "Raw-k6 Euclidean",
}

COHORT_LABELS = {
    "B1": "B1\nmatched species",
    "B2_seen_genus": "B2\nunseen species\nseen genus",
    "B2_unseen_genus": "B2\nunseen genus",
}

def panel(ax, letter):
    ax.text(
        -0.12, 1.08, letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )

def clean(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", alpha=0.18, lw=0.6)

def save(fig, name, folder):
    fig.savefig(folder / f"{name}.png",
                dpi=600, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf",
                bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg",
                bbox_inches="tight")
    plt.close(fig)

# ============================================================
# FIGURE 1
# Study design
# ============================================================

fig = plt.figure(figsize=(7.2, 6.0))
gs = gridspec.GridSpec(
    3, 2,
    hspace=0.48,
    wspace=0.30
)

# A
ax = fig.add_subplot(gs[0,0])
panel(ax, "A")
ax.axis("off")
ax.set_title("Dataset reconstruction and QC", loc="left")

nodes = [
    ("GTDB R232 metadata\n878,998 genomes", 0.82),
    ("Strict metadata QC\n172,933 candidates", 0.59),
    ("Primary selection\n3,200 genomes", 0.36),
    ("Sequence QC + redundancy audit\n3,154 retained", 0.13),
]

for text, y in nodes:
    box = FancyBboxPatch(
        (0.10, y-0.075), 0.80, 0.15,
        boxstyle="round,pad=0.015",
        facecolor="#F4F4F4",
        edgecolor="#555555",
        lw=0.8
    )
    ax.add_patch(box)
    ax.text(0.50, y, text,
            ha="center", va="center")

for y1, y2 in [(0.745,0.665),(0.515,0.435),(0.285,0.205)]:
    ax.annotate(
        "",
        xy=(0.50,y2),
        xytext=(0.50,y1),
        arrowprops=dict(arrowstyle="->", lw=0.9)
    )

ax.set_xlim(0,1)
ax.set_ylim(0,1)

# B
ax = fig.add_subplot(gs[0,1])
panel(ax, "B")

cohort_names = [
    "Development A",
    "B1",
    "B2 seen-genus",
    "B2 unseen-genus"
]

cohort_n = [2374,391,196,193]

bars = ax.bar(
    np.arange(4),
    cohort_n,
    edgecolor="black",
    lw=0.5
)

ax.set_xticks(np.arange(4))
ax.set_xticklabels(
    ["A", "B1", "B2-seen", "B2-unseen"]
)
ax.set_ylabel("Number of genomes")
ax.set_title("Final analysis cohorts", loc="left")

for b,v in zip(bars, cohort_n):
    ax.text(
        b.get_x()+b.get_width()/2,
        v+40,
        f"{v:,}",
        ha="center",
        fontsize=7.5
    )

clean(ax)

# C
ax = fig.add_subplot(gs[1,0])
panel(ax, "C")
ax.axis("off")
ax.set_title("Hierarchical generalization", loc="left")

items = [
    (0.17, "G2", "Unseen species"),
    (0.50, "G3", "Unseen genus"),
    (0.83, "G4", "Unseen family"),
]

for x, code, txt in items:
    ax.scatter(x,0.62,s=1600,
               facecolors="white",
               edgecolors="black")
    ax.text(x,0.62,code,
            ha="center",va="center",
            fontweight="bold")
    ax.text(x,0.27,txt,
            ha="center",va="center")

ax.annotate(
    "",
    xy=(0.92,0.88),
    xytext=(0.08,0.88),
    arrowprops=dict(arrowstyle="->",lw=1.0)
)

ax.text(
    0.5,0.93,
    "Increasing taxonomic novelty",
    ha="center"
)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

# D
ax = fig.add_subplot(gs[1,1])
panel(ax, "D")
ax.axis("off")
ax.set_title("HGV representation", loc="left")

arch = [
    ("Whole\nassembly",0.02,0.16),
    ("Canonical\nk=6\n2,080-D",0.24,0.18),
    ("256",0.50,0.10),
    ("128",0.66,0.10),
    ("64-D\nL2",0.82,0.14),
]

for txt,x,w in arch:
    box=FancyBboxPatch(
        (x,0.47),w,0.28,
        boxstyle="round,pad=0.012",
        facecolor="#F4F4F4",
        edgecolor="#555555",
        lw=0.8
    )
    ax.add_patch(box)
    ax.text(x+w/2,0.61,txt,
            ha="center",va="center")

for a,b in zip(arch[:-1],arch[1:]):
    ax.annotate(
        "",
        xy=(b[1],0.61),
        xytext=(a[1]+a[2],0.61),
        arrowprops=dict(arrowstyle="->",lw=0.8)
    )

ax.text(
    0.5,0.22,
    "2,080 → 64 dimensions (32.5× reduction)",
    ha="center",
    fontweight="bold"
)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

# E
ax = fig.add_subplot(gs[2,0])
panel(ax, "E")
ax.axis("off")
ax.set_title("Selective distillation design", loc="left")

for txt,x,y in [
    ("Contrastive\nobjective",0.07,0.61),
    ("Dashing2\nteacher",0.07,0.18),
    ("HGV\nstudent",0.69,0.40),
]:
    box=FancyBboxPatch(
        (x,y),0.22,0.20,
        boxstyle="round,pad=0.012",
        facecolor="#F4F4F4",
        lw=0.8
    )
    ax.add_patch(box)
    ax.text(x+0.11,y+0.10,txt,
            ha="center",va="center")

ax.annotate(
    "",xy=(0.69,0.54),xytext=(0.29,0.70),
    arrowprops=dict(arrowstyle="->",lw=0.9)
)
ax.annotate(
    "",xy=(0.69,0.46),xytext=(0.29,0.28),
    arrowprops=dict(arrowstyle="->",lw=0.9)
)

ax.text(
    0.49,0.24,
    "Informative negatives:\nteacher distance < 0.98",
    ha="center"
)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

# F
ax = fig.add_subplot(gs[2,1])
panel(ax, "F")
ax.axis("off")
ax.set_title("Reference-screening use case", loc="left")

steps = [
    ("Query",0.01),
    ("64-D\nHGV",0.20),
    ("Vector\nsearch",0.39),
    ("Top-K",0.58),
    ("skani /\nFastANI",0.77),
]

for txt,x in steps:
    box=FancyBboxPatch(
        (x,0.45),0.15,0.23,
        boxstyle="round,pad=0.01",
        facecolor="#F4F4F4",
        lw=0.8
    )
    ax.add_patch(box)
    ax.text(x+0.075,0.565,txt,
            ha="center",va="center")

for a,b in zip(steps[:-1],steps[1:]):
    ax.annotate(
        "",
        xy=(b[1],0.565),
        xytext=(a[1]+0.15,0.565),
        arrowprops=dict(arrowstyle="->",lw=0.8)
    )

ax.text(
    0.5,0.22,
    "First-stage screening → confirmatory ANI",
    ha="center",
    fontweight="bold"
)

ax.set_xlim(0,1)
ax.set_ylim(0,1)

fig.suptitle(
    "HGV-V3 study design and evaluation framework",
    fontsize=11
)

save(fig,"Figure1_study_design",MAIN)

# ============================================================
# FIGURE 2
# Internal G2/G3/G4
# ============================================================

fig, axes = plt.subplots(
    1,3,
    figsize=(7.2,3.2)
)

metrics = [
    ("AUROC_mean","AUROC_sd_split_seed","AUROC"),
    ("AUPRC_mean","AUPRC_sd_split_seed","AUPRC"),
    ("MCC_mean","MCC_sd_split_seed","MCC"),
]

methods = internal["method"].unique().tolist()
protocols = ["G2","G3","G4"]

offsets = np.linspace(-0.27,0.27,len(methods))

for ax,(mean_col,sd_col,title),letter in zip(
    axes,metrics,["A","B","C"]
):
    panel(ax,letter)

    for mi,m in enumerate(methods):
        z = internal[internal["method"]==m]

        xs=[]
        ys=[]
        es=[]

        for pi,p in enumerate(protocols):
            row=z[z["protocol"]==p]

            if len(row)!=1:
                continue

            xs.append(pi+offsets[mi])
            ys.append(row.iloc[0][mean_col])
            es.append(row.iloc[0][sd_col])

        ax.errorbar(
            xs,ys,yerr=es,
            fmt="o",
            ms=4.5,
            capsize=2,
            lw=0.8,
            color=COLORS[m],
            label=SHORT[m]
        )

    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["G2\nspecies","G3\ngenus","G4\nfamily"]
    )

    ax.set_ylabel(title)
    ax.set_title(
        f"{title} across taxonomic holdouts",
        loc="left"
    )

    if title in ["AUROC","AUPRC"]:
        ax.set_ylim(0.93,1.01)
    else:
        ax.set_ylim(0.72,1.01)

    clean(ax)

axes[0].legend(
    frameon=False,
    fontsize=6.3,
    loc="lower left"
)

fig.suptitle(
    "Internal hierarchical generalization",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.94])
save(fig,"Figure2_internal_generalization",MAIN)

# ============================================================
# FIGURE 3
# External method comparison
# ============================================================

fig, axes = plt.subplots(
    1,3,
    figsize=(7.2,3.4),
    sharey=True
)

targets = [
    ("B1","species_R1","Species Recall@1"),
    ("B2_seen_genus","genus_R1","Genus Recall@1"),
    ("B2_unseen_genus","family_R1","Family Recall@1"),
]

for ax,(cohort,metric,title),letter in zip(
    axes,targets,["A","B","C"]
):
    panel(ax,letter)

    z = external[
        (external["cohort"]==cohort) &
        (external["metric"]==metric)
    ].dropna(subset=["mean"]).copy()

    if z.empty:
        ax.text(
            0.5,0.5,
            f"{metric}\nnot available",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        continue

    z["short"]=z["method"].map(SHORT)

    z=z.sort_values(
        "mean",
        ascending=True
    )

    y=np.arange(len(z))

    ax.errorbar(
        z["mean"],
        y,
        xerr=z["sd_model_seed"].fillna(0),
        fmt="o",
        capsize=2,
        lw=0.8
    )

    ax.set_yticks(y)
    ax.set_yticklabels(z["short"])
    ax.set_xlim(0,1.03)

    ax.set_xlabel(title)
    ax.set_title(
        COHORT_LABELS[cohort].replace("\n"," "),
        loc="left"
    )

    clean(ax)

fig.suptitle(
    "Cross-source external retrieval across increasing taxonomic novelty",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.93])
save(fig,"Figure3_external_generalization",MAIN)

# ============================================================
# FIGURE 4
# E6 ablation + formal contrasts
# ============================================================

fig, axes = plt.subplots(
    2,2,
    figsize=(7.2,5.4)
)

modes=["D0","D1","D2","D3","D4","D5"]
x=np.arange(6)

for ax,metric,sd,title,letter in [
    (axes[0,0],"AUROC_mean","AUROC_sd","AUROC","A"),
    (axes[0,1],"AUPRC_mean","AUPRC_sd","AUPRC","B"),
]:
    panel(ax,letter)

    z=e6m.set_index("mode").loc[modes]

    ax.errorbar(
        x,
        z[metric],
        yerr=z[sd],
        fmt="o",
        ms=6,
        capsize=3,
        lw=1
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0.92,1.005)
    ax.set_ylabel(title)
    ax.set_title(
        f"D0–D5 {title} (mean ± SD, n=5 seeds)",
        loc="left"
    )

    clean(ax)

for ax,metric,letter in [
    (axes[1,0],"AUROC","C"),
    (axes[1,1],"AUPRC","D"),
]:
    panel(ax,letter)

    z=e6c[
        e6c["metric"]==metric
    ].copy()

    z=z.iloc[::-1].reset_index(drop=True)

    y=np.arange(len(z))

    center=z["observed_delta_mean_5seed"].to_numpy()
    lo=z["ci95_low"].to_numpy()
    hi=z["ci95_high"].to_numpy()

    xerr=np.vstack([
        center-lo,
        hi-center
    ])

    ax.errorbar(
        center,y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        lw=1
    )

    ax.axvline(
        0,
        color="black",
        ls="--",
        lw=0.8
    )

    ax.set_yticks(y)
    ax.set_yticklabels(z["comparison"])

    ax.set_xlabel(f"Δ{metric}")
    ax.set_title(
        "D3 paired bootstrap contrasts",
        loc="left"
    )

    clean(ax)

fig.suptitle(
    "Selective distillation ablation and formal paired analysis",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.95])
save(fig,"Figure4_distillation_ablation",MAIN)

# ============================================================
# FIGURE 5
# Biological fidelity
# ============================================================

fig, axes = plt.subplots(
    1,3,
    figsize=(7.2,3.5)
)

# A taxonomy
ax=axes[0]
panel(ax,"A")

order=[
    "same_species",
    "same_genus_diff_species",
    "same_family_diff_genus",
    "different_family"
]

labels=[
    "Same\nspecies",
    "Same genus /\ndiff. species",
    "Same family /\ndiff. genus",
    "Different\nfamily"
]

z=e10tax.set_index(
    "taxonomic_stratum"
).loc[order]

xx=np.arange(4)

ax.errorbar(
    xx,
    z["hgv_distance_mean"],
    yerr=z["hgv_distance_sd"],
    fmt="o-",
    capsize=3,
    lw=1.2
)

ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylabel("HGV embedding distance")
ax.set_title(
    "Taxonomic distance gradient",
    loc="left"
)

clean(ax)

# B ANI / AF correlations
ax=axes[1]
panel(ax,"B")

subset_order=[
    "all_reported",
    "positive",
    "pair_type:hard"
]

display=[
    "All reported",
    "Same species",
    "Hard negatives"
]

width=0.14

for ti,target in enumerate(["ANI","AF_min"]):
    vals=[]

    for subset in subset_order:
        q=e10corr[
            (e10corr["subset"]==subset) &
            (e10corr["target"]==target)
        ]

        vals.append(
            q.iloc[0]["spearman_r"]
            if len(q)
            else np.nan
        )

    xpos=np.arange(3)+(ti-0.5)*width

    ax.scatter(
        xpos,
        vals,
        s=38,
        label="ANI" if target=="ANI" else "AFmin"
    )

ax.axhline(0,color="black",lw=0.7,ls="--")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(display,rotation=20,ha="right")
ax.set_ylabel("Spearman ρ")
ax.set_title(
    "Association with ANI and AF",
    loc="left"
)
ax.legend(frameon=False)
clean(ax)

# C confound
ax=axes[2]
panel(ax,"C")

order_conf=["all","positive","hard"]

z=e10conf.set_index(
    "stratum"
).loc[order_conf].reset_index()

y=np.arange(len(z))

ax.scatter(
    z["raw_spearman_hgv_vs_ani_distance"],
    y+0.09,
    s=42,
    label="Raw"
)

ax.scatter(
    z["partial_rank_correlation"],
    y-0.09,
    s=42,
    label="Adjusted"
)

for i,row in z.iterrows():
    ax.plot(
        [
            row["raw_spearman_hgv_vs_ani_distance"],
            row["partial_rank_correlation"]
        ],
        [i+0.09,i-0.09],
        color="#999999",
        lw=0.8
    )

ax.axvline(0,color="black",ls="--",lw=0.7)
ax.set_yticks(y)
ax.set_yticklabels(
    ["All reported","Same species","Hard negatives"]
)
ax.set_xlabel("Rank association")
ax.set_title(
    "After covariate adjustment",
    loc="left"
)
ax.legend(frameon=False)
clean(ax)

fig.suptitle(
    "Biological fidelity of the HGV embedding space",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.93])
save(fig,"Figure5_biological_fidelity",MAIN)

# ============================================================
# FIGURE 6
# Screening and scalability
# ============================================================

fig = plt.figure(figsize=(7.2,5.6))
gs = gridspec.GridSpec(
    2,2,
    hspace=0.42,
    wspace=0.34
)

# A Recall@K
ax=fig.add_subplot(gs[0,0])
panel(ax,"A")

K=[1,5,10,20,50]

for _,row in e11.iterrows():
    vals=[
        row[f"Recall@{k}"]
        for k in K
    ]

    ax.plot(
        K,vals,
        marker="o",
        lw=1.3,
        label=COHORT_LABELS[row["external_type"]].replace("\n"," ")
    )

ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_ylim(0.45,1.02)
ax.set_xlabel("Top-K candidates")
ax.set_ylabel("Recall@K")
ax.set_title("Reference recovery",loc="left")
ax.legend(frameon=False,fontsize=6.3)
clean(ax)

# B candidate reduction
ax=fig.add_subplot(gs[0,1])
panel(ax,"B")

red=[
    e11.iloc[0][f"candidate_reduction@{k}"]*100
    for k in K
]

ax.plot(
    K,red,
    marker="o",
    lw=1.3
)

ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.set_ylabel("Database eliminated (%)")
ax.set_xlabel("Top-K candidates retained")
ax.set_ylim(97,100.05)
ax.set_title("Candidate reduction",loc="left")
clean(ax)

# C scaling
ax=fig.add_subplot(gs[1,0])
panel(ax,"C")

ax.plot(
    e12["N_reference"],
    e12["reference_feature_wall_s_median"],
    marker="o",
    label="Reference feature extraction"
)

ax.plot(
    e12["N_reference"],
    e12["end_to_end_wall_s_median"],
    marker="s",
    label="End-to-end"
)

ax.set_xlabel("Reference genomes")
ax.set_ylabel("Wall time (s)")
ax.set_title("Real-FASTA runtime scaling",loc="left")
ax.legend(frameon=False,fontsize=6.5)
clean(ax)

# D cheap stages
ax=fig.add_subplot(gs[1,1])
panel(ax,"D")

row=e12[
    e12["N_reference"]==2374
].iloc[0]

names=[
    "Reference\nfeatures",
    "50 query\nfeatures",
    "Reference\ninference",
    "Query\ninference",
    "Warm vector\nscan"
]

vals=[
    row["reference_feature_wall_s_median"],
    row["query_feature_wall_s_median"],
    row["reference_model_inference_wall_s_median"],
    row["query_model_inference_wall_s_median"],
    row["exact_scan_warm_median_s_median"]
]

ax.bar(
    np.arange(len(vals)),
    vals
)

ax.set_yscale("log")
ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(names,rotation=25,ha="right")
ax.set_ylabel("Seconds (log scale)")
ax.set_title(
    "Runtime decomposition at N=2,374",
    loc="left"
)
clean(ax)

ax.text(
    0.97,0.96,
    f"Vector index = {row['vector_index_bytes_median']/1024:.1f} KiB\n"
    f"{row['bytes_per_genome_median']:.0f} bytes/genome",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
    bbox=dict(
        facecolor="white",
        edgecolor="#999999",
        boxstyle="round,pad=0.25"
    )
)

fig.suptitle(
    "Reference-screening utility and computational scalability",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.95])
save(fig,"Figure6_screening_scalability",MAIN)

# ============================================================
# FIGURE S1
# Sensitivity
# ============================================================

fig,axes=plt.subplots(
    2,2,
    figsize=(7.0,5.0)
)

canonical=e7[
    e7["setting"]=="canonical_k6_dim64_margin1_lambda0p25"
].iloc[0]

configs=[
    (
        axes[0,0],
        ["k5","canonical_k6_dim64_margin1_lambda0p25","k7"],
        ["k=5","k=6","k=7"],
        "k-mer size",
        "A"
    ),
    (
        axes[0,1],
        ["dim32","canonical_k6_dim64_margin1_lambda0p25","dim128"],
        ["32","64","128"],
        "Embedding dimension",
        "B"
    ),
    (
        axes[1,0],
        ["margin0p5","canonical_k6_dim64_margin1_lambda0p25","margin1p5"],
        ["0.5","1.0","1.5"],
        "Contrastive margin",
        "C"
    ),
    (
        axes[1,1],
        ["lambda0p1","canonical_k6_dim64_margin1_lambda0p25","lambda0p5"],
        ["0.10","0.25","0.50"],
        "Distillation weight λ",
        "D"
    )
]

for ax,settings,labels,title,letter in configs:
    panel(ax,letter)

    z=e7.set_index("setting").loc[settings]

    xx=np.arange(3)

    ax.errorbar(
        xx,
        z["test_AUROC_mean"],
        yerr=z["test_AUROC_sd"],
        marker="o",
        capsize=3,
        label="AUROC"
    )

    ax.errorbar(
        xx,
        z["test_AUPRC_mean"],
        yerr=z["test_AUPRC_sd"],
        marker="s",
        capsize=3,
        label="AUPRC"
    )

    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.88,1.0)
    ax.set_title(title,loc="left")
    clean(ax)

axes[0,0].set_ylabel("Performance")
axes[1,0].set_ylabel("Performance")
axes[0,0].legend(frameon=False)

fig.suptitle(
    "Sensitivity analysis of HGV-V3",
    fontsize=11
)

fig.tight_layout(rect=[0,0,1,0.95])
save(fig,"FigureS1_sensitivity",SUPP)

# ============================================================
# FIGURE S2
# Fragmentation
# ============================================================

fig,ax=plt.subplots(figsize=(5.6,3.4))
panel(ax,"A")

order=["full","100kb","50kb","20kb","5kb","5x5kb"]
labels=["Full","100 kb","50 kb","20 kb","5 kb","5×5 kb"]

z=e8.set_index("condition").loc[order]

xx=np.arange(6)

ax.errorbar(
    xx,
    z["AUROC_mean"],
    yerr=z["AUROC_sd"],
    marker="o",
    capsize=3,
    label="AUROC"
)

ax.errorbar(
    xx,
    z["AUPRC_mean"],
    yerr=z["AUPRC_sd"],
    marker="s",
    capsize=3,
    label="AUPRC"
)

ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylim(0.55,1.01)
ax.set_ylabel("Performance")
ax.set_xlabel("Available genomic context")
ax.set_title(
    "Fragmentation sensitivity",
    loc="left"
)
ax.legend(frameon=False)
clean(ax)

save(fig,"FigureS2_fragmentation",SUPP)

# ============================================================
# FIGURE S3
# Integrated gradients
# ============================================================

fig,ax=plt.subplots(figsize=(5.8,4.3))
panel(ax,"A")

top=e9.nsmallest(
    20,
    "rank"
).sort_values(
    "mean_abs_pair_IG",
    ascending=True
)

y=np.arange(len(top))

ax.barh(
    y,
    top["mean_abs_pair_IG"]
)

ax.set_yticks(y)
ax.set_yticklabels(top["canonical_kmer"])
ax.set_xlabel("Mean absolute pair integrated gradient")
ax.set_title(
    "Top canonical 6-mer feature sensitivities",
    loc="left"
)

clean(ax)

ax.text(
    0.98,0.02,
    "Feature sensitivity only;\nnot functional motif evidence",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7
)

save(fig,"FigureS3_integrated_gradients",SUPP)

# ============================================================
# FIGURE S4
# GTDB tree consistency
# ============================================================

fig,ax=plt.subplots(figsize=(5.4,4.0))
panel(ax,"A")

hb=ax.hexbin(
    e10tree["gtdb_tree_patristic_distance"],
    e10tree["hgv_distance_mean_3seed"],
    gridsize=45,
    mincnt=1,
    cmap="viridis"
)

rho=e10treesum.iloc[0]["spearman_rho"]
pperm=e10treesum.iloc[0]["mantel_style_two_sided_p"]

ax.text(
    0.04,0.96,
    f"Spearman ρ = {rho:.3f}\n"
    f"Permutation P = {pperm:.4f}",
    transform=ax.transAxes,
    va="top"
)

ax.set_xlabel("GTDB bac120 patristic distance")
ax.set_ylabel("HGV distance")
ax.set_title(
    "GTDB tree-distance consistency",
    loc="left"
)

clean(ax)

save(fig,"FigureS4_GTDB_tree_consistency",SUPP)

# ============================================================
# CONTACT SHEET
# ============================================================

import matplotlib.image as mpimg

pngs=[
    MAIN/"Figure1_study_design.png",
    MAIN/"Figure2_internal_generalization.png",
    MAIN/"Figure3_external_generalization.png",
    MAIN/"Figure4_distillation_ablation.png",
    MAIN/"Figure5_biological_fidelity.png",
    MAIN/"Figure6_screening_scalability.png",
]

fig,axes=plt.subplots(
    3,2,
    figsize=(14,15)
)

for ax,p in zip(axes.flat,pngs):
    ax.axis("off")
    img=mpimg.imread(p)
    ax.imshow(img)
    ax.set_title(p.stem,fontsize=9)

fig.tight_layout()

fig.savefig(
    OUT/"MAIN_FIGURES_CONTACT_SHEET.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# AUDIT
# ============================================================

expected = pngs + [
    SUPP/"FigureS1_sensitivity.png",
    SUPP/"FigureS2_fragmentation.png",
    SUPP/"FigureS3_integrated_gradients.png",
    SUPP/"FigureS4_GTDB_tree_consistency.png",
]

missing=[p for p in expected if not p.exists()]

with open(AUDIT/"FINAL_FIGURE_AUDIT_v5.txt","w") as f:
    f.write("HGV-V3 FINAL MANUSCRIPT FIGURE AUDIT v5\n")
    f.write("="*70+"\n")

    for p in expected:
        f.write(
            f"{'PASS' if p.exists() else 'FAIL'}\t{p}\n"
        )

    f.write("\n")
    f.write(
        f"FINAL_FIGURE_GATE={'PASS' if not missing else 'FAIL'}\n"
    )

print("="*80)
print("HGV-V3 FINAL MANUSCRIPT FIGURES v5")
print("="*80)

for p in expected:
    print(
        "[PASS]" if p.exists() else "[FAIL]",
        p.name
    )

print()
print(
    "FINAL_FIGURE_GATE="
    + ("PASS" if not missing else "FAIL")
)

print("OUTPUT =",OUT)
print("CONTACT =",OUT/"MAIN_FIGURES_CONTACT_SHEET.png")
print("="*80)

if missing:
    raise SystemExit(2)
