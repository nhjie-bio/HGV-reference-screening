from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ============================================================
# PATHS
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

V6 = BASE / "18_figures" / \
    "HGV_V3_FINAL_MANUSCRIPT_PACKAGE_v6_20260926"

SRC = V6 / "source_data"

OUT = BASE / "18_figures" / \
    "HGV_V3_PUBLICATION_FIGURES_v7_20260926"

MAIN = OUT / "main"
SUPP = OUT / "supplementary"
AUDIT = OUT / "audit"

for d in [MAIN, SUPP, AUDIT]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# READ FROZEN DATA
# ============================================================

internal = pd.read_csv(
    SRC / "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv",
    sep="\t"
)

external = pd.read_csv(
    SRC / "E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv",
    sep="\t"
)

e6m = pd.read_csv(SRC / "e6_modes.tsv", sep="\t")
e6c = pd.read_csv(SRC / "e6_contrasts.tsv", sep="\t")

e10tax = pd.read_csv(SRC / "e10_tax.tsv", sep="\t")
e10corr = pd.read_csv(SRC / "e10_corr.tsv", sep="\t")
e10conf = pd.read_csv(SRC / "e10_confound.tsv", sep="\t")

e11 = pd.read_csv(SRC / "e11.tsv", sep="\t")
e12 = pd.read_csv(SRC / "e12.tsv", sep="\t")

# ============================================================
# SCIENTIFIC GATES
# ============================================================

assert internal["method"].nunique() == 11
assert len(internal) == 33

assert external["method"].nunique() == 11
assert len(external) == 33

# ============================================================
# METHOD LABELS
# ============================================================

def pretty_method(x):
    s = str(x).lower()

    if "hgv" in s:
        return "HGV"
    if "dnabert" in s:
        return "DNABERT-S"
    if "dashing" in s:
        return "Dashing2"
    if "fastani" in s:
        return "FastANI"
    if "skani" in s:
        return "skani"
    if "mash" in s:
        return "Mash"

    if "rawk6" in s and "euclid" in s:
        return "Raw-k6 Euclidean"
    if "rawk6" in s and "cosine" in s:
        return "Raw-k6 Cosine"
    if "rawk6" in s and "bray" in s:
        return "Raw-k6 Bray-Curtis"

    if "pca64" in s and "euclid" in s:
        return "PCA64 Euclidean"
    if "pca64" in s and "l2" in s:
        return "PCA64 L2"

    return str(x)

ORDER = [
    "HGV",
    "Raw-k6 Euclidean",
    "Raw-k6 Cosine",
    "Raw-k6 Bray-Curtis",
    "PCA64 Euclidean",
    "PCA64 L2",
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
    "DNABERT-S",
]

internal["Method"] = internal["method"].map(pretty_method)
external["Method"] = external["method"].map(pretty_method)

# ============================================================
# COLOR SYSTEM
# ============================================================

# deliberately restrained
NAVY = "#183A5A"
BLUE = "#2F6B9A"
TEAL = "#2A8C82"
AMBER = "#D49A2A"
RED = "#B85C5C"

DARK = "#263238"
MID = "#687078"
LIGHT = "#DDE3E8"
VERY_LIGHT = "#F5F7F9"
HGV_BG = "#EAF2F8"

PROTOCOL_COLORS = {
    "G2": NAVY,
    "G3": TEAL,
    "G4": AMBER,
}

PROTOCOL_MARKERS = {
    "G2": "o",
    "G3": "s",
    "G4": "^",
}

COHORT_COLORS = {
    "B1": NAVY,
    "B2_seen_genus": TEAL,
    "B2_unseen_genus": RED,
}

COHORT_MARKERS = {
    "B1": "o",
    "B2_seen_genus": "s",
    "B2_unseen_genus": "^",
}

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def panel(ax, label):
    ax.text(
        -0.08, 1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top"
    )

def clean(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=LIGHT,
            linewidth=0.55,
            alpha=0.85,
            zorder=0
        )

def shade_hgv(ax, y_value):
    ax.axhspan(
        y_value - 0.42,
        y_value + 0.42,
        color=HGV_BG,
        zorder=-5
    )

def save(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            MAIN / f"{name}.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )
    plt.close(fig)

# ============================================================
# FIGURE 1
# CLEAN STUDY DESIGN
# ============================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(7.1, 4.3),
    gridspec_kw={"height_ratios": [1.05, 0.95]}
)

# ------------------------------------------------------------
# A: dataset / evaluation
# ------------------------------------------------------------

ax = axes[0]
panel(ax, "A")
ax.axis("off")

def card(ax, x, y, w, h, title, subtitle=None,
         face="white", edge=LIGHT):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9
    )
    ax.add_patch(box)

    ax.text(
        x+w/2,
        y+h*(0.62 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=DARK
    )

    if subtitle:
        ax.text(
            x+w/2,
            y+h*0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.1,
            color=MID
        )

ax.text(
    0.00, 0.97,
    "Benchmark reconstruction and hierarchical evaluation",
    fontsize=8.8,
    fontweight="bold",
    color=DARK,
    va="top"
)

# dataset pipeline
dataset = [
    (0.02, "GTDB R232", "878,998 records"),
    (0.20, "Strict QC", "172,933"),
    (0.38, "Selected", "3,200"),
    (0.56, "Frozen", "3,154 genomes"),
]

for x, title, subtitle in dataset:
    card(
        ax, x, 0.61, 0.14, 0.18,
        title, subtitle,
        face=VERY_LIGHT
    )

for i in range(len(dataset)-1):
    x1 = dataset[i][0] + 0.14
    x2 = dataset[i+1][0]

    ax.annotate(
        "",
        xy=(x2, 0.70),
        xytext=(x1, 0.70),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.9,
            color=MID
        )
    )

# cohorts
for x, title, subtitle in [
    (0.04, "A", "2,374 development"),
    (0.23, "B1", "391 matched species"),
    (0.42, "B2-seen", "196 unseen species"),
    (0.61, "B2-unseen", "193 unseen genus"),
]:
    card(
        ax, x, 0.20, 0.15, 0.18,
        title, subtitle,
        face="white"
    )

# holdouts
for x, title, subtitle in [
    (0.79, "G2", "species holdout"),
    (0.79, "G3", "genus holdout"),
    (0.79, "G4", "family holdout"),
]:
    pass

card(
    ax, 0.79, 0.61, 0.17, 0.18,
    "G2 → G3 → G4",
    "species → genus → family",
    face=HGV_BG,
    edge="#B9CEDD"
)

card(
    ax, 0.79, 0.20, 0.17, 0.18,
    "External",
    "B1 → B2-seen → B2-unseen",
    face=HGV_BG,
    edge="#B9CEDD"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ------------------------------------------------------------
# B: model / deployment
# ------------------------------------------------------------

ax = axes[1]
panel(ax, "B")
ax.axis("off")

ax.text(
    0.00, 0.97,
    "Representation and intended deployment",
    fontsize=8.8,
    fontweight="bold",
    color=DARK,
    va="top"
)

pipeline = [
    (0.02, "Whole genome", None),
    (0.21, "Canonical k=6", "2,080-D"),
    (0.42, "Encoder", "256 → 128"),
    (0.62, "HGV", "64-D"),
    (0.81, "Top-K", "vector screening"),
]

for x, title, subtitle in pipeline:
    face = HGV_BG if title == "HGV" else VERY_LIGHT

    card(
        ax, x, 0.48, 0.15, 0.20,
        title, subtitle,
        face=face,
        edge="#B9CEDD" if title == "HGV" else LIGHT
    )

for i in range(len(pipeline)-1):
    x1 = pipeline[i][0] + 0.15
    x2 = pipeline[i+1][0]

    ax.annotate(
        "",
        xy=(x2, 0.58),
        xytext=(x1, 0.58),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.9,
            color=MID
        )
    )

card(
    ax, 0.36, 0.12, 0.21, 0.16,
    "Dashing2 teacher",
    "selective informative negatives",
    face="#FBF6EA",
    edge="#DFCDA6"
)

ax.annotate(
    "",
    xy=(0.49, 0.47),
    xytext=(0.46, 0.28),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=AMBER
    )
)

card(
    ax, 0.78, 0.10, 0.18, 0.16,
    "Confirmatory ANI",
    "skani / FastANI",
    face="white"
)

ax.annotate(
    "",
    xy=(0.87, 0.27),
    xytext=(0.87, 0.48),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=NAVY
    )
)

ax.text(
    0.64, 0.20,
    "32.5× dimension reduction",
    ha="center",
    fontsize=6.6,
    color=NAVY,
    fontweight="bold"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.subplots_adjust(
    left=0.05,
    right=0.99,
    top=0.98,
    bottom=0.04,
    hspace=0.20
)

save(fig, "Figure1_study_design_v7")

# ============================================================
# FIGURE 2
# INTERNAL 11 METHOD DOT-RANGE
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 7.2),
    sharey=True
)

metric_specs = [
    ("AUROC_mean", "AUROC_sd_split_seed", "AUROC", (0.86, 1.01)),
    ("AUPRC_mean", "AUPRC_sd_split_seed", "AUPRC", (0.84, 1.01)),
    ("MCC_mean", "MCC_sd_split_seed", "MCC", (0.55, 1.01)),
    ("coverage_mean", "coverage_sd_split_seed", "Coverage", (-0.02, 1.02)),
]

y_map = {
    method: len(ORDER)-1-i
    for i, method in enumerate(ORDER)
}

offset = {
    "G2": +0.18,
    "G3": 0.00,
    "G4": -0.18,
}

for ax, spec, letter in zip(
    axes.flat,
    metric_specs,
    ["A", "B", "C", "D"]
):
    mean_col, sd_col, title, xlim = spec

    panel(ax, letter)

    shade_hgv(
        ax,
        y_map["HGV"]
    )

    for protocol in ["G2", "G3", "G4"]:
        q = internal[
            internal["protocol"] == protocol
        ].copy()

        for _, r in q.iterrows():
            method = r["Method"]

            if method not in y_map:
                continue

            y = (
                y_map[method]
                + offset[protocol]
            )

            mean = r[mean_col]
            sd = r[sd_col]

            ax.errorbar(
                mean,
                y,
                xerr=sd,
                fmt=PROTOCOL_MARKERS[protocol],
                ms=4.0,
                color=PROTOCOL_COLORS[protocol],
                ecolor=PROTOCOL_COLORS[protocol],
                capsize=1.8,
                elinewidth=0.8,
                markeredgewidth=0,
                zorder=3
            )

    ax.set_xlim(*xlim)
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        color=DARK
    )

    ax.set_xlabel(title)

    ax.set_yticks(
        [y_map[m] for m in ORDER]
    )

    ax.set_yticklabels(
        ORDER
    )

    clean(ax, "x")

# legends
handles = []

for p in ["G2", "G3", "G4"]:
    h = plt.Line2D(
        [0], [0],
        marker=PROTOCOL_MARKERS[p],
        linestyle="",
        color=PROTOCOL_COLORS[p],
        markersize=5,
        label=p
    )
    handles.append(h)

axes[0,0].legend(
    handles=handles,
    title="Holdout",
    frameon=False,
    loc="lower left"
)

fig.text(
    0.50, 0.995,
    "Internal 11-method comparison across hierarchical taxonomic holdouts",
    ha="center",
    va="top",
    fontsize=9.5,
    fontweight="bold",
    color=DARK
)

fig.subplots_adjust(
    left=0.28,
    right=0.98,
    top=0.95,
    bottom=0.07,
    hspace=0.23,
    wspace=0.18
)

save(fig, "Figure2_internal_11method_v7")

# ============================================================
# FIGURE 3
# EXTERNAL BENCHMARK + HGV OPERATING BOUNDARY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 7.0)
)

external_specs = [
    ("R1_mean", "R1_sd", "Recall@1", (0.0, 1.02)),
    ("MRR_mean", "MRR_sd", "MRR", (0.0, 1.02)),
    ("pair_coverage_mean", "pair_coverage_sd", "Pair coverage", (-0.02, 1.02)),
]

cohort_offset = {
    "B1": +0.18,
    "B2_seen_genus": 0.00,
    "B2_unseen_genus": -0.18,
}

y_map = {
    method: len(ORDER)-1-i
    for i, method in enumerate(ORDER)
}

for ax, spec, letter in zip(
    axes.flat[:3],
    external_specs,
    ["A", "B", "C"]
):
    mean_col, sd_col, title, xlim = spec

    panel(ax, letter)

    shade_hgv(
        ax,
        y_map["HGV"]
    )

    for cohort in [
        "B1",
        "B2_seen_genus",
        "B2_unseen_genus"
    ]:
        q = external[
            external["cohort"] == cohort
        ]

        for _, r in q.iterrows():
            method = r["Method"]

            if method not in y_map:
                continue

            mean = r[mean_col]
            sd = r[sd_col]

            y = (
                y_map[method]
                + cohort_offset[cohort]
            )

            ax.errorbar(
                mean,
                y,
                xerr=sd,
                fmt=COHORT_MARKERS[cohort],
                ms=4.0,
                color=COHORT_COLORS[cohort],
                ecolor=COHORT_COLORS[cohort],
                capsize=1.8,
                elinewidth=0.8,
                markeredgewidth=0
            )

    ax.set_xlim(*xlim)
    ax.set_xlabel(title)

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        color=DARK
    )

    ax.set_yticks(
        [y_map[m] for m in ORDER]
    )

    if ax in [axes[0,0], axes[1,0]]:
        ax.set_yticklabels(ORDER)
    else:
        ax.set_yticklabels([])

    clean(ax, "x")

# panel D = HGV final Recall@K
ax = axes[1,1]
panel(ax, "D")

K = [1, 5, 10, 20, 50]

label_map = {
    "B1": "B1 matched species",
    "B2_seen_genus":
        "B2 unseen species / seen genus",
    "B2_unseen_genus":
        "B2 unseen genus",
}

for _, r in e11.iterrows():
    cohort = r["external_type"]

    vals = [
        r[f"Recall@{k}"]
        for k in K
    ]

    ax.plot(
        K,
        vals,
        marker=COHORT_MARKERS[cohort],
        color=COHORT_COLORS[cohort],
        linewidth=1.3,
        markersize=4.5,
        label=label_map[cohort]
    )

    # direct label at K=50
    ax.text(
        54,
        vals[-1],
        label_map[cohort],
        va="center",
        fontsize=6.2,
        color=COHORT_COLORS[cohort]
    )

ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels(
    [str(k) for k in K]
)

ax.set_xlim(0.8, 95)
ax.set_ylim(0.45, 1.02)

ax.set_xlabel("Top-K references")
ax.set_ylabel("Recall@K")

ax.set_title(
    "HGV screening boundary",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "y")

# cohort legend only once
handles = []

for c in [
    "B1",
    "B2_seen_genus",
    "B2_unseen_genus"
]:
    handles.append(
        plt.Line2D(
            [0], [0],
            marker=COHORT_MARKERS[c],
            linestyle="",
            color=COHORT_COLORS[c],
            markersize=5,
            label=label_map[c]
        )
    )

axes[0,0].legend(
    handles=handles,
    frameon=False,
    loc="lower left",
    fontsize=5.9
)

fig.text(
    0.50, 0.995,
    "Cross-source external evaluation and reference-screening boundary",
    ha="center",
    va="top",
    fontsize=9.5,
    fontweight="bold",
    color=DARK
)

fig.subplots_adjust(
    left=0.28,
    right=0.93,
    top=0.95,
    bottom=0.07,
    hspace=0.26,
    wspace=0.18
)

save(fig, "Figure3_external_generalization_v7")

# ============================================================
# FIGURE 4
# DISTILLATION
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 4.0)
)

MODE_DESC = {
    "D0": "Contrastive only",
    "D1": "All-pair distillation",
    "D2": "Negative-only distillation",
    "D3": "Selective informative-negative",
    "D4": "Shuffled-teacher control",
    "D5": "Selective without output L2",
}

modes = ["D0","D1","D2","D3","D4","D5"]

# A
ax = axes[0]
panel(ax, "A")

z = e6m.set_index("mode").loc[modes]
y = np.arange(len(modes))[::-1]

for i, mode in enumerate(modes):
    yp = y[i]

    face = NAVY if mode == "D3" else MID

    ax.errorbar(
        z.loc[mode, "AUROC_mean"],
        yp + 0.10,
        xerr=z.loc[mode, "AUROC_sd"],
        fmt="o",
        ms=4,
        color=face,
        capsize=2,
        lw=0.8
    )

    ax.errorbar(
        z.loc[mode, "AUPRC_mean"],
        yp - 0.10,
        xerr=z.loc[mode, "AUPRC_sd"],
        fmt="s",
        ms=4,
        color=AMBER if mode == "D3" else "#A7A7A7",
        capsize=2,
        lw=0.8
    )

ax.set_yticks(y)
ax.set_yticklabels([
    f"{m}  {MODE_DESC[m]}"
    for m in modes
])

ax.set_xlim(0.93, 0.99)
ax.set_xlabel("Performance")

ax.set_title(
    "Distillation ablations",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "x")

# B/C
for ax, metric, letter in [
    (axes[1], "AUROC", "B"),
    (axes[2], "AUPRC", "C"),
]:
    panel(ax, letter)

    q = e6c[
        e6c["metric"] == metric
    ].copy()

    q = q.iloc[::-1].reset_index(
        drop=True
    )

    yy = np.arange(len(q))

    effect = q[
        "observed_delta_mean_5seed"
    ].to_numpy()

    lo = q["ci95_low"].to_numpy()
    hi = q["ci95_high"].to_numpy()

    ax.errorbar(
        effect,
        yy,
        xerr=np.vstack([
            effect-lo,
            hi-effect
        ]),
        fmt="o",
        ms=4.5,
        color=NAVY,
        ecolor=MID,
        capsize=2,
        lw=0.9
    )

    ax.axvline(
        0,
        color=DARK,
        linestyle="--",
        linewidth=0.7
    )

    ax.set_yticks(yy)
    ax.set_yticklabels(
        q["comparison"]
    )

    ax.set_xlabel(
        f"D3 − comparator Δ{metric}"
    )

    ax.set_title(
        f"{metric} paired bootstrap",
        loc="left",
        fontweight="bold",
        color=DARK
    )

    clean(ax, "x")

fig.subplots_adjust(
    left=0.30,
    right=0.99,
    top=0.92,
    bottom=0.15,
    wspace=0.46
)

save(fig, "Figure4_distillation_v7")

# ============================================================
# FIGURE 5
# BIOLOGICAL FIDELITY
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 3.9)
)

# ------------------------------------------------------------
# A taxonomy
# ------------------------------------------------------------

ax = axes[0]
panel(ax, "A")

order_tax = [
    "same_species",
    "same_genus_diff_species",
    "same_family_diff_genus",
    "different_family",
]

labels_tax = [
    "Same species",
    "Same genus / diff. species",
    "Same family / diff. genus",
    "Different family",
]

q = e10tax.set_index(
    "taxonomic_stratum"
).loc[order_tax]

yy = np.arange(4)[::-1]

ax.errorbar(
    q["hgv_distance_mean"],
    yy,
    xerr=q["hgv_distance_sd"],
    fmt="o",
    color=NAVY,
    ecolor=MID,
    capsize=2,
    ms=5
)

ax.set_yticks(yy)
ax.set_yticklabels(labels_tax)

ax.set_xlabel("HGV embedding distance")

ax.set_title(
    "Taxonomic separation",
    loc="left",
    fontweight="bold",
    color=DARK
)

for i, (_, r) in enumerate(q.iterrows()):
    ax.text(
        r["hgv_distance_mean"] + 0.05,
        yy[i],
        f'{r["hgv_distance_mean"]:.3f}',
        va="center",
        fontsize=6.2,
        color=DARK
    )

clean(ax, "x")

# ------------------------------------------------------------
# B correlation forest
# ------------------------------------------------------------

ax = axes[1]
panel(ax, "B")

subset_order = [
    ("all_reported", "All reported"),
    ("positive", "Same-species positives"),
    ("pair_type:hard", "Hard negatives"),
]

yy = np.arange(3)[::-1]

for j, target in enumerate(["ANI", "AF_min"]):
    vals = []

    for subset, label in subset_order:
        r = e10corr[
            (e10corr["subset"] == subset) &
            (e10corr["target"] == target)
        ]

        vals.append(
            r.iloc[0]["spearman_r"]
            if len(r)
            else np.nan
        )

    shift = +0.10 if target == "ANI" else -0.10

    ax.scatter(
        vals,
        yy + shift,
        marker="o" if target == "ANI" else "s",
        s=28,
        color=NAVY if target == "ANI" else AMBER,
        label="ANI" if target == "ANI" else "AFmin"
    )

    for xval, yval in zip(vals, yy + shift):
        if np.isfinite(xval):
            ax.text(
                xval - 0.025 if xval < 0 else xval + 0.025,
                yval,
                f"{xval:.3f}",
                ha="right" if xval < 0 else "left",
                va="center",
                fontsize=5.8
            )

ax.axvline(
    0,
    color=DARK,
    linestyle="--",
    linewidth=0.7
)

ax.set_yticks(yy)
ax.set_yticklabels([
    x[1] for x in subset_order
])

ax.set_xlabel("Spearman ρ")

ax.set_title(
    "Genomic fidelity",
    loc="left",
    fontweight="bold",
    color=DARK
)

ax.legend(
    frameon=False,
    loc="upper right"
)

clean(ax, "x")

# ------------------------------------------------------------
# C confound adjustment
# ------------------------------------------------------------

ax = axes[2]
panel(ax, "C")

order_conf = [
    ("all", "All reported"),
    ("positive", "Same-species positives"),
    ("hard", "Hard negatives"),
]

yy = np.arange(3)[::-1]

for i, (key, label) in enumerate(order_conf):
    r = e10conf[
        e10conf["stratum"] == key
    ].iloc[0]

    raw = r[
        "raw_spearman_hgv_vs_ani_distance"
    ]

    adj = r[
        "partial_rank_correlation"
    ]

    y = yy[i]

    ax.plot(
        [raw, adj],
        [y, y],
        color="#B8BDC2",
        linewidth=1.3
    )

    ax.scatter(
        raw, y,
        color=NAVY,
        s=28,
        zorder=3,
        label="Raw" if i == 0 else None
    )

    ax.scatter(
        adj, y,
        color=AMBER,
        marker="s",
        s=28,
        zorder=3,
        label="Adjusted" if i == 0 else None
    )

    ax.text(
        max(raw, adj) + 0.035,
        y,
        f'{raw:.3f} → {adj:.3f}\nP={r["permutation_two_sided_p"]:.3f}',
        va="center",
        fontsize=5.8
    )

ax.axvline(
    0,
    color=DARK,
    linestyle="--",
    linewidth=0.7
)

ax.set_yticks(yy)
ax.set_yticklabels([
    x[1] for x in order_conf
])

ax.set_xlabel("Rank association")

ax.set_title(
    "Covariate adjustment",
    loc="left",
    fontweight="bold",
    color=DARK
)

ax.legend(
    frameon=False,
    loc="upper left"
)

clean(ax, "x")

fig.subplots_adjust(
    left=0.20,
    right=0.98,
    top=0.91,
    bottom=0.18,
    wspace=0.50
)

save(fig, "Figure5_biological_fidelity_v7")

# ============================================================
# FIGURE 6
# COMPUTATIONAL PROFILE ONLY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 5.3)
)

# A storage
ax = axes[0,0]
panel(ax, "A")

ax.plot(
    e12["N_reference"],
    e12["vector_index_bytes_median"] / 1024,
    marker="o",
    color=NAVY,
    linewidth=1.4,
    markersize=4
)

ax.set_xlabel("Reference genomes")
ax.set_ylabel("64-D index size (KiB)")

ax.set_title(
    "Vector index storage",
    loc="left",
    fontweight="bold"
)

ax.text(
    0.97, 0.08,
    "256 bytes/genome",
    transform=ax.transAxes,
    ha="right",
    color=NAVY,
    fontsize=6.5,
    fontweight="bold"
)

clean(ax, "y")

# B feature extraction
ax = axes[0,1]
panel(ax, "B")

ax.plot(
    e12["N_reference"],
    e12["reference_feature_wall_s_median"],
    marker="o",
    color=NAVY,
    linewidth=1.4,
    markersize=4
)

ax.set_xlabel("Reference genomes")
ax.set_ylabel("Wall time (s)")

ax.set_title(
    "Reference feature extraction",
    loc="left",
    fontweight="bold"
)

clean(ax, "y")

# C end-to-end comparison
ax = axes[1,0]
panel(ax, "C")

ax.plot(
    e12["N_reference"],
    e12["reference_feature_wall_s_median"],
    marker="o",
    color=NAVY,
    linewidth=1.3,
    markersize=4,
    label="Feature extraction"
)

ax.plot(
    e12["N_reference"],
    e12["end_to_end_wall_s_median"],
    marker="s",
    color=AMBER,
    linewidth=1.3,
    markersize=4,
    label="End-to-end"
)

ax.set_xlabel("Reference genomes")
ax.set_ylabel("Wall time (s)")

ax.set_title(
    "End-to-end scaling",
    loc="left",
    fontweight="bold"
)

ax.legend(
    frameon=False,
    loc="upper left"
)

clean(ax, "y")

# D decomposition
ax = axes[1,1]
panel(ax, "D")

r = e12.loc[
    e12["N_reference"].idxmax()
]

labels = [
    "Reference features",
    "50 query features",
    "Reference inference",
    "Query inference",
    "Warm vector scan",
]

vals = np.array([
    r["reference_feature_wall_s_median"],
    r["query_feature_wall_s_median"],
    r["reference_model_inference_wall_s_median"],
    r["query_model_inference_wall_s_median"],
    r["exact_scan_warm_median_s_median"],
])

yy = np.arange(len(labels))[::-1]

bars = ax.barh(
    yy,
    vals,
    color=[
        NAVY,
        "#4E7896",
        "#7D9AB0",
        "#A8BCCB",
        "#CDD9E1",
    ]
)

ax.set_xscale("log")
ax.set_yticks(yy)
ax.set_yticklabels(labels)

ax.set_xlabel("Seconds (log scale)")

ax.set_title(
    "Runtime decomposition (N=2,374)",
    loc="left",
    fontweight="bold"
)

for bar, value in zip(bars, vals):
    ax.text(
        value * 1.15,
        bar.get_y() + bar.get_height()/2,
        f"{value:.4g} s",
        va="center",
        fontsize=5.8
    )

clean(ax, "x")

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.93,
    bottom=0.12,
    hspace=0.32,
    wspace=0.33
)

save(fig, "Figure6_computational_profile_v7")

# ============================================================
# CONTACT SHEET
# ============================================================

import matplotlib.image as mpimg

files = [
    MAIN / "Figure1_study_design_v7.png",
    MAIN / "Figure2_internal_11method_v7.png",
    MAIN / "Figure3_external_generalization_v7.png",
    MAIN / "Figure4_distillation_v7.png",
    MAIN / "Figure5_biological_fidelity_v7.png",
    MAIN / "Figure6_computational_profile_v7.png",
]

fig, axes = plt.subplots(
    3, 2,
    figsize=(13.5, 15)
)

for ax, p in zip(
    axes.flat,
    files
):
    ax.axis("off")

    if p.exists():
        ax.imshow(
            mpimg.imread(p)
        )
        ax.set_title(
            p.stem,
            fontsize=8
        )

fig.tight_layout()

fig.savefig(
    OUT / "CONTACT_SHEET_v7.png",
    dpi=170,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# AUDIT
# ============================================================

missing = [
    p for p in files
    if not p.exists()
]

with open(
    AUDIT / "PUBLICATION_FIGURE_AUDIT_v7.txt",
    "w"
) as f:
    f.write(
        "HGV-V3 PUBLICATION FIGURE AUDIT v7\n"
    )
    f.write("="*72 + "\n")

    for p in files:
        f.write(
            f"{'PASS' if p.exists() else 'FAIL'}\t{p.name}\n"
        )

    f.write("\n")

    if missing:
        f.write(
            "PUBLICATION_FIGURE_GATE=FAIL\n"
        )
    else:
        f.write(
            "PUBLICATION_FIGURE_GATE=PASS\n"
        )

print("="*72)

for p in files:
    print(
        "[PASS]" if p.exists() else "[FAIL]",
        p.name
    )

print()
print(
    "PUBLICATION_FIGURE_GATE="
    + ("PASS" if not missing else "FAIL")
)

print(
    "CONTACT SHEET =",
    OUT / "CONTACT_SHEET_v7.png"
)

print("="*72)

if missing:
    raise SystemExit(2)

