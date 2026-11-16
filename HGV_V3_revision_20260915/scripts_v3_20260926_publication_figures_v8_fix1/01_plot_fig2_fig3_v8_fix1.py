from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "HGV_V3_revision_20260915"
)

SRC = (
    BASE /
    "18_figures" /
    "HGV_V3_FINAL_MANUSCRIPT_PACKAGE_v6_20260926" /
    "source_data"
)

OUT = (
    BASE /
    "18_figures" /
    "HGV_V3_PUBLICATION_FIGURES_v8_fix1_20260926"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA
# ============================================================

internal = pd.read_csv(
    SRC / "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv",
    sep="\t"
)

e11 = pd.read_csv(
    SRC / "e11.tsv",
    sep="\t"
)


# ============================================================
# METHOD NAMES
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

    if "pca64" in s and "euclid" in s:
        return "PCA64 Euclidean"

    return str(x)


internal["Method"] = (
    internal["method"]
    .map(pretty_method)
)


# ============================================================
# COLORS
# ============================================================

NAVY = "#173F5F"
TEAL = "#3A8D89"
AMBER = "#C58E2A"
RED = "#AA5A5A"

DARK = "#242A30"
GRAY = "#707980"
LIGHT = "#DCE3E8"
PALE = "#EDF4F7"


PROTOCOLS = [
    "G2",
    "G3",
    "G4"
]

PCOLOR = {
    "G2": NAVY,
    "G3": TEAL,
    "G4": AMBER,
}

PMARKER = {
    "G2": "o",
    "G3": "s",
    "G4": "^",
}


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 8,

    "axes.titlesize": 9,

    "axes.labelsize": 8,

    "xtick.labelsize": 7.5,

    "ytick.labelsize": 7.5,

    "legend.fontsize": 7,

    "axes.linewidth": 0.75,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

    "figure.facecolor": "white",

    "axes.facecolor": "white",
})


def panel(ax, label):

    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top"
    )


def clean(ax, axis="x"):

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    if axis:

        ax.grid(
            axis=axis,
            color=LIGHT,
            linewidth=0.55,
            alpha=0.8,
            zorder=0
        )


def save(fig, name):

    for ext in [
        "png",
        "pdf",
        "svg"
    ]:

        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )

    plt.close(fig)


# ============================================================
# FIGURE 2
# ============================================================

fig = plt.figure(
    figsize=(7.2, 6.0)
)

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[
        0.78,
        1.22
    ],
    hspace=0.48,
    wspace=0.50
)


# ============================================================
# FIGURE 2A
# HGV hierarchical generalization
# ============================================================

ax = fig.add_subplot(
    gs[0, :]
)

panel(ax, "A")


hgv = internal[
    internal["Method"] == "HGV"
].copy()


metric_specs = [

    (
        "AUROC",
        "AUROC_mean",
        "AUROC_sd_split_seed",
    ),

    (
        "AUPRC",
        "AUPRC_mean",
        "AUPRC_sd_split_seed",
    ),

    (
        "MCC",
        "MCC_mean",
        "MCC_sd_split_seed",
    ),
]


metric_y = {
    "AUROC": 2,
    "AUPRC": 1,
    "MCC": 0
}


offset = {
    "G2": +0.18,
    "G3": 0,
    "G4": -0.18
}


for metric, mean_col, sd_col in metric_specs:

    for protocol in PROTOCOLS:

        q = hgv[
            hgv["protocol"] == protocol
        ]

        if len(q) != 1:

            raise RuntimeError(
                f"HGV {protocol} missing"
            )

        row = q.iloc[0]

        mean = float(
            row[mean_col]
        )

        sd = float(
            row[sd_col]
        )

        y = (
            metric_y[metric]
            + offset[protocol]
        )

        ax.errorbar(

            mean,
            y,

            xerr=sd,

            fmt=PMARKER[protocol],

            color=PCOLOR[protocol],

            ecolor=PCOLOR[protocol],

            markersize=5.5,

            capsize=2.2,

            linewidth=1,

            zorder=3
        )

        # fixed numeric column
        ax.text(

            1.007,
            y,

            f"{mean:.3f}",

            ha="left",
            va="center",

            fontsize=6.5,

            color=PCOLOR[protocol]
        )


ax.set_xlim(
    0.76,
    1.045
)

ax.set_yticks(
    [2, 1, 0]
)

ax.set_yticklabels(
    [
        "AUROC",
        "AUPRC",
        "MCC"
    ]
)

ax.set_xlabel(
    "Performance"
)

ax.set_title(
    "HGV performance under increasingly strict taxonomic holdouts",
    loc="left",
    fontweight="bold",
    pad=8
)

clean(
    ax,
    "x"
)


handles = [

    Line2D(
        [0],
        [0],
        marker=PMARKER[p],
        linestyle="",
        color=PCOLOR[p],
        markersize=6,
        label={
            "G2": "G2 species",
            "G3": "G3 genus",
            "G4": "G4 family",
        }[p]
    )

    for p in PROTOCOLS
]


ax.legend(
    handles=handles,
    frameon=False,
    ncol=3,
    loc="lower left",
    bbox_to_anchor=(
        0.00,
        -0.45
    )
)


# ============================================================
# FIGURE 2B
# Representation baselines
# ============================================================

ax = fig.add_subplot(
    gs[1, 0]
)

panel(ax, "B")


methods = [

    "HGV",

    "Raw-k6 Euclidean",

    "PCA64 Euclidean",

    "DNABERT-S",
]


ypos = {
    m: len(methods)-1-i
    for i, m in enumerate(methods)
}


ax.axhspan(

    ypos["HGV"] - 0.40,

    ypos["HGV"] + 0.40,

    color=PALE,

    zorder=-5
)


for method in methods:

    q = internal[
        internal["Method"] == method
    ]

    for protocol in PROTOCOLS:

        r = q[
            q["protocol"] == protocol
        ]

        if len(r) != 1:
            continue

        row = r.iloc[0]

        mean = float(
            row["MCC_mean"]
        )

        sd = float(
            row["MCC_sd_split_seed"]
        )

        y = (
            ypos[method]
            + offset[protocol]
        )

        ax.errorbar(

            mean,
            y,

            xerr=sd,

            fmt=PMARKER[protocol],

            color=PCOLOR[protocol],

            ecolor=PCOLOR[protocol],

            markersize=5.2,

            capsize=2,

            linewidth=0.9
        )


ax.set_xlim(
    0.50,
    1.015
)

ax.set_yticks(
    [
        ypos[m]
        for m in methods
    ]
)

ax.set_yticklabels(
    methods
)

ax.set_xlabel(
    "MCC"
)

ax.set_title(
    "Compact-representation baselines",
    loc="left",
    fontweight="bold",
    pad=8
)

clean(
    ax,
    "x"
)


# ============================================================
# FIGURE 2C
# Native method coverage-performance
# ============================================================

ax = fig.add_subplot(
    gs[1, 1]
)

panel(ax, "C")


native = [
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
]


METHOD_MARKERS = {

    "Mash": "o",

    "Dashing2": "s",

    "skani": "D",

    "FastANI": "P",
}


for method in native:

    q = internal[
        internal["Method"] == method
    ]

    xs = []
    ys = []

    for protocol in PROTOCOLS:

        r = q[
            q["protocol"] == protocol
        ]

        if len(r) != 1:
            continue

        row = r.iloc[0]

        x = float(
            row["coverage_mean"]
        )

        y = float(
            row["MCC_mean"]
        )

        xs.append(x)
        ys.append(y)

        ax.scatter(

            x,
            y,

            marker=METHOD_MARKERS[method],

            s=42,

            color=PCOLOR[protocol],

            edgecolor="white",

            linewidth=0.55,

            zorder=4
        )

    ax.plot(

        xs,
        ys,

        color="#B6BDC2",

        linewidth=0.8,

        zorder=1
    )


# fixed label locations instead of point annotation
legend_method = [

    Line2D(
        [0],
        [0],
        marker=METHOD_MARKERS[m],
        linestyle="",
        color=DARK,
        markerfacecolor=DARK,
        markersize=5,
        label=m
    )

    for m in native
]


ax.legend(

    handles=legend_method,

    frameon=False,

    loc="lower left",

    fontsize=6.5
)


ax.set_xlim(
    0.50,
    1.025
)

ax.set_ylim(
    0.985,
    1.0015
)

ax.set_xlabel(
    "Pair coverage"
)

ax.set_ylabel(
    "Conditional MCC"
)

ax.set_title(
    "Native methods: coverage versus MCC",
    loc="left",
    fontweight="bold",
    pad=8
)

clean(
    ax,
    "both"
)


fig.text(

    0.02,
    0.015,

    "Error bars: SD across taxonomy split seeds. "
    "Complete 11-method metrics are reported in Table 2.",

    fontsize=6.3,

    color=GRAY
)


fig.subplots_adjust(
    left=0.18,
    right=0.98,
    top=0.95,
    bottom=0.09
)


save(
    fig,
    "Figure2_internal_generalization_v8_fix1"
)


# ============================================================
# FIGURE 3
# ============================================================

def pick_col(df, candidates):

    for c in candidates:

        if c in df.columns:

            return c

    raise KeyError(
        candidates
    )


TYPE = pick_col(
    e11,
    [
        "external_type",
        "cohort"
    ]
)


R1 = pick_col(
    e11,
    [
        "Recall@1",
        "R1",
        "R1_mean"
    ]
)


MRR = pick_col(
    e11,
    [
        "MRR",
        "MRR_mean"
    ]
)


KCOL = {}

for k in [
    1,
    5,
    10,
    20,
    50
]:

    KCOL[k] = pick_col(
        e11,
        [
            f"Recall@{k}",
            f"R{k}",
            f"R{k}_mean"
        ]
    )


cohorts = [

    "B1",

    "B2_seen_genus",

    "B2_unseen_genus",
]


labels = {

    "B1":
        "B1 matched species",

    "B2_seen_genus":
        "B2 unseen species / seen genus",

    "B2_unseen_genus":
        "B2 unseen genus",
}


colors = {

    "B1": NAVY,

    "B2_seen_genus": TEAL,

    "B2_unseen_genus": RED,
}


fig = plt.figure(
    figsize=(7.2, 6.1)
)


gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[
        0.95,
        1.05
    ],
    hspace=0.45
)


# ============================================================
# FIGURE 3A
# ============================================================

ax = fig.add_subplot(
    gs[0]
)

panel(
    ax,
    "A"
)


yy = np.arange(3)[::-1]


for i, cohort in enumerate(cohorts):

    q = e11[
        e11[TYPE] == cohort
    ]

    if len(q) != 1:

        raise RuntimeError(
            cohort
        )

    row = q.iloc[0]

    r1 = float(
        row[R1]
    )

    mrr = float(
        row[MRR]
    )

    y = yy[i]


    # connector
    ax.plot(

        [r1, mrr],

        [y, y],

        color="#B8BFC4",

        linewidth=1.2
    )


    # Recall@1
    ax.scatter(

        r1,
        y,

        s=52,

        color=colors[cohort],

        marker="o",

        zorder=3
    )


    # MRR
    ax.scatter(

        mrr,
        y,

        s=52,

        facecolor="white",

        edgecolor=colors[cohort],

        marker="s",

        linewidth=1.5,

        zorder=3
    )


    # values placed in fixed columns, not beside points
    ax.text(

        1.035,
        y + 0.10,

        f"R@1  {r1:.3f}",

        fontsize=6.8,

        color=colors[cohort],

        va="center"
    )


    ax.text(

        1.035,
        y - 0.10,

        f"MRR  {mrr:.3f}",

        fontsize=6.8,

        color=colors[cohort],

        va="center"
    )


ax.set_xlim(
    0.48,
    1.18
)

ax.set_ylim(
    -0.45,
    2.45
)

ax.set_yticks(
    yy
)

ax.set_yticklabels(
    [
        labels[c]
        for c in cohorts
    ]
)

ax.set_xlabel(
    "Retrieval performance"
)

ax.set_title(
    "HGV external operating boundary",
    loc="left",
    fontweight="bold",
    pad=8
)

clean(
    ax,
    "x"
)


metric_legend = [

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=DARK,
        markersize=6,
        label="Recall@1"
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor=DARK,
        markersize=6,
        label="MRR"
    ),
]


ax.legend(
    handles=metric_legend,
    frameon=False,
    loc="lower left"
)


# ============================================================
# FIGURE 3B
# ============================================================

ax = fig.add_subplot(
    gs[1]
)

panel(
    ax,
    "B"
)


K = [
    1,
    5,
    10,
    20,
    50
]


for cohort in cohorts:

    row = e11[
        e11[TYPE] == cohort
    ].iloc[0]

    vals = [
        float(
            row[KCOL[k]]
        )
        for k in K
    ]

    ax.plot(

        K,

        vals,

        color=colors[cohort],

        linewidth=1.7,

        marker="o",

        markersize=5.0,

        label=labels[cohort]
    )


    # left endpoint number only
    ax.text(

        0.92,

        vals[0],

        f"{vals[0]:.3f}",

        ha="right",

        va="center",

        fontsize=6.6,

        color=colors[cohort]
    )


    # right endpoint direct label; number included
    ax.text(

        55,

        vals[-1],

        f"{labels[cohort]}  ({vals[-1]:.3f})",

        va="center",

        fontsize=6.6,

        color=colors[cohort]
    )


ax.set_xscale(
    "log"
)

ax.set_xticks(
    K
)

ax.set_xticklabels(
    [
        str(k)
        for k in K
    ]
)

ax.set_xlim(
    0.75,
    105
)

ax.set_ylim(
    0.50,
    1.025
)

ax.set_xlabel(
    "Top-K references retained"
)

ax.set_ylabel(
    "Recall@K"
)

ax.set_title(
    "Recall as the candidate budget increases",
    loc="left",
    fontweight="bold",
    pad=8
)

clean(
    ax,
    "y"
)


fig.text(

    0.02,
    0.015,

    "Complete 11-method external results are reported in Table 3; "
    "this panel isolates the final E11 HGV screening analysis.",

    fontsize=6.3,

    color=GRAY
)


fig.subplots_adjust(
    left=0.28,
    right=0.84,
    top=0.95,
    bottom=0.09
)


save(
    fig,
    "Figure3_external_screening_v8_fix1"
)


print("=" * 72)

print(
    "V8 FIX1 COMPLETE"
)

print("=" * 72)

print(
    OUT /
    "Figure2_internal_generalization_v8_fix1.png"
)

print(
    OUT /
    "Figure3_external_screening_v8_fix1.png"
)

