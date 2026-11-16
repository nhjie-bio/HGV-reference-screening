from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# PATH
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

SRC = (
    BASE /
    "18_figures" /
    "HGV_V3_FINAL_MANUSCRIPT_PACKAGE_v6_20260926" /
    "source_data"
)

OUT = (
    BASE /
    "18_figures" /
    "HGV_V3_PUBLICATION_FIGURES_v8_PROTOTYPE_20260926"
)

OUT.mkdir(parents=True, exist_ok=True)

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
# METHOD NORMALIZATION
# ============================================================

def method_name(x):
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

internal["Method"] = internal["method"].map(method_name)

# ============================================================
# COLORS
# restrained journal palette
# ============================================================

NAVY = "#153B5B"
TEAL = "#3B8C88"
OCHRE = "#C9952E"

DARK = "#252B31"
GRAY = "#747C83"
LIGHT = "#DCE2E6"
PALE_BLUE = "#EDF4F8"

PROTOCOLS = ["G2", "G3", "G4"]

PCOLOR = {
    "G2": NAVY,
    "G3": TEAL,
    "G4": OCHRE,
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
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def panel(ax, label):
    ax.text(
        -0.10, 1.06,
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
            alpha=0.85,
            zorder=0
        )

def save(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT / f"{stem}.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight"
        )

    plt.close(fig)

# ============================================================
# FIGURE 2
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.25, 4.15),
    gridspec_kw={
        "width_ratios": [0.90, 1.10, 1.10]
    }
)

# ------------------------------------------------------------
# A. HGV hierarchical generalization
# ------------------------------------------------------------

ax = axes[0]
panel(ax, "A")

hgv = internal[
    internal["Method"] == "HGV"
].copy()

metric_rows = [
    (
        "AUROC",
        "AUROC_mean",
        "AUROC_sd_split_seed"
    ),
    (
        "AUPRC",
        "AUPRC_mean",
        "AUPRC_sd_split_seed"
    ),
    (
        "MCC",
        "MCC_mean",
        "MCC_sd_split_seed"
    ),
]

y_base = {
    "AUROC": 2,
    "AUPRC": 1,
    "MCC": 0,
}

protocol_offset = {
    "G2": +0.16,
    "G3": 0.00,
    "G4": -0.16,
}

for metric, mean_col, sd_col in metric_rows:
    y0 = y_base[metric]

    for protocol in PROTOCOLS:
        q = hgv[
            hgv["protocol"] == protocol
        ]

        if len(q) != 1:
            raise RuntimeError(
                f"HGV {protocol} not unique"
            )

        r = q.iloc[0]

        mean = float(r[mean_col])
        sd = float(r[sd_col])

        y = y0 + protocol_offset[protocol]

        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt=PMARKER[protocol],
            markersize=5,
            color=PCOLOR[protocol],
            ecolor=PCOLOR[protocol],
            capsize=2,
            linewidth=0.9,
            zorder=3
        )

        ax.text(
            mean + 0.010,
            y,
            f"{mean:.3f}",
            va="center",
            fontsize=6.3,
            color=PCOLOR[protocol]
        )

ax.set_xlim(0.78, 1.025)

ax.set_yticks(
    [2, 1, 0]
)

ax.set_yticklabels(
    ["AUROC", "AUPRC", "MCC"]
)

ax.set_xlabel(
    "Performance"
)

ax.set_title(
    "HGV across taxonomic holdouts",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "x")

# ------------------------------------------------------------
# B. Representation baseline context
# use MCC because AUROC/AUPRC are strongly ceiling-compressed
# ------------------------------------------------------------

ax = axes[1]
panel(ax, "B")

rep_methods = [
    "HGV",
    "Raw-k6 Euclidean",
    "PCA64 Euclidean",
    "DNABERT-S",
]

y_method = {
    m: len(rep_methods)-1-i
    for i, m in enumerate(rep_methods)
}

# HGV highlight
ax.axhspan(
    y_method["HGV"] - 0.40,
    y_method["HGV"] + 0.40,
    color=PALE_BLUE,
    zorder=-5
)

for method in rep_methods:
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

        mean = float(row["MCC_mean"])
        sd = float(
            row["MCC_sd_split_seed"]
        )

        y = (
            y_method[method] +
            protocol_offset[protocol]
        )

        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt=PMARKER[protocol],
            color=PCOLOR[protocol],
            ecolor=PCOLOR[protocol],
            markersize=5,
            capsize=2,
            linewidth=0.9,
            zorder=3
        )

ax.set_xlim(0.50, 1.02)

ax.set_yticks(
    [y_method[m] for m in rep_methods]
)

ax.set_yticklabels(
    rep_methods
)

ax.set_xlabel(
    "MCC"
)

ax.set_title(
    "Representation baseline context",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "x")

# ------------------------------------------------------------
# C. Native methods:
# conditional MCC versus pair coverage
# ------------------------------------------------------------

ax = axes[2]
panel(ax, "C")

native_methods = [
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
]

label_offset = {
    "Mash": (-14, +10),
    "Dashing2": (-18, -13),
    "skani": (+6, -12),
    "FastANI": (+6, +9),
}

for method in native_methods:
    q = internal[
        internal["Method"] == method
    ].copy()

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
            marker=PMARKER[protocol],
            s=34,
            color=PCOLOR[protocol],
            edgecolor="white",
            linewidth=0.45,
            zorder=4
        )

    # subtle line joins G2 -> G3 -> G4
    ax.plot(
        xs,
        ys,
        color="#AEB5BA",
        linewidth=0.8,
        zorder=2
    )

    if len(xs) >= 2:
        xm = xs[1]
        ym = ys[1]

        dx, dy = label_offset[method]

        ax.annotate(
            method,
            (xm, ym),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.2,
            color=DARK
        )

ax.set_xlim(0.50, 1.035)
ax.set_ylim(0.965, 1.005)

ax.set_xlabel(
    "Pair coverage"
)

ax.set_ylabel(
    "Conditional MCC"
)

ax.set_title(
    "Native-method coverage context",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "both")

# protocol legend
handles = [
    Line2D(
        [0], [0],
        marker=PMARKER[p],
        linestyle="",
        markerfacecolor=PCOLOR[p],
        markeredgecolor="none",
        markersize=5.5,
        label=p
    )
    for p in PROTOCOLS
]

fig.legend(
    handles=handles,
    labels=[
        "G2 species holdout",
        "G3 genus holdout",
        "G4 family holdout"
    ],
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.50, -0.005)
)

fig.text(
    0.01,
    -0.035,
    "Error bars: SD across taxonomy split seeds. "
    "Full 11-method AUROC/AUPRC/MCC/coverage values are reported in Table 2.",
    fontsize=6.2,
    color=GRAY
)

fig.subplots_adjust(
    left=0.12,
    right=0.99,
    top=0.91,
    bottom=0.16,
    wspace=0.48
)

save(
    fig,
    "Figure2_internal_generalization_v8"
)

# ============================================================
# FIGURE 3
# HGV EXTERNAL OPERATING BOUNDARY
# ============================================================

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"None of columns found: {candidates}\n"
        f"Available: {list(df.columns)}"
    )

TYPE_COL = pick_col(
    e11,
    ["external_type", "cohort"]
)

R1_COL = pick_col(
    e11,
    ["Recall@1", "R1", "R1_mean"]
)

MRR_COL = pick_col(
    e11,
    ["MRR", "MRR_mean", "mean_MRR"]
)

KCOLS = {}

for k in [1, 5, 10, 20, 50]:
    KCOLS[k] = pick_col(
        e11,
        [
            f"Recall@{k}",
            f"R{k}",
            f"R{k}_mean"
        ]
    )

cohort_order = [
    "B1",
    "B2_seen_genus",
    "B2_unseen_genus",
]

cohort_labels = {
    "B1":
        "B1 matched species",

    "B2_seen_genus":
        "B2 unseen species /\nseen genus",

    "B2_unseen_genus":
        "B2 unseen genus",
}

cohort_colors = {
    "B1": NAVY,
    "B2_seen_genus": TEAL,
    "B2_unseen_genus": "#A65757",
}

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.55),
    gridspec_kw={
        "width_ratios": [0.92, 1.08]
    }
)

# ------------------------------------------------------------
# A. R@1 and MRR
# ------------------------------------------------------------

ax = axes[0]
panel(ax, "A")

yy = np.arange(3)[::-1]

for i, cohort in enumerate(
    cohort_order
):
    q = e11[
        e11[TYPE_COL] == cohort
    ]

    if len(q) != 1:
        raise RuntimeError(
            f"Expected one {cohort} row"
        )

    r = q.iloc[0]

    r1 = float(r[R1_COL])
    mrr = float(r[MRR_COL])

    y = yy[i]

    # connector
    ax.plot(
        [r1, mrr],
        [y, y],
        color="#B6BDC2",
        linewidth=1.2,
        zorder=1
    )

    ax.scatter(
        r1,
        y,
        s=42,
        color=cohort_colors[cohort],
        marker="o",
        zorder=3,
        label="Recall@1" if i == 0 else None
    )

    ax.scatter(
        mrr,
        y,
        s=42,
        facecolor="white",
        edgecolor=cohort_colors[cohort],
        linewidth=1.3,
        marker="s",
        zorder=3,
        label="MRR" if i == 0 else None
    )

    ax.text(
        r1 - 0.015,
        y + 0.13,
        f"R@1 {r1:.3f}",
        ha="right",
        fontsize=6.3,
        color=cohort_colors[cohort]
    )

    ax.text(
        mrr + 0.015,
        y - 0.13,
        f"MRR {mrr:.3f}",
        ha="left",
        fontsize=6.3,
        color=cohort_colors[cohort]
    )

ax.set_xlim(0.48, 1.025)

ax.set_yticks(yy)

ax.set_yticklabels([
    cohort_labels[c]
    for c in cohort_order
])

ax.set_xlabel(
    "Retrieval performance"
)

ax.set_title(
    "External operating boundary",
    loc="left",
    fontweight="bold",
    color=DARK
)

ax.legend(
    frameon=False,
    loc="lower left"
)

clean(ax, "x")

# ------------------------------------------------------------
# B. Recall@K
# ------------------------------------------------------------

ax = axes[1]
panel(ax, "B")

K = [1, 5, 10, 20, 50]

for cohort in cohort_order:
    q = e11[
        e11[TYPE_COL] == cohort
    ].iloc[0]

    vals = [
        float(q[KCOLS[k]])
        for k in K
    ]

    ax.plot(
        K,
        vals,
        color=cohort_colors[cohort],
        linewidth=1.6,
        marker="o",
        markersize=4.5
    )

    ax.text(
        55,
        vals[-1],
        cohort_labels[cohort].replace(
            "\n", " "
        ),
        fontsize=6.2,
        va="center",
        color=cohort_colors[cohort]
    )

    # exact labels at K=1 and K=50
    ax.text(
        K[0],
        vals[0] + 0.012,
        f"{vals[0]:.3f}",
        ha="center",
        fontsize=5.8,
        color=cohort_colors[cohort]
    )

    ax.text(
        K[-1],
        vals[-1] + 0.012,
        f"{vals[-1]:.3f}",
        ha="center",
        fontsize=5.8,
        color=cohort_colors[cohort]
    )

ax.set_xscale("log")

ax.set_xticks(K)

ax.set_xticklabels(
    [str(k) for k in K]
)

ax.set_xlim(0.8, 95)
ax.set_ylim(0.50, 1.025)

ax.set_xlabel(
    "Top-K references retained"
)

ax.set_ylabel(
    "Recall@K"
)

ax.set_title(
    "Candidate-budget response",
    loc="left",
    fontweight="bold",
    color=DARK
)

clean(ax, "y")

fig.text(
    0.01,
    -0.025,
    "This figure uses the final E11 HGV screening block. "
    "The complete 11-method external benchmark is reported separately in Table 3.",
    fontsize=6.2,
    color=GRAY
)

fig.subplots_adjust(
    left=0.20,
    right=0.88,
    top=0.90,
    bottom=0.18,
    wspace=0.42
)

save(
    fig,
    "Figure3_external_screening_v8"
)

print("=" * 72)
print("V8 PROTOTYPE COMPLETE")
print("=" * 72)
print(OUT / "Figure2_internal_generalization_v8.png")
print(OUT / "Figure3_external_screening_v8.png")
