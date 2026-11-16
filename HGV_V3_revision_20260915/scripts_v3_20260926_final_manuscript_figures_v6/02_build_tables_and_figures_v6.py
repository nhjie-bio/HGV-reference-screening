from pathlib import Path
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import Normalize

# ============================================================
# PATHS
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

ROOT = BASE / "18_figures" / \
    "HGV_V3_FINAL_MANUSCRIPT_PACKAGE_v6_20260926"

SRC = ROOT / "source_data"
MAIN = ROOT / "main_figures"
SUPP = ROOT / "supplementary_figures"
TABLE = ROOT / "tables"
AUDIT = ROOT / "audit"
CAP = ROOT / "captions"

for x in [MAIN, SUPP, TABLE, AUDIT, CAP]:
    x.mkdir(parents=True, exist_ok=True)

# ============================================================
# READ FROZEN SOURCES
# ============================================================

internal = pd.read_csv(
    SRC / "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv",
    sep="\t"
)

internal_split = pd.read_csv(
    SRC / "E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv",
    sep="\t"
)

external = pd.read_csv(
    SRC / "E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv",
    sep="\t"
)

e6m = pd.read_csv(SRC / "e6_modes.tsv", sep="\t")
e6c = pd.read_csv(SRC / "e6_contrasts.tsv", sep="\t")
e7 = pd.read_csv(SRC / "e7.tsv", sep="\t")
e8 = pd.read_csv(SRC / "e8.tsv", sep="\t")
e9 = pd.read_csv(SRC / "e9_ig.tsv", sep="\t")
e10corr = pd.read_csv(SRC / "e10_corr.tsv", sep="\t")
e10tax = pd.read_csv(SRC / "e10_tax.tsv", sep="\t")
e10conf = pd.read_csv(SRC / "e10_confound.tsv", sep="\t")
e10tree = pd.read_csv(SRC / "e10_tree_pairs.tsv", sep="\t")
e10treesum = pd.read_csv(SRC / "e10_tree_summary.tsv", sep="\t")
e11 = pd.read_csv(SRC / "e11.tsv", sep="\t")
e12 = pd.read_csv(SRC / "e12.tsv", sep="\t")

# ============================================================
# STRICT SCIENTIFIC GATES
# ============================================================

assert len(internal) == 33, len(internal)
assert internal["method"].nunique() == 11
assert set(internal["protocol"]) == {"G2", "G3", "G4"}

assert len(internal_split) == 99, len(internal_split)
assert internal_split["method"].nunique() == 11

assert len(external) == 33, len(external)
assert external["method"].nunique() == 11
assert set(external["cohort"]) == {
    "B1",
    "B2_seen_genus",
    "B2_unseen_genus"
}

assert len(e6m) == 6
assert len(e6c) == 10
assert len(e7) == 9
assert len(e8) == 6
assert len(e10tax) == 4
assert len(e11) == 3
assert len(e12) == 5

# ============================================================
# METHOD LABELS
# ============================================================

def pretty_method(x):
    s = str(x)
    sl = s.lower()

    if "hgv" in sl:
        return "HGV"

    if "dnabert" in sl:
        return "DNABERT-S"

    if "dashing" in sl:
        return "Dashing2"

    if "fastani" in sl:
        return "FastANI"

    if "skani" in sl:
        return "skani"

    if sl == "mash" or "mash" in sl:
        return "Mash"

    if "rawk6" in sl and "bray" in sl:
        return "Raw-k6 Bray-Curtis"

    if "rawk6" in sl and "cosine" in sl:
        return "Raw-k6 Cosine"

    if "rawk6" in sl and "euclid" in sl:
        return "Raw-k6 Euclidean"

    if "pca64" in sl and "euclid" in sl:
        return "PCA64 Euclidean"

    if "pca64" in sl and "l2" in sl:
        return "PCA64 L2"

    return s


METHOD_ORDER = [
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
internal_split["Method"] = internal_split["method"].map(pretty_method)
external["Method"] = external["method"].map(pretty_method)

found = set(internal["Method"])
missing_methods = [m for m in METHOD_ORDER if m not in found]

if missing_methods:
    print("[WARN] Expected labels not matched:")
    print(missing_methods)
    METHOD_ORDER = [
        m for m in METHOD_ORDER if m in found
    ] + [
        m for m in internal["Method"].unique()
        if m not in METHOD_ORDER
    ]

# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ACCENT = "#204B73"
SECOND = "#B45F06"
MUTED = "#6E7781"
LIGHT = "#F3F5F7"
GRID = "#D9DEE3"

def panel(ax, letter):
    ax.text(
        -0.10, 1.06, letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top"
    )

def clean(ax, grid="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if grid == "x":
        ax.grid(axis="x", color=GRID, lw=0.5, alpha=0.7)
    elif grid == "y":
        ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)

def save(fig, stem, folder):
    fig.savefig(
        folder / f"{stem}.png",
        dpi=600,
        bbox_inches="tight"
    )
    fig.savefig(
        folder / f"{stem}.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        folder / f"{stem}.svg",
        bbox_inches="tight"
    )
    plt.close(fig)

# ============================================================
# TABLE HELPERS
# ============================================================

def fmt_pm(mean, sd, digits=3):
    if pd.isna(mean):
        return "NA"

    if pd.isna(sd):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {sd:.{digits}f}"

def save_table(df, name):
    df.to_csv(
        TABLE / f"{name}.tsv",
        sep="\t",
        index=False
    )

    df.to_csv(
        TABLE / f"{name}.csv",
        index=False
    )

# ============================================================
# TABLE 1
# ============================================================

table1 = pd.DataFrame([
    ["Development A", 2374,
     "Development/reference index",
     "Internal G2/G3/G4 evaluation"],

    ["External B1", 391,
     "External matched-species cohort",
     "Cross-source matched-species retrieval"],

    ["External B2 seen-genus", 196,
     "External unseen-species cohort",
     "Species unseen; genus represented in A"],

    ["External B2 unseen-genus", 193,
     "External distant cohort",
     "Genus unseen in A"],

    ["Total", 3154,
     "Frozen sequence-QC-passed cohort",
     "Final HGV-V3 benchmark"],
], columns=[
    "Cohort",
    "Genomes",
    "Role",
    "Evaluation scope"
])

save_table(table1, "Table1_dataset_design")

# ============================================================
# TABLE 2 - INTERNAL 11 METHOD
# ============================================================

rows = []

for method in METHOD_ORDER:
    row = {"Method": method}

    for protocol in ["G2", "G3", "G4"]:
        q = internal[
            (internal["Method"] == method) &
            (internal["protocol"] == protocol)
        ]

        if q.empty:
            continue

        r = q.iloc[0]

        row[f"{protocol} coverage"] = \
            f"{r['coverage_mean']:.3f}"

        row[f"{protocol} AUROC"] = fmt_pm(
            r["AUROC_mean"],
            r["AUROC_sd_split_seed"]
        )

        row[f"{protocol} AUPRC"] = fmt_pm(
            r["AUPRC_mean"],
            r["AUPRC_sd_split_seed"]
        )

        row[f"{protocol} MCC"] = fmt_pm(
            r["MCC_mean"],
            r["MCC_sd_split_seed"]
        )

    rows.append(row)

table2 = pd.DataFrame(rows)
save_table(table2, "Table2_internal_11method_benchmark")

save_table(
    internal,
    "Table2_source_internal_11method_long"
)

# ============================================================
# TABLE 3 - EXTERNAL 11 METHOD
# ============================================================

rows = []

cohort_map = {
    "B1": "B1 species",
    "B2_seen_genus": "B2 seen-genus",
    "B2_unseen_genus": "B2 unseen-genus",
}

for method in METHOD_ORDER:
    row = {"Method": method}

    for cohort, lab in cohort_map.items():
        q = external[
            (external["Method"] == method) &
            (external["cohort"] == cohort)
        ]

        if q.empty:
            continue

        r = q.iloc[0]

        row[f"{lab} endpoint"] = r["endpoint"]

        row[f"{lab} coverage"] = \
            f"{r['pair_coverage_mean']:.3f}"

        row[f"{lab} R@1"] = fmt_pm(
            r["R1_mean"],
            r["R1_sd"]
        )

        row[f"{lab} R@5"] = fmt_pm(
            r["R5_mean"],
            r["R5_sd"]
        )

        row[f"{lab} R@10"] = fmt_pm(
            r["R10_mean"],
            r["R10_sd"]
        )

        row[f"{lab} MRR"] = fmt_pm(
            r["MRR_mean"],
            r["MRR_sd"]
        )

    rows.append(row)

table3 = pd.DataFrame(rows)
save_table(table3, "Table3_external_11method_benchmark")

save_table(
    external,
    "Table3_source_external_11method_long"
)

# ============================================================
# TABLE 4 - E6
# ============================================================

MODE_DESC = {
    "D0": "Contrastive only",
    "D1": "All-pair distillation",
    "D2": "Negative-only distillation",
    "D3": "Selective informative-negative distillation",
    "D4": "Shuffled-teacher control",
    "D5": "Selective distillation without output L2",
}

t4 = e6m.copy()
t4.insert(
    1,
    "description",
    t4["mode"].map(MODE_DESC)
)

save_table(t4, "Table4_E6_ablation")
save_table(e6c, "Table4B_E6_formal_contrasts")

# ============================================================
# TABLE 5 - SCREENING / SCALABILITY
# ============================================================

save_table(e11, "Table5A_reference_screening")
save_table(e12, "Table5B_scalability")

# ============================================================
# SUPPLEMENTARY TABLES
# ============================================================

save_table(e7, "TableS1_sensitivity")
save_table(e8, "TableS2_fragmentation")

save_table(
    e9.nsmallest(30, "rank"),
    "TableS3_top30_integrated_gradients"
)

save_table(e10tax, "TableS4A_taxonomic_fidelity")
save_table(e10corr, "TableS4B_ANI_AF_correlations")
save_table(e10conf, "TableS4C_confound_adjustment")
save_table(e10treesum, "TableS5_GTDB_tree_consistency")

# ============================================================
# EXCEL SOURCE-DATA WORKBOOK
# ============================================================

try:
    from openpyxl import Workbook
    from openpyxl.styles import (
        Font, PatternFill, Border,
        Side, Alignment
    )
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)

    sheets = {
        "T1_Dataset": table1,
        "T2_Internal": table2,
        "T2_Internal_Raw": internal,
        "T3_External": table3,
        "T3_External_Raw": external,
        "T4_Ablation": t4,
        "T4_Contrasts": e6c,
        "T5_Screening": e11,
        "T5_Scalability": e12,
        "S1_Sensitivity": e7,
        "S2_Fragmentation": e8,
        "S3_IG_Top30": e9.nsmallest(30, "rank"),
        "S4_Taxonomy": e10tax,
        "S4_Correlation": e10corr,
        "S4_Confound": e10conf,
        "S5_Tree": e10treesum,
    }

    header_fill = PatternFill(
        "solid",
        fgColor="1F4E78"
    )

    imported_font = Font(
        color="008000"
    )

    header_font = Font(
        color="FFFFFF",
        bold=True
    )

    thin = Side(
        style="thin",
        color="B7C3D0"
    )

    for name, df in sheets.items():
        ws = wb.create_sheet(name)
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"

        for j, col in enumerate(df.columns, 1):
            c = ws.cell(1, j, col)
            c.fill = header_fill
            c.font = header_font
            c.alignment = Alignment(
                horizontal="center",
                vertical="center"
            )

        for i, row in enumerate(
            df.itertuples(index=False),
            2
        ):
            for j, val in enumerate(row, 1):
                cell = ws.cell(i, j)

                if pd.isna(val):
                    cell.value = None
                else:
                    cell.value = val

                cell.font = imported_font
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=False
                )

        for j, col in enumerate(df.columns, 1):
            max_len = len(str(col))

            for i in range(2, min(len(df)+2, 200)):
                v = ws.cell(i, j).value
                if v is not None:
                    max_len = max(
                        max_len,
                        len(str(v))
                    )

            ws.column_dimensions[
                get_column_letter(j)
            ].width = min(max(max_len + 2, 10), 34)

        ws.auto_filter.ref = ws.dimensions

    # README
    ws = wb.create_sheet("README", 0)
    ws["A1"] = "HGV-V3 Final Source Data v6"
    ws["A1"].font = Font(
        bold=True,
        size=14
    )

    ws["A3"] = "Internal benchmark"
    ws["B3"] = \
        "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv"

    ws["A4"] = "External benchmark"
    ws["B4"] = \
        "E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv"

    ws["A5"] = "Replicate unit"
    ws["B5"] = \
        "taxonomy split seed for internal protocol summaries"

    ws["A6"] = "Coverage caveat"
    ws["B6"] = \
        "FastANI/skani conditional metrics must be interpreted with reported coverage"

    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 80

    wb.save(
        TABLE / "HGV_V3_Source_Data_v6.xlsx"
    )

except Exception as e:
    print(
        "[WARN] Excel workbook skipped:",
        repr(e)
    )

# ============================================================
# HEATMAP HELPER
# ============================================================

def matrix_from(
    df,
    value,
    columns,
    method_col="Method"
):
    p = df.pivot(
        index=method_col,
        columns=columns,
        values=value
    )

    p = p.reindex(METHOD_ORDER)

    return p


def draw_heatmap(
    ax,
    mat,
    title,
    vmin,
    vmax,
    cmap="cividis",
    fmt=".3f",
    show_y=True,
):
    arr = mat.to_numpy(dtype=float)

    im = ax.imshow(
        arr,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )

    ax.set_xticks(
        np.arange(len(mat.columns))
    )
    ax.set_xticklabels(
        mat.columns,
        fontsize=7
    )

    ax.set_yticks(
        np.arange(len(mat.index))
    )

    if show_y:
        ax.set_yticklabels(
            mat.index,
            fontsize=7
        )
    else:
        ax.set_yticklabels([])

    threshold = (
        vmin + 0.58 * (vmax - vmin)
    )

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]

            if np.isnan(v):
                txt = "NA"
                color = "#666666"
            else:
                txt = format(v, fmt)
                color = (
                    "white"
                    if v > threshold
                    else "black"
                )

            ax.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=6.2,
                color=color
            )

    if "HGV" in mat.index:
        r = list(mat.index).index("HGV")
        ax.add_patch(
            Rectangle(
                (-0.49, r - 0.49),
                len(mat.columns) - 0.02,
                0.98,
                fill=False,
                edgecolor=ACCENT,
                linewidth=1.8
            )
        )

    ax.tick_params(
        length=0
    )

    return im

# ============================================================
# FIGURE 1 - CLEAN STUDY DESIGN
# ============================================================

fig = plt.figure(
    figsize=(7.2, 4.9),
    constrained_layout=True
)

gs = fig.add_gridspec(
    2, 1,
    height_ratios=[1.08, 0.92]
)

# ---------------- A ----------------

ax = fig.add_subplot(gs[0])
panel(ax, "A")
ax.axis("off")

ax.text(
    0.01, 0.96,
    "Dataset and evaluation architecture",
    fontweight="bold",
    fontsize=9,
    va="top"
)

def card(ax, x, y, w, h, title, subtitle="", fc="white"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fc,
        edgecolor="#66717C",
        linewidth=0.8
    )
    ax.add_patch(box)

    ax.text(
        x+w/2,
        y+h*0.61,
        title,
        ha="center",
        va="center",
        fontsize=7.3,
        fontweight="bold"
    )

    if subtitle:
        ax.text(
            x+w/2,
            y+h*0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.4,
            color="#4C5661"
        )

# curation column
cards = [
    (0.02, 0.60, "GTDB R232", "878,998 metadata records"),
    (0.02, 0.39, "Strict QC", "172,933 candidates"),
    (0.02, 0.18, "Cohort selection", "3,200 genomes"),
    (0.02, -0.03, "Frozen set", "3,154 genomes"),
]

for x, y, t, s in cards:
    card(
        ax, x, y,
        0.19, 0.14,
        t, s,
        LIGHT
    )

for y1, y2 in [
    (0.60, 0.53),
    (0.39, 0.32),
    (0.18, 0.11),
]:
    ax.annotate(
        "",
        xy=(0.115, y2),
        xytext=(0.115, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8,
            color=MUTED
        )
    )

# cohorts
ax.text(
    0.31, 0.78,
    "Frozen cohorts",
    ha="center",
    fontweight="bold",
    fontsize=7.5
)

cohorts = [
    ("A", "2,374", 0.25, 0.52),
    ("B1", "391", 0.25, 0.29),
    ("B2-seen", "196", 0.25, 0.06),
    ("B2-unseen", "193", 0.25, -0.17),
]

for t, s, x, y in cohorts:
    card(
        ax, x, y,
        0.14, 0.15,
        t, f"{s} genomes",
        "white"
    )

# evaluation
ax.text(
    0.59, 0.78,
    "Taxonomic holdout",
    ha="center",
    fontweight="bold",
    fontsize=7.5
)

evals = [
    ("G2", "unseen species", 0.47),
    ("G3", "unseen genus", 0.58),
    ("G4", "unseen family", 0.69),
]

for code, text_, x in evals:
    card(
        ax, x, 0.42,
        0.095, 0.19,
        code, text_,
        "#F7F9FB"
    )

ax.annotate(
    "",
    xy=(0.79, 0.67),
    xytext=(0.46, 0.67),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.8,
        color=MUTED
    )
)

ax.text(
    0.625, 0.70,
    "increasing novelty",
    ha="center",
    fontsize=6.4,
    color=MUTED
)

# external
card(
    ax, 0.49, 0.05,
    0.29, 0.18,
    "Cross-source external evaluation",
    "B1 matched species → B2 seen genus → B2 unseen genus",
    "#F7F9FB"
)

# endpoint
card(
    ax, 0.84, 0.34,
    0.14, 0.24,
    "Frozen evaluation",
    "11 methods\ninternal + external",
    "#E9F1F7"
)

ax.annotate(
    "",
    xy=(0.84, 0.46),
    xytext=(0.78, 0.46),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=ACCENT
    )
)

ax.set_xlim(0, 1)
ax.set_ylim(-0.22, 1.0)

# ---------------- B ----------------

ax = fig.add_subplot(gs[1])
panel(ax, "B")
ax.axis("off")

ax.text(
    0.01, 0.95,
    "Representation, regularization and deployment",
    fontweight="bold",
    fontsize=9,
    va="top"
)

pipeline = [
    ("Whole genome", 0.02, 0.52, 0.14),
    ("Canonical k=6", 0.20, 0.52, 0.15),
    ("2,080-D", 0.39, 0.52, 0.11),
    ("256 → 128", 0.54, 0.52, 0.12),
    ("64-D HGV", 0.70, 0.52, 0.13),
]

for t, x, y, w in pipeline:
    card(
        ax, x, y, w, 0.18,
        t, "",
        LIGHT
    )

for a, b in zip(
    pipeline[:-1],
    pipeline[1:]
):
    ax.annotate(
        "",
        xy=(b[1], 0.61),
        xytext=(a[1]+a[4], 0.61),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.9,
            color=MUTED
        )
    )

card(
    ax, 0.39, 0.12,
    0.18, 0.18,
    "Dashing2 teacher",
    "selective informative negatives",
    "#FFF6EA"
)

ax.annotate(
    "",
    xy=(0.63, 0.51),
    xytext=(0.48, 0.30),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=SECOND
    )
)

card(
    ax, 0.86, 0.52,
    0.11, 0.18,
    "Top-K",
    "vector retrieval",
    "#E9F1F7"
)

card(
    ax, 0.86, 0.12,
    0.11, 0.18,
    "ANI",
    "skani / FastANI",
    "#E9F1F7"
)

ax.annotate(
    "",
    xy=(0.86, 0.61),
    xytext=(0.83, 0.61),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=ACCENT
    )
)

ax.annotate(
    "",
    xy=(0.915, 0.30),
    xytext=(0.915, 0.52),
    arrowprops=dict(
        arrowstyle="->",
        lw=0.9,
        color=ACCENT
    )
)

ax.text(
    0.62, 0.20,
    "32.5× dimensional reduction",
    ha="center",
    color=ACCENT,
    fontweight="bold",
    fontsize=7
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

save(
    fig,
    "Figure1_study_design_v6",
    MAIN
)

# ============================================================
# FIGURE 2 - INTERNAL 11-METHOD HEATMAP
# ============================================================

fig, axes = plt.subplots(
    1, 4,
    figsize=(7.2, 5.0),
    constrained_layout=True
)

protocol_labels = {
    "G2": "G2\nspecies",
    "G3": "G3\ngenus",
    "G4": "G4\nfamily",
}

tmp = internal.copy()
tmp["Protocol"] = tmp["protocol"].map(
    protocol_labels
)

for ax, value, title, vmin, vmax, letter in [
    (
        axes[0],
        "AUROC_mean",
        "AUROC",
        0.85, 1.0,
        "A"
    ),
    (
        axes[1],
        "AUPRC_mean",
        "AUPRC",
        0.85, 1.0,
        "B"
    ),
    (
        axes[2],
        "MCC_mean",
        "MCC",
        0.50, 1.0,
        "C"
    ),
    (
        axes[3],
        "coverage_mean",
        "Coverage",
        0.0, 1.0,
        "D"
    ),
]:
    panel(ax, letter)

    mat = matrix_from(
        tmp,
        value,
        "Protocol"
    )

    mat = mat[
        [
            "G2\nspecies",
            "G3\ngenus",
            "G4\nfamily"
        ]
    ]

    im = draw_heatmap(
        ax,
        mat,
        title,
        vmin,
        vmax,
        show_y=(ax is axes[0])
    )

fig.colorbar(
    axes[0].images[0],
    ax=axes[:2],
    location="bottom",
    fraction=0.035,
    pad=0.04,
    label="AUROC / AUPRC"
)

fig.colorbar(
    axes[2].images[0],
    ax=[axes[2]],
    location="bottom",
    fraction=0.035,
    pad=0.04,
    label="MCC"
)

fig.colorbar(
    axes[3].images[0],
    ax=[axes[3]],
    location="bottom",
    fraction=0.035,
    pad=0.04,
    label="Coverage"
)

save(
    fig,
    "Figure2_internal_11method_v6",
    MAIN
)

# ============================================================
# FIGURE 3 - EXTERNAL 11-METHOD HEATMAP
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 5.0),
    constrained_layout=True
)

cohort_labels = {
    "B1": "B1\nspecies",
    "B2_seen_genus": "B2-seen\ngenus",
    "B2_unseen_genus": "B2-unseen\nfamily",
}

tmp = external.copy()
tmp["Cohort"] = tmp["cohort"].map(
    cohort_labels
)

for ax, value, title, letter in [
    (
        axes[0],
        "R1_mean",
        "Recall@1",
        "A"
    ),
    (
        axes[1],
        "MRR_mean",
        "MRR",
        "B"
    ),
    (
        axes[2],
        "pair_coverage_mean",
        "Pair coverage",
        "C"
    ),
]:
    panel(ax, letter)

    mat = matrix_from(
        tmp,
        value,
        "Cohort"
    )

    mat = mat[
        [
            "B1\nspecies",
            "B2-seen\ngenus",
            "B2-unseen\nfamily"
        ]
    ]

    draw_heatmap(
        ax,
        mat,
        title,
        0.0,
        1.0,
        show_y=(ax is axes[0])
    )

fig.colorbar(
    axes[0].images[0],
    ax=axes,
    location="bottom",
    fraction=0.025,
    pad=0.035,
    label="Proportion"
)

save(
    fig,
    "Figure3_external_11method_v6",
    MAIN
)

# ============================================================
# FIGURE 4 - E6
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 3.8),
    constrained_layout=True
)

# A: D0-D5 interval plot
ax = axes[0]
panel(ax, "A")

modes = ["D0","D1","D2","D3","D4","D5"]

q = e6m.set_index("mode").loc[modes]
y = np.arange(len(q))[::-1]

ax.errorbar(
    q["AUROC_mean"],
    y + 0.10,
    xerr=q["AUROC_sd"],
    fmt="o",
    color=ACCENT,
    capsize=2,
    label="AUROC"
)

ax.errorbar(
    q["AUPRC_mean"],
    y - 0.10,
    xerr=q["AUPRC_sd"],
    fmt="s",
    color=SECOND,
    capsize=2,
    label="AUPRC"
)

ax.set_yticks(y)
ax.set_yticklabels(
    [
        f"{m}  {MODE_DESC[m]}"
        for m in modes
    ]
)

ax.set_xlabel("Performance")
ax.set_title(
    "Distillation ablations",
    loc="left",
    fontweight="bold"
)
ax.legend(frameon=False)
clean(ax, "x")

# B/C: formal contrasts
for ax, metric, letter in [
    (axes[1], "AUROC", "B"),
    (axes[2], "AUPRC", "C"),
]:
    panel(ax, letter)

    z = e6c[
        e6c["metric"] == metric
    ].copy()

    z = z.iloc[::-1].reset_index(
        drop=True
    )

    y = np.arange(len(z))

    center = z[
        "observed_delta_mean_5seed"
    ].to_numpy()

    left = center - z["ci95_low"].to_numpy()
    right = z["ci95_high"].to_numpy() - center

    ax.errorbar(
        center,
        y,
        xerr=np.vstack([left, right]),
        fmt="o",
        color=ACCENT,
        ecolor=MUTED,
        capsize=2
    )

    ax.axvline(
        0,
        color="#222222",
        lw=0.8,
        ls="--"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        z["comparison"]
    )

    ax.set_xlabel(
        f"D3 − comparator Δ{metric}"
    )

    ax.set_title(
        f"{metric} paired bootstrap",
        loc="left",
        fontweight="bold"
    )

    x_right = ax.get_xlim()[1]

    for i, r in z.iterrows():
        p = r["p_holm"]

        ax.text(
            x_right,
            i,
            f"Pₕ={p:.3f}",
            fontsize=5.8,
            ha="right",
            va="center"
        )

    clean(ax, "x")

save(
    fig,
    "Figure4_distillation_v6",
    MAIN
)

# ============================================================
# FIGURE 5 - BIOLOGICAL FIDELITY
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 3.8),
    constrained_layout=True
)

# A taxonomy intervals
ax = axes[0]
panel(ax, "A")

tax_order = [
    "same_species",
    "same_genus_diff_species",
    "same_family_diff_genus",
    "different_family",
]

tax_labels = [
    "Same species",
    "Same genus / diff. species",
    "Same family / diff. genus",
    "Different family",
]

z = e10tax.set_index(
    "taxonomic_stratum"
).loc[tax_order]

y = np.arange(4)[::-1]

ax.errorbar(
    z["hgv_distance_mean"],
    y,
    xerr=z["hgv_distance_sd"],
    fmt="o",
    color=ACCENT,
    ecolor=MUTED,
    capsize=2
)

ax.set_yticks(y)
ax.set_yticklabels(tax_labels)
ax.set_xlabel("HGV embedding distance")
ax.set_title(
    "Taxonomic distance gradient",
    loc="left",
    fontweight="bold"
)
clean(ax, "x")

# B correlation forest
ax = axes[1]
panel(ax, "B")

subset_defs = [
    ("all_reported", "All reported"),
    ("positive", "Same-species positives"),
    ("pair_type:hard", "Hard negatives"),
]

rows = []

for subset, label in subset_defs:
    for target in ["ANI", "AF_min"]:
        q = e10corr[
            (e10corr["subset"] == subset) &
            (e10corr["target"] == target)
        ]

        if len(q):
            rows.append([
                label,
                target,
                q.iloc[0]["spearman_r"]
            ])

cor = pd.DataFrame(
    rows,
    columns=[
        "Subset",
        "Target",
        "rho"
    ]
)

labels = [
    x[1]
    for x in subset_defs
]

yy = np.arange(len(labels))[::-1]

for j, target in enumerate(
    ["ANI", "AF_min"]
):
    vals = []

    for lab in labels:
        q = cor[
            (cor["Subset"] == lab) &
            (cor["Target"] == target)
        ]

        vals.append(
            q.iloc[0]["rho"]
            if len(q)
            else np.nan
        )

    ax.scatter(
        vals,
        yy + (0.10 if j == 0 else -0.10),
        s=26,
        marker="o" if j == 0 else "s",
        color=ACCENT if j == 0 else SECOND,
        label="ANI" if j == 0 else "AFmin"
    )

ax.axvline(
    0,
    color="#333333",
    ls="--",
    lw=0.7
)

ax.set_yticks(yy)
ax.set_yticklabels(labels)
ax.set_xlabel("Spearman ρ")
ax.set_title(
    "Genomic fidelity",
    loc="left",
    fontweight="bold"
)
ax.legend(frameon=False)
clean(ax, "x")

# C adjusted
ax = axes[2]
panel(ax, "C")

conf_order = ["all","positive","hard"]

z = e10conf.set_index(
    "stratum"
).loc[conf_order].reset_index()

labels = [
    "All reported",
    "Same-species positives",
    "Hard negatives",
]

yy = np.arange(3)[::-1]

for i, r in z.iterrows():
    yv = yy[i]

    raw = r[
        "raw_spearman_hgv_vs_ani_distance"
    ]

    adj = r[
        "partial_rank_correlation"
    ]

    ax.plot(
        [raw, adj],
        [yv, yv],
        color="#A0A6AD",
        lw=1.2
    )

    ax.scatter(
        raw,
        yv,
        color=ACCENT,
        s=30,
        label="Raw" if i == 0 else None
    )

    ax.scatter(
        adj,
        yv,
        color=SECOND,
        s=30,
        marker="s",
        label="Adjusted" if i == 0 else None
    )

    ax.text(
        max(raw, adj) + 0.025,
        yv,
        f"P={r['permutation_two_sided_p']:.3f}",
        va="center",
        fontsize=5.8
    )

ax.axvline(
    0,
    color="#333333",
    ls="--",
    lw=0.7
)

ax.set_yticks(yy)
ax.set_yticklabels(labels)
ax.set_xlabel("Rank association")
ax.set_title(
    "Covariate adjustment",
    loc="left",
    fontweight="bold"
)
ax.legend(frameon=False)
clean(ax, "x")

save(
    fig,
    "Figure5_biological_fidelity_v6",
    MAIN
)

# ============================================================
# FIGURE 6 - SCREENING / SCALABILITY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 5.4),
    constrained_layout=True
)

K = [1,5,10,20,50]

SCREEN_LABELS = {
    "B1": "B1 matched species",
    "B2_seen_genus":
        "B2 unseen species / seen genus",
    "B2_unseen_genus":
        "B2 unseen genus",
}

# A recall
ax = axes[0,0]
panel(ax, "A")

for _, r in e11.iterrows():
    vals = [
        r[f"Recall@{k}"]
        for k in K
    ]

    ax.plot(
        K,
        vals,
        marker="o",
        lw=1.2,
        label=SCREEN_LABELS[
            r["external_type"]
        ]
    )

ax.set_xscale("log")
ax.set_xticks(K)

reductions = [
    e11.iloc[0][
        f"candidate_reduction@{k}"
    ] * 100
    for k in K
]

ax.set_xticklabels([
    f"{k}\n{red:.1f}% cut"
    for k, red in zip(K, reductions)
])

ax.set_ylim(0.45,1.02)
ax.set_ylabel("Recall@K")
ax.set_xlabel("Candidate budget")
ax.set_title(
    "Reference recovery",
    loc="left",
    fontweight="bold"
)
ax.legend(
    frameon=False,
    fontsize=5.9
)
clean(ax, "y")

# B storage
ax = axes[0,1]
panel(ax, "B")

ax.plot(
    e12["N_reference"],
    e12["vector_index_bytes_median"] /
    1024.0,
    marker="o",
    color=ACCENT
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
    "256 bytes / genome",
    transform=ax.transAxes,
    ha="right",
    fontsize=6.5,
    color=ACCENT
)

clean(ax, "y")

# C runtime scaling
ax = axes[1,0]
panel(ax, "C")

ax.plot(
    e12["N_reference"],
    e12[
        "reference_feature_wall_s_median"
    ],
    marker="o",
    color=ACCENT,
    label="Reference feature extraction"
)

ax.plot(
    e12["N_reference"],
    e12[
        "end_to_end_wall_s_median"
    ],
    marker="s",
    color=SECOND,
    label="End-to-end"
)

ax.set_xlabel("Reference genomes")
ax.set_ylabel("Wall time (s)")
ax.set_title(
    "Real-FASTA scaling",
    loc="left",
    fontweight="bold"
)
ax.legend(
    frameon=False,
    fontsize=6.2
)
clean(ax, "y")

# D decomposition
ax = axes[1,1]
panel(ax, "D")

r = e12.loc[
    e12["N_reference"].idxmax()
]

names = [
    "Reference\nfeatures",
    "50 query\nfeatures",
    "Reference\ninference",
    "Query\ninference",
    "Warm vector\nscan"
]

vals = [
    r["reference_feature_wall_s_median"],
    r["query_feature_wall_s_median"],
    r["reference_model_inference_wall_s_median"],
    r["query_model_inference_wall_s_median"],
    r["exact_scan_warm_median_s_median"],
]

yy = np.arange(len(names))[::-1]

ax.barh(
    yy,
    vals,
    color=ACCENT
)

ax.set_xscale("log")
ax.set_yticks(yy)
ax.set_yticklabels(names)
ax.set_xlabel("Seconds (log scale)")
ax.set_title(
    "Runtime decomposition (N=2,374)",
    loc="left",
    fontweight="bold"
)

clean(ax, "x")

save(
    fig,
    "Figure6_screening_scalability_v6",
    MAIN
)

# ============================================================
# FIGURE S1 - SENSITIVITY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.0, 4.7),
    constrained_layout=True
)

configs = [
    (
        axes[0,0],
        [
            "k5",
            "canonical_k6_dim64_margin1_lambda0p25",
            "k7"
        ],
        ["k=5","k=6","k=7"],
        "k-mer size",
        "A"
    ),
    (
        axes[0,1],
        [
            "dim32",
            "canonical_k6_dim64_margin1_lambda0p25",
            "dim128"
        ],
        ["32","64","128"],
        "Embedding dimension",
        "B"
    ),
    (
        axes[1,0],
        [
            "margin0p5",
            "canonical_k6_dim64_margin1_lambda0p25",
            "margin1p5"
        ],
        ["0.5","1.0","1.5"],
        "Contrastive margin",
        "C"
    ),
    (
        axes[1,1],
        [
            "lambda0p1",
            "canonical_k6_dim64_margin1_lambda0p25",
            "lambda0p5"
        ],
        ["0.10","0.25","0.50"],
        "Distillation weight λ",
        "D"
    ),
]

for ax, settings, labs, title, letter in configs:
    panel(ax, letter)

    z = e7.set_index(
        "setting"
    ).loc[settings]

    x = np.arange(3)

    ax.errorbar(
        x - 0.05,
        z["test_AUROC_mean"],
        yerr=z["test_AUROC_sd"],
        fmt="o",
        color=ACCENT,
        capsize=2,
        label="AUROC"
    )

    ax.errorbar(
        x + 0.05,
        z["test_AUPRC_mean"],
        yerr=z["test_AUPRC_sd"],
        fmt="s",
        color=SECOND,
        capsize=2,
        label="AUPRC"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0.88,1.0)
    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )
    clean(ax, "y")

axes[0,0].legend(
    frameon=False
)

save(
    fig,
    "FigureS1_sensitivity_v6",
    SUPP
)

# ============================================================
# FIGURE S2 - FRAGMENTATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.8,3.3),
    constrained_layout=True
)

panel(ax, "A")

order = [
    "full","100kb","50kb",
    "20kb","5kb","5x5kb"
]

labels = [
    "Full",
    "100 kb",
    "50 kb",
    "20 kb",
    "5 kb",
    "5×5 kb"
]

z = e8.set_index(
    "condition"
).loc[order]

y = np.arange(6)[::-1]

ax.errorbar(
    z["AUROC_mean"],
    y + 0.09,
    xerr=z["AUROC_sd"],
    fmt="o",
    color=ACCENT,
    capsize=2,
    label="AUROC"
)

ax.errorbar(
    z["AUPRC_mean"],
    y - 0.09,
    xerr=z["AUPRC_sd"],
    fmt="s",
    color=SECOND,
    capsize=2,
    label="AUPRC"
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Performance")
ax.set_title(
    "Fragmentation sensitivity",
    loc="left",
    fontweight="bold"
)
ax.legend(frameon=False)
clean(ax, "x")

save(
    fig,
    "FigureS2_fragmentation_v6",
    SUPP
)

# ============================================================
# FIGURE S3 - IG
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.6,4.1),
    constrained_layout=True
)

panel(ax, "A")

top = e9.nsmallest(
    20,
    "rank"
).sort_values(
    "mean_abs_pair_IG"
)

y = np.arange(len(top))

ax.barh(
    y,
    top["mean_abs_pair_IG"],
    color=ACCENT
)

ax.set_yticks(y)
ax.set_yticklabels(
    top["canonical_kmer"]
)

ax.set_xlabel(
    "Mean absolute pair integrated gradient"
)

ax.set_title(
    "Top canonical 6-mer feature sensitivities",
    loc="left",
    fontweight="bold"
)

ax.text(
    0.98,0.03,
    "Feature sensitivity only;\n"
    "not functional motif evidence",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=6.5,
    color=MUTED
)

clean(ax, "x")

save(
    fig,
    "FigureS3_integrated_gradients_v6",
    SUPP
)

# ============================================================
# FIGURE S4 - GTDB TREE
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.5,4.1),
    constrained_layout=True
)

panel(ax, "A")

x = e10tree[
    "gtdb_tree_patristic_distance"
].to_numpy()

y = e10tree[
    "hgv_distance_mean_3seed"
].to_numpy()

hb = ax.hexbin(
    x,
    y,
    gridsize=48,
    mincnt=1,
    cmap="viridis"
)

# binned median trend
edges = np.linspace(
    np.nanmin(x),
    np.nanmax(x),
    18
)

centres = []
medians = []

for lo, hi in zip(
    edges[:-1],
    edges[1:]
):
    mask = (
        (x >= lo) &
        (x < hi)
    )

    if mask.sum() >= 20:
        centres.append(
            (lo + hi) / 2
        )
        medians.append(
            np.nanmedian(y[mask])
        )

ax.plot(
    centres,
    medians,
    color="white",
    lw=1.5,
    marker="o",
    ms=2.5,
    label="Binned median"
)

summary = e10treesum.iloc[0]

ax.text(
    0.04,0.96,
    f"Spearman ρ = {summary['spearman_rho']:.3f}\n"
    f"Permutation P = {summary['mantel_style_two_sided_p']:.4f}",
    transform=ax.transAxes,
    va="top",
    fontsize=7,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.90,
        pad=2
    )
)

ax.set_xlabel(
    "GTDB bac120 patristic distance"
)

ax.set_ylabel(
    "HGV embedding distance"
)

ax.set_title(
    "Weak GTDB tree-distance consistency",
    loc="left",
    fontweight="bold"
)

clean(ax, "y")

save(
    fig,
    "FigureS4_GTDB_tree_consistency_v6",
    SUPP
)

# ============================================================
# DRAFT CAPTIONS
# ============================================================

caption_text = """HGV-V3 FIGURE PACKAGE v6

Figure 1.
Study design, hierarchical taxonomic evaluation, HGV representation,
selective teacher regularization, and intended first-stage reference-screening workflow.

Figure 2.
Internal 11-method comparison across species-, genus-, and family-level
taxonomic holdouts. Values are protocol-level means. Coverage must be interpreted
jointly with conditional metrics for methods with incomplete pair coverage.

Figure 3.
Cross-source external 11-method reference retrieval. B1 evaluates matched-species
retrieval; B2 seen-genus evaluates genus-level retrieval for unseen species;
B2 unseen-genus evaluates family-level retrieval for queries outside development genera.

Figure 4.
Ablation of distillation strategy. D3 is selective informative-negative distillation.
Formal paired bootstrap contrasts do not support universal superiority of D3.

Figure 5.
Biological fidelity of the HGV embedding space. HGV distance increases with
taxonomic separation and remains associated with ANI-related structure after adjustment
for measured composition/assembly covariates. Hard-negative residual association is weak.

Figure 6.
Reference-screening utility and computational profile. HGV should be interpreted
as a first-stage vector prefilter followed by confirmatory ANI analysis rather than
as a replacement for skani or FastANI.

Figure S1.
Sensitivity to k-mer size, embedding dimension, contrastive margin and distillation weight.
k=7 gives slightly higher discrimination than k=6 but at substantially larger input dimension;
k=6 is retained as the frozen accuracy-compactness trade-off.

Figure S2.
Fragmentation sensitivity. Performance declines as available genomic context is reduced;
multiple short fragments partially recover discrimination relative to a single short fragment.

Figure S3.
Integrated-gradient feature sensitivity of canonical 6-mers. These values represent
model sensitivity and should not be interpreted as functional motif evidence.

Figure S4.
Association between HGV distance and GTDB bac120 patristic distance. The relationship
is weak but statistically detectable and is not evidence of phylogeny reconstruction.
"""

(CAP / "DRAFT_FIGURE_CAPTIONS_v6.txt").write_text(
    caption_text
)

# ============================================================
# FINAL AUDIT
# ============================================================

main_expected = [
    MAIN / "Figure1_study_design_v6.png",
    MAIN / "Figure2_internal_11method_v6.png",
    MAIN / "Figure3_external_11method_v6.png",
    MAIN / "Figure4_distillation_v6.png",
    MAIN / "Figure5_biological_fidelity_v6.png",
    MAIN / "Figure6_screening_scalability_v6.png",
]

supp_expected = [
    SUPP / "FigureS1_sensitivity_v6.png",
    SUPP / "FigureS2_fragmentation_v6.png",
    SUPP / "FigureS3_integrated_gradients_v6.png",
    SUPP / "FigureS4_GTDB_tree_consistency_v6.png",
]

tables_expected = [
    TABLE / "Table1_dataset_design.tsv",
    TABLE / "Table2_internal_11method_benchmark.tsv",
    TABLE / "Table3_external_11method_benchmark.tsv",
    TABLE / "Table4_E6_ablation.tsv",
    TABLE / "Table4B_E6_formal_contrasts.tsv",
    TABLE / "Table5A_reference_screening.tsv",
    TABLE / "Table5B_scalability.tsv",
]

missing = [
    p for p in
    main_expected +
    supp_expected +
    tables_expected
    if not p.exists()
]

audit = [
    "HGV-V3 FINAL MANUSCRIPT PACKAGE AUDIT v6",
    "=" * 80,
    f"internal methods = {internal['method'].nunique()}",
    f"internal protocol rows = {len(internal)}",
    f"internal split rows = {len(internal_split)}",
    f"external methods = {external['method'].nunique()}",
    f"external summary rows = {len(external)}",
    f"main figures = {sum(p.exists() for p in main_expected)}/6",
    f"supp figures = {sum(p.exists() for p in supp_expected)}/4",
    f"main table blocks = {sum(p.exists() for p in tables_expected)}/{len(tables_expected)}",
    "",
]

if missing:
    audit.append("FINAL_PACKAGE_GATE=FAIL")
    audit += [
        f"MISSING\t{p}"
        for p in missing
    ]
else:
    audit.append("FINAL_PACKAGE_GATE=PASS")

(AUDIT / "FINAL_PACKAGE_AUDIT_v6.txt").write_text(
    "\n".join(audit) + "\n"
)

print()
print("\n".join(audit))
print()
print("ROOT =", ROOT)

if missing:
    raise SystemExit(2)
