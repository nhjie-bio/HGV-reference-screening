from pathlib import Path
import re
import sys
import math
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

ROOT = BASE / "11_manuscript" / \
    "HGV_V3_final_manuscript_figures_v3_20260926"

SRC = ROOT / "00_source_snapshot"
NORM = ROOT / "01_normalized_source_data"
MAIN = ROOT / "02_figures" / "main"
SUPP = ROOT / "02_figures" / "supplementary"
AUDIT = ROOT / "03_audit"
TABLES = ROOT / "04_tables"

for d in [NORM, MAIN, SUPP, AUDIT, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

COLORS = {
    "HGV": "#0072B2",
    "Raw-k6": "#009E73",
    "PCA64": "#E69F00",
    "Mash": "#CC79A7",
    "Dashing2": "#D55E00",
    "DNABERT-S": "#56B4E9",
    "FastANI": "#999999",
    "skani": "#000000",
}

MODE_COLORS = {
    "D0": "#999999",
    "D1": "#56B4E9",
    "D2": "#E69F00",
    "D3": "#0072B2",
    "D4": "#CC79A7",
    "D5": "#D55E00",
}

# ============================================================
# HELPERS
# ============================================================

def clean_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_col(df, aliases):
    mapping = {clean_name(c): c for c in df.columns}

    for a in aliases:
        aa = clean_name(a)
        if aa in mapping:
            return mapping[aa]

    for a in aliases:
        aa = clean_name(a)
        for k, v in mapping.items():
            if aa and (aa in k or k in aa):
                return v

    return None


def read_table(path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return None


def all_tables():
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() not in {".tsv", ".csv"}:
            continue
        df = read_table(p)
        if df is None or len(df.columns) == 0:
            continue
        out.append((p, df))
    return out


TABLE_CACHE = all_tables()


def text_blob(df, n=100):
    try:
        x = df.head(n).astype(str)
        return " ".join(x.values.ravel()).lower()
    except Exception:
        return ""


def path_score(path, positive=(), negative=()):
    s = str(path).lower()
    score = 0
    for x in positive:
        if x.lower() in s:
            score += 4
    for x in negative:
        if x.lower() in s:
            score -= 5
    return score


def save_source_choice(name, path, df, score):
    with open(AUDIT / "SELECTED_SOURCES.tsv", "a") as f:
        f.write(
            f"{name}\t{path}\t{len(df)}\t"
            f"{len(df.columns)}\t{score}\n"
        )


def choose_table(name, scorer):
    best = None

    for path, df in TABLE_CACHE:
        try:
            score = scorer(path, df)
        except Exception:
            continue

        if score is None:
            continue

        if best is None or score > best[0]:
            best = (score, path, df)

    if best is None or best[0] <= 0:
        raise RuntimeError(
            f"Could not identify a trustworthy source table for {name}"
        )

    score, path, df = best
    save_source_choice(name, path, df, score)

    print(
        f"[SOURCE] {name:18s} "
        f"score={score:3d} rows={len(df):6d} "
        f"{path}"
    )

    return df.copy(), path


def metric_column(df, metric):
    aliases = {
        "AUROC": ["AUROC", "ROC_AUC", "ROC-AUC", "auc"],
        "AUPRC": ["AUPRC", "AUPR", "PR_AUC",
                  "average_precision"],
        "MCC": ["MCC", "matthews_corrcoef"],
        "MRR": ["MRR", "mean_reciprocal_rank"],
    }
    return find_col(df, aliases[metric])


def add_panel(ax, label):
    ax.text(
        -0.12, 1.07, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top"
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)


def save_fig(fig, basename, folder):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            folder / f"{basename}.{ext}",
            bbox_inches="tight",
            dpi=600 if ext == "png" else None
        )
    plt.close(fig)


def pretty_method(x):
    s = str(x).lower()

    if "hgv" in s:
        return "HGV"

    if "raw" in s and ("euclid" in s or "l2" in s):
        return "Raw-k6"

    if "pca" in s and "euclid" in s:
        return "PCA64"

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

    return str(x)


# ============================================================
# FROZEN DATASET COUNTS
# ============================================================

EXPECTED_COUNTS = {
    "A": 2374,
    "B1": 391,
    "B2_seen": 196,
    "B2_unseen": 193,
}

assert sum(EXPECTED_COUNTS.values()) == 3154


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================

def score_internal(path, df):
    pcol = find_col(
        df, ["protocol", "evaluation_protocol",
             "holdout", "split_protocol"]
    )
    mcol = find_col(df, ["method", "model"])

    if pcol is None or mcol is None:
        return -100

    blob = text_blob(df, 200)

    score = 0

    if "g2" in blob:
        score += 8
    if "g3" in blob:
        score += 8
    if "g4" in blob:
        score += 8

    for metric in ["AUROC", "AUPRC", "MCC"]:
        if metric_column(df, metric):
            score += 5

    if len(df) == 99:
        score += 10
    if len(df) == 33:
        score += 5

    score += path_score(
        path,
        positive=["internal", "e5", "unified", "final", "freeze"],
        negative=["invalid", "old", "tmp"]
    )

    return score


def score_e6_modes(path, df):
    mcol = find_col(df, ["mode", "ablation_mode", "group"])
    blob = text_blob(df, 200)

    score = 0

    if mcol:
        vals = " ".join(df[mcol].astype(str).tolist()).lower()
        for d in ["d0", "d1", "d2", "d3", "d4", "d5"]:
            if d in vals:
                score += 5

    if "d0" in blob and "d5" in blob:
        score += 10

    if metric_column(df, "AUROC"):
        score += 5
    if metric_column(df, "AUPRC"):
        score += 5

    score += path_score(
        path,
        positive=["e6", "mode", "ablation", "final", "freeze"],
        negative=["contrast", "invalid", "old"]
    )

    return score


def score_e6_contrast(path, df):
    score = path_score(
        path,
        positive=["e6", "contrast", "paired", "bootstrap",
                  "final", "freeze"],
        negative=["invalid", "old"]
    )

    if len(df) == 10:
        score += 20

    for names in [
        ["contrast", "comparison", "control"],
        ["ci_low", "lower", "lcl"],
        ["ci_high", "upper", "ucl"],
    ]:
        if find_col(df, names):
            score += 5

    blob = text_blob(df, 100)
    if "d3" in blob:
        score += 10

    return score


def score_e7(path, df):
    score = path_score(
        path,
        positive=["e7", "sensitivity", "final", "corrected"],
        negative=["invalid", "old"]
    )

    if len(df) == 9:
        score += 20

    scol = find_col(df, ["setting", "configuration", "variant"])
    if scol:
        score += 5

    if metric_column(df, "AUROC"):
        score += 5

    return score


def score_e8(path, df):
    score = path_score(
        path,
        positive=["e8", "fragment", "robust", "final"],
        negative=["invalid", "old"]
    )

    if len(df) == 6:
        score += 20

    blob = text_blob(df, 100)

    for token in ["100", "50", "20", "5"]:
        if token in blob:
            score += 2

    if metric_column(df, "AUROC"):
        score += 5

    return score


def score_e9(path, df):
    score = path_score(
        path,
        positive=["e9", "explain", "attribution", "confound"],
        negative=["e10", "invalid", "old"]
    )

    blob = text_blob(df, 100)

    for token in [
        "gc", "length", "n50",
        "contig", "completeness"
    ]:
        if token in blob:
            score += 4

    return score


def score_e10_taxonomy(path, df):
    score = path_score(
        path,
        positive=["e10", "taxonomy", "taxonomic",
                  "strata", "biological"],
        negative=["invalid", "old"]
    )

    blob = text_blob(df, 200)

    targets = [
        "same species",
        "same genus",
        "same family",
        "different family",
    ]

    for x in targets:
        if x in blob:
            score += 8

    return score


def score_e10_pairs(path, df):
    ani = find_col(df, ["ANI", "ani_percent", "skani_ani"])
    af = find_col(df, ["AF_min", "AF", "alignment_fraction"])
    hgv = find_col(
        df,
        ["hgv_distance", "embedding_distance",
         "student_distance", "distance_hgv"]
    )

    if hgv is None:
        return -100

    score = 0

    if ani:
        score += 20
    if af:
        score += 20

    if len(df) >= 100:
        score += 10

    score += path_score(
        path,
        positive=["e10", "ani", "af", "fidelity",
                  "pair", "biological"],
        negative=["invalid", "old"]
    )

    return score


def score_e10_confound(path, df):
    score = path_score(
        path,
        positive=["e10", "confound", "partial", "adjust"],
        negative=["e9", "invalid", "old"]
    )

    if len(df) == 3:
        score += 20

    for names in [
        ["raw", "raw_rho", "raw_spearman"],
        ["partial", "partial_rho", "adjusted"],
    ]:
        if find_col(df, names):
            score += 8

    return score


def score_e11(path, df):
    score = path_score(
        path,
        positive=["e11", "screen", "retrieval",
                  "external", "reference"],
        negative=["invalid", "old"]
    )

    if len(df) == 3:
        score += 15

    if find_col(df, ["MRR"]):
        score += 10

    for k in [1, 5, 10, 20, 50]:
        if find_col(
            df,
            [f"Recall@{k}", f"Recall_{k}",
             f"R{k}", f"recall{k}"]
        ):
            score += 4

    blob = text_blob(df, 100)

    if "b1" in blob:
        score += 5
    if "seen" in blob:
        score += 5
    if "unseen" in blob:
        score += 5

    return score


def score_e12(path, df):
    score = path_score(
        path,
        positive=["e12", "scalability", "runtime",
                  "benchmark", "final"],
        negative=["v1", "invalid", "old"]
    )

    if len(df) == 5:
        score += 15

    if find_col(
        df,
        ["seconds", "time_sec", "runtime_seconds",
         "elapsed_seconds", "mean_seconds"]
    ):
        score += 15

    return score


# ============================================================
# SELECT SOURCES
# ============================================================

(AUDIT / "SELECTED_SOURCES.tsv").write_text(
    "block\tpath\trows\tcolumns\tscore\n"
)

internal, p_internal = choose_table(
    "internal", score_internal
)

e6_modes, p_e6_modes = choose_table(
    "e6_modes", score_e6_modes
)

e6_contrasts, p_e6_contrasts = choose_table(
    "e6_contrasts", score_e6_contrast
)

e7, p_e7 = choose_table(
    "e7", score_e7
)

e8, p_e8 = choose_table(
    "e8", score_e8
)

e9, p_e9 = choose_table(
    "e9", score_e9
)

e10_tax, p_e10_tax = choose_table(
    "e10_taxonomy", score_e10_taxonomy
)

e10_pairs, p_e10_pairs = choose_table(
    "e10_pairs", score_e10_pairs
)

e10_conf, p_e10_conf = choose_table(
    "e10_confound", score_e10_confound
)

e11, p_e11 = choose_table(
    "e11", score_e11
)

e12, p_e12 = choose_table(
    "e12", score_e12
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_internal(df):
    p = find_col(
        df,
        ["protocol", "evaluation_protocol",
         "holdout", "split_protocol"]
    )
    m = find_col(df, ["method", "model"])

    if not p or not m:
        raise ValueError("Internal table lacks protocol/method")

    out = pd.DataFrame()
    out["protocol"] = df[p].astype(str)
    out["method_raw"] = df[m].astype(str)
    out["method"] = out["method_raw"].map(pretty_method)

    seed = find_col(
        df,
        ["taxonomy_split_seed", "split_seed",
         "tax_seed"]
    )

    if seed:
        out["split_seed"] = df[seed].astype(str)
    else:
        out["split_seed"] = "summary"

    for metric in ["AUROC", "AUPRC", "MCC"]:
        c = metric_column(df, metric)
        if c:
            out[metric] = pd.to_numeric(
                df[c], errors="coerce"
            )

    out = out[
        out["protocol"]
        .str.upper()
        .isin(["G2", "G3", "G4"])
    ]

    return out.dropna(
        subset=[x for x in ["AUROC", "AUPRC"]
                if x in out.columns],
        how="all"
    )


def normalize_e6_modes(df):
    mode = find_col(
        df, ["mode", "ablation_mode", "group"]
    )

    if not mode:
        raise ValueError("Cannot identify D0-D5 mode column")

    metric = find_col(df, ["metric", "measure"])
    value = find_col(df, ["value", "mean", "score"])

    id_seed = find_col(
        df,
        ["model_seed", "seed", "run_seed"]
    )

    if metric and value:
        ids = [mode]
        if id_seed:
            ids.append(id_seed)

        tmp = df[
            ids + [metric, value]
        ].copy()

        tmp[value] = pd.to_numeric(
            tmp[value], errors="coerce"
        )

        tmp[metric] = (
            tmp[metric].astype(str).str.upper()
        )

        out = tmp.pivot_table(
            index=ids,
            columns=metric,
            values=value,
            aggfunc="mean"
        ).reset_index()

    else:
        out = pd.DataFrame()
        out["mode"] = df[mode].astype(str)

        if id_seed:
            out["seed"] = df[id_seed].astype(str)

        for x in ["AUROC", "AUPRC", "MCC"]:
            c = metric_column(df, x)
            if c:
                out[x] = pd.to_numeric(
                    df[c], errors="coerce"
                )

        return out

    out = out.rename(columns={mode: "mode"})

    if id_seed:
        out = out.rename(columns={id_seed: "seed"})

    return out


def normalize_e6_contrasts(df):
    comp = find_col(
        df,
        ["contrast", "comparison", "control",
         "comparator"]
    )
    metric = find_col(df, ["metric", "measure"])

    est = find_col(
        df,
        ["estimate", "difference", "delta",
         "mean_difference", "effect"]
    )

    lo = find_col(
        df,
        ["ci_low", "lower", "lower_ci",
         "lcl", "ci2.5"]
    )

    hi = find_col(
        df,
        ["ci_high", "upper", "upper_ci",
         "ucl", "ci97.5"]
    )

    if not all([comp, est, lo, hi]):
        raise ValueError(
            "Formal E6 contrast table lacks "
            "comparison/effect/CI columns"
        )

    out = pd.DataFrame({
        "comparison": df[comp].astype(str),
        "effect": pd.to_numeric(
            df[est], errors="coerce"
        ),
        "ci_low": pd.to_numeric(
            df[lo], errors="coerce"
        ),
        "ci_high": pd.to_numeric(
            df[hi], errors="coerce"
        ),
    })

    if metric:
        out["metric"] = (
            df[metric].astype(str).str.upper()
        )
    else:
        out["metric"] = "AUROC"

    pcol = find_col(
        df,
        ["holm_p", "p_holm", "adjusted_p",
         "p_adj", "padj"]
    )

    if pcol:
        out["p_holm"] = pd.to_numeric(
            df[pcol], errors="coerce"
        )

    return out


def normalize_e7(df):
    s = find_col(
        df,
        ["setting", "configuration",
         "variant", "label"]
    )

    if not s:
        s = df.columns[0]

    out = pd.DataFrame({
        "setting": df[s].astype(str)
    })

    for x in ["AUROC", "AUPRC", "MCC"]:
        c = metric_column(df, x)
        if c:
            out[x] = pd.to_numeric(
                df[c], errors="coerce"
            )

    return out


def normalize_e8(df):
    c = find_col(
        df,
        ["fragment", "condition",
         "setting", "fragmentation"]
    )

    if not c:
        c = df.columns[0]

    auc = metric_column(df, "AUROC")

    if not auc:
        raise ValueError("E8 lacks AUROC")

    return pd.DataFrame({
        "condition": df[c].astype(str),
        "AUROC": pd.to_numeric(
            df[auc], errors="coerce"
        ),
    })


def normalize_e9(df):
    cov = find_col(
        df,
        ["covariate", "feature",
         "variable", "confound"]
    )

    rho = find_col(
        df,
        ["spearman", "rho",
         "correlation", "spearman_rho"]
    )

    if cov and rho:
        return pd.DataFrame({
            "covariate": df[cov].astype(str),
            "rho": pd.to_numeric(
                df[rho], errors="coerce"
            ),
        })

    # fallback: table with covariates as columns
    rows = []
    aliases = {
        "ΔGC": ["delta_gc", "gc_difference", "gc"],
        "Genome length": ["genome_length", "length"],
        "N50": ["n50"],
        "Contigs": ["contig_count", "contigs"],
        "Completeness": ["completeness"],
    }

    for name, aa in aliases.items():
        x = find_col(df, aa)
        if x:
            v = pd.to_numeric(
                df[x], errors="coerce"
            ).dropna()
            if len(v):
                rows.append({
                    "covariate": name,
                    "rho": float(v.iloc[0])
                })

    if not rows:
        raise ValueError(
            "Cannot normalize E9 confound table"
        )

    return pd.DataFrame(rows)


def normalize_taxonomy(df):
    cat = find_col(
        df,
        ["taxonomic_stratum", "stratum",
         "taxonomy_group", "category"]
    )

    dist = find_col(
        df,
        ["hgv_distance", "embedding_distance",
         "distance", "mean_distance"]
    )

    if not cat or not dist:
        raise ValueError(
            "Cannot identify taxonomy stratum/distance"
        )

    out = pd.DataFrame({
        "stratum": df[cat].astype(str),
        "distance": pd.to_numeric(
            df[dist], errors="coerce"
        ),
    })

    return out.dropna()


def normalize_pairs(df):
    ani = find_col(
        df,
        ["ANI", "ani_percent", "skani_ani"]
    )

    af = find_col(
        df,
        ["AF_min", "af_min", "AF",
         "alignment_fraction"]
    )

    hgv = find_col(
        df,
        ["hgv_distance", "embedding_distance",
         "student_distance", "distance_hgv"]
    )

    if hgv is None:
        raise ValueError("No HGV distance in E10 pair table")

    out = pd.DataFrame({
        "HGV_distance": pd.to_numeric(
            df[hgv], errors="coerce"
        )
    })

    if ani:
        out["ANI"] = pd.to_numeric(
            df[ani], errors="coerce"
        )

    if af:
        out["AF"] = pd.to_numeric(
            df[af], errors="coerce"
        )

    return out


def normalize_confound(df):
    subset = find_col(
        df,
        ["subset", "group", "stratum", "scope"]
    )

    raw = find_col(
        df,
        ["raw_rho", "raw_spearman",
         "raw_rank_correlation", "raw"]
    )

    partial = find_col(
        df,
        ["partial_rho", "partial_spearman",
         "partial_rank_correlation",
         "adjusted"]
    )

    p = find_col(
        df,
        ["permutation_p", "p_value",
         "pvalue", "p"]
    )

    if not raw or not partial:
        raise ValueError(
            "No raw/partial correlation columns"
        )

    out = pd.DataFrame({
        "subset":
            df[subset].astype(str)
            if subset else
            [f"row{i+1}" for i in range(len(df))],

        "raw_rho":
            pd.to_numeric(
                df[raw], errors="coerce"
            ),

        "partial_rho":
            pd.to_numeric(
                df[partial], errors="coerce"
            ),
    })

    if p:
        out["permutation_p"] = pd.to_numeric(
            df[p], errors="coerce"
        )

    return out


def normalize_e11(df):
    cohort = find_col(
        df,
        ["cohort", "endpoint",
         "group", "external_group"]
    )

    if not cohort:
        cohort = df.columns[0]

    out = pd.DataFrame({
        "cohort": df[cohort].astype(str)
    })

    mrr = metric_column(df, "MRR")
    if mrr:
        out["MRR"] = pd.to_numeric(
            df[mrr], errors="coerce"
        )

    for k in [1, 5, 10, 20, 50]:
        c = find_col(
            df,
            [
                f"Recall@{k}",
                f"Recall_{k}",
                f"Recall{k}",
                f"R{k}"
            ]
        )

        if c:
            out[f"R{k}"] = pd.to_numeric(
                df[c], errors="coerce"
            )

    return out


def normalize_e12(df):
    stage = find_col(
        df,
        ["stage", "component",
         "operation", "benchmark"]
    )

    sec = find_col(
        df,
        ["seconds", "time_sec",
         "runtime_seconds",
         "elapsed_seconds",
         "mean_seconds"]
    )

    if not stage or not sec:
        raise ValueError(
            "Cannot identify E12 stage/runtime"
        )

    out = pd.DataFrame({
        "stage": df[stage].astype(str),
        "seconds": pd.to_numeric(
            df[sec], errors="coerce"
        ),
    })

    return out.dropna()


I = normalize_internal(internal)
E6M = normalize_e6_modes(e6_modes)
E6C = normalize_e6_contrasts(e6_contrasts)
E7 = normalize_e7(e7)
E8 = normalize_e8(e8)
E9 = normalize_e9(e9)
TAX = normalize_taxonomy(e10_tax)
PAIR = normalize_pairs(e10_pairs)
CONF = normalize_confound(e10_conf)
E11 = normalize_e11(e11)
E12 = normalize_e12(e12)

NORMALIZED = {
    "fig2_internal.tsv": I,
    "fig4_e6_modes.tsv": E6M,
    "fig4_e6_contrasts.tsv": E6C,
    "figS1_e7_sensitivity.tsv": E7,
    "figS2_e8_fragmentation.tsv": E8,
    "figS3_e9_confound.tsv": E9,
    "fig5_taxonomy.tsv": TAX,
    "fig5_pairwise_ani_af.tsv": PAIR,
    "fig5_confound_adjusted.tsv": CONF,
    "fig3_fig6_e11_screening.tsv": E11,
    "fig6_e12_runtime.tsv": E12,
}

for fname, df in NORMALIZED.items():
    df.to_csv(
        NORM / fname,
        sep="\t",
        index=False
    )


# ============================================================
# TABLES FOR MANUSCRIPT
# ============================================================

# Internal summary
summary_rows = []

for (protocol, method), g in I.groupby(
    ["protocol", "method"]
):
    row = {
        "protocol": protocol,
        "method": method,
        "n_rows": len(g),
    }

    for metric in ["AUROC", "AUPRC", "MCC"]:
        if metric in g:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_sd"] = g[metric].std(ddof=1)

    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(
    TABLES / "Table_internal_generalization.tsv",
    sep="\t",
    index=False
)

E11.to_csv(
    TABLES / "Table_external_screening.tsv",
    sep="\t",
    index=False
)

E6M.to_csv(
    TABLES / "Table_E6_ablation.tsv",
    sep="\t",
    index=False
)


# ============================================================
# FIGURE 1
# ============================================================

fig = plt.figure(figsize=(7.2, 6.1))
gs = gridspec.GridSpec(
    3, 2,
    height_ratios=[1.05, 1, 1],
    hspace=0.42,
    wspace=0.26
)

# A Dataset
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
add_panel(ax, "A")

nodes = [
    (0.05, 0.73, 0.90, 0.16,
     "GTDB R232 bacterial metadata\n878,998 genomes"),
    (0.05, 0.48, 0.90, 0.16,
     "Strict metadata QC\n172,933 candidates"),
    (0.05, 0.23, 0.90, 0.16,
     "Primary cohort selection\n3,200 genomes"),
    (0.05, -0.02, 0.90, 0.16,
     "Sequence QC + redundancy audit\n3,154 retained genomes"),
]

for x, y, w, h, txt in nodes:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012",
        linewidth=0.8,
        facecolor="#F4F4F4",
        edgecolor="#444444"
    )
    ax.add_patch(box)
    ax.text(
        x+w/2, y+h/2,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

for y1, y2 in [
    (0.73, 0.64),
    (0.48, 0.39),
    (0.23, 0.14),
]:
    ax.annotate(
        "",
        xy=(0.50, y2),
        xytext=(0.50, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8
        )
    )

ax.set_xlim(0, 1)
ax.set_ylim(-0.08, 1.0)
ax.set_title(
    "Dataset reconstruction and QC",
    loc="left",
    pad=4
)

# B Cohorts
ax = fig.add_subplot(gs[0, 1])
add_panel(ax, "B")

labs = [
    "Development\nA",
    "B1\nmatched species",
    "B2\nseen genus",
    "B2\nunseen genus",
]

vals = [2374, 391, 196, 193]

bars = ax.bar(
    np.arange(4),
    vals,
    edgecolor="black",
    linewidth=0.5
)

ax.set_xticks(np.arange(4))
ax.set_xticklabels(labs)
ax.set_ylabel("Genomes")
ax.set_title(
    "Final 3,154-genome analysis cohort",
    loc="left"
)

for b, v in zip(bars, vals):
    ax.text(
        b.get_x()+b.get_width()/2,
        v+40,
        f"{v:,}",
        ha="center",
        fontsize=7
    )

clean_axes(ax)

# C Generalization design
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
add_panel(ax, "C")
ax.set_title(
    "Hierarchical taxonomic holdout",
    loc="left"
)

labels = [
    ("G2", "Unseen species\nGenus may be seen", 0.13),
    ("G3", "Unseen genus", 0.50),
    ("G4", "Unseen family", 0.87),
]

for tag, txt, x in labels:
    circ = plt.Circle(
        (x, 0.62),
        0.085,
        fill=False,
        linewidth=1.4
    )
    ax.add_patch(circ)
    ax.text(
        x, 0.62,
        tag,
        ha="center",
        va="center",
        fontweight="bold"
    )
    ax.text(
        x, 0.30,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

ax.annotate(
    "",
    xy=(0.91, 0.90),
    xytext=(0.08, 0.90),
    arrowprops=dict(
        arrowstyle="->",
        lw=1.0
    )
)

ax.text(
    0.50, 0.94,
    "Increasing taxonomic novelty",
    ha="center",
    fontsize=7
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# D Architecture
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
add_panel(ax, "D")
ax.set_title(
    "Hyper-compact HGV representation",
    loc="left"
)

arch = [
    ("Whole\nassembly", 0.03, 0.58, 0.18),
    ("Canonical\nk=6\n2,080-D", 0.27, 0.58, 0.18),
    ("256", 0.51, 0.58, 0.12),
    ("128", 0.68, 0.58, 0.12),
    ("64-D\nL2", 0.84, 0.58, 0.13),
]

for txt, x, y, w in arch:
    box = FancyBboxPatch(
        (x, y), w, 0.22,
        boxstyle="round,pad=0.01",
        linewidth=0.8,
        facecolor="#F4F4F4"
    )
    ax.add_patch(box)
    ax.text(
        x+w/2, y+0.11,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

for a, b in zip(arch[:-1], arch[1:]):
    ax.annotate(
        "",
        xy=(b[1], 0.69),
        xytext=(a[1]+a[3], 0.69),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8
        )
    )

ax.text(
    0.50, 0.31,
    "2,080 → 64 dimensions (32.5× reduction)",
    ha="center",
    fontsize=8,
    fontweight="bold"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# E Distillation
ax = fig.add_subplot(gs[2, 0])
ax.axis("off")
add_panel(ax, "E")
ax.set_title(
    "Selective teacher regularization",
    loc="left"
)

for txt, x, y in [
    ("Contrastive\nsupervision", 0.08, 0.58),
    ("Dashing2\nteacher", 0.08, 0.20),
    ("HGV\nstudent", 0.69, 0.39),
]:
    box = FancyBboxPatch(
        (x, y), 0.23, 0.20,
        boxstyle="round,pad=0.01",
        facecolor="#F4F4F4",
        linewidth=0.8
    )
    ax.add_patch(box)
    ax.text(
        x+0.115, y+0.10,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

ax.annotate(
    "",
    xy=(0.69, 0.51),
    xytext=(0.31, 0.68),
    arrowprops=dict(
        arrowstyle="->",
        lw=1
    )
)

ax.annotate(
    "",
    xy=(0.69, 0.45),
    xytext=(0.31, 0.30),
    arrowprops=dict(
        arrowstyle="->",
        lw=1
    )
)

ax.text(
    0.49, 0.24,
    "Informative negatives\nteacher distance < τ",
    ha="center",
    fontsize=7
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# F use case
ax = fig.add_subplot(gs[2, 1])
ax.axis("off")
add_panel(ax, "F")
ax.set_title(
    "Intended screening workflow",
    loc="left"
)

steps = [
    ("Query\ngenome", 0.01),
    ("64-D\nHGV", 0.22),
    ("Vector\nsearch", 0.43),
    ("Top-K\nreferences", 0.64),
    ("skani /\nFastANI", 0.84),
]

for txt, x in steps:
    box = FancyBboxPatch(
        (x, 0.42), 0.15, 0.25,
        boxstyle="round,pad=0.01",
        facecolor="#F4F4F4",
        linewidth=0.8
    )
    ax.add_patch(box)
    ax.text(
        x+0.075, 0.545,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

for a, b in zip(steps[:-1], steps[1:]):
    ax.annotate(
        "",
        xy=(b[1], 0.545),
        xytext=(a[1]+0.15, 0.545),
        arrowprops=dict(
            arrowstyle="->",
            lw=0.8
        )
    )

ax.text(
    0.50, 0.20,
    "First-stage prefilter → confirmatory ANI",
    ha="center",
    fontsize=8,
    fontweight="bold"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.suptitle(
    "HGV-V3 study design and evaluation framework",
    fontsize=11,
    y=0.995
)

save_fig(fig, "Figure1_HGV_V3_study_design", MAIN)


# ============================================================
# FIGURE 2
# ============================================================

plot_methods = [
    "HGV",
    "Raw-k6",
    "PCA64",
    "Mash",
    "Dashing2",
    "DNABERT-S",
]

P = I[I["method"].isin(plot_methods)].copy()

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 5.6)
)

for ax, metric, label in zip(
    axes.flat[:3],
    ["AUROC", "AUPRC", "MCC"],
    ["A", "B", "C"]
):
    add_panel(ax, label)

    if metric not in P.columns:
        ax.text(
            0.5, 0.5,
            f"{metric} unavailable",
            ha="center",
            transform=ax.transAxes
        )
        continue

    protocols = ["G2", "G3", "G4"]

    width = 0.11
    offsets = np.linspace(
        -0.28,
        0.28,
        len(plot_methods)
    )

    for mi, method in enumerate(plot_methods):
        q = P[P["method"] == method]

        if q.empty:
            continue

        for pi, protocol in enumerate(protocols):
            z = q[q["protocol"].str.upper() == protocol]

            vals = z[metric].dropna().to_numpy()

            if len(vals) == 0:
                continue

            x = pi + offsets[mi]

            if len(vals) > 1:
                jitter = np.linspace(
                    -0.018, 0.018, len(vals)
                )
                ax.scatter(
                    x+jitter,
                    vals,
                    s=11,
                    alpha=0.55,
                    color=COLORS.get(method)
                )

            ax.scatter(
                [x],
                [np.mean(vals)],
                s=30,
                color=COLORS.get(method),
                edgecolor="black",
                linewidth=0.35,
                zorder=5,
                label=method if pi == 0 else None
            )

    ax.set_xticks(range(3))
    ax.set_xticklabels([
        "G2\nspecies",
        "G3\ngenus",
        "G4\nfamily"
    ])

    ax.set_ylabel(metric)

    if metric in {"AUROC", "AUPRC"}:
        ax.set_ylim(0.85, 1.01)

    ax.set_title(
        f"{metric} across taxonomic holdouts",
        loc="left"
    )

    clean_axes(ax)

axes[0, 0].legend(
    ncol=2,
    frameon=False,
    loc="lower left"
)

# Compactness
ax = axes[1, 1]
add_panel(ax, "D")

compact = pd.DataFrame({
    "Representation": [
        "Raw-k6",
        "PCA64",
        "HGV"
    ],
    "Dimensions": [
        2080,
        64,
        64
    ]
})

# G2 preferred here for identical task context
for method in compact["Representation"]:
    z = P[
        (P["method"] == method) &
        (P["protocol"].str.upper() == "G2")
    ]

    if len(z) and "AUROC" in z:
        y = z["AUROC"].mean()
        x = compact.loc[
            compact["Representation"] == method,
            "Dimensions"
        ].iloc[0]

        ax.scatter(
            x, y,
            s=60,
            color=COLORS.get(method),
            edgecolor="black",
            linewidth=0.5
        )

        ax.annotate(
            method,
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7
        )

ax.set_xscale("log")
ax.set_xlabel("Representation dimension (log scale)")
ax.set_ylabel("G2 AUROC")
ax.set_ylim(0.90, 1.01)
ax.set_title(
    "Compactness–performance trade-off",
    loc="left"
)
clean_axes(ax)

fig.suptitle(
    "Hierarchical taxonomic generalization of HGV-V3",
    fontsize=11
)

fig.tight_layout(rect=[0, 0, 1, 0.965])

save_fig(
    fig,
    "Figure2_hierarchical_generalization",
    MAIN
)


# ============================================================
# FIGURE 3
# ============================================================

def standard_cohort(x):
    s = str(x).lower()

    if "b1" in s or "matched" in s:
        return "B1 matched species"

    if (
        "seen" in s and
        "unseen" not in s.split("seen")[0]
    ):
        if "unseen genus" not in s:
            return "B2 unseen species / seen genus"

    if (
        "unseen genus" in s or
        "b2_unseen" in s or
        "b2-unseen" in s
    ):
        return "B2 unseen genus"

    if "seen-genus" in s or "seen genus" in s:
        return "B2 unseen species / seen genus"

    return str(x)


E11["cohort_display"] = E11["cohort"].map(
    standard_cohort
)

order_cohort = [
    "B1 matched species",
    "B2 unseen species / seen genus",
    "B2 unseen genus",
]

fig = plt.figure(figsize=(7.2, 4.8))
gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[1.1, 1],
    hspace=0.36,
    wspace=0.30
)

# A MRR / R1
ax = fig.add_subplot(gs[:, 0])
add_panel(ax, "A")

metrics = [
    x for x in ["MRR", "R1"]
    if x in E11.columns
]

xbase = np.arange(len(order_cohort))
offs = np.linspace(
    -0.10, 0.10, max(1, len(metrics))
)

for j, metric in enumerate(metrics):
    vals = []

    for c in order_cohort:
        z = E11[
            E11["cohort_display"] == c
        ]

        vals.append(
            z[metric].mean()
            if len(z)
            else np.nan
        )

    ax.scatter(
        xbase+offs[j],
        vals,
        s=55,
        label=metric,
        edgecolor="black",
        linewidth=0.4
    )

    ax.plot(
        xbase+offs[j],
        vals,
        linewidth=1.0,
        alpha=0.55
    )

ax.set_xticks(xbase)
ax.set_xticklabels(
    [
        "Matched\nspecies",
        "Unseen species\nseen genus",
        "Unseen\ngenus",
    ]
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Retrieval performance")
ax.set_title(
    "External generalization boundary",
    loc="left"
)
ax.legend(frameon=False)
clean_axes(ax)

# B Recall K
ax = fig.add_subplot(gs[0, 1])
add_panel(ax, "B")

ks = [
    k for k in [1, 5, 10, 20, 50]
    if f"R{k}" in E11.columns
]

for cohort in order_cohort:
    z = E11[
        E11["cohort_display"] == cohort
    ]

    if z.empty:
        continue

    vals = [
        z[f"R{k}"].mean()
        for k in ks
    ]

    ax.plot(
        ks,
        vals,
        marker="o",
        linewidth=1.4,
        label=cohort
    )

ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Top-K candidates")
ax.set_ylabel("Recall@K")
ax.set_title(
    "Recall as candidate budget increases",
    loc="left"
)
clean_axes(ax)

# C interpretation schematic
ax = fig.add_subplot(gs[1, 1])
add_panel(ax, "C")
ax.axis("off")

items = [
    (
        0.05, 0.68,
        "Matched species",
        "strong"
    ),
    (
        0.05, 0.40,
        "Unseen species / seen genus",
        "useful"
    ),
    (
        0.05, 0.12,
        "Unseen genus",
        "clear limitation"
    ),
]

for x, y, left, right in items:
    ax.text(
        x, y,
        left,
        va="center",
        fontsize=8
    )

    ax.annotate(
        "",
        xy=(0.68, y),
        xytext=(0.45, y),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.9
        )
    )

    ax.text(
        0.72, y,
        right,
        va="center",
        fontweight="bold",
        fontsize=8
    )

ax.set_title(
    "Observed deployment boundary",
    loc="left"
)

fig.suptitle(
    "External reference-retrieval performance",
    fontsize=11
)

save_fig(
    fig,
    "Figure3_external_generalization",
    MAIN
)


# ============================================================
# FIGURE 4
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 5.4)
)

modes = ["D0", "D1", "D2", "D3", "D4", "D5"]

for ax, metric, label in [
    (axes[0, 0], "AUROC", "A"),
    (axes[0, 1], "AUPRC", "B"),
]:
    add_panel(ax, label)

    if metric not in E6M.columns:
        ax.text(
            0.5, 0.5,
            f"{metric} unavailable",
            transform=ax.transAxes,
            ha="center"
        )
        continue

    means = []

    for i, mode in enumerate(modes):
        z = E6M[
            E6M["mode"]
            .astype(str)
            .str.upper()
            .eq(mode)
        ]

        vals = z[metric].dropna().to_numpy()

        if len(vals) == 0:
            means.append(np.nan)
            continue

        if len(vals) > 1:
            jitter = np.linspace(
                -0.05,
                0.05,
                len(vals)
            )

            ax.scatter(
                i+jitter,
                vals,
                s=18,
                alpha=0.60,
                color=MODE_COLORS[mode]
            )

        mean = np.mean(vals)
        means.append(mean)

        ax.scatter(
            i, mean,
            s=55,
            marker="D",
            color=MODE_COLORS[mode],
            edgecolor="black",
            linewidth=0.45,
            zorder=4
        )

    ax.set_xticks(range(6))
    ax.set_xticklabels(modes)
    ax.set_ylabel(metric)
    ax.set_title(
        f"D0–D5 {metric}",
        loc="left"
    )
    clean_axes(ax)

# Forest plots
for ax, metric, label in [
    (axes[1, 0], "AUROC", "C"),
    (axes[1, 1], "AUPRC", "D"),
]:
    add_panel(ax, label)

    z = E6C[
        E6C["metric"]
        .astype(str)
        .str.upper()
        .str.contains(metric)
    ].copy()

    if z.empty:
        ax.text(
            0.5, 0.5,
            f"No formal {metric} contrasts",
            transform=ax.transAxes,
            ha="center"
        )
        continue

    # Prefer D3 contrasts
    z = z[
        z["comparison"]
        .astype(str)
        .str.contains(
            "D3",
            case=False,
            regex=False
        )
    ]

    z = z.reset_index(drop=True)

    y = np.arange(len(z))

    err_left = (
        z["effect"] -
        z["ci_low"]
    ).clip(lower=0)

    err_right = (
        z["ci_high"] -
        z["effect"]
    ).clip(lower=0)

    ax.errorbar(
        z["effect"],
        y,
        xerr=np.vstack(
            [err_left, err_right]
        ),
        fmt="o",
        color=COLORS["HGV"],
        ecolor="#444444",
        capsize=2,
        linewidth=1
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        z["comparison"]
        .str.replace("D3", "D3 ", regex=False)
    )

    ax.set_xlabel(
        f"Paired Δ{metric}"
    )

    ax.set_title(
        f"D3 formal paired contrasts",
        loc="left"
    )

    clean_axes(ax)

fig.suptitle(
    "Mechanistic ablation of selective distillation",
    fontsize=11
)

fig.tight_layout(rect=[0, 0, 1, 0.965])

save_fig(
    fig,
    "Figure4_distillation_ablation",
    MAIN
)


# ============================================================
# FIGURE 5
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.2, 5.8)
)

# A taxonomy
ax = axes[0, 0]
add_panel(ax, "A")

desired = [
    "same species",
    "same genus",
    "same family",
    "different family",
]

groups = []
labels = []

for target in desired:
    z = TAX[
        TAX["stratum"]
        .str.lower()
        .str.contains(
            target,
            regex=False
        )
    ]["distance"].dropna().to_numpy()

    if len(z):
        groups.append(z)
        labels.append(target)

if all(len(g) > 3 for g in groups):
    ax.violinplot(
        groups,
        showmeans=False,
        showmedians=True
    )

    ax.boxplot(
        groups,
        widths=0.13,
        showfliers=False
    )

    ax.set_xticks(
        np.arange(1, len(groups)+1)
    )

else:
    means = [
        np.mean(g)
        for g in groups
    ]

    ax.plot(
        range(len(means)),
        means,
        marker="o",
        linewidth=1.4
    )

    ax.set_xticks(
        range(len(means))
    )

ax.set_xticklabels(
    [
        "Same\nspecies",
        "Same genus /\ndifferent species",
        "Same family /\ndifferent genus",
        "Different\nfamily",
    ][:len(groups)]
)

ax.set_ylabel("HGV embedding distance")
ax.set_title(
    "Distance increases with taxonomic separation",
    loc="left"
)

clean_axes(ax)

# B ANI
ax = axes[0, 1]
add_panel(ax, "B")

z = PAIR[
    ["HGV_distance", "ANI"]
].dropna() if "ANI" in PAIR else pd.DataFrame()

if len(z):
    ani = z["ANI"].to_numpy()

    if np.nanmax(ani) <= 1.1:
        ani = ani * 100.0

    hb = ax.hexbin(
        ani,
        z["HGV_distance"],
        gridsize=45,
        mincnt=1,
        cmap="viridis"
    )

    rho = pd.Series(
        ani
    ).corr(
        z["HGV_distance"],
        method="spearman"
    )

    ax.text(
        0.04, 0.94,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        va="top"
    )

ax.set_xlabel("ANI (%)")
ax.set_ylabel("HGV embedding distance")
ax.set_title(
    "HGV distance versus ANI",
    loc="left"
)
clean_axes(ax)

# C AF
ax = axes[1, 0]
add_panel(ax, "C")

z = PAIR[
    ["HGV_distance", "AF"]
].dropna() if "AF" in PAIR else pd.DataFrame()

if len(z):
    af = z["AF"].to_numpy()

    hb = ax.hexbin(
        af,
        z["HGV_distance"],
        gridsize=45,
        mincnt=1,
        cmap="viridis"
    )

    rho = pd.Series(
        af
    ).corr(
        z["HGV_distance"],
        method="spearman"
    )

    ax.text(
        0.04, 0.94,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        va="top"
    )

ax.set_xlabel("Alignment fraction")
ax.set_ylabel("HGV embedding distance")
ax.set_title(
    "HGV distance versus alignment fraction",
    loc="left"
)
clean_axes(ax)

# D confound adjustment
ax = axes[1, 1]
add_panel(ax, "D")

z = CONF.copy()

y = np.arange(len(z))

ax.scatter(
    z["raw_rho"],
    y+0.08,
    s=40,
    label="Raw rank association"
)

ax.scatter(
    z["partial_rho"],
    y-0.08,
    s=40,
    label="Confound-adjusted"
)

for i in range(len(z)):
    ax.plot(
        [
            z.iloc[i]["partial_rho"],
            z.iloc[i]["raw_rho"]
        ],
        [i-0.08, i+0.08],
        color="#999999",
        linewidth=0.8
    )

ax.axvline(
    0,
    color="black",
    linewidth=0.7,
    linestyle="--"
)

ax.set_yticks(y)
ax.set_yticklabels(
    z["subset"]
)

ax.set_xlabel("Rank association")
ax.set_title(
    "Association after covariate adjustment",
    loc="left"
)

ax.legend(
    frameon=False,
    loc="best"
)

clean_axes(ax)

fig.suptitle(
    "Biological fidelity and confounding of HGV distance",
    fontsize=11
)

fig.tight_layout(rect=[0, 0, 1, 0.965])

save_fig(
    fig,
    "Figure5_biological_fidelity",
    MAIN
)


# ============================================================
# FIGURE 6
# ============================================================

fig = plt.figure(figsize=(7.2, 6.0))
gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[0.9, 1.1],
    hspace=0.40,
    wspace=0.36
)

# A workflow
ax = fig.add_subplot(gs[0, :])
add_panel(ax, "A")
ax.axis("off")

steps = [
    ("Query genome", 0.02),
    ("k-mer\nextraction", 0.20),
    ("64-D HGV", 0.38),
    ("Vector\nscreen", 0.56),
    ("Top-K", 0.74),
    ("skani /\nFastANI", 0.88),
]

for txt, x in steps:
    w = 0.12

    box = FancyBboxPatch(
        (x, 0.36), w, 0.30,
        boxstyle="round,pad=0.01",
        facecolor="#F4F4F4",
        linewidth=0.8
    )

    ax.add_patch(box)

    ax.text(
        x+w/2,
        0.51,
        txt,
        ha="center",
        va="center",
        fontsize=7
    )

for a, b in zip(steps[:-1], steps[1:]):
    ax.annotate(
        "",
        xy=(b[1], 0.51),
        xytext=(a[1]+0.12, 0.51),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8
        )
    )

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1)
ax.set_title(
    "Reference-screening deployment workflow",
    loc="left"
)

# B Recall
ax = fig.add_subplot(gs[1, 0])
add_panel(ax, "B")

ks = [
    k for k in [1, 5, 10, 20, 50]
    if f"R{k}" in E11.columns
]

for cohort in order_cohort:
    z = E11[
        E11["cohort_display"] == cohort
    ]

    if z.empty:
        continue

    vals = [
        z[f"R{k}"].mean()
        for k in ks
    ]

    ax.plot(
        ks,
        vals,
        marker="o",
        linewidth=1.3
    )

ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels(
    [str(k) for k in ks]
)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Top-K")
ax.set_ylabel("Recall@K")
ax.set_title(
    "Candidate recovery",
    loc="left"
)
clean_axes(ax)

# C candidate reduction
ax = fig.add_subplot(gs[1, 1])
add_panel(ax, "C")

REF_N = EXPECTED_COUNTS["A"]

candidate_ks = np.array(
    [1, 5, 10, 20, 50]
)

reduction = (
    1 -
    candidate_ks / REF_N
) * 100

ax.plot(
    candidate_ks,
    reduction,
    marker="o",
    linewidth=1.4
)

ax.set_xscale("log")
ax.set_xticks(candidate_ks)
ax.set_xticklabels(
    candidate_ks.astype(str)
)
ax.set_xlabel("References retained")
ax.set_ylabel("Database eliminated (%)")
ax.set_ylim(95, 100.1)
ax.set_title(
    "Candidate reduction",
    loc="left"
)
clean_axes(ax)

# D runtime
ax = fig.add_subplot(gs[1, 2])
add_panel(ax, "D")

q = E12.copy()
q = q.sort_values(
    "seconds",
    ascending=True
)

ax.barh(
    np.arange(len(q)),
    q["seconds"]
)

ax.set_yticks(
    np.arange(len(q))
)

ax.set_yticklabels(
    q["stage"]
)

if (
    q["seconds"].min() > 0 and
    q["seconds"].max() /
    q["seconds"].min() > 100
):
    ax.set_xscale("log")

ax.set_xlabel("Seconds")
ax.set_title(
    "Real-FASTA runtime decomposition",
    loc="left"
)

clean_axes(ax)

fig.suptitle(
    "Reference screening and computational profile",
    fontsize=11
)

save_fig(
    fig,
    "Figure6_screening_scalability",
    MAIN
)


# ============================================================
# FIGURE S1
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(7.2, 2.7)
)

# categories
groups = [
    (
        "k-mer size",
        E7[
            E7["setting"]
            .str.lower()
            .str.contains(
                r"k.?=|k5|k6|k7",
                regex=True
            )
        ]
    ),
    (
        "Embedding dimension",
        E7[
            E7["setting"]
            .str.lower()
            .str.contains(
                r"dim|32|128",
                regex=True
            )
        ]
    ),
    (
        "Margin / λ",
        E7[
            E7["setting"]
            .str.lower()
            .str.contains(
                r"margin|lambda|λ",
                regex=True
            )
        ]
    ),
]

for ax, (title, z), label in zip(
    axes,
    groups,
    ["A", "B", "C"]
):
    add_panel(ax, label)

    if len(z):
        x = np.arange(len(z))

        ax.plot(
            x,
            z["AUROC"],
            marker="o",
            label="AUROC"
        )

        if "AUPRC" in z:
            ax.plot(
                x,
                z["AUPRC"],
                marker="s",
                label="AUPRC"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            z["setting"],
            rotation=35,
            ha="right"
        )

    ax.set_ylim(0.90, 1.00)
    ax.set_title(
        title,
        loc="left"
    )
    clean_axes(ax)

axes[0].set_ylabel("Performance")
axes[-1].legend(
    frameon=False
)

fig.suptitle(
    "Sensitivity analysis",
    fontsize=10
)

fig.tight_layout(rect=[0, 0, 1, 0.93])

save_fig(
    fig,
    "FigureS1_sensitivity",
    SUPP
)


# ============================================================
# FIGURE S2
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.2, 3.3)
)

add_panel(ax, "A")

preferred_order = [
    "Full",
    "100 kb",
    "50 kb",
    "20 kb",
    "5 kb",
    "5×5 kb",
]

def frag_rank(x):
    s = str(x).lower().replace(" ", "")

    if "full" in s:
        return 0
    if "100" in s:
        return 1
    if "50" in s and "5x5" not in s:
        return 2
    if "20" in s:
        return 3
    if "5x5" in s or "5×5" in s:
        return 5
    if "5" in s:
        return 4

    return 99


z = E8.copy()
z["_rank"] = z["condition"].map(
    frag_rank
)
z = z.sort_values("_rank")

x = np.arange(len(z))

ax.plot(
    x,
    z["AUROC"],
    marker="o",
    linewidth=1.5
)

ax.set_xticks(x)
ax.set_xticklabels(
    z["condition"]
)

ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC")
ax.set_xlabel("Available genomic context")
ax.set_title(
    "Fragmentation sensitivity of frozen full-assembly models",
    loc="left"
)

clean_axes(ax)

save_fig(
    fig,
    "FigureS2_fragmentation",
    SUPP
)


# ============================================================
# FIGURE S3
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.2, 3.4)
)

add_panel(ax, "A")

z = E9.copy()
z = z.sort_values(
    "rho",
    ascending=True
)

ax.barh(
    np.arange(len(z)),
    z["rho"]
)

ax.set_yticks(
    np.arange(len(z))
)

ax.set_yticklabels(
    z["covariate"]
)

ax.set_xlabel(
    "Spearman correlation with HGV distance"
)

ax.set_title(
    "Compositional and assembly covariate sensitivity",
    loc="left"
)

clean_axes(ax)

save_fig(
    fig,
    "FigureS3_explainability_confounding",
    SUPP
)


# ============================================================
# CONTACT SHEET
# ============================================================

import matplotlib.image as mpimg

main_pngs = [
    MAIN / "Figure1_HGV_V3_study_design.png",
    MAIN / "Figure2_hierarchical_generalization.png",
    MAIN / "Figure3_external_generalization.png",
    MAIN / "Figure4_distillation_ablation.png",
    MAIN / "Figure5_biological_fidelity.png",
    MAIN / "Figure6_screening_scalability.png",
]

fig, axes = plt.subplots(
    3, 2,
    figsize=(14, 15)
)

for ax, p in zip(
    axes.flat,
    main_pngs
):
    ax.axis("off")

    if p.exists():
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(
            p.stem,
            fontsize=9
        )
    else:
        ax.text(
            0.5, 0.5,
            f"MISSING\n{p.name}",
            ha="center",
            va="center"
        )

fig.tight_layout()

fig.savefig(
    ROOT /
    "02_figures" /
    "MAIN_FIGURES_CONTACT_SHEET.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FINAL AUDIT
# ============================================================

required = main_pngs + [
    SUPP / "FigureS1_sensitivity.png",
    SUPP / "FigureS2_fragmentation.png",
    SUPP / "FigureS3_explainability_confounding.png",
]

missing = [
    str(x)
    for x in required
    if not x.exists()
]

with open(
    AUDIT / "FINAL_FIGURE_AUDIT.txt",
    "w"
) as f:

    f.write(
        "HGV-V3 FINAL FIGURE AUDIT v3\n"
    )
    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        f"source tables discovered = "
        f"{len(TABLE_CACHE)}\n"
    )

    f.write(
        f"main figures expected = 6\n"
    )

    f.write(
        f"supplementary expected = 3\n"
    )

    f.write(
        f"missing outputs = "
        f"{len(missing)}\n"
    )

    for x in missing:
        f.write(
            f"MISSING\t{x}\n"
        )

    if not missing:
        f.write(
            "\nFINAL_FIGURE_GATE=PASS\n"
        )
    else:
        f.write(
            "\nFINAL_FIGURE_GATE=FAIL\n"
        )

print()
print("=" * 80)

if missing:
    print("FINAL_FIGURE_GATE=FAIL")
    for x in missing:
        print("MISSING:", x)
    sys.exit(2)

print("FINAL_FIGURE_GATE=PASS")
print(
    "Main figures:",
    MAIN
)
print(
    "Supplementary:",
    SUPP
)
print(
    "Normalized source data:",
    NORM
)
print(
    "Contact sheet:",
    ROOT /
    "02_figures" /
    "MAIN_FIGURES_CONTACT_SHEET.png"
)
print("=" * 80)

