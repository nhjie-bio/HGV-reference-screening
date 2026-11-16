#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("HGV_V3_revision_20260915")
OUT = (
    BASE
    / "16_scalability/"
      "E12B_synchronized_resource_v1_20260930"
)

Q = (
    OUT
    / "01_manifests/"
      "E12B_QUERIES_50.tsv"
)

E5 = (
    BASE
    / "09_baselines/"
      "E5_external_unified_v1_20260920/"
      "06_external_final_freeze_v2_20260922/"
      "E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv"
)

q = pd.read_csv(Q, sep="\t", dtype=str)
df = pd.read_csv(E5, sep="\t", dtype=str)

qset = set(q["assembly_accession"])

x = df[
    df["query_accession"].isin(qset)
].copy()

if "cohort" in x.columns:
    x = x[x["cohort"].astype(str).eq("B1")].copy()

# HGV must correspond to the exact frozen G1 seed20260915 checkpoint.
is_hgv = (
    x["method_type"].astype(str).str.upper().eq("HGV")
    if "method_type" in x.columns
    else x["method"].astype(str).str.upper().str.contains("HGV")
)

keep = (
    (~is_hgv)
    |
    (
        is_hgv
        &
        x["model_seed"].astype(str).eq("20260915")
    )
)

x = x[keep].copy()

for c in [
    "eligible",
    "query_covered",
    "pair_coverage_query",
    "first_relevant_rank",
]:
    x[c] = pd.to_numeric(
        x[c],
        errors="coerce",
    )

def canonical(name):
    s = str(name).lower()

    if "hgv" in s:
        return "HGV"

    if "raw" in s and "bray" in s:
        return "Raw6_BrayCurtis"

    if "raw" in s and "cos" in s:
        return "Raw6_Cosine"

    if "raw" in s and "euclid" in s:
        return "Raw6_Euclidean"

    if "pca" in s and "l2" in s:
        return "PCA64_L2"

    if "pca" in s and "euclid" in s:
        return "PCA64_Euclidean"

    if "dashing" in s:
        return "Dashing2"

    if "fastani" in s.replace("_", ""):
        return "FastANI"

    if "skani" in s:
        return "skani"

    if "dnabert" in s:
        return "DNABERT-S"

    if "mash" in s:
        return "Mash"

    return str(name)

x["canonical_method"] = x["method"].map(canonical)

expected = {
    "HGV",
    "Raw6_BrayCurtis",
    "Raw6_Cosine",
    "Raw6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash",
    "Dashing2",
    "FastANI",
    "skani",
    "DNABERT-S",
}

observed = set(x["canonical_method"])

missing = expected - observed

if missing:
    print(
        x[
            [
                "method_type",
                "method",
                "model_seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["method_type", "method", "model_seed"]
        )
        .to_string(index=False)
    )

    raise RuntimeError(
        f"Missing methods after canonicalization: {sorted(missing)}"
    )

rows = []

for method in sorted(expected):
    z = x[
        x["canonical_method"].eq(method)
    ].copy()

    # exactly one record per query for synchronized Q50
    nq = z["query_accession"].nunique()

    if nq != 50:
        raise RuntimeError(
            f"{method}: query count={nq}, expected 50; rows={len(z)}"
        )

    # If duplicates remain per query, fail rather than silently average.
    dup = z["query_accession"].duplicated().sum()

    if dup:
        raise RuntimeError(
            f"{method}: duplicated query rows={dup}"
        )

    covered = z["query_covered"].fillna(0).eq(1)

    success = z["first_relevant_rank"].eq(1)

    rows.append(
        {
            "method": method,
            "n_queries": 50,
            "q50_recall_at_1_all_queries":
                float(success.mean()),
            "query_hit_coverage":
                float(covered.mean()),
            "mean_pair_coverage":
                float(
                    z["pair_coverage_query"].mean()
                ),
            "top1_correct_given_hit":
                float(
                    success[covered].mean()
                )
                if covered.any()
                else np.nan,
            "n_hit_queries":
                int(covered.sum()),
            "n_top1_success":
                int(success.sum()),
        }
    )

out = pd.DataFrame(rows)

path = (
    OUT
    / "04_summary/"
      "E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv"
)

path.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(
    path,
    sep="\t",
    index=False,
)

print(out.to_string(index=False))
print()
print("WROTE", path)
