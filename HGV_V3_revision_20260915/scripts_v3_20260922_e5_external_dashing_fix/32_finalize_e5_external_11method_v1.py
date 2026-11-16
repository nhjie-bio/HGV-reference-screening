#!/usr/bin/env python3

from pathlib import Path
import hashlib
import shutil
import pandas as pd
import numpy as np


BASE = Path(
    "HGV_V3_revision_20260915"
)

ROOT = (
    BASE /
    "09_baselines/E5_external_unified_v1_20260920"
)

OLD = (
    ROOT /
    "02_native_external_full_v1/source_data"
)

REPAIR = (
    ROOT /
    "04_dashing2_external_repair_v2_20260922"
)

OUT = (
    ROOT /
    "05_external_final_freeze_v1_20260922"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

METHOD = "Dashing2_SetSketch_k31_S1024"

QUERY_IN = (
    REPAIR /
    "E5_EXTERNAL_ALL11_QUERY_LEVEL_REPAIRED.tsv"
)

OLD_SEED = (
    OLD /
    "E5_EXTERNAL_ALL11_SEED_LEVEL.tsv"
)

OLD_SUMMARY = (
    OLD /
    "E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv"
)

REPAIR_SUMMARY = (
    REPAIR /
    "DASHING2_REPAIRED_SUMMARY.tsv"
)


for p in [
    QUERY_IN,
    OLD_SEED,
    OLD_SUMMARY,
    REPAIR_SUMMARY,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


# ============================================================
# Helpers
# ============================================================

def fail(msg):
    raise RuntimeError(msg)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def norm(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("@", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_col(df, aliases, required=True):

    lookup = {
        norm(c): c
        for c in df.columns
    }

    for a in aliases:
        k = norm(a)

        if k in lookup:
            return lookup[k]

    if required:
        fail(
            f"Cannot locate column aliases={aliases}\n"
            f"Available={list(df.columns)}"
        )

    return None


def set_value(
    row,
    columns,
    aliases,
    value,
    required=True,
):
    lookup = {
        norm(c): c
        for c in columns
    }

    for a in aliases:
        k = norm(a)

        if k in lookup:
            row[
                lookup[k]
            ] = value
            return lookup[k]

    if required:
        fail(
            f"Cannot set aliases={aliases}\n"
            f"Available={list(columns)}"
        )

    return None


# ============================================================
# 1. Read repaired final query-level table
# ============================================================

query = pd.read_csv(
    QUERY_IN,
    sep="\t",
    dtype=str,
)

required_query = [
    "method_type",
    "method",
    "model_seed",
    "cohort",
    "query_accession",
    "endpoint",
    "eligible",
    "n_candidates",
    "query_covered",
    "pair_coverage_query",
    "top1_reference",
    "top1_score",
    "first_relevant_rank",
]

missing = [
    c
    for c in required_query
    if c not in query.columns
]

if missing:
    fail(
        f"Final query table missing columns: {missing}"
    )


d2q = query[
    query["method"] == METHOD
].copy()

if len(d2q) != 780:
    fail(
        f"Dashing2 query rows !=780: {len(d2q)}"
    )

if (
    d2q["query_accession"]
    .nunique()
    != 780
):
    fail(
        "Dashing2 queries not unique"
    )


expected_query_counts = {
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193,
}

observed = (
    d2q["cohort"]
    .value_counts()
    .to_dict()
)

if observed != expected_query_counts:
    fail(
        f"Query cohort counts mismatch: {observed}"
    )


# ============================================================
# 2. Recompute Dashing2 deterministic seed-level rows
# ============================================================

seed_metrics = []

for cohort, endpoint in [
    ("B1", "species"),
    ("B2_seen_genus", "genus"),
    ("B2_unseen_genus", "family"),
]:

    g = d2q[
        d2q["cohort"] == cohort
    ].copy()

    elig = g[
        pd.to_numeric(
            g["eligible"],
            errors="raise",
        ) == 1
    ].copy()

    covered = elig[
        pd.to_numeric(
            elig["query_covered"],
            errors="raise",
        ) == 1
    ].copy()

    ranks = pd.to_numeric(
        elig["first_relevant_rank"],
        errors="coerce",
    )

    ranks_cov = pd.to_numeric(
        covered["first_relevant_rank"],
        errors="coerce",
    )

    if ranks.isna().any():
        fail(
            f"{cohort}: eligible rank NA"
        )

    if ranks_cov.isna().any():
        fail(
            f"{cohort}: covered rank NA"
        )

    n_queries = len(g)
    n_eligible = len(elig)
    n_covered = len(covered)

    def rk(r, k, denominator):
        if denominator == 0:
            return np.nan

        return float(
            (r <= k).sum()
            / denominator
        )

    def mrr(r, denominator):
        if denominator == 0:
            return np.nan

        return float(
            (1.0 / r).sum()
            / denominator
        )

    seed_metrics.append(
        {
            "method_type":
                "native_external",

            "method":
                METHOD,

            "model_seed":
                "NA",

            "cohort":
                cohort,

            "endpoint":
                endpoint,

            "n_queries":
                n_queries,

            "n_references":
                2374,

            "n_eligible":
                n_eligible,

            "n_covered":
                n_covered,

            "query_coverage":
                float(
                    pd.to_numeric(
                        g["query_covered"],
                        errors="raise",
                    ).mean()
                ),

            "pair_coverage_mean":
                float(
                    pd.to_numeric(
                        g[
                            "pair_coverage_query"
                        ],
                        errors="raise",
                    ).mean()
                ),

            "R1":
                rk(
                    ranks,
                    1,
                    n_eligible,
                ),

            "R5":
                rk(
                    ranks,
                    5,
                    n_eligible,
                ),

            "R10":
                rk(
                    ranks,
                    10,
                    n_eligible,
                ),

            "MRR":
                mrr(
                    ranks,
                    n_eligible,
                ),

            "R1_covered":
                rk(
                    ranks_cov,
                    1,
                    n_covered,
                ),

            "R5_covered":
                rk(
                    ranks_cov,
                    5,
                    n_covered,
                ),

            "R10_covered":
                rk(
                    ranks_cov,
                    10,
                    n_covered,
                ),

            "MRR_covered":
                mrr(
                    ranks_cov,
                    n_covered,
                ),
        }
    )


calc = pd.DataFrame(
    seed_metrics
)


# Eligibility hard gates.
elig_expected = {
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 143,
}

for _, r in calc.iterrows():

    cohort = r["cohort"]

    if int(
        r["n_eligible"]
    ) != elig_expected[cohort]:

        fail(
            f"{cohort}: eligible mismatch "
            f"{r['n_eligible']} "
            f"!= {elig_expected[cohort]}"
        )


# ============================================================
# 3. Replace ONLY Dashing2 in existing seed-level table
# ============================================================

old_seed = pd.read_csv(
    OLD_SEED,
    sep="\t",
)

print(
    "SEED LEVEL COLUMNS:",
    list(old_seed.columns),
)

method_col = find_col(
    old_seed,
    ["method"],
)

cohort_col = find_col(
    old_seed,
    ["cohort"],
)

endpoint_col = find_col(
    old_seed,
    ["endpoint"],
)

seed_mask = (
    old_seed[method_col]
    .astype(str)
    == METHOD
)

if int(seed_mask.sum()) != 3:
    fail(
        f"Expected 3 old Dashing2 seed rows, "
        f"found {int(seed_mask.sum())}"
    )


new_seed_rows = []

for _, metric in calc.iterrows():

    hit = old_seed[
        seed_mask
        &
        (
            old_seed[cohort_col]
            .astype(str)
            == str(metric["cohort"])
        )
    ]

    if len(hit) != 1:
        fail(
            f"Seed template rows for "
            f"{metric['cohort']} = {len(hit)}"
        )

    row = hit.iloc[0].copy()

    cols = old_seed.columns

    # Identity / counts.
    set_value(
        row, cols,
        ["method_type"],
        "native_external",
        required=False,
    )

    set_value(
        row, cols,
        ["method"],
        METHOD,
    )

    set_value(
        row, cols,
        ["model_seed"],
        "NA",
        required=False,
    )

    set_value(
        row, cols,
        ["cohort"],
        metric["cohort"],
    )

    set_value(
        row, cols,
        ["endpoint"],
        metric["endpoint"],
    )

    set_value(
        row, cols,
        ["n_queries"],
        metric["n_queries"],
    )

    set_value(
        row, cols,
        ["n_references"],
        metric["n_references"],
    )

    set_value(
        row, cols,
        ["n_eligible"],
        metric["n_eligible"],
    )

    set_value(
        row, cols,
        [
            "n_covered",
            "n_queries_covered",
        ],
        metric["n_covered"],
    )

    # Coverage.
    set_value(
        row, cols,
        ["query_coverage"],
        metric["query_coverage"],
    )

    set_value(
        row, cols,
        [
            "pair_coverage_mean",
            "pair_coverage",
        ],
        metric["pair_coverage_mean"],
    )

    # All-query eligible retrieval.
    set_value(
        row, cols,
        ["R1", "R@1"],
        metric["R1"],
    )

    set_value(
        row, cols,
        ["R5", "R@5"],
        metric["R5"],
    )

    set_value(
        row, cols,
        ["R10", "R@10"],
        metric["R10"],
    )

    set_value(
        row, cols,
        ["MRR"],
        metric["MRR"],
    )

    # Covered-only retrieval.
    set_value(
        row, cols,
        [
            "R1_covered",
            "R1_covered_only",
            "R@1_covered",
        ],
        metric["R1_covered"],
    )

    set_value(
        row, cols,
        [
            "R5_covered",
            "R5_covered_only",
            "R@5_covered",
        ],
        metric["R5_covered"],
    )

    set_value(
        row, cols,
        [
            "R10_covered",
            "R10_covered_only",
            "R@10_covered",
        ],
        metric["R10_covered"],
    )

    set_value(
        row, cols,
        [
            "MRR_covered",
            "MRR_covered_only",
        ],
        metric["MRR_covered"],
    )

    new_seed_rows.append(
        row
    )


seed_final = pd.concat(
    [
        old_seed[
            ~seed_mask
        ],
        pd.DataFrame(
            new_seed_rows
        ),
    ],
    ignore_index=True,
)


if len(seed_final) != len(old_seed):
    fail(
        "Seed-level row count changed"
    )


# ============================================================
# 4. Replace ONLY Dashing2 in method-summary table
# ============================================================

old_summary = pd.read_csv(
    OLD_SUMMARY,
    sep="\t",
)

print(
    "METHOD SUMMARY COLUMNS:",
    list(old_summary.columns),
)

method_col_s = find_col(
    old_summary,
    ["method"],
)

cohort_col_s = find_col(
    old_summary,
    ["cohort"],
)

summary_mask = (
    old_summary[
        method_col_s
    ].astype(str)
    == METHOD
)

if int(summary_mask.sum()) != 3:
    fail(
        f"Expected 3 old Dashing2 summary rows, "
        f"found {int(summary_mask.sum())}"
    )


new_summary_rows = []

for _, metric in calc.iterrows():

    hit = old_summary[
        summary_mask
        &
        (
            old_summary[
                cohort_col_s
            ].astype(str)
            == str(
                metric["cohort"]
            )
        )
    ]

    if len(hit) != 1:
        fail(
            f"Summary template rows for "
            f"{metric['cohort']} = {len(hit)}"
        )

    row = hit.iloc[0].copy()

    cols = old_summary.columns

    set_value(
        row, cols,
        ["method_type"],
        "native_external",
        required=False,
    )

    set_value(
        row, cols,
        ["method"],
        METHOD,
    )

    set_value(
        row, cols,
        ["cohort"],
        metric["cohort"],
    )

    set_value(
        row, cols,
        ["endpoint"],
        metric["endpoint"],
    )

    set_value(
        row, cols,
        [
            "n_model_seeds",
            "n_seeds",
        ],
        1,
    )

    set_value(
        row, cols,
        ["n_queries"],
        metric["n_queries"],
    )

    set_value(
        row, cols,
        ["n_references"],
        metric["n_references"],
    )

    set_value(
        row, cols,
        ["n_eligible"],
        metric["n_eligible"],
    )

    # mean/sd pairs.
    metric_aliases = {
        "query_coverage": [
            "query_coverage"
        ],

        "pair_coverage_mean": [
            "pair_coverage",
            "pair_coverage_mean",
        ],

        "R1": [
            "R1",
            "R@1",
        ],

        "R5": [
            "R5",
            "R@5",
        ],

        "R10": [
            "R10",
            "R@10",
        ],

        "MRR": [
            "MRR",
        ],

        "R1_covered": [
            "R1_covered",
            "R1_covered_only",
            "R@1_covered",
        ],

        "R5_covered": [
            "R5_covered",
            "R5_covered_only",
            "R@5_covered",
        ],

        "R10_covered": [
            "R10_covered",
            "R10_covered_only",
            "R@10_covered",
        ],

        "MRR_covered": [
            "MRR_covered",
            "MRR_covered_only",
        ],
    }


    for key, aliases in metric_aliases.items():

        value = metric[key]

        mean_aliases = []

        sd_aliases = []

        for a in aliases:

            mean_aliases.extend(
                [
                    f"{a}_mean",
                    f"mean_{a}",
                ]
            )

            sd_aliases.extend(
                [
                    f"{a}_sd",
                    f"sd_{a}",
                ]
            )

        # Existing summary tables should contain
        # explicit mean/sd names. If not, fail rather
        # than silently invent a schema.
        set_value(
            row,
            cols,
            mean_aliases,
            value,
        )

        set_value(
            row,
            cols,
            sd_aliases,
            0.0,
        )

    new_summary_rows.append(
        row
    )


summary_final = pd.concat(
    [
        old_summary[
            ~summary_mask
        ],
        pd.DataFrame(
            new_summary_rows
        ),
    ],
    ignore_index=True,
)


if len(summary_final) != len(
    old_summary
):
    fail(
        "Method-summary row count changed"
    )


# ============================================================
# 5. Verify that ONLY Dashing2 changed
# ============================================================

old_seed_non = (
    old_seed[
        ~seed_mask
    ]
    .reset_index(drop=True)
)

new_seed_non = (
    seed_final[
        seed_final["method"]
        != METHOD
    ]
    .reset_index(drop=True)
)

if not old_seed_non.equals(
    new_seed_non
):
    fail(
        "Non-Dashing2 seed-level rows changed"
    )


old_summary_non = (
    old_summary[
        ~summary_mask
    ]
    .reset_index(drop=True)
)

new_summary_non = (
    summary_final[
        summary_final["method"]
        != METHOD
    ]
    .reset_index(drop=True)
)

if not old_summary_non.equals(
    new_summary_non
):
    fail(
        "Non-Dashing2 method-summary rows changed"
    )


# ============================================================
# 6. Write final frozen E5 external source data
# ============================================================

QUERY_OUT = (
    OUT /
    "E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv"
)

SEED_OUT = (
    OUT /
    "E5_EXTERNAL_ALL11_SEED_LEVEL_FINAL.tsv"
)

SUMMARY_OUT = (
    OUT /
    "E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv"
)

D2_OUT = (
    OUT /
    "E5_EXTERNAL_DASHING2_FINAL_METRICS.tsv"
)


query.to_csv(
    QUERY_OUT,
    sep="\t",
    index=False,
)

seed_final.to_csv(
    SEED_OUT,
    sep="\t",
    index=False,
)

summary_final.to_csv(
    SUMMARY_OUT,
    sep="\t",
    index=False,
)

calc.to_csv(
    D2_OUT,
    sep="\t",
    index=False,
)


# ============================================================
# 7. Freeze provenance
# ============================================================

artifacts = [
    QUERY_IN,
    OLD_SEED,
    OLD_SUMMARY,
    REPAIR_SUMMARY,
    QUERY_OUT,
    SEED_OUT,
    SUMMARY_OUT,
    D2_OUT,
]

sha_rows = []

for p in artifacts:

    sha_rows.append(
        {
            "path":
                str(p),
            "sha256":
                sha256_file(p),
        }
    )


SHA_OUT = (
    OUT /
    "E5_EXTERNAL_FINAL_SHA256.tsv"
)

pd.DataFrame(
    sha_rows
).to_csv(
    SHA_OUT,
    sep="\t",
    index=False,
)


# ============================================================
# 8. Freeze report
# ============================================================

REPORT = (
    OUT /
    "E5_EXTERNAL_FINAL_FREEZE.txt"
)

with REPORT.open(
    "w"
) as f:

    f.write(
        "HGV-V3 E5 EXTERNAL "
        "11-METHOD FINAL FREEZE v1\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Freeze date: 2026-09-22\n\n"
    )

    f.write(
        "Status:\n"
        "  E5 external = COMPLETE\n"
        "  E5 external = FROZEN\n\n"
    )

    f.write(
        "Repair applied:\n"
        "  Dashing2 external rectangular "
        "matrix orientation corrected.\n"
    )

    f.write(
        "  Physical Dashing2 matrix: "
        "2374 reference x 780 query.\n"
    )

    f.write(
        "  Retrieval matrix used: "
        "780 query x 2374 reference.\n\n"
    )

    f.write(
        "No Dashing2 similarity recomputation "
        "was performed.\n"
    )

    f.write(
        "No other baseline was recomputed.\n"
    )

    f.write(
        "All non-Dashing2 query/seed/summary "
        "results were preserved.\n\n"
    )

    f.write(
        "Corrected Dashing2 external metrics:\n\n"
    )

    f.write(
        calc.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "Invalid historical Dashing2 retrieval "
        "results produced from the wrong "
        "rectangular orientation are retained "
        "for provenance only and must not be "
        "used in manuscript tables or claims.\n\n"
    )

    f.write(
        "E5_EXTERNAL_FINAL_FREEZE=PASS\n"
    )


print()
print("=" * 105)
print("E5 EXTERNAL 11-METHOD FINALIZATION")
print("=" * 105)

print()
print("Dashing2 final metrics:")
print(
    calc[
        [
            "cohort",
            "endpoint",
            "n_queries",
            "n_eligible",
            "R1",
            "R5",
            "R10",
            "MRR",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Query rows:",
    len(query),
)

print(
    "Seed-level rows:",
    len(seed_final),
)

print(
    "Method-summary rows:",
    len(summary_final),
)

print()
print(
    "Output:",
    OUT,
)

print()
print(
    "E5_EXTERNAL_FINAL_FREEZE=PASS"
)
