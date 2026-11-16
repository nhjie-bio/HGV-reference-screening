#!/usr/bin/env python3

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

CORE = (
    BASE /
    "08_main_generalization/"
    "blind_test_evaluation_v1_20260917"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

CORE_SPLIT = (
    CORE /
    "test_metrics_split_level.tsv"
)

CORE_PRIMARY = (
    CORE /
    "PRIMARY_ALLPAIR_TEST_SUMMARY.tsv"
)

NATIVE_SPLIT = (
    E5 /
    "07_unified_internal/"
    "native_blind_test_evaluation_v1_20260917/"
    "native_test_metrics_split_level.tsv"
)

OUT = (
    E5 /
    "07_unified_internal/"
    "core_native_merge_v1_20260917"
)


if OUT.exists():
    raise RuntimeError(
        f"Output already exists:\n{OUT}\n"
        "Do not overwrite. "
        "Use a new version directory."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


METHOD_ORDER = [
    "HGV_Dashing2_tau0p98",

    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",

    "PCA64_Euclidean",
    "PCA64_L2",

    "Mash",
    "Dashing2_SetSketch",
    "FastANI",
    "skani",
]

PROTOCOL_ORDER = [
    "G2",
    "G3",
    "G4",
]

EXPECTED_TEST = {
    ("G2", "20260915"): 3872,
    ("G2", "20260916"): 3874,
    ("G2", "20260917"): 3874,

    ("G3", "20260915"): 3916,
    ("G3", "20260916"): 3852,
    ("G3", "20260917"): 3894,

    ("G4", "20260915"): 3832,
    ("G4", "20260916"): 3894,
    ("G4", "20260917"): 3874,
}


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def safe_precision(
    tp,
    fp,
):

    denom = tp + fp

    if denom == 0:
        return 0.0

    return tp / denom


print("=" * 104)
print(
    "E5 CORE + NATIVE INTERNAL "
    "BLIND-TEST MERGE v1"
)
print("=" * 104)


# ================================================================
# 1. Load frozen Core split-level results.
# ================================================================

print()
print(
    "===== LOAD CORE SPLIT-LEVEL RESULTS ====="
)

core = pd.read_csv(
    CORE_SPLIT,
    sep="\t",
    dtype={
        "split_seed": str,
    },
)

print(
    "raw core shape =",
    core.shape
)

required_core = [
    "method_type",
    "method",
    "protocol",
    "split_seed",
    "subset",
    "n_model_seeds",
    "n_pairs",

    "AUROC_mean",
    "AUPRC_mean",
    "MCC_mean",
    "F1_mean",
    "BalancedAccuracy_mean",
    "Precision_mean",
    "Recall_mean",
    "Specificity_mean",
]

for c in required_core:

    if c not in core.columns:

        raise RuntimeError(
            f"Core split file missing: {c}"
        )


core = (
    core[
        core["subset"] == "all"
    ]
    .copy()
)


print(
    "core subset=all rows =",
    len(core)
)


if len(core) != 54:

    raise RuntimeError(
        f"Expected 54 Core all-pair "
        f"split-level rows, "
        f"found {len(core)}"
    )


if set(
    core["method"].unique()
) != set(
    METHOD_ORDER[:6]
):

    raise RuntimeError(
        "Unexpected Core method set"
    )


if set(
    core["protocol"].unique()
) != set(
    PROTOCOL_ORDER
):

    raise RuntimeError(
        "Unexpected Core protocol set"
    )


if set(
    core["split_seed"].astype(str).unique()
) != {
    "20260915",
    "20260916",
    "20260917",
}:

    raise RuntimeError(
        "Unexpected Core split seeds"
    )


if core[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate Core "
        "method/protocol/split_seed"
    )


if not (
    pd.to_numeric(
        core["n_model_seeds"],
        errors="raise",
    )
    == 3
).all():

    raise RuntimeError(
        "Core split-level results "
        "do not all average 3 model seeds"
    )


for r in core.itertuples(
    index=False
):

    key = (
        str(r.protocol),
        str(r.split_seed),
    )

    expected = EXPECTED_TEST[
        key
    ]

    if int(r.n_pairs) != expected:

        raise RuntimeError(
            f"Core {r.method} "
            f"{r.protocol} "
            f"{r.split_seed}: "
            f"n_pairs={r.n_pairs}, "
            f"expected={expected}"
        )


# ================================================================
# 2. Normalize Core into unified schema.
# ================================================================

core_norm = pd.DataFrame({
    "method_type":
        core["method_type"].astype(str),

    "method":
        core["method"].astype(str),

    "protocol":
        core["protocol"].astype(str),

    "split_seed":
        core["split_seed"].astype(str),

    "source_family":
        "frozen_core",

    "replicate_structure":
        "3_model_seeds_mean_within_split",

    "n_model_seeds":
        pd.to_numeric(
            core["n_model_seeds"]
        ).astype(int),

    "n_pairs_total":
        pd.to_numeric(
            core["n_pairs"]
        ).astype(int),

    "n_pairs_covered":
        pd.to_numeric(
            core["n_pairs"]
        ).astype(int),

    "n_pairs_missing":
        0,

    "coverage":
        1.0,

    "positive_coverage":
        1.0,

    "negative_coverage":
        1.0,

    "AUROC":
        pd.to_numeric(
            core["AUROC_mean"]
        ),

    "AUPRC":
        pd.to_numeric(
            core["AUPRC_mean"]
        ),

    "MCC":
        pd.to_numeric(
            core["MCC_mean"]
        ),

    "F1":
        pd.to_numeric(
            core["F1_mean"]
        ),

    "BalancedAccuracy":
        pd.to_numeric(
            core[
                "BalancedAccuracy_mean"
            ]
        ),

    "Precision":
        pd.to_numeric(
            core["Precision_mean"]
        ),

    "Recall":
        pd.to_numeric(
            core["Recall_mean"]
        ),

    "Specificity":
        pd.to_numeric(
            core["Specificity_mean"]
        ),

    "coverage_type":
        "complete",

    "metric_scope":
        "all_test_pairs",
})


# Preserve within-split model-seed SD
# as provenance/secondary uncertainty.
for src_col, dst_col in [
    (
        "AUROC_sd_model_seed",
        "within_split_AUROC_sd_model_seed",
    ),
    (
        "AUPRC_sd_model_seed",
        "within_split_AUPRC_sd_model_seed",
    ),
    (
        "MCC_sd_model_seed",
        "within_split_MCC_sd_model_seed",
    ),
    (
        "F1_sd_model_seed",
        "within_split_F1_sd_model_seed",
    ),
    (
        "BalancedAccuracy_sd_model_seed",
        "within_split_BalancedAccuracy_sd_model_seed",
    ),
]:

    if src_col in core.columns:

        core_norm[
            dst_col
        ] = pd.to_numeric(
            core[src_col]
        )

    else:

        core_norm[
            dst_col
        ] = np.nan


# ================================================================
# 3. Load native split-level blind-test results.
# ================================================================

print()
print(
    "===== LOAD NATIVE SPLIT-LEVEL RESULTS ====="
)

native = pd.read_csv(
    NATIVE_SPLIT,
    sep="\t",
    dtype={
        "split_seed": str,
    },
)

print(
    "native shape =",
    native.shape
)


required_native = [
    "method",
    "protocol",
    "split_seed",

    "coverage_type",
    "metric_scope",

    "n_total",
    "n_covered",
    "n_missing",

    "coverage",
    "positive_coverage",
    "negative_coverage",

    "test_auroc_covered",
    "test_auprc_covered",
    "test_mcc_covered",
    "test_f1_covered",
    "test_balanced_accuracy_covered",
    "test_sensitivity_covered",
    "test_specificity_covered",

    "tp",
    "fp",
    "tn",
    "fn",
]

for c in required_native:

    if c not in native.columns:

        raise RuntimeError(
            f"Native split file missing: {c}"
        )


if len(native) != 36:

    raise RuntimeError(
        f"Expected 36 native rows, "
        f"found {len(native)}"
    )


if set(
    native["method"].unique()
) != set(
    METHOD_ORDER[6:]
):

    raise RuntimeError(
        "Unexpected native method set"
    )


if native[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate native "
        "method/protocol/split_seed"
    )


for r in native.itertuples(
    index=False
):

    key = (
        str(r.protocol),
        str(r.split_seed),
    )

    expected = EXPECTED_TEST[
        key
    ]

    if int(r.n_total) != expected:

        raise RuntimeError(
            f"Native {r.method} "
            f"{r.protocol} "
            f"{r.split_seed}: "
            f"n_total={r.n_total}, "
            f"expected={expected}"
        )


# ================================================================
# 4. Normalize native results.
# ================================================================

precision_values = []

for r in native.itertuples(
    index=False
):

    precision_values.append(
        safe_precision(
            int(r.tp),
            int(r.fp),
        )
    )


native_norm = pd.DataFrame({
    "method_type":
        "native_baseline",

    "method":
        native["method"].astype(str),

    "protocol":
        native["protocol"].astype(str),

    "split_seed":
        native["split_seed"].astype(str),

    "source_family":
        "e5_native",

    "replicate_structure":
        "single_native_method_per_split",

    "n_model_seeds":
        np.nan,

    "n_pairs_total":
        pd.to_numeric(
            native["n_total"]
        ).astype(int),

    "n_pairs_covered":
        pd.to_numeric(
            native["n_covered"]
        ).astype(int),

    "n_pairs_missing":
        pd.to_numeric(
            native["n_missing"]
        ).astype(int),

    "coverage":
        pd.to_numeric(
            native["coverage"]
        ),

    "positive_coverage":
        pd.to_numeric(
            native[
                "positive_coverage"
            ]
        ),

    "negative_coverage":
        pd.to_numeric(
            native[
                "negative_coverage"
            ]
        ),

    "AUROC":
        pd.to_numeric(
            native[
                "test_auroc_covered"
            ]
        ),

    "AUPRC":
        pd.to_numeric(
            native[
                "test_auprc_covered"
            ]
        ),

    "MCC":
        pd.to_numeric(
            native[
                "test_mcc_covered"
            ]
        ),

    "F1":
        pd.to_numeric(
            native[
                "test_f1_covered"
            ]
        ),

    "BalancedAccuracy":
        pd.to_numeric(
            native[
                "test_balanced_accuracy_covered"
            ]
        ),

    "Precision":
        precision_values,

    "Recall":
        pd.to_numeric(
            native[
                "test_sensitivity_covered"
            ]
        ),

    "Specificity":
        pd.to_numeric(
            native[
                "test_specificity_covered"
            ]
        ),

    "coverage_type":
        native[
            "coverage_type"
        ].astype(str),

    "metric_scope":
        native[
            "metric_scope"
        ].astype(str),
})


for c in [
    "within_split_AUROC_sd_model_seed",
    "within_split_AUPRC_sd_model_seed",
    "within_split_MCC_sd_model_seed",
    "within_split_F1_sd_model_seed",
    "within_split_BalancedAccuracy_sd_model_seed",
]:

    native_norm[c] = np.nan


# ================================================================
# 5. Merge.
# ================================================================

combined = pd.concat(
    [
        core_norm,
        native_norm,
    ],
    ignore_index=True,
)


if len(combined) != 90:

    raise RuntimeError(
        f"Expected 90 unified "
        f"split-level rows, "
        f"found {len(combined)}"
    )


if combined[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate unified "
        "method/protocol/split_seed"
    )


if set(
    combined["method"].unique()
) != set(
    METHOD_ORDER
):

    raise RuntimeError(
        "Unified method set mismatch"
    )


# Stable publication-oriented ordering.

combined["_method_order"] = (
    combined["method"]
    .map({
        m: i
        for i, m
        in enumerate(
            METHOD_ORDER
        )
    })
)

combined["_protocol_order"] = (
    combined["protocol"]
    .map({
        p: i
        for i, p
        in enumerate(
            PROTOCOL_ORDER
        )
    })
)


combined = (
    combined.sort_values(
        [
            "_method_order",
            "_protocol_order",
            "split_seed",
        ]
    )
    .drop(
        columns=[
            "_method_order",
            "_protocol_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ================================================================
# 6. Protocol-level mean +/- SD.
#
# Replicate unit = taxonomy split seed.
# HGV model seeds have already been averaged within split.
# ================================================================

METRICS = [
    "coverage",
    "positive_coverage",
    "negative_coverage",

    "AUROC",
    "AUPRC",
    "MCC",
    "F1",
    "BalancedAccuracy",
    "Precision",
    "Recall",
    "Specificity",
]


summary_rows = []


for (
    method,
    protocol
), g in combined.groupby(
    [
        "method",
        "protocol",
    ],
    sort=False,
):

    if len(g) != 3:

        raise RuntimeError(
            f"{method} {protocol}: "
            f"expected 3 split seeds, "
            f"found {len(g)}"
        )


    row = {
        "method":
            method,

        "method_type":
            g[
                "method_type"
            ].iloc[0],

        "protocol":
            protocol,

        "n_split_seeds":
            3,

        "replicate_unit":
            "taxonomy_split_seed",

        "coverage_type":
            g[
                "coverage_type"
            ].iloc[0],

        "metric_scope":
            g[
                "metric_scope"
            ].iloc[0],

        "n_pairs_total_sum":
            int(
                g[
                    "n_pairs_total"
                ].sum()
            ),

        "n_pairs_covered_sum":
            int(
                g[
                    "n_pairs_covered"
                ].sum()
            ),

        "n_pairs_missing_sum":
            int(
                g[
                    "n_pairs_missing"
                ].sum()
            ),
    }


    for metric in METRICS:

        values = pd.to_numeric(
            g[metric],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        row[
            metric + "_mean"
        ] = float(
            np.mean(values)
        )

        row[
            metric + "_sd_split_seed"
        ] = float(
            np.std(
                values,
                ddof=1,
            )
        )


    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


if len(summary) != 30:

    raise RuntimeError(
        f"Expected 30 protocol summary rows, "
        f"found {len(summary)}"
    )


summary["_method_order"] = (
    summary["method"]
    .map({
        m: i
        for i, m
        in enumerate(
            METHOD_ORDER
        )
    })
)

summary["_protocol_order"] = (
    summary["protocol"]
    .map({
        p: i
        for i, p
        in enumerate(
            PROTOCOL_ORDER
        )
    })
)

summary = (
    summary.sort_values(
        [
            "_method_order",
            "_protocol_order",
        ]
    )
    .drop(
        columns=[
            "_method_order",
            "_protocol_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ================================================================
# 7. Critical audit:
# Recomputed Core summary MUST reproduce frozen
# PRIMARY_ALLPAIR_TEST_SUMMARY.tsv.
# ================================================================

print()
print(
    "===== REPRODUCE FROZEN CORE PRIMARY SUMMARY ====="
)


primary = pd.read_csv(
    CORE_PRIMARY,
    sep="\t",
)

primary = (
    primary[
        primary["subset"] == "all"
    ]
    .copy()
)


if len(primary) != 18:

    raise RuntimeError(
        f"Expected 18 frozen Core "
        f"primary rows, found "
        f"{len(primary)}"
    )


core_summary = summary[
    summary["method"].isin(
        METHOD_ORDER[:6]
    )
].copy()


if len(core_summary) != 18:

    raise RuntimeError(
        "Unified Core summary "
        "does not contain 18 rows"
    )


COMPARE = [
    (
        "AUROC_mean",
        "AUROC_mean",
    ),
    (
        "AUROC_sd_split_seed",
        "AUROC_sd_split_seed",
    ),

    (
        "AUPRC_mean",
        "AUPRC_mean",
    ),
    (
        "AUPRC_sd_split_seed",
        "AUPRC_sd_split_seed",
    ),

    (
        "MCC_mean",
        "MCC_mean",
    ),
    (
        "MCC_sd_split_seed",
        "MCC_sd_split_seed",
    ),

    (
        "F1_mean",
        "F1_mean",
    ),
    (
        "F1_sd_split_seed",
        "F1_sd_split_seed",
    ),

    (
        "BalancedAccuracy_mean",
        "BalancedAccuracy_mean",
    ),
    (
        "BalancedAccuracy_sd_split_seed",
        "BalancedAccuracy_sd_split_seed",
    ),

    (
        "Precision_mean",
        "Precision_mean",
    ),
    (
        "Precision_sd_split_seed",
        "Precision_sd_split_seed",
    ),

    (
        "Recall_mean",
        "Recall_mean",
    ),
    (
        "Recall_sd_split_seed",
        "Recall_sd_split_seed",
    ),

    (
        "Specificity_mean",
        "Specificity_mean",
    ),
    (
        "Specificity_sd_split_seed",
        "Specificity_sd_split_seed",
    ),
]


mismatch_rows = []


for p in primary.itertuples(
    index=False
):

    method = str(
        p.method
    )

    protocol = str(
        p.protocol
    )


    q = core_summary[
        (
            core_summary[
                "method"
            ] == method
        )
        &
        (
            core_summary[
                "protocol"
            ] == protocol
        )
    ]


    if len(q) != 1:

        raise RuntimeError(
            f"Cannot match frozen Core "
            f"summary: "
            f"{method} {protocol}"
        )


    q = q.iloc[0]


    for frozen_col, new_col in COMPARE:

        frozen_val = float(
            getattr(
                p,
                frozen_col
            )
        )

        new_val = float(
            q[
                new_col
            ]
        )


        if not np.isclose(
            frozen_val,
            new_val,
            rtol=1e-10,
            atol=1e-12,
        ):

            mismatch_rows.append({
                "method":
                    method,

                "protocol":
                    protocol,

                "metric":
                    frozen_col,

                "frozen":
                    frozen_val,

                "recomputed":
                    new_val,

                "abs_difference":
                    abs(
                        frozen_val -
                        new_val
                    ),
            })


if mismatch_rows:

    mismatch = pd.DataFrame(
        mismatch_rows
    )

    mismatch.to_csv(
        OUT /
        "CORE_REPRODUCTION_MISMATCH.tsv",
        sep="\t",
        index=False,
    )

    raise RuntimeError(
        f"Core frozen summary reproduction "
        f"FAILED with "
        f"{len(mismatch_rows)} mismatches"
    )


print(
    "Core frozen summary reproduction:"
)
print(
    "18/18 rows, all metrics PASS"
)


# ================================================================
# 8. Long table.
# ================================================================

long_rows = []


for r in summary.itertuples(
    index=False
):

    d = r._asdict()

    for metric in METRICS:

        long_rows.append({
            "method":
                d["method"],

            "method_type":
                d["method_type"],

            "protocol":
                d["protocol"],

            "metric":
                metric,

            "mean":
                d[
                    metric + "_mean"
                ],

            "sd_split_seed":
                d[
                    metric +
                    "_sd_split_seed"
                ],

            "n_split_seeds":
                3,

            "coverage_type":
                d[
                    "coverage_type"
                ],

            "metric_scope":
                d[
                    "metric_scope"
                ],
        })


long_df = pd.DataFrame(
    long_rows
)


# ================================================================
# 9. Compact primary comparison table.
# ================================================================

primary_cols = [
    "method",
    "method_type",
    "protocol",

    "coverage_mean",
    "coverage_sd_split_seed",

    "positive_coverage_mean",
    "negative_coverage_mean",

    "AUROC_mean",
    "AUROC_sd_split_seed",

    "AUPRC_mean",
    "AUPRC_sd_split_seed",

    "MCC_mean",
    "MCC_sd_split_seed",

    "F1_mean",
    "F1_sd_split_seed",

    "BalancedAccuracy_mean",
    "BalancedAccuracy_sd_split_seed",

    "metric_scope",
    "coverage_type",
]


primary_table = (
    summary[
        primary_cols
    ]
    .copy()
)


# ================================================================
# 10. Global gates.
# ================================================================

complete_methods = [
    "HGV_Dashing2_tau0p98",
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash",
    "Dashing2_SetSketch",
]


complete_rows = combined[
    combined["method"].isin(
        complete_methods
    )
]


if not np.allclose(
    complete_rows["coverage"],
    1.0,
):

    raise RuntimeError(
        "A complete-coverage method "
        "has coverage != 1"
    )


ani_rows = combined[
    combined["method"].isin(
        [
            "FastANI",
            "skani",
        ]
    )
]


if not np.allclose(
    ani_rows[
        "positive_coverage"
    ],
    1.0,
):

    raise RuntimeError(
        "FastANI/skani positive "
        "coverage is not uniformly 1"
    )


# ================================================================
# 11. Write outputs.
# ================================================================

split_file = (
    OUT /
    "E5_CORE_NATIVE_SPLIT_LEVEL.tsv"
)

summary_file = (
    OUT /
    "E5_CORE_NATIVE_PROTOCOL_SUMMARY.tsv"
)

long_file = (
    OUT /
    "E5_CORE_NATIVE_LONG.tsv"
)

primary_file = (
    OUT /
    "E5_CORE_NATIVE_PRIMARY_TABLE.tsv"
)


combined.to_csv(
    split_file,
    sep="\t",
    index=False,
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False,
)

long_df.to_csv(
    long_file,
    sep="\t",
    index=False,
)

primary_table.to_csv(
    primary_file,
    sep="\t",
    index=False,
)


# ================================================================
# 12. Input provenance.
# ================================================================

input_sha = pd.DataFrame([
    {
        "input":
            "CORE_SPLIT",

        "file":
            str(CORE_SPLIT),

        "sha256":
            sha256_file(
                CORE_SPLIT
            ),
    },
    {
        "input":
            "CORE_PRIMARY",

        "file":
            str(CORE_PRIMARY),

        "sha256":
            sha256_file(
                CORE_PRIMARY
            ),
    },
    {
        "input":
            "NATIVE_SPLIT",

        "file":
            str(NATIVE_SPLIT),

        "sha256":
            sha256_file(
                NATIVE_SPLIT
            ),
    },
])


input_sha_file = (
    OUT /
    "INPUT_SHA256.tsv"
)

input_sha.to_csv(
    input_sha_file,
    sep="\t",
    index=False,
)


# ================================================================
# 13. Report.
# ================================================================

report = (
    OUT /
    "E5_CORE_NATIVE_MERGE_REPORT.txt"
)


display_cols = [
    "method",
    "protocol",

    "coverage_mean",
    "positive_coverage_mean",
    "negative_coverage_mean",

    "AUROC_mean",
    "AUROC_sd_split_seed",

    "AUPRC_mean",
    "AUPRC_sd_split_seed",

    "MCC_mean",
    "MCC_sd_split_seed",

    "F1_mean",
    "BalancedAccuracy_mean",

    "metric_scope",
]


with open(
    report,
    "w",
) as f:

    f.write(
        "E5 Core + Native Internal "
        "Blind-Test Merge v1\n"
    )

    f.write(
        "=" * 84 +
        "\n\n"
    )

    f.write(
        "Statistical replicate unit:\n"
        "  taxonomy split seed\n\n"
    )

    f.write(
        "For HGV/Core results, "
        "3 model seeds were averaged "
        "within each split before "
        "cross-split aggregation.\n"
    )

    f.write(
        "No run-level pooling was used.\n\n"
    )

    f.write(
        "Core frozen summary reproduction:\n"
        "  PASS: 18/18 rows and all "
        "reported metrics reproduced.\n\n"
    )

    f.write(
        "Unified split-level rows: "
        f"{len(combined)}\n"
    )

    f.write(
        "Unified protocol-summary rows: "
        f"{len(summary)}\n"
    )

    f.write(
        "Methods: "
        f"{combined['method'].nunique()}\n\n"
    )

    f.write(
        "FastANI/skani metrics remain "
        "reported-pair-only metrics; "
        "missing scores were not imputed.\n"
    )

    f.write(
        "Coverage must accompany their "
        "AUROC/AUPRC/MCC/F1/BAcc values.\n\n"
    )

    f.write(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: PASS\n"
    )


# ================================================================
# 14. Result hashes + script provenance.
# ================================================================

result_files = [
    split_file,
    summary_file,
    long_file,
    primary_file,
    input_sha_file,
    report,
]


sha_file = (
    OUT /
    "RESULT_SHA256.txt"
)


with open(
    sha_file,
    "w",
) as f:

    for p in result_files:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


script_path = Path(
    __file__
).resolve()


with open(
    OUT /
    "SCRIPT_PROVENANCE.txt",
    "w",
) as f:

    f.write(
        f"script={script_path}\n"
    )

    f.write(
        "script_sha256="
        f"{sha256_file(script_path)}\n"
    )

    f.write(
        f"python={sys.version}\n"
    )


# ================================================================
# Final console report.
# ================================================================

print()
print("=" * 104)
print(
    "E5 CORE + NATIVE PRIMARY COMPARISON"
)
print("=" * 104)

print(
    primary_table.to_string(
        index=False
    )
)

print()
print(
    "Unified split-level rows =",
    len(combined)
)

print(
    "Unified protocol rows    =",
    len(summary)
)

print(
    "Methods                  =",
    combined[
        "method"
    ].nunique()
)

print(
    "Core frozen reproduction = PASS"
)

print()
print("Output:")
print(OUT)

print()
print("STATUS = PASS")
