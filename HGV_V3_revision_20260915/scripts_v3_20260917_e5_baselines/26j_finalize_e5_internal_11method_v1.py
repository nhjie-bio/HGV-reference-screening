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

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

CORE = (
    E5 /
    "07_unified_internal/"
    "core_native_merge_v2_20260917"
)

CORE_FREEZE = (
    E5 /
    "07_unified_internal/"
    "core_native_merge_v2_freeze_v1_20260917"
)

DNA = (
    E5 /
    "06_dnaberts/"
    "11_blind_test_freeze_v1_20260918"
)

OUT = (
    E5 /
    "07_unified_internal/"
    "e5_internal_11method_final_v1_20260918"
)


if OUT.exists():
    raise RuntimeError(
        f"Output already exists:\n{OUT}\n"
        "Do not overwrite a frozen/final merge."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


# ======================================================================
# Input files
# ======================================================================

CORE_SPLIT = (
    CORE /
    "E5_CORE_NATIVE_SPLIT_LEVEL.tsv"
)

CORE_PROTOCOL = (
    CORE /
    "E5_CORE_NATIVE_PROTOCOL_SUMMARY.tsv"
)

CORE_PRIMARY = (
    CORE /
    "E5_CORE_NATIVE_PRIMARY_TABLE.tsv"
)

CORE_LONG = (
    CORE /
    "E5_CORE_NATIVE_LONG.tsv"
)

CORE_FREEZE_FILE = (
    CORE_FREEZE /
    "CORE_NATIVE_MERGE_FREEZE.txt"
)

CORE_FREEZE_SHA = (
    CORE_FREEZE /
    "FREEZE_SHA256.txt"
)


DNA_SPLIT = (
    DNA /
    "DNABERTS_BLIND_TEST_SPLIT_LEVEL.tsv"
)

DNA_PROTOCOL = (
    DNA /
    "DNABERTS_BLIND_TEST_PROTOCOL_SUMMARY.tsv"
)

DNA_PRIMARY = (
    DNA /
    "DNABERTS_PRIMARY_BLIND_TEST_SUMMARY.tsv"
)

DNA_FREEZE_FILE = (
    DNA /
    "DNABERTS_BLIND_TEST_FREEZE.txt"
)

DNA_FREEZE_SHA = (
    DNA /
    "FREEZE_SHA256.txt"
)


for p in [
    CORE_SPLIT,
    CORE_PROTOCOL,
    CORE_PRIMARY,
    CORE_LONG,
    CORE_FREEZE_FILE,
    CORE_FREEZE_SHA,
    DNA_SPLIT,
    DNA_PROTOCOL,
    DNA_PRIMARY,
    DNA_FREEZE_FILE,
    DNA_FREEZE_SHA,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


print("=" * 100)
print(
    "HGV-V3 E5 INTERNAL 11-METHOD FINAL MERGE v1"
)
print("=" * 100)

print()
print(
    "Scope:"
)

print(
    "G2 species holdout / "
    "G3 genus holdout / "
    "G4 family holdout"
)

print(
    "This merge finalizes the INTERNAL "
    "11-method comparison block only."
)


# ======================================================================
# Load frozen Core+Native
# ======================================================================

core_split = pd.read_csv(
    CORE_SPLIT,
    sep="\t",
)

core_protocol = pd.read_csv(
    CORE_PROTOCOL,
    sep="\t",
)

core_primary = pd.read_csv(
    CORE_PRIMARY,
    sep="\t",
)

core_long = pd.read_csv(
    CORE_LONG,
    sep="\t",
)


# ======================================================================
# Hard audit existing 10-method block
# ======================================================================

if core_split.shape != (
    90,
    28,
):
    raise RuntimeError(
        f"Unexpected core split shape: "
        f"{core_split.shape}"
    )


if core_protocol.shape != (
    30,
    32,
):
    raise RuntimeError(
        f"Unexpected core protocol shape: "
        f"{core_protocol.shape}"
    )


if core_primary.shape != (
    30,
    19,
):
    raise RuntimeError(
        f"Unexpected core primary shape: "
        f"{core_primary.shape}"
    )


if core_long.shape != (
    330,
    9,
):
    raise RuntimeError(
        f"Unexpected core long shape: "
        f"{core_long.shape}"
    )


core_methods = sorted(
    core_split[
        "method"
    ]
    .astype(str)
    .unique()
)


if len(
    core_methods
) != 10:
    raise RuntimeError(
        f"Expected 10 existing methods; "
        f"found {len(core_methods)}"
    )


if "DNABERT-S" in core_methods:
    raise RuntimeError(
        "DNABERT-S already exists "
        "in core block"
    )


expected_protocols = {
    "G2",
    "G3",
    "G4",
}


for df_name, df in [
    ("core_split", core_split),
    ("core_protocol", core_protocol),
    ("core_primary", core_primary),
    ("core_long", core_long),
]:

    if set(
        df[
            "protocol"
        ].astype(str)
    ) != expected_protocols:

        raise RuntimeError(
            f"{df_name}: unexpected "
            "protocol set"
        )


if core_split[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate core split-level key"
    )


if core_protocol[
    [
        "method",
        "protocol",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate core protocol key"
    )


if core_primary[
    [
        "method",
        "protocol",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate core primary key"
    )


if core_long[
    [
        "method",
        "protocol",
        "metric",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate core long key"
    )


print()
print(
    "Existing frozen Core+Native block:"
)

print(
    "methods       =",
    len(core_methods)
)

print(
    "split rows    =",
    len(core_split)
)

print(
    "protocol rows =",
    len(core_protocol)
)

print(
    "primary rows  =",
    len(core_primary)
)

print(
    "long rows     =",
    len(core_long)
)


# ======================================================================
# Load frozen DNABERT-S
# ======================================================================

dna_split_raw = pd.read_csv(
    DNA_SPLIT,
    sep="\t",
)

dna_protocol_raw = pd.read_csv(
    DNA_PROTOCOL,
    sep="\t",
)

dna_primary_raw = pd.read_csv(
    DNA_PRIMARY,
    sep="\t",
)


if dna_split_raw.shape != (
    9,
    29,
):
    raise RuntimeError(
        f"Unexpected DNABERT-S split shape: "
        f"{dna_split_raw.shape}"
    )


if dna_protocol_raw.shape != (
    3,
    25,
):
    raise RuntimeError(
        "Unexpected DNABERT-S "
        f"protocol shape: "
        f"{dna_protocol_raw.shape}"
    )


if dna_primary_raw.shape != (
    3,
    15,
):
    raise RuntimeError(
        "Unexpected DNABERT-S "
        f"primary shape: "
        f"{dna_primary_raw.shape}"
    )


if set(
    dna_split_raw[
        "method"
    ].astype(str)
) != {
    "DNABERT-S"
}:
    raise RuntimeError(
        "Unexpected DNABERT-S method name"
    )


if set(
    dna_split_raw[
        "protocol"
    ].astype(str)
) != expected_protocols:
    raise RuntimeError(
        "Unexpected DNABERT-S protocols"
    )


if not np.allclose(
    pd.to_numeric(
        dna_split_raw[
            "coverage"
        ]
    ),
    1.0,
):
    raise RuntimeError(
        "DNABERT-S coverage is not 1.0"
    )


if not np.allclose(
    pd.to_numeric(
        dna_split_raw[
            "positive_coverage"
        ]
    ),
    1.0,
):
    raise RuntimeError(
        "DNABERT-S positive coverage "
        "is not 1.0"
    )


if not np.allclose(
    pd.to_numeric(
        dna_split_raw[
            "negative_coverage"
        ]
    ),
    1.0,
):
    raise RuntimeError(
        "DNABERT-S negative coverage "
        "is not 1.0"
    )


if set(
    pd.to_numeric(
        dna_split_raw[
            "n_model_seeds"
        ]
    ).astype(int)
) != {
    1
}:
    raise RuntimeError(
        "DNABERT-S must retain "
        "n_model_seeds=1"
    )


if set(
    dna_split_raw[
        "metric_scope"
    ].astype(str)
) != {
    "all_test_pairs"
}:
    raise RuntimeError(
        "Unexpected DNABERT-S "
        "metric scope"
    )


if int(
    pd.to_numeric(
        dna_split_raw[
            "n_pairs"
        ]
    ).sum()
) != 34882:
    raise RuntimeError(
        "DNABERT-S total test pairs "
        "!=34882"
    )


# ======================================================================
# Infer deterministic-single-run convention from existing frozen block
# ======================================================================

deterministic_core = (
    core_split[
        pd.to_numeric(
            core_split[
                "n_model_seeds"
            ],
            errors="coerce",
        )
        ==
        1
    ]
)


if len(
    deterministic_core
) == 0:
    raise RuntimeError(
        "No deterministic existing "
        "baseline found to infer "
        "within-split SD convention"
    )


within_sd_cols = [
    "within_split_AUROC_sd_model_seed",
    "within_split_AUPRC_sd_model_seed",
    "within_split_MCC_sd_model_seed",
    "within_split_F1_sd_model_seed",
    "within_split_BalancedAccuracy_sd_model_seed",
]


# For deterministic baselines, retain the
# existing block's convention:
# if all values are missing -> use NaN;
# if all finite values are zero -> use 0.
deterministic_sd_values = {}


for c in within_sd_cols:

    vals = pd.to_numeric(
        deterministic_core[
            c
        ],
        errors="coerce",
    )


    finite = vals[
        np.isfinite(
            vals
        )
    ]


    if len(
        finite
    ) == 0:

        deterministic_sd_values[
            c
        ] = np.nan

    elif np.allclose(
        finite,
        0.0,
    ):

        deterministic_sd_values[
            c
        ] = 0.0

    else:

        raise RuntimeError(
            f"Existing deterministic "
            f"baseline has unexpected "
            f"within-split SD values "
            f"in {c}"
        )


print()
print(
    "Existing deterministic-baseline "
    "within-split SD convention:"
)

for c in within_sd_cols:

    print(
        f"  {c} = "
        f"{deterministic_sd_values[c]}"
    )


# ======================================================================
# Convert DNABERT-S -> exact 28-column split-level schema
# ======================================================================

dna_split = pd.DataFrame({
    "method_type":
        "PretrainedDNA_LM",

    "method":
        dna_split_raw[
            "method"
        ].astype(str),

    "protocol":
        dna_split_raw[
            "protocol"
        ].astype(str),

    "split_seed":
        dna_split_raw[
            "split_seed"
        ],

    "source_family":
        "dnaberts_frozen",

    "replicate_structure":
        "deterministic_single_run",

    "n_model_seeds":
        pd.to_numeric(
            dna_split_raw[
                "n_model_seeds"
            ]
        ),

    "n_pairs_total":
        pd.to_numeric(
            dna_split_raw[
                "n_pairs"
            ]
        ),

    "n_pairs_covered":
        pd.to_numeric(
            dna_split_raw[
                "n_pairs"
            ]
        ),

    "n_pairs_missing":
        0,

    "coverage":
        pd.to_numeric(
            dna_split_raw[
                "coverage"
            ]
        ),

    "positive_coverage":
        pd.to_numeric(
            dna_split_raw[
                "positive_coverage"
            ]
        ),

    "negative_coverage":
        pd.to_numeric(
            dna_split_raw[
                "negative_coverage"
            ]
        ),

    "AUROC":
        pd.to_numeric(
            dna_split_raw[
                "auroc"
            ]
        ),

    "AUPRC":
        pd.to_numeric(
            dna_split_raw[
                "auprc"
            ]
        ),

    "MCC":
        pd.to_numeric(
            dna_split_raw[
                "mcc"
            ]
        ),

    "F1":
        pd.to_numeric(
            dna_split_raw[
                "f1"
            ]
        ),

    "BalancedAccuracy":
        pd.to_numeric(
            dna_split_raw[
                "balanced_accuracy"
            ]
        ),

    "Precision":
        pd.to_numeric(
            dna_split_raw[
                "precision"
            ]
        ),

    "Recall":
        pd.to_numeric(
            dna_split_raw[
                "recall"
            ]
        ),

    "Specificity":
        pd.to_numeric(
            dna_split_raw[
                "specificity"
            ]
        ),

    "coverage_type":
        "complete",

    "metric_scope":
        "all_test_pairs",

    "within_split_AUROC_sd_model_seed":
        deterministic_sd_values[
            "within_split_AUROC_sd_model_seed"
        ],

    "within_split_AUPRC_sd_model_seed":
        deterministic_sd_values[
            "within_split_AUPRC_sd_model_seed"
        ],

    "within_split_MCC_sd_model_seed":
        deterministic_sd_values[
            "within_split_MCC_sd_model_seed"
        ],

    "within_split_F1_sd_model_seed":
        deterministic_sd_values[
            "within_split_F1_sd_model_seed"
        ],

    "within_split_BalancedAccuracy_sd_model_seed":
        deterministic_sd_values[
            "within_split_BalancedAccuracy_sd_model_seed"
        ],
})


dna_split = dna_split[
    core_split.columns
]


if dna_split.shape != (
    9,
    28,
):
    raise RuntimeError(
        "Converted DNABERT-S split-level "
        "shape is wrong"
    )


# ======================================================================
# Recompute DNABERT-S protocol summary FROM converted split-level rows
# using the exact same statistical unit: taxonomy split seeds.
# ======================================================================

protocol_rows = []


metric_map = {
    "coverage":
        "coverage",

    "positive_coverage":
        "positive_coverage",

    "negative_coverage":
        "negative_coverage",

    "AUROC":
        "AUROC",

    "AUPRC":
        "AUPRC",

    "MCC":
        "MCC",

    "F1":
        "F1",

    "BalancedAccuracy":
        "BalancedAccuracy",

    "Precision":
        "Precision",

    "Recall":
        "Recall",

    "Specificity":
        "Specificity",
}


for protocol in [
    "G2",
    "G3",
    "G4",
]:

    x = (
        dna_split[
            dna_split[
                "protocol"
            ]
            ==
            protocol
        ]
        .copy()
    )


    if len(
        x
    ) != 3:
        raise RuntimeError(
            f"{protocol}: DNABERT-S "
            "does not have 3 split seeds"
        )


    row = {
        "method":
            "DNABERT-S",

        "method_type":
            "PretrainedDNA_LM",

        "protocol":
            protocol,

        "n_split_seeds":
            3,

        "replicate_unit":
            "taxonomy_split_seed",

        "coverage_type":
            "complete",

        "metric_scope":
            "all_test_pairs",

        "n_pairs_total_sum":
            int(
                pd.to_numeric(
                    x[
                        "n_pairs_total"
                    ]
                ).sum()
            ),

        "n_pairs_covered_sum":
            int(
                pd.to_numeric(
                    x[
                        "n_pairs_covered"
                    ]
                ).sum()
            ),

        "n_pairs_missing_sum":
            int(
                pd.to_numeric(
                    x[
                        "n_pairs_missing"
                    ]
                ).sum()
            ),
    }


    for source_col, output_prefix in (
        metric_map.items()
    ):

        vals = pd.to_numeric(
            x[
                source_col
            ]
        ).to_numpy(
            dtype=np.float64
        )


        row[
            f"{output_prefix}_mean"
        ] = float(
            np.mean(
                vals
            )
        )


        row[
            f"{output_prefix}_sd_split_seed"
        ] = float(
            np.std(
                vals,
                ddof=1,
            )
        )


    protocol_rows.append(
        row
    )


dna_protocol = pd.DataFrame(
    protocol_rows
)


dna_protocol = dna_protocol[
    core_protocol.columns
]


if dna_protocol.shape != (
    3,
    32,
):
    raise RuntimeError(
        "Converted DNABERT-S protocol "
        "summary shape is wrong"
    )


# ======================================================================
# Cross-check against DNABERT-S's own 26I protocol summary
# ======================================================================

dna_proto_by_key = {
    str(r.protocol):
        r
    for r in (
        dna_protocol_raw.itertuples(
            index=False
        )
    )
}


crosswalk = {
    "coverage_mean":
        "coverage_mean",

    "coverage_sd_split_seed":
        "coverage_sd",

    "AUROC_mean":
        "auroc_mean",

    "AUROC_sd_split_seed":
        "auroc_sd",

    "AUPRC_mean":
        "auprc_mean",

    "AUPRC_sd_split_seed":
        "auprc_sd",

    "MCC_mean":
        "mcc_mean",

    "MCC_sd_split_seed":
        "mcc_sd",

    "F1_mean":
        "f1_mean",

    "F1_sd_split_seed":
        "f1_sd",

    "BalancedAccuracy_mean":
        "balanced_accuracy_mean",

    "BalancedAccuracy_sd_split_seed":
        "balanced_accuracy_sd",

    "Precision_mean":
        "precision_mean",

    "Precision_sd_split_seed":
        "precision_sd",

    "Recall_mean":
        "recall_mean",

    "Recall_sd_split_seed":
        "recall_sd",

    "Specificity_mean":
        "specificity_mean",

    "Specificity_sd_split_seed":
        "specificity_sd",
}


for r in dna_protocol.itertuples(
    index=False
):

    source = dna_proto_by_key[
        str(
            r.protocol
        )
    ]


    for final_col, dna_col in (
        crosswalk.items()
    ):

        a = float(
            getattr(
                r,
                final_col
            )
        )

        b = float(
            getattr(
                source,
                dna_col
            )
        )


        if not np.isclose(
            a,
            b,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"DNABERT-S protocol "
                f"summary reproduction "
                f"failed: "
                f"{r.protocol} "
                f"{final_col}: "
                f"{a} != {b}"
            )


print()
print(
    "DNABERT-S protocol-summary "
    "reproduction: PASS"
)


# ======================================================================
# Build DNABERT-S primary table from protocol summary
# ======================================================================

dna_primary = pd.DataFrame({
    "method":
        dna_protocol[
            "method"
        ],

    "method_type":
        dna_protocol[
            "method_type"
        ],

    "protocol":
        dna_protocol[
            "protocol"
        ],

    "coverage_mean":
        dna_protocol[
            "coverage_mean"
        ],

    "coverage_sd_split_seed":
        dna_protocol[
            "coverage_sd_split_seed"
        ],

    "positive_coverage_mean":
        dna_protocol[
            "positive_coverage_mean"
        ],

    "negative_coverage_mean":
        dna_protocol[
            "negative_coverage_mean"
        ],

    "AUROC_mean":
        dna_protocol[
            "AUROC_mean"
        ],

    "AUROC_sd_split_seed":
        dna_protocol[
            "AUROC_sd_split_seed"
        ],

    "AUPRC_mean":
        dna_protocol[
            "AUPRC_mean"
        ],

    "AUPRC_sd_split_seed":
        dna_protocol[
            "AUPRC_sd_split_seed"
        ],

    "MCC_mean":
        dna_protocol[
            "MCC_mean"
        ],

    "MCC_sd_split_seed":
        dna_protocol[
            "MCC_sd_split_seed"
        ],

    "F1_mean":
        dna_protocol[
            "F1_mean"
        ],

    "F1_sd_split_seed":
        dna_protocol[
            "F1_sd_split_seed"
        ],

    "BalancedAccuracy_mean":
        dna_protocol[
            "BalancedAccuracy_mean"
        ],

    "BalancedAccuracy_sd_split_seed":
        dna_protocol[
            "BalancedAccuracy_sd_split_seed"
        ],

    "metric_scope":
        dna_protocol[
            "metric_scope"
        ],

    "coverage_type":
        dna_protocol[
            "coverage_type"
        ],
})


dna_primary = dna_primary[
    core_primary.columns
]


if dna_primary.shape != (
    3,
    19,
):
    raise RuntimeError(
        "Converted DNABERT-S primary "
        "table shape is wrong"
    )


# Cross-check main metrics against
# DNABERT-S's own primary table.
dna_primary_source = {
    str(r.protocol):
        r
    for r in (
        dna_primary_raw.itertuples(
            index=False
        )
    )
}


primary_crosswalk = {
    "coverage_mean":
        "coverage_mean",

    "coverage_sd_split_seed":
        "coverage_sd",

    "AUROC_mean":
        "auroc_mean",

    "AUROC_sd_split_seed":
        "auroc_sd",

    "AUPRC_mean":
        "auprc_mean",

    "AUPRC_sd_split_seed":
        "auprc_sd",

    "MCC_mean":
        "mcc_mean",

    "MCC_sd_split_seed":
        "mcc_sd",

    "F1_mean":
        "f1_mean",

    "F1_sd_split_seed":
        "f1_sd",

    "BalancedAccuracy_mean":
        "balanced_accuracy_mean",

    "BalancedAccuracy_sd_split_seed":
        "balanced_accuracy_sd",
}


for r in dna_primary.itertuples(
    index=False
):

    source = dna_primary_source[
        str(
            r.protocol
        )
    ]


    for final_col, dna_col in (
        primary_crosswalk.items()
    ):

        a = float(
            getattr(
                r,
                final_col
            )
        )

        b = float(
            getattr(
                source,
                dna_col
            )
        )


        if not np.isclose(
            a,
            b,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"DNABERT-S primary "
                f"reproduction failed: "
                f"{r.protocol} "
                f"{final_col}"
            )


print(
    "DNABERT-S primary-table "
    "reproduction: PASS"
)


# ======================================================================
# Build long table
# ======================================================================

long_metrics = [
    (
        "coverage",
        "coverage_mean",
        "coverage_sd_split_seed",
    ),

    (
        "positive_coverage",
        "positive_coverage_mean",
        "positive_coverage_sd_split_seed",
    ),

    (
        "negative_coverage",
        "negative_coverage_mean",
        "negative_coverage_sd_split_seed",
    ),

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

    (
        "F1",
        "F1_mean",
        "F1_sd_split_seed",
    ),

    (
        "BalancedAccuracy",
        "BalancedAccuracy_mean",
        "BalancedAccuracy_sd_split_seed",
    ),

    (
        "Precision",
        "Precision_mean",
        "Precision_sd_split_seed",
    ),

    (
        "Recall",
        "Recall_mean",
        "Recall_sd_split_seed",
    ),

    (
        "Specificity",
        "Specificity_mean",
        "Specificity_sd_split_seed",
    ),
]


dna_long_rows = []


for r in dna_protocol.itertuples(
    index=False
):

    for metric, mean_col, sd_col in (
        long_metrics
    ):

        dna_long_rows.append({
            "method":
                "DNABERT-S",

            "method_type":
                "PretrainedDNA_LM",

            "protocol":
                str(
                    r.protocol
                ),

            "metric":
                metric,

            "mean":
                float(
                    getattr(
                        r,
                        mean_col
                    )
                ),

            "sd_split_seed":
                float(
                    getattr(
                        r,
                        sd_col
                    )
                ),

            "n_split_seeds":
                3,

            "coverage_type":
                "complete",

            "metric_scope":
                "all_test_pairs",
        })


dna_long = pd.DataFrame(
    dna_long_rows
)


dna_long = dna_long[
    core_long.columns
]


if dna_long.shape != (
    33,
    9,
):
    raise RuntimeError(
        "Converted DNABERT-S long "
        "table shape is wrong"
    )


# ======================================================================
# Merge
# ======================================================================

final_split = pd.concat(
    [
        core_split,
        dna_split,
    ],
    ignore_index=True,
)


final_protocol = pd.concat(
    [
        core_protocol,
        dna_protocol,
    ],
    ignore_index=True,
)


final_primary = pd.concat(
    [
        core_primary,
        dna_primary,
    ],
    ignore_index=True,
)


final_long = pd.concat(
    [
        core_long,
        dna_long,
    ],
    ignore_index=True,
)


# ======================================================================
# Verify old 10-method block numerically/structurally unchanged
# ======================================================================

pd.testing.assert_frame_equal(
    final_split.iloc[
        :len(
            core_split
        )
    ].reset_index(
        drop=True
    ),
    core_split.reset_index(
        drop=True
    ),
    check_dtype=False,
)


pd.testing.assert_frame_equal(
    final_protocol.iloc[
        :len(
            core_protocol
        )
    ].reset_index(
        drop=True
    ),
    core_protocol.reset_index(
        drop=True
    ),
    check_dtype=False,
)


pd.testing.assert_frame_equal(
    final_primary.iloc[
        :len(
            core_primary
        )
    ].reset_index(
        drop=True
    ),
    core_primary.reset_index(
        drop=True
    ),
    check_dtype=False,
)


pd.testing.assert_frame_equal(
    final_long.iloc[
        :len(
            core_long
        )
    ].reset_index(
        drop=True
    ),
    core_long.reset_index(
        drop=True
    ),
    check_dtype=False,
)


print()
print(
    "Frozen 10-method block "
    "preservation: PASS"
)


# ======================================================================
# Final hard checks
# ======================================================================

if final_split.shape != (
    99,
    28,
):
    raise RuntimeError(
        f"Final split shape wrong: "
        f"{final_split.shape}"
    )


if final_protocol.shape != (
    33,
    32,
):
    raise RuntimeError(
        f"Final protocol shape wrong: "
        f"{final_protocol.shape}"
    )


if final_primary.shape != (
    33,
    19,
):
    raise RuntimeError(
        f"Final primary shape wrong: "
        f"{final_primary.shape}"
    )


if final_long.shape != (
    363,
    9,
):
    raise RuntimeError(
        f"Final long shape wrong: "
        f"{final_long.shape}"
    )


final_methods = sorted(
    final_split[
        "method"
    ]
    .astype(str)
    .unique()
)


if len(
    final_methods
) != 11:
    raise RuntimeError(
        f"Expected 11 methods; "
        f"found {len(final_methods)}"
    )


if "DNABERT-S" not in (
    final_methods
):
    raise RuntimeError(
        "DNABERT-S absent "
        "from final methods"
    )


if final_split[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate final split key"
    )


if final_protocol[
    [
        "method",
        "protocol",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate final protocol key"
    )


if final_primary[
    [
        "method",
        "protocol",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate final primary key"
    )


if final_long[
    [
        "method",
        "protocol",
        "metric",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate final long key"
    )


# DNABERT-S must remain deterministic.
final_dna_split = (
    final_split[
        final_split[
            "method"
        ]
        ==
        "DNABERT-S"
    ]
)


if set(
    pd.to_numeric(
        final_dna_split[
            "n_model_seeds"
        ]
    ).astype(int)
) != {
    1
}:
    raise RuntimeError(
        "Final DNABERT-S "
        "n_model_seeds changed"
    )


# HGV must remain the 3-model-seed
# within-split mean.
hgv = final_split[
    final_split[
        "method"
    ]
    ==
    "HGV_Dashing2_tau0p98"
]


if set(
    pd.to_numeric(
        hgv[
            "n_model_seeds"
        ]
    ).astype(int)
) != {
    3
}:
    raise RuntimeError(
        "HGV n_model_seeds "
        "was altered"
    )


# Every method must have exactly:
# 3 protocols x 3 split seeds = 9 rows.
split_counts = (
    final_split.groupby(
        "method"
    )
    .size()
)


if not (
    split_counts == 9
).all():
    raise RuntimeError(
        "Not every method has "
        "9 split-level rows"
    )


protocol_counts = (
    final_protocol.groupby(
        "method"
    )
    .size()
)


if not (
    protocol_counts == 3
).all():
    raise RuntimeError(
        "Not every method has "
        "3 protocol-summary rows"
    )


long_counts = (
    final_long.groupby(
        "method"
    )
    .size()
)


if not (
    long_counts == 33
).all():
    raise RuntimeError(
        "Not every method has "
        "33 long-format rows"
    )


# ======================================================================
# Write outputs
# ======================================================================

SPLIT_OUT = (
    OUT /
    "E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv"
)

PROTOCOL_OUT = (
    OUT /
    "E5_INTERNAL_11METHOD_PROTOCOL_SUMMARY.tsv"
)

PRIMARY_OUT = (
    OUT /
    "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv"
)

LONG_OUT = (
    OUT /
    "E5_INTERNAL_11METHOD_LONG.tsv"
)


final_split.to_csv(
    SPLIT_OUT,
    sep="\t",
    index=False,
)

final_protocol.to_csv(
    PROTOCOL_OUT,
    sep="\t",
    index=False,
)

final_primary.to_csv(
    PRIMARY_OUT,
    sep="\t",
    index=False,
)

final_long.to_csv(
    LONG_OUT,
    sep="\t",
    index=False,
)


# ======================================================================
# Method catalog
# ======================================================================

catalog_rows = []


for method in final_methods:

    x = final_split[
        final_split[
            "method"
        ]
        ==
        method
    ]


    method_types = sorted(
        x[
            "method_type"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    coverage_types = sorted(
        x[
            "coverage_type"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    scopes = sorted(
        x[
            "metric_scope"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    n_model = sorted(
        set(
            pd.to_numeric(
                x[
                    "n_model_seeds"
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )
    )


    catalog_rows.append({
        "method":
            method,

        "method_type":
            ";".join(
                method_types
            ),

        "n_model_seeds":
            ";".join(
                str(v)
                for v in n_model
            ),

        "n_split_seeds":
            3,

        "coverage_type":
            ";".join(
                coverage_types
            ),

        "metric_scope":
            ";".join(
                scopes
            ),

        "n_split_level_rows":
            len(
                x
            ),
    })


catalog = pd.DataFrame(
    catalog_rows
)


CATALOG_OUT = (
    OUT /
    "E5_INTERNAL_11METHOD_METHOD_CATALOG.tsv"
)


catalog.to_csv(
    CATALOG_OUT,
    sep="\t",
    index=False,
)


# ======================================================================
# Provenance
# ======================================================================

script_path = Path(
    __file__
).resolve()


input_rows = []


for name, path in [
    (
        "CORE_SPLIT",
        CORE_SPLIT,
    ),
    (
        "CORE_PROTOCOL",
        CORE_PROTOCOL,
    ),
    (
        "CORE_PRIMARY",
        CORE_PRIMARY,
    ),
    (
        "CORE_LONG",
        CORE_LONG,
    ),
    (
        "CORE_FREEZE",
        CORE_FREEZE_FILE,
    ),
    (
        "CORE_FREEZE_SHA",
        CORE_FREEZE_SHA,
    ),
    (
        "DNABERT_SPLIT",
        DNA_SPLIT,
    ),
    (
        "DNABERT_PROTOCOL",
        DNA_PROTOCOL,
    ),
    (
        "DNABERT_PRIMARY",
        DNA_PRIMARY,
    ),
    (
        "DNABERT_FREEZE",
        DNA_FREEZE_FILE,
    ),
    (
        "DNABERT_FREEZE_SHA",
        DNA_FREEZE_SHA,
    ),
]:

    input_rows.append({
        "input":
            name,

        "file":
            str(
                path
            ),

        "sha256":
            sha256_file(
                path
            ),
    })


INPUT_OUT = (
    OUT /
    "INPUT_SHA256.tsv"
)


pd.DataFrame(
    input_rows
).to_csv(
    INPUT_OUT,
    sep="\t",
    index=False,
)


PROVENANCE = (
    OUT /
    "SCRIPT_PROVENANCE.txt"
)


PROVENANCE.write_text(
    f"script={script_path}\n"
    f"script_sha256={sha256_file(script_path)}\n"
    f"python={sys.version}\n"
    f"numpy={np.__version__}\n"
    f"pandas={pd.__version__}\n"
)


# ======================================================================
# Report
# ======================================================================

dna_primary_final = (
    final_primary[
        final_primary[
            "method"
        ]
        ==
        "DNABERT-S"
    ]
    .copy()
)


REPORT = (
    OUT /
    "E5_INTERNAL_11METHOD_FINAL_REPORT.txt"
)


with open(
    REPORT,
    "w",
) as f:

    f.write(
        "HGV-V3 E5 Internal "
        "11-Method Final Merge v1\n"
    )

    f.write(
        "=" * 88 +
        "\n\n"
    )

    f.write(
        "Scope:\n"
        "G2 species holdout\n"
        "G3 genus holdout\n"
        "G4 family holdout\n\n"
    )

    f.write(
        "This freeze covers the internal "
        "G2/G3/G4 benchmark block. "
        "It does not by itself finalize "
        "external B1/B2 validation.\n\n"
    )

    f.write(
        f"Methods: "
        f"{len(final_methods)}\n"
    )

    f.write(
        f"Split-level rows: "
        f"{len(final_split)}\n"
    )

    f.write(
        f"Protocol-summary rows: "
        f"{len(final_protocol)}\n"
    )

    f.write(
        f"Primary-table rows: "
        f"{len(final_primary)}\n"
    )

    f.write(
        f"Long-format rows: "
        f"{len(final_long)}\n\n"
    )

    f.write(
        "Primary variability unit:\n"
        "taxonomy split seed "
        "(3 seeds per protocol)\n\n"
    )

    f.write(
        "Replicate structure:\n"
        "- HGV: 3 model seeds averaged "
        "within each taxonomy split before "
        "cross-split summary.\n"
        "- deterministic baselines: "
        "n_model_seeds=1.\n"
        "- DNABERT-S: frozen pretrained "
        "deterministic baseline, "
        "n_model_seeds=1.\n\n"
    )

    f.write(
        "Coverage interpretation:\n"
        "- complete methods are evaluated "
        "on all fixed test pairs.\n"
        "- native ANI methods retain their "
        "previous frozen covered-only/"
        "conditional metric scope and "
        "coverage reporting.\n"
        "- no missing native ANI scores "
        "are imputed.\n\n"
    )

    f.write(
        "DNABERT-S PRIMARY RESULTS\n"
    )

    f.write(
        "-" * 88 +
        "\n"
    )

    f.write(
        dna_primary_final.to_string(
            index=False
        )
    )

    f.write(
        "\n\nMETHODS\n"
    )

    f.write(
        "-" * 88 +
        "\n"
    )

    for method in final_methods:

        f.write(
            f"{method}\n"
        )


    f.write(
        "\nFrozen prior Core+Native "
        "rows preserved unchanged: PASS\n"
    )

    f.write(
        "DNABERT-S 26I reproduction: PASS\n"
    )

    f.write(
        "Final row-count audits: PASS\n"
    )

    f.write(
        "Final duplicate-key audits: PASS\n"
    )

    f.write(
        "\nSTATUS: FROZEN\n"
    )


# ======================================================================
# Result hashes
# ======================================================================

RESULT_SHA = (
    OUT /
    "RESULT_SHA256.txt"
)


result_files = [
    SPLIT_OUT,
    PROTOCOL_OUT,
    PRIMARY_OUT,
    LONG_OUT,
    CATALOG_OUT,
    INPUT_OUT,
    PROVENANCE,
    REPORT,
]


with open(
    RESULT_SHA,
    "w",
) as f:

    for p in result_files:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ======================================================================
# Dedicated freeze descriptor
# ======================================================================

FREEZE_TXT = (
    OUT /
    "E5_INTERNAL_11METHOD_FREEZE.txt"
)


FREEZE_TXT.write_text(
    """HGV-V3 E5 Internal 11-Method Freeze v1

Scope:
Internal fixed-pair G2/G3/G4 benchmark.

Methods:
11

Split-level rows:
99

Protocol-summary rows:
33

Primary-table rows:
33

Long-format rows:
363

DNABERT-S:
included
n_model_seeds=1
coverage=complete
metric_scope=all_test_pairs

HGV:
3 model seeds averaged within each taxonomy split.

Primary cross-split variability:
3 taxonomy split seeds.

FastANI/skani:
retain frozen conditional/covered-only evaluation semantics.
Missing native scores are not imputed.

External B1/B2:
not finalized by this internal freeze.

STATUS:
FROZEN
"""
)


FREEZE_SHA = (
    OUT /
    "FREEZE_SHA256.txt"
)


with open(
    FREEZE_SHA,
    "w",
) as f:

    for p in [
        FREEZE_TXT,
        RESULT_SHA,
        INPUT_OUT,
        REPORT,
    ]:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ======================================================================
# Console output
# ======================================================================

print()
print("=" * 100)
print(
    "FINAL E5 INTERNAL 11-METHOD AUDIT"
)
print("=" * 100)

print(
    "methods =",
    len(final_methods)
)

print(
    "split-level rows =",
    len(final_split)
)

print(
    "protocol-summary rows =",
    len(final_protocol)
)

print(
    "primary rows =",
    len(final_primary)
)

print(
    "long rows =",
    len(final_long)
)

print()
print(
    "METHODS:"
)

for method in final_methods:

    print(
        " ",
        method
    )


print()
print(
    "DNABERT-S PRIMARY:"
)

print(
    dna_primary_final[
        [
            "protocol",
            "coverage_mean",
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
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Frozen 10-method preservation = PASS"
)

print(
    "DNABERT-S reproduction = PASS"
)

print(
    "duplicate-key audit = PASS"
)

print()
print(
    "Output:"
)

print(
    OUT
)

print()
print(
    "STATUS = FROZEN"
)
