#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

THRESHOLD_FILE = (
    E5 /
    "00_native_validation/"
    "threshold_freeze_v1_20260917/"
    "FROZEN_NATIVE_VALIDATION_THRESHOLDS.tsv"
)

OUT = (
    E5 /
    "07_unified_internal/"
    "native_blind_test_evaluation_v1_20260917"
)


if OUT.exists():
    raise RuntimeError(
        f"Output already exists:\n{OUT}\n"
        "Do not overwrite frozen/evaluated outputs."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


EXPECTED = {
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


METHODS = {

    "Mash": {
        "root":
            E5 /
            "02_mash/"
            "internal_cache_join_v1/"
            "results",

        "filename":
            "mash_fixed_pair_results.tsv",

        "score_column":
            "mash_similarity",

        "fallback_ani":
            None,

        "coverage_type":
            "complete",
    },

    "Dashing2_SetSketch": {
        "root":
            E5 /
            "03_dashing2/"
            "internal_cache_join_v1/"
            "results",

        "filename":
            "dashing2_fixed_pair_results.tsv",

        "score_column":
            "dashing2_similarity",

        "fallback_ani":
            None,

        "coverage_type":
            "complete",
    },

    "FastANI": {
        "root":
            E5 /
            "05_fastani/"
            "internal_bidirectional_v2/"
            "results",

        "filename":
            "fastani_symmetric_fixed_pair_results.tsv",

        "score_column":
            "fastani_symmetric_similarity",

        "fallback_ani":
            "fastani_symmetric_ani",

        "coverage_type":
            "censored_native",
    },

    "skani": {
        "root":
            E5 /
            "04_skani/"
            "internal_v1/"
            "results",

        "filename":
            "skani_fixed_pair_results.tsv",

        "score_column":
            "skani_similarity",

        "fallback_ani":
            "skani_ani",

        "coverage_type":
            "censored_native",
    },
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


def load_score(
    df,
    method,
    spec,
):

    c = spec["score_column"]

    if c in df.columns:

        return (
            c,
            pd.to_numeric(
                df[c],
                errors="coerce",
            ),
        )


    fallback = spec[
        "fallback_ani"
    ]

    if (
        fallback is not None
        and fallback in df.columns
    ):

        ani = pd.to_numeric(
            df[fallback],
            errors="coerce",
        )

        return (
            fallback + "/100",
            ani / 100.0,
        )


    raise RuntimeError(
        f"{method}: score column "
        "not found"
    )


def classification_metrics(
    y,
    score,
    threshold,
):

    pred = (
        score >= threshold
    ).astype(int)

    mcc = float(
        matthews_corrcoef(
            y,
            pred,
        )
    )

    f1 = float(
        f1_score(
            y,
            pred,
            zero_division=0,
        )
    )

    bacc = float(
        balanced_accuracy_score(
            y,
            pred,
        )
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else float("nan")
    )

    return {
        "mcc":
            mcc,

        "f1":
            f1,

        "balanced_accuracy":
            bacc,

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "tp":
            int(tp),

        "fp":
            int(fp),

        "tn":
            int(tn),

        "fn":
            int(fn),
    }


# ================================================================
# Load frozen validation thresholds
# ================================================================

print("=" * 100)
print(
    "E5 NATIVE BLIND-TEST EVALUATION v1"
)
print("=" * 100)

print()
print(
    "Frozen threshold source:"
)
print(THRESHOLD_FILE)


if not THRESHOLD_FILE.is_file():
    raise FileNotFoundError(
        THRESHOLD_FILE
    )


thresholds = pd.read_csv(
    THRESHOLD_FILE,
    sep="\t",
    dtype={
        "split_seed": str,
    },
)


required_thr = [
    "method",
    "protocol",
    "split_seed",
    "frozen_threshold",
]

for c in required_thr:
    if c not in thresholds.columns:
        raise RuntimeError(
            f"Threshold file missing {c}"
        )


if len(thresholds) != 36:
    raise RuntimeError(
        f"Expected 36 thresholds, "
        f"found {len(thresholds)}"
    )


if thresholds[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate frozen threshold key"
    )


print(
    "Frozen thresholds =",
    len(thresholds)
)

print(
    "Threshold SHA256  =",
    sha256_file(
        THRESHOLD_FILE
    )
)


# ================================================================
# Blind-test evaluation
# ================================================================

split_rows = []
input_sha_rows = []


for method, spec in METHODS.items():

    print()
    print("=" * 100)
    print(method)
    print("=" * 100)

    for protocol in [
        "G2",
        "G3",
        "G4",
    ]:

        for seed in [
            "20260915",
            "20260916",
            "20260917",
        ]:

            expected = EXPECTED[
                (protocol, seed)
            ]


            thr_match = thresholds[
                (thresholds["method"] == method)
                &
                (thresholds["protocol"] == protocol)
                &
                (
                    thresholds[
                        "split_seed"
                    ].astype(str)
                    == seed
                )
            ]


            if len(thr_match) != 1:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    f"threshold rows="
                    f"{len(thr_match)}"
                )


            threshold = float(
                thr_match.iloc[0][
                    "frozen_threshold"
                ]
            )


            src = (
                spec["root"] /
                f"{protocol}_seed{seed}" /
                spec["filename"]
            )


            if not src.is_file():

                raise FileNotFoundError(
                    src
                )


            df = pd.read_csv(
                src,
                sep="\t",
            )


            if len(df) != expected:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    f"expected {expected}, "
                    f"found {len(df)}"
                )


            if "label" not in df.columns:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "label missing"
                )


            y = pd.to_numeric(
                df["label"],
                errors="raise",
            ).astype(int)


            if set(y.unique()) != {0, 1}:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "non-binary labels"
                )


            score_source, score = (
                load_score(
                    df,
                    method,
                    spec,
                )
            )


            covered = (
                score.notna()
            )


            n_total = int(
                len(df)
            )

            n_covered = int(
                covered.sum()
            )

            n_missing = (
                n_total -
                n_covered
            )


            pos = (
                y == 1
            )

            neg = (
                y == 0
            )


            n_pos = int(
                pos.sum()
            )

            n_neg = int(
                neg.sum()
            )


            n_pos_covered = int(
                (
                    pos &
                    covered
                ).sum()
            )

            n_neg_covered = int(
                (
                    neg &
                    covered
                ).sum()
            )


            coverage = (
                n_covered /
                n_total
            )

            positive_coverage = (
                n_pos_covered /
                n_pos
            )

            negative_coverage = (
                n_neg_covered /
                n_neg
            )


            if (
                spec[
                    "coverage_type"
                ]
                == "complete"
                and n_missing != 0
            ):

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "unexpected missing "
                    "test score"
                )


            y_cov = (
                y[covered]
                .to_numpy(
                    dtype=int
                )
            )

            s_cov = (
                score[covered]
                .to_numpy(
                    dtype=float
                )
            )


            if len(y_cov) == 0:
                raise RuntimeError(
                    "No covered test pairs"
                )


            if set(
                np.unique(
                    y_cov
                ).tolist()
            ) != {0, 1}:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "covered test subset "
                    "does not contain "
                    "both classes"
                )


            if not np.isfinite(
                s_cov
            ).all():

                raise RuntimeError(
                    "Non-finite covered score"
                )


            auroc = float(
                roc_auc_score(
                    y_cov,
                    s_cov,
                )
            )

            auprc = float(
                average_precision_score(
                    y_cov,
                    s_cov,
                )
            )


            cls = classification_metrics(
                y_cov,
                s_cov,
                threshold,
            )


            metric_scope = (
                "all_test_pairs"
                if n_missing == 0
                else "reported_pairs_only"
            )


            n_one_direction = 0

            if (
                method == "FastANI"
                and
                "fastani_pair_status"
                in df.columns
            ):

                n_one_direction = int(
                    (
                        df[
                            "fastani_pair_status"
                        ]
                        ==
                        "ONE_DIRECTION_ONLY"
                    ).sum()
                )


            row = {
                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "coverage_type":
                    spec[
                        "coverage_type"
                    ],

                "metric_scope":
                    metric_scope,

                "score_source":
                    score_source,

                "frozen_threshold":
                    threshold,

                "n_total":
                    n_total,

                "n_covered":
                    n_covered,

                "n_missing":
                    n_missing,

                "n_one_direction":
                    n_one_direction,

                "coverage":
                    coverage,

                "n_positive":
                    n_pos,

                "n_positive_covered":
                    n_pos_covered,

                "positive_coverage":
                    positive_coverage,

                "n_negative":
                    n_neg,

                "n_negative_covered":
                    n_neg_covered,

                "negative_coverage":
                    negative_coverage,

                "test_auroc_covered":
                    auroc,

                "test_auprc_covered":
                    auprc,

                "test_mcc_covered":
                    cls["mcc"],

                "test_f1_covered":
                    cls["f1"],

                "test_balanced_accuracy_covered":
                    cls[
                        "balanced_accuracy"
                    ],

                "test_sensitivity_covered":
                    cls[
                        "sensitivity"
                    ],

                "test_specificity_covered":
                    cls[
                        "specificity"
                    ],

                "tp":
                    cls["tp"],

                "fp":
                    cls["fp"],

                "tn":
                    cls["tn"],

                "fn":
                    cls["fn"],

                "score_min_covered":
                    float(
                        np.min(s_cov)
                    ),

                "score_median_covered":
                    float(
                        np.median(s_cov)
                    ),

                "score_mean_covered":
                    float(
                        np.mean(s_cov)
                    ),

                "score_max_covered":
                    float(
                        np.max(s_cov)
                    ),

                "test_file":
                    str(src),

                "test_file_sha256":
                    sha256_file(src),
            }


            split_rows.append(
                row
            )


            input_sha_rows.append({
                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "file":
                    str(src),

                "sha256":
                    sha256_file(src),
            })


            print(
                f"{protocol} seed{seed}: "
                f"coverage="
                f"{coverage:.6f} "
                f"pos="
                f"{positive_coverage:.6f} "
                f"neg="
                f"{negative_coverage:.6f} "
                f"AUROC="
                f"{auroc:.6f} "
                f"AUPRC="
                f"{auprc:.6f} "
                f"MCC="
                f"{cls['mcc']:.6f} "
                f"thr="
                f"{threshold:.10f}"
            )


split_df = pd.DataFrame(
    split_rows
)


if len(split_df) != 36:

    raise RuntimeError(
        f"Expected 36 test result rows, "
        f"found {len(split_df)}"
    )


# ================================================================
# Protocol-level mean +/- SD across the 3 taxonomy split seeds
# ================================================================

METRICS = [
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
]


summary_rows = []

for (
    method,
    protocol
), g in split_df.groupby(
    [
        "method",
        "protocol",
    ],
    sort=True,
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

        "protocol":
            protocol,

        "n_split_seeds":
            3,

        "coverage_type":
            g[
                "coverage_type"
            ].iloc[0],

        "metric_scope":
            g[
                "metric_scope"
            ].iloc[0],
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
            metric + "_sd"
        ] = float(
            np.std(
                values,
                ddof=1,
            )
        )


    row[
        "n_total_sum"
    ] = int(
        g["n_total"].sum()
    )

    row[
        "n_covered_sum"
    ] = int(
        g["n_covered"].sum()
    )

    row[
        "n_missing_sum"
    ] = int(
        g["n_missing"].sum()
    )


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ================================================================
# Long metric summary, useful for manuscript / plotting later
# ================================================================

long_rows = []

for r in summary_df.itertuples(
    index=False
):

    rd = r._asdict()

    for metric in METRICS:

        long_rows.append({
            "method":
                rd["method"],

            "protocol":
                rd["protocol"],

            "metric":
                metric,

            "mean":
                rd[
                    metric + "_mean"
                ],

            "sd":
                rd[
                    metric + "_sd"
                ],

            "n_split_seeds":
                3,

            "coverage_type":
                rd[
                    "coverage_type"
                ],

            "metric_scope":
                rd[
                    "metric_scope"
                ],
        })


long_df = pd.DataFrame(
    long_rows
)


# ================================================================
# Sanity gates
# ================================================================

complete = split_df[
    split_df["method"].isin(
        [
            "Mash",
            "Dashing2_SetSketch",
        ]
    )
]

if not np.allclose(
    complete["coverage"],
    1.0,
):
    raise RuntimeError(
        "Mash/Dashing test coverage "
        "not complete"
    )


ani = split_df[
    split_df["method"].isin(
        [
            "FastANI",
            "skani",
        ]
    )
]


all_ani_positive_complete = bool(
    np.allclose(
        ani[
            "positive_coverage"
        ],
        1.0,
    )
)


# ================================================================
# Write results
# ================================================================

split_file = (
    OUT /
    "native_test_metrics_split_level.tsv"
)

summary_file = (
    OUT /
    "native_test_metrics_protocol_summary.tsv"
)

long_file = (
    OUT /
    "native_test_metrics_protocol_summary_long.tsv"
)

input_sha_file = (
    OUT /
    "NATIVE_TEST_INPUT_SHA256.tsv"
)


split_df.to_csv(
    split_file,
    sep="\t",
    index=False,
)

summary_df.to_csv(
    summary_file,
    sep="\t",
    index=False,
)

long_df.to_csv(
    long_file,
    sep="\t",
    index=False,
)

pd.DataFrame(
    input_sha_rows
).to_csv(
    input_sha_file,
    sep="\t",
    index=False,
)


# ================================================================
# Report
# ================================================================

report = (
    OUT /
    "NATIVE_BLIND_TEST_EVALUATION_REPORT.txt"
)


with open(report, "w") as f:

    f.write(
        "E5 Native Blind-Test Evaluation v1\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )

    f.write(
        "Threshold source:\n"
        f"{THRESHOLD_FILE}\n\n"
    )

    f.write(
        "Thresholds were frozen on "
        "validation before this evaluator "
        "read blind-test results.\n\n"
    )

    f.write(
        "Mash/Dashing2 metrics use all "
        "test pairs because coverage is 100%.\n"
    )

    f.write(
        "FastANI/skani discrimination and "
        "threshold metrics use reported "
        "test pairs only; missing scores "
        "remain missing and coverage is "
        "reported separately.\n\n"
    )

    f.write(
        "All FastANI/skani positive "
        "test coverage = 1.0: "
        f"{all_ani_positive_complete}\n\n"
    )

    display_cols = [
        "method",
        "protocol",
        "coverage_mean",
        "coverage_sd",
        "positive_coverage_mean",
        "negative_coverage_mean",
        "test_auroc_covered_mean",
        "test_auroc_covered_sd",
        "test_auprc_covered_mean",
        "test_auprc_covered_sd",
        "test_mcc_covered_mean",
        "test_mcc_covered_sd",
        "test_f1_covered_mean",
        "test_balanced_accuracy_covered_mean",
        "metric_scope",
    ]

    f.write(
        summary_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: PASS\n"
    )


# ================================================================
# Freeze provenance
# ================================================================

files_to_hash = [
    THRESHOLD_FILE,
    split_file,
    summary_file,
    long_file,
    input_sha_file,
    report,
]


freeze_sha = (
    OUT /
    "RESULT_SHA256.txt"
)


with open(
    freeze_sha,
    "w",
) as f:

    for p in files_to_hash:

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


print()
print("=" * 100)
print(
    "NATIVE BLIND-TEST PROTOCOL SUMMARY"
)
print("=" * 100)

print(
    summary_df[
        [
            "method",
            "protocol",

            "coverage_mean",
            "coverage_sd",

            "positive_coverage_mean",
            "negative_coverage_mean",

            "test_auroc_covered_mean",
            "test_auroc_covered_sd",

            "test_auprc_covered_mean",
            "test_auprc_covered_sd",

            "test_mcc_covered_mean",
            "test_mcc_covered_sd",

            "test_f1_covered_mean",

            "test_balanced_accuracy_covered_mean",

            "metric_scope",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "All FastANI/skani positive "
    "test coverage=1.0 =",
    all_ani_positive_complete
)

print()
print("Output:")
print(OUT)

print()
print("STATUS = PASS")
