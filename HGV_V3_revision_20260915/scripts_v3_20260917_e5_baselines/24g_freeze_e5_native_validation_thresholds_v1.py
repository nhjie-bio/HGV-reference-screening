#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required in hgv_env"
    ) from e


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

OUT = (
    E5 /
    "00_native_validation/"
    "threshold_freeze_v1_20260917"
)


# ------------------------------------------------------------
# Never overwrite a previous freeze.
# ------------------------------------------------------------

if OUT.exists():
    raise RuntimeError(
        f"Freeze directory already exists:\n{OUT}\n"
        "Do not overwrite it. "
        "Use a new version directory if rerunning."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


EXPECTED = {
    ("G2", "20260915"): 3874,
    ("G2", "20260916"): 3872,
    ("G2", "20260917"): 3874,

    ("G3", "20260915"): 3764,
    ("G3", "20260916"): 3764,
    ("G3", "20260917"): 3830,

    ("G4", "20260915"): 3916,
    ("G4", "20260916"): 3786,
    ("G4", "20260917"): 3832,
}


METHODS = {
    "Mash": {
        "root":
            E5 /
            "02_mash/"
            "validation_cache_join_v1/"
            "results",

        "filename":
            "mash_validation_results.tsv",

        "score_column":
            "mash_similarity",

        "native_score":
            "Mash distance",

        "unified_score":
            "1 - Mash distance",

        "complete_expected":
            True,
    },

    "Dashing2_SetSketch": {
        "root":
            E5 /
            "03_dashing2/"
            "validation_cache_join_v1/"
            "results",

        "filename":
            "dashing2_validation_results.tsv",

        "score_column":
            "dashing2_similarity",

        "native_score":
            "Dashing2 SetSketch similarity",

        "unified_score":
            "native similarity",

        "complete_expected":
            True,
    },

    "FastANI": {
        "root":
            E5 /
            "05_fastani/"
            "validation_bidirectional_v1/"
            "results",

        "filename":
            "fastani_symmetric_fixed_pair_results.tsv",

        "score_column":
            "fastani_symmetric_similarity",

        "fallback_ani_column":
            "fastani_symmetric_ani",

        "native_score":
            "mean bidirectional FastANI ANI (%)",

        "unified_score":
            "symmetric ANI / 100",

        "complete_expected":
            False,
    },

    "skani": {
        "root":
            E5 /
            "04_skani/"
            "validation_v1/"
            "results",

        "filename":
            "skani_fixed_pair_results.tsv",

        "score_column":
            "skani_similarity",

        "fallback_ani_column":
            "skani_ani",

        "native_score":
            "skani ANI (%)",

        "unified_score":
            "ANI / 100",

        "complete_expected":
            False,
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


def confusion_metrics(
    tp,
    fp,
    tn,
    fn,
):

    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    p = tp + fn
    n = tn + fp

    tpr = (
        tp / p
        if p > 0
        else float("nan")
    )

    tnr = (
        tn / n
        if n > 0
        else float("nan")
    )

    bacc = (
        (tpr + tnr) / 2.0
    )

    f1_den = (
        2 * tp +
        fp +
        fn
    )

    f1 = (
        2 * tp / f1_den
        if f1_den > 0
        else 0.0
    )

    mcc_den = math.sqrt(
        (tp + fp) *
        (tp + fn) *
        (tn + fp) *
        (tn + fn)
    )

    if mcc_den > 0:
        mcc = (
            tp * tn -
            fp * fn
        ) / mcc_den
    else:
        mcc = 0.0

    return {
        "mcc":
            float(mcc),

        "balanced_accuracy":
            float(bacc),

        "f1":
            float(f1),

        "sensitivity":
            float(tpr),

        "specificity":
            float(tnr),

        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,
    }


def build_threshold_candidates(
    labels,
    scores,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(labels) != len(scores):
        raise RuntimeError(
            "label/score length mismatch"
        )

    if len(labels) == 0:
        raise RuntimeError(
            "No covered validation pairs"
        )

    if not np.isfinite(
        scores
    ).all():
        raise RuntimeError(
            "Non-finite score detected"
        )

    unique_labels = set(
        labels.tolist()
    )

    if unique_labels != {0, 1}:
        raise RuntimeError(
            "Covered validation subset "
            "must contain both classes"
        )

    df = pd.DataFrame({
        "score":
            scores,

        "label":
            labels,
    })

    grouped = (
        df.groupby(
            "score",
            sort=True,
        )["label"]
        .agg(
            count="count",
            positive="sum",
        )
        .sort_index(
            ascending=False
        )
    )

    p_total = int(
        labels.sum()
    )

    n_total = int(
        len(labels) -
        p_total
    )

    rows = []

    # Candidate immediately above maximum score:
    # predicts every pair negative.
    max_score = float(
        np.max(scores)
    )

    above_max = float(
        np.nextafter(
            max_score,
            np.inf,
        )
    )

    m = confusion_metrics(
        tp=0,
        fp=0,
        tn=n_total,
        fn=p_total,
    )

    rows.append({
        "threshold":
            above_max,

        "candidate_source":
            "above_max_sentinel",

        **m,
    })


    # For each observed score t:
    # positive iff score >= t.
    tp = 0
    fp = 0

    for threshold, r in (
        grouped.iterrows()
    ):

        count = int(
            r["count"]
        )

        positive = int(
            r["positive"]
        )

        negative = (
            count -
            positive
        )

        tp += positive
        fp += negative

        fn = (
            p_total -
            tp
        )

        tn = (
            n_total -
            fp
        )

        m = confusion_metrics(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

        rows.append({
            "threshold":
                float(threshold),

            "candidate_source":
                "observed_score",

            **m,
        })


    out = pd.DataFrame(
        rows
    )

    out[
        "distance_to_0p5"
    ] = (
        out["threshold"] -
        0.5
    ).abs()


    # Frozen tie-break:
    #
    # 1 max MCC
    # 2 max balanced accuracy
    # 3 max F1
    # 4 threshold closest to 0.5
    # 5 lower threshold
    #
    ranked = (
        out.sort_values(
            by=[
                "mcc",
                "balanced_accuracy",
                "f1",
                "distance_to_0p5",
                "threshold",
            ],

            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],

            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    ranked.insert(
        0,
        "selection_rank",
        np.arange(
            1,
            len(ranked) + 1
        ),
    )

    return ranked


def load_score(
    df,
    method,
    spec,
):

    score_col = (
        spec[
            "score_column"
        ]
    )

    if score_col in df.columns:

        score = pd.to_numeric(
            df[score_col],
            errors="coerce",
        )

        return (
            score_col,
            score,
        )


    fallback = spec.get(
        "fallback_ani_column"
    )

    if (
        fallback is not None
        and fallback in df.columns
    ):

        ani = pd.to_numeric(
            df[fallback],
            errors="coerce",
        )

        score = (
            ani / 100.0
        )

        return (
            f"{fallback}/100",
            score,
        )


    raise RuntimeError(
        f"{method}: neither "
        f"{score_col} nor fallback "
        "score column exists"
    )


print("=" * 96)
print(
    "E5 NATIVE VALIDATION "
    "THRESHOLD FREEZE v1"
)
print("=" * 96)

print()
print(
    "Threshold policy:"
)
print(
    "1) maximum validation MCC"
)
print(
    "2) tie -> maximum Balanced Accuracy"
)
print(
    "3) tie -> maximum F1"
)
print(
    "4) tie -> threshold closest to 0.5"
)
print(
    "5) tie -> lower threshold"
)

print()
print(
    "Prediction rule:"
)
print(
    "positive iff unified_similarity >= threshold"
)

print()
print(
    "Missing native ANI:"
)
print(
    "not imputed; excluded from threshold search; "
    "coverage frozen separately"
)


selected_rows = []
coverage_rows = []
candidate_frames = []
input_sha_rows = []


for method, spec in (
    METHODS.items()
):

    print()
    print("=" * 96)
    print(method)
    print("=" * 96)

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
                    f"expected {expected} rows, "
                    f"got {len(df)}"
                )


            if "label" not in df.columns:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "label column missing"
                )


            labels = pd.to_numeric(
                df["label"],
                errors="raise",
            ).astype(
                np.int64
            )


            if set(
                labels.unique()
            ) != {0, 1}:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "labels are not binary"
                )


            score_source_col, score = (
                load_score(
                    df,
                    method,
                    spec,
                )
            )


            covered = (
                score.notna()
            )

            if (
                score[covered] <
                0.0
            ).any():

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "unified similarity < 0"
                )

            if (
                score[covered] >
                1.0000001
            ).any():

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "unified similarity > 1"
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


            pos_mask = (
                labels == 1
            )

            neg_mask = (
                labels == 0
            )

            n_pos = int(
                pos_mask.sum()
            )

            n_neg = int(
                neg_mask.sum()
            )

            pos_covered = int(
                (
                    covered &
                    pos_mask
                ).sum()
            )

            neg_covered = int(
                (
                    covered &
                    neg_mask
                ).sum()
            )


            coverage = (
                n_covered /
                n_total
            )

            pos_coverage = (
                pos_covered /
                n_pos
            )

            neg_coverage = (
                neg_covered /
                n_neg
            )


            if (
                spec[
                    "complete_expected"
                ]
                and n_missing != 0
            ):

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "unexpected missing score"
                )


            labels_cov = (
                labels[
                    covered
                ]
                .to_numpy(
                    dtype=np.int64
                )
            )

            scores_cov = (
                score[
                    covered
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )


            if set(
                np.unique(
                    labels_cov
                ).tolist()
            ) != {0, 1}:

                raise RuntimeError(
                    f"{method} "
                    f"{protocol} seed{seed}: "
                    "covered subset lacks "
                    "one class"
                )


            val_auroc = float(
                roc_auc_score(
                    labels_cov,
                    scores_cov,
                )
            )

            val_auprc = float(
                average_precision_score(
                    labels_cov,
                    scores_cov,
                )
            )


            candidates = (
                build_threshold_candidates(
                    labels_cov,
                    scores_cov,
                )
            )


            best = (
                candidates.iloc[0]
            )


            metric_scope = (
                "all_validation_pairs"
                if n_missing == 0
                else "reported_pairs_only"
            )


            selected_rows.append({
                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "score_source_column":
                    score_source_col,

                "native_score":
                    spec[
                        "native_score"
                    ],

                "unified_score":
                    spec[
                        "unified_score"
                    ],

                "prediction_rule":
                    "score >= threshold",

                "metric_scope":
                    metric_scope,

                "missing_policy":
                    "not_imputed",

                "n_total":
                    n_total,

                "n_covered":
                    n_covered,

                "n_missing":
                    n_missing,

                "coverage":
                    coverage,

                "n_positive_total":
                    n_pos,

                "n_positive_covered":
                    pos_covered,

                "positive_coverage":
                    pos_coverage,

                "n_negative_total":
                    n_neg,

                "n_negative_covered":
                    neg_covered,

                "negative_coverage":
                    neg_coverage,

                "val_auroc_covered":
                    val_auroc,

                "val_auprc_covered":
                    val_auprc,

                "frozen_threshold":
                    float(
                        best[
                            "threshold"
                        ]
                    ),

                "val_mcc_covered":
                    float(
                        best["mcc"]
                    ),

                "val_balanced_accuracy_covered":
                    float(
                        best[
                            "balanced_accuracy"
                        ]
                    ),

                "val_f1_covered":
                    float(
                        best["f1"]
                    ),

                "val_sensitivity_covered":
                    float(
                        best[
                            "sensitivity"
                        ]
                    ),

                "val_specificity_covered":
                    float(
                        best[
                            "specificity"
                        ]
                    ),

                "tp":
                    int(
                        best["tp"]
                    ),

                "fp":
                    int(
                        best["fp"]
                    ),

                "tn":
                    int(
                        best["tn"]
                    ),

                "fn":
                    int(
                        best["fn"]
                    ),

                "threshold_candidate_source":
                    best[
                        "candidate_source"
                    ],

                "n_threshold_candidates":
                    len(candidates),

                "validation_file":
                    str(src),

                "validation_file_sha256":
                    sha256_file(src),
            })


            coverage_rows.append({
                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "n_total":
                    n_total,

                "n_covered":
                    n_covered,

                "n_missing":
                    n_missing,

                "coverage":
                    coverage,

                "positive_coverage":
                    pos_coverage,

                "negative_coverage":
                    neg_coverage,

                "metric_scope":
                    metric_scope,
            })


            c = (
                candidates.copy()
            )

            c.insert(
                0,
                "method",
                method,
            )

            c.insert(
                1,
                "protocol",
                protocol,
            )

            c.insert(
                2,
                "split_seed",
                seed,
            )

            candidate_frames.append(
                c
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
                f"{pos_coverage:.6f} "
                f"neg="
                f"{neg_coverage:.6f} "
                f"AUROC="
                f"{val_auroc:.6f} "
                f"AUPRC="
                f"{val_auprc:.6f} "
                f"thr="
                f"{float(best['threshold']):.10f} "
                f"MCC="
                f"{float(best['mcc']):.6f}"
            )


selected = pd.DataFrame(
    selected_rows
)

coverage_df = pd.DataFrame(
    coverage_rows
)

input_sha_df = pd.DataFrame(
    input_sha_rows
)

candidate_df = pd.concat(
    candidate_frames,
    ignore_index=True,
)


# ------------------------------------------------------------
# Global gates.
# ------------------------------------------------------------

if len(selected) != 36:

    raise RuntimeError(
        f"Expected 36 frozen thresholds, "
        f"got {len(selected)}"
    )


if selected[
    [
        "method",
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate method/protocol/seed "
        "threshold row"
    )


for method in [
    "Mash",
    "Dashing2_SetSketch",
]:

    x = selected[
        selected["method"] ==
        method
    ]

    if not np.allclose(
        x["coverage"],
        1.0,
    ):
        raise RuntimeError(
            f"{method} coverage is "
            "not exactly complete"
        )


# Positive coverage for native ANI is audited,
# but this is an empirical result rather than
# an imputation rule.

fast_skani = selected[
    selected["method"].isin(
        [
            "FastANI",
            "skani",
        ]
    )
]

all_native_positive_complete = bool(
    np.allclose(
        fast_skani[
            "positive_coverage"
        ],
        1.0,
    )
)


# ------------------------------------------------------------
# Save frozen outputs.
# ------------------------------------------------------------

threshold_file = (
    OUT /
    "FROZEN_NATIVE_VALIDATION_THRESHOLDS.tsv"
)

coverage_file = (
    OUT /
    "NATIVE_VALIDATION_COVERAGE.tsv"
)

candidate_file = (
    OUT /
    "NATIVE_VALIDATION_THRESHOLD_SEARCH_CANDIDATES.tsv.gz"
)

input_sha_file = (
    OUT /
    "INPUT_VALIDATION_SHA256.tsv"
)


selected.to_csv(
    threshold_file,
    sep="\t",
    index=False,
)

coverage_df.to_csv(
    coverage_file,
    sep="\t",
    index=False,
)

candidate_df.to_csv(
    candidate_file,
    sep="\t",
    index=False,
    compression="gzip",
)

input_sha_df.to_csv(
    input_sha_file,
    sep="\t",
    index=False,
)


policy_file = (
    OUT /
    "THRESHOLD_POLICY.txt"
)

policy_file.write_text(
    "E5 native validation threshold policy\n"
    "=====================================\n\n"
    "Selection data: validation only.\n"
    "Blind test is not read by this script.\n\n"
    "Unified prediction rule:\n"
    "  positive iff similarity_score >= threshold\n\n"
    "Selection rule:\n"
    "  1. maximum validation MCC\n"
    "  2. tie -> maximum Balanced Accuracy\n"
    "  3. tie -> maximum F1\n"
    "  4. tie -> threshold closest to 0.5\n"
    "  5. tie -> lower threshold\n\n"
    "Candidate thresholds:\n"
    "  every unique observed covered validation score\n"
    "  plus one finite threshold immediately above max score.\n\n"
    "Missing-score policy:\n"
    "  no imputation.\n"
    "  FastANI/skani threshold metrics use reported pairs only.\n"
    "  Coverage is frozen and must be reported beside metrics.\n\n"
    "Mash score = 1 - Mash distance.\n"
    "Dashing2 score = native SetSketch similarity.\n"
    "FastANI score = mean bidirectional ANI / 100, "
    "only when both directions report.\n"
    "skani score = ANI / 100 when reported.\n"
)


report_file = (
    OUT /
    "NATIVE_VALIDATION_THRESHOLD_FREEZE_REPORT.txt"
)


with open(
    report_file,
    "w",
) as f:

    f.write(
        "E5 Native Validation Threshold Freeze v1\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )

    f.write(
        "Frozen thresholds: 36 "
        "(4 methods x 3 protocols x 3 split seeds)\n\n"
    )

    f.write(
        "No test file was used for threshold selection.\n"
    )

    f.write(
        "Missing FastANI/skani scores were not imputed.\n"
    )

    f.write(
        "FastANI/skani threshold metrics are "
        "reported-pair metrics.\n\n"
    )

    f.write(
        "All FastANI/skani positive validation "
        "coverage = 1.0: "
        f"{all_native_positive_complete}\n\n"
    )

    f.write(
        selected[
            [
                "method",
                "protocol",
                "split_seed",
                "coverage",
                "positive_coverage",
                "negative_coverage",
                "val_auroc_covered",
                "val_auprc_covered",
                "frozen_threshold",
                "val_mcc_covered",
                "val_balanced_accuracy_covered",
                "val_f1_covered",
                "metric_scope",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: FROZEN\n"
    )


# ------------------------------------------------------------
# Freeze SHA256.
# ------------------------------------------------------------

freeze_files = [
    threshold_file,
    coverage_file,
    candidate_file,
    input_sha_file,
    policy_file,
    report_file,
]


sha_out = (
    OUT /
    "FREEZE_SHA256.txt"
)

with open(
    sha_out,
    "w",
) as f:

    for p in freeze_files:

        f.write(
            f"{sha256_file(p)}  {p}\n"
        )


# Script provenance.
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
        f"script_sha256="
        f"{sha256_file(script_path)}\n"
    )

    f.write(
        f"python={sys.version}\n"
    )


print()
print("=" * 96)
print(
    "FROZEN NATIVE VALIDATION THRESHOLDS"
)
print("=" * 96)

print(
    selected[
        [
            "method",
            "protocol",
            "split_seed",
            "coverage",
            "positive_coverage",
            "negative_coverage",
            "frozen_threshold",
            "val_mcc_covered",
            "val_balanced_accuracy_covered",
            "val_f1_covered",
            "metric_scope",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "n frozen thresholds =",
    len(selected)
)

print(
    "FastANI/skani all "
    "positive coverage=1.0 =",
    all_native_positive_complete
)

print()
print(
    "Freeze directory:"
)
print(OUT)

print()
print(
    "Threshold file:"
)
print(threshold_file)

print()
print("STATUS = FROZEN")
