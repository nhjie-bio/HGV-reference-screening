#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
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

PAIR_ROOT = (
    BASE /
    "05_pairs/"
    "pair_manifests_v1_20260916"
)

EMB_DIR = (
    E5 /
    "06_dnaberts/"
    "07_production_embeddings_v1_20260918"
)

EMB_FREEZE = (
    E5 /
    "06_dnaberts/"
    "08_production_embeddings_freeze_v1_20260918/"
    "DNABERTS_EMBEDDING_FREEZE.txt"
)

EMB_FILE = (
    EMB_DIR /
    "DNABERTS_GENOME_EMBEDDINGS_L2_FLOAT32.npy"
)

EMB_MANIFEST = (
    EMB_DIR /
    "DNABERTS_GENOME_EMBEDDING_MANIFEST.tsv"
)

OUT = (
    E5 /
    "06_dnaberts/"
    "09_validation_threshold_freeze_v1_20260918"
)


if OUT.exists():
    raise RuntimeError(
        f"Output already exists:\n{OUT}\n"
        "Do not overwrite a threshold-freeze directory."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
)


EXPECTED_VAL = {
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


PROTOCOLS = [
    "G2",
    "G3",
    "G4",
]

SEEDS = [
    "20260915",
    "20260916",
    "20260917",
]


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


def binary_metrics_from_counts(
    tp,
    fp,
    tn,
    fn,
):

    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

    n_pos = tp + fn
    n_neg = tn + fp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / n_pos
        if n_pos > 0
        else 0.0
    )

    specificity = (
        tn / n_neg
        if n_neg > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    f1 = (
        2.0 * tp
        /
        (2.0 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0.0
    )

    denom = math.sqrt(
        (tp + fp)
        *
        (tp + fn)
        *
        (tn + fp)
        *
        (tn + fn)
    )

    mcc = (
        ((tp * tn) - (fp * fn))
        / denom
        if denom > 0
        else 0.0
    )

    return {
        "mcc":
            float(mcc),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "f1":
            float(f1),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "specificity":
            float(specificity),

        "tp":
            tp,

        "fp":
            fp,

        "tn":
            tn,

        "fn":
            fn,
    }


def auroc_rank(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=np.int8,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    n_pos = int(
        np.sum(
            y == 1
        )
    )

    n_neg = int(
        np.sum(
            y == 0
        )
    )

    if n_pos == 0 or n_neg == 0:

        return float("nan")


    # Average ranks for ties.
    order = np.argsort(
        score,
        kind="mergesort",
    )

    sorted_score = score[
        order
    ]

    ranks = np.empty(
        len(score),
        dtype=np.float64,
    )

    i = 0

    while i < len(score):

        j = i + 1

        while (
            j < len(score)
            and
            sorted_score[j]
            ==
            sorted_score[i]
        ):
            j += 1

        # 1-based average rank for positions
        # i ... j-1.
        average_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = average_rank

        i = j


    rank_sum_pos = float(
        ranks[
            y == 1
        ].sum()
    )

    auc = (
        rank_sum_pos
        -
        (
            n_pos *
            (n_pos + 1)
            /
            2.0
        )
    ) / (
        n_pos *
        n_neg
    )

    return float(
        auc
    )


def average_precision_grouped(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=np.int8,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    n_pos = int(
        np.sum(
            y == 1
        )
    )

    if n_pos == 0:
        return float("nan")


    order = np.argsort(
        -score,
        kind="mergesort",
    )

    ys = y[
        order
    ]

    ss = score[
        order
    ]


    tp = 0
    fp = 0

    previous_recall = 0.0
    ap = 0.0

    i = 0

    while i < len(
        ss
    ):

        j = i + 1

        while (
            j < len(ss)
            and
            ss[j] == ss[i]
        ):

            j += 1


        group_y = ys[
            i:j
        ]

        tp += int(
            np.sum(
                group_y == 1
            )
        )

        fp += int(
            np.sum(
                group_y == 0
            )
        )


        recall = (
            tp /
            n_pos
        )

        precision = (
            tp /
            (tp + fp)
        )


        ap += (
            recall -
            previous_recall
        ) * precision


        previous_recall = (
            recall
        )

        i = j


    return float(
        ap
    )


def build_threshold_search(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=np.int8,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )


    n_pos = int(
        np.sum(
            y == 1
        )
    )

    n_neg = int(
        np.sum(
            y == 0
        )
    )


    if (
        n_pos == 0
        or
        n_neg == 0
    ):

        raise RuntimeError(
            "Threshold search requires "
            "both classes"
        )


    order = np.argsort(
        -score,
        kind="mergesort",
    )

    ys = y[
        order
    ]

    ss = score[
        order
    ]


    rows = []


    # Sentinel above maximum:
    # positive iff score >= threshold
    # => all negative.
    sentinel = np.nextafter(
        float(
            ss[0]
        ),
        np.inf,
    )


    metrics = (
        binary_metrics_from_counts(
            tp=0,
            fp=0,
            tn=n_neg,
            fn=n_pos,
        )
    )


    rows.append({
        "threshold":
            sentinel,

        "threshold_source":
            "nextafter_max_plus_inf",

        "threshold_distance_to_0p5":
            abs(
                sentinel -
                0.5
            ),

        **metrics,
    })


    tp = 0
    fp = 0

    i = 0


    while i < len(
        ss
    ):

        threshold = float(
            ss[i]
        )

        j = i + 1

        while (
            j < len(ss)
            and
            ss[j] == ss[i]
        ):

            j += 1


        group_y = ys[
            i:j
        ]


        tp += int(
            np.sum(
                group_y == 1
            )
        )

        fp += int(
            np.sum(
                group_y == 0
            )
        )


        fn = (
            n_pos -
            tp
        )

        tn = (
            n_neg -
            fp
        )


        metrics = (
            binary_metrics_from_counts(
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )


        rows.append({
            "threshold":
                threshold,

            "threshold_source":
                "observed_validation_score",

            "threshold_distance_to_0p5":
                abs(
                    threshold -
                    0.5
                ),

            **metrics,
        })


        i = j


    search = pd.DataFrame(
        rows
    )


    ranked = search.sort_values(
        [
            "mcc",
            "balanced_accuracy",
            "f1",
            "threshold_distance_to_0p5",
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


    best = ranked.iloc[
        0
    ]


    return (
        search,
        best,
    )


print("=" * 100)
print(
    "DNABERT-S VALIDATION SCORING "
    "+ THRESHOLD FREEZE v1"
)
print("=" * 100)

print()
print(
    "IMPORTANT:"
)

print(
    "This script reads VALIDATION pairs only."
)

print(
    "No test-pair file is read."
)

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
    "positive iff DNABERT-S cosine >= threshold"
)


# ================================================================
# Load embedding matrix + accession map.
# ================================================================

embedding = np.load(
    EMB_FILE,
    mmap_mode="r",
)


if (
    embedding.shape
    !=
    (
        3154,
        768,
    )
):

    raise RuntimeError(
        f"Unexpected embedding shape: "
        f"{embedding.shape}"
    )


if (
    embedding.dtype
    !=
    np.float32
):

    raise RuntimeError(
        f"Unexpected embedding dtype: "
        f"{embedding.dtype}"
    )


manifest = pd.read_csv(
    EMB_MANIFEST,
    sep="\t",
    dtype={
        "assembly_accession":
            str,
    },
)


required_manifest = [
    "embedding_index",
    "assembly_accession",
    "normalized_l2_norm",
]


for c in required_manifest:

    if c not in manifest.columns:

        raise RuntimeError(
            f"Embedding manifest "
            f"missing {c}"
        )


if len(
    manifest
) != 3154:

    raise RuntimeError(
        "Embedding manifest "
        "row count !=3154"
    )


if manifest[
    "assembly_accession"
].duplicated().any():

    raise RuntimeError(
        "Duplicate accession "
        "in embedding manifest"
    )


indices = pd.to_numeric(
    manifest[
        "embedding_index"
    ],
    errors="raise",
).astype(int)


if indices.tolist() != list(
    range(
        3154
    )
):

    raise RuntimeError(
        "Embedding indices are not "
        "exactly 0..3153"
    )


index_by_accession = dict(
    zip(
        manifest[
            "assembly_accession"
        ].astype(str),
        indices,
    )
)


norms = np.linalg.norm(
    np.asarray(
        embedding,
        dtype=np.float32,
    ),
    axis=1,
)


if not np.allclose(
    norms,
    1.0,
    rtol=1e-5,
    atol=1e-6,
):

    raise RuntimeError(
        "Stored L2 embeddings "
        "do not have unit norm"
    )


print()
print(
    "Embedding matrix:",
    embedding.shape,
    embedding.dtype
)

print(
    "Embedding accession map:",
    len(
        index_by_accession
    )
)

print(
    "L2 norm check: PASS"
)


# ================================================================
# Score validation pairs.
# ================================================================

threshold_rows = []
validation_summary_rows = []
search_frames = []
input_hash_rows = []
scored_files = []


for protocol in PROTOCOLS:

    print()
    print("=" * 100)
    print(protocol)
    print("=" * 100)


    for seed in SEEDS:

        expected = EXPECTED_VAL[
            (
                protocol,
                seed,
            )
        ]


        pair_file = (
            PAIR_ROOT /
            f"{protocol}_seed{seed}" /
            "val_pairs.tsv"
        )


        if not pair_file.is_file():

            raise FileNotFoundError(
                pair_file
            )


        # Explicitly ensure this is validation,
        # never test.
        if pair_file.name != (
            "val_pairs.tsv"
        ):

            raise RuntimeError(
                "Non-validation pair file "
                "was requested"
            )


        pairs = pd.read_csv(
            pair_file,
            sep="\t",
            dtype={
                "pair_id":
                    str,

                "protocol":
                    str,

                "split_seed":
                    str,

                "role":
                    str,

                "accession_a":
                    str,

                "accession_b":
                    str,
            },
        )


        if len(
            pairs
        ) != expected:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"expected {expected}, "
                f"found {len(pairs)}"
            )


        required_pair_cols = [
            "pair_id",
            "protocol",
            "split_seed",
            "role",
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
        ]


        for c in required_pair_cols:

            if c not in pairs.columns:

                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    f"missing column {c}"
                )


        if pairs[
            "pair_id"
        ].duplicated().any():

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "duplicate pair_id"
            )


        if set(
            pairs[
                "protocol"
            ].astype(str)
        ) != {
            protocol
        }:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "protocol field mismatch"
            )


        if set(
            pairs[
                "split_seed"
            ].astype(str)
        ) != {
            seed
        }:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "split_seed field mismatch"
            )


        if set(
            pairs[
                "role"
            ].astype(str)
        ) != {
            "val"
        }:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "role is not exclusively val"
            )


        y = pd.to_numeric(
            pairs[
                "label"
            ],
            errors="raise",
        ).astype(
            np.int8
        ).to_numpy()


        if set(
            np.unique(
                y
            ).tolist()
        ) != {
            0,
            1,
        }:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "non-binary labels"
            )


        n_pos = int(
            np.sum(
                y == 1
            )
        )

        n_neg = int(
            np.sum(
                y == 0
            )
        )


        if n_pos != n_neg:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"validation is not balanced "
                f"(pos={n_pos}, neg={n_neg})"
            )


        expected_positive_type = (
            pairs[
                "pair_type"
            ].astype(str)
            ==
            "positive"
        ).to_numpy()


        if not np.array_equal(
            expected_positive_type,
            y == 1,
        ):

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "label/pair_type mismatch"
            )


        if (
            pairs[
                "accession_a"
            ].astype(str)
            ==
            pairs[
                "accession_b"
            ].astype(str)
        ).any():

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "self pair found"
            )


        missing_a = sorted(
            set(
                pairs[
                    "accession_a"
                ].astype(str)
            )
            -
            set(
                index_by_accession
            )
        )


        missing_b = sorted(
            set(
                pairs[
                    "accession_b"
                ].astype(str)
            )
            -
            set(
                index_by_accession
            )
        )


        if missing_a or missing_b:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"missing embedding accessions; "
                f"A={missing_a[:5]}, "
                f"B={missing_b[:5]}"
            )


        idx_a = (
            pairs[
                "accession_a"
            ]
            .astype(str)
            .map(
                index_by_accession
            )
            .to_numpy(
                dtype=np.int64
            )
        )


        idx_b = (
            pairs[
                "accession_b"
            ]
            .astype(str)
            .map(
                index_by_accession
            )
            .to_numpy(
                dtype=np.int64
            )
        )


        # Frozen L2 vectors:
        # cosine similarity = dot product.
        #
        # Accumulate in float64 to reduce
        # summation error while preserving
        # the frozen float32 vector values.
        emb_a = np.asarray(
            embedding[
                idx_a
            ],
            dtype=np.float64,
        )

        emb_b = np.asarray(
            embedding[
                idx_b
            ],
            dtype=np.float64,
        )


        score = np.sum(
            emb_a *
            emb_b,
            axis=1,
            dtype=np.float64,
        )


        if not np.isfinite(
            score
        ).all():

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "non-finite cosine score"
            )


        # Unit vectors should stay within tiny
        # floating tolerance around [-1, 1].
        if (
            score.min()
            <
            -1.00001
            or
            score.max()
            >
            1.00001
        ):

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"cosine out of range "
                f"[{score.min()}, {score.max()}]"
            )


        auroc = (
            auroc_rank(
                y,
                score,
            )
        )

        auprc = (
            average_precision_grouped(
                y,
                score,
            )
        )


        search, best = (
            build_threshold_search(
                y,
                score,
            )
        )


        threshold = float(
            best[
                "threshold"
            ]
        )


        search.insert(
            0,
            "method",
            "DNABERT-S"
        )

        search.insert(
            1,
            "protocol",
            protocol
        )

        search.insert(
            2,
            "split_seed",
            seed
        )


        search_frames.append(
            search
        )


        predicted = (
            score >= threshold
        ).astype(
            np.int8
        )


        tp = int(
            np.sum(
                (y == 1)
                &
                (predicted == 1)
            )
        )

        fp = int(
            np.sum(
                (y == 0)
                &
                (predicted == 1)
            )
        )

        tn = int(
            np.sum(
                (y == 0)
                &
                (predicted == 0)
            )
        )

        fn = int(
            np.sum(
                (y == 1)
                &
                (predicted == 0)
            )
        )


        final_metrics = (
            binary_metrics_from_counts(
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )


        # Ensure selected threshold metrics match
        # the recorded search row.
        for metric in [
            "mcc",
            "balanced_accuracy",
            "f1",
        ]:

            if not np.isclose(
                final_metrics[
                    metric
                ],
                float(
                    best[
                        metric
                    ]
                ),
                rtol=1e-12,
                atol=1e-12,
            ):

                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    "threshold metric "
                    "reproduction failed"
                )


        scored = pairs.copy()

        scored[
            "dnaberts_embedding_index_a"
        ] = idx_a

        scored[
            "dnaberts_embedding_index_b"
        ] = idx_b

        scored[
            "dnaberts_cosine_similarity"
        ] = score


        split_out = (
            OUT /
            f"{protocol}_seed{seed}"
        )

        split_out.mkdir(
            parents=True,
            exist_ok=False,
        )


        scored_file = (
            split_out /
            "dnaberts_validation_scores.tsv"
        )


        scored.to_csv(
            scored_file,
            sep="\t",
            index=False,
        )


        scored_files.append(
            scored_file
        )


        threshold_rows.append({
            "method":
                "DNABERT-S",

            "protocol":
                protocol,

            "split_seed":
                seed,

            "role":
                "validation",

            "coverage":
                1.0,

            "positive_coverage":
                1.0,

            "negative_coverage":
                1.0,

            "metric_scope":
                "all_validation_pairs",

            "n_pairs":
                len(
                    pairs
                ),

            "n_positive":
                n_pos,

            "n_negative":
                n_neg,

            "frozen_threshold":
                threshold,

            "validation_auroc":
                auroc,

            "validation_auprc":
                auprc,

            "validation_mcc":
                final_metrics[
                    "mcc"
                ],

            "validation_balanced_accuracy":
                final_metrics[
                    "balanced_accuracy"
                ],

            "validation_f1":
                final_metrics[
                    "f1"
                ],

            "validation_precision":
                final_metrics[
                    "precision"
                ],

            "validation_recall":
                final_metrics[
                    "recall"
                ],

            "validation_specificity":
                final_metrics[
                    "specificity"
                ],

            "tp":
                final_metrics[
                    "tp"
                ],

            "fp":
                final_metrics[
                    "fp"
                ],

            "tn":
                final_metrics[
                    "tn"
                ],

            "fn":
                final_metrics[
                    "fn"
                ],

            "score_min":
                float(
                    score.min()
                ),

            "score_median":
                float(
                    np.median(
                        score
                    )
                ),

            "score_mean":
                float(
                    score.mean()
                ),

            "score_max":
                float(
                    score.max()
                ),

            "n_unique_scores":
                int(
                    np.unique(
                        score
                    ).size
                ),
        })


        validation_summary_rows.append({
            "protocol":
                protocol,

            "split_seed":
                seed,

            "n_pairs":
                len(
                    pairs
                ),

            "n_positive":
                n_pos,

            "n_negative":
                n_neg,

            "auroc":
                auroc,

            "auprc":
                auprc,

            "frozen_threshold":
                threshold,

            "mcc":
                final_metrics[
                    "mcc"
                ],

            "balanced_accuracy":
                final_metrics[
                    "balanced_accuracy"
                ],

            "f1":
                final_metrics[
                    "f1"
                ],

            "score_positive_mean":
                float(
                    score[
                        y == 1
                    ].mean()
                ),

            "score_positive_median":
                float(
                    np.median(
                        score[
                            y == 1
                        ]
                    )
                ),

            "score_negative_mean":
                float(
                    score[
                        y == 0
                    ].mean()
                ),

            "score_negative_median":
                float(
                    np.median(
                        score[
                            y == 0
                        ]
                    )
                ),
        })


        input_hash_rows.append({
            "input_type":
                "validation_pair_manifest",

            "protocol":
                protocol,

            "split_seed":
                seed,

            "path":
                str(
                    pair_file
                ),

            "sha256":
                sha256_file(
                    pair_file
                ),
        })


        print(
            f"{protocol} seed{seed}: "
            f"n={len(pairs)} "
            f"AUROC={auroc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"thr={threshold:.10f} "
            f"MCC="
            f"{final_metrics['mcc']:.6f} "
            f"BAcc="
            f"{final_metrics['balanced_accuracy']:.6f} "
            f"F1="
            f"{final_metrics['f1']:.6f}"
        )


# ================================================================
# Final frozen tables.
# ================================================================

threshold_df = pd.DataFrame(
    threshold_rows
)


if len(
    threshold_df
) != 9:

    raise RuntimeError(
        f"Expected 9 frozen thresholds, "
        f"found {len(threshold_df)}"
    )


if threshold_df[
    [
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate threshold key"
    )


summary_df = pd.DataFrame(
    validation_summary_rows
)


search_df = pd.concat(
    search_frames,
    ignore_index=True,
)


THRESHOLD_FILE = (
    OUT /
    "FROZEN_DNABERTS_VALIDATION_THRESHOLDS.tsv"
)

SUMMARY_FILE = (
    OUT /
    "DNABERTS_VALIDATION_SUMMARY.tsv"
)

SEARCH_FILE = (
    OUT /
    "DNABERTS_VALIDATION_THRESHOLD_SEARCH.tsv"
)


threshold_df.to_csv(
    THRESHOLD_FILE,
    sep="\t",
    index=False,
)

summary_df.to_csv(
    SUMMARY_FILE,
    sep="\t",
    index=False,
)

search_df.to_csv(
    SEARCH_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Threshold policy.
# ================================================================

POLICY_FILE = (
    OUT /
    "THRESHOLD_POLICY.txt"
)


POLICY_FILE.write_text(
    """DNABERT-S E5 validation threshold policy

Score:
cosine similarity between frozen DNABERT-S genome embeddings.

Prediction:
positive iff cosine_similarity >= threshold.

Threshold source:
validation pairs only.

Tie-breaking:
1. maximum validation MCC
2. maximum Balanced Accuracy
3. maximum F1
4. threshold closest to 0.5
5. lower threshold

Threshold candidates:
- every unique observed validation score
- one nextafter(max_score, +inf) sentinel

Coverage:
100% for all validation pairs.

Test access:
No test-pair manifest is read by this script.
No test score is used for threshold selection.

Status:
FROZEN
"""
)


# ================================================================
# Input provenance.
# ================================================================

input_hash_rows.extend([
    {
        "input_type":
            "embedding_l2_matrix",

        "protocol":
            "",

        "split_seed":
            "",

        "path":
            str(
                EMB_FILE
            ),

        "sha256":
            sha256_file(
                EMB_FILE
            ),
    },

    {
        "input_type":
            "embedding_manifest",

        "protocol":
            "",

        "split_seed":
            "",

        "path":
            str(
                EMB_MANIFEST
            ),

        "sha256":
            sha256_file(
                EMB_MANIFEST
            ),
    },

    {
        "input_type":
            "embedding_freeze",

        "protocol":
            "",

        "split_seed":
            "",

        "path":
            str(
                EMB_FREEZE
            ),

        "sha256":
            sha256_file(
                EMB_FREEZE
            ),
    },
])


INPUT_SHA_FILE = (
    OUT /
    "INPUT_SHA256.tsv"
)


pd.DataFrame(
    input_hash_rows
).to_csv(
    INPUT_SHA_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Report.
# ================================================================

REPORT = (
    OUT /
    "DNABERTS_VALIDATION_THRESHOLD_FREEZE_REPORT.txt"
)


with open(
    REPORT,
    "w",
) as f:

    f.write(
        "DNABERT-S Validation Threshold Freeze v1\n"
    )

    f.write(
        "=" * 84 +
        "\n\n"
    )

    f.write(
        "No blind-test pair manifest was read.\n\n"
    )

    f.write(
        "Score: cosine similarity between "
        "frozen L2-normalized 768-D "
        "DNABERT-S genome embeddings.\n\n"
    )

    f.write(
        "Threshold policy:\n"
        "1. max validation MCC\n"
        "2. max Balanced Accuracy\n"
        "3. max F1\n"
        "4. closest threshold to 0.5\n"
        "5. lower threshold\n\n"
    )

    f.write(
        threshold_df[
            [
                "protocol",
                "split_seed",
                "n_pairs",
                "coverage",
                "validation_auroc",
                "validation_auprc",
                "frozen_threshold",
                "validation_mcc",
                "validation_balanced_accuracy",
                "validation_f1",
                "score_min",
                "score_median",
                "score_max",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nn frozen thresholds = "
        f"{len(threshold_df)}\n"
    )

    f.write(
        "validation coverage = 1.0 "
        "for all splits\n"
    )

    f.write(
        "STATUS: FROZEN\n"
    )


# ================================================================
# Script provenance + freeze hashes.
# ================================================================

script_path = Path(
    __file__
).resolve()


SCRIPT_PROVENANCE = (
    OUT /
    "SCRIPT_PROVENANCE.txt"
)


SCRIPT_PROVENANCE.write_text(
    f"script={script_path}\n"
    f"script_sha256={sha256_file(script_path)}\n"
    f"python={sys.version}\n"
    f"numpy={np.__version__}\n"
    f"pandas={pd.__version__}\n"
)


files_to_hash = [
    THRESHOLD_FILE,
    SUMMARY_FILE,
    SEARCH_FILE,
    POLICY_FILE,
    INPUT_SHA_FILE,
    REPORT,
    SCRIPT_PROVENANCE,
] + scored_files


FREEZE_SHA = (
    OUT /
    "FREEZE_SHA256.txt"
)


with open(
    FREEZE_SHA,
    "w",
) as f:

    for p in files_to_hash:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ================================================================
# Console.
# ================================================================

print()
print("=" * 100)
print(
    "FROZEN DNABERT-S VALIDATION THRESHOLDS"
)
print("=" * 100)

print(
    threshold_df[
        [
            "protocol",
            "split_seed",
            "n_pairs",
            "validation_auroc",
            "validation_auprc",
            "frozen_threshold",
            "validation_mcc",
            "validation_balanced_accuracy",
            "validation_f1",
            "score_min",
            "score_median",
            "score_max",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "n frozen thresholds =",
    len(
        threshold_df
    )
)

print(
    "all validation coverage=1.0 =",
    bool(
        np.allclose(
            threshold_df[
                "coverage"
            ],
            1.0,
        )
    )
)

print()
print(
    "Threshold file:"
)

print(
    THRESHOLD_FILE
)

print()
print(
    "STATUS = FROZEN"
)
