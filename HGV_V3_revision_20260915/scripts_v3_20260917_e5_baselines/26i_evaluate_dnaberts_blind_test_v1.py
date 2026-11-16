#!/usr/bin/env python3

from pathlib import Path
import hashlib
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

EMB_FILE = (
    EMB_DIR /
    "DNABERTS_GENOME_EMBEDDINGS_L2_FLOAT32.npy"
)

EMB_MANIFEST = (
    EMB_DIR /
    "DNABERTS_GENOME_EMBEDDING_MANIFEST.tsv"
)

THRESHOLD_DIR = (
    E5 /
    "06_dnaberts/"
    "09_validation_threshold_freeze_v1_20260918"
)

THRESHOLD_FILE = (
    THRESHOLD_DIR /
    "FROZEN_DNABERTS_VALIDATION_THRESHOLDS.tsv"
)

THRESHOLD_POLICY = (
    THRESHOLD_DIR /
    "THRESHOLD_POLICY.txt"
)

OUT = (
    E5 /
    "06_dnaberts/"
    "10_blind_test_evaluation_v1_20260918"
)


if OUT.exists():

    raise RuntimeError(
        f"Output already exists:\n"
        f"{OUT}\n"
        "Do not overwrite blind-test output."
    )


OUT.mkdir(
    parents=True,
    exist_ok=False,
)


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


def binary_metrics(
    y,
    pred,
):

    y = np.asarray(
        y,
        dtype=np.int8,
    )

    pred = np.asarray(
        pred,
        dtype=np.int8,
    )


    tp = int(
        np.sum(
            (y == 1)
            &
            (pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            &
            (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            &
            (pred == 0)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            &
            (pred == 0)
        )
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall +
        specificity
    ) / 2.0


    f1 = (
        2.0 * tp
        /
        (
            2.0 * tp +
            fp +
            fn
        )
        if (
            2 * tp +
            fp +
            fn
        ) > 0
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
        (
            tp * tn -
            fp * fn
        )
        /
        denom
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
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "specificity":
            float(
                specificity
            ),

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


    if (
        n_pos == 0
        or
        n_neg == 0
    ):

        return float("nan")


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

    while i < len(
        score
    ):

        j = i + 1

        while (
            j < len(score)
            and
            sorted_score[j]
            ==
            sorted_score[i]
        ):

            j += 1


        avg_rank = (
            (i + 1) +
            j
        ) / 2.0


        ranks[
            order[i:j]
        ] = avg_rank


        i = j


    sum_pos_ranks = float(
        ranks[
            y == 1
        ].sum()
    )


    auc = (
        sum_pos_ranks
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


def average_precision(
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


        group = ys[
            i:j
        ]


        tp += int(
            np.sum(
                group == 1
            )
        )

        fp += int(
            np.sum(
                group == 0
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


print("=" * 100)
print(
    "DNABERT-S BLIND TEST EVALUATION v1"
)
print("=" * 100)

print()
print(
    "Threshold source:"
)

print(
    THRESHOLD_FILE
)

print()
print(
    "IMPORTANT:"
)

print(
    "Frozen validation thresholds are used directly."
)

print(
    "No threshold search or test retuning is performed."
)


# ================================================================
# Preconditions
# ================================================================

for p in [
    EMB_FILE,
    EMB_MANIFEST,
    THRESHOLD_FILE,
    THRESHOLD_POLICY,
]:

    if not p.is_file():

        raise FileNotFoundError(
            p
        )


# ================================================================
# Embeddings
# ================================================================

embedding = np.load(
    EMB_FILE,
    mmap_mode="r",
)


if embedding.shape != (
    3154,
    768,
):

    raise RuntimeError(
        f"Unexpected embedding shape "
        f"{embedding.shape}"
    )


if embedding.dtype != (
    np.float32
):

    raise RuntimeError(
        f"Unexpected embedding dtype "
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
        "Duplicate embedding accession"
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
        "Embedding indices "
        "not 0..3153"
    )


index_by_accession = dict(
    zip(
        manifest[
            "assembly_accession"
        ].astype(str),
        indices,
    )
)


# ================================================================
# Frozen thresholds
# ================================================================

thresholds = pd.read_csv(
    THRESHOLD_FILE,
    sep="\t",
    dtype={
        "protocol":
            str,

        "split_seed":
            str,
    },
)


if len(
    thresholds
) != 9:

    raise RuntimeError(
        f"Expected 9 thresholds, "
        f"found {len(thresholds)}"
    )


if thresholds[
    [
        "protocol",
        "split_seed",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate frozen threshold key"
    )


threshold_map = {}


for r in thresholds.itertuples(
    index=False
):

    key = (
        str(
            r.protocol
        ),
        str(
            r.split_seed
        ),
    )

    threshold_map[
        key
    ] = float(
        r.frozen_threshold
    )


expected_keys = set(
    EXPECTED_TEST.keys()
)


if set(
    threshold_map
) != expected_keys:

    raise RuntimeError(
        "Frozen threshold keys "
        "do not match expected "
        "G2/G3/G4 × 3 seeds"
    )


print()
print(
    "Embedding matrix:",
    embedding.shape,
    embedding.dtype
)

print(
    "Frozen thresholds:",
    len(
        threshold_map
    )
)

print(
    "Threshold SHA256:",
    sha256_file(
        THRESHOLD_FILE
    )
)


# ================================================================
# Blind test
# ================================================================

split_rows = []
score_files = []
input_hash_rows = []


for protocol in PROTOCOLS:

    print()
    print("=" * 100)
    print(protocol)
    print("=" * 100)


    for seed in SEEDS:

        key = (
            protocol,
            seed,
        )


        threshold = (
            threshold_map[
                key
            ]
        )


        pair_file = (
            PAIR_ROOT /
            f"{protocol}_seed{seed}" /
            "test_pairs.tsv"
        )


        if not pair_file.is_file():

            raise FileNotFoundError(
                pair_file
            )


        expected_n = (
            EXPECTED_TEST[
                key
            ]
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
        ) != expected_n:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"expected {expected_n}, "
                f"found {len(pairs)}"
            )


        required_cols = [
            "pair_id",
            "protocol",
            "split_seed",
            "role",
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
        ]


        for c in required_cols:

            if c not in pairs.columns:

                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    f"missing {c}"
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
                "protocol mismatch"
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
                "seed mismatch"
            )


        if set(
            pairs[
                "role"
            ].astype(str)
        ) != {
            "test"
        }:

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "role != test"
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
                "non-binary label"
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
                f"unbalanced test set "
                f"pos={n_pos}, "
                f"neg={n_neg}"
            )


        positive_type = (
            pairs[
                "pair_type"
            ].astype(str)
            ==
            "positive"
        ).to_numpy()


        if not np.array_equal(
            positive_type,
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


        missing_a = (
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


        missing_b = (
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
                "missing embedding accession"
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


        a = np.asarray(
            embedding[
                idx_a
            ],
            dtype=np.float64,
        )

        b = np.asarray(
            embedding[
                idx_b
            ],
            dtype=np.float64,
        )


        score = np.sum(
            a * b,
            axis=1,
            dtype=np.float64,
        )


        if not np.isfinite(
            score
        ).all():

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "non-finite score"
            )


        if (
            score.min() < -1.00001
            or
            score.max() > 1.00001
        ):

            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "cosine out of range"
            )


        pred = (
            score >= threshold
        ).astype(
            np.int8
        )


        auroc = (
            auroc_rank(
                y,
                score,
            )
        )

        auprc = (
            average_precision(
                y,
                score,
            )
        )

        metrics = (
            binary_metrics(
                y,
                pred,
            )
        )


        split_rows.append({
            "method":
                "DNABERT-S",

            "protocol":
                protocol,

            "split_seed":
                seed,

            "subset":
                "all",

            "metric_scope":
                "all_test_pairs",

            "coverage":
                1.0,

            "positive_coverage":
                1.0,

            "negative_coverage":
                1.0,

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

            "auroc":
                auroc,

            "auprc":
                auprc,

            "mcc":
                metrics[
                    "mcc"
                ],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "f1":
                metrics[
                    "f1"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "specificity":
                metrics[
                    "specificity"
                ],

            "tp":
                metrics[
                    "tp"
                ],

            "fp":
                metrics[
                    "fp"
                ],

            "tn":
                metrics[
                    "tn"
                ],

            "fn":
                metrics[
                    "fn"
                ],

            "n_model_seeds":
                1,

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
        })


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

        scored[
            "frozen_validation_threshold"
        ] = threshold

        scored[
            "dnaberts_prediction"
        ] = pred


        split_dir = (
            OUT /
            f"{protocol}_seed{seed}"
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=False,
        )


        scored_file = (
            split_dir /
            "dnaberts_blind_test_scores.tsv"
        )


        scored.to_csv(
            scored_file,
            sep="\t",
            index=False,
        )


        score_files.append(
            scored_file
        )


        input_hash_rows.append({
            "input_type":
                "blind_test_pair_manifest",

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
            f"thr={threshold:.10f} "
            f"AUROC={auroc:.6f} "
            f"AUPRC={auprc:.6f} "
            f"MCC={metrics['mcc']:.6f} "
            f"BAcc="
            f"{metrics['balanced_accuracy']:.6f} "
            f"F1={metrics['f1']:.6f}"
        )


# ================================================================
# Split-level table
# ================================================================

split_df = pd.DataFrame(
    split_rows
)


if len(
    split_df
) != 9:

    raise RuntimeError(
        f"Expected 9 split-level rows, "
        f"found {len(split_df)}"
    )


if int(
    split_df[
        "n_pairs"
    ].sum()
) != 34882:

    raise RuntimeError(
        "Total blind-test pair "
        "count !=34882"
    )


SPLIT_FILE = (
    OUT /
    "DNABERTS_BLIND_TEST_SPLIT_LEVEL.tsv"
)


split_df.to_csv(
    SPLIT_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Protocol summaries:
# primary variability unit = 3 split seeds.
# ================================================================

metric_cols = [
    "coverage",
    "auroc",
    "auprc",
    "mcc",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "specificity",
]


protocol_rows = []


for protocol in PROTOCOLS:

    x = split_df[
        split_df[
            "protocol"
        ]
        ==
        protocol
    ].copy()


    if len(
        x
    ) != 3:

        raise RuntimeError(
            f"{protocol}: expected "
            "3 split seeds"
        )


    row = {
        "method":
            "DNABERT-S",

        "protocol":
            protocol,

        "subset":
            "all",

        "metric_scope":
            "all_test_pairs",

        "n_split_seeds":
            3,

        "n_model_seeds":
            1,

        "total_pairs_across_splits":
            int(
                x[
                    "n_pairs"
                ].sum()
            ),
    }


    for metric in metric_cols:

        vals = pd.to_numeric(
            x[
                metric
            ]
        ).to_numpy(
            dtype=np.float64
        )


        row[
            f"{metric}_mean"
        ] = float(
            np.mean(
                vals
            )
        )


        row[
            f"{metric}_sd"
        ] = float(
            np.std(
                vals,
                ddof=1,
            )
        )


    protocol_rows.append(
        row
    )


protocol_df = pd.DataFrame(
    protocol_rows
)


PROTOCOL_FILE = (
    OUT /
    "DNABERTS_BLIND_TEST_PROTOCOL_SUMMARY.tsv"
)


protocol_df.to_csv(
    PROTOCOL_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Primary compact table
# ================================================================

primary_rows = []


for r in protocol_df.itertuples(
    index=False
):

    primary_rows.append({
        "method":
            r.method,

        "protocol":
            r.protocol,

        "n_split_seeds":
            r.n_split_seeds,

        "coverage_mean":
            r.coverage_mean,

        "coverage_sd":
            r.coverage_sd,

        "auroc_mean":
            r.auroc_mean,

        "auroc_sd":
            r.auroc_sd,

        "auprc_mean":
            r.auprc_mean,

        "auprc_sd":
            r.auprc_sd,

        "mcc_mean":
            r.mcc_mean,

        "mcc_sd":
            r.mcc_sd,

        "balanced_accuracy_mean":
            r.balanced_accuracy_mean,

        "balanced_accuracy_sd":
            r.balanced_accuracy_sd,

        "f1_mean":
            r.f1_mean,

        "f1_sd":
            r.f1_sd,
    })


primary_df = pd.DataFrame(
    primary_rows
)


PRIMARY_FILE = (
    OUT /
    "DNABERTS_PRIMARY_BLIND_TEST_SUMMARY.tsv"
)


primary_df.to_csv(
    PRIMARY_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Input provenance
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
            "frozen_validation_thresholds",

        "protocol":
            "",

        "split_seed":
            "",

        "path":
            str(
                THRESHOLD_FILE
            ),

        "sha256":
            sha256_file(
                THRESHOLD_FILE
            ),
    },

    {
        "input_type":
            "threshold_policy",

        "protocol":
            "",

        "split_seed":
            "",

        "path":
            str(
                THRESHOLD_POLICY
            ),

        "sha256":
            sha256_file(
                THRESHOLD_POLICY
            ),
    },
])


INPUT_SHA = (
    OUT /
    "INPUT_SHA256.tsv"
)


pd.DataFrame(
    input_hash_rows
).to_csv(
    INPUT_SHA,
    sep="\t",
    index=False,
)


# ================================================================
# Report
# ================================================================

REPORT = (
    OUT /
    "DNABERTS_BLIND_TEST_REPORT.txt"
)


with open(
    REPORT,
    "w",
) as f:

    f.write(
        "DNABERT-S Blind Test Evaluation v1\n"
    )

    f.write(
        "=" * 84 +
        "\n\n"
    )

    f.write(
        "Thresholds were frozen using "
        "validation pairs before blind-test "
        "evaluation.\n"
    )

    f.write(
        "No test-specific threshold search "
        "or retuning was performed.\n\n"
    )

    f.write(
        f"Frozen threshold file SHA256:\n"
        f"{sha256_file(THRESHOLD_FILE)}\n\n"
    )

    f.write(
        "Total test pair occurrences: "
        f"{int(split_df['n_pairs'].sum())}\n"
    )

    f.write(
        "Coverage: 1.0 for every split.\n"
    )

    f.write(
        "Statistical variability unit: "
        "three taxonomy split seeds per protocol.\n"
    )

    f.write(
        "DNABERT-S is deterministic here; "
        "n_model_seeds=1.\n\n"
    )

    f.write(
        "SPLIT-LEVEL RESULTS\n"
    )

    f.write(
        "-" * 84 +
        "\n"
    )

    f.write(
        split_df[
            [
                "protocol",
                "split_seed",
                "n_pairs",
                "frozen_threshold",
                "auroc",
                "auprc",
                "mcc",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "specificity",
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nPROTOCOL SUMMARY "
        "(mean +/- sample SD over "
        "three split seeds)\n"
    )

    f.write(
        "-" * 84 +
        "\n"
    )

    f.write(
        primary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: PASS\n"
    )


# ================================================================
# Script provenance
# ================================================================

script_path = Path(
    __file__
).resolve()


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


# ================================================================
# Freeze hashes
# ================================================================

HASH_FILE = (
    OUT /
    "RESULT_SHA256.txt"
)


files_to_hash = [
    SPLIT_FILE,
    PROTOCOL_FILE,
    PRIMARY_FILE,
    INPUT_SHA,
    REPORT,
    PROVENANCE,
] + score_files


with open(
    HASH_FILE,
    "w",
) as f:

    for p in files_to_hash:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ================================================================
# Console
# ================================================================

print()
print("=" * 100)
print(
    "DNABERT-S BLIND TEST SPLIT-LEVEL RESULTS"
)
print("=" * 100)

print(
    split_df[
        [
            "protocol",
            "split_seed",
            "n_pairs",
            "frozen_threshold",
            "auroc",
            "auprc",
            "mcc",
            "balanced_accuracy",
            "f1",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print(
    "DNABERT-S BLIND TEST PROTOCOL SUMMARY"
)
print("=" * 100)

print(
    primary_df.to_string(
        index=False
    )
)


print()
print(
    "total test pair occurrences =",
    int(
        split_df[
            "n_pairs"
        ].sum()
    )
)

print(
    "coverage all splits =",
    bool(
        np.allclose(
            split_df[
                "coverage"
            ],
            1.0,
        )
    )
)

print()
print(
    "STATUS = PASS"
)
