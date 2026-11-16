#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


BASE = Path(
    "HGV_V3_revision_20260915"
)

FEATURE_NPY = (
    BASE
    / "03_features/canonical_k6_full3154_v1/"
      "canonical_k6_features.float32.npy"
)

FEATURE_ROWS = (
    BASE
    / "03_features/canonical_k6_full3154_v1/"
      "canonical_k6_rows.tsv"
)

SPLIT_ROOT = (
    BASE
    / "04_splits/g1_g4_v1_20260916/manifests"
)

PAIR_ROOT = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
)

TRAIN_ROOT = (
    BASE
    / "08_main_generalization/"
      "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
)

TEST_PROTOCOL = (
    BASE
    / "08_main_generalization/"
      "test_eval_protocol_freeze_v1_20260917/"
      "TEST_EVAL_PROTOCOL.txt"
)

OUT = (
    BASE
    / "08_main_generalization/"
      "validation_threshold_freeze_v1_20260917"
)

PCA_SEED = 20260915

PROTOCOLS = [
    "G2",
    "G3",
    "G4",
]

SPLIT_SEEDS = [
    20260915,
    20260916,
    20260917,
]

MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
]

BASELINES = [
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
]


def clean(x):
    return "" if x is None else str(x).strip()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def read_tsv(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        r = csv.DictReader(
            f,
            delimiter="\t",
        )

        return (
            list(r.fieldnames or []),
            list(r),
        )


def write_tsv(
    path,
    rows,
    fields,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )

        w.writeheader()
        w.writerows(rows)


def detect_col(
    fields,
    candidates,
    description,
):

    for c in candidates:
        if c in fields:
            return c

    raise RuntimeError(
        f"Cannot detect {description}: {fields}"
    )


def detect_accession(fields):

    return detect_col(
        fields,
        [
            "assembly_accession",
            "assembly_accession_final",
            "accession",
        ],
        "accession",
    )


def confusion_metrics(
    tp,
    fp,
    tn,
    fn,
):

    tp = float(tp)
    fp = float(fp)
    tn = float(tn)
    fn = float(fn)

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

    balanced = (
        recall + specificity
    ) / 2.0

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    denom = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        (
            tp * tn
            - fp * fn
        )
        / denom
        if denom > 0
        else 0.0
    )

    return {
        "MCC": mcc,
        "F1": f1,
        "BalancedAccuracy": balanced,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
    }


def freeze_threshold(
    labels,
    scores,
):

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(y) != len(s):
        raise RuntimeError(
            "labels/scores length mismatch"
        )

    if len(y) == 0:
        raise RuntimeError(
            "Empty validation set."
        )

    if not np.all(
        np.isfinite(s)
    ):
        raise RuntimeError(
            "Non-finite validation score."
        )

    if set(
        np.unique(y).tolist()
    ) != {0, 1}:
        raise RuntimeError(
            "Validation labels must contain both classes."
        )

    order = np.argsort(
        -s,
        kind="mergesort",
    )

    ys = y[order]
    ss = s[order]

    total_pos = int(
        np.sum(ys == 1)
    )

    total_neg = int(
        np.sum(ys == 0)
    )

    candidates = []

    # ----------------------------------------------------------
    # Candidate above maximum:
    # predicts everything negative.
    # ----------------------------------------------------------

    threshold = float(
        np.nextafter(
            ss[0],
            np.inf,
        )
    )

    m = confusion_metrics(
        tp=0,
        fp=0,
        tn=total_neg,
        fn=total_pos,
    )

    candidates.append({
        "threshold": threshold,
        **m,
    })

    # ----------------------------------------------------------
    # Move threshold downward one unique score group at a time.
    # ----------------------------------------------------------

    tp = 0
    fp = 0

    i = 0
    n = len(ss)

    while i < n:

        current = ss[i]

        j = i

        while (
            j < n
            and ss[j] == current
        ):
            if ys[j] == 1:
                tp += 1
            else:
                fp += 1

            j += 1

        fn = total_pos - tp
        tn = total_neg - fp

        if j < n:

            nxt = ss[j]

            threshold = float(
                (
                    current
                    + nxt
                )
                / 2.0
            )

        else:

            threshold = float(
                np.nextafter(
                    current,
                    -np.inf,
                )
            )

        m = confusion_metrics(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

        candidates.append({
            "threshold": threshold,
            **m,
        })

        i = j

    # ----------------------------------------------------------
    # Frozen tie-breaking:
    #
    # 1 MCC
    # 2 Balanced Accuracy
    # 3 F1
    # 4 closest to 0.5
    # 5 lower threshold
    # ----------------------------------------------------------

    best = max(
        candidates,
        key=lambda r: (
            r["MCC"],
            r["BalancedAccuracy"],
            r["F1"],
            -abs(
                r["threshold"]
                - 0.5
            ),
            -r["threshold"],
        ),
    )

    return best


def ranking_metrics(
    labels,
    scores,
):

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    return {
        "AUROC": float(
            roc_auc_score(
                y,
                s,
            )
        ),
        "AUPRC": float(
            average_precision_score(
                y,
                s,
            )
        ),
    }


print("=" * 78)
print("HGV-V3 VALIDATION THRESHOLD FREEZE")
print("=" * 78)

print(
    "test_used=False"
)

print(
    "external_used=False"
)

print(
    "THIS SCRIPT DOES NOT READ test_pairs.tsv"
)

if OUT.exists():
    raise SystemExit(
        f"ERROR: output already exists: {OUT}"
    )

if not TEST_PROTOCOL.is_file():
    raise SystemExit(
        "ERROR: missing frozen test protocol."
    )

OUT.mkdir(
    parents=True,
)

(
    OUT
    / "pca_models"
).mkdir()

(
    OUT
    / "baseline_validation_predictions"
).mkdir()

# ==============================================================
# 1. Load canonical features
# ==============================================================

print()
print("[1] LOAD FEATURES")

ff, fr = read_tsv(
    FEATURE_ROWS
)

feature_acc_col = (
    detect_accession(ff)
)

feature_accessions = [
    clean(r[feature_acc_col])
    for r in fr
]

if (
    len(feature_accessions)
    != len(
        set(feature_accessions)
    )
):
    raise RuntimeError(
        "Duplicate feature accessions."
    )

acc_to_idx = {
    a: i
    for i, a
    in enumerate(
        feature_accessions
    )
}

X = np.load(
    FEATURE_NPY,
    mmap_mode="r",
)

if X.shape != (
    3154,
    2080,
):
    raise RuntimeError(
        f"Bad feature shape: {X.shape}"
    )

print(
    f"feature_shape={X.shape}"
)

print(
    f"feature_accessions="
    f"{len(acc_to_idx)}"
)

# ==============================================================
# 2. Verify all 27 frozen HGV runs
# ==============================================================

print()
print("[2] VERIFY 27 HGV RUNS")

expected_runs = []

for protocol in PROTOCOLS:
    for split_seed in SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:

            run_dir = (
                TRAIN_ROOT
                / protocol
                / f"split{split_seed}"
                / f"model{model_seed}"
            )

            expected_runs.append(
                run_dir
            )

            required = [
                run_dir
                / "RUN_COMPLETE",

                run_dir
                / "best_model.pt",

                run_dir
                / "val_predictions.tsv",

                run_dir
                / "run_summary.json",

                run_dir
                / "config.json",
            ]

            for p in required:
                if not p.is_file():
                    raise RuntimeError(
                        f"Missing frozen run file: {p}"
                    )

print(
    f"complete_runs="
    f"{len(expected_runs)}"
)

# ==============================================================
# 3. HGV validation-derived thresholds
# ==============================================================

print()
print("[3] FREEZE HGV VALIDATION THRESHOLDS")

threshold_rows = []

for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        for model_seed in MODEL_SEEDS:

            run_dir = (
                TRAIN_ROOT
                / protocol
                / f"split{split_seed}"
                / f"model{model_seed}"
            )

            pred_path = (
                run_dir
                / "val_predictions.tsv"
            )

            fields, rows = read_tsv(
                pred_path
            )

            if (
                "label" not in fields
                or "similarity_score"
                not in fields
            ):
                raise RuntimeError(
                    f"Bad validation predictions: {pred_path}"
                )

            y = np.asarray([
                int(
                    float(
                        r["label"]
                    )
                )
                for r in rows
            ])

            s = np.asarray([
                float(
                    r[
                        "similarity_score"
                    ]
                )
                for r in rows
            ])

            rank = ranking_metrics(
                y,
                s,
            )

            frozen = freeze_threshold(
                y,
                s,
            )

            threshold_rows.append({
                "method_type":
                    "HGV",

                "method":
                    "HGV_Dashing2_tau0p98",

                "protocol":
                    protocol,

                "split_seed":
                    split_seed,

                "model_seed":
                    model_seed,

                "threshold":
                    frozen[
                        "threshold"
                    ],

                "val_AUROC":
                    rank[
                        "AUROC"
                    ],

                "val_AUPRC":
                    rank[
                        "AUPRC"
                    ],

                "val_MCC":
                    frozen[
                        "MCC"
                    ],

                "val_F1":
                    frozen[
                        "F1"
                    ],

                "val_BalancedAccuracy":
                    frozen[
                        "BalancedAccuracy"
                    ],

                "val_Precision":
                    frozen[
                        "Precision"
                    ],

                "val_Recall":
                    frozen[
                        "Recall"
                    ],

                "n_val_pairs":
                    len(rows),

                "source_sha256":
                    sha256(
                        pred_path
                    ),
            })

            print(
                f"HGV {protocol} "
                f"split={split_seed} "
                f"model={model_seed} "
                f"thr="
                f"{frozen['threshold']:.8f} "
                f"MCC="
                f"{frozen['MCC']:.6f}"
            )

# ==============================================================
# 4. Baseline thresholds
# ==============================================================

print()
print("[4] FREEZE BASELINE VALIDATION THRESHOLDS")

for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        split_id = (
            f"{protocol}_seed{split_seed}"
        )

        split_path = (
            SPLIT_ROOT
            / f"{split_id}.tsv"
        )

        val_pair_path = (
            PAIR_ROOT
            / split_id
            / "val_pairs.tsv"
        )

        sf, sr = read_tsv(
            split_path
        )

        split_acc_col = (
            detect_accession(sf)
        )

        role_col = detect_col(
            sf,
            [
                "role",
                "split_role",
                "set",
            ],
            "split role",
        )

        train_acc = sorted({
            clean(
                r[
                    split_acc_col
                ]
            )
            for r in sr
            if clean(
                r[
                    role_col
                ]
            ).lower() == "train"
        })

        if not train_acc:
            raise RuntimeError(
                f"No train genomes: {split_id}"
            )

        pf, pr = read_tsv(
            val_pair_path
        )

        a_col = detect_col(
            pf,
            [
                "accession_a",
                "assembly_accession_a",
                "genome_a",
                "a_accession",
            ],
            "pair endpoint A",
        )

        b_col = detect_col(
            pf,
            [
                "accession_b",
                "assembly_accession_b",
                "genome_b",
                "b_accession",
            ],
            "pair endpoint B",
        )

        y_col = detect_col(
            pf,
            [
                "label",
                "y",
                "target",
                "same_species",
            ],
            "pair label",
        )

        acc_a = [
            clean(r[a_col])
            for r in pr
        ]

        acc_b = [
            clean(r[b_col])
            for r in pr
        ]

        y = np.asarray([
            int(
                float(
                    r[y_col]
                )
            )
            for r in pr
        ])

        try:
            idx_a = np.asarray([
                acc_to_idx[a]
                for a in acc_a
            ])

            idx_b = np.asarray([
                acc_to_idx[b]
                for b in acc_b
            ])

            train_idx = np.asarray([
                acc_to_idx[a]
                for a in train_acc
            ])

        except KeyError as e:
            raise RuntimeError(
                f"Unknown accession in {split_id}: {e}"
            )

        A = np.asarray(
            X[idx_a],
            dtype=np.float32,
        )

        B = np.asarray(
            X[idx_b],
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Raw-k6 Bray-Curtis
        # Features are L1-normalized, so:
        # BC distance = 0.5 * L1 distance
        # ------------------------------------------------------

        bray = (
            1.0
            - 0.5
            * np.abs(
                A.astype(
                    np.float64
                )
                - B.astype(
                    np.float64
                )
            ).sum(
                axis=1
            )
        )

        # ------------------------------------------------------
        # Raw-k6 cosine
        # ------------------------------------------------------

        A64 = A.astype(
            np.float64
        )

        B64 = B.astype(
            np.float64
        )

        norm_a = np.linalg.norm(
            A64,
            axis=1,
        )

        norm_b = np.linalg.norm(
            B64,
            axis=1,
        )

        denom = (
            norm_a
            * norm_b
        )

        cosine = np.divide(
            np.sum(
                A64 * B64,
                axis=1,
            ),
            denom,
            out=np.zeros_like(
                denom
            ),
            where=(
                denom > 0
            ),
        )

        # ------------------------------------------------------
        # Raw-k6 Euclidean ranking score
        # Larger score = more similar.
        # ------------------------------------------------------

        raw_euclidean = (
            -np.linalg.norm(
                A64 - B64,
                axis=1,
            )
        )

        # ------------------------------------------------------
        # PCA64:
        # fit TRAIN genomes only.
        # ------------------------------------------------------

        X_train = np.asarray(
            X[train_idx],
            dtype=np.float32,
        )

        pca = PCA(
            n_components=64,
            svd_solver="randomized",
            random_state=PCA_SEED,
        )

        pca.fit(
            X_train
        )

        unique_val_acc = sorted(
            set(acc_a)
            | set(acc_b)
        )

        unique_idx = np.asarray([
            acc_to_idx[a]
            for a in unique_val_acc
        ])

        P = pca.transform(
            np.asarray(
                X[unique_idx],
                dtype=np.float32,
            )
        ).astype(
            np.float32
        )

        pca_lookup = {
            a: P[i]
            for i, a
            in enumerate(
                unique_val_acc
            )
        }

        PA = np.stack([
            pca_lookup[a]
            for a in acc_a
        ])

        PB = np.stack([
            pca_lookup[b]
            for b in acc_b
        ])

        pca_euclidean = (
            -np.linalg.norm(
                PA.astype(
                    np.float64
                )
                - PB.astype(
                    np.float64
                ),
                axis=1,
            )
        )

        PA_l2 = (
            PA
            / np.clip(
                np.linalg.norm(
                    PA,
                    axis=1,
                    keepdims=True,
                ),
                1e-12,
                None,
            )
        )

        PB_l2 = (
            PB
            / np.clip(
                np.linalg.norm(
                    PB,
                    axis=1,
                    keepdims=True,
                ),
                1e-12,
                None,
            )
        )

        pca_l2 = (
            1.0
            - np.linalg.norm(
                PA_l2.astype(
                    np.float64
                )
                - PB_l2.astype(
                    np.float64
                ),
                axis=1,
            )
            / 2.0
        )

        # ------------------------------------------------------
        # Save frozen PCA transform.
        # Test evaluator will NOT refit PCA.
        # ------------------------------------------------------

        pca_model_path = (
            OUT
            / "pca_models"
            / f"{split_id}.npz"
        )

        np.savez_compressed(
            pca_model_path,

            mean=
                pca.mean_.astype(
                    np.float32
                ),

            components=
                pca.components_.astype(
                    np.float32
                ),

            explained_variance_ratio=
                pca.explained_variance_ratio_.astype(
                    np.float32
                ),

            explained_variance_ratio_sum=
                np.asarray(
                    [
                        pca.explained_variance_ratio_.sum()
                    ],
                    dtype=np.float64,
                ),

            pca_seed=
                np.asarray(
                    [PCA_SEED],
                    dtype=np.int64,
                ),

            train_genomes=
                np.asarray(
                    [len(train_acc)],
                    dtype=np.int64,
                ),
        )

        method_scores = {
            "RawK6_BrayCurtis":
                bray,

            "RawK6_Cosine":
                cosine,

            "RawK6_Euclidean":
                raw_euclidean,

            "PCA64_Euclidean":
                pca_euclidean,

            "PCA64_L2":
                pca_l2,
        }

        # ------------------------------------------------------
        # Save validation baseline predictions for audit.
        # ------------------------------------------------------

        prediction_rows = []

        for i in range(
            len(pr)
        ):

            prediction_rows.append({
                "accession_a":
                    acc_a[i],

                "accession_b":
                    acc_b[i],

                "label":
                    int(
                        y[i]
                    ),

                "RawK6_BrayCurtis":
                    float(
                        bray[i]
                    ),

                "RawK6_Cosine":
                    float(
                        cosine[i]
                    ),

                "RawK6_Euclidean":
                    float(
                        raw_euclidean[i]
                    ),

                "PCA64_Euclidean":
                    float(
                        pca_euclidean[i]
                    ),

                "PCA64_L2":
                    float(
                        pca_l2[i]
                    ),
            })

        baseline_pred_path = (
            OUT
            / "baseline_validation_predictions"
            / f"{split_id}.tsv"
        )

        write_tsv(
            baseline_pred_path,
            prediction_rows,
            [
                "accession_a",
                "accession_b",
                "label",
                "RawK6_BrayCurtis",
                "RawK6_Cosine",
                "RawK6_Euclidean",
                "PCA64_Euclidean",
                "PCA64_L2",
            ],
        )

        for method in BASELINES:

            scores = (
                method_scores[
                    method
                ]
            )

            rank = ranking_metrics(
                y,
                scores,
            )

            frozen = freeze_threshold(
                y,
                scores,
            )

            threshold_rows.append({
                "method_type":
                    "baseline",

                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    split_seed,

                "model_seed":
                    "",

                "threshold":
                    frozen[
                        "threshold"
                    ],

                "val_AUROC":
                    rank[
                        "AUROC"
                    ],

                "val_AUPRC":
                    rank[
                        "AUPRC"
                    ],

                "val_MCC":
                    frozen[
                        "MCC"
                    ],

                "val_F1":
                    frozen[
                        "F1"
                    ],

                "val_BalancedAccuracy":
                    frozen[
                        "BalancedAccuracy"
                    ],

                "val_Precision":
                    frozen[
                        "Precision"
                    ],

                "val_Recall":
                    frozen[
                        "Recall"
                    ],

                "n_val_pairs":
                    len(y),

                "source_sha256":
                    sha256(
                        baseline_pred_path
                    ),
            })

            print(
                f"{method:20s} "
                f"{protocol} "
                f"split={split_seed} "
                f"thr="
                f"{frozen['threshold']:.8f} "
                f"MCC="
                f"{frozen['MCC']:.6f}"
            )

# ==============================================================
# 5. Write frozen threshold table
# ==============================================================

print()
print("[5] WRITE THRESHOLD FREEZE")

expected_thresholds = (
    27
    + (
        len(PROTOCOLS)
        * len(SPLIT_SEEDS)
        * len(BASELINES)
    )
)

if (
    len(threshold_rows)
    != expected_thresholds
):
    raise RuntimeError(
        f"Threshold count mismatch: "
        f"{len(threshold_rows)} "
        f"!= {expected_thresholds}"
    )

threshold_path = (
    OUT
    / "FROZEN_VALIDATION_THRESHOLDS.tsv"
)

write_tsv(
    threshold_path,
    threshold_rows,
    [
        "method_type",
        "method",
        "protocol",
        "split_seed",
        "model_seed",
        "threshold",
        "val_AUROC",
        "val_AUPRC",
        "val_MCC",
        "val_F1",
        "val_BalancedAccuracy",
        "val_Precision",
        "val_Recall",
        "n_val_pairs",
        "source_sha256",
    ],
)

# ==============================================================
# 6. Provenance
# ==============================================================

provenance = {
    "status":
        "PASS",

    "freeze_stage":
        "validation-only",

    "test_used":
        False,

    "external_used":
        False,

    "hgv_thresholds":
        27,

    "baseline_thresholds":
        (
            len(PROTOCOLS)
            * len(SPLIT_SEEDS)
            * len(BASELINES)
        ),

    "total_thresholds":
        len(
            threshold_rows
        ),

    "threshold_rule":
        (
            "maximize validation MCC; tie-break "
            "BalancedAccuracy, F1, closest-to-0.5, "
            "then lower threshold"
        ),

    "pca_fit":
        (
            "TRAIN genomes only for each taxonomy split"
        ),

    "pca_random_state":
        PCA_SEED,

    "test_protocol_sha256":
        sha256(
            TEST_PROTOCOL
        ),

    "feature_npy_sha256":
        sha256(
            FEATURE_NPY
        ),

    "feature_rows_sha256":
        sha256(
            FEATURE_ROWS
        ),
}

(
    OUT
    / "threshold_freeze_provenance.json"
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# ==============================================================
# 7. Human report
# ==============================================================

report = (
    OUT
    / "VALIDATION_THRESHOLD_FREEZE_REPORT.txt"
)

with open(
    report,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 VALIDATION THRESHOLD FREEZE\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "STATUS: PASS\n\n"
    )

    f.write(
        "Test used: NO\n"
    )

    f.write(
        "External used: NO\n\n"
    )

    f.write(
        "Frozen HGV thresholds: 27\n"
    )

    f.write(
        "Frozen baseline thresholds: 45\n"
    )

    f.write(
        "Total frozen thresholds: 72\n\n"
    )

    f.write(
        "HGV threshold source: "
        "saved validation predictions only\n"
    )

    f.write(
        "Baseline threshold source: "
        "validation pair manifests only\n"
    )

    f.write(
        "PCA fit scope: "
        "TRAIN genomes only\n\n"
    )

    f.write(
        "Threshold selection:\n"
    )

    f.write(
        "  1. maximum validation MCC\n"
    )

    f.write(
        "  2. higher validation Balanced Accuracy\n"
    )

    f.write(
        "  3. higher validation F1\n"
    )

    f.write(
        "  4. threshold closest to 0.5\n"
    )

    f.write(
        "  5. lower numerical threshold\n\n"
    )

    f.write(
        "G2/G3/G4 test pairs were not opened "
        "by this script.\n\n"
    )

    f.write(
        "VALIDATION_THRESHOLD_FREEZE=PASS\n"
    )

# ==============================================================
# 8. SHA256
# ==============================================================

files = sorted([
    p
    for p in OUT.rglob("*")
    if (
        p.is_file()
        and p.name
        != "SHA256SUMS.txt"
    )
])

with open(
    OUT
    / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:

    for p in files:

        f.write(
            f"{sha256(p)}  "
            f"{p.relative_to(OUT)}\n"
        )

print()
print("=" * 78)
print("VALIDATION THRESHOLD FREEZE COMPLETE")
print("=" * 78)

print(
    f"Output: {OUT}"
)

print(
    "hgv_thresholds=27"
)

print(
    "baseline_thresholds=45"
)

print(
    "total_thresholds=72"
)

print(
    "test_used=False"
)

print(
    "external_used=False"
)

print(
    "VALIDATION_THRESHOLD_FREEZE=PASS"
)
