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

SPLIT = (
    BASE
    / "04_splits/g1_g4_v1_20260916/manifests/"
      "G1_seed20260915.tsv"
)

VAL_PAIRS = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916/"
      "G1_seed20260915/val_pairs.tsv"
)

TRAIN_ROOT = (
    BASE
    / "09_external_validation/"
      "canonical_A_model_training_v1_20260917"
)

PROTOCOL = (
    BASE
    / "09_external_validation/"
      "canonical_validation_threshold_protocol_v1_20260917/"
      "CANONICAL_VALIDATION_THRESHOLD_PROTOCOL.txt"
)

OUT = (
    BASE
    / "09_external_validation/"
      "canonical_validation_threshold_freeze_v1_20260917"
)

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

PCA_SEED = 20260915


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


def find_col(
    fields,
    candidates,
    name,
):

    for c in candidates:

        if c in fields:
            return c

    raise RuntimeError(
        f"Cannot detect {name}: {fields}"
    )


def confusion(
    y,
    s,
    threshold,
):

    pred = (
        s >= threshold
    ).astype(
        np.int64
    )

    tp = int(
        np.sum(
            (pred == 1)
            & (y == 1)
        )
    )

    fp = int(
        np.sum(
            (pred == 1)
            & (y == 0)
        )
    )

    tn = int(
        np.sum(
            (pred == 0)
            & (y == 0)
        )
    )

    fn = int(
        np.sum(
            (pred == 0)
            & (y == 1)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    bal = (
        recall + specificity
    ) / 2.0

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    denom = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn)
        / denom
        if denom
        else 0.0
    )

    return {
        "MCC": mcc,
        "F1": f1,
        "BalancedAccuracy": bal,
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

    if set(
        np.unique(y).tolist()
    ) != {0, 1}:

        raise RuntimeError(
            "Validation must contain both classes."
        )

    unique = np.unique(s)

    candidates = []

    candidates.append(
        float(
            np.nextafter(
                unique[-1],
                np.inf,
            )
        )
    )

    if len(unique) > 1:

        mids = (
            unique[:-1]
            + unique[1:]
        ) / 2.0

        candidates.extend(
            mids.tolist()
        )

    candidates.append(
        float(
            np.nextafter(
                unique[0],
                -np.inf,
            )
        )
    )

    best = None

    for t in candidates:

        m = confusion(
            y,
            s,
            t,
        )

        row = {
            "threshold":
                float(t),
            **m,
        }

        key = (
            row["MCC"],
            row["BalancedAccuracy"],
            row["F1"],
            -abs(
                row["threshold"]
                - 0.5
            ),
            -row["threshold"],
        )

        if (
            best is None
            or key > best[0]
        ):

            best = (
                key,
                row,
            )

    return best[1]


if OUT.exists():

    raise SystemExit(
        f"ERROR: output already exists: {OUT}"
    )

if not PROTOCOL.is_file():

    raise RuntimeError(
        "Missing frozen threshold protocol."
    )

print("=" * 78)
print("HGV-V3 CANONICAL VALIDATION THRESHOLD FREEZE")
print("=" * 78)

print("G1_test_used=False")
print("external_used=False")

OUT.mkdir(
    parents=True,
)

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

ff, fr = read_tsv(
    FEATURE_ROWS
)

acc_col = find_col(
    ff,
    [
        "assembly_accession",
        "assembly_accession_final",
        "accession",
    ],
    "feature accession",
)

accessions = [
    clean(r[acc_col])
    for r in fr
]

acc_to_idx = {
    a: i
    for i, a
    in enumerate(accessions)
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

# ------------------------------------------------------------
# Split train genomes
# ------------------------------------------------------------

sf, sr = read_tsv(
    SPLIT
)

split_acc = find_col(
    sf,
    [
        "assembly_accession",
        "assembly_accession_final",
        "accession",
    ],
    "split accession",
)

role_col = find_col(
    sf,
    [
        "role",
        "split_role",
        "set",
    ],
    "role",
)

train_acc = sorted({
    clean(
        r[split_acc]
    )
    for r in sr
    if clean(
        r[role_col]
    ).lower() == "train"
})

if not train_acc:

    raise RuntimeError(
        "No G1 train accessions."
    )

print(
    f"G1_train_genomes={len(train_acc)}"
)

# ------------------------------------------------------------
# Validation pairs
# ------------------------------------------------------------

pf, pr = read_tsv(
    VAL_PAIRS
)

a_col = find_col(
    pf,
    [
        "accession_a",
        "assembly_accession_a",
        "genome_a",
        "a_accession",
    ],
    "endpoint A",
)

b_col = find_col(
    pf,
    [
        "accession_b",
        "assembly_accession_b",
        "genome_b",
        "b_accession",
    ],
    "endpoint B",
)

y_col = find_col(
    pf,
    [
        "label",
        "y",
        "target",
        "same_species",
    ],
    "label",
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

idx_a = np.asarray([
    acc_to_idx[a]
    for a in acc_a
])

idx_b = np.asarray([
    acc_to_idx[b]
    for b in acc_b
])

A = np.asarray(
    X[idx_a],
    dtype=np.float32,
).astype(
    np.float64
)

B = np.asarray(
    X[idx_b],
    dtype=np.float32,
).astype(
    np.float64
)

rows_out = []

# ------------------------------------------------------------
# HGV thresholds
# ------------------------------------------------------------

print()
print("[1] HGV VALIDATION THRESHOLDS")

for model_seed in MODEL_SEEDS:

    run_dir = (
        TRAIN_ROOT
        / "G1"
        / "split20260915"
        / f"model{model_seed}"
    )

    complete = (
        run_dir
        / "RUN_COMPLETE"
    )

    pred_path = (
        run_dir
        / "val_predictions.tsv"
    )

    ckpt_path = (
        run_dir
        / "best_model.pt"
    )

    if not complete.is_file():

        raise RuntimeError(
            f"Incomplete run: {run_dir}"
        )

    if not pred_path.is_file():

        raise RuntimeError(
            f"Missing validation predictions: {pred_path}"
        )

    vf, vr = read_tsv(
        pred_path
    )

    if (
        "label" not in vf
        or "similarity_score" not in vf
    ):

        raise RuntimeError(
            f"Bad prediction schema: {pred_path}"
        )

    vy = np.asarray([
        int(float(r["label"]))
        for r in vr
    ])

    vs = np.asarray([
        float(
            r["similarity_score"]
        )
        for r in vr
    ])

    frozen = freeze_threshold(
        vy,
        vs,
    )

    auc = float(
        roc_auc_score(
            vy,
            vs,
        )
    )

    ap = float(
        average_precision_score(
            vy,
            vs,
        )
    )

    rows_out.append({
        "method_type":
            "HGV",

        "method":
            "HGV_Dashing2_tau0p98",

        "model_seed":
            model_seed,

        "threshold":
            frozen["threshold"],

        "val_AUROC":
            auc,

        "val_AUPRC":
            ap,

        "val_MCC":
            frozen["MCC"],

        "val_F1":
            frozen["F1"],

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
            len(vr),

        "source_sha256":
            sha256(
                pred_path
            ),

        "checkpoint_sha256":
            sha256(
                ckpt_path
            ),
    })

    print(
        f"HGV model={model_seed} "
        f"threshold="
        f"{frozen['threshold']:.8f} "
        f"MCC="
        f"{frozen['MCC']:.6f} "
        f"AUC={auc:.6f}"
    )

# ------------------------------------------------------------
# Raw baselines
# ------------------------------------------------------------

print()
print("[2] BASELINE VALIDATION THRESHOLDS")

bray = (
    1.0
    - 0.5
    * np.abs(
        A - B
    ).sum(
        axis=1
    )
)

norm_a = np.linalg.norm(
    A,
    axis=1,
)

norm_b = np.linalg.norm(
    B,
    axis=1,
)

denom = (
    norm_a
    * norm_b
)

cosine = np.divide(
    np.sum(
        A * B,
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

raw_euclidean = (
    -np.linalg.norm(
        A - B,
        axis=1,
    )
)

# ------------------------------------------------------------
# PCA64 fit on G1 train only
# ------------------------------------------------------------

train_idx = np.asarray([
    acc_to_idx[a]
    for a in train_acc
])

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

all_val_acc = sorted(
    set(acc_a)
    | set(acc_b)
)

all_val_idx = np.asarray([
    acc_to_idx[a]
    for a in all_val_acc
])

P = pca.transform(
    np.asarray(
        X[all_val_idx],
        dtype=np.float32,
    )
).astype(
    np.float32
)

lookup = {
    a: P[i]
    for i, a
    in enumerate(
        all_val_acc
    )
}

PA = np.stack([
    lookup[a]
    for a in acc_a
])

PB = np.stack([
    lookup[b]
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

baseline_prediction_rows = []

for i in range(
    len(y)
):

    row = {
        "accession_a":
            acc_a[i],

        "accession_b":
            acc_b[i],

        "label":
            int(y[i]),
    }

    for method in BASELINES:

        row[
            method
        ] = float(
            method_scores[
                method
            ][i]
        )

    baseline_prediction_rows.append(
        row
    )

baseline_pred_path = (
    OUT
    / "baseline_validation_predictions.tsv"
)

write_tsv(
    baseline_pred_path,
    baseline_prediction_rows,
    [
        "accession_a",
        "accession_b",
        "label",
        *BASELINES,
    ],
)

for method in BASELINES:

    scores = (
        method_scores[
            method
        ]
    )

    frozen = freeze_threshold(
        y,
        scores,
    )

    auc = float(
        roc_auc_score(
            y,
            scores,
        )
    )

    ap = float(
        average_precision_score(
            y,
            scores,
        )
    )

    rows_out.append({
        "method_type":
            "baseline",

        "method":
            method,

        "model_seed":
            "",

        "threshold":
            frozen["threshold"],

        "val_AUROC":
            auc,

        "val_AUPRC":
            ap,

        "val_MCC":
            frozen["MCC"],

        "val_F1":
            frozen["F1"],

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

        "checkpoint_sha256":
            "",
    })

    print(
        f"{method:20s} "
        f"threshold="
        f"{frozen['threshold']:.8f} "
        f"MCC="
        f"{frozen['MCC']:.6f} "
        f"AUC={auc:.6f}"
    )

# ------------------------------------------------------------
# Save PCA
# ------------------------------------------------------------

np.savez_compressed(
    OUT
    / "PCA64_G1_TRAIN_FROZEN.npz",

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

    random_state=
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

# ------------------------------------------------------------
# Threshold table
# ------------------------------------------------------------

if len(rows_out) != 8:

    raise RuntimeError(
        f"Expected 8 thresholds, got {len(rows_out)}"
    )

threshold_path = (
    OUT
    / "FROZEN_CANONICAL_VALIDATION_THRESHOLDS.tsv"
)

write_tsv(
    threshold_path,
    rows_out,
    [
        "method_type",
        "method",
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
        "checkpoint_sha256",
    ],
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = (
    OUT
    / "CANONICAL_VALIDATION_THRESHOLD_FREEZE_REPORT.txt"
)

with open(
    report,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 CANONICAL VALIDATION THRESHOLD FREEZE\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "STATUS: PASS\n\n"
    )

    f.write(
        "G1 test used: NO\n"
    )

    f.write(
        "External B1/B2 used: NO\n\n"
    )

    f.write(
        "Frozen HGV thresholds: 3\n"
    )

    f.write(
        "Frozen baseline thresholds: 5\n"
    )

    f.write(
        "Total thresholds: 8\n\n"
    )

    f.write(
        f"G1 train genomes for PCA: "
        f"{len(train_acc)}\n"
    )

    f.write(
        "PCA refit on validation: NO\n"
    )

    f.write(
        "PCA fit scope: G1 TRAIN genomes only\n\n"
    )

    f.write(
        "Threshold selection:\n"
    )

    f.write(
        "  maximum validation MCC\n"
    )

    f.write(
        "  tie-break BalancedAccuracy -> F1 "
        "-> closest to 0.5 -> lower threshold\n\n"
    )

    f.write(
        "CANONICAL_VALIDATION_THRESHOLD_FREEZE=PASS\n"
    )

provenance = {
    "status":
        "PASS",

    "G1_test_used":
        False,

    "external_used":
        False,

    "hgv_thresholds":
        3,

    "baseline_thresholds":
        5,

    "total_thresholds":
        8,

    "PCA_train_genomes":
        len(train_acc),

    "PCA_random_state":
        PCA_SEED,

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "feature_sha256":
        sha256(
            FEATURE_NPY
        ),

    "validation_pair_sha256":
        sha256(
            VAL_PAIRS
        ),
}

(
    OUT
    / "canonical_threshold_freeze_provenance.json"
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# ------------------------------------------------------------
# SHA256
# ------------------------------------------------------------

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
print("CANONICAL VALIDATION THRESHOLD FREEZE COMPLETE")
print("=" * 78)

print(
    f"Output: {OUT}"
)

print(
    "hgv_thresholds=3"
)

print(
    "baseline_thresholds=5"
)

print(
    "total_thresholds=8"
)

print(
    "G1_test_used=False"
)

print(
    "external_used=False"
)

print(
    "CANONICAL_VALIDATION_THRESHOLD_FREEZE=PASS"
)
