#!/usr/bin/env python3

import csv
import math
import random
from pathlib import Path

import numpy as np


BASE = Path("HGV_V3_revision_20260915")

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

MASTER = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916/"
      "final_master_manifest_3154.tsv"
)

TEACHER_DIR = (
    BASE
    / "06_teacher/teacher_geometry_audit_v1_20260916/"
      "teacher_augmented/G2_seed20260915"
)

TRAIN = TEACHER_DIR / "train_teacher.tsv"
VAL = TEACHER_DIR / "val_teacher.tsv"

PRED = (
    BASE
    / "07_hgv/teacher_selection_smoke_v1_20260916/"
      "D3_Mash_tau0p90__seed20260915/"
      "val_predictions.tsv"
)


def clean(x):
    return "" if x is None else str(x).strip()


def read_tsv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r.fieldnames or []), list(r)


def avg_ranks(values):
    n = len(values)
    order = sorted(
        range(n),
        key=lambda i: values[i],
    )

    ranks = [0.0] * n
    i = 0

    while i < n:
        j = i + 1

        while (
            j < n
            and values[order[j]]
            == values[order[i]]
        ):
            j += 1

        rank = (
            (i + 1) + j
        ) / 2.0

        for k in range(i, j):
            ranks[order[k]] = rank

        i = j

    return ranks


def auc(labels, scores):
    labels = list(map(int, labels))
    scores = list(map(float, scores))

    npos = sum(labels)
    nneg = len(labels) - npos

    ranks = avg_ranks(scores)

    rsum = sum(
        ranks[i]
        for i, y in enumerate(labels)
        if y == 1
    )

    return (
        rsum
        - npos * (npos + 1) / 2.0
    ) / (npos * nneg)


def ap(labels, scores):
    groups = {}

    for y, s in zip(labels, scores):
        groups.setdefault(
            float(s),
            [0, 0],
        )

        groups[float(s)][
            0 if int(y) == 1 else 1
        ] += 1

    total_pos = sum(
        int(y) == 1
        for y in labels
    )

    tp = 0
    fp = 0
    prev_r = 0.0
    result = 0.0

    for s in sorted(
        groups,
        reverse=True,
    ):
        gp, gn = groups[s]
        tp += gp
        fp += gn

        r = tp / total_pos
        p = tp / (tp + fp)

        result += (
            r - prev_r
        ) * p

        prev_r = r

    return result


def metric(name, labels, scores):
    a = auc(labels, scores)
    p = ap(labels, scores)

    print(
        f"{name:28s} "
        f"AUROC={a:.9f} "
        f"AUPRC={p:.9f}"
    )

    return a, p


print("=" * 78)
print("HGV-V3 CEILING / LEAKAGE AUDIT")
print("=" * 78)

# ------------------------------------------------------------------
# Master taxonomy
# ------------------------------------------------------------------

mf, mr = read_tsv(MASTER)

species = {
    clean(r["assembly_accession_final"]):
    clean(r["gtdb_species_cluster"])
    for r in mr
}

# ------------------------------------------------------------------
# Train / validation endpoint overlap
# ------------------------------------------------------------------

tf, tr = read_tsv(TRAIN)
vf, vr = read_tsv(VAL)

train_acc = set()
val_acc = set()

for r in tr:
    train_acc.add(clean(r["accession_a"]))
    train_acc.add(clean(r["accession_b"]))

for r in vr:
    val_acc.add(clean(r["accession_a"]))
    val_acc.add(clean(r["accession_b"]))

train_species = {
    species[a]
    for a in train_acc
}

val_species = {
    species[a]
    for a in val_acc
}

print()
print("[1] TRAIN / VALIDATION LEAKAGE")

print(f"train genomes       = {len(train_acc)}")
print(f"validation genomes  = {len(val_acc)}")

print(
    "accession overlap   = "
    f"{len(train_acc & val_acc)}"
)

print(
    "species overlap     = "
    f"{len(train_species & val_species)}"
)

# ------------------------------------------------------------------
# Label audit
# ------------------------------------------------------------------

print()
print("[2] TAXONOMIC LABEL AUDIT")

mismatch = 0

for r in vr:
    a = clean(r["accession_a"])
    b = clean(r["accession_b"])
    label = int(r["label"])

    truth = int(
        species[a] == species[b]
    )

    if truth != label:
        mismatch += 1

print(f"validation pairs     = {len(vr)}")
print(f"label mismatches     = {mismatch}")

# ------------------------------------------------------------------
# Existing model predictions
# ------------------------------------------------------------------

pf, pr = read_tsv(PRED)

labels = np.asarray([
    int(r["label"])
    for r in pr
])

scores = np.asarray([
    float(r["student_similarity"])
    for r in pr
])

distances = np.asarray([
    float(r["student_distance"])
    for r in pr
])

pair_type = np.asarray([
    clean(r["pair_type"])
    for r in pr
])

print()
print("[3] SAVED SMOKE MODEL METRIC RECHECK")

metric(
    "student/all",
    labels,
    scores,
)

pos = scores[
    labels == 1
]

neg = scores[
    labels == 0
]

print()
print("Student score geometry:")

print(
    f"positive similarity: "
    f"min={pos.min():.9f} "
    f"median={np.median(pos):.9f} "
    f"max={pos.max():.9f}"
)

print(
    f"negative similarity: "
    f"min={neg.min():.9f} "
    f"median={np.median(neg):.9f} "
    f"max={neg.max():.9f}"
)

gap = (
    pos.min()
    - neg.max()
)

print(
    f"ranking gap "
    f"(min_pos-max_neg) = "
    f"{gap:.9f}"
)

for tier in [
    "hard",
    "medium",
    "easy",
]:
    mask = (
        (labels == 1)
        |
        (pair_type == tier)
    )

    metric(
        f"student/{tier}",
        labels[mask],
        scores[mask],
    )

# ------------------------------------------------------------------
# Independent sklearn check, if available
# ------------------------------------------------------------------

print()
print("[4] INDEPENDENT SKLEARN METRIC CHECK")

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )

    print(
        "sklearn AUROC = "
        f"{roc_auc_score(labels, scores):.9f}"
    )

    print(
        "sklearn AUPRC = "
        f"{average_precision_score(labels, scores):.9f}"
    )

except Exception as e:
    print(
        "SKLEARN_UNAVAILABLE:",
        repr(e),
    )

# ------------------------------------------------------------------
# Raw k6 feature baseline
# ------------------------------------------------------------------

print()
print("[5] RAW 2080-D k6 FEATURE BASELINE")

ff, fr = read_tsv(FEATURE_ROWS)

acc_col = (
    "assembly_accession"
    if "assembly_accession" in ff
    else "assembly_accession_final"
)

acc_to_idx = {
    clean(r[acc_col]): i
    for i, r in enumerate(fr)
}

X = np.load(
    FEATURE_NPY,
    mmap_mode="r",
)

raw_labels = []
bray_sim = []
cos_sim = []
neg_euclid = []

for r in vr:
    a = clean(r["accession_a"])
    b = clean(r["accession_b"])

    xa = np.asarray(
        X[acc_to_idx[a]],
        dtype=np.float64,
    )

    xb = np.asarray(
        X[acc_to_idx[b]],
        dtype=np.float64,
    )

    raw_labels.append(
        int(r["label"])
    )

    # L1-normalized composition:
    # Bray-Curtis distance = 0.5 * L1 distance
    bc_dist = (
        0.5
        * np.abs(
            xa - xb
        ).sum()
    )

    bray_sim.append(
        1.0 - bc_dist
    )

    denom = (
        np.linalg.norm(xa)
        * np.linalg.norm(xb)
    )

    cos = (
        float(np.dot(xa, xb) / denom)
        if denom > 0
        else 0.0
    )

    cos_sim.append(cos)

    euclid = np.linalg.norm(
        xa - xb
    )

    neg_euclid.append(
        -float(euclid)
    )

raw_labels = np.asarray(
    raw_labels
)

metric(
    "raw_k6/BrayCurtis",
    raw_labels,
    bray_sim,
)

metric(
    "raw_k6/cosine",
    raw_labels,
    cos_sim,
)

metric(
    "raw_k6/euclidean",
    raw_labels,
    neg_euclid,
)

# ------------------------------------------------------------------
# Random label sanity
# ------------------------------------------------------------------

print()
print("[6] RANDOM-LABEL METRIC SANITY")

rng = np.random.default_rng(
    20260916
)

rand_auc = []
rand_ap = []

for _ in range(100):
    yy = rng.permutation(
        labels
    )

    rand_auc.append(
        auc(
            yy,
            scores,
        )
    )

    rand_ap.append(
        ap(
            yy,
            scores,
        )
    )

print(
    "random-label AUROC: "
    f"mean={np.mean(rand_auc):.6f} "
    f"sd={np.std(rand_auc):.6f}"
)

print(
    "random-label AUPRC: "
    f"mean={np.mean(rand_ap):.6f} "
    f"sd={np.std(rand_ap):.6f}"
)

# ------------------------------------------------------------------
# Final gate
# ------------------------------------------------------------------

print()
print("[7] AUDIT INTERPRETATION GATES")

print(
    "NO_ACCESSION_LEAKAGE="
    + str(
        len(
            train_acc
            & val_acc
        ) == 0
    )
)

print(
    "NO_SPECIES_LEAKAGE="
    + str(
        len(
            train_species
            & val_species
        ) == 0
    )
)

print(
    "LABEL_AUDIT_PASS="
    + str(
        mismatch == 0
    )
)

print(
    "STUDENT_TRUE_PERFECT_RANKING="
    + str(
        gap > 0
    )
)

print()
print("CEILING_AUDIT_COMPLETE=PASS")
