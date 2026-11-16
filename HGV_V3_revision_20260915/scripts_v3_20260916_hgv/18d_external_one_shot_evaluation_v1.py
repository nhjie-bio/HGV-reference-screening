#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import cdist
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


BASE = Path(
    "HGV_V3_revision_20260915"
)

MASTER = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916/"
      "final_master_manifest_3154.tsv"
)

EXTERNAL = (
    BASE
    / "04_splits/g1_g4_v1_20260916/manifests/"
      "external_holdout_780.tsv"
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

TRAIN_ROOT = (
    BASE
    / "09_external_validation/"
      "canonical_A_model_training_v1_20260917"
)

THRESHOLD_ROOT = (
    BASE
    / "09_external_validation/"
      "canonical_validation_threshold_freeze_v1_20260917"
)

THRESHOLD_FILE = (
    THRESHOLD_ROOT
    / "FROZEN_CANONICAL_VALIDATION_THRESHOLDS.tsv"
)

PCA_FILE = (
    THRESHOLD_ROOT
    / "PCA64_G1_TRAIN_FROZEN.npz"
)

PROTOCOL = (
    BASE
    / "09_external_validation/"
      "external_eval_protocol_freeze_v1_20260917/"
      "EXTERNAL_EVAL_PROTOCOL.txt"
)

OUT = (
    BASE
    / "09_external_validation/"
      "external_one_shot_evaluation_v1_20260917"
)


MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
]

HGV_METHOD = (
    "HGV_Dashing2_tau0p98"
)

BASELINES = [
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
]

COHORTS = [
    "B1",
    "B2_seen_genus",
    "B2_unseen_genus",
]

EMBED_BATCH = 1024


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


def confusion_binary(
    y,
    score,
    threshold,
):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    s = np.asarray(
        score,
        dtype=np.float64,
    )

    pred = (
        s >= threshold
    ).astype(np.int64)

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
        2.0
        * precision
        * recall
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
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


class HGV(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            2080,
            256,
        )

        self.bn1 = nn.BatchNorm1d(
            256
        )

        self.fc2 = nn.Linear(
            256,
            128,
        )

        self.bn2 = nn.BatchNorm1d(
            128
        )

        self.fc3 = nn.Linear(
            128,
            64,
        )

    def forward(self, x):

        x = F.relu(
            self.bn1(
                self.fc1(x)
            )
        )

        x = F.relu(
            self.bn2(
                self.fc2(x)
            )
        )

        x = self.fc3(x)

        return F.normalize(
            x,
            p=2,
            dim=1,
        )


@torch.no_grad()
def embed_all(
    model,
    X_device,
):

    model.eval()

    out = []

    for start in range(
        0,
        X_device.shape[0],
        EMBED_BATCH,
    ):

        end = min(
            start + EMBED_BATCH,
            X_device.shape[0],
        )

        out.append(
            model(
                X_device[
                    start:end
                ]
            ).cpu()
        )

    return (
        torch.cat(
            out,
            dim=0,
        )
        .numpy()
        .astype(
            np.float32
        )
    )


def l2_similarity_matrix(
    Q,
    R,
):

    Q = np.asarray(
        Q,
        dtype=np.float64,
    )

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    q2 = np.sum(
        Q * Q,
        axis=1,
        keepdims=True,
    )

    r2 = np.sum(
        R * R,
        axis=1,
    )[None, :]

    d2 = (
        q2
        + r2
        - 2.0
        * (
            Q @ R.T
        )
    )

    d2 = np.maximum(
        d2,
        0.0,
    )

    d = np.sqrt(
        d2
    )

    return (
        1.0
        - d / 2.0
    )


def negative_euclidean_matrix(
    Q,
    R,
):

    Q = np.asarray(
        Q,
        dtype=np.float64,
    )

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    q2 = np.sum(
        Q * Q,
        axis=1,
        keepdims=True,
    )

    r2 = np.sum(
        R * R,
        axis=1,
    )[None, :]

    d2 = (
        q2
        + r2
        - 2.0
        * (
            Q @ R.T
        )
    )

    return -np.sqrt(
        np.maximum(
            d2,
            0.0,
        )
    )


def cosine_matrix(
    Q,
    R,
):

    Q = np.asarray(
        Q,
        dtype=np.float64,
    )

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    qn = np.linalg.norm(
        Q,
        axis=1,
        keepdims=True,
    )

    rn = np.linalg.norm(
        R,
        axis=1,
        keepdims=True,
    )

    Qn = (
        Q
        / np.clip(
            qn,
            1e-15,
            None,
        )
    )

    Rn = (
        R
        / np.clip(
            rn,
            1e-15,
            None,
        )
    )

    return (
        Qn @ Rn.T
    )


def bray_similarity_matrix(
    Q,
    R,
    chunk=32,
):

    Q = np.asarray(
        Q,
        dtype=np.float64,
    )

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    out = np.empty(
        (
            Q.shape[0],
            R.shape[0],
        ),
        dtype=np.float64,
    )

    for start in range(
        0,
        Q.shape[0],
        chunk,
    ):

        end = min(
            start + chunk,
            Q.shape[0],
        )

        dist = cdist(
            Q[start:end],
            R,
            metric="cityblock",
        )

        out[start:end] = (
            1.0
            - 0.5 * dist
        )

        print(
            f"  BrayCurtis rows "
            f"{end}/{Q.shape[0]}"
        )

    return out


def first_relevant_rank(
    order,
    ref_tax,
    query_tax,
):

    if not query_tax:
        return None

    hit = np.flatnonzero(
        ref_tax[order]
        == query_tax
    )

    if len(hit) == 0:
        return None

    return int(
        hit[0] + 1
    )


def retrieval_summary(
    ranks,
):

    valid = [
        int(x)
        for x in ranks
        if x is not None
    ]

    n = len(valid)

    if n == 0:

        return {
            "eligible": 0,
            "R1": float("nan"),
            "R5": float("nan"),
            "R10": float("nan"),
            "MRR": float("nan"),
        }

    a = np.asarray(
        valid,
        dtype=np.int64,
    )

    return {
        "eligible": n,
        "R1": float(
            np.mean(
                a <= 1
            )
        ),
        "R5": float(
            np.mean(
                a <= 5
            )
        ),
        "R10": float(
            np.mean(
                a <= 10
            )
        ),
        "MRR": float(
            np.mean(
                1.0 / a
            )
        ),
    }


def safe_mean_sd(
    values,
):

    vals = [
        float(x)
        for x in values
        if x is not None
        and math.isfinite(
            float(x)
        )
    ]

    if not vals:
        return (
            float("nan"),
            float("nan"),
        )

    mean = statistics.mean(
        vals
    )

    sd = (
        statistics.stdev(
            vals
        )
        if len(vals) > 1
        else 0.0
    )

    return mean, sd


# ==============================================================
# PRECHECK
# ==============================================================

print("=" * 78)
print("HGV-V3 LOCKED EXTERNAL ONE-SHOT EVALUATION")
print("=" * 78)

print("G1_test_used=False")
print("external_performance_now_opened=True")

if OUT.exists():

    raise SystemExit(
        f"ERROR: output already exists: {OUT}"
    )

if not PROTOCOL.is_file():

    raise RuntimeError(
        "Missing frozen external protocol."
    )

# ------------------------------------------------------------
# Load thresholds
# ------------------------------------------------------------

tf, tr = read_tsv(
    THRESHOLD_FILE
)

hgv_threshold = {}
baseline_threshold = {}

for r in tr:

    if r[
        "method_type"
    ] == "HGV":

        seed = int(
            r[
                "model_seed"
            ]
        )

        hgv_threshold[
            seed
        ] = float(
            r[
                "threshold"
            ]
        )

    else:

        baseline_threshold[
            r[
                "method"
            ]
        ] = float(
            r[
                "threshold"
            ]
        )

if len(hgv_threshold) != 3:
    raise RuntimeError(
        "Expected 3 HGV thresholds."
    )

if len(baseline_threshold) != 5:
    raise RuntimeError(
        "Expected 5 baseline thresholds."
    )

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

mf, mr = read_tsv(
    MASTER
)

ef, er = read_tsv(
    EXTERNAL
)

ff, fr = read_tsv(
    FEATURE_ROWS
)

m_acc_col = find_col(
    mf,
    [
        "assembly_accession_final",
        "assembly_accession",
    ],
    "master accession",
)

f_acc_col = find_col(
    ff,
    [
        "assembly_accession",
        "assembly_accession_final",
    ],
    "feature accession",
)

species_col = (
    "gtdb_species_cluster"
)

genus_col = (
    "gtdb_genus"
)

family_col = (
    "gtdb_family"
)

A_rows = [
    r
    for r in mr
    if clean(
        r[
            "final_cohort"
        ]
    ) == "A"
]

if len(A_rows) != 2374:

    raise RuntimeError(
        f"Expected 2374 A genomes; "
        f"got {len(A_rows)}"
    )

cohort_counts = defaultdict(
    int
)

for r in er:

    cohort_counts[
        clean(
            r[
                "final_cohort"
            ]
        )
    ] += 1

expected_counts = {
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193,
}

if dict(
    cohort_counts
) != expected_counts:

    raise RuntimeError(
        f"Unexpected external cohorts: "
        f"{dict(cohort_counts)}"
    )

# ------------------------------------------------------------
# Feature mapping
# ------------------------------------------------------------

feature_accessions = [
    clean(
        r[
            f_acc_col
        ]
    )
    for r in fr
]

acc_to_idx = {
    a: i
    for i, a
    in enumerate(
        feature_accessions
    )
}

if len(acc_to_idx) != 3154:

    raise RuntimeError(
        "Feature accession count != 3154"
    )

A_acc = [
    clean(
        r[
            m_acc_col
        ]
    )
    for r in A_rows
]

E_acc = [
    clean(
        r[
            "assembly_accession_final"
        ]
    )
    for r in er
]

if set(A_acc) & set(E_acc):

    raise RuntimeError(
        "External accession overlap with A."
    )

for a in (
    A_acc
    + E_acc
):

    if a not in acc_to_idx:

        raise RuntimeError(
            f"Missing feature accession: {a}"
        )

A_idx = np.asarray([
    acc_to_idx[
        a
    ]
    for a in A_acc
])

E_idx = np.asarray([
    acc_to_idx[
        a
    ]
    for a in E_acc
])

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

# ------------------------------------------------------------
# Taxonomy arrays
# ------------------------------------------------------------

A_species = np.asarray([
    clean(r[species_col])
    for r in A_rows
], dtype=object)

A_genus = np.asarray([
    clean(r[genus_col])
    for r in A_rows
], dtype=object)

A_family = np.asarray([
    clean(r[family_col])
    for r in A_rows
], dtype=object)

E_species = np.asarray([
    clean(r[species_col])
    for r in er
], dtype=object)

E_genus = np.asarray([
    clean(r[genus_col])
    for r in er
], dtype=object)

E_family = np.asarray([
    clean(r[family_col])
    for r in er
], dtype=object)

E_cohort = np.asarray([
    clean(
        r[
            "final_cohort"
        ]
    )
    for r in er
], dtype=object)

# ------------------------------------------------------------
# Verify taxonomy design before scoring
# ------------------------------------------------------------

A_species_set = set(
    A_species.tolist()
)

A_genus_set = set(
    A_genus.tolist()
)

A_family_set = set(
    A_family.tolist()
)

b1 = (
    E_cohort == "B1"
)

b2s = (
    E_cohort
    == "B2_seen_genus"
)

b2u = (
    E_cohort
    == "B2_unseen_genus"
)

if not np.all([
    x in A_species_set
    for x in E_species[b1]
]):

    raise RuntimeError(
        "B1 species-seen condition failed."
    )

if np.any([
    x in A_species_set
    for x in E_species[b2s]
]):

    raise RuntimeError(
        "B2-seen contains species seen in A."
    )

if not np.all([
    x in A_genus_set
    for x in E_genus[b2s]
]):

    raise RuntimeError(
        "B2-seen genus condition failed."
    )

if np.any([
    x in A_species_set
    for x in E_species[b2u]
]):

    raise RuntimeError(
        "B2-unseen contains species seen in A."
    )

if np.any([
    x in A_genus_set
    for x in E_genus[b2u]
]):

    raise RuntimeError(
        "B2-unseen contains genus seen in A."
    )

family_seen_b2u = np.asarray([
    x in A_family_set
    for x in E_family[b2u]
])

if int(
    family_seen_b2u.sum()
) != 143:

    raise RuntimeError(
        "Expected 143 family-seen B2-unseen queries."
    )

print(
    "external_taxonomy_design=PASS"
)

print(
    "A_reference_genomes=2374"
)

print(
    "external_queries=780"
)

# ==============================================================
# Output setup
# ==============================================================

OUT.mkdir(
    parents=True,
)

seed_metric_rows = []
query_rows = []

# ==============================================================
# Evaluation helper
# ==============================================================

def evaluate_scores(
    method_type,
    method,
    model_seed,
    score_all,
    threshold,
):

    print()
    print(
        f"EVALUATE {method} "
        f"seed={model_seed or 'NA'}"
    )

    for cohort in COHORTS:

        mask = (
            E_cohort == cohort
        )

        idx_global = np.flatnonzero(
            mask
        )

        S = score_all[
            mask
        ]

        q_species = E_species[
            mask
        ]

        q_genus = E_genus[
            mask
        ]

        q_family = E_family[
            mask
        ]

        species_ranks = []
        genus_ranks = []
        family_ranks = []

        max_scores = []
        false_counts = []

        for local_i, global_i in enumerate(
            idx_global
        ):

            scores = S[
                local_i
            ]

            order = np.argsort(
                -scores,
                kind="mergesort",
            )

            top = int(
                order[0]
            )

            sp_rank = (
                first_relevant_rank(
                    order,
                    A_species,
                    q_species[
                        local_i
                    ],
                )
            )

            ge_rank = (
                first_relevant_rank(
                    order,
                    A_genus,
                    q_genus[
                        local_i
                    ],
                )
            )

            fa_rank = (
                first_relevant_rank(
                    order,
                    A_family,
                    q_family[
                        local_i
                    ],
                )
            )

            species_ranks.append(
                sp_rank
            )

            genus_ranks.append(
                ge_rank
            )

            family_ranks.append(
                fa_rank
            )

            max_score = float(
                scores[top]
            )

            n_above = int(
                np.sum(
                    scores
                    >= threshold
                )
            )

            max_scores.append(
                max_score
            )

            false_counts.append(
                n_above
            )

            query_rows.append({
                "method_type":
                    method_type,

                "method":
                    method,

                "model_seed":
                    model_seed,

                "cohort":
                    cohort,

                "query_accession":
                    E_acc[
                        global_i
                    ],

                "query_species":
                    q_species[
                        local_i
                    ],

                "query_genus":
                    q_genus[
                        local_i
                    ],

                "query_family":
                    q_family[
                        local_i
                    ],

                "threshold":
                    threshold,

                "max_similarity":
                    max_score,

                "references_above_threshold":
                    n_above,

                "any_reference_above_threshold":
                    int(
                        n_above > 0
                    ),

                "top1_reference":
                    A_acc[
                        top
                    ],

                "top1_species":
                    A_species[
                        top
                    ],

                "top1_genus":
                    A_genus[
                        top
                    ],

                "top1_family":
                    A_family[
                        top
                    ],

                "species_first_rank":
                    (
                        ""
                        if sp_rank is None
                        else sp_rank
                    ),

                "genus_first_rank":
                    (
                        ""
                        if ge_rank is None
                        else ge_rank
                    ),

                "family_first_rank":
                    (
                        ""
                        if fa_rank is None
                        else fa_rank
                    ),
            })

        sp = retrieval_summary(
            species_ranks
        )

        ge = retrieval_summary(
            genus_ranks
        )

        fa = retrieval_summary(
            family_ranks
        )

        max_scores_arr = np.asarray(
            max_scores,
            dtype=np.float64,
        )

        row = {
            "method_type":
                method_type,

            "method":
                method,

            "model_seed":
                model_seed,

            "cohort":
                cohort,

            "n_queries":
                int(
                    mask.sum()
                ),

            "n_references":
                2374,

            "threshold":
                threshold,

            "species_eligible":
                sp["eligible"],

            "species_R1":
                sp["R1"],

            "species_R5":
                sp["R5"],

            "species_R10":
                sp["R10"],

            "species_MRR":
                sp["MRR"],

            "genus_eligible":
                ge["eligible"],

            "genus_R1":
                ge["R1"],

            "genus_R5":
                ge["R5"],

            "genus_R10":
                ge["R10"],

            "genus_MRR":
                ge["MRR"],

            "family_eligible":
                fa["eligible"],

            "family_R1":
                fa["R1"],

            "family_R5":
                fa["R5"],

            "family_R10":
                fa["R10"],

            "family_MRR":
                fa["MRR"],

            "max_similarity_mean":
                float(
                    np.mean(
                        max_scores_arr
                    )
                ),

            "max_similarity_sd_queries":
                float(
                    np.std(
                        max_scores_arr,
                        ddof=1,
                    )
                ),

            "max_similarity_median":
                float(
                    np.median(
                        max_scores_arr
                    )
                ),

            "max_similarity_q95":
                float(
                    np.quantile(
                        max_scores_arr,
                        0.95,
                    )
                ),

            "pair_AUROC":
                float("nan"),

            "pair_AUPRC":
                float("nan"),

            "pair_MCC":
                float("nan"),

            "pair_F1":
                float("nan"),

            "pair_BalancedAccuracy":
                float("nan"),

            "pair_Precision":
                float("nan"),

            "pair_Recall":
                float("nan"),

            "pair_Specificity":
                float("nan"),

            "open_pair_FPR":
                float("nan"),

            "open_query_any_match_rate":
                float("nan"),

            "open_mean_false_refs_per_query":
                float("nan"),
        }

        if cohort == "B1":

            labels = (
                q_species[:, None]
                == A_species[
                    None, :
                ]
            ).astype(
                np.int64
            )

            yflat = labels.ravel()
            sflat = S.ravel()

            row[
                "pair_AUROC"
            ] = float(
                roc_auc_score(
                    yflat,
                    sflat,
                )
            )

            row[
                "pair_AUPRC"
            ] = float(
                average_precision_score(
                    yflat,
                    sflat,
                )
            )

            cm = confusion_binary(
                yflat,
                sflat,
                threshold,
            )

            row[
                "pair_MCC"
            ] = cm["MCC"]

            row[
                "pair_F1"
            ] = cm["F1"]

            row[
                "pair_BalancedAccuracy"
            ] = cm[
                "BalancedAccuracy"
            ]

            row[
                "pair_Precision"
            ] = cm[
                "Precision"
            ]

            row[
                "pair_Recall"
            ] = cm[
                "Recall"
            ]

            row[
                "pair_Specificity"
            ] = cm[
                "Specificity"
            ]

        else:

            predicted_positive = (
                S >= threshold
            )

            row[
                "open_pair_FPR"
            ] = float(
                np.mean(
                    predicted_positive
                )
            )

            row[
                "open_query_any_match_rate"
            ] = float(
                np.mean(
                    np.any(
                        predicted_positive,
                        axis=1,
                    )
                )
            )

            row[
                "open_mean_false_refs_per_query"
            ] = float(
                np.mean(
                    np.sum(
                        predicted_positive,
                        axis=1,
                    )
                )
            )

        seed_metric_rows.append(
            row
        )

        print(
            f"  {cohort}: "
            f"species_R1="
            f"{row['species_R1']} "
            f"genus_R1="
            f"{row['genus_R1']} "
            f"family_R1="
            f"{row['family_R1']} "
            f"open_FPR="
            f"{row['open_pair_FPR']}"
        )


# ==============================================================
# Feature matrices
# ==============================================================

X_A = np.asarray(
    X[A_idx],
    dtype=np.float32,
)

X_E = np.asarray(
    X[E_idx],
    dtype=np.float32,
)

# ==============================================================
# HGV x 3 seeds
# ==============================================================

print()
print("[1] HGV EXTERNAL EVALUATION")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Writable copy avoids mmap warning.
X_device = torch.tensor(
    np.array(
        X,
        dtype=np.float32,
        copy=True,
    ),
    dtype=torch.float32,
    device=device,
)

for seed in MODEL_SEEDS:

    run_dir = (
        TRAIN_ROOT
        / "G1"
        / "split20260915"
        / f"model{seed}"
    )

    ckpt = (
        run_dir
        / "best_model.pt"
    )

    expected_row = [
        r
        for r in tr
        if (
            r[
                "method_type"
            ] == "HGV"
            and int(
                r[
                    "model_seed"
                ]
            ) == seed
        )
    ]

    if len(
        expected_row
    ) != 1:

        raise RuntimeError(
            f"Missing threshold row for seed {seed}"
        )

    expected_sha = (
        expected_row[0][
            "checkpoint_sha256"
        ]
    )

    actual_sha = sha256(
        ckpt
    )

    if actual_sha != expected_sha:

        raise RuntimeError(
            f"Checkpoint SHA mismatch "
            f"for seed {seed}"
        )

    state = torch.load(
        ckpt,
        map_location=device,
        weights_only=False,
    )

    model = HGV().to(
        device
    )

    model.load_state_dict(
        state[
            "model_state_dict"
        ]
    )

    Z = embed_all(
        model,
        X_device,
    )

    score = (
        l2_similarity_matrix(
            Z[E_idx],
            Z[A_idx],
        )
    )

    evaluate_scores(
        "HGV",
        HGV_METHOD,
        seed,
        score,
        hgv_threshold[
            seed
        ],
    )

    del model
    del state
    del Z
    del score

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================
# Raw baselines
# ==============================================================

print()
print("[2] RAW-K6 BASELINES")

print(
    "RawK6_BrayCurtis..."
)

bray = bray_similarity_matrix(
    X_E,
    X_A,
)

evaluate_scores(
    "baseline",
    "RawK6_BrayCurtis",
    "",
    bray,
    baseline_threshold[
        "RawK6_BrayCurtis"
    ],
)

del bray

print(
    "RawK6_Cosine..."
)

cosine = cosine_matrix(
    X_E,
    X_A,
)

evaluate_scores(
    "baseline",
    "RawK6_Cosine",
    "",
    cosine,
    baseline_threshold[
        "RawK6_Cosine"
    ],
)

del cosine

print(
    "RawK6_Euclidean..."
)

raw_euclidean = (
    negative_euclidean_matrix(
        X_E,
        X_A,
    )
)

evaluate_scores(
    "baseline",
    "RawK6_Euclidean",
    "",
    raw_euclidean,
    baseline_threshold[
        "RawK6_Euclidean"
    ],
)

del raw_euclidean


# ==============================================================
# Frozen PCA baselines
# ==============================================================

print()
print("[3] FROZEN PCA64 BASELINES")

p = np.load(
    PCA_FILE
)

pca_mean = p[
    "mean"
].astype(
    np.float32
)

pca_components = p[
    "components"
].astype(
    np.float32
)

P_A = (
    (
        X_A
        - pca_mean
    )
    @ pca_components.T
).astype(
    np.float32
)

P_E = (
    (
        X_E
        - pca_mean
    )
    @ pca_components.T
).astype(
    np.float32
)

pca_euc = (
    negative_euclidean_matrix(
        P_E,
        P_A,
    )
)

evaluate_scores(
    "baseline",
    "PCA64_Euclidean",
    "",
    pca_euc,
    baseline_threshold[
        "PCA64_Euclidean"
    ],
)

del pca_euc

P_A_l2 = (
    P_A
    / np.clip(
        np.linalg.norm(
            P_A,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )
)

P_E_l2 = (
    P_E
    / np.clip(
        np.linalg.norm(
            P_E,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )
)

pca_l2 = (
    l2_similarity_matrix(
        P_E_l2,
        P_A_l2,
    )
)

evaluate_scores(
    "baseline",
    "PCA64_L2",
    "",
    pca_l2,
    baseline_threshold[
        "PCA64_L2"
    ],
)

del pca_l2


# ==============================================================
# Save query-level results
# ==============================================================

print()
print("[4] WRITE QUERY-LEVEL RESULTS")

query_fields = [
    "method_type",
    "method",
    "model_seed",
    "cohort",
    "query_accession",
    "query_species",
    "query_genus",
    "query_family",
    "threshold",
    "max_similarity",
    "references_above_threshold",
    "any_reference_above_threshold",
    "top1_reference",
    "top1_species",
    "top1_genus",
    "top1_family",
    "species_first_rank",
    "genus_first_rank",
    "family_first_rank",
]

write_tsv(
    OUT
    / "external_query_level_results.tsv",
    query_rows,
    query_fields,
)


# ==============================================================
# Save seed/method-level metrics
# ==============================================================

metric_fields = [
    "method_type",
    "method",
    "model_seed",
    "cohort",
    "n_queries",
    "n_references",
    "threshold",

    "species_eligible",
    "species_R1",
    "species_R5",
    "species_R10",
    "species_MRR",

    "genus_eligible",
    "genus_R1",
    "genus_R5",
    "genus_R10",
    "genus_MRR",

    "family_eligible",
    "family_R1",
    "family_R5",
    "family_R10",
    "family_MRR",

    "pair_AUROC",
    "pair_AUPRC",
    "pair_MCC",
    "pair_F1",
    "pair_BalancedAccuracy",
    "pair_Precision",
    "pair_Recall",
    "pair_Specificity",

    "open_pair_FPR",
    "open_query_any_match_rate",
    "open_mean_false_refs_per_query",

    "max_similarity_mean",
    "max_similarity_sd_queries",
    "max_similarity_median",
    "max_similarity_q95",
]

write_tsv(
    OUT
    / "external_metrics_seed_level.tsv",
    seed_metric_rows,
    metric_fields,
)


# ==============================================================
# Aggregate HGV seeds
# ==============================================================

print()
print("[5] AGGREGATE HGV SEEDS")

summary_rows = []

numeric_metrics = [
    x
    for x in metric_fields
    if x not in {
        "method_type",
        "method",
        "model_seed",
        "cohort",
        "n_queries",
        "n_references",
        "threshold",
        "species_eligible",
        "genus_eligible",
        "family_eligible",
    }
]

groups = defaultdict(
    list
)

for r in seed_metric_rows:

    groups[
        (
            r[
                "method_type"
            ],
            r[
                "method"
            ],
            r[
                "cohort"
            ],
        )
    ].append(
        r
    )

for key, rows in sorted(
    groups.items()
):

    mtype, method, cohort = key

    expected = (
        3
        if mtype == "HGV"
        else 1
    )

    if len(rows) != expected:

        raise RuntimeError(
            f"Unexpected replicate count "
            f"{key}: {len(rows)}"
        )

    for metric in numeric_metrics:

        mean, sd = safe_mean_sd([
            r[
                metric
            ]
            for r in rows
        ])

        summary_rows.append({
            "method_type":
                mtype,

            "method":
                method,

            "cohort":
                cohort,

            "metric":
                metric,

            "n_model_seeds":
                len(rows),

            "mean":
                mean,

            "sd_model_seed":
                sd,
        })

write_tsv(
    OUT
    / "external_metrics_summary_long.tsv",
    summary_rows,
    [
        "method_type",
        "method",
        "cohort",
        "metric",
        "n_model_seeds",
        "mean",
        "sd_model_seed",
    ],
)


# ==============================================================
# Human-readable report
# ==============================================================

print()
print("[6] WRITE REPORT")


def lookup_summary(
    method,
    cohort,
    metric,
):

    rows = [
        r
        for r in summary_rows
        if (
            r[
                "method"
            ] == method
            and r[
                "cohort"
            ] == cohort
            and r[
                "metric"
            ] == metric
        )
    ]

    if len(rows) != 1:

        raise RuntimeError(
            f"Summary lookup failed: "
            f"{method}/{cohort}/{metric}"
        )

    return (
        rows[0][
            "mean"
        ],
        rows[0][
            "sd_model_seed"
        ],
    )


report = (
    OUT
    / "EXTERNAL_EVALUATION_REPORT.txt"
)

methods = [
    HGV_METHOD,
    *BASELINES,
]

with open(
    report,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 LOCKED EXTERNAL ONE-SHOT EVALUATION\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "STATUS: PASS\n\n"
    )

    f.write(
        "Dataset-A references: 2374\n"
    )

    f.write(
        "External queries: 780\n"
    )

    f.write(
        "B1: 391\n"
    )

    f.write(
        "B2_seen_genus: 196\n"
    )

    f.write(
        "B2_unseen_genus: 193\n\n"
    )

    f.write(
        "G1 test used: NO\n"
    )

    f.write(
        "External tuning: NO\n\n"
    )

    f.write(
        "B1 MATCHED-SPECIES CROSS-SOURCE\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    for method in methods:

        r1, r1sd = (
            lookup_summary(
                method,
                "B1",
                "species_R1",
            )
        )

        r10, r10sd = (
            lookup_summary(
                method,
                "B1",
                "species_R10",
            )
        )

        mrr, mrrsd = (
            lookup_summary(
                method,
                "B1",
                "species_MRR",
            )
        )

        auc, aucsd = (
            lookup_summary(
                method,
                "B1",
                "pair_AUROC",
            )
        )

        ap, apsd = (
            lookup_summary(
                method,
                "B1",
                "pair_AUPRC",
            )
        )

        mcc, mccsd = (
            lookup_summary(
                method,
                "B1",
                "pair_MCC",
            )
        )

        f.write(
            f"{method:24s} "
            f"R@1={r1:.6f}"
            f"+/-{r1sd:.6f}; "
            f"R@10={r10:.6f}"
            f"+/-{r10sd:.6f}; "
            f"MRR={mrr:.6f}"
            f"+/-{mrrsd:.6f}; "
            f"AUROC={auc:.6f}"
            f"+/-{aucsd:.6f}; "
            f"AUPRC={ap:.6f}"
            f"+/-{apsd:.6f}; "
            f"MCC={mcc:.6f}"
            f"+/-{mccsd:.6f}\n"
        )

    f.write("\n")
    f.write(
        "B2 SEEN-GENUS / UNSEEN-SPECIES\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    for method in methods:

        gr1, gr1sd = (
            lookup_summary(
                method,
                "B2_seen_genus",
                "genus_R1",
            )
        )

        gr10, gr10sd = (
            lookup_summary(
                method,
                "B2_seen_genus",
                "genus_R10",
            )
        )

        gmrr, gmrrsd = (
            lookup_summary(
                method,
                "B2_seen_genus",
                "genus_MRR",
            )
        )

        fpr, fprsd = (
            lookup_summary(
                method,
                "B2_seen_genus",
                "open_pair_FPR",
            )
        )

        qfp, qfpsd = (
            lookup_summary(
                method,
                "B2_seen_genus",
                "open_query_any_match_rate",
            )
        )

        f.write(
            f"{method:24s} "
            f"genusR@1={gr1:.6f}"
            f"+/-{gr1sd:.6f}; "
            f"genusR@10={gr10:.6f}"
            f"+/-{gr10sd:.6f}; "
            f"genusMRR={gmrr:.6f}"
            f"+/-{gmrrsd:.6f}; "
            f"pairFPR={fpr:.6f}"
            f"+/-{fprsd:.6f}; "
            f"queryAnyFalseMatch="
            f"{qfp:.6f}"
            f"+/-{qfpsd:.6f}\n"
        )

    f.write("\n")
    f.write(
        "B2 UNSEEN-GENUS / UNSEEN-SPECIES\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    f.write(
        "Family retrieval applies to 143/193 queries "
        "whose family exists in Dataset A.\n\n"
    )

    for method in methods:

        fr1, fr1sd = (
            lookup_summary(
                method,
                "B2_unseen_genus",
                "family_R1",
            )
        )

        fr10, fr10sd = (
            lookup_summary(
                method,
                "B2_unseen_genus",
                "family_R10",
            )
        )

        fmrr, fmrrsd = (
            lookup_summary(
                method,
                "B2_unseen_genus",
                "family_MRR",
            )
        )

        fpr, fprsd = (
            lookup_summary(
                method,
                "B2_unseen_genus",
                "open_pair_FPR",
            )
        )

        qfp, qfpsd = (
            lookup_summary(
                method,
                "B2_unseen_genus",
                "open_query_any_match_rate",
            )
        )

        f.write(
            f"{method:24s} "
            f"familyR@1={fr1:.6f}"
            f"+/-{fr1sd:.6f}; "
            f"familyR@10={fr10:.6f}"
            f"+/-{fr10sd:.6f}; "
            f"familyMRR={fmrr:.6f}"
            f"+/-{fmrrsd:.6f}; "
            f"pairFPR={fpr:.6f}"
            f"+/-{fprsd:.6f}; "
            f"queryAnyFalseMatch="
            f"{qfp:.6f}"
            f"+/-{qfpsd:.6f}\n"
        )

    f.write("\n")

    f.write(
        "No external result was used to alter model, "
        "teacher, tau, lambda, checkpoint, PCA transform, "
        "model seed set, or validation-derived threshold.\n\n"
    )

    f.write(
        "EXTERNAL_ONE_SHOT_EVALUATION=PASS\n"
    )


# ==============================================================
# Audit + provenance
# ==============================================================

audit_rows = [
    {
        "cohort":
            "B1",

        "n":
            391,

        "species_seen_A":
            391,

        "genus_seen_A":
            391,

        "family_seen_A":
            391,
    },
    {
        "cohort":
            "B2_seen_genus",

        "n":
            196,

        "species_seen_A":
            0,

        "genus_seen_A":
            196,

        "family_seen_A":
            196,
    },
    {
        "cohort":
            "B2_unseen_genus",

        "n":
            193,

        "species_seen_A":
            0,

        "genus_seen_A":
            0,

        "family_seen_A":
            143,
    },
]

write_tsv(
    OUT
    / "EXTERNAL_COHORT_AUDIT.tsv",
    audit_rows,
    [
        "cohort",
        "n",
        "species_seen_A",
        "genus_seen_A",
        "family_seen_A",
    ],
)

provenance = {
    "status":
        "PASS",

    "reference_genomes":
        2374,

    "external_queries":
        780,

    "B1":
        391,

    "B2_seen_genus":
        196,

    "B2_unseen_genus":
        193,

    "G1_test_used":
        False,

    "external_tuning":
        False,

    "HGV_model_seeds":
        MODEL_SEEDS,

    "threshold_file_sha256":
        sha256(
            THRESHOLD_FILE
        ),

    "PCA_file_sha256":
        sha256(
            PCA_FILE
        ),

    "external_manifest_sha256":
        sha256(
            EXTERNAL
        ),

    "protocol_sha256":
        sha256(
            PROTOCOL
        ),

    "python":
        sys.version,

    "numpy":
        np.__version__,

    "torch":
        torch.__version__,

    "platform":
        platform.platform(),
}

(
    OUT
    / "external_evaluation_provenance.json"
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
# SHA256
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
print("EXTERNAL ONE-SHOT EVALUATION COMPLETE")
print("=" * 78)

print(
    f"Output: {OUT}"
)

print(
    "G1_test_used=False"
)

print(
    "external_tuning=False"
)

print(
    "EXTERNAL_ONE_SHOT_EVALUATION=PASS"
)
