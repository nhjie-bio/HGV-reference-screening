#!/usr/bin/env python3

import os
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path(
    "HGV_V3_revision_20260915"
)

FEATURE_DIR = (
    BASE
    / "03_features/canonical_k6_full3154_v1"
)

FEATURE_NPY = (
    FEATURE_DIR
    / "canonical_k6_features.float32.npy"
)

FEATURE_ROWS = (
    FEATURE_DIR
    / "canonical_k6_rows.tsv"
)

PAIR_DIR = (
    BASE
    / "06_teacher/teacher_geometry_audit_v1_20260916"
    / "teacher_augmented/G2_seed20260915"
)

TRAIN_TSV = (
    PAIR_DIR
    / "train_teacher.tsv"
)

VAL_TSV = (
    PAIR_DIR
    / "val_teacher.tsv"
)

PREFLIGHT_DIR = (
    BASE
    / "07_hgv/teacher_selection_preflight_v1_20260916"
)

TRAINING_DEF = (
    PREFLIGHT_DIR
    / "proposed_training_definition.json"
)


FULL_OUT = (
    BASE
    / "07_hgv/teacher_selection_training_v4_20260916"
)

SMOKE_OUT = (
    BASE
    / "07_hgv/teacher_selection_smoke_v4_20260916"
)


MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
]


CONFIGS = [
    {
        "config_id": "D0_contrastive",
        "variant": "D0",
        "teacher": "none",
        "teacher_field": None,
        "tau": None,
        "lambda_dist": 0.0,
    },

    {
        "config_id": "D3_Mash_tau0p90",
        "variant": "D3",
        "teacher": "Mash",
        "teacher_field": "mash_distance",
        "tau": 0.90,
        "lambda_dist": 0.25,
    },

    {
        "config_id": "D3_Dashing2_tau0p90",
        "variant": "D3",
        "teacher": "Dashing2_SetSketch",
        "teacher_field": "dashing2_set_distance",
        "tau": 0.90,
        "lambda_dist": 0.25,
    },

    {
        "config_id": "D3_Dashing2_tau0p95",
        "variant": "D3",
        "teacher": "Dashing2_SetSketch",
        "teacher_field": "dashing2_set_distance",
        "tau": 0.95,
        "lambda_dist": 0.25,
    },

    {
        "config_id": "D3_Dashing2_tau0p98",
        "variant": "D3",
        "teacher": "Dashing2_SetSketch",
        "teacher_field": "dashing2_set_distance",
        "tau": 0.98,
        "lambda_dist": 0.25,
    },
]


# ----------------------------------------------------------------------
# Frozen HGV hyperparameters
# ----------------------------------------------------------------------

INPUT_DIM = 2080
HIDDEN1 = 256
HIDDEN2 = 128
EMBED_DIM = 64

MARGIN = 1.0

LR = 5e-4
WEIGHT_DECAY = 1e-4

TRAIN_BATCH = 512
VAL_BATCH = 1024

MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0

EPOCH_EXAMPLES = 100000
EPOCH_POSITIVE = 50000
EPOCH_NEGATIVE = 50000

IMPROVEMENT_EPS = 1e-8
LOSS_TIE_EPS = 1e-8


def checkpoint_better(
    new_score,
    new_contrastive,
    best_score,
    best_contrastive,
):
    """
    Ceiling-aware lexicographic checkpoint rule.

    Primary criterion:
        maximize validation ranking score
        (AUROC + AUPRC) / 2

    Secondary criterion:
        if ranking scores are tied within IMPROVEMENT_EPS,
        minimize teacher-independent validation contrastive loss.
    """

    if new_score > best_score + IMPROVEMENT_EPS:
        return True, "ranking_score"

    if abs(new_score - best_score) <= IMPROVEMENT_EPS:
        if new_contrastive < best_contrastive - LOSS_TIE_EPS:
            return True, "contrastive_tiebreak"

    return False, "none"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def clean(x):
    return "" if x is None else str(x).strip()


def read_tsv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        raise RuntimeError(
            f"No TSV header: {path}"
        )

    return fields, rows


def write_tsv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for r in rows:
            writer.writerow({
                k: r.get(k, "")
                for k in fields
            })

    tmp.replace(path)


def write_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    tmp.replace(path)


def sha256_bytes(b):
    return hashlib.sha256(
        b
    ).hexdigest()


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


def safe_mean(vals):
    if not vals:
        return float("nan")

    return float(
        sum(vals) / len(vals)
    )


def safe_sd(vals):
    if len(vals) < 2:
        return 0.0

    return float(
        statistics.stdev(vals)
    )


def average_ranks(values):
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

        avg_rank = (
            (i + 1 + j) / 2.0
        )

        for k in range(i, j):
            ranks[
                order[k]
            ] = avg_rank

        i = j

    return ranks


def roc_auc(labels, scores):
    labels = list(
        map(int, labels)
    )

    scores = list(
        map(float, scores)
    )

    npos = sum(labels)
    nneg = (
        len(labels) - npos
    )

    if npos == 0 or nneg == 0:
        return float("nan")

    ranks = average_ranks(
        scores
    )

    rank_sum_pos = sum(
        ranks[i]
        for i, y in enumerate(labels)
        if y == 1
    )

    return float(
        (
            rank_sum_pos
            - npos * (npos + 1) / 2.0
        )
        / (npos * nneg)
    )


def average_precision(labels, scores):
    labels = list(
        map(int, labels)
    )

    scores = list(
        map(float, scores)
    )

    npos = sum(labels)

    if npos == 0:
        return float("nan")

    groups = defaultdict(
        lambda: [0, 0]
    )

    for y, score in zip(
        labels,
        scores,
    ):
        groups[score][
            0 if y == 1 else 1
        ] += 1

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    for score in sorted(
        groups.keys(),
        reverse=True,
    ):
        gp, gn = groups[
            score
        ]

        tp += gp
        fp += gn

        recall = (
            tp / npos
        )

        precision = (
            tp / (tp + fp)
        )

        ap += (
            recall
            - prev_recall
        ) * precision

        prev_recall = recall

    return float(ap)


def detect_acc_col(fields):
    for c in [
        "assembly_accession",
        "assembly_accession_final",
        "accession",
    ]:
        if c in fields:
            return c

    raise RuntimeError(
        "Feature-row accession "
        "column not detected"
    )


def set_seed(seed):
    random.seed(seed)

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class HGVEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            INPUT_DIM,
            HIDDEN1,
        )

        self.bn1 = nn.BatchNorm1d(
            HIDDEN1
        )

        self.fc2 = nn.Linear(
            HIDDEN1,
            HIDDEN2,
        )

        self.bn2 = nn.BatchNorm1d(
            HIDDEN2
        )

        self.fc3 = nn.Linear(
            HIDDEN2,
            EMBED_DIM,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)

        x = F.normalize(
            x,
            p=2,
            dim=1,
        )

        return x


# ----------------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------------

def prepare_data():

    X = np.load(
        FEATURE_NPY
    ).astype(
        np.float32,
        copy=False,
    )

    if X.shape != (
        3154,
        2080,
    ):
        raise RuntimeError(
            f"Unexpected feature shape: "
            f"{X.shape}"
        )

    fields, rows = read_tsv(
        FEATURE_ROWS
    )

    acc_col = detect_acc_col(
        fields
    )

    acc_to_index = {}

    for i, r in enumerate(rows):
        acc = clean(
            r[acc_col]
        )

        if acc in acc_to_index:
            raise RuntimeError(
                f"Duplicate accession: {acc}"
            )

        acc_to_index[
            acc
        ] = i

    train_fields, train_rows = read_tsv(
        TRAIN_TSV
    )

    val_fields, val_rows = read_tsv(
        VAL_TSV
    )

    def convert(rows):

        a_idx = []
        b_idx = []
        labels = []
        mash = []
        d2 = []
        pair_types = []
        acc_a = []
        acc_b = []

        for r in rows:
            a = clean(
                r["accession_a"]
            )

            b = clean(
                r["accession_b"]
            )

            a_idx.append(
                acc_to_index[a]
            )

            b_idx.append(
                acc_to_index[b]
            )

            labels.append(
                int(r["label"])
            )

            mash.append(
                float(
                    r["mash_distance"]
                )
            )

            d2.append(
                float(
                    r[
                        "dashing2_set_distance"
                    ]
                )
            )

            pair_types.append(
                clean(
                    r.get(
                        "pair_type",
                        "",
                    )
                )
            )

            acc_a.append(a)
            acc_b.append(b)

        return {
            "a_idx": np.asarray(
                a_idx,
                dtype=np.int64,
            ),
            "b_idx": np.asarray(
                b_idx,
                dtype=np.int64,
            ),
            "labels": np.asarray(
                labels,
                dtype=np.float32,
            ),
            "mash_distance": np.asarray(
                mash,
                dtype=np.float32,
            ),
            "dashing2_set_distance": np.asarray(
                d2,
                dtype=np.float32,
            ),
            "pair_type": np.asarray(
                pair_types,
                dtype=object,
            ),
            "accession_a": np.asarray(
                acc_a,
                dtype=object,
            ),
            "accession_b": np.asarray(
                acc_b,
                dtype=object,
            ),
        }

    return (
        X,
        convert(train_rows),
        convert(val_rows),
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

@torch.no_grad()
def embed_all(
    model,
    X_device,
    batch_size=1024,
):

    model.eval()

    result = []

    for start in range(
        0,
        X_device.shape[0],
        batch_size,
    ):
        end = min(
            start + batch_size,
            X_device.shape[0],
        )

        z = model(
            X_device[
                start:end
            ]
        )

        result.append(
            z
        )

    return torch.cat(
        result,
        dim=0,
    )


@torch.no_grad()
def validate(
    model,
    X_device,
    val,
    config,
    device,
):

    Z = embed_all(
        model,
        X_device,
        batch_size=1024,
    )

    a_idx = torch.as_tensor(
        val["a_idx"],
        dtype=torch.long,
        device=device,
    )

    b_idx = torch.as_tensor(
        val["b_idx"],
        dtype=torch.long,
        device=device,
    )

    y = torch.as_tensor(
        val["labels"],
        dtype=torch.float32,
        device=device,
    )

    distances = []

    for start in range(
        0,
        len(y),
        VAL_BATCH,
    ):
        end = min(
            start + VAL_BATCH,
            len(y),
        )

        za = Z[
            a_idx[start:end]
        ]

        zb = Z[
            b_idx[start:end]
        ]

        d = torch.linalg.vector_norm(
            za - zb,
            ord=2,
            dim=1,
        )

        distances.append(
            d
        )

    d = torch.cat(
        distances
    )

    similarity = (
        1.0
        - d / 2.0
    )

    contrastive = (
        y * d.square()
        +
        (1.0 - y)
        * F.relu(
            MARGIN - d
        ).square()
    ).mean()

    labels_np = (
        y.detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    distance_np = (
        d.detach()
        .cpu()
        .numpy()
    )

    similarity_np = (
        similarity.detach()
        .cpu()
        .numpy()
    )

    auc = roc_auc(
        labels_np,
        similarity_np,
    )

    auprc = average_precision(
        labels_np,
        similarity_np,
    )

    score = (
        auc + auprc
    ) / 2.0

    distill_loss = float("nan")

    distill_mask_np = np.zeros(
        len(labels_np),
        dtype=bool,
    )

    teacher_np = np.full(
        len(labels_np),
        np.nan,
        dtype=np.float32,
    )

    if (
        config[
            "teacher_field"
        ]
        is not None
    ):

        teacher_np = val[
            config[
                "teacher_field"
            ]
        ]

        distill_mask_np = (
            (labels_np == 1)
            |
            (
                (labels_np == 0)
                &
                (
                    teacher_np
                    < config["tau"]
                )
            )
        )

        if distill_mask_np.any():

            mask = torch.as_tensor(
                distill_mask_np,
                dtype=torch.bool,
                device=device,
            )

            teacher_t = torch.as_tensor(
                teacher_np,
                dtype=torch.float32,
                device=device,
            )

            distill_loss = (
                (
                    d[mask] / 2.0
                    - teacher_t[mask]
                )
                .square()
                .mean()
                .item()
            )

    result = {
        "val_AUROC": auc,
        "val_AUPRC": auprc,
        "val_selection_score": score,
        "val_contrastive_loss": (
            contrastive.item()
        ),
        "val_distill_loss": (
            distill_loss
        ),
        "distance": distance_np,
        "similarity": similarity_np,
        "teacher": teacher_np,
        "distill_mask": (
            distill_mask_np
        ),
    }

    return result


# ----------------------------------------------------------------------
# Training one run
# ----------------------------------------------------------------------

def train_one(
    X,
    train,
    val,
    config,
    seed,
    run_dir,
    smoke=False,
):

    if run_dir.exists():
        if (
            run_dir
            / "RUN_COMPLETE"
        ).is_file():

            print(
                f"SKIP COMPLETE: "
                f"{run_dir.name}"
            )

            return

        raise RuntimeError(
            "Incomplete output already "
            f"exists: {run_dir}"
        )

    tmp_dir = Path(
        str(run_dir)
        + f".tmp_{os.getpid()}"
    )

    if tmp_dir.exists():
        shutil.rmtree(
            tmp_dir
        )

    tmp_dir.mkdir(
        parents=True
    )

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X_device = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    model = HGVEncoder().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    labels_np = train[
        "labels"
    ].astype(
        np.int64
    )

    pos_pool = np.flatnonzero(
        labels_np == 1
    )

    neg_pool = np.flatnonzero(
        labels_np == 0
    )

    if (
        len(pos_pool) != 9088
        or len(neg_pool) != 9088
    ):
        raise RuntimeError(
            "Unexpected frozen "
            "training pool sizes"
        )

    max_epochs = (
        2
        if smoke
        else MAX_EPOCHS
    )

    epoch_positive = (
        5000
        if smoke
        else EPOCH_POSITIVE
    )

    epoch_negative = (
        5000
        if smoke
        else EPOCH_NEGATIVE
    )

    history = []

    best_score = -math.inf
    best_contrastive = math.inf
    best_epoch = 0
    best_reason = ""
    epochs_without_improvement = 0

    best_path = (
        tmp_dir
        / "best_model.pt"
    )

    print()
    print(
        f"RUN {config['config_id']} "
        f"seed={seed} "
        f"device={device}"
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        rng = np.random.default_rng(
            seed
            + epoch * 1000003
        )

        pos_draw = pos_pool[
            rng.integers(
                0,
                len(pos_pool),
                size=epoch_positive,
            )
        ]

        neg_draw = neg_pool[
            rng.integers(
                0,
                len(neg_pool),
                size=epoch_negative,
            )
        ]

        sampled = np.concatenate([
            pos_draw,
            neg_draw,
        ])

        sampled = sampled[
            rng.permutation(
                len(sampled)
            )
        ]

        sampler_hash = sha256_bytes(
            sampled
            .astype("<i8")
            .tobytes()
        )

        epoch_total = 0.0
        epoch_con = 0.0
        epoch_dist = 0.0
        epoch_distill_batches = 0
        seen = 0

        for start in range(
            0,
            len(sampled),
            TRAIN_BATCH,
        ):

            batch_ids = sampled[
                start:
                start + TRAIN_BATCH
            ]

            a_idx = torch.as_tensor(
                train["a_idx"][
                    batch_ids
                ],
                dtype=torch.long,
                device=device,
            )

            b_idx = torch.as_tensor(
                train["b_idx"][
                    batch_ids
                ],
                dtype=torch.long,
                device=device,
            )

            y = torch.as_tensor(
                train["labels"][
                    batch_ids
                ],
                dtype=torch.float32,
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # BN-safe Siamese forward:
            # concatenate both pair endpoints so that za and zb
            # are normalized using the same mini-batch statistics.
            x_pair = torch.cat(
                [
                    X_device[a_idx],
                    X_device[b_idx],
                ],
                dim=0,
            )

            z_pair = model(
                x_pair
            )

            za, zb = torch.chunk(
                z_pair,
                2,
                dim=0,
            )

            d = torch.linalg.vector_norm(
                za - zb,
                ord=2,
                dim=1,
            )

            loss_con = (
                y * d.square()
                +
                (1.0 - y)
                * F.relu(
                    MARGIN - d
                ).square()
            ).mean()

            loss_dist = torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

            if (
                config[
                    "teacher_field"
                ]
                is not None
            ):

                teacher = torch.as_tensor(
                    train[
                        config[
                            "teacher_field"
                        ]
                    ][batch_ids],
                    dtype=torch.float32,
                    device=device,
                )

                mask = (
                    (y == 1)
                    |
                    (
                        (y == 0)
                        &
                        (
                            teacher
                            < config["tau"]
                        )
                    )
                )

                if mask.any():

                    loss_dist = (
                        (
                            d[mask] / 2.0
                            - teacher[mask]
                        )
                        .square()
                        .mean()
                    )

                    epoch_distill_batches += 1

            loss = (
                loss_con
                +
                config[
                    "lambda_dist"
                ]
                * loss_dist
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            n_batch = len(
                batch_ids
            )

            epoch_total += (
                loss.item()
                * n_batch
            )

            epoch_con += (
                loss_con.item()
                * n_batch
            )

            epoch_dist += (
                loss_dist.item()
                * n_batch
            )

            seen += n_batch

        val_result = validate(
            model,
            X_device,
            val,
            config,
            device,
        )

        row = {
            "epoch": epoch,
            "sampled_examples": seen,
            "sampler_sha256": sampler_hash,
            "train_total_loss": (
                epoch_total / seen
            ),
            "train_contrastive_loss": (
                epoch_con / seen
            ),
            "train_distill_loss": (
                epoch_dist / seen
            ),
            "distill_batches": (
                epoch_distill_batches
            ),
            "val_AUROC": (
                val_result[
                    "val_AUROC"
                ]
            ),
            "val_AUPRC": (
                val_result[
                    "val_AUPRC"
                ]
            ),
            "val_selection_score": (
                val_result[
                    "val_selection_score"
                ]
            ),
            "val_contrastive_loss": (
                val_result[
                    "val_contrastive_loss"
                ]
            ),
            "val_distill_loss": (
                val_result[
                    "val_distill_loss"
                ]
            ),
        }

        history.append(row)

        score = row[
            "val_selection_score"
        ]

        improved, improvement_reason = checkpoint_better(
            new_score=score,
            new_contrastive=row["val_contrastive_loss"],
            best_score=best_score,
            best_contrastive=best_contrastive,
        )

        if improved:

            best_score = score
            best_contrastive = row["val_contrastive_loss"]
            best_epoch = epoch
            best_reason = improvement_reason
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "val_selection_score": (
                        best_score
                    ),
                    "val_contrastive_loss": (
                        best_contrastive
                    ),
                    "checkpoint_reason": (
                        best_reason
                    ),
                },
                best_path,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:02d} "
            f"train={row['train_total_loss']:.6f} "
            f"AUC={row['val_AUROC']:.6f} "
            f"AUPRC={row['val_AUPRC']:.6f} "
            f"score={score:.6f} "
            f"best={best_score:.6f} "
            f"best_con={best_contrastive:.6f} "
            f"update_reason={improvement_reason} "
            f"best_reason={best_reason} "
            f"patience="
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

        if (
            not smoke
            and
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                f"EARLY_STOP epoch={epoch}"
            )
            break

    # ------------------------------------------------------------------
    # Restore best checkpoint
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    final_val = validate(
        model,
        X_device,
        val,
        config,
        device,
    )

    # ------------------------------------------------------------------
    # Validation predictions
    # ------------------------------------------------------------------

    pred_rows = []

    for i in range(
        len(val["labels"])
    ):

        pred_rows.append({
            "accession_a": (
                val[
                    "accession_a"
                ][i]
            ),
            "accession_b": (
                val[
                    "accession_b"
                ][i]
            ),
            "label": int(
                val["labels"][i]
            ),
            "pair_type": (
                val[
                    "pair_type"
                ][i]
            ),
            "teacher": (
                config["teacher"]
            ),
            "tau": (
                ""
                if config["tau"] is None
                else config["tau"]
            ),
            "teacher_distance": (
                ""
                if (
                    config[
                        "teacher_field"
                    ]
                    is None
                )
                else float(
                    final_val[
                        "teacher"
                    ][i]
                )
            ),
            "distill_mask": int(
                final_val[
                    "distill_mask"
                ][i]
            ),
            "student_distance": float(
                final_val[
                    "distance"
                ][i]
            ),
            "student_similarity": float(
                final_val[
                    "similarity"
                ][i]
            ),
        })

    write_tsv(
        tmp_dir
        / "val_predictions.tsv",
        pred_rows,
        [
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
            "teacher",
            "tau",
            "teacher_distance",
            "distill_mask",
            "student_distance",
            "student_similarity",
        ],
    )

    # ------------------------------------------------------------------
    # Stratum AUROC/AUPRC
    # positive vs each negative tier
    # ------------------------------------------------------------------

    stratum_rows = []

    labels_all = val[
        "labels"
    ].astype(int)

    scores_all = final_val[
        "similarity"
    ]

    pair_types = val[
        "pair_type"
    ]

    for stratum in [
        "all",
        "hard",
        "medium",
        "easy",
    ]:

        if stratum == "all":
            mask = np.ones(
                len(labels_all),
                dtype=bool,
            )

        else:
            mask = (
                (labels_all == 1)
                |
                (
                    pair_types
                    == stratum
                )
            )

        y = labels_all[
            mask
        ]

        s = scores_all[
            mask
        ]

        if (
            len(np.unique(y))
            < 2
        ):
            continue

        stratum_rows.append({
            "stratum": stratum,
            "n_pairs": len(y),
            "positive_n": int(
                (y == 1).sum()
            ),
            "negative_n": int(
                (y == 0).sum()
            ),
            "AUROC": roc_auc(
                y,
                s,
            ),
            "AUPRC": average_precision(
                y,
                s,
            ),
        })

    write_tsv(
        tmp_dir
        / "val_stratum_metrics.tsv",
        stratum_rows,
        [
            "stratum",
            "n_pairs",
            "positive_n",
            "negative_n",
            "AUROC",
            "AUPRC",
        ],
    )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    write_tsv(
        tmp_dir
        / "history.tsv",
        history,
        [
            "epoch",
            "sampled_examples",
            "sampler_sha256",
            "train_total_loss",
            "train_contrastive_loss",
            "train_distill_loss",
            "distill_batches",
            "val_AUROC",
            "val_AUPRC",
            "val_selection_score",
            "val_contrastive_loss",
            "val_distill_loss",
        ],
    )

    summary = {
        "status": "PASS",
        "smoke": smoke,
        "config_id": (
            config[
                "config_id"
            ]
        ),
        "variant": (
            config["variant"]
        ),
        "teacher": (
            config["teacher"]
        ),
        "tau": (
            config["tau"]
        ),
        "lambda_dist": (
            config[
                "lambda_dist"
            ]
        ),
        "model_seed": seed,
        "device": str(device),
        "torch_version": (
            torch.__version__
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_checkpoint_reason": (
            best_reason
        ),
        "best_checkpoint_val_contrastive_loss": (
            best_contrastive
        ),
        "epochs_ran": (
            len(history)
        ),
        "best_val_AUROC": (
            final_val[
                "val_AUROC"
            ]
        ),
        "best_val_AUPRC": (
            final_val[
                "val_AUPRC"
            ]
        ),
        "best_val_selection_score": (
            final_val[
                "val_selection_score"
            ]
        ),
        "best_val_contrastive_loss": (
            final_val[
                "val_contrastive_loss"
            ]
        ),
        "best_val_distill_loss": (
            final_val[
                "val_distill_loss"
            ]
        ),
        "training_examples_per_epoch": (
            epoch_positive
            + epoch_negative
        ),
        "positive_examples_per_epoch": (
            epoch_positive
        ),
        "negative_examples_per_epoch": (
            epoch_negative
        ),
        "test_used": False,
        "external_used": False,
    }

    write_json(
        tmp_dir
        / "run_summary.json",
        summary,
    )

    # run-level checksums

    hash_files = [
        p
        for p in sorted(
            tmp_dir.iterdir()
        )
        if (
            p.is_file()
            and
            p.name
            not in {
                "SHA256SUMS.txt",
                "RUN_COMPLETE",
            }
        )
    ]

    with open(
        tmp_dir
        / "SHA256SUMS.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for p in hash_files:
            f.write(
                f"{sha256_file(p)}  "
                f"{p.name}\n"
            )

    (
        tmp_dir
        / "RUN_COMPLETE"
    ).write_text(
        "PASS\n",
        encoding="utf-8",
    )

    tmp_dir.rename(
        run_dir
    )

    print(
        f"RUN_PASS "
        f"{run_dir.name} "
        f"score={summary['best_val_selection_score']:.6f}"
    )


# ----------------------------------------------------------------------
# Aggregate completed runs
# ----------------------------------------------------------------------

def aggregate(root):

    run_rows = []

    for run_dir in sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and (
            p
            / "RUN_COMPLETE"
        ).is_file()
    ):

        summary_path = (
            run_dir
            / "run_summary.json"
        )

        if not summary_path.is_file():
            continue

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        run_rows.append({
            "config_id": (
                summary[
                    "config_id"
                ]
            ),
            "variant": (
                summary["variant"]
            ),
            "teacher": (
                summary["teacher"]
            ),
            "tau": (
                ""
                if summary[
                    "tau"
                ] is None
                else summary["tau"]
            ),
            "model_seed": (
                summary[
                    "model_seed"
                ]
            ),
            "best_epoch": (
                summary[
                    "best_epoch"
                ]
            ),
            "epochs_ran": (
                summary[
                    "epochs_ran"
                ]
            ),
            "val_AUROC": (
                summary[
                    "best_val_AUROC"
                ]
            ),
            "val_AUPRC": (
                summary[
                    "best_val_AUPRC"
                ]
            ),
            "val_selection_score": (
                summary[
                    "best_val_selection_score"
                ]
            ),
            "val_contrastive_loss": (
                summary[
                    "best_val_contrastive_loss"
                ]
            ),
            "val_distill_loss": (
                summary[
                    "best_val_distill_loss"
                ]
            ),
        })

    write_tsv(
        root
        / "run_summary.tsv",
        run_rows,
        [
            "config_id",
            "variant",
            "teacher",
            "tau",
            "model_seed",
            "best_epoch",
            "epochs_ran",
            "val_AUROC",
            "val_AUPRC",
            "val_selection_score",
            "val_contrastive_loss",
            "val_distill_loss",
        ],
    )

    by_config = defaultdict(
        list
    )

    for r in run_rows:
        by_config[
            r["config_id"]
        ].append(r)

    config_rows = []

    for config_id in sorted(
        by_config
    ):

        rows = by_config[
            config_id
        ]

        aucs = [
            float(r["val_AUROC"])
            for r in rows
        ]

        aps = [
            float(r["val_AUPRC"])
            for r in rows
        ]

        scores = [
            float(
                r[
                    "val_selection_score"
                ]
            )
            for r in rows
        ]

        config_rows.append({
            "config_id": config_id,
            "variant": (
                rows[0]["variant"]
            ),
            "teacher": (
                rows[0]["teacher"]
            ),
            "tau": (
                rows[0]["tau"]
            ),
            "n_seeds": len(rows),
            "mean_AUROC": (
                safe_mean(aucs)
            ),
            "sd_AUROC": (
                safe_sd(aucs)
            ),
            "mean_AUPRC": (
                safe_mean(aps)
            ),
            "sd_AUPRC": (
                safe_sd(aps)
            ),
            "mean_selection_score": (
                safe_mean(scores)
            ),
            "sd_selection_score": (
                safe_sd(scores)
            ),
        })

    write_tsv(
        root
        / "config_summary.tsv",
        config_rows,
        [
            "config_id",
            "variant",
            "teacher",
            "tau",
            "n_seeds",
            "mean_AUROC",
            "sd_AUROC",
            "mean_AUPRC",
            "sd_AUPRC",
            "mean_selection_score",
            "sd_selection_score",
        ],
    )

    return (
        run_rows,
        config_rows,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "full",
        ],
        default="full",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if not TRAINING_DEF.is_file():
        raise RuntimeError(
            f"Missing frozen training "
            f"definition: {TRAINING_DEF}"
        )

    root = (
        SMOKE_OUT
        if args.mode == "smoke"
        else FULL_OUT
    )

    if root.exists():
        if not args.resume:
            raise RuntimeError(
                f"Output exists: {root}\n"
                "Use --resume only if you "
                "intend to continue this "
                "same frozen grid."
            )
    else:
        root.mkdir(
            parents=True,
            exist_ok=False,
        )

    (
        root
        / "SOURCE_TRAINING_DEFINITION.sha256"
    ).write_text(
        f"{sha256_file(TRAINING_DEF)}  "
        f"{TRAINING_DEF}\n",
        encoding="utf-8",
    )

    X, train, val = (
        prepare_data()
    )

    if args.mode == "smoke":

        smoke_config = next(
            c
            for c in CONFIGS
            if c["config_id"]
            == "D3_Mash_tau0p90"
        )

        jobs = [
            (
                smoke_config,
                20260915,
            )
        ]

    else:

        jobs = [
            (
                config,
                seed,
            )
            for config in CONFIGS
            for seed in MODEL_SEEDS
        ]

    print("=" * 78)
    print(
        "HGV-V3 TEACHER-SELECTION "
        f"{args.mode.upper()}"
    )
    print("=" * 78)

    print(
        f"jobs={len(jobs)}"
    )

    print(
        "test_used=False "
        "external_used=False"
    )

    for config, seed in jobs:

        run_name = (
            f"{config['config_id']}"
            f"__seed{seed}"
        )

        run_dir = (
            root
            / run_name
        )

        train_one(
            X=X,
            train=train,
            val=val,
            config=config,
            seed=seed,
            run_dir=run_dir,
            smoke=(
                args.mode
                == "smoke"
            ),
        )

        aggregate(
            root
        )

    run_rows, config_rows = (
        aggregate(root)
    )

    report = (
        root
        / "HGV_TEACHER_SELECTION_TRAINING_REPORT.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HGV-V3 TEACHER-SELECTION TRAINING\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"MODE: {args.mode}\n"
        )

        f.write(
            f"Completed runs: "
            f"{len(run_rows)}\n"
        )

        f.write(
            "Test used: NO\n"
        )

        f.write(
            "External genomes used: NO\n\n"
        )

        for r in sorted(
            config_rows,
            key=lambda x: (
                -float(
                    x[
                        "mean_selection_score"
                    ]
                ),
                x["config_id"],
            ),
        ):
            f.write(
                f"{r['config_id']}: "
                f"n={r['n_seeds']}; "
                f"AUROC="
                f"{float(r['mean_AUROC']):.6f}"
                f" +/- "
                f"{float(r['sd_AUROC']):.6f}; "
                f"AUPRC="
                f"{float(r['mean_AUPRC']):.6f}"
                f" +/- "
                f"{float(r['sd_AUPRC']):.6f}; "
                f"score="
                f"{float(r['mean_selection_score']):.6f}"
                f" +/- "
                f"{float(r['sd_selection_score']):.6f}\n"
            )

        f.write("\n")

        f.write(
            "PRIMARY TEACHER STATUS: "
            "NOT FROZEN BY TRAINER\n"
        )

        f.write(
            "Teacher choice must be made "
            "from Development validation "
            "together with the frozen "
            "teacher-geometry audit.\n"
        )

    print()
    print("=" * 78)
    print(
        "HGV TEACHER-SELECTION "
        "TRAINING COMPLETE"
    )
    print("=" * 78)

    print(f"Output: {root}")

    print()
    print(
        "HGV_TEACHER_SELECTION_TRAINING=PASS"
    )


if __name__ == "__main__":
    main()
