#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

TEACHER_ROOT = (
    BASE
    / "06_teacher/g1_teacher_augmentation_v2_20260917/"
      "teacher_augmented"
)

FINAL_FREEZE = (
    BASE
    / "07_hgv/final_hgv_config_freeze_v1_20260916/"
      "FINAL_HGV_CONFIG.txt"
)

EVAL_FREEZE = (
    BASE
    / "09_external_validation/"
      "canonical_A_model_protocol_freeze_v1_20260917/"
      "CANONICAL_A_MODEL_PROTOCOL.txt"
)

OUT = (
    BASE
    / "09_external_validation/"
      "canonical_A_model_training_v1_20260917"
)


PROTOCOLS = [
    "G1",
]

SPLIT_SEEDS = [
    20260915,
]

MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
]


INPUT_DIM = 2080
H1 = 256
H2 = 128
EMBED_DIM = 64

MARGIN = 1.0
TAU = 0.98
LAMBDA_DIST = 0.25

LR = 5e-4
WEIGHT_DECAY = 1e-4

PAIR_BATCH = 512
EMBED_BATCH = 1024

POS_PER_EPOCH = 50000
NEG_PER_EPOCH = 50000

MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0

EPS = 1e-8


def args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--resume",
        action="store_true",
    )

    p.add_argument(
        "--protocol",
        choices=[
            "all",
            "G2",
            "G3",
            "G4",
        ],
        default="all",
    )

    return p.parse_args()


def clean(x):
    if x is None:
        return ""
    return str(x).strip()


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
        f"Cannot detect {name}; fields={fields}"
    )


def feature_accession_col(fields):
    return find_col(
        fields,
        [
            "assembly_accession",
            "assembly_accession_final",
            "accession",
        ],
        "feature accession",
    )


def pair_columns(fields):

    a = find_col(
        fields,
        [
            "accession_a",
            "assembly_accession_a",
            "genome_a",
            "a_accession",
        ],
        "endpoint A",
    )

    b = find_col(
        fields,
        [
            "accession_b",
            "assembly_accession_b",
            "genome_b",
            "b_accession",
        ],
        "endpoint B",
    )

    y = find_col(
        fields,
        [
            "label",
            "y",
            "same_species",
            "target",
        ],
        "label",
    )

    pair_type = None

    for c in [
        "pair_type",
        "negative_tier",
        "neg_type",
        "stratum",
    ]:
        if c in fields:
            pair_type = c
            break

    return a, b, y, pair_type


def dashing_col(fields):

    preferred = [
        "dashing2_set_distance",
        "dashing2_distance",
        "dashing_distance",
        "d2_distance",
        "teacher_distance_dashing2",
    ]

    for c in preferred:
        if c in fields:
            return c

    hits = [
        c
        for c in fields
        if (
            "dashing" in c.lower()
            and "distance" in c.lower()
        )
    ]

    if len(hits) == 1:
        return hits[0]

    raise RuntimeError(
        "Cannot uniquely identify Dashing2 distance "
        f"column. hits={hits}; fields={fields}"
    )


def deterministic(seed):

    random.seed(seed)
    np.random.seed(
        seed % (2**32)
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HGV(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            INPUT_DIM,
            H1,
        )

        self.bn1 = nn.BatchNorm1d(
            H1,
        )

        self.fc2 = nn.Linear(
            H1,
            H2,
        )

        self.bn2 = nn.BatchNorm1d(
            H2,
        )

        self.fc3 = nn.Linear(
            H2,
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

        return F.normalize(
            x,
            p=2,
            dim=1,
        )


def load_features():

    fields, rows = read_tsv(
        FEATURE_ROWS
    )

    c = feature_accession_col(
        fields
    )

    accessions = [
        clean(r[c])
        for r in rows
    ]

    if len(accessions) != len(set(accessions)):
        raise RuntimeError(
            "Duplicate feature accessions."
        )

    X = np.load(
        FEATURE_NPY,
        mmap_mode="r",
    )

    if X.shape != (
        3154,
        INPUT_DIM,
    ):
        raise RuntimeError(
            f"Bad feature shape: {X.shape}"
        )

    acc_to_idx = {
        a: i
        for i, a
        in enumerate(accessions)
    }

    return X, acc_to_idx


def load_pairs(
    path,
    acc_to_idx,
):

    fields, rows = read_tsv(
        path
    )

    a_col, b_col, y_col, type_col = (
        pair_columns(fields)
    )

    teacher_col = dashing_col(
        fields
    )

    A = []
    B = []
    Y = []
    T = []

    acc_a = []
    acc_b = []
    pair_types = []

    for r in rows:

        a = clean(r[a_col])
        b = clean(r[b_col])

        if (
            a not in acc_to_idx
            or b not in acc_to_idx
        ):
            raise RuntimeError(
                f"Unknown accession: {a}, {b}"
            )

        y = int(
            float(r[y_col])
        )

        t = float(
            r[teacher_col]
        )

        if y not in (0, 1):
            raise RuntimeError(
                f"Invalid label: {y}"
            )

        if (
            not math.isfinite(t)
            or t < 0
            or t > 1
        ):
            raise RuntimeError(
                f"Invalid teacher distance: {t}"
            )

        A.append(
            acc_to_idx[a]
        )

        B.append(
            acc_to_idx[b]
        )

        Y.append(y)
        T.append(t)

        acc_a.append(a)
        acc_b.append(b)

        pair_types.append(
            clean(r[type_col])
            if type_col is not None
            else ""
        )

    return {
        "a_idx": np.asarray(
            A,
            dtype=np.int64,
        ),
        "b_idx": np.asarray(
            B,
            dtype=np.int64,
        ),
        "labels": np.asarray(
            Y,
            dtype=np.int64,
        ),
        "teacher": np.asarray(
            T,
            dtype=np.float32,
        ),
        "accession_a": np.asarray(
            acc_a,
            dtype=object,
        ),
        "accession_b": np.asarray(
            acc_b,
            dtype=object,
        ),
        "pair_type": np.asarray(
            pair_types,
            dtype=object,
        ),
        "teacher_column": teacher_col,
    }


def sample_epoch(
    train,
    model_seed,
    epoch,
):

    pos = np.flatnonzero(
        train["labels"] == 1
    )

    neg = np.flatnonzero(
        train["labels"] == 0
    )

    epoch_seed = (
        model_seed
        + epoch * 1000003
    ) % (2**32)

    rng = np.random.default_rng(
        epoch_seed
    )

    p = rng.choice(
        pos,
        POS_PER_EPOCH,
        replace=True,
    )

    n = rng.choice(
        neg,
        NEG_PER_EPOCH,
        replace=True,
    )

    ids = np.concatenate(
        [p, n]
    )

    rng.shuffle(ids)

    digest = hashlib.sha256(
        ids.astype(
            np.int64
        ).tobytes()
    ).hexdigest()

    return ids, digest


@torch.no_grad()
def embed_all(
    model,
    X,
):

    model.eval()

    out = []

    for start in range(
        0,
        X.shape[0],
        EMBED_BATCH,
    ):
        end = min(
            start + EMBED_BATCH,
            X.shape[0],
        )

        out.append(
            model(
                X[start:end]
            )
        )

    return torch.cat(
        out,
        dim=0,
    )


@torch.no_grad()
def evaluate(
    model,
    X,
    pairs,
):

    Z = embed_all(
        model,
        X,
    )

    a = torch.as_tensor(
        pairs["a_idx"],
        dtype=torch.long,
        device=X.device,
    )

    b = torch.as_tensor(
        pairs["b_idx"],
        dtype=torch.long,
        device=X.device,
    )

    y = torch.as_tensor(
        pairs["labels"],
        dtype=torch.float32,
        device=X.device,
    )

    teacher = torch.as_tensor(
        pairs["teacher"],
        dtype=torch.float32,
        device=X.device,
    )

    d = torch.linalg.vector_norm(
        Z[a] - Z[b],
        ord=2,
        dim=1,
    )

    score = (
        1.0
        - d / 2.0
    )

    con = (
        y * d.square()
        +
        (1.0 - y)
        * F.relu(
            MARGIN - d
        ).square()
    ).mean()

    mask = (
        (y > 0.5)
        |
        (
            (y < 0.5)
            & (teacher < TAU)
        )
    )

    if int(mask.sum()) > 0:

        dist = F.mse_loss(
            d[mask] / 2.0,
            teacher[mask],
        )

    else:
        dist = torch.zeros(
            (),
            device=X.device,
        )

    y_np = (
        y.cpu()
        .numpy()
        .astype(np.int64)
    )

    score_np = (
        score.cpu()
        .numpy()
        .astype(np.float64)
    )

    d_np = (
        d.cpu()
        .numpy()
        .astype(np.float64)
    )

    auc = float(
        roc_auc_score(
            y_np,
            score_np,
        )
    )

    auprc = float(
        average_precision_score(
            y_np,
            score_np,
        )
    )

    return {
        "AUROC": auc,
        "AUPRC": auprc,
        "score": (
            auc + auprc
        ) / 2.0,
        "contrastive": float(
            con.item()
        ),
        "distill": float(
            dist.item()
        ),
        "similarity": score_np,
        "distance": d_np,
    }


def better(
    new_score,
    new_con,
    best_score,
    best_con,
):

    if (
        new_score
        > best_score + EPS
    ):
        return (
            True,
            "ranking_score",
        )

    if (
        abs(
            new_score
            - best_score
        ) <= EPS
        and new_con
        < best_con - EPS
    ):
        return (
            True,
            "contrastive_tiebreak",
        )

    return (
        False,
        "none",
    )


def save_history(
    path,
    history,
):

    fields = [
        "epoch",
        "sampler_sha256",
        "train_total_loss",
        "train_contrastive_loss",
        "train_distill_loss",
        "val_AUROC",
        "val_AUPRC",
        "val_selection_score",
        "val_contrastive_loss",
        "val_distill_loss",
        "update_reason",
        "best_epoch",
        "patience",
    ]

    write_tsv(
        path,
        history,
        fields,
    )


def save_val_predictions(
    path,
    val,
    result,
):

    rows = []

    for i in range(
        len(val["labels"])
    ):

        rows.append({
            "accession_a":
                val["accession_a"][i],

            "accession_b":
                val["accession_b"][i],

            "label":
                int(
                    val["labels"][i]
                ),

            "pair_type":
                val["pair_type"][i],

            "distance":
                float(
                    result[
                        "distance"
                    ][i]
                ),

            "similarity_score":
                float(
                    result[
                        "similarity"
                    ][i]
                ),
        })

    write_tsv(
        path,
        rows,
        [
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
            "distance",
            "similarity_score",
        ],
    )


def run_one(
    protocol,
    split_seed,
    model_seed,
    X,
    acc_to_idx,
    resume,
):

    split_id = (
        f"{protocol}_seed{split_seed}"
    )

    run_id = (
        f"{protocol}"
        f"__split{split_seed}"
        f"__model{model_seed}"
    )

    run_dir = (
        OUT
        / protocol
        / f"split{split_seed}"
        / f"model{model_seed}"
    )

    complete = (
        run_dir
        / "RUN_COMPLETE"
    )

    if complete.is_file():

        print(
            f"SKIP_COMPLETE {run_id}"
        )

        return

    if (
        run_dir.exists()
        and not resume
    ):
        raise RuntimeError(
            f"Incomplete output exists: {run_dir}. "
            "Use --resume."
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        TEACHER_ROOT
        / split_id
        / "train_teacher.tsv"
    )

    val_path = (
        TEACHER_ROOT
        / split_id
        / "val_teacher.tsv"
    )

    # IMPORTANT:
    # test_pairs.tsv is intentionally not referenced
    # anywhere in this training script.

    train = load_pairs(
        train_path,
        acc_to_idx,
    )

    val = load_pairs(
        val_path,
        acc_to_idx,
    )

    config = {
        "protocol": protocol,
        "split_seed": split_seed,
        "model_seed": model_seed,

        "teacher":
            "Dashing2 SetSketch",

        "tau": TAU,
        "lambda_dist":
            LAMBDA_DIST,

        "architecture":
            "2080-256-128-64",

        "pair_forward":
            "BN-safe concatenated endpoints",

        "output_normalization":
            "L2",

        "margin": MARGIN,
        "optimizer": "AdamW",

        "learning_rate": LR,

        "weight_decay":
            WEIGHT_DECAY,

        "pair_batch":
            PAIR_BATCH,

        "train_examples_per_epoch":
            100000,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "checkpoint_primary":
            "mean(AUROC,AUPRC)",

        "checkpoint_tiebreak":
            "minimum validation contrastive loss",

        "test_used":
            False,

        "external_used":
            False,

        "train_teacher_sha256":
            sha256(
                train_path
            ),

        "val_teacher_sha256":
            sha256(
                val_path
            ),

        "final_freeze_sha256":
            sha256(
                FINAL_FREEZE
            ),

        "eval_protocol_sha256":
            sha256(
                EVAL_FREEZE
            ),
    }

    (
        run_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    deterministic(
        model_seed
    )

    model = HGV().to(
        X.device
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    best_path = (
        run_dir
        / "best_model.pt"
    )

    last_path = (
        run_dir
        / "last_state.pt"
    )

    history_path = (
        run_dir
        / "history.tsv"
    )

    best_score = -math.inf
    best_con = math.inf
    best_epoch = 0
    best_reason = ""

    patience_count = 0
    history = []

    start_epoch = 1

    if (
        resume
        and last_path.is_file()
    ):

        state = torch.load(
            last_path,
            map_location=X.device,
            weights_only=False,
        )

        model.load_state_dict(
            state[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            state[
                "optimizer_state_dict"
            ]
        )

        best_score = float(
            state[
                "best_score"
            ]
        )

        best_con = float(
            state[
                "best_con"
            ]
        )

        best_epoch = int(
            state[
                "best_epoch"
            ]
        )

        best_reason = str(
            state[
                "best_reason"
            ]
        )

        patience_count = int(
            state[
                "patience_count"
            ]
        )

        history = list(
            state[
                "history"
            ]
        )

        start_epoch = (
            int(
                state[
                    "epoch"
                ]
            )
            + 1
        )

        print(
            f"RESUME {run_id} "
            f"epoch={start_epoch}"
        )

    print()
    print(
        f"RUN {run_id} "
        f"train={len(train['labels'])} "
        f"val={len(val['labels'])}"
    )

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        model.train()

        ids, sampler_sha = (
            sample_epoch(
                train,
                model_seed,
                epoch,
            )
        )

        total_sum = 0.0
        con_sum = 0.0
        dist_sum = 0.0

        n_seen = 0

        for start in range(
            0,
            len(ids),
            PAIR_BATCH,
        ):

            batch = ids[
                start:
                start + PAIR_BATCH
            ]

            a = torch.as_tensor(
                train[
                    "a_idx"
                ][batch],
                dtype=torch.long,
                device=X.device,
            )

            b = torch.as_tensor(
                train[
                    "b_idx"
                ][batch],
                dtype=torch.long,
                device=X.device,
            )

            y = torch.as_tensor(
                train[
                    "labels"
                ][batch],
                dtype=torch.float32,
                device=X.device,
            )

            teacher = (
                torch.as_tensor(
                    train[
                        "teacher"
                    ][batch],
                    dtype=torch.float32,
                    device=X.device,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # BN-safe Siamese forward:
            # both endpoints share the same
            # mini-batch normalization statistics.
            xpair = torch.cat(
                [
                    X[a],
                    X[b],
                ],
                dim=0,
            )

            zpair = model(
                xpair
            )

            za, zb = torch.chunk(
                zpair,
                2,
                dim=0,
            )

            d = (
                torch.linalg
                .vector_norm(
                    za - zb,
                    ord=2,
                    dim=1,
                )
            )

            loss_con = (
                y * d.square()
                +
                (1.0 - y)
                * F.relu(
                    MARGIN - d
                ).square()
            ).mean()

            mask = (
                (y > 0.5)
                |
                (
                    (y < 0.5)
                    & (
                        teacher
                        < TAU
                    )
                )
            )

            loss_dist = (
                F.mse_loss(
                    d[mask] / 2.0,
                    teacher[mask],
                )
                if int(
                    mask.sum()
                ) > 0
                else torch.zeros(
                    (),
                    device=X.device,
                )
            )

            loss = (
                loss_con
                +
                LAMBDA_DIST
                * loss_dist
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            bs = len(batch)

            total_sum += (
                float(
                    loss.item()
                )
                * bs
            )

            con_sum += (
                float(
                    loss_con.item()
                )
                * bs
            )

            dist_sum += (
                float(
                    loss_dist.item()
                )
                * bs
            )

            n_seen += bs

        train_total = (
            total_sum
            / n_seen
        )

        train_con = (
            con_sum
            / n_seen
        )

        train_dist = (
            dist_sum
            / n_seen
        )

        vr = evaluate(
            model,
            X,
            val,
        )

        improved, reason = (
            better(
                vr["score"],
                vr["contrastive"],
                best_score,
                best_con,
            )
        )

        if improved:

            best_score = (
                vr["score"]
            )

            best_con = (
                vr[
                    "contrastive"
                ]
            )

            best_epoch = epoch
            best_reason = reason

            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_selection_score":
                        best_score,

                    "val_contrastive_loss":
                        best_con,

                    "checkpoint_reason":
                        reason,

                    "config":
                        config,
                },
                best_path,
            )

        else:
            patience_count += 1

        row = {
            "epoch":
                epoch,

            "sampler_sha256":
                sampler_sha,

            "train_total_loss":
                train_total,

            "train_contrastive_loss":
                train_con,

            "train_distill_loss":
                train_dist,

            "val_AUROC":
                vr["AUROC"],

            "val_AUPRC":
                vr["AUPRC"],

            "val_selection_score":
                vr["score"],

            "val_contrastive_loss":
                vr[
                    "contrastive"
                ],

            "val_distill_loss":
                vr[
                    "distill"
                ],

            "update_reason":
                reason,

            "best_epoch":
                best_epoch,

            "patience":
                patience_count,
        }

        history.append(row)

        save_history(
            history_path,
            history,
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "best_score":
                    best_score,

                "best_con":
                    best_con,

                "best_epoch":
                    best_epoch,

                "best_reason":
                    best_reason,

                "patience_count":
                    patience_count,

                "history":
                    history,
            },
            last_path,
        )

        print(
            f"epoch={epoch:02d} "
            f"train={train_total:.6f} "
            f"AUC={vr['AUROC']:.6f} "
            f"AUPRC={vr['AUPRC']:.6f} "
            f"score={vr['score']:.6f} "
            f"best={best_score:.6f} "
            f"best_con={best_con:.6f} "
            f"reason={reason} "
            f"patience="
            f"{patience_count}/{PATIENCE}"
        )

        if (
            patience_count
            >= PATIENCE
        ):

            print(
                f"EARLY_STOP "
                f"epoch={epoch}"
            )

            break

    best = torch.load(
        best_path,
        map_location=X.device,
        weights_only=False,
    )

    model.load_state_dict(
        best[
            "model_state_dict"
        ]
    )

    vr = evaluate(
        model,
        X,
        val,
    )

    save_val_predictions(
        run_dir
        / "val_predictions.tsv",
        val,
        vr,
    )

    summary = {
        "status":
            "PASS",

        "protocol":
            protocol,

        "split_seed":
            split_seed,

        "model_seed":
            model_seed,

        "best_epoch":
            int(
                best[
                    "epoch"
                ]
            ),

        "best_checkpoint_reason":
            best[
                "checkpoint_reason"
            ],

        "val_AUROC":
            float(
                vr[
                    "AUROC"
                ]
            ),

        "val_AUPRC":
            float(
                vr[
                    "AUPRC"
                ]
            ),

        "val_selection_score":
            float(
                vr[
                    "score"
                ]
            ),

        "val_contrastive_loss":
            float(
                vr[
                    "contrastive"
                ]
            ),

        "val_distill_loss":
            float(
                vr[
                    "distill"
                ]
            ),

        "epochs_ran":
            len(history),

        "train_pairs":
            len(
                train[
                    "labels"
                ]
            ),

        "val_pairs":
            len(
                val[
                    "labels"
                ]
            ),

        "test_used":
            False,

        "external_used":
            False,
    }

    (
        run_dir
        / "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    complete.write_text(
        "RUN_COMPLETE\n",
        encoding="utf-8",
    )

    print(
        f"RUN_PASS {run_id} "
        f"val_score="
        f"{vr['score']:.6f}"
    )


def main():

    a = args()

    if not FINAL_FREEZE.is_file():
        raise RuntimeError(
            "Missing final model freeze."
        )

    if not EVAL_FREEZE.is_file():
        raise RuntimeError(
            "Missing main evaluation freeze."
        )

    protocols = (
        PROTOCOLS
        if a.protocol == "all"
        else [
            a.protocol
        ]
    )

    if (
        OUT.exists()
        and not a.resume
    ):
        raise RuntimeError(
            f"Output exists: {OUT}\n"
            "Use --resume only for the same frozen grid."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        FINAL_FREEZE,
        OUT
        / "FINAL_HGV_CONFIG_FROZEN_COPY.txt",
    )

    shutil.copy2(
        EVAL_FREEZE,
        OUT
        / "MAIN_EVAL_PROTOCOL_FROZEN_COPY.txt",
    )

    (
        OUT
        / "environment.txt"
    ).write_text(
        "\n".join([
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"torch={torch.__version__}",
            f"numpy={np.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
            (
                "gpu="
                + (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "NA"
                )
            ),
            "test_used=False",
            "external_used=False",
        ])
        + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "HGV-V3 CANONICAL DATASET-A "
        "G1 TRAINING"
    )
    print("=" * 78)

    print(
        f"protocols={protocols}"
    )

    print(
        f"planned_runs="
        f"{len(protocols) * 3 * 3}"
    )

    print(
        "test_used=False"
    )

    print(
        "external_used=False"
    )

    print(
        "TRAIN SCRIPT DOES NOT READ "
        "test_pairs.tsv"
    )

    X_np, acc_to_idx = (
        load_features()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X = torch.as_tensor(
        np.asarray(
            X_np,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )

    for protocol in protocols:

        for split_seed in SPLIT_SEEDS:

            for model_seed in MODEL_SEEDS:

                run_one(
                    protocol,
                    split_seed,
                    model_seed,
                    X,
                    acc_to_idx,
                    a.resume,
                )

    expected = (
        len(protocols)
        * len(SPLIT_SEEDS)
        * len(MODEL_SEEDS)
    )

    completed = 0

    for protocol in protocols:

        for split_seed in SPLIT_SEEDS:

            for model_seed in MODEL_SEEDS:

                f = (
                    OUT
                    / protocol
                    / f"split{split_seed}"
                    / f"model{model_seed}"
                    / "RUN_COMPLETE"
                )

                if f.is_file():
                    completed += 1

    if completed != expected:
        raise RuntimeError(
            f"Only {completed}/{expected} "
            "runs complete."
        )

    (
        OUT
        / "TRAIN_GRID_COMPLETE"
    ).write_text(
        (
            f"completed={completed}\n"
            "test_used=False\n"
            "external_used=False\n"
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "CANONICAL DATASET-A TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        f"completed_runs="
        f"{completed}/{expected}"
    )

    print(
        "test_used=False"
    )

    print(
        "external_used=False"
    )

    print(
        "HGV_CANONICAL_A_TRAINING=PASS"
    )


if __name__ == "__main__":
    main()
