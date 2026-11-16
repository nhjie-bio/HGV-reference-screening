#!/usr/bin/env python3

from pathlib import Path
import copy
import json
import math
import os
import random
import re
import shutil
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scipy.stats import spearmanr

from hgv_common import (
    extract_acc,
    parse_dashing_allpairs,
    read_list,
    sha256_file,
)


BASE = Path(
    os.environ.get(
        "HGV_BASE",
        "HGV_V3_revision_20260915",
    )
).resolve()

FEATURE_ROOT = (
    BASE /
    "03_features/canonical_k6_full3154_v1"
)

PAIR_ROOT = (
    BASE /
    "05_pairs/pair_manifests_v1_20260916"
)

TEACHER_ROOT = (
    BASE /
    "06_teacher/teacher_cache_v1_20260916"
)

FINAL_CONFIG = (
    BASE /
    "07_hgv/final_hgv_config_freeze_v1_20260916/"
    "FINAL_HGV_CONFIG.txt"
)

MAIN_ROOT = (
    BASE /
    "08_main_generalization/"
    "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
)

OUT = (
    BASE /
    "10_ablation/"
    "E6_D0_D5_G2_seed20260915_5seed_v2_20260921"
)

RUNS = OUT / "runs"
SRC = OUT / "source_data"
PROV = OUT / "provenance"

for d in [OUT, RUNS, SRC, PROV]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ======================================================================
# Frozen E6 constants
# ======================================================================

SPLIT_SEED = "20260915"

MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
]

MODES = [
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]

TAU = 0.98
LAMBDA_DIST = 0.25
MARGIN = 1.0

LR = 5e-4
WEIGHT_DECAY = 1e-4
BATCH = 512

N_POS_EPOCH = 50000
N_NEG_EPOCH = 50000

MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0

INPUT_DIM = 2080
H1 = 256
H2 = 128
OUT_DIM = 64


def fail(msg):
    raise RuntimeError(msg)


# ======================================================================
# Reproducibility
# ======================================================================

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

torch.set_num_threads(
    int(
        os.environ.get(
            "HGV_TORCH_CPU_THREADS",
            "16",
        )
    )
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("HGV-V3 E6 D0-D5 5-SEED MECHANISTIC ABLATION")
print("=" * 100)
print("Device:", device)
print("Canonical split: G2 seed20260915")
print("Modes:", MODES)
print("Model seeds:", MODEL_SEEDS)
print("Output:", OUT)


# ======================================================================
# Preserve final source/config provenance
# ======================================================================

if FINAL_CONFIG.is_file():
    shutil.copy2(
        FINAL_CONFIG,
        PROV / "FINAL_HGV_CONFIG_FROZEN_COPY.txt",
    )


# Find likely final BN-safe trainer source and preserve it.
script_root = (
    BASE /
    "scripts_v3_20260916_hgv"
)

trainer_candidates = []

if script_root.is_dir():
    for p in script_root.glob("*.py"):
        try:
            txt = p.read_text(
                errors="replace"
            )
        except Exception:
            continue

        score = 0

        if "bn_safe" in txt.lower():
            score += 3

        if (
            "hgv_d3_dashing2_tau0p98_bn_safe"
            in txt.lower()
        ):
            score += 10

        if "CUBLAS_WORKSPACE_CONFIG" in txt:
            score += 1

        if score:
            trainer_candidates.append(
                (
                    score,
                    p,
                    txt,
                )
            )

trainer_candidates.sort(
    key=lambda x: (
        -x[0],
        str(x[1]),
    )
)

trainer_source = None
trainer_text = ""

if trainer_candidates:
    _, trainer_source, trainer_text = trainer_candidates[0]

    shutil.copy2(
        trainer_source,
        PROV /
        f"FINAL_BN_SAFE_TRAINER_SOURCE_COPY__{trainer_source.name}",
    )


# Detect the distillation regression loss from final source if possible.
low = trainer_text.lower()

if "smooth_l1" in low:
    DISTILL_LOSS = "smooth_l1"

elif "mse_loss" in low:
    DISTILL_LOSS = "mse"

elif (
    "teacher_distance" in low
    and (
        ".pow(2)" in low
        or "** 2" in low
        or "**2" in low
    )
):
    DISTILL_LOSS = "mse"

else:
    # The project documents define L_dist but not the function name.
    # MSE is used as the explicit new E6 frozen implementation.
    DISTILL_LOSS = "mse"
    warnings.warn(
        "Could not infer distillation regression loss from copied "
        "final source; E6 explicitly freezes L_dist=MSE."
    )


(PROV / "E6_DISTILLATION_LOSS_FREEZE.txt").write_text(
    f"DISTILL_LOSS={DISTILL_LOSS}\n"
    f"trainer_source={trainer_source}\n"
)


# ======================================================================
# Feature matrix + accession order discovery
# ======================================================================

def discover_feature_matrix():
    cands = []

    for p in FEATURE_ROOT.rglob("*.npy"):
        try:
            x = np.load(
                p,
                mmap_mode="r",
            )
        except Exception:
            continue

        if x.shape == (3154, 2080):
            score = 0

            n = p.name.lower()

            if "feature" in n:
                score += 3
            if "canonical" in n:
                score += 2
            if "k6" in n:
                score += 2

            cands.append(
                (
                    score,
                    p,
                )
            )

    if not cands:
        fail(
            "Cannot find frozen (3154,2080) feature matrix"
        )

    cands.sort(
        key=lambda z: (
            -z[0],
            str(z[1]),
        )
    )

    return cands[0][1]


FEATURE_NPY = discover_feature_matrix()

X_np = np.load(
    FEATURE_NPY,
    mmap_mode="r",
)

if X_np.shape != (3154, 2080):
    fail(
        f"Unexpected feature matrix: {X_np.shape}"
    )


def discover_feature_order():
    candidates = []

    for p in FEATURE_ROOT.rglob("*.tsv"):
        try:
            if p.stat().st_size > 50_000_000:
                continue

            d = pd.read_csv(
                p,
                sep="\t",
                dtype=str,
            )
        except Exception:
            continue

        acc_col = None

        for c in [
            "assembly_accession",
            "accession",
            "genome_accession",
        ]:
            if c in d.columns:
                acc_col = c
                break

        if acc_col is None:
            continue

        if len(d) != 3154:
            continue

        if d[acc_col].nunique() != 3154:
            continue

        idx_col = None

        for c in [
            "feature_index",
            "row_index",
            "index",
            "matrix_index",
        ]:
            if c in d.columns:
                z = pd.to_numeric(
                    d[c],
                    errors="coerce",
                )

                if (
                    z.notna().all()
                    and set(z.astype(int))
                    == set(range(3154))
                ):
                    idx_col = c
                    break

        score = 0

        name = p.name.lower()

        if "manifest" in name:
            score += 5
        if "index" in name:
            score += 4
        if "feature" in name:
            score += 3
        if idx_col:
            score += 10

        candidates.append(
            (
                score,
                p,
                d,
                acc_col,
                idx_col,
            )
        )

    if not candidates:
        fail(
            "Cannot identify frozen feature accession mapping"
        )

    candidates.sort(
        key=lambda z: (
            -z[0],
            str(z[1]),
        )
    )

    _, p, d, acc_col, idx_col = candidates[0]

    if idx_col:
        d = d.assign(
            __idx=pd.to_numeric(
                d[idx_col]
            ).astype(int)
        ).sort_values(
            "__idx"
        )

    return (
        p,
        d[acc_col].astype(str).tolist(),
    )


FEATURE_MAP_TSV, feature_acc = discover_feature_order()

if len(feature_acc) != 3154:
    fail("Feature accession order count !=3154")

feature_idx = {
    a: i
    for i, a in enumerate(feature_acc)
}


(PROV / "FEATURE_SOURCE.txt").write_text(
    f"matrix={FEATURE_NPY}\n"
    f"mapping={FEATURE_MAP_TSV}\n"
    f"shape={X_np.shape}\n"
)


# ======================================================================
# Locate canonical G2 seed15 pair manifests
# ======================================================================

def locate_pair(role):
    filenames = {
        "train": "train_unique_pair_pool.tsv",
        "val": "val_pairs.tsv",
        "test": "test_pairs.tsv",
    }

    if role not in filenames:
        fail(
            f"Unsupported pair role: {role}"
        )

    target = (
        PAIR_ROOT /
        f"G2_seed{SPLIT_SEED}" /
        filenames[role]
    )

    if not target.is_file():
        fail(
            f"Missing frozen G2 pair manifest: {target}"
        )

    return target


TRAIN_TSV = locate_pair("train")
VAL_TSV = locate_pair("val")
TEST_TSV = locate_pair("test")


def read_pairs(path):
    d = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    ca = None
    cb = None
    cl = None

    for c in [
        "accession_a",
        "genome_a",
        "assembly_accession_a",
    ]:
        if c in d.columns:
            ca = c
            break

    for c in [
        "accession_b",
        "genome_b",
        "assembly_accession_b",
    ]:
        if c in d.columns:
            cb = c
            break

    for c in [
        "label",
        "same_species",
        "y",
    ]:
        if c in d.columns:
            cl = c
            break

    if not all([ca, cb, cl]):
        fail(
            f"Pair schema unsupported: {path}\ncolumns={list(d.columns)}"
        )

    a = d[ca].astype(str).to_numpy()
    b = d[cb].astype(str).to_numpy()

    y = pd.to_numeric(
        d[cl],
        errors="raise",
    ).astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        fail(
            f"Non-binary labels in {path}: {np.unique(y)}"
        )

    missing = [
        x
        for x in set(a) | set(b)
        if x not in feature_idx
    ]

    if missing:
        fail(
            f"Pair accessions missing from features: {missing[:10]}"
        )

    ia = np.asarray(
        [
            feature_idx[x]
            for x in a
        ],
        dtype=np.int64,
    )

    ib = np.asarray(
        [
            feature_idx[x]
            for x in b
        ],
        dtype=np.int64,
    )

    return {
        "df": d,
        "accession_a": a,
        "accession_b": b,
        "ia": ia,
        "ib": ib,
        "y": y.astype(np.float32),
    }


train = read_pairs(TRAIN_TSV)
val = read_pairs(VAL_TSV)
test = read_pairs(TEST_TSV)

if len(train["y"]) != 18176:
    fail(
        f"Expected G2 seed15 train pairs=18176; found {len(train['y'])}"
    )

if len(val["y"]) != 3874:
    fail(
        f"Expected G2 seed15 val pairs=3874; found {len(val['y'])}"
    )

if len(test["y"]) != 3872:
    fail(
        f"Expected G2 seed15 test pairs=3872; found {len(test['y'])}"
    )


# ======================================================================
# Teacher matrix
# ======================================================================

D2_SIM = (
    TEACHER_ROOT /
    "dashing2_set/dashing2_set_similarity.tsv"
)

if not D2_SIM.is_file():
    fail(
        f"Missing production Dashing2 teacher matrix: {D2_SIM}"
    )


list_cands = list(
    TEACHER_ROOT.rglob(
        "development_A_fasta.list"
    )
)

if len(list_cands) != 1:
    fail(
        "Cannot uniquely locate Development-A teacher FASTA order: "
        f"{list_cands}"
    )

A_LIST = list_cands[0]

teacher_paths = read_list(
    A_LIST
)

teacher_acc = [
    extract_acc(x)
    for x in teacher_paths
]

if (
    len(teacher_acc) != 2374
    or any(x is None for x in teacher_acc)
    or len(set(teacher_acc)) != 2374
):
    fail(
        "Invalid Development-A teacher accession order"
    )


print("Loading Dashing2 teacher matrix...")

teacher_similarity = parse_dashing_allpairs(
    D2_SIM,
    teacher_acc,
)

if teacher_similarity.shape != (2374, 2374):
    fail(
        f"Teacher matrix shape unexpected: {teacher_similarity.shape}"
    )

finite_teacher = teacher_similarity[
    np.isfinite(
        teacher_similarity
    )
]

if (
    finite_teacher.min() < -1e-6
    or finite_teacher.max() > 1.000001
):
    fail(
        "Teacher matrix is not in expected similarity range [0,1]"
    )

teacher_map = {
    a: i
    for i, a in enumerate(teacher_acc)
}


def attach_teacher(pair):
    td = np.empty(
        len(pair["y"]),
        dtype=np.float32,
    )

    missing = []

    for i, (a, b) in enumerate(
        zip(
            pair["accession_a"],
            pair["accession_b"],
        )
    ):
        if (
            a not in teacher_map
            or b not in teacher_map
        ):
            missing.append(
                (
                    a,
                    b,
                )
            )
            td[i] = np.nan
            continue

        sim = teacher_similarity[
            teacher_map[a],
            teacher_map[b],
        ]

        if not np.isfinite(sim):
            missing.append(
                (
                    a,
                    b,
                )
            )
            td[i] = np.nan

        else:
            td[i] = 1.0 - float(sim)

    if missing:
        fail(
            f"Teacher missing for {len(missing)} pairs. "
            f"Examples={missing[:10]}"
        )

    pair["teacher_distance"] = td


for x in [
    train,
    val,
    test,
]:
    attach_teacher(x)


# ======================================================================
# Model
# ======================================================================

class HGV(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()

        self.normalize = normalize

        self.fc1 = nn.Linear(
            INPUT_DIM,
            H1,
        )

        self.bn1 = nn.BatchNorm1d(
            H1
        )

        self.fc2 = nn.Linear(
            H1,
            H2,
        )

        self.bn2 = nn.BatchNorm1d(
            H2
        )

        self.fc3 = nn.Linear(
            H2,
            OUT_DIM,
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

        z = self.fc3(x)

        if self.normalize:
            z = F.normalize(
                z,
                p=2,
                dim=1,
            )

        return z


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def distill_loss(student, teacher):
    if DISTILL_LOSS == "smooth_l1":
        return F.smooth_l1_loss(
            student,
            teacher,
        )

    return F.mse_loss(
        student,
        teacher,
    )


# Feature matrix on device.
X = torch.from_numpy(
    np.asarray(
        X_np,
        dtype=np.float32,
    ).copy()
).to(
    device
)


def encode_pair_batch(model, ia, ib):
    ta = torch.as_tensor(
        ia,
        dtype=torch.long,
        device=device,
    )

    tb = torch.as_tensor(
        ib,
        dtype=torch.long,
        device=device,
    )

    # BN-safe: concatenate A/B endpoint features and use one forward.
    xx = torch.cat(
        [
            X.index_select(0, ta),
            X.index_select(0, tb),
        ],
        dim=0,
    )

    zz = model(xx)

    n = len(ia)

    za = zz[:n]
    zb = zz[n:]

    return za, zb


def pair_scores(
    model,
    pair,
    batch_size=2048,
):
    model.eval()

    scores = []
    dnorm = []

    with torch.no_grad():
        for s in range(
            0,
            len(pair["y"]),
            batch_size,
        ):
            e = min(
                s + batch_size,
                len(pair["y"]),
            )

            za, zb = encode_pair_batch(
                model,
                pair["ia"][s:e],
                pair["ib"][s:e],
            )

            d = torch.linalg.vector_norm(
                za - zb,
                dim=1,
            )

            dn = d / 2.0

            sc = 1.0 - dn

            scores.append(
                sc.detach()
                .cpu()
                .numpy()
            )

            dnorm.append(
                dn.detach()
                .cpu()
                .numpy()
            )

    return (
        np.concatenate(scores),
        np.concatenate(dnorm),
    )


def contrastive_from_dist(d, y):
    return (
        y * d.pow(2)
        +
        (1.0 - y)
        *
        torch.relu(
            MARGIN - d
        ).pow(2)
    ).mean()


def validation_loss(
    model,
    pair,
    batch_size=2048,
):
    model.eval()

    vals = []

    with torch.no_grad():
        for s in range(
            0,
            len(pair["y"]),
            batch_size,
        ):
            e = min(
                s + batch_size,
                len(pair["y"]),
            )

            za, zb = encode_pair_batch(
                model,
                pair["ia"][s:e],
                pair["ib"][s:e],
            )

            d = torch.linalg.vector_norm(
                za - zb,
                dim=1,
            )

            y = torch.as_tensor(
                pair["y"][s:e],
                dtype=torch.float32,
                device=device,
            )

            vals.append(
                float(
                    contrastive_from_dist(
                        d,
                        y,
                    ).detach().cpu()
                )
            )

    return float(
        np.mean(vals)
    )


def choose_threshold(scores, y):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    s = scores[order]
    yy = y[order]

    P = int(
        yy.sum()
    )

    N = len(yy) - P

    ctp = np.cumsum(
        yy == 1
    )

    cfp = np.cumsum(
        yy == 0
    )

    ends = np.r_[
        np.where(
            np.diff(s) != 0
        )[0],
        len(s) - 1,
    ]

    best = None

    for k in ends:
        tp = int(ctp[k])
        fp = int(cfp[k])
        fn = P - tp
        tn = N - fp

        denom = math.sqrt(
            max(
                (tp + fp)
                * (tp + fn)
                * (tn + fp)
                * (tn + fn),
                0,
            )
        )

        mcc = (
            ((tp * tn) - (fp * fn)) / denom
            if denom > 0
            else 0.0
        )

        tpr = (
            tp / P
            if P
            else 0.0
        )

        tnr = (
            tn / N
            if N
            else 0.0
        )

        bacc = (
            tpr + tnr
        ) / 2.0

        f1den = (
            2 * tp
            + fp
            + fn
        )

        f1 = (
            2 * tp / f1den
            if f1den
            else 0.0
        )

        thr = float(
            s[k]
        )

        key = (
            mcc,
            bacc,
            f1,
            -abs(
                thr - 0.5
            ),
            -thr,
        )

        if best is None or key > best[0]:
            best = (
                key,
                thr,
            )

    return best[1]


def metric_dict(scores, y, threshold):
    y = np.asarray(
        y,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    pred = (
        scores >= threshold
    ).astype(int)

    return {
        "auroc":
            float(
                roc_auc_score(
                    y,
                    scores,
                )
            ),
        "auprc":
            float(
                average_precision_score(
                    y,
                    scores,
                )
            ),
        "mcc":
            float(
                matthews_corrcoef(
                    y,
                    pred,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),
        "f1":
            float(
                f1_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),
        "precision":
            float(
                precision_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),
        "recall":
            float(
                recall_score(
                    y,
                    pred,
                    zero_division=0,
                )
            ),
    }


def teacher_rho(
    student_d,
    pair,
    mask,
):
    if mask.sum() < 3:
        return np.nan

    r = spearmanr(
        student_d[mask],
        pair["teacher_distance"][mask],
    ).statistic

    return float(r)


# ======================================================================
# D0-D5 semantics
# ======================================================================

def mode_mask(mode, labels, td):
    pos = labels == 1
    neg = labels == 0

    if mode == "D0":
        return np.zeros(
            len(labels),
            dtype=bool,
        )

    if mode == "D1":
        return np.ones(
            len(labels),
            dtype=bool,
        )

    if mode == "D2":
        return neg

    if mode in {
        "D3",
        "D4",
        "D5",
    }:
        return (
            pos
            |
            (
                neg
                &
                (td < TAU)
            )
        )

    fail(
        f"Unknown E6 mode: {mode}"
    )


def train_one(mode, seed):
    out = (
        RUNS /
        f"{mode}_modelseed{seed}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete = out / "RUN_COMPLETE"

    if complete.is_file():
        print(
            "SKIP COMPLETE:",
            out.name,
        )

        return json.loads(
            (
                out /
                "metrics.json"
            ).read_text()
        )

    normalize = (
        mode != "D5"
    )

    set_seed(seed)

    model = HGV(
        normalize=normalize
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    labels = train["y"]
    td = train["teacher_distance"]

    pos_pool = np.where(
        labels == 1
    )[0]

    neg_pool = np.where(
        labels == 0
    )[0]

    if (
        len(pos_pool) == 0
        or len(neg_pool) == 0
    ):
        fail(
            "Training pair pool lacks positive or negative pairs"
        )

    mask_full = mode_mask(
        mode,
        labels,
        td,
    )

    teacher_for_pair = td.copy()

    if mode == "D4":
        rng_shuffle = np.random.default_rng(
            seed + 4000004
        )

        vals = teacher_for_pair[
            mask_full
        ].copy()

        rng_shuffle.shuffle(
            vals
        )

        teacher_for_pair[
            mask_full
        ] = vals

    history = []

    best_state = None
    best_epoch = None
    best_score = -np.inf
    best_val_con = np.inf
    best_reason = None

    no_improve = 0

    t_run = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        rng = np.random.default_rng(
            seed * 1000
            + epoch
        )

        ep_pos = rng.choice(
            pos_pool,
            size=N_POS_EPOCH,
            replace=True,
        )

        ep_neg = rng.choice(
            neg_pool,
            size=N_NEG_EPOCH,
            replace=True,
        )

        ep_idx = np.concatenate(
            [
                ep_pos,
                ep_neg,
            ]
        )

        rng.shuffle(
            ep_idx
        )

        model.train()

        epoch_losses = []
        epoch_con = []
        epoch_distill = []

        for s in range(
            0,
            len(ep_idx),
            BATCH,
        ):
            idx = ep_idx[
                s:s + BATCH
            ]

            za, zb = encode_pair_batch(
                model,
                train["ia"][idx],
                train["ib"][idx],
            )

            d = torch.linalg.vector_norm(
                za - zb,
                dim=1,
            )

            yb = torch.as_tensor(
                labels[idx],
                dtype=torch.float32,
                device=device,
            )

            con = contrastive_from_dist(
                d,
                yb,
            )

            loss = con

            dl = torch.tensor(
                0.0,
                device=device,
            )

            if mode != "D0":
                mb_np = mask_full[
                    idx
                ]

                if mb_np.any():
                    mb = torch.as_tensor(
                        mb_np,
                        dtype=torch.bool,
                        device=device,
                    )

                    target = torch.as_tensor(
                        teacher_for_pair[
                            idx
                        ],
                        dtype=torch.float32,
                        device=device,
                    )

                    student = d / 2.0

                    dl = distill_loss(
                        student[mb],
                        target[mb],
                    )

                    loss = (
                        con
                        +
                        LAMBDA_DIST
                        * dl
                    )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

            epoch_con.append(
                float(
                    con.detach()
                    .cpu()
                )
            )

            epoch_distill.append(
                float(
                    dl.detach()
                    .cpu()
                )
            )

        val_scores, val_dn = pair_scores(
            model,
            val,
        )

        val_auc = float(
            roc_auc_score(
                val["y"],
                val_scores,
            )
        )

        val_ap = float(
            average_precision_score(
                val["y"],
                val_scores,
            )
        )

        selection = (
            val_auc
            + val_ap
        ) / 2.0

        val_con = validation_loss(
            model,
            val,
        )

        improved = False
        reason = ""

        if selection > best_score + 1e-8:
            improved = True
            reason = "ranking_score"

        elif (
            abs(
                selection
                - best_score
            ) <= 1e-8
            and val_con
            < best_val_con - 1e-12
        ):
            improved = True
            reason = "contrastive_tiebreak"

        if improved:
            best_score = selection
            best_val_con = val_con
            best_epoch = epoch
            best_reason = reason

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "state_dict":
                        best_state,
                    "mode":
                        mode,
                    "model_seed":
                        seed,
                    "split":
                        "G2_seed20260915",
                    "normalize":
                        normalize,
                    "tau":
                        TAU,
                    "lambda_dist":
                        LAMBDA_DIST,
                    "margin":
                        MARGIN,
                    "distill_loss":
                        DISTILL_LOSS,
                    "best_epoch":
                        best_epoch,
                    "best_selection":
                        best_score,
                },
                out /
                "best_model.pt",
            )

            no_improve = 0

        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss":
                np.mean(epoch_losses),
            "train_contrastive":
                np.mean(epoch_con),
            "train_distill":
                np.mean(epoch_distill),
            "val_auroc":
                val_auc,
            "val_auprc":
                val_ap,
            "val_selection":
                selection,
            "val_contrastive":
                val_con,
            "checkpoint_improved":
                int(improved),
            "checkpoint_reason":
                reason,
        })

        print(
            f"{mode} seed={seed} "
            f"epoch={epoch:02d} "
            f"loss={np.mean(epoch_losses):.6f} "
            f"val_auc={val_auc:.6f} "
            f"val_ap={val_ap:.6f} "
            f"sel={selection:.6f} "
            f"patience={no_improve}/{PATIENCE}"
        )

        if no_improve >= PATIENCE:
            break

    if best_state is None:
        fail(
            f"No checkpoint selected: {mode} seed={seed}"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    val_scores, val_dn = pair_scores(
        model,
        val,
    )

    threshold = choose_threshold(
        val_scores,
        val["y"],
    )

    val_metric = metric_dict(
        val_scores,
        val["y"],
        threshold,
    )

    test_scores, test_dn = pair_scores(
        model,
        test,
    )

    test_metric = metric_dict(
        test_scores,
        test["y"],
        threshold,
    )

    pos_mask = (
        test["y"] == 1
    )

    neg_mask = (
        test["y"] == 0
    )

    infneg_mask = (
        neg_mask
        &
        (
            test["teacher_distance"]
            < TAU
        )
    )

    rho_all = teacher_rho(
        test_dn,
        test,
        np.ones(
            len(test_dn),
            dtype=bool,
        ),
    )

    rho_pos = teacher_rho(
        test_dn,
        test,
        pos_mask,
    )

    rho_neg = teacher_rho(
        test_dn,
        test,
        neg_mask,
    )

    rho_infneg = teacher_rho(
        test_dn,
        test,
        infneg_mask,
    )

    pred = test["df"].copy()

    pred["model_seed"] = seed
    pred["mode"] = mode
    pred["student_score"] = test_scores
    pred["student_distance_norm"] = test_dn
    pred["teacher_distance"] = test[
        "teacher_distance"
    ]

    pred.to_csv(
        out /
        "test_pair_predictions.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        out /
        "epoch_history.tsv",
        sep="\t",
        index=False,
    )

    metrics = {
        "mode":
            mode,
        "model_seed":
            seed,
        "split_protocol":
            "G2",
        "split_seed":
            SPLIT_SEED,
        "best_epoch":
            best_epoch,
        "checkpoint_reason":
            best_reason,
        "validation_threshold":
            threshold,
        "val_auroc":
            val_metric["auroc"],
        "val_auprc":
            val_metric["auprc"],
        "val_mcc":
            val_metric["mcc"],
        "test_auroc":
            test_metric["auroc"],
        "test_auprc":
            test_metric["auprc"],
        "test_mcc":
            test_metric["mcc"],
        "test_balanced_accuracy":
            test_metric[
                "balanced_accuracy"
            ],
        "test_f1":
            test_metric["f1"],
        "test_precision":
            test_metric["precision"],
        "test_recall":
            test_metric["recall"],
        "teacher_spearman_all":
            rho_all,
        "teacher_spearman_positive":
            rho_pos,
        "teacher_spearman_negative":
            rho_neg,
        "teacher_spearman_informative_negative":
            rho_infneg,
        "normalize_output":
            normalize,
        "distill_loss":
            DISTILL_LOSS,
        "tau":
            TAU,
        "lambda_dist":
            LAMBDA_DIST,
        "margin":
            MARGIN,
        "elapsed_seconds":
            time.perf_counter()
            - t_run,
    }

    (
        out /
        "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    (
        out /
        "RUN_COMPLETE"
    ).write_text(
        "RUN_COMPLETE\n"
    )

    return metrics


# ======================================================================
# Execute 6 x 5 = 30 frozen ablation runs
# ======================================================================

results = []

for mode in MODES:
    for seed in MODEL_SEEDS:
        results.append(
            train_one(
                mode,
                seed,
            )
        )

seed_df = pd.DataFrame(
    results
).sort_values(
    [
        "mode",
        "model_seed",
    ]
)

seed_df.to_csv(
    SRC /
    "E6_D0_D5_SEED_LEVEL_METRICS.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# 5-seed descriptive CI
# This is NOT the later species-block formal statistical analysis.
# ======================================================================

metrics_for_summary = [
    "test_auroc",
    "test_auprc",
    "test_mcc",
    "test_balanced_accuracy",
    "test_f1",
    "teacher_spearman_all",
    "teacher_spearman_positive",
    "teacher_spearman_informative_negative",
]


def bootstrap_mean_ci(
    x,
    seed=20260920,
    nboot=10000,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    b = rng.choice(
        x,
        size=(
            nboot,
            len(x),
        ),
        replace=True,
    ).mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                b,
                0.025,
            )
        ),
        float(
            np.quantile(
                b,
                0.975,
            )
        ),
    )


summary_rows = []

for mode, g in seed_df.groupby(
    "mode",
    sort=True,
):
    row = {
        "mode": mode,
        "n_model_seeds": len(g),
    }

    for m in metrics_for_summary:
        x = pd.to_numeric(
            g[m],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        lo, hi = bootstrap_mean_ci(
            x,
            seed=20260920
            + sum(
                ord(c)
                for c in mode + m
            ),
        )

        row[m + "_mean"] = np.nanmean(x)
        row[m + "_sd"] = np.nanstd(
            x,
            ddof=1,
        )
        row[m + "_seed_bootstrap_ci95_low"] = lo
        row[m + "_seed_bootstrap_ci95_high"] = hi

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SRC /
    "E6_D0_D5_5SEED_SUMMARY.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# Paired D3-minus-comparator model-seed effects.
# No significance p-values here; formal statistics is later.
# ======================================================================

delta_rows = []

d3 = seed_df[
    seed_df["mode"] == "D3"
].set_index(
    "model_seed"
)

for comp in [
    "D0",
    "D1",
    "D2",
    "D4",
    "D5",
]:
    other = seed_df[
        seed_df["mode"] == comp
    ].set_index(
        "model_seed"
    )

    common = sorted(
        set(
            d3.index
        )
        &
        set(
            other.index
        )
    )

    for m in [
        "test_auroc",
        "test_auprc",
        "test_mcc",
    ]:
        delta = np.asarray(
            [
                float(
                    d3.loc[
                        s,
                        m,
                    ]
                )
                -
                float(
                    other.loc[
                        s,
                        m,
                    ]
                )
                for s in common
            ],
            dtype=float,
        )

        lo, hi = bootstrap_mean_ci(
            delta,
            seed=20260920
            + sum(
                ord(c)
                for c in comp + m
            ),
        )

        delta_rows.append({
            "comparison":
                f"D3_minus_{comp}",
            "metric":
                m,
            "n_paired_model_seeds":
                len(delta),
            "mean_delta":
                float(
                    delta.mean()
                ),
            "sd_delta":
                float(
                    delta.std(
                        ddof=1
                    )
                ),
            "seed_bootstrap_ci95_low":
                lo,
            "seed_bootstrap_ci95_high":
                hi,
            "n_positive_delta":
                int(
                    (delta > 0).sum()
                ),
            "n_negative_delta":
                int(
                    (delta < 0).sum()
                ),
            "n_zero_delta":
                int(
                    (delta == 0).sum()
                ),
        })


delta_df = pd.DataFrame(
    delta_rows
)

delta_df.to_csv(
    SRC /
    "E6_D3_PAIRED_EFFECTS.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# Gate
# ======================================================================

expected_runs = {
    (
        m,
        s,
    )
    for m in MODES
    for s in MODEL_SEEDS
}

observed_runs = {
    (
        str(r["mode"]),
        int(r["model_seed"]),
    )
    for _, r in seed_df.iterrows()
}

if observed_runs != expected_runs:
    fail(
        f"E6 completion gate failed: "
        f"expected={len(expected_runs)}, "
        f"observed={len(observed_runs)}"
    )

for m in MODES:
    if (
        seed_df[
            seed_df["mode"] == m
        ]["model_seed"].nunique()
        != 5
    ):
        fail(
            f"{m}: does not have five model seeds"
        )


config = {
    "protocol":
        "G2",
    "taxonomy_split_seed":
        SPLIT_SEED,
    "model_seeds":
        MODEL_SEEDS,
    "modes":
        {
            "D0":
                "contrastive_only",
            "D1":
                "full_distillation_all_pairs",
            "D2":
                "negative_only_all_interspecies",
            "D3":
                "positives_plus_informative_negatives_teacher_distance_lt_0.98",
            "D4":
                "D3_mask_with_shuffled_teacher_targets",
            "D5":
                "D3_without_output_L2_normalization",
        },
    "input_dim":
        INPUT_DIM,
    "architecture":
        [
            2080,
            256,
            128,
            64,
        ],
    "batchnorm_safe_forward":
        "concat_A_B_then_forward_then_split",
    "margin":
        MARGIN,
    "tau":
        TAU,
    "lambda_dist":
        LAMBDA_DIST,
    "distillation_loss":
        DISTILL_LOSS,
    "optimizer":
        "AdamW",
    "learning_rate":
        LR,
    "weight_decay":
        WEIGHT_DECAY,
    "batch_size":
        BATCH,
    "epoch_positive_samples":
        N_POS_EPOCH,
    "epoch_negative_samples":
        N_NEG_EPOCH,
    "max_epochs":
        MAX_EPOCHS,
    "patience":
        PATIENCE,
    "gradient_clip":
        GRAD_CLIP,
    "precision":
        "FP32",
    "checkpoint_primary":
        "mean_validation_AUROC_AUPRC",
    "checkpoint_tie_break":
        "lower_validation_contrastive_loss_within_1e-8",
    "threshold":
        "validation_max_MCC_then_BAcc_then_F1_then_closest_0.5_then_lower",
    "device":
        str(device),
}


(
    OUT /
    "E6_FROZEN_CONFIG.json"
).write_text(
    json.dumps(
        config,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


report = OUT / "E6_FINAL_REPORT.txt"

with report.open("w") as f:
    f.write(
        "HGV-V3 E6 D0-D5 MECHANISTIC ABLATION\n"
    )
    f.write("=" * 80 + "\n")
    f.write(
        "Canonical protocol: G2 taxonomy split seed 20260915\n"
    )
    f.write(
        "Model seeds: 20260915,20260916,20260917,20260918,20260919\n"
    )
    f.write(
        "Runs: 6 modes x 5 seeds = 30\n"
    )
    f.write(
        f"Device: {device}\n"
    )
    f.write(
        f"Distillation regression loss: {DISTILL_LOSS}\n"
    )
    f.write(
        "No external data were used for training/model selection.\n"
    )
    f.write(
        "Test threshold was selected on validation only.\n"
    )
    f.write(
        "Seed-bootstrap CI in this block is descriptive model-seed uncertainty; "
        "it is not the later species-block formal statistics.\n\n"
    )

    f.write(
        summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nD3 paired effects:\n"
    )

    f.write(
        delta_df.to_string(
            index=False
        )
    )

    f.write("\n")


hash_files = [
    FEATURE_NPY,
    FEATURE_MAP_TSV,
    TRAIN_TSV,
    VAL_TSV,
    TEST_TSV,
    D2_SIM,
    SRC /
    "E6_D0_D5_SEED_LEVEL_METRICS.tsv",
    SRC /
    "E6_D0_D5_5SEED_SUMMARY.tsv",
    SRC /
    "E6_D3_PAIRED_EFFECTS.tsv",
    OUT /
    "E6_FROZEN_CONFIG.json",
    report,
]

with (
    OUT /
    "E6_KEY_SHA256.txt"
).open("w") as f:

    for p in hash_files:
        f.write(
            f"{sha256_file(p)}  {p}\n"
        )


(
    OUT /
    "RUN_COMPLETE"
).write_text(
    "E6_D0_D5_30RUN_COMPLETE\n"
)


print()
print("=" * 100)
print("E6 COMPLETE")
print("=" * 100)
print()
print(
    summary_df.to_string(
        index=False
    )
)
print()
print("Paired D3 effects:")
print(
    delta_df.to_string(
        index=False
    )
)
print()
print("Output:", OUT)
