#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

PAIR_ROOT = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
)

PAIR_SHA = (
    PAIR_ROOT
    / "SHA256SUMS.txt"
)

TRAIN_ROOT = (
    BASE
    / "08_main_generalization/"
      "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
)

THRESHOLD_ROOT = (
    BASE
    / "08_main_generalization/"
      "validation_threshold_freeze_v1_20260917"
)

THRESHOLD_FILE = (
    THRESHOLD_ROOT
    / "FROZEN_VALIDATION_THRESHOLDS.tsv"
)

THRESHOLD_SHA = (
    THRESHOLD_ROOT
    / "SHA256SUMS.txt"
)

TEST_PROTOCOL_ROOT = (
    BASE
    / "08_main_generalization/"
      "test_eval_protocol_freeze_v1_20260917"
)

TEST_PROTOCOL = (
    TEST_PROTOCOL_ROOT
    / "TEST_EVAL_PROTOCOL.txt"
)

TEST_PROTOCOL_SHA = (
    TEST_PROTOCOL_ROOT
    / "SHA256SUMS.txt"
)

OUT = (
    BASE
    / "08_main_generalization/"
      "blind_test_evaluation_v1_20260917"
)


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

HGV_METHOD = (
    "HGV_Dashing2_tau0p98"
)

METRICS = [
    "AUROC",
    "AUPRC",
    "MCC",
    "F1",
    "BalancedAccuracy",
    "Precision",
    "Recall",
    "Specificity",
]

INPUT_DIM = 2080
EMBED_BATCH = 1024


def clean(x):
    return "" if x is None else str(x).strip()


def sha256(path):
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


def verify_sha_manifest(
    root,
    manifest,
):

    if not manifest.is_file():
        raise RuntimeError(
            f"Missing SHA256 manifest: {manifest}"
        )

    checked = 0

    with open(
        manifest,
        "r",
        encoding="utf-8",
    ) as f:

        for raw in f:

            raw = raw.strip()

            if not raw:
                continue

            parts = raw.split(
                None,
                1,
            )

            if len(parts) != 2:
                raise RuntimeError(
                    f"Malformed SHA line: {raw}"
                )

            expected = parts[0]
            rel = parts[1].strip()

            if rel.startswith("*"):
                rel = rel[1:]

            p = root / rel

            if not p.is_file():
                raise RuntimeError(
                    f"SHA file missing: {p}"
                )

            actual = sha256(p)

            if actual != expected:
                raise RuntimeError(
                    f"SHA mismatch: {p}"
                )

            checked += 1

    return checked


def sha_manifest_lookup(
    manifest,
):

    result = {}

    with open(
        manifest,
        "r",
        encoding="utf-8",
    ) as f:

        for raw in f:

            raw = raw.strip()

            if not raw:
                continue

            digest, rel = raw.split(
                None,
                1,
            )

            rel = rel.strip()

            if rel.startswith("*"):
                rel = rel[1:]

            result[rel] = digest

    return result


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


def accession_col(fields):

    return detect_col(
        fields,
        [
            "assembly_accession",
            "assembly_accession_final",
            "accession",
        ],
        "accession",
    )


def pair_columns(fields):

    a = detect_col(
        fields,
        [
            "accession_a",
            "assembly_accession_a",
            "genome_a",
            "a_accession",
        ],
        "endpoint A",
    )

    b = detect_col(
        fields,
        [
            "accession_b",
            "assembly_accession_b",
            "genome_b",
            "b_accession",
        ],
        "endpoint B",
    )

    y = detect_col(
        fields,
        [
            "label",
            "y",
            "target",
            "same_species",
        ],
        "label",
    )

    ptype = None

    for c in [
        "pair_type",
        "negative_tier",
        "neg_type",
        "stratum",
    ]:
        if c in fields:
            ptype = c
            break

    if ptype is None:
        raise RuntimeError(
            "Cannot detect pair-type column."
        )

    return (
        a,
        b,
        y,
        ptype,
    )


def canonical_pair_type(
    label,
    raw,
):

    if label == 1:
        return "positive"

    s = clean(raw).lower()

    if "hard" in s:
        return "hard"

    if "medium" in s:
        return "medium"

    if "easy" in s:
        return "easy"

    raise RuntimeError(
        f"Unknown negative pair type: {raw}"
    )


def confusion_metrics(
    labels,
    scores,
    threshold,
):

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    pred = (
        s >= float(threshold)
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
        if (
            precision + recall
        ) > 0
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
        "BalancedAccuracy":
            balanced,
        "Precision":
            precision,
        "Recall":
            recall,
        "Specificity":
            specificity,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def metric_row(
    labels,
    scores,
    threshold,
):

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(np.unique(y)) != 2:
        raise RuntimeError(
            "Subset does not contain both classes."
        )

    rank = {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    s,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    s,
                )
            ),
    }

    cls = confusion_metrics(
        y,
        s,
        threshold,
    )

    return {
        **rank,
        **cls,
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

    chunks = []

    for start in range(
        0,
        X_device.shape[0],
        EMBED_BATCH,
    ):

        end = min(
            start + EMBED_BATCH,
            X_device.shape[0],
        )

        chunks.append(
            model(
                X_device[
                    start:end
                ]
            ).cpu()
        )

    return torch.cat(
        chunks,
        dim=0,
    ).numpy().astype(
        np.float32
    )


def mean_sd(values):

    vals = [
        float(v)
        for v in values
        if math.isfinite(
            float(v)
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
        if len(vals) >= 2
        else 0.0
    )

    return mean, sd


# ==============================================================
# 0. Preflight BEFORE opening test
# ==============================================================

print("=" * 78)
print("HGV-V3 LOCKED BLIND TEST EVALUATION")
print("=" * 78)

print(
    "TEST WILL BE OPENED BY THIS SCRIPT"
)

print(
    "external_used=False"
)

if OUT.exists():
    raise SystemExit(
        f"ERROR: final output already exists: {OUT}"
    )

print()
print("[0] VERIFY ALL FROZEN INPUTS")

n_thr_sha = verify_sha_manifest(
    THRESHOLD_ROOT,
    THRESHOLD_SHA,
)

n_protocol_sha = (
    verify_sha_manifest(
        TEST_PROTOCOL_ROOT,
        TEST_PROTOCOL_SHA,
    )
)

if not (
    TRAIN_ROOT
    / "TRAIN_GRID_COMPLETE"
).is_file():

    raise RuntimeError(
        "Missing TRAIN_GRID_COMPLETE."
    )

fields_thr, rows_thr = read_tsv(
    THRESHOLD_FILE
)

if len(rows_thr) != 72:
    raise RuntimeError(
        f"Expected 72 frozen thresholds, "
        f"found {len(rows_thr)}"
    )

# Verify all 27 checkpoints exist.
for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        for model_seed in MODEL_SEEDS:

            run_dir = (
                TRAIN_ROOT
                / protocol
                / f"split{split_seed}"
                / f"model{model_seed}"
            )

            for fname in [
                "RUN_COMPLETE",
                "best_model.pt",
                "run_summary.json",
                "config.json",
            ]:

                p = (
                    run_dir
                    / fname
                )

                if not p.is_file():
                    raise RuntimeError(
                        f"Missing frozen run input: {p}"
                    )

            config = json.loads(
                (
                    run_dir
                    / "config.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            if config.get(
                "test_used"
            ) is not False:

                raise RuntimeError(
                    f"Run claims test use: {run_dir}"
                )

            if config.get(
                "external_used"
            ) is not False:

                raise RuntimeError(
                    f"Run claims external use: {run_dir}"
                )

print(
    f"threshold_sha_files_checked="
    f"{n_thr_sha}"
)

print(
    f"protocol_sha_files_checked="
    f"{n_protocol_sha}"
)

print(
    "frozen_thresholds=72"
)

print(
    "frozen_checkpoints=27"
)

# ==============================================================
# 1. Load frozen thresholds
# ==============================================================

print()
print("[1] LOAD FROZEN THRESHOLDS")

hgv_threshold = {}
baseline_threshold = {}

for r in rows_thr:

    mtype = clean(
        r["method_type"]
    )

    method = clean(
        r["method"]
    )

    protocol = clean(
        r["protocol"]
    )

    split_seed = int(
        r["split_seed"]
    )

    model_seed_raw = clean(
        r["model_seed"]
    )

    threshold = float(
        r["threshold"]
    )

    if mtype == "HGV":

        model_seed = int(
            model_seed_raw
        )

        key = (
            method,
            protocol,
            split_seed,
            model_seed,
        )

        if key in hgv_threshold:
            raise RuntimeError(
                f"Duplicate HGV threshold: {key}"
            )

        hgv_threshold[
            key
        ] = threshold

    elif mtype == "baseline":

        key = (
            method,
            protocol,
            split_seed,
        )

        if key in baseline_threshold:
            raise RuntimeError(
                f"Duplicate baseline threshold: {key}"
            )

        baseline_threshold[
            key
        ] = threshold

    else:
        raise RuntimeError(
            f"Unknown method_type: {mtype}"
        )

if len(hgv_threshold) != 27:
    raise RuntimeError(
        f"Expected 27 HGV thresholds, "
        f"got {len(hgv_threshold)}"
    )

if len(baseline_threshold) != 45:
    raise RuntimeError(
        f"Expected 45 baseline thresholds, "
        f"got {len(baseline_threshold)}"
    )

# ==============================================================
# 2. Load features
# ==============================================================

print()
print("[2] LOAD FROZEN FEATURES")

ff, fr = read_tsv(
    FEATURE_ROWS
)

fc = accession_col(
    ff
)

feature_accessions = [
    clean(r[fc])
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
    INPUT_DIM,
):
    raise RuntimeError(
        f"Bad feature shape: {X.shape}"
    )

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

X_device = torch.as_tensor(
    np.asarray(
        X,
        dtype=np.float32,
    ),
    dtype=torch.float32,
    device=device,
)

print(
    f"feature_shape={X.shape}"
)

print(
    f"device={device}"
)

if torch.cuda.is_available():

    print(
        "gpu="
        + torch.cuda.get_device_name(
            0
        )
    )

# ==============================================================
# 3. FIRST OPENING OF TEST
# ==============================================================

print()
print("[3] OPEN FROZEN TEST MANIFESTS")

OUT.mkdir(
    parents=True,
)

(
    OUT
    / "hgv_test_predictions"
).mkdir()

(
    OUT
    / "baseline_test_predictions"
).mkdir()

pair_sha_lookup = (
    sha_manifest_lookup(
        PAIR_SHA
    )
)

test_data = {}
test_open_rows = []

for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        split_id = (
            f"{protocol}_seed{split_seed}"
        )

        rel = (
            f"{split_id}/test_pairs.tsv"
        )

        path = (
            PAIR_ROOT
            / split_id
            / "test_pairs.tsv"
        )

        if rel not in pair_sha_lookup:
            raise RuntimeError(
                f"Frozen pair SHA missing: {rel}"
            )

        actual_sha = sha256(
            path
        )

        expected_sha = (
            pair_sha_lookup[
                rel
            ]
        )

        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Test pair SHA mismatch: {path}"
            )

        pf, pr = read_tsv(
            path
        )

        (
            a_col,
            b_col,
            y_col,
            ptype_col,
        ) = pair_columns(
            pf
        )

        acc_a = []
        acc_b = []
        labels = []
        ptypes = []

        for r in pr:

            a = clean(
                r[a_col]
            )

            b = clean(
                r[b_col]
            )

            if (
                a not in acc_to_idx
                or b not in acc_to_idx
            ):
                raise RuntimeError(
                    f"Unknown test accession: "
                    f"{a}, {b}"
                )

            y = int(
                float(
                    r[y_col]
                )
            )

            if y not in (
                0,
                1,
            ):
                raise RuntimeError(
                    f"Invalid test label: {y}"
                )

            ptype = (
                canonical_pair_type(
                    y,
                    r[
                        ptype_col
                    ],
                )
            )

            acc_a.append(a)
            acc_b.append(b)
            labels.append(y)
            ptypes.append(ptype)

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        if set(
            np.unique(
                labels
            ).tolist()
        ) != {0, 1}:
            raise RuntimeError(
                f"Test classes invalid: {split_id}"
            )

        test_data[
            (
                protocol,
                split_seed,
            )
        ] = {
            "path":
                path,

            "sha256":
                actual_sha,

            "accession_a":
                np.asarray(
                    acc_a,
                    dtype=object,
                ),

            "accession_b":
                np.asarray(
                    acc_b,
                    dtype=object,
                ),

            "labels":
                labels,

            "pair_type":
                np.asarray(
                    ptypes,
                    dtype=object,
                ),

            "a_idx":
                np.asarray(
                    [
                        acc_to_idx[a]
                        for a in acc_a
                    ],
                    dtype=np.int64,
                ),

            "b_idx":
                np.asarray(
                    [
                        acc_to_idx[b]
                        for b in acc_b
                    ],
                    dtype=np.int64,
                ),
        }

        counts = {
            k: int(
                np.sum(
                    np.asarray(
                        ptypes
                    ) == k
                )
            )
            for k in [
                "positive",
                "hard",
                "medium",
                "easy",
            ]
        }

        test_open_rows.append({
            "protocol":
                protocol,

            "split_seed":
                split_seed,

            "test_pairs":
                len(labels),

            "positive":
                counts[
                    "positive"
                ],

            "hard":
                counts[
                    "hard"
                ],

            "medium":
                counts[
                    "medium"
                ],

            "easy":
                counts[
                    "easy"
                ],

            "sha256":
                actual_sha,
        })

        print(
            f"OPENED {split_id} "
            f"pairs={len(labels)} "
            f"pos={counts['positive']} "
            f"hard={counts['hard']} "
            f"medium={counts['medium']} "
            f"easy={counts['easy']}"
        )

write_tsv(
    OUT
    / "TEST_OPEN_AUDIT.tsv",
    test_open_rows,
    [
        "protocol",
        "split_seed",
        "test_pairs",
        "positive",
        "hard",
        "medium",
        "easy",
        "sha256",
    ],
)

# ==============================================================
# Helper: evaluate all + strata
# ==============================================================

def append_metric_rows(
    output,
    method_type,
    method,
    protocol,
    split_seed,
    model_seed,
    labels,
    ptypes,
    scores,
    threshold,
):

    subsets = {
        "all":
            np.ones(
                len(labels),
                dtype=bool,
            ),

        "hard":
            (
                (labels == 1)
                |
                (ptypes == "hard")
            ),

        "medium":
            (
                (labels == 1)
                |
                (ptypes == "medium")
            ),

        "easy":
            (
                (labels == 1)
                |
                (ptypes == "easy")
            ),
    }

    for subset, mask in subsets.items():

        y = labels[
            mask
        ]

        s = scores[
            mask
        ]

        m = metric_row(
            y,
            s,
            threshold,
        )

        output.append({
            "method_type":
                method_type,

            "method":
                method,

            "protocol":
                protocol,

            "split_seed":
                split_seed,

            "model_seed":
                model_seed,

            "subset":
                subset,

            "n_pairs":
                int(
                    len(y)
                ),

            "n_positive":
                int(
                    np.sum(
                        y == 1
                    )
                ),

            "n_negative":
                int(
                    np.sum(
                        y == 0
                    )
                ),

            "threshold":
                float(
                    threshold
                ),

            "AUROC":
                m[
                    "AUROC"
                ],

            "AUPRC":
                m[
                    "AUPRC"
                ],

            "MCC":
                m[
                    "MCC"
                ],

            "F1":
                m[
                    "F1"
                ],

            "BalancedAccuracy":
                m[
                    "BalancedAccuracy"
                ],

            "Precision":
                m[
                    "Precision"
                ],

            "Recall":
                m[
                    "Recall"
                ],

            "Specificity":
                m[
                    "Specificity"
                ],

            "TP":
                m["TP"],

            "FP":
                m["FP"],

            "TN":
                m["TN"],

            "FN":
                m["FN"],
        })


metric_rows = []

# ==============================================================
# 4. HGV locked checkpoint evaluation
# ==============================================================

print()
print("[4] EVALUATE 27 FROZEN HGV CHECKPOINTS")

for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        td = test_data[
            (
                protocol,
                split_seed,
            )
        ]

        a_idx = td[
            "a_idx"
        ]

        b_idx = td[
            "b_idx"
        ]

        labels = td[
            "labels"
        ]

        ptypes = td[
            "pair_type"
        ]

        for model_seed in MODEL_SEEDS:

            run_dir = (
                TRAIN_ROOT
                / protocol
                / f"split{split_seed}"
                / f"model{model_seed}"
            )

            ckpt_path = (
                run_dir
                / "best_model.pt"
            )

            checkpoint = torch.load(
                ckpt_path,
                map_location=device,
                weights_only=False,
            )

            model = HGV().to(
                device
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            Z = embed_all(
                model,
                X_device,
            )

            d = np.linalg.norm(
                Z[a_idx].astype(
                    np.float64
                )
                - Z[b_idx].astype(
                    np.float64
                ),
                axis=1,
            )

            scores = (
                1.0
                - d / 2.0
            )

            key = (
                HGV_METHOD,
                protocol,
                split_seed,
                model_seed,
            )

            threshold = (
                hgv_threshold[
                    key
                ]
            )

            append_metric_rows(
                metric_rows,
                "HGV",
                HGV_METHOD,
                protocol,
                split_seed,
                model_seed,
                labels,
                ptypes,
                scores,
                threshold,
            )

            pred = (
                scores
                >= threshold
            ).astype(
                np.int64
            )

            pred_rows = []

            for i in range(
                len(labels)
            ):

                pred_rows.append({
                    "accession_a":
                        td[
                            "accession_a"
                        ][i],

                    "accession_b":
                        td[
                            "accession_b"
                        ][i],

                    "label":
                        int(
                            labels[i]
                        ),

                    "pair_type":
                        ptypes[i],

                    "distance":
                        float(
                            d[i]
                        ),

                    "similarity_score":
                        float(
                            scores[i]
                        ),

                    "frozen_threshold":
                        float(
                            threshold
                        ),

                    "predicted_label":
                        int(
                            pred[i]
                        ),
                })

            pred_path = (
                OUT
                / "hgv_test_predictions"
                / (
                    f"{protocol}"
                    f"_split{split_seed}"
                    f"_model{model_seed}.tsv"
                )
            )

            write_tsv(
                pred_path,
                pred_rows,
                [
                    "accession_a",
                    "accession_b",
                    "label",
                    "pair_type",
                    "distance",
                    "similarity_score",
                    "frozen_threshold",
                    "predicted_label",
                ],
            )

            all_row = (
                metric_rows[-4]
            )

            print(
                f"HGV {protocol} "
                f"split={split_seed} "
                f"model={model_seed} "
                f"AUC="
                f"{all_row['AUROC']:.6f} "
                f"AUPRC="
                f"{all_row['AUPRC']:.6f} "
                f"MCC="
                f"{all_row['MCC']:.6f}"
            )

            del model
            del checkpoint

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ==============================================================
# 5. Frozen baseline evaluation
# ==============================================================

print()
print("[5] EVALUATE FROZEN RAW/PCA BASELINES")

for protocol in PROTOCOLS:

    for split_seed in SPLIT_SEEDS:

        td = test_data[
            (
                protocol,
                split_seed,
            )
        ]

        labels = td[
            "labels"
        ]

        ptypes = td[
            "pair_type"
        ]

        idx_a = td[
            "a_idx"
        ]

        idx_b = td[
            "b_idx"
        ]

        A = np.asarray(
            X[idx_a],
            dtype=np.float32,
        )

        B = np.asarray(
            X[idx_b],
            dtype=np.float32,
        )

        A64 = A.astype(
            np.float64
        )

        B64 = B.astype(
            np.float64
        )

        # Raw Bray-Curtis similarity.
        bray = (
            1.0
            - 0.5
            * np.abs(
                A64 - B64
            ).sum(
                axis=1
            )
        )

        # Raw cosine.
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

        # Raw Euclidean ranking score.
        raw_euclidean = (
            -np.linalg.norm(
                A64 - B64,
                axis=1,
            )
        )

        # Frozen PCA parameters.
        pca_path = (
            THRESHOLD_ROOT
            / "pca_models"
            / (
                f"{protocol}"
                f"_seed{split_seed}.npz"
            )
        )

        pca_npz = np.load(
            pca_path
        )

        pca_mean = (
            pca_npz[
                "mean"
            ].astype(
                np.float32
            )
        )

        pca_components = (
            pca_npz[
                "components"
            ].astype(
                np.float32
            )
        )

        unique_idx = np.unique(
            np.concatenate(
                [
                    idx_a,
                    idx_b,
                ]
            )
        )

        X_test_unique = np.asarray(
            X[unique_idx],
            dtype=np.float32,
        )

        P = (
            (
                X_test_unique
                - pca_mean
            )
            @ pca_components.T
        ).astype(
            np.float32
        )

        p_lookup = {
            int(idx):
                P[i]
            for i, idx
            in enumerate(
                unique_idx
            )
        }

        PA = np.stack([
            p_lookup[
                int(i)
            ]
            for i in idx_a
        ])

        PB = np.stack([
            p_lookup[
                int(i)
            ]
            for i in idx_b
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

        # Save one prediction table per split.
        pred_rows = []

        for i in range(
            len(labels)
        ):

            row = {
                "accession_a":
                    td[
                        "accession_a"
                    ][i],

                "accession_b":
                    td[
                        "accession_b"
                    ][i],

                "label":
                    int(
                        labels[i]
                    ),

                "pair_type":
                    ptypes[i],
            }

            for method in BASELINES:

                score = float(
                    method_scores[
                        method
                    ][i]
                )

                threshold = (
                    baseline_threshold[
                        (
                            method,
                            protocol,
                            split_seed,
                        )
                    ]
                )

                row[
                    method
                ] = score

                row[
                    method
                    + "_pred"
                ] = int(
                    score
                    >= threshold
                )

            pred_rows.append(
                row
            )

        baseline_pred_path = (
            OUT
            / "baseline_test_predictions"
            / (
                f"{protocol}"
                f"_seed{split_seed}.tsv"
            )
        )

        baseline_fields = [
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
        ]

        for method in BASELINES:
            baseline_fields.extend([
                method,
                method + "_pred",
            ])

        write_tsv(
            baseline_pred_path,
            pred_rows,
            baseline_fields,
        )

        for method in BASELINES:

            scores = (
                method_scores[
                    method
                ]
            )

            threshold = (
                baseline_threshold[
                    (
                        method,
                        protocol,
                        split_seed,
                    )
                ]
            )

            before = len(
                metric_rows
            )

            append_metric_rows(
                metric_rows,
                "baseline",
                method,
                protocol,
                split_seed,
                "",
                labels,
                ptypes,
                scores,
                threshold,
            )

            all_row = (
                metric_rows[
                    before
                ]
            )

            print(
                f"{method:20s} "
                f"{protocol} "
                f"split={split_seed} "
                f"AUC="
                f"{all_row['AUROC']:.6f} "
                f"AUPRC="
                f"{all_row['AUPRC']:.6f} "
                f"MCC="
                f"{all_row['MCC']:.6f}"
            )

# ==============================================================
# 6. Save run-level metrics
# ==============================================================

print()
print("[6] WRITE RUN-LEVEL TEST METRICS")

run_metric_fields = [
    "method_type",
    "method",
    "protocol",
    "split_seed",
    "model_seed",
    "subset",
    "n_pairs",
    "n_positive",
    "n_negative",
    "threshold",
    "AUROC",
    "AUPRC",
    "MCC",
    "F1",
    "BalancedAccuracy",
    "Precision",
    "Recall",
    "Specificity",
    "TP",
    "FP",
    "TN",
    "FN",
]

write_tsv(
    OUT
    / "test_metrics_run_level.tsv",
    metric_rows,
    run_metric_fields,
)

# ==============================================================
# 7. Aggregate model seeds within taxonomy split
# ==============================================================

print()
print("[7] AGGREGATE MODEL SEEDS WITHIN SPLIT")

groups = defaultdict(list)

for r in metric_rows:

    key = (
        r["method_type"],
        r["method"],
        r["protocol"],
        int(
            r["split_seed"]
        ),
        r["subset"],
    )

    groups[
        key
    ].append(
        r
    )

split_rows = []

for key, rows in sorted(
    groups.items()
):

    (
        method_type,
        method,
        protocol,
        split_seed,
        subset,
    ) = key

    expected = (
        3
        if method_type == "HGV"
        else 1
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Unexpected replicate count "
            f"for {key}: {len(rows)}"
        )

    out = {
        "method_type":
            method_type,

        "method":
            method,

        "protocol":
            protocol,

        "split_seed":
            split_seed,

        "subset":
            subset,

        "n_model_seeds":
            len(rows),

        "n_pairs":
            int(
                rows[0][
                    "n_pairs"
                ]
            ),
    }

    for metric in METRICS:

        mean, sd = mean_sd([
            r[
                metric
            ]
            for r in rows
        ])

        out[
            metric + "_mean"
        ] = mean

        out[
            metric
            + "_sd_model_seed"
        ] = sd

    split_rows.append(
        out
    )

split_fields = [
    "method_type",
    "method",
    "protocol",
    "split_seed",
    "subset",
    "n_model_seeds",
    "n_pairs",
]

for metric in METRICS:
    split_fields.extend([
        metric + "_mean",
        metric + "_sd_model_seed",
    ])

write_tsv(
    OUT
    / "test_metrics_split_level.tsv",
    split_rows,
    split_fields,
)

# ==============================================================
# 8. Aggregate taxonomy split seeds
# ==============================================================

print()
print("[8] AGGREGATE 3 TAXONOMY SPLIT SEEDS")

protocol_groups = defaultdict(
    list
)

for r in split_rows:

    key = (
        r[
            "method_type"
        ],
        r[
            "method"
        ],
        r[
            "protocol"
        ],
        r[
            "subset"
        ],
    )

    protocol_groups[
        key
    ].append(
        r
    )

protocol_rows = []

for key, rows in sorted(
    protocol_groups.items()
):

    (
        method_type,
        method,
        protocol,
        subset,
    ) = key

    if len(rows) != 3:
        raise RuntimeError(
            f"Expected 3 split seeds "
            f"for {key}, got {len(rows)}"
        )

    out = {
        "method_type":
            method_type,

        "method":
            method,

        "protocol":
            protocol,

        "subset":
            subset,

        "n_split_seeds":
            3,
    }

    for metric in METRICS:

        values = [
            r[
                metric
                + "_mean"
            ]
            for r in rows
        ]

        mean, sd = mean_sd(
            values
        )

        out[
            metric + "_mean"
        ] = mean

        out[
            metric
            + "_sd_split_seed"
        ] = sd

    protocol_rows.append(
        out
    )

protocol_fields = [
    "method_type",
    "method",
    "protocol",
    "subset",
    "n_split_seeds",
]

for metric in METRICS:
    protocol_fields.extend([
        metric + "_mean",
        metric + "_sd_split_seed",
    ])

write_tsv(
    OUT
    / "test_metrics_protocol_summary.tsv",
    protocol_rows,
    protocol_fields,
)

primary_rows = [
    r
    for r in protocol_rows
    if r[
        "subset"
    ] == "all"
]

write_tsv(
    OUT
    / "PRIMARY_ALLPAIR_TEST_SUMMARY.tsv",
    primary_rows,
    protocol_fields,
)

# ==============================================================
# 9. Human-readable report
# ==============================================================

print()
print("[9] WRITE FINAL TEST REPORT")

report_path = (
    OUT
    / "BLIND_TEST_EVALUATION_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 LOCKED BLIND TEST EVALUATION\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "STATUS: PASS\n\n"
    )

    f.write(
        "G2/G3/G4 test used: YES\n"
    )

    f.write(
        "External B1/B2 used: NO\n\n"
    )

    f.write(
        "Frozen HGV checkpoints evaluated: 27\n"
    )

    f.write(
        "Frozen HGV thresholds used: 27\n"
    )

    f.write(
        "Frozen baseline thresholds used: 45\n"
    )

    f.write(
        "PCA refit on test: NO\n"
    )

    f.write(
        "Threshold retuning on test: NO\n\n"
    )

    f.write(
        "Statistical unit: taxonomy split seed\n"
    )

    f.write(
        "HGV model seeds were averaged within "
        "each taxonomy split before protocol-level "
        "mean +/- SD.\n\n"
    )

    f.write(
        "PRIMARY ALL-PAIR TEST RESULTS\n"
    )

    f.write(
        "-" * 78
        + "\n"
    )

    order_methods = [
        HGV_METHOD,
        "RawK6_BrayCurtis",
        "RawK6_Cosine",
        "RawK6_Euclidean",
        "PCA64_Euclidean",
        "PCA64_L2",
    ]

    lookup = {
        (
            r["method"],
            r["protocol"],
        ): r
        for r in primary_rows
    }

    for protocol in PROTOCOLS:

        f.write(
            f"\n[{protocol}]\n"
        )

        for method in order_methods:

            r = lookup[
                (
                    method,
                    protocol,
                )
            ]

            f.write(
                f"{method:24s} "
                f"AUROC="
                f"{r['AUROC_mean']:.6f}"
                f" +/- "
                f"{r['AUROC_sd_split_seed']:.6f}; "
                f"AUPRC="
                f"{r['AUPRC_mean']:.6f}"
                f" +/- "
                f"{r['AUPRC_sd_split_seed']:.6f}; "
                f"MCC="
                f"{r['MCC_mean']:.6f}"
                f" +/- "
                f"{r['MCC_sd_split_seed']:.6f}\n"
            )

    f.write("\n")

    f.write(
        "No model, teacher, tau, checkpoint, PCA "
        "transform, or decision threshold was selected "
        "using test results.\n\n"
    )

    f.write(
        "BLIND_TEST_EVALUATION=PASS\n"
    )

# ==============================================================
# 10. Provenance
# ==============================================================

provenance = {
    "status":
        "PASS",

    "evaluation_started_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "test_used":
        True,

    "external_used":
        False,

    "protocols":
        PROTOCOLS,

    "split_seeds":
        SPLIT_SEEDS,

    "model_seeds":
        MODEL_SEEDS,

    "hgv_checkpoints":
        27,

    "frozen_hgv_thresholds":
        27,

    "frozen_baseline_thresholds":
        45,

    "pca_refit_on_test":
        False,

    "test_threshold_tuning":
        False,

    "aggregation":
        (
            "average 3 model seeds within each taxonomy "
            "split, then mean+SD across 3 taxonomy split seeds"
        ),

    "device":
        str(
            device
        ),

    "torch":
        torch.__version__,

    "numpy":
        np.__version__,

    "python":
        sys.version,

    "platform":
        platform.platform(),

    "threshold_file_sha256":
        sha256(
            THRESHOLD_FILE
        ),

    "test_protocol_sha256":
        sha256(
            TEST_PROTOCOL
        ),
}

(
    OUT
    / "blind_test_provenance.json"
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
# 11. SHA256
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
print("LOCKED BLIND TEST EVALUATION COMPLETE")
print("=" * 78)

print(
    f"Output: {OUT}"
)

print(
    "test_used=True"
)

print(
    "external_used=False"
)

print(
    "BLIND_TEST_EVALUATION=PASS"
)
