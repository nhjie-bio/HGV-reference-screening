#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# HGV-V3 E6
# Canonical G2 locked test evaluation
#
# Frozen semantics:
#
# D0-D4:
#   HGV output = L2-normalized 64-D embedding
#
# D5:
#   HGV output = raw 64-D embedding (NO L2)
#
# Pair distance:
#   d = Euclidean(z_a, z_b)
#
# Score:
#   similarity_score = 1 - d / 2
#
# Prediction:
#   similarity_score >= frozen validation threshold
#
# IMPORTANT:
# - same frozen G2 test for all 30 cells
# - no threshold selection
# - no retraining
# - no model selection
# - no post-test tuning
# ============================================================


BASE = Path(
    "HGV_V3_revision_20260915"
)

E6ROOT = (
    BASE
    / "10_ablation"
    / "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

MANIFEST_DIR = (
    E6ROOT
    / "04_canonical_30cell_manifest_v1_20260923"
)

MANIFEST = (
    MANIFEST_DIR
    / "E6_30CELL_CANONICAL_MANIFEST.tsv"
)

THRESHOLD_DIR = (
    E6ROOT
    / "05_validation_threshold_freeze_v1_20260924"
)

THRESHOLD_FILE = (
    THRESHOLD_DIR
    / "E6_FROZEN_VALIDATION_THRESHOLDS.tsv"
)

THRESHOLD_FREEZE = (
    THRESHOLD_DIR
    / "E6_VALIDATION_THRESHOLD_FREEZE_v1.txt"
)

SOURCE_17C2 = (
    BASE
    / "scripts_v3_20260916_hgv"
    / "17c2_blind_test_evaluation_v1.py"
)

FEATURE_ROOT = (
    BASE
    / "03_features"
    / "canonical_k6_full3154_v1"
)

TEST_PAIRS = (
    BASE
    / "05_pairs"
    / "pair_manifests_v1_20260916"
    / "G2_seed20260915"
    / "test_pairs.tsv"
)

OUT = (
    E6ROOT
    / "06_canonical_g2_test_v1_20260924"
)

PRED_DIR = (
    OUT
    / "cell_predictions"
)

METRICS_TSV = (
    OUT
    / "E6_G2_TEST_RUN_METRICS.tsv"
)

SUMMARY_TSV = (
    OUT
    / "E6_G2_TEST_MODE_SUMMARY.tsv"
)

NORM_TSV = (
    OUT
    / "E6_G2_EMBEDDING_NORM_AUDIT.tsv"
)

PARITY_TSV = (
    OUT
    / "E6_D3_MAIN_TEST_PARITY.tsv"
)

PROVENANCE_JSON = (
    OUT
    / "E6_G2_TEST_PROVENANCE.json"
)

REPORT = (
    OUT
    / "E6_G2_TEST_REPORT.txt"
)

SHA_FILE = (
    OUT
    / "E6_G2_TEST_SHA256.txt"
)

FREEZE_FILE = (
    OUT
    / "E6_G2_TEST_FREEZE_v1.txt"
)


EXPECTED_MANIFEST_SHA = (
    "dde063cbc4697aeec43b80480223509427a37c0afafa182cac1008aaf268fd26"
)

EXPECTED_THRESHOLD_SHA = (
    "ad75bdb8508fe520e1a879302932b2561af21ab07425e4d4cf2acdae53f9e5bc"
)

EXPECTED_17C2_SHA = (
    "44a7e6e03881ccb87f393af490c85f7e1f60f9d24b3870af187ccac7412fe2ad"
)

EXPECTED_TEST_SHA = (
    "4ff432b84bcfe76743e6ca7d6ce842a094744e5be726527aaf74e09bc9e474ba"
)

EXPECTED_CELLS = 30
EXPECTED_TEST_PAIRS = 3872

MAIN_D3_SEEDS = {
    20260915,
    20260916,
    20260917,
}


# ============================================================
# Basic utilities
# ============================================================

def die(msg):
    print(
        f"[FAIL] {msg}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_tsv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        rows = list(reader)

        return (
            reader.fieldnames,
            rows,
        )


def write_tsv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_freeze(path):
    out = {}

    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        if "=" not in line:
            continue

        k, v = line.split(
            "=",
            1,
        )

        out[
            k.strip()
        ] = v.strip()

    return out


# ============================================================
# Feature cache resolver
# ============================================================

ACCESSION_NAMES = [
    "assembly_accession",
    "accession",
    "genome_accession",
    "accession_id",
]


def detect_accession_col(fields):

    for x in ACCESSION_NAMES:
        if x in fields:
            return x

    for f in fields:
        lf = f.lower()

        if (
            "accession" in lf
            and "paired" not in lf
        ):
            return f

    return None


def resolve_feature_matrix():

    candidates = []

    for p in sorted(
        FEATURE_ROOT.rglob("*.npy")
    ):

        try:
            x = np.load(
                p,
                mmap_mode="r",
            )

        except Exception:
            continue

        if (
            x.ndim == 2
            and tuple(
                x.shape
            ) == (
                3154,
                2080,
            )
        ):
            candidates.append(
                p
            )

    if len(
        candidates
    ) != 1:
        die(
            "Expected exactly one "
            "(3154,2080) feature matrix under "
            f"{FEATURE_ROOT}; found "
            f"{len(candidates)}:\n"
            + "\n".join(
                str(x)
                for x in candidates
            )
        )

    return candidates[0]


def resolve_feature_rows():

    candidates = []

    for p in sorted(
        FEATURE_ROOT.rglob("*.tsv")
    ):

        try:
            fields, rows = read_tsv(
                p
            )

        except Exception:
            continue

        if (
            not fields
            or len(rows) != 3154
        ):
            continue

        acc_col = detect_accession_col(
            fields
        )

        if acc_col is None:
            continue

        acc = [
            r[
                acc_col
            ].strip()
            for r in rows
        ]

        if (
            len(set(acc))
            != 3154
        ):
            continue

        candidates.append(
            (
                p,
                acc_col,
                acc,
            )
        )

    if not candidates:
        die(
            "Could not resolve 3154-row "
            "feature row table."
        )

    if len(
        candidates
    ) == 1:
        return candidates[0]

    # Prefer files explicitly describing row order.
    preferred = [
        x
        for x in candidates
        if (
            "row" in x[0].name.lower()
            or "index" in x[0].name.lower()
        )
    ]

    if len(
        preferred
    ) == 1:
        return preferred[0]

    # If several tables encode exactly the same accession order,
    # choosing one is harmless for matrix indexing.
    first_acc = candidates[0][2]

    same_order = all(
        x[2] == first_acc
        for x in candidates
    )

    if same_order:
        candidates = sorted(
            candidates,
            key=lambda x: (
                0
                if "row"
                in x[0].name.lower()
                else 1,
                len(
                    str(
                        x[0]
                    )
                ),
                str(
                    x[0]
                ),
            ),
        )

        return candidates[0]

    die(
        "Multiple incompatible 3154-row "
        "accession tables found:\n"
        + "\n".join(
            str(x[0])
            for x in candidates
        )
    )


# ============================================================
# Metrics
# ============================================================

def classification_metrics(
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
        s >= threshold
    ).astype(
        np.int64
    )

    tp = int(
        np.sum(
            (y == 1)
            & (pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            & (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            & (pred == 0)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            & (pred == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (
            tp + fn
        ) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (
            tn + fp
        ) > 0
        else 0.0
    )

    bacc = (
        recall
        + specificity
    ) / 2.0

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
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
        "MCC":
            float(
                mcc
            ),

        "F1":
            float(
                f1
            ),

        "BalancedAccuracy":
            float(
                bacc
            ),

        "Precision":
            float(
                precision
            ),

        "Recall":
            float(
                recall
            ),

        "Specificity":
            float(
                specificity
            ),

        "TP":
            tp,

        "FP":
            fp,

        "TN":
            tn,

        "FN":
            fn,

        "pred":
            pred,
    }


def full_metrics(
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

    cls = classification_metrics(
        y,
        s,
        threshold,
    )

    return {
        **rank,
        **cls,
    }


# ============================================================
# Mode-aware HGV
# ============================================================

class HGV(nn.Module):

    def __init__(
        self,
        normalize_output=True,
    ):
        super().__init__()

        self.normalize_output = (
            normalize_output
        )

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

    def forward(
        self,
        x,
    ):

        x = F.relu(
            self.bn1(
                self.fc1(
                    x
                )
            )
        )

        x = F.relu(
            self.bn2(
                self.fc2(
                    x
                )
            )
        )

        x = self.fc3(
            x
        )

        if self.normalize_output:

            x = F.normalize(
                x,
                p=2,
                dim=1,
            )

        return x


@torch.no_grad()
def embed_all(
    model,
    X,
    device,
    batch_size=1024,
):

    model.eval()

    chunks = []

    for start in range(
        0,
        len(X),
        batch_size,
    ):

        stop = min(
            start + batch_size,
            len(X),
        )

        xb = torch.as_tensor(
            np.asarray(
                X[
                    start:stop
                ],
                dtype=np.float32,
            ).copy(),
            dtype=torch.float32,
            device=device,
        )

        z = model(
            xb
        )

        chunks.append(
            z.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    return np.concatenate(
        chunks,
        axis=0,
    )


# ============================================================
# Original frozen D3 blind-test prediction resolver
# ============================================================

def resolve_original_d3_prediction_dir():

    root = (
        BASE
        / "08_main_generalization"
    )

    target = (
        "G2_split20260915_"
        "model20260915.tsv"
    )

    hits = [
        p
        for p in root.rglob(
            target
        )
        if (
            p.parent.name
            == "hgv_test_predictions"
        )
    ]

    if len(
        hits
    ) != 1:
        die(
            "Expected exactly one original "
            "main D3 prediction file for "
            "G2 split20260915 model20260915; "
            f"found {len(hits)}:\n"
            + "\n".join(
                str(x)
                for x in hits
            )
        )

    pred_dir = hits[0].parent

    for seed in sorted(
        MAIN_D3_SEEDS
    ):

        p = (
            pred_dir
            / (
                "G2_split20260915_"
                f"model{seed}.tsv"
            )
        )

        if not p.is_file():
            die(
                "Missing original D3 prediction: "
                f"{p}"
            )

    return pred_dir


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "HGV-V3 E6 CANONICAL G2 "
        "LOCKED TEST EVALUATION v1"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # PRE-TEST GATES
    # --------------------------------------------------------

    required = [
        MANIFEST,
        THRESHOLD_FILE,
        THRESHOLD_FREEZE,
        SOURCE_17C2,
        TEST_PAIRS,
    ]

    for p in required:

        if not p.is_file():
            die(
                f"Missing required file: {p}"
            )

    manifest_sha = sha256_file(
        MANIFEST
    )

    threshold_sha = sha256_file(
        THRESHOLD_FILE
    )

    source17c2_sha = sha256_file(
        SOURCE_17C2
    )

    test_sha = sha256_file(
        TEST_PAIRS
    )

    if (
        manifest_sha
        != EXPECTED_MANIFEST_SHA
    ):
        die(
            "Manifest SHA mismatch."
        )

    if (
        threshold_sha
        != EXPECTED_THRESHOLD_SHA
    ):
        die(
            "Frozen threshold SHA mismatch."
        )

    if (
        source17c2_sha
        != EXPECTED_17C2_SHA
    ):
        die(
            "Original 17c2 SHA mismatch."
        )

    if (
        test_sha
        != EXPECTED_TEST_SHA
    ):
        die(
            "Canonical G2 test SHA mismatch."
        )

    freeze08 = parse_freeze(
        THRESHOLD_FREEZE
    )

    if (
        freeze08.get(
            "E6_30CELL_VALIDATION_THRESHOLD_FREEZE"
        )
        != "PASS"
    ):
        die(
            "08 threshold freeze is not PASS."
        )

    if (
        freeze08.get(
            "POST_TEST_TUNING"
        )
        != "NO"
    ):
        die(
            "08 freeze post-test tuning gate failed."
        )

    if (
        freeze08.get(
            "FROZEN_THRESHOLD_SHA256"
        )
        != threshold_sha
    ):
        die(
            "08 freeze threshold SHA mismatch."
        )

    print(
        "[PASS] 07 manifest SHA"
    )

    print(
        "[PASS] 08 threshold SHA/freeze"
    )

    print(
        "[PASS] original 17c2 SHA"
    )

    print(
        "[PASS] canonical G2 test SHA"
    )

    # --------------------------------------------------------
    # Do not overwrite test results.
    # --------------------------------------------------------

    if (
        OUT.exists()
        and any(
            OUT.iterdir()
        )
    ):
        die(
            "Output directory already contains files:\n"
            f"{OUT}\n"
            "Do not overwrite a locked test evaluation."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resolve feature cache
    # --------------------------------------------------------

    feature_npy = (
        resolve_feature_matrix()
    )

    (
        feature_rows_tsv,
        feature_acc_col,
        feature_accessions,
    ) = resolve_feature_rows()

    print(
        f"[PASS] feature matrix: "
        f"{feature_npy}"
    )

    print(
        f"[PASS] feature rows: "
        f"{feature_rows_tsv}"
    )

    X = np.load(
        feature_npy,
        mmap_mode="r",
    )

    if (
        tuple(
            X.shape
        )
        != (
            3154,
            2080,
        )
    ):
        die(
            f"Feature shape invalid: {X.shape}"
        )

    acc_to_idx = {
        a:
            i
        for i, a
        in enumerate(
            feature_accessions
        )
    }

    # --------------------------------------------------------
    # Read the canonical G2 TEST
    #
    # TEST IS NOW OPENED.
    # No model/threshold choice may change beyond this point.
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "TEST OPENING"
    )

    print(
        "=" * 100
    )

    print(
        "TEST_DATA_READ=YES"
    )

    test_fields, test_rows = read_tsv(
        TEST_PAIRS
    )

    if len(
        test_rows
    ) != EXPECTED_TEST_PAIRS:
        die(
            "Canonical G2 test pair count mismatch: "
            f"{len(test_rows)} "
            f"!= {EXPECTED_TEST_PAIRS}"
        )

    required_test_cols = {
        "accession_a",
        "accession_b",
        "label",
        "pair_type",
    }

    if not required_test_cols.issubset(
        set(
            test_fields
        )
    ):
        die(
            f"Unexpected test schema: "
            f"{test_fields}"
        )

    accession_a = [
        r[
            "accession_a"
        ].strip()
        for r in test_rows
    ]

    accession_b = [
        r[
            "accession_b"
        ].strip()
        for r in test_rows
    ]

    labels = np.asarray(
        [
            int(
                r[
                    "label"
                ]
            )
            for r in test_rows
        ],
        dtype=np.int64,
    )

    pair_types = [
        r[
            "pair_type"
        ]
        for r in test_rows
    ]

    missing = sorted(
        (
            set(
                accession_a
            )
            | set(
                accession_b
            )
        )
        - set(
            acc_to_idx
        )
    )

    if missing:
        die(
            "Test accessions missing from feature "
            f"matrix: n={len(missing)}"
        )

    a_idx = np.asarray(
        [
            acc_to_idx[
                a
            ]
            for a in accession_a
        ],
        dtype=np.int64,
    )

    b_idx = np.asarray(
        [
            acc_to_idx[
                b
            ]
            for b in accession_b
        ],
        dtype=np.int64,
    )

    print(
        f"[PASS] test pairs = "
        f"{len(test_rows)}"
    )

    print(
        f"[INFO] positive = "
        f"{int(np.sum(labels == 1))}"
    )

    print(
        f"[INFO] negative = "
        f"{int(np.sum(labels == 0))}"
    )

    # --------------------------------------------------------
    # Read 30-cell manifest and frozen thresholds
    # --------------------------------------------------------

    manifest_fields, manifest_rows = read_tsv(
        MANIFEST
    )

    if len(
        manifest_rows
    ) != 30:
        die(
            "Manifest is not 30 cells."
        )

    threshold_fields, threshold_rows = read_tsv(
        THRESHOLD_FILE
    )

    if len(
        threshold_rows
    ) != 30:
        die(
            "Threshold freeze is not 30 cells."
        )

    threshold_lookup = {}

    for r in threshold_rows:

        key = (
            r[
                "mode"
            ],
            int(
                r[
                    "model_seed"
                ]
            ),
        )

        if key in threshold_lookup:
            die(
                f"Duplicate threshold key: {key}"
            )

        threshold_lookup[
            key
        ] = float(
            r[
                "threshold"
            ]
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] device={device}"
    )

    if (
        device.type
        == "cuda"
    ):
        print(
            f"[INFO] GPU="
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Original D3 frozen blind predictions
    # --------------------------------------------------------

    old_d3_pred_dir = (
        resolve_original_d3_prediction_dir()
    )

    print(
        "[PASS] original D3 blind-test "
        f"predictions: {old_d3_pred_dir}"
    )

    # --------------------------------------------------------
    # Evaluate all 30 cells
    # --------------------------------------------------------

    metrics_rows = []
    norm_rows = []
    parity_rows = []

    all_combined_rows = []

    print()
    print(
        "=" * 100
    )

    print(
        "EVALUATE 30 FROZEN CELLS"
    )

    print(
        "=" * 100
    )

    for n, row in enumerate(
        manifest_rows,
        start=1,
    ):

        cell_id = row[
            "cell_id"
        ]

        mode = row[
            "mode"
        ]

        split_seed = int(
            row[
                "split_seed"
            ]
        )

        model_seed = int(
            row[
                "model_seed"
            ]
        )

        source_block = row[
            "source_block"
        ]

        if split_seed != 20260915:
            die(
                f"{cell_id}: split seed is not canonical G2 seed15."
            )

        key = (
            mode,
            model_seed,
        )

        if key not in threshold_lookup:
            die(
                f"{cell_id}: missing frozen threshold."
            )

        threshold = threshold_lookup[
            key
        ]

        # Checkpoint is in the same frozen run directory
        # as val_predictions.tsv.
        val_pred = Path(
            row[
                "val_predictions_tsv"
            ]
        )

        run_dir = (
            val_pred.parent
        )

        checkpoint_path = (
            run_dir
            / "best_model.pt"
        )

        if not checkpoint_path.is_file():
            die(
                f"{cell_id}: missing checkpoint "
                f"{checkpoint_path}"
            )

        checkpoint_sha = sha256_file(
            checkpoint_path
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        normalize_output = (
            mode != "D5"
        )

        model = HGV(
            normalize_output=
                normalize_output
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        Z = embed_all(
            model,
            X,
            device,
        )

        norms = np.linalg.norm(
            Z.astype(
                np.float64
            ),
            axis=1,
        )

        d = np.linalg.norm(
            Z[
                a_idx
            ].astype(
                np.float64
            )
            - Z[
                b_idx
            ].astype(
                np.float64
            ),
            axis=1,
        )

        scores = (
            1.0
            - d / 2.0
        )

        m = full_metrics(
            labels,
            scores,
            threshold,
        )

        pred = m[
            "pred"
        ]

        pred_rows = []

        for i in range(
            len(
                labels
            )
        ):

            out_row = {
                "cell_id":
                    cell_id,

                "mode":
                    mode,

                "model_seed":
                    model_seed,

                "accession_a":
                    accession_a[
                        i
                    ],

                "accession_b":
                    accession_b[
                        i
                    ],

                "label":
                    int(
                        labels[
                            i
                        ]
                    ),

                "pair_type":
                    pair_types[
                        i
                    ],

                "distance":
                    repr(
                        float(
                            d[
                                i
                            ]
                        )
                    ),

                "similarity_score":
                    repr(
                        float(
                            scores[
                                i
                            ]
                        )
                    ),

                "frozen_threshold":
                    repr(
                        float(
                            threshold
                        )
                    ),

                "predicted_label":
                    int(
                        pred[
                            i
                        ]
                    ),
            }

            pred_rows.append(
                out_row
            )

            all_combined_rows.append(
                dict(
                    out_row
                )
            )

        pred_path = (
            PRED_DIR
            / f"{cell_id}.tsv"
        )

        pred_fields = [
            "cell_id",
            "mode",
            "model_seed",
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
            "distance",
            "similarity_score",
            "frozen_threshold",
            "predicted_label",
        ]

        write_tsv(
            pred_path,
            pred_rows,
            pred_fields,
        )

        metrics_rows.append({
            "cell_id":
                cell_id,

            "mode":
                mode,

            "protocol":
                "G2",

            "split_seed":
                20260915,

            "model_seed":
                model_seed,

            "source_block":
                source_block,

            "n_pairs":
                len(
                    labels
                ),

            "n_positive":
                int(
                    np.sum(
                        labels == 1
                    )
                ),

            "n_negative":
                int(
                    np.sum(
                        labels == 0
                    )
                ),

            "threshold":
                repr(
                    threshold
                ),

            "AUROC":
                repr(
                    m[
                        "AUROC"
                    ]
                ),

            "AUPRC":
                repr(
                    m[
                        "AUPRC"
                    ]
                ),

            "MCC":
                repr(
                    m[
                        "MCC"
                    ]
                ),

            "F1":
                repr(
                    m[
                        "F1"
                    ]
                ),

            "BalancedAccuracy":
                repr(
                    m[
                        "BalancedAccuracy"
                    ]
                ),

            "Precision":
                repr(
                    m[
                        "Precision"
                    ]
                ),

            "Recall":
                repr(
                    m[
                        "Recall"
                    ]
                ),

            "Specificity":
                repr(
                    m[
                        "Specificity"
                    ]
                ),

            "TP":
                m[
                    "TP"
                ],

            "FP":
                m[
                    "FP"
                ],

            "TN":
                m[
                    "TN"
                ],

            "FN":
                m[
                    "FN"
                ],

            "checkpoint_sha256":
                checkpoint_sha,

            "prediction_sha256":
                sha256_file(
                    pred_path
                ),
        })

        norm_rows.append({
            "cell_id":
                cell_id,

            "mode":
                mode,

            "model_seed":
                model_seed,

            "output_normalization":
                (
                    "none"
                    if mode == "D5"
                    else "L2"
                ),

            "embedding_norm_mean":
                repr(
                    float(
                        np.mean(
                            norms
                        )
                    )
                ),

            "embedding_norm_sd":
                repr(
                    float(
                        np.std(
                            norms,
                            ddof=1,
                        )
                    )
                ),

            "embedding_norm_min":
                repr(
                    float(
                        np.min(
                            norms
                        )
                    )
                ),

            "embedding_norm_max":
                repr(
                    float(
                        np.max(
                            norms
                        )
                    )
                ),
        })

        print(
            f"[{n:02d}/30] "
            f"{cell_id:<18} "
            f"AUC={m['AUROC']:.6f} "
            f"AUPRC={m['AUPRC']:.6f} "
            f"MCC={m['MCC']:.6f} "
            f"F1={m['F1']:.6f} "
            f"BAcc={m['BalancedAccuracy']:.6f}"
        )

        # ----------------------------------------------------
        # Frozen-main D3 parity
        # ----------------------------------------------------

        if (
            mode == "D3"
            and model_seed
            in MAIN_D3_SEEDS
        ):

            old_path = (
                old_d3_pred_dir
                / (
                    "G2_split20260915_"
                    f"model{model_seed}.tsv"
                )
            )

            old_fields, old_rows = read_tsv(
                old_path
            )

            if len(
                old_rows
            ) != len(
                pred_rows
            ):
                die(
                    f"D3 seed {model_seed}: "
                    "old/new prediction row count mismatch."
                )

            pair_match = True
            label_match = True
            pred_match = True

            old_distance = []
            old_score = []

            for i, old in enumerate(
                old_rows
            ):

                new = pred_rows[
                    i
                ]

                if (
                    old[
                        "accession_a"
                    ]
                    != new[
                        "accession_a"
                    ]
                    or old[
                        "accession_b"
                    ]
                    != new[
                        "accession_b"
                    ]
                    or old[
                        "pair_type"
                    ]
                    != new[
                        "pair_type"
                    ]
                ):
                    pair_match = False

                if (
                    int(
                        old[
                            "label"
                        ]
                    )
                    != int(
                        new[
                            "label"
                        ]
                    )
                ):
                    label_match = False

                if (
                    int(
                        old[
                            "predicted_label"
                        ]
                    )
                    != int(
                        new[
                            "predicted_label"
                        ]
                    )
                ):
                    pred_match = False

                old_distance.append(
                    float(
                        old[
                            "distance"
                        ]
                    )
                )

                old_score.append(
                    float(
                        old[
                            "similarity_score"
                        ]
                    )
                )

            old_distance = np.asarray(
                old_distance,
                dtype=np.float64,
            )

            old_score = np.asarray(
                old_score,
                dtype=np.float64,
            )

            max_distance_delta = float(
                np.max(
                    np.abs(
                        old_distance
                        - d
                    )
                )
            )

            max_score_delta = float(
                np.max(
                    np.abs(
                        old_score
                        - scores
                    )
                )
            )

            numeric_match = (
                max_distance_delta
                <= 1e-6
                and max_score_delta
                <= 1e-6
            )

            parity_status = (
                "PASS"
                if (
                    pair_match
                    and label_match
                    and pred_match
                    and numeric_match
                )
                else "FAIL"
            )

            parity_rows.append({
                "mode":
                    "D3",

                "model_seed":
                    model_seed,

                "pair_order_match":
                    str(
                        pair_match
                    ),

                "label_match":
                    str(
                        label_match
                    ),

                "predicted_label_match":
                    str(
                        pred_match
                    ),

                "max_abs_distance_delta":
                    repr(
                        max_distance_delta
                    ),

                "max_abs_score_delta":
                    repr(
                        max_score_delta
                    ),

                "numeric_tolerance":
                    "1e-6",

                "original_prediction":
                    str(
                        old_path
                    ),

                "status":
                    parity_status,
            })

        del Z
        del model
        del checkpoint

        if (
            device.type
            == "cuda"
        ):
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # D3 parity hard gate
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "D3 FROZEN-MAIN TEST PARITY"
    )

    print(
        "=" * 100
    )

    if len(
        parity_rows
    ) != 3:
        die(
            "Expected 3 frozen-main D3 parity rows."
        )

    for r in parity_rows:

        print(
            "D3 seed="
            f"{r['model_seed']} "
            f"pair={r['pair_order_match']} "
            f"label={r['label_match']} "
            f"pred={r['predicted_label_match']} "
            f"max_score_delta="
            f"{float(r['max_abs_score_delta']):.3e} "
            f"{r['status']}"
        )

    write_tsv(
        PARITY_TSV,
        parity_rows,
        [
            "mode",
            "model_seed",
            "pair_order_match",
            "label_match",
            "predicted_label_match",
            "max_abs_distance_delta",
            "max_abs_score_delta",
            "numeric_tolerance",
            "original_prediction",
            "status",
        ],
    )

    if any(
        r[
            "status"
        ] != "PASS"
        for r in parity_rows
    ):
        die(
            "D3 frozen-main TEST parity failed. "
            "Do not proceed to E6 mechanism/statistics."
        )

    print(
        "D3_FROZEN_MAIN_TEST_PARITY=PASS"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metric_fields = [
        "cell_id",
        "mode",
        "protocol",
        "split_seed",
        "model_seed",
        "source_block",
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
        "checkpoint_sha256",
        "prediction_sha256",
    ]

    write_tsv(
        METRICS_TSV,
        metrics_rows,
        metric_fields,
    )

    write_tsv(
        NORM_TSV,
        norm_rows,
        [
            "cell_id",
            "mode",
            "model_seed",
            "output_normalization",
            "embedding_norm_mean",
            "embedding_norm_sd",
            "embedding_norm_min",
            "embedding_norm_max",
        ],
    )

    combined_path = (
        OUT
        / "E6_G2_TEST_PAIR_PREDICTIONS_LONG.tsv"
    )

    write_tsv(
        combined_path,
        all_combined_rows,
        [
            "cell_id",
            "mode",
            "model_seed",
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

    # --------------------------------------------------------
    # Mode-level descriptive summary
    # 5 model seeds = initialization variability.
    # Formal biological CI comes later in step 11.
    # --------------------------------------------------------

    summary_rows = []

    metric_names = [
        "AUROC",
        "AUPRC",
        "MCC",
        "F1",
        "BalancedAccuracy",
    ]

    for mode in [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]:

        rs = [
            r
            for r in metrics_rows
            if r[
                "mode"
            ] == mode
        ]

        if len(
            rs
        ) != 5:
            die(
                f"{mode}: expected 5 test runs."
            )

        outrow = {
            "mode":
                mode,

            "n_model_seeds":
                5,

            "variability_unit":
                "model_initialization_seed",
        }

        for metric in metric_names:

            vals = np.asarray(
                [
                    float(
                        r[
                            metric
                        ]
                    )
                    for r in rs
                ],
                dtype=np.float64,
            )

            outrow[
                metric
                + "_mean"
            ] = repr(
                float(
                    np.mean(
                        vals
                    )
                )
            )

            outrow[
                metric
                + "_sd_model_seed"
            ] = repr(
                float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                )
            )

        summary_rows.append(
            outrow
        )

    summary_fields = [
        "mode",
        "n_model_seeds",
        "variability_unit",
    ]

    for metric in metric_names:
        summary_fields.extend([
            metric
            + "_mean",

            metric
            + "_sd_model_seed",
        ])

    write_tsv(
        SUMMARY_TSV,
        summary_rows,
        summary_fields,
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = {
        "status":
            "PASS",

        "stage":
            "E6 canonical G2 locked test",

        "protocol":
            "G2",

        "split_seed":
            20260915,

        "test_pairs":
            str(
                TEST_PAIRS
            ),

        "test_pairs_sha256":
            test_sha,

        "test_pair_count":
            len(
                test_rows
            ),

        "manifest_sha256":
            manifest_sha,

        "threshold_sha256":
            threshold_sha,

        "original_17c2_sha256":
            source17c2_sha,

        "feature_matrix":
            str(
                feature_npy
            ),

        "feature_matrix_sha256":
            sha256_file(
                feature_npy
            ),

        "feature_rows":
            str(
                feature_rows_tsv
            ),

        "feature_rows_sha256":
            sha256_file(
                feature_rows_tsv
            ),

        "cells":
            30,

        "score_definition":
            "1 - euclidean_distance / 2",

        "decision_rule":
            "similarity_score >= frozen validation threshold",

        "output_normalization":
            {
                "D0":
                    "L2",
                "D1":
                    "L2",
                "D2":
                    "L2",
                "D3":
                    "L2",
                "D4":
                    "L2",
                "D5":
                    "none",
            },

        "d3_frozen_main_test_parity":
            "PASS",

        "test_used":
            True,

        "external_used":
            False,

        "threshold_reselected":
            False,

        "retraining":
            False,

        "post_test_tuning":
            False,

        "device":
            str(
                device
            ),

        "torch_version":
            torch.__version__,

        "numpy_version":
            np.__version__,

        "platform":
            platform.platform(),
    }

    PROVENANCE_JSON.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    lines = [
        "=" * 80,
        "HGV-V3 E6 CANONICAL G2 TEST REPORT",
        "=" * 80,
        "",
        "STATUS: PASS",
        "",
        "Protocol: G2",
        "Split seed: 20260915",
        f"Test pairs: {len(test_rows)}",
        "Cells evaluated: 30",
        "",
        "Test used: YES",
        "External used: NO",
        "Threshold reselection: NO",
        "Retraining: NO",
        "Post-test tuning: NO",
        "",
        "Score definition:",
        "  similarity_score = 1 - EuclideanDistance / 2",
        "",
        "Output normalization:",
        "  D0-D4 = L2",
        "  D5    = none",
        "",
        "D3 frozen-main test parity:",
        "  seeds 20260915/16/17 = PASS",
        "",
        "Mode summaries:",
    ]

    for r in summary_rows:

        lines.append(
            f"  {r['mode']}: "
            f"AUROC="
            f"{float(r['AUROC_mean']):.6f}"
            f" +/- "
            f"{float(r['AUROC_sd_model_seed']):.6f}; "
            f"AUPRC="
            f"{float(r['AUPRC_mean']):.6f}"
            f" +/- "
            f"{float(r['AUPRC_sd_model_seed']):.6f}; "
            f"MCC="
            f"{float(r['MCC_mean']):.6f}"
            f" +/- "
            f"{float(r['MCC_sd_model_seed']):.6f}"
        )

    lines += [
        "",
        "NOTE:",
        "  The five model seeds represent initialization variability.",
        "  They are not five independent biological replicates.",
        "  Species-block bootstrap/statistical contrasts are performed later.",
        "",
        "E6_CANONICAL_G2_TEST=PASS",
        "",
    ]

    REPORT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SHA
    # --------------------------------------------------------

    output_files = sorted([
        p
        for p in OUT.rglob(
            "*"
        )
        if (
            p.is_file()
            and p
            != SHA_FILE
            and p
            != FREEZE_FILE
        )
    ])

    sha_lines = []

    for p in output_files:

        sha_lines.append(
            f"{sha256_file(p)}  "
            f"{p.relative_to(OUT)}"
        )

    SHA_FILE.write_text(
        "\n".join(
            sha_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final freeze
    # --------------------------------------------------------

    freeze_lines = [
        "HGV-V3 E6 CANONICAL G2 TEST FREEZE",
        "VERSION=v1",
        "DATE=2026-09-24",
        "STATUS=PASS",
        "PROTOCOL=G2",
        "SPLIT_SEED=20260915",
        "CELLS=30",
        f"TEST_PAIRS={len(test_rows)}",
        f"TEST_SHA256={test_sha}",
        f"MANIFEST_SHA256={manifest_sha}",
        f"THRESHOLD_SHA256={threshold_sha}",
        f"ORIGINAL_17C2_SHA256={source17c2_sha}",
        "D0_OUTPUT_NORMALIZATION=L2",
        "D1_OUTPUT_NORMALIZATION=L2",
        "D2_OUTPUT_NORMALIZATION=L2",
        "D3_OUTPUT_NORMALIZATION=L2",
        "D4_OUTPUT_NORMALIZATION=L2",
        "D5_OUTPUT_NORMALIZATION=NONE",
        "SCORE=1-EUCLIDEAN_DISTANCE/2",
        "D3_FROZEN_MAIN_TEST_PARITY=PASS",
        "TEST_DATA_READ=YES",
        "EXTERNAL_DATA_READ=NO",
        "THRESHOLD_RESELECTION=NO",
        "RETRAINING=NO",
        "POST_TEST_TUNING=NO",
        "E6_CANONICAL_G2_TEST=PASS",
    ]

    FREEZE_FILE.write_text(
        "\n".join(
            freeze_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL"
    )

    print(
        "=" * 100
    )

    for r in summary_rows:

        print(
            f"{r['mode']} "
            f"AUROC="
            f"{float(r['AUROC_mean']):.6f}"
            f" +/- "
            f"{float(r['AUROC_sd_model_seed']):.6f} "
            f"AUPRC="
            f"{float(r['AUPRC_mean']):.6f}"
            f" +/- "
            f"{float(r['AUPRC_sd_model_seed']):.6f} "
            f"MCC="
            f"{float(r['MCC_mean']):.6f}"
            f" +/- "
            f"{float(r['MCC_sd_model_seed']):.6f}"
        )

    print()
    print(
        "D3_FROZEN_MAIN_TEST_PARITY=PASS"
    )

    print(
        "THRESHOLD_RESELECTION=NO"
    )

    print(
        "POST_TEST_TUNING=NO"
    )

    print(
        f"Output={OUT}"
    )

    print()
    print(
        "E6_CANONICAL_G2_TEST=PASS"
    )


if __name__ == "__main__":
    main()
