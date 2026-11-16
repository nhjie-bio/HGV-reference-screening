#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# HGV-V3 E6
# 30-cell VALIDATION-ONLY threshold freeze
#
# Exact threshold semantics copied from frozen 17c1:
#
# Candidate thresholds:
#   1. nextafter(max_score, +inf)
#   2. midpoint between adjacent unique score groups
#   3. nextafter(min_score, -inf)
#
# Selection:
#   1. max MCC
#   2. max Balanced Accuracy
#   3. max F1
#   4. closest threshold to 0.5
#   5. lower numerical threshold
#
# IMPORTANT:
# - validation predictions ONLY
# - NO test read
# - NO external read
# - NO retraining
# - NO overwrite
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

MANIFEST_FREEZE = (
    MANIFEST_DIR
    / "E6_30CELL_MANIFEST_FREEZE_v1.txt"
)

ORIGINAL_17C1 = (
    BASE
    / "scripts_v3_20260916_hgv"
    / "17c1_freeze_validation_thresholds_v1.py"
)

ORIGINAL_MAIN_THRESHOLDS = (
    BASE
    / "08_main_generalization"
    / "validation_threshold_freeze_v1_20260917"
    / "FROZEN_VALIDATION_THRESHOLDS.tsv"
)

OUTDIR = (
    E6ROOT
    / "05_validation_threshold_freeze_v1_20260924"
)

FROZEN_TSV = (
    OUTDIR
    / "E6_FROZEN_VALIDATION_THRESHOLDS.tsv"
)

PARITY_TSV = (
    OUTDIR
    / "E6_D3_MAIN_THRESHOLD_PARITY.tsv"
)

AUDIT_JSON = (
    OUTDIR
    / "E6_VALIDATION_THRESHOLD_FREEZE_AUDIT.json"
)

REPORT = (
    OUTDIR
    / "E6_VALIDATION_THRESHOLD_FREEZE_REPORT.txt"
)

SHA_FILE = (
    OUTDIR
    / "E6_VALIDATION_THRESHOLD_SHA256.txt"
)

FREEZE_FILE = (
    OUTDIR
    / "E6_VALIDATION_THRESHOLD_FREEZE_v1.txt"
)

EXPECTED_MANIFEST_SHA256 = (
    "dde063cbc4697aeec43b80480223509427a37c0afafa182cac1008aaf268fd26"
)

EXPECTED_17C1_SHA256 = (
    "2ded4d5af7f33f6cf2f86c50ac57b7971271256b6267e92a962034bf1337d052"
)

EXPECTED_CELLS = 30
EXPECTED_VAL_PAIRS = 3874

D3_MAIN_SEEDS = {
    20260915,
    20260916,
    20260917,
}


# ============================================================
# Utilities
# ============================================================

def die(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
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
        return list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )


def write_tsv(path, rows, fieldnames):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        w.writeheader()
        w.writerows(rows)


def parse_freeze_file(path):
    out = {}

    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()

    return out


# ============================================================
# Exact confusion metrics
# ============================================================

def confusion_metrics(
    tp,
    fp,
    tn,
    fn,
):
    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)

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
        "MCC": float(mcc),
        "F1": float(f1),
        "BalancedAccuracy": float(balanced),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
    }


# ============================================================
# Exact frozen 17c1 threshold selection
# ============================================================

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

    if len(y) == 0:
        raise RuntimeError(
            "Empty validation set."
        )

    if len(y) != len(s):
        raise RuntimeError(
            "Label/score length mismatch."
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

    # Frozen source used descending stable mergesort.
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

    # --------------------------------------------------------
    # Candidate above maximum:
    # predicts everything negative.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Move downward one unique score group at a time.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Frozen tie-breaking:
    #
    # 1 MCC
    # 2 Balanced Accuracy
    # 3 F1
    # 4 closest to 0.5
    # 5 lower threshold
    # --------------------------------------------------------

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

    best = dict(best)
    best["candidate_count"] = len(
        candidates
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


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print(
        "HGV-V3 E6 30-CELL VALIDATION THRESHOLD FREEZE v1"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Gate 1: source files
    # --------------------------------------------------------

    required = [
        MANIFEST,
        MANIFEST_FREEZE,
        ORIGINAL_17C1,
        ORIGINAL_MAIN_THRESHOLDS,
    ]

    for p in required:
        if not p.is_file():
            die(
                f"Missing required file: {p}"
            )

    # --------------------------------------------------------
    # Gate 2: 07 manifest integrity
    # --------------------------------------------------------

    actual_manifest_sha = sha256_file(
        MANIFEST
    )

    if (
        actual_manifest_sha
        != EXPECTED_MANIFEST_SHA256
    ):
        die(
            "07 manifest SHA mismatch:\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"actual  ={actual_manifest_sha}"
        )

    freeze07 = parse_freeze_file(
        MANIFEST_FREEZE
    )

    if (
        freeze07.get(
            "E6_30CELL_MANIFEST"
        )
        != "PASS"
    ):
        die(
            "07 manifest freeze does not say PASS."
        )

    if (
        freeze07.get(
            "TEST_DATA_READ"
        )
        != "NO"
    ):
        die(
            "07 freeze TEST_DATA_READ is not NO."
        )

    if (
        freeze07.get(
            "MANIFEST_SHA256"
        )
        != actual_manifest_sha
    ):
        die(
            "07 freeze manifest SHA does not "
            "match current manifest."
        )

    print(
        "[PASS] 07 canonical manifest integrity"
    )
    print(
        f"[INFO] manifest_sha256={actual_manifest_sha}"
    )

    # --------------------------------------------------------
    # Gate 3: frozen 17c1 source integrity
    # --------------------------------------------------------

    actual_17c1_sha = sha256_file(
        ORIGINAL_17C1
    )

    if (
        actual_17c1_sha
        != EXPECTED_17C1_SHA256
    ):
        die(
            "Original 17c1 SHA mismatch:\n"
            f"expected={EXPECTED_17C1_SHA256}\n"
            f"actual  ={actual_17c1_sha}"
        )

    print(
        "[PASS] original 17c1 source SHA"
    )
    print(
        f"[INFO] 17c1_sha256={actual_17c1_sha}"
    )

    # --------------------------------------------------------
    # Do not overwrite.
    # --------------------------------------------------------

    if (
        OUTDIR.exists()
        and any(
            OUTDIR.iterdir()
        )
    ):
        die(
            "Output directory already contains files:\n"
            f"{OUTDIR}\n"
            "Do not overwrite a threshold freeze."
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read canonical 30 cells
    # --------------------------------------------------------

    manifest_rows = read_tsv(
        MANIFEST
    )

    if len(
        manifest_rows
    ) != EXPECTED_CELLS:
        die(
            "Manifest cell count mismatch: "
            f"{len(manifest_rows)} "
            f"!= {EXPECTED_CELLS}"
        )

    expected_modes = {
        "D0": 5,
        "D1": 5,
        "D2": 5,
        "D3": 5,
        "D4": 5,
        "D5": 5,
    }

    mode_counts = {
        mode: sum(
            r["mode"] == mode
            for r in manifest_rows
        )
        for mode in expected_modes
    }

    if (
        mode_counts
        != expected_modes
    ):
        die(
            f"Mode counts invalid: {mode_counts}"
        )

    # --------------------------------------------------------
    # Freeze thresholds
    # --------------------------------------------------------

    threshold_rows = []

    for idx, row in enumerate(
        manifest_rows,
        start=1,
    ):

        cell_id = row["cell_id"]
        mode = row["mode"]

        split_seed = int(
            row["split_seed"]
        )

        model_seed = int(
            row["model_seed"]
        )

        if split_seed != 20260915:
            die(
                f"{cell_id}: unexpected split seed "
                f"{split_seed}"
            )

        pred_path = Path(
            row[
                "val_predictions_tsv"
            ]
        )

        if not pred_path.is_file():
            die(
                f"{cell_id}: missing validation file: "
                f"{pred_path}"
            )

        pred_sha = sha256_file(
            pred_path
        )

        if (
            pred_sha
            != row[
                "val_predictions_sha256"
            ]
        ):
            die(
                f"{cell_id}: validation SHA mismatch."
            )

        preds = read_tsv(
            pred_path
        )

        if len(
            preds
        ) != EXPECTED_VAL_PAIRS:
            die(
                f"{cell_id}: expected "
                f"{EXPECTED_VAL_PAIRS} validation pairs, "
                f"found {len(preds)}"
            )

        if not preds:
            die(
                f"{cell_id}: empty validation predictions."
            )

        required_cols = {
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
            "distance",
            "similarity_score",
        }

        observed_cols = set(
            preds[0].keys()
        )

        if not required_cols.issubset(
            observed_cols
        ):
            die(
                f"{cell_id}: invalid validation schema: "
                f"{sorted(observed_cols)}"
            )

        y = np.asarray(
            [
                int(
                    r["label"]
                )
                for r in preds
            ],
            dtype=np.int64,
        )

        s = np.asarray(
            [
                float(
                    r[
                        "similarity_score"
                    ]
                )
                for r in preds
            ],
            dtype=np.float64,
        )

        frozen = freeze_threshold(
            y,
            s,
        )

        rank = ranking_metrics(
            y,
            s,
        )

        threshold_rows.append({
            "cell_id":
                cell_id,

            "mode":
                mode,

            "protocol":
                "G2",

            "split_seed":
                split_seed,

            "model_seed":
                model_seed,

            "source_block":
                row[
                    "source_block"
                ],

            "threshold":
                repr(
                    frozen[
                        "threshold"
                    ]
                ),

            "val_AUROC":
                repr(
                    rank[
                        "AUROC"
                    ]
                ),

            "val_AUPRC":
                repr(
                    rank[
                        "AUPRC"
                    ]
                ),

            "val_MCC":
                repr(
                    frozen[
                        "MCC"
                    ]
                ),

            "val_F1":
                repr(
                    frozen[
                        "F1"
                    ]
                ),

            "val_BalancedAccuracy":
                repr(
                    frozen[
                        "BalancedAccuracy"
                    ]
                ),

            "val_Precision":
                repr(
                    frozen[
                        "Precision"
                    ]
                ),

            "val_Recall":
                repr(
                    frozen[
                        "Recall"
                    ]
                ),

            "val_Specificity":
                repr(
                    frozen[
                        "Specificity"
                    ]
                ),

            "n_val_pairs":
                len(preds),

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

            "candidate_thresholds":
                frozen[
                    "candidate_count"
                ],

            "score_column":
                "similarity_score",

            "decision_rule":
                "similarity_score>=threshold",

            "validation_predictions_tsv":
                str(
                    pred_path.resolve()
                ),

            "source_sha256":
                pred_sha,
        })

        print(
            f"[{idx:02d}/30] "
            f"{cell_id:<18} "
            f"thr={frozen['threshold']:.12f} "
            f"MCC={frozen['MCC']:.6f} "
            f"BAcc="
            f"{frozen['BalancedAccuracy']:.6f} "
            f"F1={frozen['F1']:.6f}"
        )

    # --------------------------------------------------------
    # 30-cell structural gate
    # --------------------------------------------------------

    if len(
        threshold_rows
    ) != 30:
        die(
            f"Threshold row count={len(threshold_rows)} != 30"
        )

    keys = [
        (
            r["mode"],
            int(
                r["model_seed"]
            ),
        )
        for r in threshold_rows
    ]

    if len(
        set(keys)
    ) != 30:
        die(
            "Duplicate mode/model_seed thresholds."
        )

    # ========================================================
    # D3 frozen-main threshold parity
    # ========================================================

    print()
    print("=" * 100)
    print(
        "D3 FROZEN-MAIN THRESHOLD PARITY"
    )
    print("=" * 100)

    original_rows = read_tsv(
        ORIGINAL_MAIN_THRESHOLDS
    )

    original_lookup = {}

    for r in original_rows:

        if (
            r.get(
                "method_type",
                "",
            )
            != "HGV"
        ):
            continue

        if (
            r.get(
                "method",
                ""
            )
            != "HGV_Dashing2_tau0p98"
        ):
            continue

        if (
            r.get(
                "protocol",
                ""
            )
            != "G2"
        ):
            continue

        try:
            split_seed = int(
                r[
                    "split_seed"
                ]
            )

            model_seed = int(
                r[
                    "model_seed"
                ]
            )

        except Exception:
            continue

        if split_seed != 20260915:
            continue

        original_lookup[
            model_seed
        ] = r

    parity_rows = []

    parity_pass = True

    for model_seed in sorted(
        D3_MAIN_SEEDS
    ):

        e6_matches = [
            r
            for r in threshold_rows
            if (
                r["mode"] == "D3"
                and int(
                    r[
                        "model_seed"
                    ]
                )
                == model_seed
            )
        ]

        if len(
            e6_matches
        ) != 1:
            die(
                f"D3 seed {model_seed}: "
                "canonical threshold row not unique."
            )

        if (
            model_seed
            not in original_lookup
        ):
            die(
                f"D3 seed {model_seed}: "
                "missing original main threshold."
            )

        e6_row = e6_matches[0]
        old_row = original_lookup[
            model_seed
        ]

        e6_thr = float(
            e6_row[
                "threshold"
            ]
        )

        old_thr = float(
            old_row[
                "threshold"
            ]
        )

        delta = abs(
            e6_thr - old_thr
        )

        threshold_ok = (
            delta <= 1e-15
        )

        source_sha_old = (
            old_row.get(
                "source_sha256",
                ""
            )
        )

        source_sha_new = (
            e6_row[
                "source_sha256"
            ]
        )

        source_sha_ok = (
            source_sha_old
            == source_sha_new
        )

        status = (
            "PASS"
            if (
                threshold_ok
                and source_sha_ok
            )
            else "FAIL"
        )

        if status != "PASS":
            parity_pass = False

        parity_rows.append({
            "mode":
                "D3",

            "protocol":
                "G2",

            "split_seed":
                20260915,

            "model_seed":
                model_seed,

            "e6_threshold":
                repr(
                    e6_thr
                ),

            "original_main_threshold":
                repr(
                    old_thr
                ),

            "abs_delta":
                repr(
                    delta
                ),

            "threshold_match":
                str(
                    threshold_ok
                ),

            "source_sha_match":
                str(
                    source_sha_ok
                ),

            "status":
                status,
        })

        print(
            f"D3 seed={model_seed} "
            f"new={e6_thr:.17g} "
            f"old={old_thr:.17g} "
            f"delta={delta:.3e} "
            f"threshold={threshold_ok} "
            f"sourceSHA={source_sha_ok} "
            f"{status}"
        )

    if not parity_pass:
        # Write the diagnostic before stopping.
        write_tsv(
            PARITY_TSV,
            parity_rows,
            [
                "mode",
                "protocol",
                "split_seed",
                "model_seed",
                "e6_threshold",
                "original_main_threshold",
                "abs_delta",
                "threshold_match",
                "source_sha_match",
                "status",
            ],
        )

        die(
            "Frozen-main D3 threshold parity FAILED. "
            "Do not proceed to E6 test."
        )

    print(
        "D3_FROZEN_MAIN_THRESHOLD_PARITY=PASS"
    )

    # ========================================================
    # Write frozen threshold table
    # ========================================================

    threshold_fields = [
        "cell_id",
        "mode",
        "protocol",
        "split_seed",
        "model_seed",
        "source_block",
        "threshold",
        "val_AUROC",
        "val_AUPRC",
        "val_MCC",
        "val_F1",
        "val_BalancedAccuracy",
        "val_Precision",
        "val_Recall",
        "val_Specificity",
        "n_val_pairs",
        "n_positive",
        "n_negative",
        "candidate_thresholds",
        "score_column",
        "decision_rule",
        "validation_predictions_tsv",
        "source_sha256",
    ]

    write_tsv(
        FROZEN_TSV,
        threshold_rows,
        threshold_fields,
    )

    write_tsv(
        PARITY_TSV,
        parity_rows,
        [
            "mode",
            "protocol",
            "split_seed",
            "model_seed",
            "e6_threshold",
            "original_main_threshold",
            "abs_delta",
            "threshold_match",
            "source_sha_match",
            "status",
        ],
    )

    # ========================================================
    # Audit / provenance
    # ========================================================

    original_threshold_sha = (
        sha256_file(
            ORIGINAL_MAIN_THRESHOLDS
        )
    )

    audit = {
        "status":
            "PASS",

        "stage":
            "E6 validation-only threshold freeze",

        "protocol":
            "G2",

        "split_seed":
            20260915,

        "cells":
            30,

        "modes":
            {
                "D0": 5,
                "D1": 5,
                "D2": 5,
                "D3": 5,
                "D4": 5,
                "D5": 5,
            },

        "score_column":
            "similarity_score",

        "decision_rule":
            "similarity_score >= threshold",

        "threshold_candidate_rule": [
            "nextafter(max_score,+inf)",
            (
                "midpoint between adjacent "
                "unique score groups"
            ),
            "nextafter(min_score,-inf)",
        ],

        "threshold_selection_rule": [
            "maximum validation MCC",
            "higher validation Balanced Accuracy",
            "higher validation F1",
            "threshold closest to 0.5",
            "lower numerical threshold",
        ],

        "manifest_sha256":
            actual_manifest_sha,

        "original_17c1_sha256":
            actual_17c1_sha,

        "original_main_thresholds_sha256":
            original_threshold_sha,

        "d3_main_parity":
            "PASS",

        "d3_main_seeds": [
            20260915,
            20260916,
            20260917,
        ],

        "test_used":
            False,

        "external_used":
            False,

        "retraining_performed":
            False,

        "post_test_tuning":
            False,
    }

    AUDIT_JSON.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Human-readable report
    # ========================================================

    mode_summary = {}

    for mode in [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]:

        vals = [
            float(
                r[
                    "threshold"
                ]
            )
            for r in threshold_rows
            if r["mode"] == mode
        ]

        mode_summary[
            mode
        ] = {
            "n": len(vals),
            "mean": float(
                np.mean(vals)
            ),
            "sd": float(
                np.std(
                    vals,
                    ddof=1,
                )
            ),
            "min": float(
                np.min(vals)
            ),
            "max": float(
                np.max(vals)
            ),
        }

    report_lines = [
        "=" * 80,
        "HGV-V3 E6 VALIDATION THRESHOLD FREEZE REPORT",
        "=" * 80,
        "",
        "STATUS: PASS",
        "",
        "Protocol: G2",
        "Split seed: 20260915",
        "Frozen thresholds: 30",
        "",
        "Threshold source:",
        "  saved validation predictions only",
        "",
        "Score:",
        "  similarity_score",
        "",
        "Decision rule:",
        "  similarity_score >= threshold",
        "",
        "Candidate thresholds:",
        "  1. nextafter(max score, +inf)",
        "  2. midpoint between adjacent unique score groups",
        "  3. nextafter(min score, -inf)",
        "",
        "Selection:",
        "  1. maximum validation MCC",
        "  2. higher Balanced Accuracy",
        "  3. higher F1",
        "  4. threshold closest to 0.5",
        "  5. lower numerical threshold",
        "",
        "D3 frozen-main parity:",
        "  seeds 20260915/20260916/20260917 = PASS",
        "",
        "Test used: NO",
        "External used: NO",
        "Retraining: NO",
        "Post-test tuning: NO",
        "",
        "Threshold summary by mode:",
    ]

    for mode in [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]:

        s = mode_summary[
            mode
        ]

        report_lines.append(
            f"  {mode}: "
            f"n={s['n']} "
            f"mean={s['mean']:.10f} "
            f"sd={s['sd']:.10f} "
            f"min={s['min']:.10f} "
            f"max={s['max']:.10f}"
        )

    report_lines += [
        "",
        "E6_30CELL_VALIDATION_THRESHOLD_FREEZE=PASS",
        "",
    ]

    REPORT.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SHA provenance
    # ========================================================

    script_path = Path(
        __file__
    ).resolve()

    sha_lines = [
        (
            f"{sha256_file(FROZEN_TSV)}  "
            f"{FROZEN_TSV}"
        ),
        (
            f"{sha256_file(PARITY_TSV)}  "
            f"{PARITY_TSV}"
        ),
        (
            f"{sha256_file(AUDIT_JSON)}  "
            f"{AUDIT_JSON}"
        ),
        (
            f"{sha256_file(REPORT)}  "
            f"{REPORT}"
        ),
        (
            f"{actual_manifest_sha}  "
            f"{MANIFEST}"
        ),
        (
            f"{actual_17c1_sha}  "
            f"{ORIGINAL_17C1}"
        ),
        (
            f"{original_threshold_sha}  "
            f"{ORIGINAL_MAIN_THRESHOLDS}"
        ),
        (
            f"{sha256_file(script_path)}  "
            f"{script_path}"
        ),
    ]

    SHA_FILE.write_text(
        "\n".join(
            sha_lines
        )
        + "\n",
        encoding="utf-8",
    )

    frozen_tsv_sha = sha256_file(
        FROZEN_TSV
    )

    # ========================================================
    # Final freeze marker
    # ========================================================

    freeze_lines = [
        "HGV-V3 E6 30-CELL VALIDATION THRESHOLD FREEZE",
        "VERSION=v1",
        "DATE=2026-09-24",
        "STATUS=PASS",
        "PROTOCOL=G2",
        "SPLIT_SEED=20260915",
        "CELLS=30",
        "THRESHOLDS=30",
        "D0=5",
        "D1=5",
        "D2=5",
        "D3=5",
        "D4=5",
        "D5=5",
        "SCORE_COLUMN=similarity_score",
        "DECISION_RULE=similarity_score>=threshold",
        "D3_FROZEN_MAIN_PARITY=PASS",
        "TEST_DATA_READ=NO",
        "EXTERNAL_DATA_READ=NO",
        "RETRAINING=NO",
        "POST_TEST_TUNING=NO",
        f"MANIFEST_SHA256={actual_manifest_sha}",
        f"ORIGINAL_17C1_SHA256={actual_17c1_sha}",
        f"FROZEN_THRESHOLD_SHA256={frozen_tsv_sha}",
        "E6_30CELL_VALIDATION_THRESHOLD_FREEZE=PASS",
    ]

    FREEZE_FILE.write_text(
        "\n".join(
            freeze_lines
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "thresholds      = 30"
    )

    print(
        "D3 main parity  = PASS"
    )

    print(
        "test used       = NO"
    )

    print(
        "external used   = NO"
    )

    print(
        f"threshold file  = {FROZEN_TSV}"
    )

    print(
        f"threshold sha   = {frozen_tsv_sha}"
    )

    print()
    print(
        "E6_30CELL_VALIDATION_THRESHOLD_FREEZE=PASS"
    )


if __name__ == "__main__":
    main()
