#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
except Exception as e:
    print(f"TORCH_IMPORT_ERROR={repr(e)}")
    sys.exit(2)


BASE = Path("HGV_V3_revision_20260915")

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

TEACHER_DIR = (
    BASE
    / "06_teacher/teacher_geometry_audit_v1_20260916"
    / "teacher_augmented/G2_seed20260915"
)

TRAIN_TSV = (
    TEACHER_DIR
    / "train_teacher.tsv"
)

VAL_TSV = (
    TEACHER_DIR
    / "val_teacher.tsv"
)

OUT = (
    BASE
    / "07_hgv/teacher_selection_preflight_v1_20260916"
)

TAUS = [0.90, 0.95, 0.98]

TEACHERS = {
    "Mash": "mash_distance",
    "Dashing2_SetSketch": "dashing2_set_distance",
}


def die(msg):
    print()
    print(f"ERROR: {msg}")
    print("HGV_TEACHER_SELECTION_PREFLIGHT=FAIL")
    sys.exit(2)


def clean(x):
    return "" if x is None else str(x).strip()


def read_tsv(path):
    if not path.is_file():
        die(f"Missing file: {path}")

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
        fields = list(r.fieldnames or [])
        rows = list(r)

    if not fields:
        die(f"No header: {path}")

    return fields, rows


def write_tsv(path, rows, fields):
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


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def detect_accession_column(fields):
    candidates = [
        "assembly_accession_final",
        "assembly_accession",
        "accession",
    ]

    for c in candidates:
        if c in fields:
            return c

    raise RuntimeError(
        "Could not detect feature-row accession column. "
        f"Fields={fields}"
    )


def safe_float(x):
    try:
        v = float(clean(x))
    except Exception:
        return None

    if not math.isfinite(v):
        return None

    return v


def canonical_pair(a, b):
    a = clean(a)
    b = clean(b)

    return tuple(sorted((a, b)))


def mask_hash(rows, teacher_field, tau):
    """
    D3 selective distillation mask:
      all positive pairs
      +
      negative pairs with teacher_distance < tau

    Hash only the frozen UNIQUE pair identities included
    in the distillation component.
    """

    keys = []

    n_pos = 0
    n_neg_info = 0
    n_neg_total = 0

    for r in rows:
        label = int(r["label"])
        d = safe_float(r[teacher_field])

        if d is None:
            raise RuntimeError(
                f"Missing teacher value in {teacher_field}"
            )

        a, b = canonical_pair(
            r["accession_a"],
            r["accession_b"],
        )

        include = False

        if label == 1:
            include = True
            n_pos += 1
        else:
            n_neg_total += 1

            if d < tau:
                include = True
                n_neg_info += 1

        if include:
            keys.append(
                f"{a}\t{b}\t{label}\n"
            )

    keys.sort()

    h = hashlib.sha256(
        "".join(keys).encode("utf-8")
    ).hexdigest()

    return {
        "mask_sha256": h,
        "positive_distill_n": n_pos,
        "negative_total_n": n_neg_total,
        "informative_negative_n": n_neg_info,
        "informative_negative_fraction": (
            n_neg_info / n_neg_total
            if n_neg_total
            else 0.0
        ),
        "distill_unique_pairs_n": (
            n_pos + n_neg_info
        ),
    }


print("=" * 78)
print("HGV-V3 TEACHER-SELECTION PREFLIGHT")
print("=" * 78)

if OUT.exists():
    die(f"Output exists: {OUT}")

OUT.mkdir(
    parents=True,
    exist_ok=False,
)

# ----------------------------------------------------------------------
# 1. Torch/device
# ----------------------------------------------------------------------

print()
print("[1] TORCH / DEVICE")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"selected_device={device}")

gpu_name = ""

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)

    print(f"gpu_name={gpu_name}")
    print(
        "gpu_memory_GiB="
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.3f}"
    )

# ----------------------------------------------------------------------
# 2. Feature matrix
# ----------------------------------------------------------------------

print()
print("[2] FEATURE MATRIX")

if not FEATURE_NPY.is_file():
    die(f"Missing feature matrix: {FEATURE_NPY}")

X = np.load(
    FEATURE_NPY,
    mmap_mode="r",
)

print(f"feature_shape={X.shape}")
print(f"feature_dtype={X.dtype}")

if X.shape != (3154, 2080):
    die(
        f"Expected feature shape (3154,2080); "
        f"observed {X.shape}"
    )

if X.dtype != np.float32:
    die(
        f"Expected float32 features; "
        f"observed {X.dtype}"
    )

fields, feature_rows = read_tsv(
    FEATURE_ROWS
)

try:
    acc_col = detect_accession_column(
        fields
    )
except Exception as e:
    die(str(e))

if len(feature_rows) != 3154:
    die(
        f"Expected 3154 feature rows; "
        f"observed {len(feature_rows)}"
    )

acc_to_row = {}

for i, r in enumerate(feature_rows):
    acc = clean(r[acc_col])

    if not acc:
        die(
            f"Empty accession at feature row {i}"
        )

    if acc in acc_to_row:
        die(
            f"Duplicate feature accession: {acc}"
        )

    acc_to_row[acc] = i

print(f"feature_accession_column={acc_col}")
print(f"feature_accessions={len(acc_to_row)}")

# ----------------------------------------------------------------------
# 3. Pair manifests
# ----------------------------------------------------------------------

print()
print("[3] CANONICAL G2 TRAIN / VALIDATION")

train_fields, train_rows = read_tsv(
    TRAIN_TSV
)

val_fields, val_rows = read_tsv(
    VAL_TSV
)

required_pair_cols = {
    "accession_a",
    "accession_b",
    "label",
    "mash_distance",
    "dashing2_set_distance",
}

for name, fields0 in [
    ("train", train_fields),
    ("val", val_fields),
]:
    missing = required_pair_cols - set(
        fields0
    )

    if missing:
        die(
            f"{name} pair table missing: "
            + ",".join(sorted(missing))
        )

print(f"train_unique_pairs={len(train_rows)}")
print(f"val_unique_pairs={len(val_rows)}")

if len(train_rows) != 18176:
    die(
        f"Expected canonical G2 train=18176; "
        f"observed {len(train_rows)}"
    )

if len(val_rows) != 3874:
    die(
        f"Expected canonical G2 val=3874; "
        f"observed {len(val_rows)}"
    )

# ----------------------------------------------------------------------
# 4. Pair-to-feature join
# ----------------------------------------------------------------------

print()
print("[4] PAIR -> FEATURE JOIN")

join_summary = []

for role, rows in [
    ("train", train_rows),
    ("val", val_rows),
]:

    missing_acc = []
    labels = []

    for r in rows:
        a = clean(r["accession_a"])
        b = clean(r["accession_b"])

        if a not in acc_to_row:
            missing_acc.append(a)

        if b not in acc_to_row:
            missing_acc.append(b)

        labels.append(
            int(r["label"])
        )

    c0 = labels.count(0)
    c1 = labels.count(1)

    print(
        f"{role}: pairs={len(rows)} "
        f"positive={c1} "
        f"negative={c0} "
        f"missing_feature_accessions={len(missing_acc)}"
    )

    if missing_acc:
        die(
            f"{role} has pair accessions missing "
            f"from feature matrix: "
            f"{sorted(set(missing_acc))[:10]}"
        )

    if c0 != c1:
        die(
            f"{role} is not 1:1 balanced: "
            f"pos={c1}, neg={c0}"
        )

    join_summary.append({
        "role": role,
        "pairs": len(rows),
        "positive": c1,
        "negative": c0,
        "missing_feature_accessions": 0,
        "gate": "PASS",
    })

# ----------------------------------------------------------------------
# 5. Numerical feature sanity on all used genomes
# ----------------------------------------------------------------------

print()
print("[5] USED-FEATURE NUMERICAL SANITY")

used_acc = set()

for rows in [
    train_rows,
    val_rows,
]:
    for r in rows:
        used_acc.add(
            clean(r["accession_a"])
        )
        used_acc.add(
            clean(r["accession_b"])
        )

used_indices = np.array(
    sorted(
        acc_to_row[a]
        for a in used_acc
    ),
    dtype=np.int64,
)

X_used = np.asarray(
    X[used_indices],
    dtype=np.float32,
)

finite = bool(
    np.isfinite(X_used).all()
)

row_sums = X_used.sum(
    axis=1,
    dtype=np.float64,
)

max_row_sum_error = float(
    np.max(
        np.abs(
            row_sums - 1.0
        )
    )
)

print(f"used_genomes={len(used_acc)}")
print(f"features_finite={finite}")
print(
    "max_L1_row_sum_error="
    f"{max_row_sum_error:.12g}"
)

if not finite:
    die(
        "Non-finite values in used features"
    )

if max_row_sum_error > 1e-5:
    die(
        f"Unexpected feature normalization error: "
        f"{max_row_sum_error}"
    )

# ----------------------------------------------------------------------
# 6. Teacher target sanity
# ----------------------------------------------------------------------

print()
print("[6] TEACHER TARGET SANITY")

teacher_target_summary = []

for role, rows in [
    ("train", train_rows),
    ("val", val_rows),
]:

    for teacher, field in TEACHERS.items():

        vals = []

        for r in rows:
            d = safe_float(r[field])

            if d is None:
                die(
                    f"{teacher} has missing "
                    f"{role} distance"
                )

            if not (
                -1e-8
                <= d
                <= 1.0 + 1e-8
            ):
                die(
                    f"{teacher} {role} target "
                    f"outside [0,1]: {d}"
                )

            vals.append(d)

        teacher_target_summary.append({
            "role": role,
            "teacher": teacher,
            "n": len(vals),
            "min_distance": min(vals),
            "max_distance": max(vals),
            "mean_distance": (
                sum(vals) / len(vals)
            ),
            "distance_eq_1_n": sum(
                1
                for x in vals
                if x >= 1.0 - 1e-12
            ),
        })

        print(
            f"{role:5s} "
            f"{teacher:22s} "
            f"min={min(vals):.6f} "
            f"max={max(vals):.6f} "
            f"mean={sum(vals)/len(vals):.6f}"
        )

# ----------------------------------------------------------------------
# 7. D3 distillation masks
# ----------------------------------------------------------------------

print()
print("[7] D3 INFORMATIVE-NEGATIVE MASKS")

mask_rows = []

for teacher, field in TEACHERS.items():

    print()
    print(f"Teacher: {teacher}")

    for tau in TAUS:

        train_mask = mask_hash(
            train_rows,
            field,
            tau,
        )

        val_mask = mask_hash(
            val_rows,
            field,
            tau,
        )

        row = {
            "teacher": teacher,
            "tau": tau,

            "train_positive_distill_n": (
                train_mask[
                    "positive_distill_n"
                ]
            ),

            "train_negative_total_n": (
                train_mask[
                    "negative_total_n"
                ]
            ),

            "train_informative_negative_n": (
                train_mask[
                    "informative_negative_n"
                ]
            ),

            "train_informative_negative_fraction": (
                train_mask[
                    "informative_negative_fraction"
                ]
            ),

            "train_distill_unique_pairs_n": (
                train_mask[
                    "distill_unique_pairs_n"
                ]
            ),

            "train_mask_sha256": (
                train_mask[
                    "mask_sha256"
                ]
            ),

            "val_informative_negative_n": (
                val_mask[
                    "informative_negative_n"
                ]
            ),

            "val_informative_negative_fraction": (
                val_mask[
                    "informative_negative_fraction"
                ]
            ),

            "val_mask_sha256": (
                val_mask[
                    "mask_sha256"
                ]
            ),
        }

        mask_rows.append(row)

        print(
            f"tau={tau:.2f} "
            f"train_info_neg="
            f"{train_mask['informative_negative_n']}/"
            f"{train_mask['negative_total_n']} "
            f"({train_mask['informative_negative_fraction']:.6f}) "
            f"mask={train_mask['mask_sha256'][:12]}"
        )

# ----------------------------------------------------------------------
# 8. Detect redundant training configurations
# ----------------------------------------------------------------------

print()
print("[8] REDUNDANT-CONFIG DETECTION")

groups = defaultdict(list)

for r in mask_rows:
    key = (
        r["teacher"],
        r["train_mask_sha256"],
    )

    groups[key].append(
        float(r["tau"])
    )

effective_configs = []

alias_rows = []

for (
    teacher,
    mask_sha,
), taus in sorted(groups.items()):

    taus = sorted(taus)

    canonical_tau = taus[0]

    effective_configs.append({
        "teacher": teacher,
        "canonical_tau": canonical_tau,
        "equivalent_taus": (
            ",".join(
                f"{x:.2f}"
                for x in taus
            )
        ),
        "train_mask_sha256": mask_sha,
    })

    for tau in taus:
        alias_rows.append({
            "teacher": teacher,
            "tau": tau,
            "canonical_tau": canonical_tau,
            "is_redundant_alias": (
                0
                if tau == canonical_tau
                else 1
            ),
            "train_mask_sha256": mask_sha,
        })

    print(
        f"{teacher:22s} "
        f"canonical_tau={canonical_tau:.2f} "
        f"equivalent="
        f"{','.join(f'{x:.2f}' for x in taus)} "
        f"mask={mask_sha[:12]}"
    )

print(
    f"Requested configs={len(TEACHERS) * len(TAUS)}"
)

print(
    f"Effective unique configs={len(effective_configs)}"
)

# ----------------------------------------------------------------------
# 9. Freeze proposed training definition
# ----------------------------------------------------------------------

print()
print("[9] PROPOSED HGV TRAINING DEFINITION")

training_definition = {
    "scope": {
        "split": "G2_seed20260915",
        "training": "train only",
        "selection": "validation only",
        "test_used": False,
        "external_used": False,
    },

    "input": {
        "feature": "canonical reverse-complement invariant k=6",
        "dimension": 2080,
        "dtype": "float32",
        "feature_normalization": "L1",
    },

    "encoder": {
        "layers": [
            2080,
            256,
            128,
            64,
        ],
        "hidden_blocks": "Linear + BatchNorm1d + ReLU",
        "output_normalization": "L2 unit norm",
    },

    "pair_distance": {
        "metric": "Euclidean",
        "embedding_range": "[0,2]",
        "similarity_for_AUROC_AUPRC": "1 - distance/2",
    },

    "contrastive_loss": {
        "positive_label": 1,
        "margin": 1.0,
        "formula": (
            "y*d^2 + (1-y)*max(0,1-d)^2"
        ),
    },

    "distillation": {
        "teacher_target_range": "[0,1]",
        "student_target": "embedding_distance/2",
        "loss": "MSE",
        "lambda": 0.25,
        "D3_mask": (
            "all positive pairs plus negative pairs "
            "with teacher_distance < tau"
        ),
        "tau_candidates": TAUS,
    },

    "optimization": {
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "train_batch_size": 512,
        "validation_batch_size": 1024,
        "max_epochs": 50,
        "early_stopping_patience": 8,
        "gradient_clip_norm": 1.0,
        "precision": "FP32",
        "training_examples_per_epoch": 100000,
        "positive_examples_per_epoch": 50000,
        "negative_examples_per_epoch": 50000,
        "sampling": (
            "with replacement from frozen unique "
            "positive and negative pools"
        ),
    },

    "teacher_selection_training_seeds": [
        20260915,
        20260916,
        20260917,
    ],

    "validation_selection_score": (
        "(AUROC + AUPRC) / 2"
    ),

    "primary_teacher_status": (
        "NOT_FROZEN"
    ),
}

with open(
    OUT / "proposed_training_definition.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        training_definition,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------

write_tsv(
    OUT / "feature_pair_join_summary.tsv",
    join_summary,
    [
        "role",
        "pairs",
        "positive",
        "negative",
        "missing_feature_accessions",
        "gate",
    ],
)

write_tsv(
    OUT / "teacher_target_summary.tsv",
    teacher_target_summary,
    [
        "role",
        "teacher",
        "n",
        "min_distance",
        "max_distance",
        "mean_distance",
        "distance_eq_1_n",
    ],
)

write_tsv(
    OUT / "distillation_mask_audit.tsv",
    mask_rows,
    [
        "teacher",
        "tau",
        "train_positive_distill_n",
        "train_negative_total_n",
        "train_informative_negative_n",
        "train_informative_negative_fraction",
        "train_distill_unique_pairs_n",
        "train_mask_sha256",
        "val_informative_negative_n",
        "val_informative_negative_fraction",
        "val_mask_sha256",
    ],
)

write_tsv(
    OUT / "effective_training_configs.tsv",
    effective_configs,
    [
        "teacher",
        "canonical_tau",
        "equivalent_taus",
        "train_mask_sha256",
    ],
)

write_tsv(
    OUT / "config_aliases.tsv",
    alias_rows,
    [
        "teacher",
        "tau",
        "canonical_tau",
        "is_redundant_alias",
        "train_mask_sha256",
    ],
)

config = {
    "status": "PASS",
    "torch_version": torch.__version__,
    "cuda_available": bool(
        torch.cuda.is_available()
    ),
    "selected_device": device,
    "gpu_name": gpu_name,
    "feature_shape": list(X.shape),
    "feature_dtype": str(X.dtype),
    "used_genomes": len(used_acc),
    "max_L1_row_sum_error": (
        max_row_sum_error
    ),
    "canonical_split": (
        "G2_seed20260915"
    ),
    "train_pairs": len(train_rows),
    "val_pairs": len(val_rows),
    "requested_configs": 6,
    "effective_unique_configs": (
        len(effective_configs)
    ),
    "test_used": False,
    "external_used": False,
}

with open(
    OUT / "preflight_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

report = OUT / "HGV_TEACHER_SELECTION_PREFLIGHT_REPORT.txt"

with open(
    report,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "HGV-V3 TEACHER-SELECTION PREFLIGHT\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write("STATUS: PASS\n\n")

    f.write(
        f"Feature matrix: {X.shape}, "
        f"{X.dtype}\n"
    )

    f.write(
        f"Used genomes: {len(used_acc)}\n"
    )

    f.write(
        f"Canonical split: "
        f"G2_seed20260915\n"
    )

    f.write(
        f"Train unique pairs: "
        f"{len(train_rows)}\n"
    )

    f.write(
        f"Validation unique pairs: "
        f"{len(val_rows)}\n"
    )

    f.write(
        f"Device: {device}\n"
    )

    if gpu_name:
        f.write(
            f"GPU: {gpu_name}\n"
        )

    f.write("\n")

    f.write(
        "Requested teacher/tau configs: 6\n"
    )

    f.write(
        "Effective unique train masks: "
        f"{len(effective_configs)}\n\n"
    )

    for r in effective_configs:
        f.write(
            f"{r['teacher']}: "
            f"canonical_tau={r['canonical_tau']}; "
            f"equivalent_taus="
            f"{r['equivalent_taus']}; "
            f"mask_sha256="
            f"{r['train_mask_sha256']}\n"
        )

    f.write("\n")
    f.write(
        "Test pair manifests read: NO\n"
    )

    f.write(
        "External genomes used: NO\n"
    )

    f.write(
        "Primary teacher frozen: NO\n\n"
    )

    f.write(
        "HGV_TEACHER_SELECTION_PREFLIGHT=PASS\n"
    )

# SHA256

checksum_files = sorted(
    p
    for p in OUT.iterdir()
    if p.is_file()
    and p.name != "SHA256SUMS.txt"
)

with open(
    OUT / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in checksum_files:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )

print()
print("=" * 78)
print("HGV TEACHER-SELECTION PREFLIGHT PASSED")
print("=" * 78)
print(f"Output: {OUT}")
print()
print("HGV_TEACHER_SELECTION_PREFLIGHT=PASS")
