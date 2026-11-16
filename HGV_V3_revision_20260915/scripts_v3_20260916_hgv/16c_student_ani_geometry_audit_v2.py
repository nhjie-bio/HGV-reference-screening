#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


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

SPLIT = (
    BASE
    / "04_splits/g1_g4_v1_20260916/manifests/"
      "G2_seed20260915.tsv"
)

GEOMETRY = (
    BASE
    / "06_teacher/teacher_geometry_audit_v1_20260916/"
      "canonical_g2_val_teacher_skani.tsv"
)

TRAIN_ROOT = (
    BASE
    / "07_hgv/teacher_selection_training_v4_20260916"
)

OUT = (
    BASE
    / "07_hgv/student_ani_geometry_audit_v2_20260916"
)

PCA_SEED = 20260915


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


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def detect_acc_col(fields):
    for c in [
        "assembly_accession",
        "assembly_accession_final",
        "accession",
    ]:
        if c in fields:
            return c

    raise RuntimeError(
        f"Could not detect accession column: {fields}"
    )


def safe_corr(func, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    result = func(x, y)

    if hasattr(result, "statistic"):
        return float(result.statistic)

    return float(result[0])


def mean_sd(values):
    values = [
        float(x)
        for x in values
        if math.isfinite(float(x))
    ]

    if not values:
        return float("nan"), float("nan")

    mean = statistics.mean(values)

    sd = (
        statistics.stdev(values)
        if len(values) >= 2
        else 0.0
    )

    return mean, sd


class HGVEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2080, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)

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


print("=" * 78)
print("HGV-V3 STUDENT / ANI GEOMETRY AUDIT")
print("=" * 78)

if OUT.exists():
    raise SystemExit(
        f"ERROR: output already exists: {OUT}"
    )

OUT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1. Features and accession mapping
# ----------------------------------------------------------------------

print()
print("[1] LOAD FEATURES")

ff, fr = read_tsv(
    FEATURE_ROWS
)

acc_col = detect_acc_col(ff)

acc_to_idx = {
    clean(r[acc_col]): i
    for i, r in enumerate(fr)
}

X = np.load(
    FEATURE_NPY,
    mmap_mode="r",
)

assert X.shape == (3154, 2080)

print(f"feature_shape={X.shape}")
print(f"feature_accessions={len(acc_to_idx)}")

# ----------------------------------------------------------------------
# 2. Frozen G2 train / validation genomes
# ----------------------------------------------------------------------

print()
print("[2] LOAD G2 TRAIN / VALIDATION")

sf, sr = read_tsv(SPLIT)

split_acc_col = detect_acc_col(sf)

train_acc = sorted({
    clean(r[split_acc_col])
    for r in sr
    if clean(r["role"]) == "train"
})

val_acc = sorted({
    clean(r[split_acc_col])
    for r in sr
    if clean(r["role"]) == "val"
})

print(f"train_genomes={len(train_acc)}")
print(f"val_genomes={len(val_acc)}")
print(
    f"train_val_overlap="
    f"{len(set(train_acc) & set(val_acc))}"
)

assert len(train_acc) == 1662
assert len(val_acc) == 356
assert not (
    set(train_acc)
    & set(val_acc)
)

train_idx = np.asarray(
    [acc_to_idx[a] for a in train_acc],
    dtype=np.int64,
)

val_idx = np.asarray(
    [acc_to_idx[a] for a in val_acc],
    dtype=np.int64,
)

val_pos = {
    acc: i
    for i, acc in enumerate(val_acc)
}

X_train = np.asarray(
    X[train_idx],
    dtype=np.float32,
)

X_val = np.asarray(
    X[val_idx],
    dtype=np.float32,
)

write_tsv(
    OUT / "val_accession_order.tsv",
    [
        {
            "row_index": i,
            "assembly_accession": a,
        }
        for i, a in enumerate(val_acc)
    ],
    [
        "row_index",
        "assembly_accession",
    ],
)

# ----------------------------------------------------------------------
# 3. Fit PCA on TRAIN ONLY
# ----------------------------------------------------------------------

print()
print("[3] FIT PCA-64 ON TRAIN ONLY")

pca = PCA(
    n_components=64,
    svd_solver="randomized",
    random_state=PCA_SEED,
)

pca.fit(X_train)

P_val = pca.transform(
    X_val
).astype(np.float32)

P_val_l2 = (
    P_val
    / np.clip(
        np.linalg.norm(
            P_val,
            axis=1,
            keepdims=True,
        ),
        1e-12,
        None,
    )
)

np.save(
    OUT / "PCA64_val_embedding.float32.npy",
    P_val,
)

np.save(
    OUT / "PCA64_L2_val_embedding.float32.npy",
    P_val_l2.astype(np.float32),
)

pca_summary = {
    "fit_scope": "G2_seed20260915 train only",
    "train_genomes": len(train_acc),
    "validation_genomes": len(val_acc),
    "n_components": 64,
    "random_state": PCA_SEED,
    "explained_variance_ratio_sum": float(
        pca.explained_variance_ratio_.sum()
    ),
    "test_used": False,
    "external_used": False,
}

with open(
    OUT / "pca_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        pca_summary,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

print(
    "PCA64 explained variance="
    f"{pca_summary['explained_variance_ratio_sum']:.6f}"
)

# ----------------------------------------------------------------------
# 4. Validation pair table
# ----------------------------------------------------------------------

print()
print("[4] LOAD SKANI CONTROL PAIRS")

gf, gr = read_tsv(
    GEOMETRY
)

for r in gr:
    a = clean(r["accession_a"])
    b = clean(r["accession_b"])

    if a not in val_pos or b not in val_pos:
        raise RuntimeError(
            f"Non-validation endpoint: {a}, {b}"
        )

print(f"all_validation_pairs={len(gr)}")

skani_rows = [
    r
    for r in gr
    if clean(r["skani_reported"]) == "1"
]

print(f"skani_reported_pairs={len(skani_rows)}")

# ----------------------------------------------------------------------
# Helpers for pair scores
# ----------------------------------------------------------------------

def scores_from_embedding(
    embedding,
    normalize_distance=True,
):
    scores = []

    for r in gr:
        i = val_pos[
            clean(r["accession_a"])
        ]
        j = val_pos[
            clean(r["accession_b"])
        ]

        d = float(
            np.linalg.norm(
                embedding[i]
                - embedding[j]
            )
        )

        if normalize_distance:
            score = 1.0 - d / 2.0
        else:
            score = -d

        scores.append(score)

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def raw_bray_scores():
    result = []

    for r in gr:
        i = val_pos[
            clean(r["accession_a"])
        ]
        j = val_pos[
            clean(r["accession_b"])
        ]

        d = 0.5 * float(
            np.abs(
                X_val[i].astype(np.float64)
                - X_val[j].astype(np.float64)
            ).sum()
        )

        result.append(
            1.0 - d
        )

    return np.asarray(result)


def raw_cosine_scores():
    norms = np.linalg.norm(
        X_val.astype(np.float64),
        axis=1,
    )

    result = []

    for r in gr:
        i = val_pos[
            clean(r["accession_a"])
        ]
        j = val_pos[
            clean(r["accession_b"])
        ]

        den = norms[i] * norms[j]

        score = (
            float(
                np.dot(
                    X_val[i],
                    X_val[j],
                )
                / den
            )
            if den > 0
            else 0.0
        )

        result.append(score)

    return np.asarray(result)


# ----------------------------------------------------------------------
# 5. Register fixed representations
# ----------------------------------------------------------------------

print()
print("[5] BUILD FIXED BASELINES")

representations = {}

representations[
    "RawK6_BrayCurtis"
] = raw_bray_scores()

representations[
    "RawK6_Cosine"
] = raw_cosine_scores()

representations[
    "RawK6_Euclidean"
] = scores_from_embedding(
    X_val,
    normalize_distance=False,
)

representations[
    "PCA64_Euclidean"
] = scores_from_embedding(
    P_val,
    normalize_distance=False,
)

representations[
    "PCA64_L2"
] = scores_from_embedding(
    P_val_l2,
    normalize_distance=True,
)

representations[
    "Mash_native"
] = np.asarray([
    1.0 - float(
        r["mash_distance"]
    )
    for r in gr
])

representations[
    "Dashing2_native"
] = np.asarray([
    1.0 - float(
        r["dashing2_set_distance"]
    )
    for r in gr
])

# ----------------------------------------------------------------------
# 6. HGV checkpoint inference
# ----------------------------------------------------------------------

print()
print("[6] INFER ALL 15 HGV CHECKPOINTS")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

X_val_t = torch.as_tensor(
    X_val,
    dtype=torch.float32,
    device=device,
)

run_metadata = {}

embedding_dir = (
    OUT / "hgv_val_embeddings"
)

embedding_dir.mkdir()

run_dirs = sorted([
    p
    for p in TRAIN_ROOT.iterdir()
    if p.is_dir()
    and (
        p / "RUN_COMPLETE"
    ).is_file()
])

if len(run_dirs) != 15:
    raise RuntimeError(
        f"Expected 15 completed runs, found {len(run_dirs)}"
    )

for run_dir in run_dirs:

    summary = json.loads(
        (
            run_dir
            / "run_summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    ckpt = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
        weights_only=False,
    )

    model = HGVEncoder().to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        Z = model(
            X_val_t
        ).cpu().numpy().astype(
            np.float32
        )

    run_name = run_dir.name

    np.save(
        embedding_dir
        / f"{run_name}.float32.npy",
        Z,
    )

    representations[
        run_name
    ] = scores_from_embedding(
        Z,
        normalize_distance=True,
    )

    run_metadata[
        run_name
    ] = {
        "type": "HGV",
        "config_id": summary[
            "config_id"
        ],
        "teacher": summary[
            "teacher"
        ],
        "tau": summary[
            "tau"
        ],
        "seed": summary[
            "model_seed"
        ],
        "best_epoch": summary[
            "best_epoch"
        ],
    }

    print(
        f"{run_name:45s} "
        f"best_epoch={summary['best_epoch']}"
    )

# ----------------------------------------------------------------------
# 7. Discrimination metrics
# ----------------------------------------------------------------------

print()
print("[7] CLASSIFICATION CEILING CHECK")

labels = np.asarray([
    int(r["label"])
    for r in gr
])

pair_types = np.asarray([
    clean(r["pair_type"])
    for r in gr
])

classification_rows = []

for name, scores in representations.items():

    for subset in [
        "all",
        "hard",
        "medium",
        "easy",
    ]:

        if subset == "all":
            mask = np.ones(
                len(gr),
                dtype=bool,
            )
        else:
            mask = (
                (labels == 1)
                |
                (pair_types == subset)
            )

        y = labels[mask]
        s = scores[mask]

        if len(np.unique(y)) < 2:
            continue

        classification_rows.append({
            "representation": name,
            "subset": subset,
            "n": len(y),
            "AUROC": float(
                roc_auc_score(y, s)
            ),
            "AUPRC": float(
                average_precision_score(
                    y,
                    s,
                )
            ),
        })

# ----------------------------------------------------------------------
# 8. ANI / AF geometry metrics
# ----------------------------------------------------------------------

print()
print("[8] ANI / AF GEOMETRY METRICS")

skani_indices = np.asarray([
    i
    for i, r in enumerate(gr)
    if clean(r["skani_reported"]) == "1"
])

ani = np.asarray([
    float(gr[i]["skani_ani"])
    for i in skani_indices
])

af = np.asarray([
    float(gr[i]["skani_af_mean"])
    for i in skani_indices
])

sk_labels = labels[
    skani_indices
]

sk_types = pair_types[
    skani_indices
]

geometry_rows = []

for name, scores_all in representations.items():

    s_all = scores_all[
        skani_indices
    ]

    subsets = {
        "all_skani_reported": (
            np.ones(
                len(skani_indices),
                dtype=bool,
            )
        ),
        "positive_only": (
            sk_labels == 1
        ),
        "hard_negative_only": (
            sk_types == "hard"
        ),
    }

    for subset_name, mask in subsets.items():

        s = s_all[mask]
        a = ani[mask]
        f = af[mask]

        geometry_rows.append({
            "representation": name,
            "subset": subset_name,
            "n_pairs": len(s),

            "ANI_pearson": safe_corr(
                pearsonr,
                s,
                a,
            ),

            "ANI_spearman": safe_corr(
                spearmanr,
                s,
                a,
            ),

            "AF_pearson": safe_corr(
                pearsonr,
                s,
                f,
            ),

            "AF_spearman": safe_corr(
                spearmanr,
                s,
                f,
            ),

            "ANI_min": (
                float(np.min(a))
                if len(a)
                else ""
            ),

            "ANI_max": (
                float(np.max(a))
                if len(a)
                else ""
            ),
        })

# ----------------------------------------------------------------------
# 9. Aggregate HGV configurations across seeds
# ----------------------------------------------------------------------

print()
print("[9] THREE-SEED CONFIGURATION SUMMARY")

metric_lookup = defaultdict(dict)

for r in geometry_rows:
    metric_lookup[
        r["representation"]
    ][
        r["subset"]
    ] = r

config_groups = defaultdict(list)

for run_name, meta in run_metadata.items():
    config_groups[
        meta["config_id"]
    ].append(
        run_name
    )

config_summary = []

for config_id, runs in sorted(
    config_groups.items()
):

    row = {
        "config_id": config_id,
        "n_seeds": len(runs),
    }

    for subset in [
        "all_skani_reported",
        "positive_only",
        "hard_negative_only",
    ]:

        for metric in [
            "ANI_pearson",
            "ANI_spearman",
            "AF_spearman",
        ]:

            vals = [
                metric_lookup[
                    run
                ][
                    subset
                ][
                    metric
                ]
                for run in runs
            ]

            mean, sd = mean_sd(
                vals
            )

            prefix = (
                subset
                .replace(
                    "_skani_reported",
                    "",
                )
                .replace(
                    "_negative_only",
                    "",
                )
                .replace(
                    "_only",
                    "",
                )
            )

            row[
                f"{prefix}_{metric}_mean"
            ] = mean

            row[
                f"{prefix}_{metric}_sd"
            ] = sd

    config_summary.append(row)

# ----------------------------------------------------------------------
# 10. Fixed-baseline summary
# ----------------------------------------------------------------------

fixed_names = [
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash_native",
    "Dashing2_native",
]

fixed_rows = []

for name in fixed_names:

    for subset in [
        "all_skani_reported",
        "positive_only",
        "hard_negative_only",
    ]:

        r = metric_lookup[
            name
        ][
            subset
        ]

        fixed_rows.append({
            "representation": name,
            "subset": subset,
            "n_pairs": r["n_pairs"],
            "ANI_pearson": r[
                "ANI_pearson"
            ],
            "ANI_spearman": r[
                "ANI_spearman"
            ],
            "AF_spearman": r[
                "AF_spearman"
            ],
        })

# ----------------------------------------------------------------------
# 11. Write outputs
# ----------------------------------------------------------------------

write_tsv(
    OUT / "classification_metrics.tsv",
    classification_rows,
    [
        "representation",
        "subset",
        "n",
        "AUROC",
        "AUPRC",
    ],
)

write_tsv(
    OUT / "geometry_metrics_per_representation.tsv",
    geometry_rows,
    [
        "representation",
        "subset",
        "n_pairs",
        "ANI_pearson",
        "ANI_spearman",
        "AF_pearson",
        "AF_spearman",
        "ANI_min",
        "ANI_max",
    ],
)

write_tsv(
    OUT / "fixed_baseline_geometry.tsv",
    fixed_rows,
    [
        "representation",
        "subset",
        "n_pairs",
        "ANI_pearson",
        "ANI_spearman",
        "AF_spearman",
    ],
)

summary_fields = [
    "config_id",
    "n_seeds",
]

for subset in [
    "all",
    "positive",
    "hard",
]:
    for metric in [
        "ANI_pearson",
        "ANI_spearman",
        "AF_spearman",
    ]:
        summary_fields.extend([
            f"{subset}_{metric}_mean",
            f"{subset}_{metric}_sd",
        ])

write_tsv(
    OUT / "hgv_config_geometry_summary.tsv",
    config_summary,
    summary_fields,
)

# ----------------------------------------------------------------------
# 12. Human report
# ----------------------------------------------------------------------

report = (
    OUT
    / "STUDENT_ANI_GEOMETRY_AUDIT_REPORT.txt"
)

with open(
    report,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 STUDENT / ANI GEOMETRY AUDIT\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write("STATUS: PASS\n\n")

    f.write(
        f"G2 train genomes: {len(train_acc)}\n"
    )

    f.write(
        f"G2 validation genomes: {len(val_acc)}\n"
    )

    f.write(
        f"Validation pairs: {len(gr)}\n"
    )

    f.write(
        f"skani-reported pairs: "
        f"{len(skani_rows)}\n"
    )

    f.write(
        "Test used: NO\n"
    )

    f.write(
        "External used: NO\n\n"
    )

    f.write(
        "PCA fit scope: G2 train genomes only\n"
    )

    f.write(
        "PCA64 explained variance: "
        f"{pca_summary['explained_variance_ratio_sum']:.6f}\n\n"
    )

    f.write(
        "PRIMARY GEOMETRY SELECTION METRIC:\n"
    )

    f.write(
        "  positive_only ANI Spearman correlation\n"
    )

    f.write(
        "Rationale: removes the trivial same-species vs "
        "different-species ranking boundary and tests "
        "continuous within-species genomic geometry.\n\n"
    )

    f.write(
        "HGV CONFIGURATION SUMMARY\n"
    )

    for r in config_summary:

        f.write(
            f"{r['config_id']}: "
            f"positive ANI Spearman="
            f"{r['positive_ANI_spearman_mean']:.6f}"
            f" +/- "
            f"{r['positive_ANI_spearman_sd']:.6f}; "
            f"all ANI Spearman="
            f"{r['all_ANI_spearman_mean']:.6f}"
            f" +/- "
            f"{r['all_ANI_spearman_sd']:.6f}; "
            f"hard ANI Spearman="
            f"{r['hard_ANI_spearman_mean']:.6f}"
            f" +/- "
            f"{r['hard_ANI_spearman_sd']:.6f}\n"
        )

    f.write("\n")

    f.write(
        "PRIMARY TEACHER STATUS: NOT FROZEN\n"
    )

    f.write(
        "Freeze only after reviewing HGV geometry against "
        "Raw-k6, PCA-64 and native teacher references.\n\n"
    )

    f.write(
        "STUDENT_ANI_GEOMETRY_AUDIT=PASS\n"
    )

# ----------------------------------------------------------------------
# 13. SHA256
# ----------------------------------------------------------------------

files = sorted([
    p
    for p in OUT.rglob("*")
    if p.is_file()
    and p.name != "SHA256SUMS.txt"
])

with open(
    OUT / "SHA256SUMS.txt",
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
print("STUDENT / ANI GEOMETRY AUDIT COMPLETE")
print("=" * 78)
print(f"Output: {OUT}")
print()
print("STUDENT_ANI_GEOMETRY_AUDIT=PASS")
