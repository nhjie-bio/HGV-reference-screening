#!/usr/bin/env bash
set -euo pipefail
BASE="HGV_V3_revision_20260915"
V9="HGV_V3_revision_20260915/scripts_v3_20260926_remaining_all_v9"
FIX="HGV_V3_revision_20260915/scripts_v3_20260926_e7_e12_final_fix_v1"

OLD_E7="HGV_V3_revision_20260915/11_sensitivity/E7_sensitivity_v1_20260925_v6"
NEW_E7="HGV_V3_revision_20260915/11_sensitivity/E7_sensitivity_v2_gzipfix_20260925_v6"
E7_FINAL="HGV_V3_revision_20260915/11_sensitivity/E7_sensitivity_v2_gzipfix_20260925_v6/final_validation_refreeze_v2_20260926"

OLD_E12="HGV_V3_revision_20260915/16_scalability/E12_scalability_v1_20260925_v6"
NEW_E12="HGV_V3_revision_20260915/16_scalability/E12_scalability_v2_gzipfix_20260926"

E5="HGV_V3_revision_20260915/09_baselines/E5_unified_baseline_v1_20260917"
E5_AUDIT="HGV_V3_revision_20260915/20_rebuttal/HGV_V3_E5_runtime_audit_v2_20260926"

LOGDIR="HGV_V3_revision_20260915/scripts_v3_20260926_e7_e12_final_fix_v1/logs"

export HGV_BASE="HGV_V3_revision_20260915"
export HGV_RUN_TAG=20260925_v6

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32
for setting in k5 k7
do
    N=$(find \
        "$NEW_E7/runs/$setting" \
        -type f \
        -name RUN_COMPLETE \
        | wc -l)

    if [ "$N" -ne 3 ]; then
        echo "[FAIL] $setting expected 3 RUN_COMPLETE but got $N"
        exit 2
    fi
done

# Validate that the old non-k checkpoints in the copied tree were not changed.
find "$NEW_E7/runs" \
    -type f \
    -name best_model.pt \
    ! -path '*/k5/*' \
    ! -path '*/k7/*' \
    -print0 \
| sort -z \
| xargs -0 sha256sum \
> "$NEW_E7/NON_K_MODEL_SHA256_AFTER.tsv"

if ! diff -u \
    "$NEW_E7/NON_K_MODEL_SHA256_BEFORE.tsv" \
    "$NEW_E7/NON_K_MODEL_SHA256_AFTER.tsv"
then
    echo "[FAIL] A non-k5/k7 checkpoint changed."
    exit 2
fi

echo "ONLY_K5_K7_CHECKPOINTS_NEW=PASS"

# Re-audit feature caches in case the original E7 runner attempted extraction.
python - <<'PY'
from pathlib import Path
import numpy as np

BASE = Path(
    "HGV_V3_revision_20260915"
)

E7 = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v2_gzipfix_20260925_v6"
)

for k, dim in [(5,512),(7,8192)]:

    p = (
        E7
        / f"features_k{k}"
        / f"canonical_k{k}_A2374.npy"
    )

    X = np.load(
        p,
        mmap_mode="r",
    )

    assert X.shape == (2374, dim)

    sums = np.asarray(
        X,
        dtype=np.float64,
    ).sum(
        axis=1
    )

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5,
    )

    assert np.all(
        np.count_nonzero(
            X,
            axis=1,
        )
        >
        0
    )

print(
    "POST_TRAIN_FEATURE_AUDIT=PASS"
)
PY

# ======================================================================================
# 7. VALIDATION-ONLY THRESHOLD REFREEZE FOR ALL NON-CANONICAL E7 SETTINGS
# ======================================================================================

echo
echo "===== 7. E7 VALIDATION-ONLY THRESHOLD REFREEZE ====="

cat > "$FIX/07_finalize_e7_v2_validation_refreeze.py" <<'PY'
#!/usr/bin/env python3

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

BASE = Path(
    "HGV_V3_revision_20260915"
)

FIX = (
    BASE
    / "scripts_v3_20260926_e7_e12_final_fix_v1"
)

NEW_E7 = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v2_gzipfix_20260925_v6"
)

OLD_E7 = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v1_20260925_v6"
)

OUT = (
    NEW_E7
    / "final_validation_refreeze_v2_20260926"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ["HGV_BASE"] = str(BASE)
os.environ["HGV_RUN_TAG"] = "20260925_v6"

sys.path.insert(
    0,
    str(FIX)
)

import common as C

SEEDS = [
    20260915,
    20260916,
    20260917,
]

SETTINGS = {
    "k5": ("k5", 64),
    "k7": ("k7", 64),
    "dim32": ("k6", 32),
    "dim128": ("k6", 128),
    "margin0p5": ("k6", 64),
    "margin1p5": ("k6", 64),
    "lambda0p1": ("k6", 64),
    "lambda0p5": ("k6", 64),
}

CANONICAL = (
    "canonical_k6_dim64_margin1_lambda0p25"
)

def pair_indices(
    df,
    idx,
):

    aa = np.asarray(
        [
            idx[str(a)]
            for a
            in df["accession_a"]
        ],
        dtype=np.int64,
    )

    bb = np.asarray(
        [
            idx[str(a)]
            for a
            in df["accession_b"]
        ],
        dtype=np.int64,
    )

    y = pd.to_numeric(
        df["label"]
    ).to_numpy(
        dtype=np.int64
    )

    return (
        aa,
        bb,
        y,
    )

def metric_set(
    y,
    score,
    threshold,
):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    pred = (
        score >= threshold
    ).astype(
        np.int64
    )

    tp = int(
        np.sum(
            (y == 1)
            &
            (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y == 0)
            &
            (pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            &
            (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            &
            (pred == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else float("nan")
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else float("nan")
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (precision + recall)
        if (
            np.isfinite(precision)
            and
            np.isfinite(recall)
            and
            precision + recall > 0
        )
        else float("nan")
    )

    bacc = (
        (
            recall
            +
            specificity
        )
        /
        2.0
        if (
            np.isfinite(recall)
            and
            np.isfinite(specificity)
        )
        else float("nan")
    )

    den = math.sqrt(
        (tp + fp)
        *
        (tp + fn)
        *
        (tn + fp)
        *
        (tn + fn)
    )

    mcc = (
        (tp * tn - fp * fn)
        /
        den
        if den
        else float("nan")
    )

    return {
        "AUROC":
            float(
                roc_auc_score(
                    y,
                    score,
                )
            ),

        "AUPRC":
            float(
                average_precision_score(
                    y,
                    score,
                )
            ),

        "MCC":
            float(mcc),

        "F1":
            float(f1),

        "BalancedAccuracy":
            float(bacc),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "Specificity":
            float(specificity),

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "predicted_positive":
            int(
                pred.sum()
            ),
    }

def finite_value(x):

    if np.isfinite(x):
        return float(x)

    return -float("inf")

def freeze_threshold(
    y,
    s,
):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    s = np.asarray(
        s,
        dtype=np.float64,
    )

    order = np.argsort(
        -s,
        kind="mergesort",
    )

    yy = y[order]
    ss = s[order]

    P = int(
        np.sum(
            yy == 1
        )
    )

    N = int(
        np.sum(
            yy == 0
        )
    )

    tp = 0
    fp = 0

    best = None

    i = 0

    while i < len(ss):

        current = ss[i]
        j = i

        while (
            j + 1 < len(ss)
            and
            ss[j + 1] == current
        ):
            j += 1

        grp = yy[
            i:
            j + 1
        ]

        tp += int(
            np.sum(
                grp == 1
            )
        )

        fp += int(
            np.sum(
                grp == 0
            )
        )

        fn = P - tp
        tn = N - fp

        precision = (
            tp / (tp + fp)
            if tp + fp
            else float("nan")
        )

        recall = (
            tp / P
            if P
            else float("nan")
        )

        specificity = (
            tn / N
            if N
            else float("nan")
        )

        f1 = (
            2
            *
            precision
            *
            recall
            /
            (precision + recall)
            if (
                np.isfinite(precision)
                and
                np.isfinite(recall)
                and
                precision + recall > 0
            )
            else float("nan")
        )

        bacc = (
            (
                recall
                +
                specificity
            )
            /
            2.0
            if (
                np.isfinite(recall)
                and
                np.isfinite(specificity)
            )
            else float("nan")
        )

        den = math.sqrt(
            (tp + fp)
            *
            (tp + fn)
            *
            (tn + fp)
            *
            (tn + fn)
        )

        mcc = (
            (tp * tn - fp * fn)
            /
            den
            if den
            else float("nan")
        )

        if j + 1 < len(ss):

            threshold = float(
                (
                    current
                    +
                    ss[j + 1]
                )
                /
                2.0
            )

        else:

            threshold = float(
                np.nextafter(
                    current,
                    -np.inf,
                )
            )

        # Exact intended policy:
        # max MCC
        # -> higher BAcc
        # -> higher F1
        # -> closest to 0.5
        # -> lower threshold
        #
        # Undefined MCC/F1 must NEVER outrank finite candidates.
        key = (
            finite_value(
                mcc
            ),

            finite_value(
                bacc
            ),

            finite_value(
                f1
            ),

            -abs(
                threshold
                -
                0.5
            ),

            -threshold,
        )

        if (
            best is None
            or
            key > best[0]
        ):

            best = (
                key,
                threshold,
            )

        i = j + 1

    if best is None:
        raise SystemExit(
            "No validation threshold candidate"
        )

    threshold = float(
        best[1]
    )

    return (
        threshold,
        metric_set(
            y,
            s,
            threshold,
        ),
    )

def infer_scores(
    model,
    X,
    aa,
    bb,
    device,
):

    torch, _, _ = (
        C.import_torch()
    )

    parts = []

    model.eval()

    with torch.no_grad():

        for st in range(
            0,
            len(aa),
            1024,
        ):

            ai = aa[
                st:
                st + 1024
            ]

            bi = bb[
                st:
                st + 1024
            ]

            A = np.asarray(
                X[ai],
                dtype=np.float32,
            )

            B = np.asarray(
                X[bi],
                dtype=np.float32,
            )

            xx = np.concatenate(
                [
                    A,
                    B,
                ],
                axis=0,
            )

            z = model(
                torch.as_tensor(
                    xx,
                    dtype=torch.float32,
                    device=device,
                )
            )

            n = len(ai)

            za = (
                z[:n]
                .detach()
                .cpu()
                .numpy()
            )

            zb = (
                z[n:]
                .detach()
                .cpu()
                .numpy()
            )

            d = np.linalg.norm(
                za - zb,
                axis=1,
            )

            s = (
                1.0
                -
                d / 2.0
            )

            parts.append(
                s
            )

    return np.concatenate(
        parts
    )

manifest = C.load_manifest()
mc = C.manifest_columns(
    manifest
)

meta = manifest.set_index(
    mc["accession"]
)

A = sorted([
    str(acc)
    for acc, row
    in meta.iterrows()
    if C.infer_cohort(
        row,
        mc,
        accession=acc,
    )
    ==
    "A"
])

assert len(A) == 2374

# k6 frozen
X6_all, x6idx, _, _ = (
    C.feature_accession_index()
)

X6 = np.asarray(
    X6_all[
        [
            x6idx[a]
            for a
            in A
        ]
    ],
    dtype=np.float32,
)

idx6 = {
    a: i
    for i, a
    in enumerate(A)
}

features = {
    "k6":
        (
            X6,
            idx6,
        )
}

for k in [5, 7]:

    d = (
        NEW_E7
        /
        f"features_k{k}"
    )

    X = np.load(
        d
        /
        f"canonical_k{k}_A2374.npy",
        mmap_mode="r",
    )

    order = pd.read_csv(
        d
        /
        "accession_order.tsv",
        sep="\t",
    )

    idx = {
        str(a):
            i
        for i, a
        in enumerate(
            order[
                "accession"
            ].astype(str)
        )
    }

    features[
        f"k{k}"
    ] = (
        X,
        idx,
    )

val = C.load_pairs(
    "G2",
    20260915,
    "val",
)

test = C.load_pairs(
    "G2",
    20260915,
    "test",
)

assert len(val) == 3874
assert len(test) == 3872

run_rows = []
threshold_rows = []

for setting, (
    feat_name,
    embed_dim,
) in SETTINGS.items():

    X, idx = (
        features[
            feat_name
        ]
    )

    va, vb, vy = (
        pair_indices(
            val,
            idx,
        )
    )

    ta, tb, ty = (
        pair_indices(
            test,
            idx,
        )
    )

    for seed in SEEDS:

        rdir = (
            NEW_E7
            /
            "runs"
            /
            setting
            /
            f"model{seed}"
        )

        ckpt = (
            rdir
            /
            "best_model.pt"
        )

        metrics_json = (
            rdir
            /
            "metrics.json"
        )

        if not (
            ckpt.is_file()
            and
            metrics_json.is_file()
            and
            (
                rdir
                /
                "RUN_COMPLETE"
            ).is_file()
        ):

            raise SystemExit(
                f"Incomplete E7 run: {rdir}"
            )

        old = json.loads(
            metrics_json.read_text(
                encoding="utf-8",
            )
        )

        input_dim = int(
            old[
                "input_dim"
            ]
        )

        model, device, _ = (
            C.load_model_checkpoint(
                ckpt,
                input_dim=input_dim,
                embed_dim=embed_dim,
                normalize_output=True,
            )
        )

        vscore = infer_scores(
            model,
            X,
            va,
            vb,
            device,
        )

        threshold, vmet = (
            freeze_threshold(
                vy,
                vscore,
            )
        )

        # Important: threshold is now frozen.
        # Only after the validation-derived threshold has been selected
        # do we calculate test metrics.
        tscore = infer_scores(
            model,
            X,
            ta,
            tb,
            device,
        )

        tmet = metric_set(
            ty,
            tscore,
            threshold,
        )

        old_threshold = float(
            old.get(
                "val_threshold",
                float("nan"),
            )
        )

        threshold_rows.append({
            "setting":
                setting,

            "model_seed":
                seed,

            "validation_n":
                len(vy),

            "validation_score_min":
                float(
                    np.min(
                        vscore
                    )
                ),

            "validation_score_max":
                float(
                    np.max(
                        vscore
                    )
                ),

            "old_stored_threshold":
                old_threshold,

            "corrected_validation_threshold":
                threshold,

            "old_threshold_above_validation_max":
                int(
                    np.isfinite(
                        old_threshold
                    )
                    and
                    old_threshold
                    >
                    float(
                        np.max(
                            vscore
                        )
                    )
                ),
        })

        row = {
            "setting":
                setting,

            "model_seed":
                seed,

            "input_dim":
                input_dim,

            "embed_dim":
                embed_dim,

            "margin":
                float(
                    old[
                        "margin"
                    ]
                ),

            "lambda_dist":
                float(
                    old[
                        "lambda_dist"
                    ]
                ),

            "validation_threshold":
                threshold,

            "val_AUROC":
                vmet[
                    "AUROC"
                ],

            "val_AUPRC":
                vmet[
                    "AUPRC"
                ],

            "val_MCC":
                vmet[
                    "MCC"
                ],

            "val_F1":
                vmet[
                    "F1"
                ],

            "val_BalancedAccuracy":
                vmet[
                    "BalancedAccuracy"
                ],

            "source":
                (
                    "existing checkpoint; "
                    "validation-only threshold refreeze; "
                    "no test tuning"
                ),
        }

        for key, value in tmet.items():
            row[
                f"test_{key}"
            ] = value

        run_rows.append(
            row
        )

runs = pd.DataFrame(
    run_rows
)

runs.to_csv(
    OUT
    /
    "E7_FINAL_RUN_METRICS.tsv",
    sep="\t",
    index=False,
)

pd.DataFrame(
    threshold_rows
).to_csv(
    OUT
    /
    "E7_VALIDATION_THRESHOLD_FREEZE.tsv",
    sep="\t",
    index=False,
)

summary_rows = []

# canonical baseline copied unchanged from accepted E7-v1
old_summary = pd.read_csv(
    OLD_E7
    /
    "E7_SENSITIVITY_SUMMARY.tsv",
    sep="\t",
)

canonical = old_summary[
    old_summary[
        "setting"
    ]
    .astype(str)
    ==
    CANONICAL
]

assert len(canonical) == 1

c = canonical.iloc[0]

canonical_row = {
    "setting":
        CANONICAL,

    "n_seeds":
        int(
            c[
                "n_seeds"
            ]
        ),

    "test_AUROC_mean":
        float(
            c[
                "test_AUROC_mean"
            ]
        ),

    "test_AUROC_sd":
        float(
            c[
                "test_AUROC_sd"
            ]
        ),

    "test_AUPRC_mean":
        float(
            c[
                "test_AUPRC_mean"
            ]
        ),

    "test_AUPRC_sd":
        float(
            c[
                "test_AUPRC_sd"
            ]
        ),

    "test_MCC_mean":
        float(
            c[
                "test_MCC_mean"
            ]
        ),

    "test_MCC_sd":
        float(
            c[
                "test_MCC_sd"
            ]
        ),

    "test_F1_mean":
        float(
            c[
                "test_F1_mean"
            ]
        ),

    "test_F1_sd":
        float(
            c[
                "test_F1_sd"
            ]
        ),

    "test_BalancedAccuracy_mean":
        float(
            c[
                "test_BalancedAccuracy_mean"
            ]
        ),

    "test_BalancedAccuracy_sd":
        float(
            c[
                "test_BalancedAccuracy_sd"
            ]
        ),

    "input_dim":
        int(
            c[
                "input_dim"
            ]
        ),

    "embed_dim":
        int(
            c[
                "embed_dim"
            ]
        ),

    "margin":
        float(
            c[
                "margin"
            ]
        ),

    "lambda_dist":
        float(
            c[
                "lambda_dist"
            ]
        ),

    "source":
        "accepted frozen canonical baseline",
}

summary_rows.append(
    canonical_row
)

for setting, g in runs.groupby(
    "setting",
    sort=False,
):

    row = {
        "setting":
            setting,

        "n_seeds":
            len(g),

        "input_dim":
            int(
                g.iloc[0][
                    "input_dim"
                ]
            ),

        "embed_dim":
            int(
                g.iloc[0][
                    "embed_dim"
                ]
            ),

        "margin":
            float(
                g.iloc[0][
                    "margin"
                ]
            ),

        "lambda_dist":
            float(
                g.iloc[0][
                    "lambda_dist"
                ]
            ),

        "source":
            (
                "validation-only threshold refreeze; "
                "existing checkpoint"
            ),
    }

    for metric in [
        "test_AUROC",
        "test_AUPRC",
        "test_MCC",
        "test_F1",
        "test_BalancedAccuracy",
    ]:

        v = pd.to_numeric(
            g[
                metric
            ],
            errors="coerce",
        )

        row[
            metric
            +
            "_mean"
        ] = float(
            v.mean()
        )

        row[
            metric
            +
            "_sd"
        ] = float(
            v.std(
                ddof=1
            )
        )

    summary_rows.append(
        row
    )

summary = pd.DataFrame(
    summary_rows
)

cb = summary[
    summary[
        "setting"
    ]
    ==
    CANONICAL
].iloc[0]

for metric in [
    "test_AUROC",
    "test_AUPRC",
    "test_MCC",
    "test_F1",
    "test_BalancedAccuracy",
]:

    summary[
        f"delta_{metric}_vs_canonical"
    ] = (
        summary[
            metric
            +
            "_mean"
        ]
        -
        float(
            cb[
                metric
                +
                "_mean"
            ]
        )
    )

summary.to_csv(
    OUT
    /
    "E7_FINAL_SENSITIVITY_SUMMARY.tsv",
    sep="\t",
    index=False,
)

# Technical feature gate for corrected k5/k7
for k, dim in [(5,512),(7,8192)]:

    X = np.load(
        NEW_E7
        /
        f"features_k{k}"
        /
        f"canonical_k{k}_A2374.npy",
        mmap_mode="r",
    )

    if X.shape != (2374, dim):
        raise SystemExit(
            f"k{k}: wrong feature shape"
        )

    sums = np.asarray(
        X,
        dtype=np.float64,
    ).sum(
        axis=1
    )

    if not np.allclose(
        sums,
        1.0,
        atol=1e-5,
    ):
        raise SystemExit(
            f"k{k}: non-normalized rows"
        )

    if np.any(
        np.count_nonzero(
            X,
            axis=1,
        )
        ==
        0
    ):
        raise SystemExit(
            f"k{k}: zero feature row"
        )

(OUT / "E7_FINAL_PASS.txt").write_text(
    "\n".join([
        "E7_V2_FINAL=PASS",
        "K5_K7_GZIP_FIX=PASS",
        "NON_K_SETTINGS_RETRAINED=NO",
        "K5_K7_NEW_TRAINING_RUNS=6",
        "THRESHOLD_SOURCE=VALIDATION_ONLY",
        "TEST_THRESHOLD_TUNING=NO",
        "OLD_E7_MODIFIED=NO",
    ])
    +
    "\n",
    encoding="utf-8",
)

print(
    "E7_V2_FINAL=PASS"
)

print(
    summary.to_string(
        index=False
    )
)
PY

python \
    "$FIX/07_finalize_e7_v2_validation_refreeze.py" \
2>&1 | tee "$LOGDIR/07_finalize_e7_v2.log"

# ======================================================================================
# 8. E12-v2 REAL GZIP FASTA BENCHMARK
# ======================================================================================

echo
echo "===== 8. E12-v2 GZIP-AWARE REAL FASTA BENCHMARK ====="

cat > "$FIX/08_run_e12_gzip_real_fasta_v2.py" <<'PY'
#!/usr/bin/env python3

import hashlib
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "HGV_V3_revision_20260915"
)

FIX = (
    BASE
    /
    "scripts_v3_20260926_e7_e12_final_fix_v1"
)

OUT = (
    BASE
    /
    "16_scalability"
    /
    "E12_scalability_v2_gzipfix_20260926"
)

OLD = (
    BASE
    /
    "16_scalability"
    /
    "E12_scalability_v1_20260925_v6"
)

EXE = (
    FIX
    /
    "canonical_kmer_gzip_batch_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SCRATCH = (
    OUT
    /
    "scratch"
)

SCRATCH.mkdir(
    parents=True,
    exist_ok=True,
)

os.environ["HGV_BASE"] = str(BASE)
os.environ["HGV_RUN_TAG"] = "20260925_v6"

sys.path.insert(
    0,
    str(FIX)
)

import common as C

torch, _, _ = (
    C.import_torch()
)

try:
    torch.set_num_threads(
        int(
            os.environ.get(
                "OMP_NUM_THREADS",
                "48",
            )
        )
    )
except Exception:
    pass

MODEL = (
    BASE
    /
    "08_main_generalization"
    /
    "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
    /
    "G2"
    /
    "split20260915"
    /
    "model20260915"
    /
    "best_model.pt"
)

if not MODEL.is_file():
    raise SystemExit(
        f"Canonical model missing: {MODEL}"
    )

model, _, _ = (
    C.load_model_checkpoint(
        MODEL,
        input_dim=2080,
        embed_dim=64,
        normalize_output=True,
    )
)

model = model.to(
    "cpu"
)

model.eval()

A_LIST = (
    FIX
    /
    "lists"
    /
    "A2374_fasta.tsv"
)

Q_LIST = (
    FIX
    /
    "lists"
    /
    "B1_Q50_fasta.tsv"
)

a = pd.read_csv(
    A_LIST,
    sep="\t",
    header=None,
    names=[
        "accession",
        "fasta_path",
    ],
)

q = pd.read_csv(
    Q_LIST,
    sep="\t",
    header=None,
    names=[
        "accession",
        "fasta_path",
    ],
)

assert len(a) == 2374
assert len(q) == 50

N_VALUES = [
    100,
    500,
    1000,
    2000,
    2374,
]

REPEATS = 3

def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda:
            f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()

def write_list(
    df,
    path,
):

    df.to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
    )

def timed_extract(
    list_path,
    raw_path,
    tag,
):

    time_file = (
        SCRATCH
        /
        f"{tag}.time.tsv"
    )

    cmd = [
        "/usr/bin/time",
        "-f",
        "%e\t%M",
        "-o",
        str(time_file),
        str(EXE),
        "--k",
        "6",
        "--list",
        str(list_path),
        "--out",
        str(raw_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    x = (
        time_file
        .read_text()
        .strip()
        .split("\t")
    )

    wall = float(
        x[0]
    )

    maxrss_kb = int(
        x[1]
    )

    return (
        wall,
        maxrss_kb,
    )

def infer(
    X,
):

    zparts = []

    t0 = time.perf_counter()

    with torch.no_grad():

        for st in range(
            0,
            len(X),
            512,
        ):

            xx = torch.as_tensor(
                np.asarray(
                    X[
                        st:
                        st + 512
                    ],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device="cpu",
            )

            z = model(
                xx
            )

            zparts.append(
                z.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

    wall = (
        time.perf_counter()
        -
        t0
    )

    return (
        np.concatenate(
            zparts,
            axis=0,
        ),
        wall,
    )

def exact_scan(
    Q,
    R,
):

    # Unit-vector cosine and Euclidean rankings are equivalent.
    t0 = time.perf_counter()

    sim = (
        Q
        @
        R.T
    )

    _ = np.argmax(
        sim,
        axis=1,
    )

    return (
        time.perf_counter()
        -
        t0
    )

rows = []

for N in N_VALUES:

    refs = a.iloc[
        :N
    ].copy()

    ref_list = (
        SCRATCH
        /
        f"refs_N{N}.tsv"
    )

    write_list(
        refs,
        ref_list,
    )

    for rep in range(
        1,
        REPEATS + 1,
    ):

        print(
            f"[E12-v2] N={N} repeat={rep}",
            flush=True,
        )

        ref_raw = (
            SCRATCH
            /
            f"refs_N{N}_r{rep}.raw"
        )

        q_raw = (
            SCRATCH
            /
            f"query_Q50_N{N}_r{rep}.raw"
        )

        (
            ref_feature_wall,
            ref_feature_rss,
        ) = timed_extract(
            ref_list,
            ref_raw,
            f"refs_N{N}_r{rep}",
        )

        (
            q_feature_wall,
            q_feature_rss,
        ) = timed_extract(
            Q_LIST,
            q_raw,
            f"query_N{N}_r{rep}",
        )

        refX = np.fromfile(
            ref_raw,
            dtype=np.float32,
        ).reshape(
            N,
            2080,
        )

        qX = np.fromfile(
            q_raw,
            dtype=np.float32,
        ).reshape(
            50,
            2080,
        )

        ref_sums = (
            refX
            .astype(
                np.float64
            )
            .sum(
                axis=1
            )
        )

        q_sums = (
            qX
            .astype(
                np.float64
            )
            .sum(
                axis=1
            )
        )

        if not np.allclose(
            ref_sums,
            1.0,
            atol=1e-5,
        ):
            raise SystemExit(
                f"E12 N={N} reference feature normalization failure"
            )

        if not np.allclose(
            q_sums,
            1.0,
            atol=1e-5,
        ):
            raise SystemExit(
                "E12 query feature normalization failure"
            )

        refZ, ref_infer_wall = (
            infer(
                refX
            )
        )

        qZ, q_infer_wall = (
            infer(
                qX
            )
        )

        cold = exact_scan(
            qZ,
            refZ,
        )

        warm_values = []

        for _ in range(5):
            warm_values.append(
                exact_scan(
                    qZ,
                    refZ,
                )
            )

        warm_median = float(
            np.median(
                warm_values
            )
        )

        vector_bytes = int(
            refZ.nbytes
        )

        bytes_per_genome = (
            vector_bytes
            /
            N
        )

        python_rss = int(
            resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
        )

        total_wall = (
            ref_feature_wall
            +
            q_feature_wall
            +
            ref_infer_wall
            +
            q_infer_wall
            +
            cold
        )

        rows.append({
            "N_reference":
                N,

            "repeat":
                rep,

            "Q":
                50,

            "reference_cohort":
                "A",

            "query_cohort":
                "B1",

            "real_input_format":
                "gzip FASTA",

            "canonical_k":
                6,

            "input_dim":
                2080,

            "embedding_dim":
                64,

            "reference_feature_wall_s":
                ref_feature_wall,

            "query_feature_wall_s":
                q_feature_wall,

            "reference_feature_maxrss_kb":
                ref_feature_rss,

            "query_feature_maxrss_kb":
                q_feature_rss,

            "reference_model_inference_wall_s":
                ref_infer_wall,

            "query_model_inference_wall_s":
                q_infer_wall,

            "exact_scan_cold_s":
                cold,

            "exact_scan_warm_median_s":
                warm_median,

            "vector_index_bytes":
                vector_bytes,

            "bytes_per_genome":
                bytes_per_genome,

            "python_process_maxrss_kb":
                python_rss,

            "end_to_end_wall_s":
                total_wall,

            "model_checkpoint":
                str(MODEL),

            "model_sha256":
                sha256(
                    MODEL
                ),
        })

        del (
            refX,
            qX,
            refZ,
            qZ,
        )

        ref_raw.unlink(
            missing_ok=True
        )

        q_raw.unlink(
            missing_ok=True
        )

runs = pd.DataFrame(
    rows
)

runs.to_csv(
    OUT
    /
    "E12_V2_REAL_GZIP_FASTA_RUNS.tsv",
    sep="\t",
    index=False,
)

summary_rows = []

for N, g in runs.groupby(
    "N_reference",
    sort=True,
):

    row = {
        "N_reference":
            N,

        "repeats":
            len(g),

        "Q":
            50,
    }

    for col in [
        "reference_feature_wall_s",
        "query_feature_wall_s",
        "reference_feature_maxrss_kb",
        "query_feature_maxrss_kb",
        "reference_model_inference_wall_s",
        "query_model_inference_wall_s",
        "exact_scan_cold_s",
        "exact_scan_warm_median_s",
        "vector_index_bytes",
        "bytes_per_genome",
        "python_process_maxrss_kb",
        "end_to_end_wall_s",
    ]:

        vals = pd.to_numeric(
            g[col],
            errors="coerce",
        )

        row[
            col
            +
            "_median"
        ] = float(
            vals.median()
        )

        row[
            col
            +
            "_min"
        ] = float(
            vals.min()
        )

        row[
            col
            +
            "_max"
        ] = float(
            vals.max()
        )

    summary_rows.append(
        row
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT
    /
    "E12_V2_REAL_GZIP_FASTA_SUMMARY.tsv",
    sep="\t",
    index=False,
)

# Preserve the valid vector-search microbenchmark from v1 as an
# explicitly frozen copy. It does not depend on FASTA parsing.
old_vector = (
    OLD
    /
    "E12_VECTOR_MICROBENCHMARK.tsv"
)

if old_vector.is_file():

    import shutil

    shutil.copy2(
        old_vector,
        OUT
        /
        "E12_VECTOR_MICROBENCHMARK_FROZEN_V1_COPY.tsv",
    )

(OUT / "E12_V1_REAL_FASTA_INVALIDATION_NOTE.txt").write_text(
    "\n".join([
        "E12_V1_REAL_FASTA_TIMING_STATUS=DO_NOT_USE",
        "REASON=final FASTA payloads are gzip-compressed but the v1 plain C++ extractor did not decompress gzip",
        "VECTOR_MICROBENCHMARK_V1_STATUS=RETAINABLE",
        "E12_V2_REAL_FASTA_INPUT=gzip FASTA via zlib-aware canonical-kmer extractor",
        "REFERENCE=A development cohort",
        "QUERY=B1 first 50 deterministic sorted accessions",
        "N=100,500,1000,2000,2374",
        "REPEATS=3",
        "TEST_TUNING=NOT_APPLICABLE",
    ])
    +
    "\n",
    encoding="utf-8",
)

(OUT / "E12_V2_PROTOCOL.txt").write_text(
    "\n".join([
        "E12_V2_PROTOCOL=FROZEN",
        "INPUT=real gzip FASTA",
        "REFERENCE_COHORT=A",
        "QUERY_COHORT=B1",
        "QUERY_COUNT=50",
        "REFERENCE_N=100,500,1000,2000,2374",
        "REPEATS=3",
        "FEATURE=canonical RC-invariant k=6 normalized profile",
        "FEATURE_DIM=2080",
        "MODEL=canonical frozen G2 D3 seed20260915",
        "MODEL_DEVICE=CPU",
        "EMBEDDING_DIM=64",
        "SCAN=exact 64-D vector scan",
        "BUILD_AND_QUERY_TIMES_SEPARATE=YES",
        "VECTOR_MICROBENCHMARK=retained separately from v1",
        "MILLION_REAL_GENOME_CLAIM=NO",
    ])
    +
    "\n",
    encoding="utf-8",
)

(OUT / "E12_V2_PASS.txt").write_text(
    "\n".join([
        "E12_V2_GZIP_REAL_FASTA=PASS",
        "REAL_FASTA_GZIP_AWARE=YES",
        "REFERENCE_MAX_N=2374",
        "Q=50",
        "REPEATS=3",
        "OLD_E12_MODIFIED=NO",
    ])
    +
    "\n",
    encoding="utf-8",
)

print(
    "E12_V2_GZIP_REAL_FASTA=PASS"
)

print(
    summary.to_string(
        index=False
    )
)
PY

python \
    "$FIX/08_run_e12_gzip_real_fasta_v2.py" \
2>&1 | tee "$LOGDIR/08_e12_v2.log"

# ======================================================================================
# 9. ORGANIZE EXISTING E5 RUNTIME EVIDENCE
#    READ-ONLY: NO NATIVE TOOL IS RERUN.
# ======================================================================================

echo
echo "===== 9. E5 EXISTING RUNTIME AUDIT ====="

cat > "$FIX/09_audit_existing_e5_runtime_v2.py" <<'PY'
#!/usr/bin/env python3

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE
    /
    "09_baselines"
    /
    "E5_unified_baseline_v1_20260917"
)

OUT = (
    BASE
    /
    "20_rebuttal"
    /
    "HGV_V3_E5_runtime_audit_v2_20260926"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

METHOD_PATTERNS = {
    "Mash":
        re.compile(
            r"\bmash\b",
            re.I,
        ),

    "Dashing2":
        re.compile(
            r"dashing2|\bdashing\b",
            re.I,
        ),

    "skani":
        re.compile(
            r"\bskani\b",
            re.I,
        ),

    "FastANI":
        re.compile(
            r"\bfastani\b",
            re.I,
        ),

    "HGV":
        re.compile(
            r"\bhgv\b",
            re.I,
        ),
}

WALL_PATTERNS = [
    re.compile(
        r'wall[_ ]seconds\s*["\']?\s*[:=]\s*([0-9eE.+-]+)',
        re.I,
    ),

    re.compile(
        r'"wall_seconds"\s*:\s*([0-9eE.+-]+)',
        re.I,
    ),

    re.compile(
        r'elapsed_seconds\s*["\']?\s*[:=]\s*([0-9eE.+-]+)',
        re.I,
    ),
]

def infer_method(path, text):

    s = (
        str(path)
        +
        "\n"
        +
        text[:200000]
    )

    hits = [
        m
        for m, pat
        in METHOD_PATTERNS.items()
        if pat.search(
            s
        )
    ]

    if len(hits) == 1:
        return hits[0]

    # Path priority
    lower = str(
        path
    ).lower()

    if "dashing" in lower:
        return "Dashing2"

    if "fastani" in lower:
        return "FastANI"

    if "skani" in lower:
        return "skani"

    if "mash" in lower:
        return "Mash"

    if "hgv" in lower:
        return "HGV"

    return (
        ",".join(
            hits
        )
        if hits
        else
        "UNKNOWN"
    )

def infer_scope(path):

    s = str(
        path
    ).lower()

    if "external" in s:
        return "external"

    if "validation" in s:
        return "validation"

    if "internal" in s:
        return "internal"

    if "g2_" in s:
        return "G2"

    if "g3_" in s:
        return "G3"

    if "g4_" in s:
        return "G4"

    return "unspecified"

records = []
candidate_files = []

for p in sorted(
    E5.rglob("*")
):

    if (
        not p.is_file()
        or
        p.stat().st_size
        >
        25
        *
        1024
        *
        1024
    ):
        continue

    suffix = (
        p.suffix.lower()
    )

    if suffix not in {
        ".txt",
        ".md",
        ".json",
        ".log",
        ".tsv",
        ".csv",
    }:
        continue

    text = ""

    try:
        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    method = infer_method(
        p,
        text,
    )

    scope = infer_scope(
        p
    )

    values = []

    for pat in WALL_PATTERNS:

        for m in pat.finditer(
            text
        ):

            try:
                values.append(
                    float(
                        m.group(1)
                    )
                )
            except Exception:
                pass

    if suffix in {
        ".tsv",
        ".csv",
    }:

        try:
            sep = (
                "\t"
                if suffix == ".tsv"
                else ","
            )

            df = pd.read_csv(
                p,
                sep=sep,
            )

            for col in [
                "wall_seconds",
                "elapsed_seconds",
                "runtime_seconds",
                "wall_s",
            ]:

                if col in df.columns:

                    vals = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    ).dropna()

                    values.extend(
                        vals.astype(
                            float
                        ).tolist()
                    )

        except Exception:
            pass

    runtime_like = (
        bool(values)
        or
        re.search(
            r"runtime|wall|elapsed|rss|memory|throughput",
            p.name,
            re.I,
        )
        is not None
    )

    if not runtime_like:
        continue

    candidate_files.append({
        "method":
            method,

        "scope":
            scope,

        "path":
            str(
                p.relative_to(
                    BASE
                )
            ),

        "numeric_runtime_values_found":
            len(values),

        "size_bytes":
            p.stat().st_size,
    })

    for value in values:

        if (
            np.isfinite(
                value
            )
            and
            value >= 0
        ):

            records.append({
                "method":
                    method,

                "scope":
                    scope,

                "path":
                    str(
                        p.relative_to(
                            BASE
                        )
                    ),

                "wall_or_elapsed_seconds":
                    value,
            })

files_df = pd.DataFrame(
    candidate_files
)

records_df = pd.DataFrame(
    records
)

files_df.to_csv(
    OUT
    /
    "E5_RUNTIME_CANDIDATE_FILES.tsv",
    sep="\t",
    index=False,
)

records_df.to_csv(
    OUT
    /
    "E5_RUNTIME_NUMERIC_RECORDS.tsv",
    sep="\t",
    index=False,
)

summary_rows = []

if not records_df.empty:

    for (
        method,
        scope,
    ), g in records_df.groupby(
        [
            "method",
            "scope",
        ],
        sort=True,
    ):

        v = pd.to_numeric(
            g[
                "wall_or_elapsed_seconds"
            ],
            errors="coerce",
        ).dropna()

        summary_rows.append({
            "method":
                method,

            "scope":
                scope,

            "n_numeric_records":
                len(v),

            "wall_seconds_median":
                float(
                    v.median()
                ),

            "wall_seconds_min":
                float(
                    v.min()
                ),

            "wall_seconds_max":
                float(
                    v.max()
                ),

            "n_source_files":
                int(
                    g[
                        "path"
                    ]
                    .nunique()
                ),
        })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT
    /
    "E5_RUNTIME_METHOD_SCOPE_SUMMARY.tsv",
    sep="\t",
    index=False,
)

method_file_counts = {}

if not files_df.empty:

    for method in [
        "Mash",
        "Dashing2",
        "skani",
        "FastANI",
        "HGV",
    ]:

        method_file_counts[
            method
        ] = int(
            np.sum(
                files_df[
                    "method"
                ]
                ==
                method
            )
        )

else:

    for method in [
        "Mash",
        "Dashing2",
        "skani",
        "FastANI",
        "HGV",
    ]:
        method_file_counts[
            method
        ] = 0

(OUT / "E5_RUNTIME_SCOPE_DECISION.txt").write_text(
    "\n".join([
        "E5_RUNTIME_AUDIT=PASS",
        "NATIVE_TOOLS_RERUN=NO",
        "SOURCE=existing frozen E5 files only",
        "FAIR_CROSS_METHOD_RUNTIME_TABLE=REQUIRES_MANUAL_SCOPE_ALIGNMENT",
        "DO_NOT_COMPARE_UNMATCHED_INTERNAL_EXTERNAL_OR_BUILD_QUERY_TIMES=YES",
        f"METHOD_FILE_COUNTS={json.dumps(method_file_counts, sort_keys=True)}",
    ])
    +
    "\n",
    encoding="utf-8",
)

(OUT / "E5_RUNTIME_AUDIT_PASS.txt").write_text(
    "E5_RUNTIME_AUDIT=PASS\n"
    "NATIVE_TOOLS_RERUN=NO\n",
    encoding="utf-8",
)

print(
    "E5_RUNTIME_AUDIT=PASS"
)

print(
    "METHOD_FILE_COUNTS="
    +
    json.dumps(
        method_file_counts,
        sort_keys=True,
    )
)

if not summary_df.empty:
    print(
        summary_df.to_string(
            index=False
        )
    )
PY

python \
    "$FIX/09_audit_existing_e5_runtime_v2.py" \
2>&1 | tee "$LOGDIR/09_e5_runtime_audit.log"

# ======================================================================================
# 10. FINAL FREEZE / GATE
# ======================================================================================

echo
echo "===== 10. FINAL CORRECTION GATE ====="

for x in \
    "$E7_FINAL/E7_FINAL_PASS.txt" \
    "$E7_FINAL/E7_FINAL_SENSITIVITY_SUMMARY.tsv" \
    "$NEW_E12/E12_V2_PASS.txt" \
    "$NEW_E12/E12_V2_REAL_GZIP_FASTA_SUMMARY.tsv" \
    "$E5_AUDIT/E5_RUNTIME_AUDIT_PASS.txt"
do
    if [ ! -f "$x" ]; then
        echo "[FAIL] expected output missing: $x"
        exit 2
    fi
done

cat > "$FIX/HGV_V3_E7_E12_FINAL_FIX_STATUS.txt" <<EOF
HGV_V3_E7_E12_FINAL_FIX=PASS
FINISHED=$(date --iso-8601=seconds)

OLD_E7_MODIFIED=NO
OLD_E12_MODIFIED=NO
E5_NATIVE_TOOLS_RERUN=NO

E7_NEW_ROOT=$NEW_E7
E7_FINAL_ROOT=$E7_FINAL
E7_K5_K7_NEW_MODEL_RUNS=6
E7_NON_K_MODEL_RETRAINING=NO
E7_THRESHOLD_SOURCE=VALIDATION_ONLY

E12_NEW_ROOT=$NEW_E12
E12_REAL_FASTA_INPUT=GZIP_AWARE
E12_REFERENCE=A
E12_QUERY=B1_Q50
E12_MAX_REFERENCE_N=2374
E12_REPEATS=3
E12_V1_REAL_FASTA_TIMES=INVALIDATED
E12_V1_VECTOR_MICROBENCHMARK=RETAINED

E5_RUNTIME_AUDIT_ROOT=$E5_AUDIT
E5_RUNTIME_SOURCE=EXISTING_FROZEN_RESULTS_ONLY

NEXT_STEP=READ_ONLY_RESULT_REVIEW_THEN_MANUSCRIPT_TABLE_FIGURE_FREEZE
EOF

echo
echo "===================================================================================================="
echo "HGV_V3_E7_E12_FINAL_FIX=PASS"
echo "===================================================================================================="

echo
echo "E7 FINAL SUMMARY:"
column -t -s $'\t' \
    "$E7_FINAL/E7_FINAL_SENSITIVITY_SUMMARY.tsv"

echo
echo "E12-v2 FINAL SUMMARY:"
column -t -s $'\t' \
    "$NEW_E12/E12_V2_REAL_GZIP_FASTA_SUMMARY.tsv"

echo
echo "E5 RUNTIME SCOPE SUMMARY:"
if [ -s "$E5_AUDIT/E5_RUNTIME_METHOD_SCOPE_SUMMARY.tsv" ]; then
    column -t -s $'\t' \
        "$E5_AUDIT/E5_RUNTIME_METHOD_SCOPE_SUMMARY.tsv"
else
    echo "No numeric runtime rows summarized."
fi

echo
echo "FINAL STATUS:"
cat \
    "$FIX/HGV_V3_E7_E12_FINAL_FIX_STATUS.txt"

echo
echo "END=$(date --iso-8601=seconds)"
echo "===================================================================================================="
