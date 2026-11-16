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
