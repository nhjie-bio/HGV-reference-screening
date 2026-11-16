#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_TAG = "20260925_v6"
MODEL_SEEDS = [20260915, 20260916, 20260917]

NONBASE_SETTINGS = [
    "k5",
    "k7",
    "dim32",
    "dim128",
    "margin0p5",
    "margin1p5",
    "lambda0p1",
    "lambda0p5",
]

CANONICAL = "canonical_k6_dim64_margin1_lambda0p25"


def die(msg):
    raise SystemExit(f"[FAIL] {msg}")


def rel(base, p):
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def detect_fasta(path):
    try:
        with path.open("rb") as f:
            h = f.read(16)
    except Exception:
        return "READ_ERROR"

    if h.startswith(b"\x1f\x8b"):
        return "GZIP"

    if h.lstrip().startswith(b">"):
        return "PLAIN_FASTA"

    return "OTHER"


def pair_indices(df, idx):
    try:
        a = np.asarray(
            [idx[str(x)] for x in df["accession_a"]],
            dtype=np.int64,
        )

        b = np.asarray(
            [idx[str(x)] for x in df["accession_b"]],
            dtype=np.int64,
        )

    except KeyError as e:
        die(f"pair accession not found in feature index: {e}")

    y = pd.to_numeric(df["label"]).to_numpy(np.int64)

    return a, b, y


def metrics(y, score, threshold):

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    pred = (
        score >= float(threshold)
    ).astype(np.int64)

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
        if (tp + fp)
        else float("nan")
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else float("nan")
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (
            np.isfinite(precision)
            and
            np.isfinite(recall)
            and
            (precision + recall) > 0
        )
        else float("nan")
    )

    bacc = (
        (recall + specificity) / 2
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
        / den
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
            int(pred.sum()),

        "predicted_negative":
            int(
                len(pred)
                -
                pred.sum()
            ),
    }


def finite_key(x):

    if np.isfinite(x):
        return float(x)

    return -float("inf")


def corrected_freeze_threshold(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    s = np.asarray(
        score,
        dtype=np.float64,
    )

    if (
        len(y) == 0
        or
        not np.isfinite(s).all()
        or
        set(
            np.unique(y).tolist()
        )
        !=
        {0, 1}
    ):
        die(
            "invalid validation arrays"
        )

    order = np.argsort(
        -s,
        kind="mergesort",
    )

    ss = s[order]

    candidates = [
        float(
            np.nextafter(
                ss[0],
                np.inf,
            )
        )
    ]

    i = 0

    while i < len(ss):

        current = ss[i]

        j = i

        while (
            j + 1 < len(ss)
            and
            ss[j + 1]
            ==
            current
        ):
            j += 1

        if j + 1 < len(ss):

            candidates.append(
                float(
                    (
                        current
                        +
                        ss[j + 1]
                    )
                    /
                    2.0
                )
            )

        else:

            candidates.append(
                float(
                    np.nextafter(
                        current,
                        -np.inf,
                    )
                )
            )

        i = j + 1

    best = None

    for t in candidates:

        m = metrics(
            y,
            s,
            t,
        )

        key = (
            finite_key(
                m["MCC"]
            ),

            finite_key(
                m[
                    "BalancedAccuracy"
                ]
            ),

            finite_key(
                m["F1"]
            ),

            -abs(
                t - 0.5
            ),

            -t,
        )

        if (
            best is None
            or
            key > best[0]
        ):

            best = (
                key,
                t,
                m,
            )

    return (
        best[1],
        best[2],
    )


def infer_pair_scores(
    C,
    model,
    X,
    aidx,
    bidx,
    device,
    batch=1024,
):

    torch, _, _ = (
        C.import_torch()
    )

    za_parts = []
    zb_parts = []

    model.eval()

    with torch.no_grad():

        for st in range(
            0,
            len(aidx),
            batch,
        ):

            ai = aidx[
                st:
                st + batch
            ]

            bi = bidx[
                st:
                st + batch
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

            za_parts.append(
                z[:n]
                .detach()
                .cpu()
                .numpy()
            )

            zb_parts.append(
                z[n:]
                .detach()
                .cpu()
                .numpy()
            )

    za = np.concatenate(
        za_parts,
        axis=0,
    )

    zb = np.concatenate(
        zb_parts,
        axis=0,
    )

    d = np.linalg.norm(
        za - zb,
        axis=1,
    )

    score = (
        1.0
        -
        d / 2.0
    )

    return (
        za,
        zb,
        d,
        score,
    )


def raw_scores(
    X,
    aidx,
    bidx,
    chunk=256,
):

    bc_all = []
    cos_all = []

    for st in range(
        0,
        len(aidx),
        chunk,
    ):

        ai = aidx[
            st:
            st + chunk
        ]

        bi = bidx[
            st:
            st + chunk
        ]

        A = np.asarray(
            X[ai],
            dtype=np.float64,
        )

        B = np.asarray(
            X[bi],
            dtype=np.float64,
        )

        den = np.sum(
            A + B,
            axis=1,
        )

        l1 = np.sum(
            np.abs(
                A - B
            ),
            axis=1,
        )

        bc_dist = np.divide(
            l1,
            den,
            out=np.ones_like(l1),
            where=den > 0,
        )

        bc_all.append(
            1.0
            -
            bc_dist
        )

        dot = np.sum(
            A * B,
            axis=1,
        )

        na = np.linalg.norm(
            A,
            axis=1,
        )

        nb = np.linalg.norm(
            B,
            axis=1,
        )

        cos_all.append(
            np.divide(
                dot,
                na * nb,
                out=np.zeros_like(dot),
                where=(
                    (na > 0)
                    &
                    (nb > 0)
                ),
            )
        )

    return (
        np.concatenate(
            bc_all
        ),
        np.concatenate(
            cos_all
        ),
    )


def feature_diag(
    name,
    X,
):

    n = X.shape[0]

    sums = []
    l2s = []
    nnz = []

    for st in range(
        0,
        n,
        256,
    ):

        A = np.asarray(
            X[
                st:
                st + 256
            ],
            dtype=np.float64,
        )

        sums.append(
            A.sum(
                axis=1
            )
        )

        l2s.append(
            np.linalg.norm(
                A,
                axis=1,
            )
        )

        nnz.append(
            np.count_nonzero(
                A,
                axis=1,
            )
        )

    sums = np.concatenate(
        sums
    )

    l2s = np.concatenate(
        l2s
    )

    nnz = np.concatenate(
        nnz
    )

    return {
        "feature_set":
            name,

        "rows":
            int(
                X.shape[0]
            ),

        "dim":
            int(
                X.shape[1]
            ),

        "row_sum_min":
            float(
                sums.min()
            ),

        "row_sum_median":
            float(
                np.median(
                    sums
                )
            ),

        "row_sum_max":
            float(
                sums.max()
            ),

        "rows_sum_zero":
            int(
                np.sum(
                    np.isclose(
                        sums,
                        0.0,
                        atol=1e-12,
                    )
                )
            ),

        "rows_sum_near1":
            int(
                np.sum(
                    np.isclose(
                        sums,
                        1.0,
                        atol=1e-5,
                    )
                )
            ),

        "nonzero_features_median":
            float(
                np.median(
                    nnz
                )
            ),

        "l2_median":
            float(
                np.median(
                    l2s
                )
            ),
    }


def scan_runtime_evidence(
    base,
    e5root,
):

    runtime_re = re.compile(
        r"(runtime|wall|elapsed|throughput|rss|ram|memory|peak.*mem|index.*size|bytes|seconds|time_s)",
        re.I,
    )

    methods = {
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

        "DNABERT-S":
            re.compile(
                r"dnabert",
                re.I,
            ),
    }

    rows = []

    if not e5root.is_dir():

        return pd.DataFrame(
            rows
        )

    for p in sorted(
        e5root.rglob("*")
    ):

        if (
            not p.is_file()
            or
            p.stat().st_size
            >
            100
            *
            1024
            *
            1024
        ):
            continue

        suffix = (
            p.suffix.lower()
        )

        signal = ""

        text_for_methods = str(
            p
        )

        runtime_cols = []

        try:

            if suffix in {
                ".tsv",
                ".csv",
            }:

                sep = (
                    "\t"
                    if suffix == ".tsv"
                    else ","
                )

                df = pd.read_csv(
                    p,
                    sep=sep,
                    nrows=100,
                )

                cols = [
                    str(c)
                    for c
                    in df.columns
                ]

                runtime_cols = [
                    c
                    for c
                    in cols
                    if runtime_re.search(
                        c
                    )
                ]

                text_for_methods += (
                    " "
                    +
                    " ".join(
                        cols
                    )
                )

                if len(df):

                    text_for_methods += (
                        " "
                        +
                        " ".join(
                            df.astype(str)
                            .head(20)
                            .fillna("")
                            .agg(
                                " ".join,
                                axis=1,
                            )
                        )
                    )

                if (
                    not runtime_cols
                    and
                    not runtime_re.search(
                        p.name
                    )
                ):
                    continue

                signal = (
                    "runtime_columns="
                    +
                    ",".join(
                        runtime_cols
                    )
                )

            elif (
                suffix
                in {
                    ".txt",
                    ".md",
                    ".json",
                    ".log",
                }
                and
                p.stat().st_size
                <=
                10
                *
                1024
                *
                1024
            ):

                txt = (
                    p.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )

                lines = [
                    x.strip()
                    for x
                    in txt.splitlines()
                    if runtime_re.search(
                        x
                    )
                ]

                if (
                    not lines
                    and
                    not runtime_re.search(
                        p.name
                    )
                ):
                    continue

                text_for_methods += (
                    " "
                    +
                    txt[:200000]
                )

                signal = (
                    " | ".join(
                        lines[:8]
                    )[:3000]
                )

            else:
                continue

        except Exception:
            continue

        hit_methods = [
            m
            for m, pat
            in methods.items()
            if pat.search(
                text_for_methods
            )
        ]

        rows.append({
            "path":
                rel(
                    base,
                    p,
                ),

            "runtime_columns":
                ",".join(
                    runtime_cols
                ),

            "matched_methods":
                ",".join(
                    hit_methods
                ),

            "signal":
                signal,

            "size_bytes":
                p.stat().st_size,
        })

    return pd.DataFrame(
        rows
    )


def md_table(
    df,
    cols=None,
    max_rows=40,
):

    if (
        df is None
        or
        df.empty
    ):
        return "_No rows._"

    x = df.copy()

    if cols:

        x = x[
            [
                c
                for c
                in cols
                if c
                in x.columns
            ]
        ]

    x = x.head(
        max_rows
    )

    lines = [
        "| "
        +
        " | ".join(
            map(
                str,
                x.columns,
            )
        )
        +
        " |",

        "| "
        +
        " | ".join(
            [
                "---"
            ]
            *
            len(
                x.columns
            )
        )
        +
        " |",
    ]

    for _, r in x.iterrows():

        vals = []

        for c in x.columns:

            v = r[c]

            if pd.isna(v):

                s = "NA"

            elif isinstance(
                v,
                (
                    float,
                    np.floating,
                ),
            ):

                s = (
                    f"{float(v):.6f}"
                )

            else:

                s = str(v)

            vals.append(
                s.replace(
                    "|",
                    "\\|",
                )
                .replace(
                    "\n",
                    " ",
                )
            )

        lines.append(
            "| "
            +
            " | ".join(
                vals
            )
            +
            " |"
        )

    return "\n".join(
        lines
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    base = (
        args.base.resolve()
    )

    pipe = (
        Path(__file__)
        .resolve()
        .parent
    )

    if not base.is_dir():

        die(
            f"BASE not found: {base}"
        )

    if not (
        pipe
        /
        "common.py"
    ).is_file():

        die(
            "common.py missing: run this script inside scripts_v3_20260926_remaining_all_v9"
        )

    os.environ[
        "HGV_BASE"
    ] = str(base)

    sys.path.insert(
        0,
        str(pipe),
    )

    C = importlib.import_module(
        "common"
    )

    out = (
        base
        /
        "20_rebuttal"
        /
        (
            "HGV_V3_write_readiness_audit_v1_"
            +
            datetime.now()
            .astimezone()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    e7 = (
        base
        /
        "11_sensitivity"
        /
        (
            "E7_sensitivity_v1_"
            +
            FINAL_TAG
        )
    )

    if not (
        e7
        /
        "E7_SENSITIVITY_PASS.txt"
    ).is_file():

        die(
            f"E7 PASS missing: {e7}"
        )

    manifest = (
        C.load_manifest()
    )

    mc = (
        C.manifest_columns(
            manifest
        )
    )

    fmap = (
        C.get_fasta_map(
            manifest
        )
    )

    meta = (
        manifest.set_index(
            mc["accession"]
        )
    )

    A = sorted(
        [
            str(a)
            for a, r
            in meta.iterrows()
            if C.infer_cohort(
                r,
                mc,
                accession=a,
            )
            ==
            "A"
        ]
    )

    if len(A) != 2374:

        die(
            f"A count != 2374: {len(A)}"
        )

    val = (
        C.load_pairs(
            "G2",
            20260915,
            "val",
        )
    )

    test = (
        C.load_pairs(
            "G2",
            20260915,
            "test",
        )
    )

    if (
        len(val) != 3874
        or
        len(test) != 3872
    ):

        die(
            "pair counts unexpected: "
            f"val={len(val)} "
            f"test={len(test)}"
        )

    # ==========================================================
    # FASTA storage format
    # ==========================================================

    fmt_rows = []

    for a in A:

        p = Path(
            fmap[a]
        )

        fmt_rows.append({
            "accession":
                a,

            "fasta_path":
                str(p),

            "suffix":
                "".join(
                    p.suffixes
                ),

            "detected_format":
                detect_fasta(
                    p
                ),

            "size_bytes":
                (
                    p.stat().st_size
                    if p.exists()
                    else -1
                ),
        })

    fmt_df = pd.DataFrame(
        fmt_rows
    )

    fmt_df.to_csv(
        out
        /
        "FASTA_STORAGE_FORMAT_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    fmt_counts = (
        fmt_df[
            "detected_format"
        ]
        .value_counts()
        .to_dict()
    )

    plain_n = int(
        fmt_counts.get(
            "PLAIN_FASTA",
            0,
        )
    )

    gzip_n = int(
        fmt_counts.get(
            "GZIP",
            0,
        )
    )

    other_n = (
        len(A)
        -
        plain_n
        -
        gzip_n
    )

    # ==========================================================
    # Feature caches
    # ==========================================================

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

    feature_map = {
        "k6_frozen":
            (
                X6,
                idx6,
            )
    }

    for k in [
        5,
        7,
    ]:

        d = (
            e7
            /
            f"features_k{k}"
        )

        npy = (
            d
            /
            f"canonical_k{k}_A2374.npy"
        )

        order = (
            d
            /
            "accession_order.tsv"
        )

        marker = (
            d
            /
            "FEATURE_PASS.txt"
        )

        if not (
            npy.is_file()
            and
            order.is_file()
            and
            marker.is_file()
        ):

            die(
                f"missing k{k} feature cache"
            )

        X = np.load(
            npy,
            mmap_mode="r",
        )

        odf = pd.read_csv(
            order,
            sep="\t",
        )

        idx = {
            str(a): i
            for i, a
            in enumerate(
                odf[
                    "accession"
                ]
                .astype(str)
            )
        }

        feature_map[
            f"k{k}"
        ] = (
            X,
            idx,
        )

    feat_diag = pd.DataFrame(
        [
            feature_diag(
                name,
                X,
            )
            for name, (
                X,
                _,
            )
            in feature_map.items()
        ]
    )

    feat_diag.to_csv(
        out
        /
        "E7_FEATURE_DIAGNOSTICS.tsv",
        sep="\t",
        index=False,
    )

    # ==========================================================
    # Raw feature discrimination
    # ==========================================================

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )

    raw_rows = []

    for name, (
        X,
        idx,
    ) in feature_map.items():

        aa, bb, y = (
            pair_indices(
                test,
                idx,
            )
        )

        bc, cos = (
            raw_scores(
                X,
                aa,
                bb,
            )
        )

        for method, s in [
            (
                "BrayCurtis_similarity",
                bc,
            ),
            (
                "Cosine_similarity",
                cos,
            ),
        ]:

            raw_rows.append({
                "feature_set":
                    name,

                "method":
                    method,

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

                "score_min":
                    float(
                        np.min(
                            s
                        )
                    ),

                "score_max":
                    float(
                        np.max(
                            s
                        )
                    ),

                "score_sd":
                    float(
                        np.std(
                            s
                        )
                    ),

                "positive_mean":
                    float(
                        np.mean(
                            s[
                                y == 1
                            ]
                        )
                    ),

                "negative_mean":
                    float(
                        np.mean(
                            s[
                                y == 0
                            ]
                        )
                    ),
            })

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_df.to_csv(
        out
        /
        "E7_RAW_FEATURE_DISCRIMINATION.tsv",
        sep="\t",
        index=False,
    )

    # ==========================================================
    # Existing E7 checkpoints: validation-only threshold fix
    # ==========================================================

    setting_meta = {
        "k5":
            (
                "k5",
                64,
            ),

        "k7":
            (
                "k7",
                64,
            ),

        "dim32":
            (
                "k6_frozen",
                32,
            ),

        "dim128":
            (
                "k6_frozen",
                128,
            ),

        "margin0p5":
            (
                "k6_frozen",
                64,
            ),

        "margin1p5":
            (
                "k6_frozen",
                64,
            ),

        "lambda0p1":
            (
                "k6_frozen",
                64,
            ),

        "lambda0p5":
            (
                "k6_frozen",
                64,
            ),
    }

    bug_rows = []
    corrected_rows = []
    score_diag_rows = []

    for setting in NONBASE_SETTINGS:

        feat_name, expected_embed = (
            setting_meta[
                setting
            ]
        )

        X, idx = (
            feature_map[
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

        for seed in MODEL_SEEDS:

            rdir = (
                e7
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

            hist_tsv = (
                rdir
                /
                "history.tsv"
            )

            if not (
                ckpt.is_file()
                and
                metrics_json.is_file()
                and
                hist_tsv.is_file()
            ):

                die(
                    f"missing E7 run files: {rdir}"
                )

            stored = json.loads(
                metrics_json.read_text(
                    encoding="utf-8"
                )
            )

            input_dim = int(
                stored[
                    "input_dim"
                ]
            )

            embed_dim = int(
                stored[
                    "embed_dim"
                ]
            )

            if (
                input_dim
                !=
                X.shape[1]
                or
                embed_dim
                !=
                expected_embed
            ):

                die(
                    "setting mismatch "
                    f"{setting} "
                    f"seed={seed}"
                )

            model, device, _ = (
                C.load_model_checkpoint(
                    ckpt,
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    normalize_output=True,
                )
            )

            _, _, vd, vs = (
                infer_pair_scores(
                    C,
                    model,
                    X,
                    va,
                    vb,
                    device,
                )
            )

            _, _, td, ts = (
                infer_pair_scores(
                    C,
                    model,
                    X,
                    ta,
                    tb,
                    device,
                )
            )

            old_th = float(
                stored[
                    "val_threshold"
                ]
            )

            (
                corrected_th,
                vmet,
            ) = (
                corrected_freeze_threshold(
                    vy,
                    vs,
                )
            )

            old_tmet = metrics(
                ty,
                ts,
                old_th,
            )

            new_tmet = metrics(
                ty,
                ts,
                corrected_th,
            )

            first_above_max = float(
                np.nextafter(
                    np.max(vs),
                    np.inf,
                )
            )

            bug_rows.append({
                "setting":
                    setting,

                "model_seed":
                    seed,

                "old_threshold":
                    old_th,

                "val_score_min":
                    float(
                        np.min(vs)
                    ),

                "val_score_max":
                    float(
                        np.max(vs)
                    ),

                "first_candidate_above_val_max":
                    first_above_max,

                "old_threshold_above_val_max":
                    int(
                        old_th
                        >
                        float(
                            np.max(vs)
                        )
                    ),

                "old_threshold_matches_first_candidate":
                    int(
                        np.isclose(
                            old_th,
                            first_above_max,
                            rtol=0,
                            atol=1e-12,
                        )
                    ),

                "corrected_threshold":
                    corrected_th,

                "old_test_predicted_positive":
                    old_tmet[
                        "predicted_positive"
                    ],

                "corrected_test_predicted_positive":
                    new_tmet[
                        "predicted_positive"
                    ],

                "old_test_MCC":
                    old_tmet[
                        "MCC"
                    ],

                "corrected_test_MCC":
                    new_tmet[
                        "MCC"
                    ],

                "old_test_F1":
                    old_tmet[
                        "F1"
                    ],

                "corrected_test_F1":
                    new_tmet[
                        "F1"
                    ],

                "old_test_BalancedAccuracy":
                    old_tmet[
                        "BalancedAccuracy"
                    ],

                "corrected_test_BalancedAccuracy":
                    new_tmet[
                        "BalancedAccuracy"
                    ],
            })

            hist = pd.read_csv(
                hist_tsv,
                sep="\t",
            )

            score_diag_rows.append({
                "setting":
                    setting,

                "model_seed":
                    seed,

                "val_score_sd":
                    float(
                        np.std(
                            vs
                        )
                    ),

                "val_positive_mean":
                    float(
                        np.mean(
                            vs[
                                vy == 1
                            ]
                        )
                    ),

                "val_negative_mean":
                    float(
                        np.mean(
                            vs[
                                vy == 0
                            ]
                        )
                    ),

                "test_score_sd":
                    float(
                        np.std(
                            ts
                        )
                    ),

                "test_positive_mean":
                    float(
                        np.mean(
                            ts[
                                ty == 1
                            ]
                        )
                    ),

                "test_negative_mean":
                    float(
                        np.mean(
                            ts[
                                ty == 0
                            ]
                        )
                    ),

                "best_history_val_AUROC":
                    float(
                        pd.to_numeric(
                            hist[
                                "val_AUROC"
                            ],
                            errors="coerce",
                        )
                        .max()
                    ),

                "best_history_val_AUPRC":
                    float(
                        pd.to_numeric(
                            hist[
                                "val_AUPRC"
                            ],
                            errors="coerce",
                        )
                        .max()
                    ),

                "best_epoch":
                    int(
                        stored[
                            "best_epoch"
                        ]
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
                        stored[
                            "margin"
                        ]
                    ),

                "lambda_dist":
                    float(
                        stored[
                            "lambda_dist"
                        ]
                    ),

                "old_threshold":
                    old_th,

                "corrected_threshold_validation_only":
                    corrected_th,

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
            }

            for k, v in new_tmet.items():

                row[
                    f"test_{k}"
                ] = v

            corrected_rows.append(
                row
            )

    bug_df = pd.DataFrame(
        bug_rows
    )

    corrected_runs = (
        pd.DataFrame(
            corrected_rows
        )
    )

    score_diag = pd.DataFrame(
        score_diag_rows
    )

    bug_df.to_csv(
        out
        /
        "E7_THRESHOLD_BUG_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    corrected_runs.to_csv(
        out
        /
        "E7_CORRECTED_RUN_METRICS_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    score_diag.to_csv(
        out
        /
        "E7_MODEL_SCORE_DIAGNOSTICS.tsv",
        sep="\t",
        index=False,
    )

    # ==========================================================
    # Corrected summary + canonical frozen baseline
    # ==========================================================

    old_summary = pd.read_csv(
        e7
        /
        "E7_SENSITIVITY_SUMMARY.tsv",
        sep="\t",
    )

    canon = old_summary[
        old_summary[
            "setting"
        ]
        .astype(str)
        ==
        CANONICAL
    ]

    if len(canon) != 1:

        die(
            "canonical E7 summary row not unique"
        )

    summary_rows = []

    c = canon.iloc[0]

    summary_rows.append({
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
            "frozen canonical E6 D3 baseline; unchanged",
    })

    for setting, g in (
        corrected_runs.groupby(
            "setting",
            sort=False,
        )
    ):

        row = {
            "setting":
                setting,

            "n_seeds":
                len(g),

            "source":
                "existing E7 checkpoints; validation-only threshold correction; no retraining",
        }

        for m in [
            "test_AUROC",
            "test_AUPRC",
            "test_MCC",
            "test_F1",
            "test_BalancedAccuracy",
        ]:

            vals = pd.to_numeric(
                g[m],
                errors="coerce",
            )

            row[
                m
                +
                "_mean"
            ] = float(
                vals.mean()
            )

            row[
                m
                +
                "_sd"
            ] = float(
                vals.std(
                    ddof=1
                )
            )

        for col in [
            "input_dim",
            "embed_dim",
            "margin",
            "lambda_dist",
        ]:

            row[col] = (
                g.iloc[0][col]
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    cb = (
        summary[
            summary[
                "setting"
            ]
            ==
            CANONICAL
        ]
        .iloc[0]
    )

    for m in [
        "test_AUROC",
        "test_AUPRC",
        "test_MCC",
        "test_F1",
        "test_BalancedAccuracy",
    ]:

        summary[
            f"delta_{m}_vs_canonical"
        ] = (
            summary[
                m
                +
                "_mean"
            ]
            -
            float(
                cb[
                    m
                    +
                    "_mean"
                ]
            )
        )

    summary.to_csv(
        out
        /
        "E7_CORRECTED_SUMMARY_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    # ==========================================================
    # k5 / k7 diagnosis
    # ==========================================================

    fd = (
        feat_diag.set_index(
            "feature_set"
        )
    )

    raw_idx = (
        raw_df.set_index(
            [
                "feature_set",
                "method",
            ]
        )
    )

    k5_cache_ok = (
        int(
            fd.loc[
                "k5",
                "rows_sum_zero",
            ]
        )
        ==
        0
        and
        int(
            fd.loc[
                "k5",
                "rows_sum_near1",
            ]
        )
        >=
        2370
    )

    k7_cache_ok = (
        int(
            fd.loc[
                "k7",
                "rows_sum_zero",
            ]
        )
        ==
        0
        and
        int(
            fd.loc[
                "k7",
                "rows_sum_near1",
            ]
        )
        >=
        2370
    )

    k5_raw_auc = float(
        raw_idx.loc[
            (
                "k5",
                "BrayCurtis_similarity",
            ),
            "AUROC",
        ]
    )

    k7_raw_auc = float(
        raw_idx.loc[
            (
                "k7",
                "BrayCurtis_similarity",
            ),
            "AUROC",
        ]
    )

    k5_model_auc = float(
        summary.loc[
            summary[
                "setting"
            ]
            ==
            "k5",
            "test_AUROC_mean",
        ]
        .iloc[0]
    )

    k7_model_auc = float(
        summary.loc[
            summary[
                "setting"
            ]
            ==
            "k7",
            "test_AUROC_mean",
        ]
        .iloc[0]
    )

    threshold_bug_fraction = float(
        bug_df[
            "old_threshold_above_val_max"
        ]
        .mean()
    )

    threshold_bug_confirmed = (
        threshold_bug_fraction
        >=
        0.95
    )

    extractor_compatible = (
        plain_n == 2374
        and
        gzip_n == 0
        and
        other_n == 0
    )

    if not extractor_compatible:

        kdiag = (
            "FASTA_FORMAT_RISK: final FASTA is not uniformly plain FASTA; "
            "k5/k7 C++ extractor cannot decompress gzip."
        )

    elif not (
        k5_cache_ok
        and
        k7_cache_ok
    ):

        kdiag = (
            "FEATURE_CACHE_INVALID: "
            "zero/non-normalized rows detected."
        )

    elif (
        (
            k5_raw_auc
            >=
            0.80
            or
            k7_raw_auc
            >=
            0.80
        )
        and
        (
            k5_model_auc
            <=
            0.60
            or
            k7_model_auc
            <=
            0.60
        )
    ):

        kdiag = (
            "RAW_FEATURES_INFORMATIVE_BUT_HGV_COLLAPSES: "
            "k5/k7 raw features discriminate, "
            "but trained embeddings are near-random."
        )

    else:

        kdiag = (
            "NO_SIMPLE_EXTRACTION_FAILURE: "
            "interpret raw-vs-model table."
        )

    # ==========================================================
    # Existing E5 runtime evidence
    # ==========================================================

    e5root = (
        base
        /
        "09_baselines"
        /
        "E5_unified_baseline_v1_20260917"
    )

    runtime_df = (
        scan_runtime_evidence(
            base,
            e5root,
        )
    )

    runtime_df.to_csv(
        out
        /
        "E5_RUNTIME_EVIDENCE_INVENTORY.tsv",
        sep="\t",
        index=False,
    )

    method_hits = {}

    for m in [
        "Mash",
        "Dashing2",
        "skani",
        "FastANI",
        "HGV",
    ]:

        if runtime_df.empty:

            method_hits[
                m
            ] = False

        else:

            method_hits[
                m
            ] = bool(
                runtime_df[
                    "matched_methods"
                ]
                .fillna("")
                .str.contains(
                    m,
                    regex=False,
                )
                .any()
            )

    native_candidate_complete = all(
        method_hits[m]
        for m
        in [
            "Mash",
            "Dashing2",
            "skani",
            "FastANI",
        ]
    )

    # ==========================================================
    # Readiness summary
    # ==========================================================

    readiness = pd.DataFrame([
        {
            "item":
                "E7 threshold evaluation",

            "status":
                (
                    "BUG_CONFIRMED_CORRECTABLE_NO_RETRAIN"
                    if threshold_bug_confirmed
                    else
                    "NEEDS_MANUAL_REVIEW"
                ),

            "detail":
                (
                    "fraction_old_threshold_above_val_max="
                    f"{threshold_bug_fraction:.3f}"
                ),
        },

        {
            "item":
                "E7 k5/k7 feature and model integrity",

            "status":
                "AUDITED",

            "detail":
                kdiag,
        },

        {
            "item":
                "E12 C++ FASTA extractor compatibility",

            "status":
                (
                    "COMPATIBLE"
                    if extractor_compatible
                    else
                    "INCOMPATIBLE_OR_AT_RISK"
                ),

            "detail":
                (
                    f"plain={plain_n}; "
                    f"gzip={gzip_n}; "
                    f"other={other_n}"
                ),
        },

        {
            "item":
                "Existing native runtime evidence",

            "status":
                (
                    "CANDIDATE_FILES_FOUND_FOR_ALL_NATIVE_METHODS"
                    if native_candidate_complete
                    else
                    "INCOMPLETE_OR_NOT_FOUND"
                ),

            "detail":
                json.dumps(
                    method_hits,
                    sort_keys=True,
                ),
        },

        {
            "item":
                "Core manuscript writing",

            "status":
                "READY_TO_START",

            "detail":
                (
                    "Stable blocks can be drafted now; "
                    "E7 and cross-method runtime claims "
                    "remain subject to this audit."
                ),
        },
    ])

    readiness.to_csv(
        out
        /
        "READINESS_DECISION.tsv",
        sep="\t",
        index=False,
    )

    # ==========================================================
    # Markdown report
    # ==========================================================

    report = []

    report += [
        "# HGV-V3 Write-Readiness Audit",
        "",

        (
            "Generated: `"
            +
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
            +
            "`"
        ),

        "",

        (
            "This audit used existing checkpoints "
            "and frozen outputs only. "
            "No retraining and no frozen-file modification were performed."
        ),

        "",

        "## 1. E7 threshold bug",
        "",

        (
            "- Fraction of non-canonical E7 runs "
            "whose stored validation threshold "
            "is above the maximum validation score: "
            f"**{threshold_bug_fraction:.3f}**"
        ),

        (
            "- Threshold bug confirmed: "
            f"**{threshold_bug_confirmed}**"
        ),

        "",

        "### Corrected E7 summary",
        "",

        md_table(
            summary,
            [
                "setting",
                "n_seeds",
                "test_AUROC_mean",
                "test_AUPRC_mean",
                "test_MCC_mean",
                "test_F1_mean",
                "test_BalancedAccuracy_mean",
                "delta_test_AUROC_vs_canonical",
                "delta_test_AUPRC_vs_canonical",
                "delta_test_MCC_vs_canonical",
            ],
        ),

        "",

        "## 2. k5/k7 integrity",
        "",

        (
            "- Final A FASTA storage: "
            f"plain={plain_n}, "
            f"gzip={gzip_n}, "
            f"other={other_n}"
        ),

        (
            "- E12/E7 plain C++ FASTA extractor compatible: "
            f"**{extractor_compatible}**"
        ),

        (
            "- k5 cache valid: "
            f"**{k5_cache_ok}**"
        ),

        (
            "- k7 cache valid: "
            f"**{k7_cache_ok}**"
        ),

        (
            "- k5 raw Bray-Curtis AUROC: "
            f"**{k5_raw_auc:.6f}**"
        ),

        (
            "- k7 raw Bray-Curtis AUROC: "
            f"**{k7_raw_auc:.6f}**"
        ),

        (
            "- k5 HGV AUROC: "
            f"**{k5_model_auc:.6f}**"
        ),

        (
            "- k7 HGV AUROC: "
            f"**{k7_model_auc:.6f}**"
        ),

        (
            "- Diagnosis: "
            f"**{kdiag}**"
        ),

        "",

        "### Feature diagnostics",
        "",

        md_table(
            feat_diag
        ),

        "",

        "### Raw feature discrimination",
        "",

        md_table(
            raw_df
        ),

        "",

        "## 3. Existing E5 runtime/memory evidence inventory",
        "",

        (
            "- Candidate files found: "
            f"**{len(runtime_df)}**"
        ),

        (
            "- Method hits: `"
            +
            json.dumps(
                method_hits,
                sort_keys=True,
            )
            +
            "`"
        ),

        (
            "- Candidate presence does not by itself prove "
            "a fair matched runtime comparison; "
            "scope must be checked before manuscript claims."
        ),

        "",

        md_table(
            runtime_df,
            [
                "path",
                "runtime_columns",
                "matched_methods",
                "signal",
            ],
            40,
        ),

        "",

        "## 4. Readiness",
        "",

        md_table(
            readiness
        ),

        "",

        "HGV_V3_WRITE_READINESS_AUDIT=PASS",
        "NO_RETRAINING_PERFORMED=YES",
        "NO_FROZEN_INPUT_MODIFIED=YES",
    ]

    (
        out
        /
        "WRITE_READINESS_REPORT.md"
    ).write_text(
        "\n".join(
            report
        )
        +
        "\n",
        encoding="utf-8",
    )

    (
        out
        /
        "AUDIT_PASS.txt"
    ).write_text(
        "HGV_V3_WRITE_READINESS_AUDIT=PASS\n"
        "NO_RETRAINING_PERFORMED=YES\n"
        "NO_FROZEN_INPUT_MODIFIED=YES\n",
        encoding="utf-8",
    )

    print(
        "=" * 100
    )

    print(
        "HGV-V3 WRITE-READINESS AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        f"OUT={out}"
    )

    print(
        "E7_THRESHOLD_BUG_CONFIRMED="
        f"{threshold_bug_confirmed}"
    )

    print(
        "E7_THRESHOLD_BUG_FRACTION="
        f"{threshold_bug_fraction:.3f}"
    )

    print(
        f"FASTA_PLAIN={plain_n}"
    )

    print(
        f"FASTA_GZIP={gzip_n}"
    )

    print(
        f"FASTA_OTHER={other_n}"
    )

    print(
        "E12_CPP_EXTRACTOR_COMPATIBLE="
        f"{extractor_compatible}"
    )

    print(
        "K5_FEATURE_CACHE_VALID="
        f"{k5_cache_ok}"
    )

    print(
        "K7_FEATURE_CACHE_VALID="
        f"{k7_cache_ok}"
    )

    print(
        "K5_RAW_BRAY_AUROC="
        f"{k5_raw_auc:.6f}"
    )

    print(
        "K7_RAW_BRAY_AUROC="
        f"{k7_raw_auc:.6f}"
    )

    print(
        "K5_HGV_AUROC="
        f"{k5_model_auc:.6f}"
    )

    print(
        "K7_HGV_AUROC="
        f"{k7_model_auc:.6f}"
    )

    print(
        "E5_RUNTIME_CANDIDATE_FILES="
        f"{len(runtime_df)}"
    )

    print(
        "E5_NATIVE_RUNTIME_CANDIDATE_COMPLETE="
        f"{native_candidate_complete}"
    )

    print(
        "CORE_MANUSCRIPT_WRITING_READY=True"
    )

    print(
        "HGV_V3_WRITE_READINESS_AUDIT=PASS"
    )

    print(
        "NO_RETRAINING_PERFORMED=YES"
    )

    print(
        "NO_FROZEN_INPUT_MODIFIED=YES"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
