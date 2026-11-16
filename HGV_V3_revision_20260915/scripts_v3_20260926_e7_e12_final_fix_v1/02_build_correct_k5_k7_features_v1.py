#!/usr/bin/env python3

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "HGV_V3_revision_20260915"
)

FIX = BASE / "scripts_v3_20260926_e7_e12_final_fix_v1"

E7 = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v2_gzipfix_20260925_v6"
)

EXE = FIX / "canonical_kmer_gzip_batch_v1"

LIST = FIX / "lists/A2374_fasta.tsv"

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

list_df = pd.read_csv(
    LIST,
    sep="\t",
    header=None,
    names=[
        "accession",
        "fasta_path",
    ],
)

assert len(list_df) == 2374

for k, dim in [
    (5, 512),
    (7, 8192),
]:

    out = E7 / f"features_k{k}"

    marker = out / "FEATURE_PASS.txt"

    if marker.is_file():

        X = np.load(
            out / f"canonical_k{k}_A2374.npy",
            mmap_mode="r",
        )

        sums = np.asarray(
            X,
            dtype=np.float64,
        ).sum(
            axis=1
        )

        if (
            X.shape
            ==
            (2374, dim)
            and
            np.allclose(
                sums,
                1.0,
                atol=1e-5,
            )
        ):

            print(
                f"k{k}: existing corrected feature PASS -> skip"
            )

            continue

        raise SystemExit(
            f"Existing k{k} FEATURE_PASS is inconsistent"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = (
        out
        / f"canonical_k{k}_A2374.raw"
    )

    cmd = [
        str(EXE),
        "--k",
        str(k),
        "--list",
        str(LIST),
        "--out",
        str(raw),
    ]

    print(
        "RUN",
        " ".join(cmd),
        flush=True,
    )

    subprocess.run(
        cmd,
        check=True,
    )

    expected = (
        2374
        *
        dim
        *
        4
    )

    actual = raw.stat().st_size

    if actual != expected:
        raise SystemExit(
            f"k{k} raw size mismatch: "
            f"{actual} != {expected}"
        )

    X = np.fromfile(
        raw,
        dtype=np.float32,
    ).reshape(
        2374,
        dim,
    )

    sums = X.astype(
        np.float64
    ).sum(
        axis=1
    )

    nnz = np.count_nonzero(
        X,
        axis=1,
    )

    if not np.all(
        np.isfinite(X)
    ):
        raise SystemExit(
            f"k{k}: nonfinite feature values"
        )

    if not np.allclose(
        sums,
        1.0,
        atol=1e-5,
    ):
        bad = int(
            np.sum(
                ~np.isclose(
                    sums,
                    1.0,
                    atol=1e-5,
                )
            )
        )

        raise SystemExit(
            f"k{k}: {bad} rows are not L1-normalized"
        )

    if np.any(
        nnz == 0
    ):
        raise SystemExit(
            f"k{k}: zero feature row detected"
        )

    npy = (
        out
        / f"canonical_k{k}_A2374.npy"
    )

    np.save(
        npy,
        X,
    )

    pd.DataFrame({
        "accession":
            list_df[
                "accession"
            ].astype(str)
    }).to_csv(
        out
        / "accession_order.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame([
        {
            "k":
                k,

            "n_genomes":
                2374,

            "dimension":
                dim,

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

            "zero_rows":
                int(
                    np.sum(
                        nnz == 0
                    )
                ),

            "nonzero_features_median":
                float(
                    np.median(
                        nnz
                    )
                ),

            "npy_sha256":
                sha256(
                    npy
                ),
        }
    ]).to_csv(
        out
        / "FEATURE_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    marker.write_text(
        "\n".join([
            "E7_GZIP_AWARE_FEATURE=PASS",
            f"K={k}",
            "N=2374",
            f"DIM={dim}",
            "ZERO_ROWS=0",
            "ALL_ROWS_L1_NORMALIZED=YES",
            f"NPY_SHA256={sha256(npy)}",
        ])
        +
        "\n",
        encoding="utf-8",
    )

    raw.unlink()

    print(
        f"k{k}: FEATURE_PASS"
    )

print(
    "K5_K7_FEATURE_FIX=PASS"
)
