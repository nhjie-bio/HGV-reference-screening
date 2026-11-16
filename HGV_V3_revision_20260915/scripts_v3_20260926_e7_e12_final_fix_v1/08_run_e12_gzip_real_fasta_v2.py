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
