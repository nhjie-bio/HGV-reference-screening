#!/usr/bin/env python3

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from hgv_common import (
    extract_acc,
    parse_dashing_rectangular,
    parse_dashing_allpairs,
    read_list,
    run_logged,
    sha256_file,
)


BASE = Path(
    os.environ.get(
        "HGV_BASE",
        "HGV_V3_revision_20260915",
    )
).resolve()

THREADS = int(
    os.environ.get(
        "HGV_THREADS",
        "88",
    )
)

FASTANI_WORKERS = int(
    os.environ.get(
        "HGV_FASTANI_WORKERS",
        "44",
    )
)

FREEZE = (
    BASE /
    "09_baselines/E5_external_unified_v1_20260920/"
    "00_task_freeze"
)

OUT = (
    BASE /
    "09_baselines/E5_external_unified_v1_20260920/"
    "02_native_external_full_v1"
)

RAW = OUT / "raw"
SRC = OUT / "source_data"
LOG = OUT / "logs"

for d in [OUT, RAW, SRC, LOG]:
    d.mkdir(parents=True, exist_ok=True)


def fail(msg):
    raise RuntimeError(msg)


def tool(name):
    p = shutil.which(name)
    if not p:
        fail(f"Required tool not found in PATH: {name}")
    return p


def save_runtime(rows):
    pd.DataFrame(rows).to_csv(
        SRC / "E5_EXTERNAL_RUNTIME.tsv",
        sep="\t",
        index=False,
    )


# ======================================================================
# Frozen task
# ======================================================================

REF_TSV = FREEZE / "REFERENCE_A_2374.tsv"
REF_LIST = FREEZE / "REFERENCE_A_2374_FASTA.list"

QFILES = [
    (
        "B1",
        FREEZE / "QUERY_B1_391.tsv",
        FREEZE / "QUERY_B1_391_FASTA.list",
    ),
    (
        "B2_seen_genus",
        FREEZE / "QUERY_B2_SEEN_196.tsv",
        FREEZE / "QUERY_B2_SEEN_196_FASTA.list",
    ),
    (
        "B2_unseen_genus",
        FREEZE / "QUERY_B2_UNSEEN_193.tsv",
        FREEZE / "QUERY_B2_UNSEEN_193_FASTA.list",
    ),
]

required = [
    REF_TSV,
    REF_LIST,
    FREEZE / "EXTERNAL_TASK_FREEZE.txt",
    FREEZE / "EXTERNAL_TASK_SHA256.txt",
]

for _, t, l in QFILES:
    required += [t, l]

for p in required:
    if not p.is_file():
        fail(f"Missing frozen E5 external input: {p}")


ref = pd.read_csv(
    REF_TSV,
    sep="\t",
    dtype=str,
)

if len(ref) != 2374:
    fail(f"Expected 2374 references, found {len(ref)}")

if "assembly_accession" not in ref.columns:
    fail("REFERENCE_A_2374.tsv lacks assembly_accession")

ref_acc = ref["assembly_accession"].astype(str).tolist()

if len(set(ref_acc)) != 2374:
    fail("Reference accessions are not unique")

ref_fasta = read_list(REF_LIST)

if len(ref_fasta) != 2374:
    fail("Reference FASTA list count mismatch")

if any(not Path(x).is_file() for x in ref_fasta):
    fail("At least one reference FASTA is missing")


qparts = []

for cohort, tsv, flist in QFILES:
    q = pd.read_csv(
        tsv,
        sep="\t",
        dtype=str,
    )

    expected = {
        "B1": 391,
        "B2_seen_genus": 196,
        "B2_unseen_genus": 193,
    }[cohort]

    if len(q) != expected:
        fail(
            f"{cohort}: expected {expected} queries, found {len(q)}"
        )

    paths = read_list(flist)

    if len(paths) != expected:
        fail(
            f"{cohort}: FASTA count mismatch"
        )

    q = q.copy()
    q["cohort"] = cohort

    qparts.append(q)

query = pd.concat(
    qparts,
    ignore_index=True,
)

if len(query) != 780:
    fail(f"Expected 780 queries, found {len(query)}")

if query["query_accession"].nunique() != 780:
    fail("External query accessions are not unique")

query_acc = query["query_accession"].astype(str).tolist()
query_fasta = query["fasta_path"].astype(str).tolist()

if any(not Path(x).is_file() for x in query_fasta):
    fail("At least one external query FASTA is missing")

if set(query_acc) & set(ref_acc):
    fail("External query/reference accession overlap detected")


ALL_QUERY_LIST = OUT / "EXTERNAL_ALL_780_FASTA.list"

ALL_QUERY_LIST.write_text(
    "".join(
        x + "\n"
        for x in query_fasta
    )
)


# ======================================================================
# Identify taxonomy columns in A without guessing names.
# ======================================================================

def detect_tax_col(frame, query_values, tokens):
    qset = {
        str(x)
        for x in query_values
        if pd.notna(x)
        and str(x) not in {"", "nan", "None"}
    }

    best = None

    for c in frame.columns:
        vals = set(
            frame[c]
            .dropna()
            .astype(str)
            .tolist()
        )

        overlap = len(
            vals & qset
        )

        name_bonus = sum(
            int(t.lower() in c.lower())
            for t in tokens
        )

        candidate = (
            overlap,
            name_bonus,
            c,
        )

        if best is None or candidate > best:
            best = candidate

    if best is None or best[0] == 0:
        fail(
            f"Cannot identify taxonomy column for {tokens}; "
            f"best={best}"
        )

    return best[2]


species_col = detect_tax_col(
    ref,
    query["query_species"],
    ["species", "cluster", "representative"],
)

genus_col = detect_tax_col(
    ref,
    query["query_genus"],
    ["genus"],
)

family_col = detect_tax_col(
    ref,
    query["query_family"],
    ["family"],
)

print("Detected A taxonomy columns:")
print("  species:", species_col)
print("  genus  :", genus_col)
print("  family :", family_col)


ref_species = ref[species_col].astype(str).to_numpy()
ref_genus = ref[genus_col].astype(str).to_numpy()
ref_family = ref[family_col].astype(str).to_numpy()


# ======================================================================
# Tools
# ======================================================================

mash = tool("mash")
dashing2 = tool("dashing2")
fastani = tool("fastANI")
skani = tool("skani")

runtime_rows = []


def timed_command(name, stage, cmd, stdout, stderr, check=True):
    t0 = time.perf_counter()

    rc = run_logged(
        cmd,
        stdout,
        stderr,
        check=check,
    )

    dt = time.perf_counter() - t0

    runtime_rows.append({
        "method": name,
        "stage": stage,
        "wall_seconds": dt,
        "returncode": rc,
    })

    save_runtime(runtime_rows)

    return rc


# ======================================================================
# Mash
# ======================================================================

MASH_DIR = RAW / "mash"
MASH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

teacher_mash_root = (
    BASE /
    "06_teacher/teacher_cache_v1_20260916/mash"
)

a_msh_candidates = sorted(
    teacher_mash_root.rglob(
        "development_A_k21_s1000.msh"
    )
)

if len(a_msh_candidates) == 1:
    A_MSH = a_msh_candidates[0]
else:
    A_MSH = MASH_DIR / "development_A_k21_s1000.msh"

    if not A_MSH.is_file():
        prefix = A_MSH.with_suffix("")

        timed_command(
            "Mash",
            "reference_sketch",
            [
                mash,
                "sketch",
                "-p", str(THREADS),
                "-k", "21",
                "-s", "1000",
                "-l", str(REF_LIST),
                "-o", str(prefix),
            ],
            LOG / "mash_ref.stdout.txt",
            LOG / "mash_ref.stderr.txt",
        )

if not A_MSH.is_file():
    fail(f"Mash A sketch missing: {A_MSH}")


Q_MSH = MASH_DIR / "external_780_k21_s1000.msh"

if not Q_MSH.is_file():
    prefix = Q_MSH.with_suffix("")

    timed_command(
        "Mash",
        "query_sketch",
        [
            mash,
            "sketch",
            "-p", str(THREADS),
            "-k", "21",
            "-s", "1000",
            "-l", str(ALL_QUERY_LIST),
            "-o", str(prefix),
        ],
        LOG / "mash_query.stdout.txt",
        LOG / "mash_query.stderr.txt",
    )


MASH_RAW = MASH_DIR / "mash_A_vs_external.tsv"

if not MASH_RAW.is_file() or MASH_RAW.stat().st_size == 0:
    t0 = time.perf_counter()

    with MASH_RAW.open("w") as out, \
         (LOG / "mash_dist.stderr.txt").open("w") as err:

        p = subprocess.run(
            [
                mash,
                "dist",
                "-p", str(THREADS),
                str(A_MSH),
                str(Q_MSH),
            ],
            stdout=out,
            stderr=err,
        )

    runtime_rows.append({
        "method": "Mash",
        "stage": "retrieval",
        "wall_seconds": time.perf_counter() - t0,
        "returncode": p.returncode,
    })

    save_runtime(runtime_rows)

    if p.returncode != 0:
        fail("Mash external retrieval failed")


# ======================================================================
# Dashing2 SetSketch
# Exact production scientific parameters:
# k31, S1024, same SetSketch/default mode.
# No new flag is silently introduced here.
# ======================================================================

D2_DIR = RAW / "dashing2"
D2_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

D2_RECT = D2_DIR / "dashing2_external_rectangular.tsv"
D2_MODE = D2_DIR / "MODE.txt"

if not D2_RECT.is_file() or D2_RECT.stat().st_size == 0:

    rc = timed_command(
        "Dashing2",
        "rectangular_retrieval",
        [
            dashing2,
            "sketch",
            "-p", str(THREADS),
            "-k31",
            "-S1024",
            "--cmpout", str(D2_RECT),
            "-Q", str(ALL_QUERY_LIST),
            "-F", str(REF_LIST),
        ],
        LOG / "dashing2_rect.stdout.txt",
        LOG / "dashing2_rect.stderr.txt",
        check=False,
    )

    if (
        rc == 0
        and D2_RECT.is_file()
        and D2_RECT.stat().st_size > 0
    ):
        D2_MODE.write_text("RECTANGULAR\n")

    else:
        if D2_RECT.exists():
            D2_RECT.unlink()

        COMBINED = D2_DIR / "A_PLUS_EXTERNAL_3154.list"

        COMBINED.write_text(
            "".join(
                x + "\n"
                for x in ref_fasta + query_fasta
            )
        )

        D2_ALL = D2_DIR / "dashing2_all3154_similarity.tsv"

        timed_command(
            "Dashing2",
            "allpair_fallback",
            [
                dashing2,
                "sketch",
                "-p", str(THREADS),
                "-k31",
                "-S1024",
                "--cmpout", str(D2_ALL),
                "-F", str(COMBINED),
            ],
            LOG / "dashing2_all.stdout.txt",
            LOG / "dashing2_all.stderr.txt",
        )

        D2_MODE.write_text(
            "ALLPAIR_FALLBACK\n"
        )


# ======================================================================
# FastANI external query -> A reference
# Structural no-hit is preserved as missing.
# ======================================================================

FAST_DIR = RAW / "fastani"
FAST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def fastani_job(i):
    acc = query_acc[i]
    fasta = query_fasta[i]

    qdir = FAST_DIR / acc
    qdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = qdir / "fastani.tsv"
    done = qdir / "RUN_COMPLETE"

    if done.is_file():
        return {
            "query": acc,
            "returncode": 0,
            "status": "SKIP",
        }

    t0 = time.perf_counter()

    p = subprocess.run(
        [
            fastani,
            "-q", fasta,
            "--rl", str(REF_LIST),
            "-o", str(raw),
            "-t", "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    (qdir / "stdout.txt").write_text(
        p.stdout or ""
    )

    (qdir / "stderr.txt").write_text(
        p.stderr or ""
    )

    if p.returncode == 0:
        if not raw.exists():
            raw.touch()

        done.write_text(
            "RUN_COMPLETE\n"
            f"elapsed_seconds={time.perf_counter()-t0:.6f}\n"
        )

    return {
        "query": acc,
        "returncode": p.returncode,
        "status": "PASS" if p.returncode == 0 else "FAIL",
    }


t0 = time.perf_counter()

fast_fail = []

with ThreadPoolExecutor(
    max_workers=FASTANI_WORKERS
) as ex:

    futs = [
        ex.submit(
            fastani_job,
            i,
        )
        for i in range(len(query_acc))
    ]

    for fut in as_completed(futs):
        r = fut.result()

        if r["returncode"] != 0:
            fast_fail.append(r)

runtime_rows.append({
    "method": "FastANI",
    "stage": "retrieval_780_queries",
    "wall_seconds": time.perf_counter() - t0,
    "returncode": 0 if not fast_fail else 1,
})

save_runtime(runtime_rows)

if fast_fail:
    fail(
        f"FastANI failed jobs: {fast_fail[:10]}"
    )


# ======================================================================
# skani external rectangular query/reference
# ======================================================================

SKANI_DIR = RAW / "skani"
SKANI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SKANI_RAW = SKANI_DIR / "skani_external.tsv"

if not SKANI_RAW.is_file() or SKANI_RAW.stat().st_size == 0:
    timed_command(
        "skani",
        "retrieval",
        [
            skani,
            "dist",
            "-t", str(THREADS),
            "--ql", str(ALL_QUERY_LIST),
            "--rl", str(REF_LIST),
            "-o", str(SKANI_RAW),
        ],
        LOG / "skani.stdout.txt",
        LOG / "skani.stderr.txt",
    )


# ======================================================================
# Build score matrices
# ======================================================================

nq = len(query_acc)
nr = len(ref_acc)

qidx = {
    a: i
    for i, a in enumerate(query_acc)
}

ridx = {
    a: i
    for i, a in enumerate(ref_acc)
}


# Mash: higher score is better, so score=-distance.
M_mash = np.full(
    (nq, nr),
    np.nan,
    dtype=np.float32,
)

with MASH_RAW.open(errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")

        if len(p) < 3:
            p = line.split()

        if len(p) < 3:
            continue

        r = extract_acc(p[0])
        q = extract_acc(p[1])

        if r not in ridx or q not in qidx:
            continue

        try:
            dist = float(p[2])
        except Exception:
            continue

        M_mash[
            qidx[q],
            ridx[r],
        ] = -dist


# Dashing2.
mode = D2_MODE.read_text().strip()

if mode == "RECTANGULAR":
    M_d2 = parse_dashing_rectangular(
        D2_RECT,
        query_acc,
        ref_acc,
    )

elif mode == "ALLPAIR_FALLBACK":
    D2_ALL = D2_DIR / "dashing2_all3154_similarity.tsv"

    combined_acc = ref_acc + query_acc

    full = parse_dashing_allpairs(
        D2_ALL,
        combined_acc,
    )

    M_d2 = full[
        len(ref_acc):,
        :len(ref_acc),
    ]

else:
    fail(
        f"Unknown Dashing2 mode: {mode}"
    )


# FastANI.
M_fast = np.full(
    (nq, nr),
    np.nan,
    dtype=np.float32,
)

for q in query_acc:
    raw = FAST_DIR / q / "fastani.tsv"

    if not raw.exists():
        fail(f"Missing FastANI output: {raw}")

    with raw.open(errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")

            if len(p) < 3:
                continue

            qa = extract_acc(p[0])
            ra = extract_acc(p[1])

            if qa != q:
                continue

            if ra not in ridx:
                continue

            try:
                ani = float(p[2])
            except Exception:
                continue

            M_fast[
                qidx[q],
                ridx[ra],
            ] = ani


# skani.
M_skani = np.full(
    (nq, nr),
    np.nan,
    dtype=np.float32,
)

with SKANI_RAW.open(errors="replace") as f:
    for line in f:
        if line.startswith("Ref_file\t"):
            continue

        p = line.rstrip("\n").split("\t")

        if len(p) < 5:
            continue

        ra = extract_acc(p[0])
        qa = extract_acc(p[1])

        if (
            ra not in ridx
            or qa not in qidx
        ):
            continue

        try:
            ani = float(p[2])
        except Exception:
            continue

        M_skani[
            qidx[qa],
            ridx[ra],
        ] = ani


# DNABERT-S.
DNAROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "06_dnaberts/07_production_embeddings_v1_20260918"
)

DNA_MAT = (
    DNAROOT /
    "DNABERTS_GENOME_EMBEDDINGS_L2_FLOAT32.npy"
)

DNA_MAN = (
    DNAROOT /
    "DNABERTS_GENOME_EMBEDDING_MANIFEST.tsv"
)

if not DNA_MAT.is_file() or not DNA_MAN.is_file():
    fail("Frozen DNABERT-S production embeddings are missing")

dna = np.load(
    DNA_MAT,
    mmap_mode="r",
)

dman = pd.read_csv(
    DNA_MAN,
    sep="\t",
    dtype=str,
)

if dna.shape != (3154, 768):
    fail(
        f"Unexpected DNABERT-S matrix shape: {dna.shape}"
    )

if "embedding_index" not in dman.columns:
    fail("DNABERT manifest lacks embedding_index")

if "assembly_accession" not in dman.columns:
    fail("DNABERT manifest lacks assembly_accession")

dna_map = {
    r["assembly_accession"]:
        int(r["embedding_index"])
    for _, r in dman.iterrows()
}

missing_dna = [
    x
    for x in ref_acc + query_acc
    if x not in dna_map
]

if missing_dna:
    fail(
        f"DNABERT-S missing accessions: {missing_dna[:10]}"
    )

ref_idx = np.asarray(
    [
        dna_map[x]
        for x in ref_acc
    ],
    dtype=np.int64,
)

query_idx = np.asarray(
    [
        dna_map[x]
        for x in query_acc
    ],
    dtype=np.int64,
)

R = np.asarray(
    dna[ref_idx],
    dtype=np.float32,
)

Q = np.asarray(
    dna[query_idx],
    dtype=np.float32,
)

M_dna = Q @ R.T


# ======================================================================
# Evaluate native methods
# ======================================================================

endpoint_map = {
    "B1": (
        "species",
        "species_eligible",
        "query_species",
        ref_species,
    ),

    "B2_seen_genus": (
        "genus",
        "genus_eligible",
        "query_genus",
        ref_genus,
    ),

    "B2_unseen_genus": (
        "family",
        "family_eligible",
        "query_family",
        ref_family,
    ),
}


def as_int01(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def evaluate_matrix(method, M):
    rows = []

    if M.shape != (nq, nr):
        fail(
            f"{method}: invalid score matrix shape {M.shape}"
        )

    for i in range(nq):
        qr = query.iloc[i]

        cohort = qr["cohort"]

        endpoint, elig_col, qtax_col, rtax = endpoint_map[cohort]

        eligible = as_int01(
            qr[elig_col]
        )

        finite = np.where(
            np.isfinite(M[i])
        )[0]

        n_candidates = len(finite)

        if n_candidates:
            # refs are already accession-sorted;
            # stable sort gives deterministic accession tie-break.
            order_local = np.argsort(
                -M[i, finite],
                kind="mergesort",
            )

            order = finite[
                order_local
            ]

            top1 = ref_acc[
                int(order[0])
            ]

            top1_score = float(
                M[i, order[0]]
            )

        else:
            order = np.asarray(
                [],
                dtype=np.int64,
            )
            top1 = ""
            top1_score = np.nan

        first_rank = np.nan

        if eligible and n_candidates:
            target = str(
                qr[qtax_col]
            )

            hit = (
                rtax[order].astype(str)
                == target
            )

            hit_idx = np.where(hit)[0]

            if len(hit_idx):
                first_rank = int(
                    hit_idx[0]
                ) + 1

        rows.append({
            "method_type": "native_external",
            "method": method,
            "model_seed": "NA",
            "cohort": cohort,
            "query_accession": qr["query_accession"],
            "endpoint": endpoint,
            "eligible": eligible,
            "n_candidates": n_candidates,
            "query_covered": int(
                n_candidates > 0
            ),
            "pair_coverage_query": n_candidates / nr,
            "top1_reference": top1,
            "top1_score": top1_score,
            "first_relevant_rank": first_rank,
        })

    return pd.DataFrame(rows)


native = pd.concat(
    [
        evaluate_matrix(
            "Mash_k21_s1000",
            M_mash,
        ),
        evaluate_matrix(
            "Dashing2_SetSketch_k31_S1024",
            M_d2,
        ),
        evaluate_matrix(
            "FastANI_v1.34",
            M_fast,
        ),
        evaluate_matrix(
            "skani_v0.3.2",
            M_skani,
        ),
        evaluate_matrix(
            "DNABERT-S",
            M_dna,
        ),
    ],
    ignore_index=True,
)

native.to_csv(
    SRC / "E5_EXTERNAL_NATIVE_QUERY_LEVEL.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# Convert existing frozen HGV/Raw/PCA external result into the same
# primary-retrieval schema.
# ======================================================================

CORE = (
    BASE /
    "09_external_validation/"
    "external_one_shot_evaluation_v1_20260917/"
    "external_query_level_results.tsv"
)

core = pd.read_csv(
    CORE,
    sep="\t",
    dtype=str,
)

core_rows = []

for _, r in core.iterrows():
    cohort = r["cohort"]

    if cohort == "B1":
        endpoint = "species"
        rank_col = "species_first_rank"

    elif cohort == "B2_seen_genus":
        endpoint = "genus"
        rank_col = "genus_first_rank"

    elif cohort == "B2_unseen_genus":
        endpoint = "family"
        rank_col = "family_first_rank"

    else:
        continue

    rank = pd.to_numeric(
        pd.Series([r.get(rank_col)]),
        errors="coerce",
    ).iloc[0]

    core_rows.append({
        "method_type": r["method_type"],
        "method": r["method"],
        "model_seed":
            r["model_seed"]
            if pd.notna(r["model_seed"])
            and str(r["model_seed"]).strip()
            not in {"", "nan"}
            else "NA",
        "cohort": cohort,
        "query_accession": r["query_accession"],
        "endpoint": endpoint,
        "eligible": int(
            pd.notna(rank)
        ),
        "n_candidates": nr,
        "query_covered": 1,
        "pair_coverage_query": 1.0,
        "top1_reference": r["top1_reference"],
        "top1_score": pd.to_numeric(
            pd.Series([r["max_similarity"]]),
            errors="coerce",
        ).iloc[0],
        "first_relevant_rank": rank,
    })

core_u = pd.DataFrame(core_rows)


allq = pd.concat(
    [
        core_u,
        native,
    ],
    ignore_index=True,
)

allq.to_csv(
    SRC / "E5_EXTERNAL_ALL11_QUERY_LEVEL.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# Seed-level and method-level summaries
# ======================================================================

def summarize_group(g):
    elig = g[
        g["eligible"].astype(int) == 1
    ].copy()

    ranks = pd.to_numeric(
        elig["first_relevant_rank"],
        errors="coerce",
    )

    covered = elig[
        elig["query_covered"].astype(int) == 1
    ].copy()

    ranks_cov = pd.to_numeric(
        covered["first_relevant_rank"],
        errors="coerce",
    )

    def rk(x, k, denom):
        if denom == 0:
            return np.nan

        return float(
            (
                x.notna()
                & (x <= k)
            ).sum()
            / denom
        )

    def mrr(x, denom):
        if denom == 0:
            return np.nan

        return float(
            x.dropna()
            .map(lambda z: 1.0 / float(z))
            .sum()
            / denom
        )

    n_elig = len(elig)
    n_cov_elig = len(covered)

    return pd.Series({
        "n_queries": len(g),
        "n_references": nr,
        "eligible_queries": n_elig,
        "eligible_covered_queries": n_cov_elig,
        "query_coverage": float(
            g["query_covered"].astype(float).mean()
        ),
        "pair_coverage": float(
            g["pair_coverage_query"].astype(float).mean()
        ),
        "R1": rk(ranks, 1, n_elig),
        "R5": rk(ranks, 5, n_elig),
        "R10": rk(ranks, 10, n_elig),
        "MRR": mrr(ranks, n_elig),
        "R1_covered_only": rk(
            ranks_cov,
            1,
            n_cov_elig,
        ),
        "R5_covered_only": rk(
            ranks_cov,
            5,
            n_cov_elig,
        ),
        "R10_covered_only": rk(
            ranks_cov,
            10,
            n_cov_elig,
        ),
        "MRR_covered_only": mrr(
            ranks_cov,
            n_cov_elig,
        ),
    })


seed_summary = (
    allq
    .groupby(
        [
            "method_type",
            "method",
            "model_seed",
            "cohort",
            "endpoint",
        ],
        dropna=False,
        sort=True,
    )
    .apply(
        summarize_group,
        include_groups=False,
    )
    .reset_index()
)

seed_summary.to_csv(
    SRC / "E5_EXTERNAL_ALL11_SEED_LEVEL.tsv",
    sep="\t",
    index=False,
)


metric_cols = [
    "query_coverage",
    "pair_coverage",
    "R1",
    "R5",
    "R10",
    "MRR",
    "R1_covered_only",
    "R5_covered_only",
    "R10_covered_only",
    "MRR_covered_only",
]

method_rows = []

for (
    method_type,
    method,
    cohort,
    endpoint,
), g in seed_summary.groupby(
    [
        "method_type",
        "method",
        "cohort",
        "endpoint",
    ],
    sort=True,
):

    row = {
        "method_type": method_type,
        "method": method,
        "cohort": cohort,
        "endpoint": endpoint,
        "n_model_seeds": len(g),
        "n_queries": int(
            g["n_queries"].iloc[0]
        ),
        "n_references": int(
            g["n_references"].iloc[0]
        ),
        "eligible_queries": int(
            g["eligible_queries"].iloc[0]
        ),
    }

    for c in metric_cols:
        x = pd.to_numeric(
            g[c],
            errors="coerce",
        )

        row[c + "_mean"] = x.mean()

        row[c + "_sd"] = (
            x.std(ddof=1)
            if x.notna().sum() > 1
            else 0.0
        )

    method_rows.append(row)


method_summary = pd.DataFrame(
    method_rows
)

method_summary.to_csv(
    SRC / "E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv",
    sep="\t",
    index=False,
)


# ======================================================================
# Completion gate
# ======================================================================

EXPECTED_METHODS = {
    "HGV_Dashing2_tau0p98",
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash_k21_s1000",
    "Dashing2_SetSketch_k31_S1024",
    "FastANI_v1.34",
    "skani_v0.3.2",
    "DNABERT-S",
}

observed = set(
    method_summary["method"]
)

if observed != EXPECTED_METHODS:
    fail(
        "E5 method gate failed.\n"
        f"missing={sorted(EXPECTED_METHODS-observed)}\n"
        f"extra={sorted(observed-EXPECTED_METHODS)}"
    )

for method in EXPECTED_METHODS:
    co = set(
        method_summary.loc[
            method_summary["method"] == method,
            "cohort",
        ]
    )

    if co != {
        "B1",
        "B2_seen_genus",
        "B2_unseen_genus",
    }:
        fail(
            f"{method}: incomplete external cohorts: {co}"
        )


# Check dense methods.
for name, M in [
    ("Mash", M_mash),
    ("Dashing2", M_d2),
    ("DNABERT-S", M_dna),
]:
    cov = np.isfinite(M).mean()

    if cov < 0.999:
        fail(
            f"{name}: unexpected dense coverage {cov}"
        )


report = OUT / "E5_EXTERNAL_FREEZE_REPORT.txt"

with report.open("w") as f:
    f.write(
        "HGV-V3 E5 EXTERNAL 11-METHOD COMPLETION\n"
    )
    f.write("=" * 80 + "\n")
    f.write(
        "Reference A = 2374\n"
        "External queries = 780\n"
        "B1 = 391\n"
        "B2_seen_genus = 196\n"
        "B2_unseen_genus = 193\n\n"
    )

    f.write(
        "Methods = 11\n"
        "Native external methods newly executed = 5\n"
        "Core frozen methods reused = 6\n\n"
    )

    f.write(
        "FastANI/skani no-hit comparisons remain missing.\n"
        "No missing ANI value was imputed.\n\n"
    )

    f.write(
        "PRIMARY SOURCE DATA:\n"
        f"{SRC/'E5_EXTERNAL_ALL11_QUERY_LEVEL.tsv'}\n"
        f"{SRC/'E5_EXTERNAL_ALL11_SEED_LEVEL.tsv'}\n"
        f"{SRC/'E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv'}\n"
        f"{SRC/'E5_EXTERNAL_RUNTIME.tsv'}\n"
    )


hash_paths = [
    FREEZE / "EXTERNAL_TASK_SHA256.txt",
    CORE,
    SRC / "E5_EXTERNAL_ALL11_QUERY_LEVEL.tsv",
    SRC / "E5_EXTERNAL_ALL11_SEED_LEVEL.tsv",
    SRC / "E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv",
    SRC / "E5_EXTERNAL_RUNTIME.tsv",
    report,
]

with (
    OUT / "E5_EXTERNAL_KEY_SHA256.txt"
).open("w") as f:

    for p in hash_paths:
        f.write(
            f"{sha256_file(p)}  {p}\n"
        )


(OUT / "RUN_COMPLETE").write_text(
    "E5_EXTERNAL_11METHOD_COMPLETE\n"
)


print()
print("=" * 100)
print("E5 EXTERNAL COMPLETE")
print("=" * 100)
print("Methods:", len(EXPECTED_METHODS))
print("Queries:", nq)
print("References:", nr)
print()
print(
    method_summary[
        [
            "method",
            "cohort",
            "endpoint",
            "query_coverage_mean",
            "pair_coverage_mean",
            "R1_mean",
            "R5_mean",
            "R10_mean",
            "MRR_mean",
        ]
    ].to_string(index=False)
)
print()
print("Output:", OUT)
