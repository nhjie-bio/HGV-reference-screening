#!/usr/bin/env python3

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd


BASE = Path("HGV_V3_revision_20260915")

TASKROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "00_native_validation/compat_tasks"
)

OUTROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "04_skani/validation_v1/results"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pkey(x):
    return os.path.realpath(str(x))


def run_job(job):
    if job["complete"].exists():
        return {
            "query_accession": job["query_accession"],
            "returncode": 0,
            "status": "SKIPPED_COMPLETE",
            "elapsed_seconds": 0.0,
            "n_references": job["n_references"],
        }

    cmd = [
        job["skani"],
        "dist",
        "-t", "1",
        "--ql", str(job["query_list"]),
        "--rl", str(job["reference_list"]),
        "-o", str(job["raw_output"]),
    ]

    t0 = time.perf_counter()

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.perf_counter() - t0

    job["stdout"].write_text(p.stdout or "")
    job["stderr"].write_text(p.stderr or "")

    if p.returncode == 0:
        if not job["raw_output"].exists():
            job["raw_output"].touch()

        job["complete"].write_text(
            "RUN_COMPLETE\n"
            f"elapsed_seconds={elapsed:.6f}\n"
        )
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "query_accession": job["query_accession"],
        "returncode": p.returncode,
        "status": status,
        "elapsed_seconds": elapsed,
        "n_references": job["n_references"],
    }


def read_skani_output(path):
    rows = []

    if not path.exists() or path.stat().st_size == 0:
        return rows

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            # Header from skani.
            if line.startswith("Ref_file\t"):
                continue

            parts = line.split("\t")

            if len(parts) < 5:
                raise RuntimeError(
                    f"Malformed skani output in {path}: {line}"
                )

            rows.append({
                "reference_fasta_raw": parts[0],
                "query_fasta_raw": parts[1],
                "reference_key": pkey(parts[0]),
                "query_key": pkey(parts[1]),
                "ani": float(parts[2]),
                "af_ref": float(parts[3]),
                "af_query": float(parts[4]),
            })

    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--protocol",
        required=True,
        choices=["G2", "G3", "G4"],
    )

    ap.add_argument(
        "--seed",
        required=True,
        choices=["20260915", "20260916", "20260917"],
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=40,
    )

    args = ap.parse_args()

    skani = shutil.which("skani")
    if skani is None:
        raise RuntimeError("skani not found in PATH")

    version = subprocess.run(
        [skani, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()

    taskfile = (
        TASKROOT /
        f"{args.protocol}_seed{args.seed}" /
        "fastani_fixed_pairs.tsv"
    )

    if not taskfile.is_file():
        raise FileNotFoundError(taskfile)

    outdir = (
        OUTROOT /
        f"{args.protocol}_seed{args.seed}"
    )

    jobroot = outdir / "query_jobs"

    outdir.mkdir(parents=True, exist_ok=True)
    jobroot.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        taskfile,
        sep="\t",
        dtype=str,
    )

    required = [
        "pair_index",
        "accession_a",
        "accession_b",
        "fasta_a",
        "fasta_b",
        "label",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"Missing required task column: {c}"
            )

    print("=" * 76)
    print("E5 SKANI VALIDATION FIXED-PAIR BENCHMARK v1")
    print("=" * 76)
    print("protocol       :", args.protocol)
    print("split seed     :", args.seed)
    print("workers        :", args.workers)
    print("skani          :", skani)
    print("version        :", version)
    print("expected pairs :", len(df))

    all_fastas = sorted(
        set(df["fasta_a"]) |
        set(df["fasta_b"])
    )

    missing = [
        p for p in all_fastas
        if not Path(p).is_file()
    ]

    print("unique FASTA   :", len(all_fastas))
    print("missing FASTA  :", len(missing))

    if missing:
        raise RuntimeError(
            f"Missing FASTA: {missing[:10]}"
        )

    # ------------------------------------------------------------
    # One query job per accession_a.
    #
    # skani ANI is symmetric, so unlike FastANI we do not need
    # A->B + B->A averaging.
    # ------------------------------------------------------------

    jobs = []

    for accession_a, g in df.groupby(
        "accession_a",
        sort=True,
    ):
        qpaths = g["fasta_a"].unique()

        if len(qpaths) != 1:
            raise RuntimeError(
                f"{accession_a}: multiple query paths"
            )

        query_fasta = qpaths[0]

        refs = list(
            dict.fromkeys(
                g["fasta_b"].tolist()
            )
        )

        qdir = jobroot / accession_a
        qdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        query_list = qdir / "query_fasta.list"
        ref_list = qdir / "reference_fasta.list"

        query_list.write_text(
            query_fasta + "\n"
        )

        with open(ref_list, "w") as f:
            for r in refs:
                f.write(r + "\n")

        jobs.append({
            "query_accession": accession_a,
            "query_fasta": query_fasta,
            "query_list": query_list,
            "reference_list": ref_list,
            "raw_output": qdir / "skani_raw.tsv",
            "stdout": qdir / "skani.stdout.txt",
            "stderr": qdir / "skani.stderr.txt",
            "complete": qdir / "RUN_COMPLETE",
            "n_references": len(refs),
            "skani": skani,
        })

    print("query jobs     :", len(jobs))

    pd.DataFrame([
        {
            "query_accession":
                j["query_accession"],
            "query_fasta":
                j["query_fasta"],
            "query_list":
                str(j["query_list"]),
            "reference_list":
                str(j["reference_list"]),
            "n_references":
                j["n_references"],
        }
        for j in jobs
    ]).to_csv(
        outdir / "skani_query_job_manifest.tsv",
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # Run.
    # ------------------------------------------------------------

    t_all = time.perf_counter()

    run_rows = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futures = {
            ex.submit(run_job, j): j
            for j in jobs
        }

        done = 0

        for fut in as_completed(futures):
            r = fut.result()
            run_rows.append(r)

            done += 1

            if done % 25 == 0 or done == len(jobs):
                print(
                    f"[progress] {done}/{len(jobs)} "
                    "query jobs complete"
                )

    wall_seconds = (
        time.perf_counter() -
        t_all
    )

    run_df = pd.DataFrame(
        run_rows
    ).sort_values(
        "query_accession"
    )

    run_df.to_csv(
        outdir / "skani_query_run_status.tsv",
        sep="\t",
        index=False,
    )

    failed = run_df[
        run_df["returncode"] != 0
    ]

    print()
    print("query failures :", len(failed))
    print(
        "wall seconds   :",
        f"{wall_seconds:.3f}"
    )

    if len(failed):
        print(
            failed.head(20).to_string(
                index=False
            )
        )
        raise RuntimeError(
            "One or more skani jobs failed"
        )

    # ------------------------------------------------------------
    # Parse native results.
    # ------------------------------------------------------------

    raw_rows = []

    for j in jobs:
        raw_rows.extend(
            read_skani_output(
                j["raw_output"]
            )
        )

    raw = pd.DataFrame(raw_rows)

    if len(raw):
        dup = raw.duplicated(
            ["query_key", "reference_key"]
        ).sum()

        if dup:
            raise RuntimeError(
                f"Duplicate skani outputs: {dup}"
            )

        raw.to_csv(
            outdir /
            "skani_all_reported_raw.tsv",
            sep="\t",
            index=False,
        )

        lookup = {
            (
                r.query_key,
                r.reference_key,
            ): r
            for r in raw.itertuples(
                index=False
            )
        }

    else:
        lookup = {}

    expected_keys = {
        (
            pkey(a),
            pkey(b),
        )
        for a, b in zip(
            df["fasta_a"],
            df["fasta_b"],
        )
    }

    unexpected = [
        k
        for k in lookup
        if k not in expected_keys
    ]

    if unexpected:
        raise RuntimeError(
            f"Unexpected skani outputs: "
            f"{len(unexpected)}"
        )

    # ------------------------------------------------------------
    # Left join back to frozen pairs.
    # ------------------------------------------------------------

    rows = []

    for r in df.itertuples(index=False):

        hit = lookup.get(
            (
                pkey(r.fasta_a),
                pkey(r.fasta_b),
            )
        )

        d = r._asdict()

        if hit is None:
            d.update({
                "skani_reported": 0,
                "skani_ani": np.nan,
                "skani_similarity": np.nan,
                "skani_af_ref": np.nan,
                "skani_af_query": np.nan,
            })

        else:
            d.update({
                "skani_reported": 1,
                "skani_ani": hit.ani,
                "skani_similarity":
                    hit.ani / 100.0,
                "skani_af_ref":
                    hit.af_ref,
                "skani_af_query":
                    hit.af_query,
            })

        rows.append(d)

    result = pd.DataFrame(rows)

    result_path = (
        outdir /
        "skani_fixed_pair_results.tsv"
    )

    result.to_csv(
        result_path,
        sep="\t",
        index=False,
    )

    # ------------------------------------------------------------
    # Coverage audit.
    # ------------------------------------------------------------

    cov_rows = []

    groups = [("ALL", result)]

    for label, g in result.groupby(
        "label",
        dropna=False,
    ):
        groups.append(
            (f"LABEL_{label}", g)
        )

    for name, g in groups:

        n = len(g)
        nr = int(
            g["skani_reported"].sum()
        )

        cov_rows.append({
            "protocol":
                args.protocol,

            "split_seed":
                args.seed,

            "stratum":
                name,

            "n_expected":
                n,

            "n_reported":
                nr,

            "n_missing":
                n - nr,

            "coverage":
                nr / n,
        })

    coverage = pd.DataFrame(
        cov_rows
    )

    coverage.to_csv(
        outdir /
        "skani_coverage_summary.tsv",
        sep="\t",
        index=False,
    )

    audit = {
        "protocol":
            args.protocol,

        "split_seed":
            args.seed,

        "skani_executable":
            skani,

        "skani_version":
            version,

        "workers":
            args.workers,

        "threads_per_process":
            1,

        "source_task_file":
            str(taskfile),

        "source_task_sha256":
            sha256_file(taskfile),

        "expected_pairs":
            len(result),

        "reported_pairs":
            int(
                result[
                    "skani_reported"
                ].sum()
            ),

        "missing_pairs":
            int(
                (
                    result[
                        "skani_reported"
                    ] == 0
                ).sum()
            ),

        "query_jobs":
            len(jobs),

        "query_failures":
            0,

        "wall_seconds":
            wall_seconds,

        "result_file":
            str(result_path),

        "result_sha256":
            sha256_file(result_path),
    }

    with open(
        outdir /
        "SKANI_RUN_AUDIT.json",
        "w",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print("SKANI FIXED-PAIR RESULT")
    print("=" * 76)

    print(
        coverage.to_string(
            index=False
        )
    )

    print()
    print("Result:")
    print(result_path)

    print()
    print("STATUS = PASS")


if __name__ == "__main__":
    main()
