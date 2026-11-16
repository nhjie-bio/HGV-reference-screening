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
    "05_fastani/validation_bidirectional_v1/results"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def path_key(x):
    return os.path.realpath(str(x))


def run_query_job(job):
    if job["complete"].exists():
        return {
            "query_accession": job["query_accession"],
            "returncode": 0,
            "status": "SKIPPED_COMPLETE",
            "elapsed_seconds": 0.0,
            "n_references": job["n_references"],
        }

    cmd = [
        job["fastani"],
        "-q", job["query_fasta"],
        "--rl", str(job["reference_list"]),
        "-o", str(job["raw_output"]),
        "-t", "1",
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
        choices=[
            "20260915",
            "20260916",
            "20260917",
        ],
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=40,
    )

    args = ap.parse_args()

    fastani = shutil.which("fastANI")

    if fastani is None:
        raise RuntimeError("fastANI not found")

    taskdir = TASKROOT / f"{args.protocol}_seed{args.seed}"
    taskfile = taskdir / "fastani_fixed_pairs.tsv"

    if not taskfile.is_file():
        raise FileNotFoundError(taskfile)

    outdir = OUTROOT / f"{args.protocol}_seed{args.seed}"
    jobroot = outdir / "query_jobs"

    outdir.mkdir(parents=True, exist_ok=True)
    jobroot.mkdir(parents=True, exist_ok=True)

    version = subprocess.run(
        [fastani, "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()

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
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(
                f"Missing required column: {c}"
            )

    print("=" * 76)
    print("E5 FASTANI BIDIRECTIONAL VALIDATION v1")
    print("=" * 76)
    print("protocol       :", args.protocol)
    print("split seed     :", args.seed)
    print("workers        :", args.workers)
    print("fastANI        :", fastani)
    print("version        :", version)
    print("fixed pairs    :", len(df))

    # --------------------------------------------------------
    # Build BOTH directions while retaining original pair ID.
    # --------------------------------------------------------

    oriented_rows = []

    for r in df.itertuples(index=False):
        base = r._asdict()

        oriented_rows.append({
            "pair_index": base["pair_index"],
            "direction": "A_to_B",
            "query_accession": base["accession_a"],
            "reference_accession": base["accession_b"],
            "query_fasta": base["fasta_a"],
            "reference_fasta": base["fasta_b"],
        })

        oriented_rows.append({
            "pair_index": base["pair_index"],
            "direction": "B_to_A",
            "query_accession": base["accession_b"],
            "reference_accession": base["accession_a"],
            "query_fasta": base["fasta_b"],
            "reference_fasta": base["fasta_a"],
        })

    oriented = pd.DataFrame(oriented_rows)

    print("directional tasks:", len(oriented))

    if len(oriented) != 2 * len(df):
        raise RuntimeError(
            "Directional task count is not 2x fixed pairs"
        )

    # Unique directional query/ref check.
    ndup = oriented.duplicated(
        ["query_fasta", "reference_fasta"]
    ).sum()

    if ndup:
        raise RuntimeError(
            f"Duplicate directional comparisons: {ndup}"
        )

    all_fastas = sorted(
        set(oriented["query_fasta"]) |
        set(oriented["reference_fasta"])
    )

    missing_fastas = [
        p for p in all_fastas
        if not Path(p).is_file()
    ]

    print("unique FASTA     :", len(all_fastas))
    print("missing FASTA    :", len(missing_fastas))

    if missing_fastas:
        raise RuntimeError(
            f"Missing FASTA: {missing_fastas[:10]}"
        )

    oriented.to_csv(
        outdir / "fastani_directional_task_manifest.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Group all directional tasks by query genome.
    # Every test genome may now serve as query.
    # --------------------------------------------------------

    jobs = []

    for query_accession, g in oriented.groupby(
        "query_accession",
        sort=True,
    ):

        qpaths = g["query_fasta"].unique()

        if len(qpaths) != 1:
            raise RuntimeError(
                f"{query_accession}: multiple query FASTA paths"
            )

        query_fasta = qpaths[0]

        refs = list(
            dict.fromkeys(
                g["reference_fasta"].tolist()
            )
        )

        qdir = jobroot / query_accession
        qdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        reference_list = (
            qdir /
            "reference_fasta.list"
        )

        with open(reference_list, "w") as f:
            for x in refs:
                f.write(x + "\n")

        jobs.append({
            "query_accession": query_accession,
            "query_fasta": query_fasta,
            "reference_list": reference_list,
            "raw_output": (
                qdir /
                "fastani_raw.tsv"
            ),
            "stdout": (
                qdir /
                "fastani.stdout.txt"
            ),
            "stderr": (
                qdir /
                "fastani.stderr.txt"
            ),
            "complete": (
                qdir /
                "RUN_COMPLETE"
            ),
            "n_references": len(refs),
            "fastani": fastani,
        })

    print("query jobs       :", len(jobs))

    job_manifest = pd.DataFrame([
        {
            "query_accession": j["query_accession"],
            "query_fasta": j["query_fasta"],
            "reference_list": str(j["reference_list"]),
            "n_references": j["n_references"],
        }
        for j in jobs
    ])

    job_manifest.to_csv(
        outdir / "fastani_query_job_manifest.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Execute.
    # --------------------------------------------------------

    run_rows = []

    t_all = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futures = {
            ex.submit(run_query_job, j): j
            for j in jobs
        }

        done = 0

        for fut in as_completed(futures):
            result = fut.result()

            run_rows.append(result)

            done += 1

            if (
                done % 25 == 0
                or done == len(jobs)
            ):
                print(
                    f"[progress] "
                    f"{done}/{len(jobs)} "
                    f"query jobs complete"
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
        outdir /
        "fastani_query_run_status.tsv",
        sep="\t",
        index=False,
    )

    failed = run_df[
        run_df["returncode"] != 0
    ]

    print()
    print(
        "query failures   :",
        len(failed)
    )

    print(
        "wall seconds     :",
        f"{wall_seconds:.3f}"
    )

    if len(failed):
        print(
            failed.head(20).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "FastANI query job failure"
        )

    # --------------------------------------------------------
    # Read reported directional ANI.
    # --------------------------------------------------------

    raw_rows = []

    for j in jobs:
        p = j["raw_output"]

        if (
            not p.exists()
            or p.stat().st_size == 0
        ):
            continue

        with open(p) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) < 5:
                    raise RuntimeError(
                        f"Malformed FastANI line: "
                        f"{line}"
                    )

                raw_rows.append({
                    "query_fasta_raw":
                        parts[0],
                    "reference_fasta_raw":
                        parts[1],
                    "query_key":
                        path_key(parts[0]),
                    "reference_key":
                        path_key(parts[1]),
                    "ani":
                        float(parts[2]),
                    "fragments_matched":
                        int(parts[3]),
                    "fragments_total":
                        int(parts[4]),
                })

    raw = pd.DataFrame(raw_rows)

    if len(raw):
        dup = raw.duplicated(
            [
                "query_key",
                "reference_key",
            ]
        ).sum()

        if dup:
            raise RuntimeError(
                f"Duplicate FastANI outputs: "
                f"{dup}"
            )

        raw.to_csv(
            outdir /
            "fastani_directional_reported_raw.tsv",
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
            path_key(r.query_fasta),
            path_key(r.reference_fasta),
        )
        for r in oriented.itertuples(
            index=False
        )
    }

    unexpected = [
        k
        for k in lookup
        if k not in expected_keys
    ]

    if unexpected:
        raise RuntimeError(
            f"Unexpected outputs: "
            f"{len(unexpected)}"
        )

    # --------------------------------------------------------
    # Direction-level result table.
    # --------------------------------------------------------

    dir_result_rows = []

    for r in oriented.itertuples(
        index=False
    ):
        key = (
            path_key(r.query_fasta),
            path_key(r.reference_fasta),
        )

        hit = lookup.get(key)

        if hit is None:
            reported = 0
            ani = np.nan
            fm = np.nan
            ft = np.nan

        else:
            reported = 1
            ani = hit.ani
            fm = hit.fragments_matched
            ft = hit.fragments_total

        dir_result_rows.append({
            "pair_index":
                r.pair_index,
            "direction":
                r.direction,
            "query_accession":
                r.query_accession,
            "reference_accession":
                r.reference_accession,
            "query_fasta":
                r.query_fasta,
            "reference_fasta":
                r.reference_fasta,
            "reported":
                reported,
            "ani":
                ani,
            "fragments_matched":
                fm,
            "fragments_total":
                ft,
        })

    directional = pd.DataFrame(
        dir_result_rows
    )

    directional.to_csv(
        outdir /
        "fastani_directional_results.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Construct final symmetric pair-level table.
    # --------------------------------------------------------

    by_key = {}

    for r in directional.itertuples(
        index=False
    ):
        by_key[
            (
                str(r.pair_index),
                r.direction,
            )
        ] = r

    pair_rows = []

    for original in df.itertuples(
        index=False
    ):

        pidx = str(original.pair_index)

        ab = by_key[
            (pidx, "A_to_B")
        ]

        ba = by_key[
            (pidx, "B_to_A")
        ]

        n_reported = (
            int(ab.reported) +
            int(ba.reported)
        )

        if n_reported == 2:
            sym_ani = (
                float(ab.ani) +
                float(ba.ani)
            ) / 2.0

            status = "BOTH"

        elif n_reported == 1:
            sym_ani = np.nan
            status = "ONE_DIRECTION_ONLY"

        else:
            sym_ani = np.nan
            status = "NONE"

        d = original._asdict()

        d.update({
            "reported_A_to_B":
                int(ab.reported),
            "ani_A_to_B":
                ab.ani,
            "fragments_matched_A_to_B":
                ab.fragments_matched,
            "fragments_total_A_to_B":
                ab.fragments_total,

            "reported_B_to_A":
                int(ba.reported),
            "ani_B_to_A":
                ba.ani,
            "fragments_matched_B_to_A":
                ba.fragments_matched,
            "fragments_total_B_to_A":
                ba.fragments_total,

            "n_reported_directions":
                n_reported,

            "fastani_pair_status":
                status,

            "fastani_symmetric_ani":
                sym_ani,

            "fastani_symmetric_similarity":
                (
                    sym_ani / 100.0
                    if not pd.isna(sym_ani)
                    else np.nan
                ),
        })

        pair_rows.append(d)

    pair_df = pd.DataFrame(
        pair_rows
    )

    result_path = (
        outdir /
        "fastani_symmetric_fixed_pair_results.tsv"
    )

    pair_df.to_csv(
        result_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Coverage summary.
    # --------------------------------------------------------

    def coverage_stats(g):
        both = (
            g["fastani_pair_status"]
            == "BOTH"
        ).sum()

        one = (
            g["fastani_pair_status"]
            == "ONE_DIRECTION_ONLY"
        ).sum()

        none = (
            g["fastani_pair_status"]
            == "NONE"
        ).sum()

        n = len(g)

        return {
            "n_expected": n,
            "n_both": int(both),
            "n_one_direction": int(one),
            "n_none": int(none),

            "coverage_both":
                both / n,

            "coverage_any":
                (both + one) / n,
        }

    cov_rows = []

    x = coverage_stats(pair_df)

    cov_rows.append({
        "protocol": args.protocol,
        "split_seed": args.seed,
        "stratum": "ALL",
        **x,
    })

    if "label" in pair_df.columns:
        for label, g in pair_df.groupby(
            "label",
            dropna=False,
        ):
            x = coverage_stats(g)

            cov_rows.append({
                "protocol":
                    args.protocol,
                "split_seed":
                    args.seed,
                "stratum":
                    f"LABEL_{label}",
                **x,
            })

    coverage_df = pd.DataFrame(
        cov_rows
    )

    coverage_df.to_csv(
        outdir /
        "fastani_bidirectional_coverage.tsv",
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Directional asymmetry audit.
    # --------------------------------------------------------

    both = pair_df[
        pair_df["fastani_pair_status"]
        == "BOTH"
    ].copy()

    if len(both):
        both["ani_abs_direction_delta"] = (
            both["ani_A_to_B"] -
            both["ani_B_to_A"]
        ).abs()

        asymmetry = {
            "n_bidirectional":
                int(len(both)),
            "mean_abs_delta":
                float(
                    both[
                        "ani_abs_direction_delta"
                    ].mean()
                ),
            "median_abs_delta":
                float(
                    both[
                        "ani_abs_direction_delta"
                    ].median()
                ),
            "max_abs_delta":
                float(
                    both[
                        "ani_abs_direction_delta"
                    ].max()
                ),
        }

    else:
        asymmetry = {
            "n_bidirectional": 0,
            "mean_abs_delta": None,
            "median_abs_delta": None,
            "max_abs_delta": None,
        }

    with open(
        outdir /
        "FASTANI_BIDIRECTIONAL_AUDIT.json",
        "w",
    ) as f:

        json.dump(
            {
                "protocol":
                    args.protocol,
                "split_seed":
                    args.seed,
                "fastani":
                    fastani,
                "version":
                    version,
                "workers":
                    args.workers,
                "threads_per_process":
                    1,
                "fixed_pairs":
                    len(df),
                "directional_tasks":
                    len(oriented),
                "query_jobs":
                    len(jobs),
                "wall_seconds":
                    wall_seconds,
                "source_task":
                    str(taskfile),
                "source_task_sha256":
                    sha256_file(taskfile),
                "result":
                    str(result_path),
                "result_sha256":
                    sha256_file(
                        result_path
                    ),
                "asymmetry":
                    asymmetry,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print(
        "FASTANI BIDIRECTIONAL "
        "PAIR-LEVEL RESULT"
    )
    print("=" * 76)

    print(
        coverage_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Directional asymmetry:"
    )

    for k, v in asymmetry.items():
        print(
            f"{k:24s}: {v}"
        )

    print()
    print("Result:")
    print(result_path)
    print()
    print("STATUS = PASS")


if __name__ == "__main__":
    main()
