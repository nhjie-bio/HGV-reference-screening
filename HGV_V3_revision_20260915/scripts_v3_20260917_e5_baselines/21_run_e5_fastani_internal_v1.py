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

import pandas as pd


BASE = Path("HGV_V3_revision_20260915")

TASKROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "05_fastani/internal/tasks"
)

OUTROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "05_fastani/internal/results"
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


def realpath(x):
    return os.path.realpath(str(x))


def run_one(job):
    fastani = job["fastani"]
    query = job["query_fasta"]
    ref_list = job["ref_list"]
    raw_out = job["raw_out"]
    stdout_path = job["stdout_path"]
    stderr_path = job["stderr_path"]
    complete_path = job["complete_path"]

    if complete_path.exists():
        return {
            "accession_a": job["accession_a"],
            "status": "SKIPPED_COMPLETE",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "n_references": job["n_references"],
        }

    cmd = [
        fastani,
        "-q", query,
        "--rl", str(ref_list),
        "-o", str(raw_out),
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

    stdout_path.write_text(p.stdout or "")
    stderr_path.write_text(p.stderr or "")

    if p.returncode == 0:
        if not raw_out.exists():
            raw_out.touch()

        complete_path.write_text(
            "RUN_COMPLETE\n"
            f"elapsed_seconds={elapsed:.6f}\n"
        )

        status = "PASS"
    else:
        status = "FAIL"

    return {
        "accession_a": job["accession_a"],
        "status": status,
        "returncode": p.returncode,
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
        choices=["20260915", "20260916", "20260917"],
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=40,
    )

    args = ap.parse_args()

    fastani = shutil.which("fastANI")

    if fastani is None:
        raise RuntimeError("fastANI not found in PATH")

    taskdir = TASKROOT / f"{args.protocol}_seed{args.seed}"
    taskfile = taskdir / "fastani_fixed_pairs.tsv"

    if not taskfile.is_file():
        raise FileNotFoundError(taskfile)

    outdir = OUTROOT / f"{args.protocol}_seed{args.seed}"
    jobdir = outdir / "query_jobs"

    outdir.mkdir(parents=True, exist_ok=True)
    jobdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("E5 FASTANI INTERNAL FIXED-PAIR BENCHMARK")
    print("=" * 72)
    print("protocol       :", args.protocol)
    print("split seed     :", args.seed)
    print("workers        :", args.workers)
    print("fastANI        :", fastani)

    v = subprocess.run(
        [fastani, "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()

    print("version        :", v)
    print("task file      :", taskfile)

    df = pd.read_csv(taskfile, sep="\t", dtype=str)

    required = [
        "pair_index",
        "accession_a",
        "accession_b",
        "fasta_a",
        "fasta_b",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    print("expected pairs :", len(df))

    # Validate FASTA again immediately before execution.
    all_fasta = sorted(
        set(df["fasta_a"].tolist()) |
        set(df["fasta_b"].tolist())
    )

    missing = [x for x in all_fasta if not Path(x).is_file()]

    print("unique FASTA   :", len(all_fasta))
    print("missing FASTA  :", len(missing))

    if missing:
        raise RuntimeError(
            f"Missing FASTA immediately before FastANI run: {missing[:10]}"
        )

    jobs = []

    # Important:
    # Group by frozen accession_a.
    # Each FastANI query receives ONLY its fixed accession_b partners.
    for accession_a, g in df.groupby("accession_a", sort=True):

        qpaths = g["fasta_a"].unique()

        if len(qpaths) != 1:
            raise RuntimeError(
                f"{accession_a}: multiple query FASTA paths: {qpaths}"
            )

        query_fasta = qpaths[0]

        # Preserve only fixed partners.
        refs = list(dict.fromkeys(g["fasta_b"].tolist()))

        qdir = jobdir / accession_a
        qdir.mkdir(parents=True, exist_ok=True)

        ref_list = qdir / "reference_fasta.list"

        with open(ref_list, "w") as f:
            for r in refs:
                f.write(r + "\n")

        raw_out = qdir / "fastani_raw.tsv"

        jobs.append({
            "accession_a": accession_a,
            "query_fasta": query_fasta,
            "ref_list": ref_list,
            "raw_out": raw_out,
            "stdout_path": qdir / "fastani.stdout.txt",
            "stderr_path": qdir / "fastani.stderr.txt",
            "complete_path": qdir / "RUN_COMPLETE",
            "n_references": len(refs),
            "fastani": fastani,
        })

    print("query jobs     :", len(jobs))
    print()

    job_manifest = pd.DataFrame([
        {
            "accession_a": j["accession_a"],
            "query_fasta": j["query_fasta"],
            "reference_list": str(j["ref_list"]),
            "n_references": j["n_references"],
        }
        for j in jobs
    ])

    job_manifest.to_csv(
        outdir / "fastani_query_job_manifest.tsv",
        sep="\t",
        index=False,
    )

    t_all = time.perf_counter()

    run_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = {
            ex.submit(run_one, j): j
            for j in jobs
        }

        done = 0

        for fut in as_completed(future_map):
            r = fut.result()
            run_rows.append(r)

            done += 1

            if done % 25 == 0 or done == len(jobs):
                print(
                    f"[progress] {done}/{len(jobs)} query jobs complete"
                )

    total_elapsed = time.perf_counter() - t_all

    run_df = pd.DataFrame(run_rows).sort_values("accession_a")

    run_df.to_csv(
        outdir / "fastani_query_run_status.tsv",
        sep="\t",
        index=False,
    )

    failed = run_df[run_df["returncode"] != 0]

    print()
    print("query job failures :", len(failed))
    print(f"wall time seconds   : {total_elapsed:.3f}")

    if len(failed) > 0:
        print(failed.head(20).to_string(index=False))
        raise RuntimeError("One or more FastANI query jobs failed.")

    # --------------------------------------------------------
    # Parse FastANI outputs.
    # Format:
    # query reference ANI mapped_fragments total_query_fragments
    # --------------------------------------------------------

    raw_rows = []

    for j in jobs:
        p = j["raw_out"]

        if not p.exists() or p.stat().st_size == 0:
            continue

        with open(p) as f:
            for line in f:
                line = line.rstrip("\n")

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) < 5:
                    raise RuntimeError(
                        f"Unexpected FastANI output line in {p}: {line}"
                    )

                raw_rows.append({
                    "query_fasta_raw": parts[0],
                    "reference_fasta_raw": parts[1],
                    "query_fasta_key": realpath(parts[0]),
                    "reference_fasta_key": realpath(parts[1]),
                    "ani": float(parts[2]),
                    "fragments_matched": int(parts[3]),
                    "fragments_total": int(parts[4]),
                })

    raw = pd.DataFrame(raw_rows)

    if len(raw) > 0:
        raw.to_csv(
            outdir / "fastani_all_reported_raw.tsv",
            sep="\t",
            index=False,
        )

        key_dup = raw.duplicated(
            ["query_fasta_key", "reference_fasta_key"]
        ).sum()

        if key_dup != 0:
            raise RuntimeError(
                f"Duplicate FastANI query/reference outputs: {key_dup}"
            )

        result_lookup = {
            (r.query_fasta_key, r.reference_fasta_key): r
            for r in raw.itertuples(index=False)
        }
    else:
        result_lookup = {}

    merged_rows = []

    unexpected_outputs = 0

    expected_keys = set(
        (
            realpath(a),
            realpath(b),
        )
        for a, b in zip(df["fasta_a"], df["fasta_b"])
    )

    for k in result_lookup:
        if k not in expected_keys:
            unexpected_outputs += 1

    if unexpected_outputs:
        raise RuntimeError(
            f"FastANI produced {unexpected_outputs} outputs "
            "outside the frozen pair manifest."
        )

    for row in df.itertuples(index=False):

        key = (
            realpath(row.fasta_a),
            realpath(row.fasta_b),
        )

        hit = result_lookup.get(key)

        d = row._asdict()

        if hit is None:
            d.update({
                "fastani_reported": 0,
                "fastani_ani": None,
                "fastani_similarity": None,
                "fragments_matched": None,
                "fragments_total": None,
            })
        else:
            d.update({
                "fastani_reported": 1,
                "fastani_ani": hit.ani,
                "fastani_similarity": hit.ani / 100.0,
                "fragments_matched": hit.fragments_matched,
                "fragments_total": hit.fragments_total,
            })

        merged_rows.append(d)

    merged = pd.DataFrame(merged_rows)

    merged_path = outdir / "fastani_fixed_pair_results.tsv"

    merged.to_csv(
        merged_path,
        sep="\t",
        index=False,
    )

    n_expected = len(merged)
    n_reported = int(merged["fastani_reported"].sum())
    n_missing = n_expected - n_reported
    coverage = n_reported / n_expected if n_expected else float("nan")

    coverage_rows = [{
        "protocol": args.protocol,
        "split_seed": args.seed,
        "stratum": "ALL",
        "n_expected": n_expected,
        "n_reported": n_reported,
        "n_missing": n_missing,
        "coverage": coverage,
    }]

    if "label" in merged.columns:
        for label, g in merged.groupby("label", dropna=False):
            nr = int(g["fastani_reported"].sum())

            coverage_rows.append({
                "protocol": args.protocol,
                "split_seed": args.seed,
                "stratum": f"LABEL_{label}",
                "n_expected": len(g),
                "n_reported": nr,
                "n_missing": len(g) - nr,
                "coverage": nr / len(g),
            })

    coverage_df = pd.DataFrame(coverage_rows)

    coverage_df.to_csv(
        outdir / "fastani_coverage_summary.tsv",
        sep="\t",
        index=False,
    )

    report = {
        "protocol": args.protocol,
        "split_seed": args.seed,
        "fastani_executable": fastani,
        "fastani_version": v,
        "workers": args.workers,
        "threads_per_fastani_process": 1,
        "source_task_file": str(taskfile),
        "source_task_sha256": sha256_file(taskfile),
        "n_expected_pairs": n_expected,
        "n_reported_pairs": n_reported,
        "n_missing_pairs": n_missing,
        "coverage": coverage,
        "n_query_jobs": len(jobs),
        "query_job_failures": 0,
        "unexpected_outputs": 0,
        "wall_time_seconds": total_elapsed,
        "result_file": str(merged_path),
        "result_sha256": sha256_file(merged_path),
    }

    with open(outdir / "FASTANI_RUN_AUDIT.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(outdir / "FASTANI_RUN_REPORT.txt", "w") as f:
        f.write("E5 FastANI internal fixed-pair benchmark\n")
        f.write("========================================\n\n")

        for k, v in report.items():
            f.write(f"{k}: {v}\n")

        f.write("\nCoverage by stratum\n")
        f.write("-------------------\n")
        f.write(coverage_df.to_string(index=False))
        f.write("\n\nSTATUS: PASS\n")

    print()
    print("=" * 72)
    print("FASTANI FIXED-PAIR RESULT")
    print("=" * 72)
    print(f"expected pairs : {n_expected}")
    print(f"reported pairs : {n_reported}")
    print(f"missing pairs  : {n_missing}")
    print(f"coverage       : {coverage:.6f}")
    print()
    print(coverage_df.to_string(index=False))
    print()
    print("Result:")
    print(merged_path)
    print()
    print("STATUS = PASS")


if __name__ == "__main__":
    main()
