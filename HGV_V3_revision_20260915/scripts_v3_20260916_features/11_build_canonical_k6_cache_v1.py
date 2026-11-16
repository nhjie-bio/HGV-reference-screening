#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


BASE = Path("HGV_V3_revision_20260915")

DEFAULT_MANIFEST = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916"
    / "final_master_manifest_3154.tsv"
)

DEFAULT_EXTRACTOR = (
    BASE
    / "scripts_v3_20260916_features"
    / "10_canonical_k6_extract_v1"
)


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    required = {
        "assembly_accession_final",
        "final_cohort",
        "fasta_path",
    }

    missing = required - set(reader.fieldnames or [])
    if missing:
        raise RuntimeError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    return rows


def run_one(args):
    i, accession, cohort, fasta, extractor = args

    t0 = time.time()

    p = subprocess.run(
        [str(extractor), str(fasta)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.time() - t0

    if p.returncode != 0:
        return {
            "ok": False,
            "index": i,
            "accession": accession,
            "cohort": cohort,
            "fasta_path": fasta,
            "returncode": p.returncode,
            "stderr": p.stderr.strip(),
            "elapsed_sec": elapsed,
        }

    fields = p.stdout.rstrip("\n").split("\t")

    # 6 QC fields + 2080 feature dimensions.
    if len(fields) != 2086:
        return {
            "ok": False,
            "index": i,
            "accession": accession,
            "cohort": cohort,
            "fasta_path": fasta,
            "returncode": 100,
            "stderr": (
                f"Expected 2086 output fields, observed {len(fields)}"
            ),
            "elapsed_sec": elapsed,
        }

    contigs = int(fields[0])
    total_bases = int(fields[1])
    acgt_bases = int(fields[2])
    ambiguous_bases = int(fields[3])
    ambiguous_fraction = float(fields[4])
    valid_kmers = int(fields[5])

    vector = np.asarray(
        [float(x) for x in fields[6:]],
        dtype=np.float32,
    )

    if vector.shape != (2080,):
        return {
            "ok": False,
            "index": i,
            "accession": accession,
            "cohort": cohort,
            "fasta_path": fasta,
            "returncode": 101,
            "stderr": f"Bad vector shape: {vector.shape}",
            "elapsed_sec": elapsed,
        }

    vector_sum = float(vector.sum(dtype=np.float64))

    if not np.isfinite(vector).all():
        return {
            "ok": False,
            "index": i,
            "accession": accession,
            "cohort": cohort,
            "fasta_path": fasta,
            "returncode": 102,
            "stderr": "Non-finite feature value",
            "elapsed_sec": elapsed,
        }

    if abs(vector_sum - 1.0) > 1e-5:
        return {
            "ok": False,
            "index": i,
            "accession": accession,
            "cohort": cohort,
            "fasta_path": fasta,
            "returncode": 103,
            "stderr": f"Feature sum != 1: {vector_sum}",
            "elapsed_sec": elapsed,
        }

    return {
        "ok": True,
        "index": i,
        "accession": accession,
        "cohort": cohort,
        "fasta_path": fasta,
        "contigs": contigs,
        "total_bases": total_bases,
        "acgt_bases": acgt_bases,
        "ambiguous_bases": ambiguous_bases,
        "ambiguous_fraction": ambiguous_fraction,
        "valid_kmers": valid_kmers,
        "feature_sum": vector_sum,
        "elapsed_sec": elapsed,
        "vector": vector,
    }


def canonical_vocab():
    trans = str.maketrans("ACGT", "TGCA")

    kmers = []

    def decode(x):
        chars = []
        for shift in (10, 8, 6, 4, 2, 0):
            chars.append("ACGT"[(x >> shift) & 3])
        return "".join(chars)

    for x in range(4096):
        s = decode(x)
        rc = s.translate(trans)[::-1]
        if s <= rc:
            kmers.append(s)

    if len(kmers) != 2080:
        raise RuntimeError(
            f"Canonical vocabulary should contain 2080 terms; "
            f"observed {len(kmers)}"
        )

    return kmers


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )

    ap.add_argument(
        "--extractor",
        default=str(DEFAULT_EXTRACTOR),
    )

    ap.add_argument(
        "--outdir",
        required=True,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all genomes",
    )

    args = ap.parse_args()

    manifest = Path(args.manifest)
    extractor = Path(args.extractor)
    outdir = Path(args.outdir)

    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    if not extractor.is_file():
        raise SystemExit(f"Extractor not found: {extractor}")

    if outdir.exists():
        raise SystemExit(
            f"Output directory already exists; refusing overwrite: {outdir}"
        )

    rows = read_manifest(manifest)

    if args.limit > 0:
        rows = rows[:args.limit]

    n = len(rows)

    if n == 0:
        raise SystemExit("No genomes selected")

    print("=" * 72)
    print("HGV-V3 canonical k=6 feature extraction")
    print("=" * 72)
    print(f"Manifest : {manifest}")
    print(f"Genomes  : {n}")
    print(f"Workers  : {args.workers}")
    print(f"k        : 6")
    print(f"Dim      : 2080")
    print(f"Mode     : full assembly")
    print(f"RC       : canonical invariant")
    print()

    # Validate FASTA existence before doing work.
    missing = []

    tasks = []

    for i, row in enumerate(rows):
        acc = row["assembly_accession_final"].strip()
        cohort = row["final_cohort"].strip()
        fasta = Path(row["fasta_path"].strip())

        if not fasta.is_file():
            missing.append((acc, str(fasta)))

        tasks.append(
            (
                i,
                acc,
                cohort,
                str(fasta),
                str(extractor),
            )
        )

    if missing:
        print(f"Missing FASTA: {len(missing)}")
        for x in missing[:20]:
            print(x)
        raise SystemExit(2)

    # Only create output directory once preflight has passed.
    outdir.mkdir(parents=True, exist_ok=False)

    t_start = time.time()

    results = [None] * n
    failures = []

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futs = {
            ex.submit(run_one, task): task[0]
            for task in tasks
        }

        done = 0

        for fut in as_completed(futs):
            res = fut.result()

            done += 1

            if res["ok"]:
                results[res["index"]] = res
            else:
                failures.append(res)

            if (
                done <= 10
                or done % 50 == 0
                or done == n
            ):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{done:5d}/{n:5d}] "
                    f"fail={len(failures):3d} "
                    f"rate={rate:.2f} genomes/s"
                )

    elapsed_total = time.time() - t_start

    if failures:
        fail_path = outdir / "FAILED_GENOMES.tsv"

        with open(
            fail_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            fields = [
                "index",
                "accession",
                "cohort",
                "fasta_path",
                "returncode",
                "stderr",
                "elapsed_sec",
            ]

            w = csv.DictWriter(
                f,
                delimiter="\t",
                fieldnames=fields,
                lineterminator="\n",
                extrasaction="ignore",
            )

            w.writeheader()
            w.writerows(failures)

        print()
        print("FEATURE_EXTRACTION=FAIL")
        print(f"Failed genomes: {len(failures)}")
        print(f"See: {fail_path}")
        sys.exit(3)

    X = np.stack(
        [r["vector"] for r in results],
        axis=0,
    ).astype(np.float32, copy=False)

    if X.shape != (n, 2080):
        raise RuntimeError(
            f"Unexpected matrix shape: {X.shape}"
        )

    row_sums = X.sum(axis=1, dtype=np.float64)

    max_sum_error = float(
        np.max(np.abs(row_sums - 1.0))
    )

    if max_sum_error > 1e-5:
        raise RuntimeError(
            f"Feature normalization gate failed: {max_sum_error}"
        )

    # Save binary feature matrix.
    matrix_path = outdir / "canonical_k6_features.float32.npy"
    np.save(matrix_path, X)

    # Row order / identity.
    rows_path = outdir / "canonical_k6_rows.tsv"

    with open(
        rows_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = [
            "row_index",
            "assembly_accession",
            "final_cohort",
            "fasta_path",
        ]

        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        w.writeheader()

        for i, r in enumerate(results):
            w.writerow({
                "row_index": i,
                "assembly_accession": r["accession"],
                "final_cohort": r["cohort"],
                "fasta_path": r["fasta_path"],
            })

    # Per-genome extraction QC.
    qc_path = outdir / "canonical_k6_extraction_qc.tsv"

    with open(
        qc_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = [
            "row_index",
            "assembly_accession",
            "final_cohort",
            "contigs",
            "total_bases",
            "acgt_bases",
            "ambiguous_bases",
            "ambiguous_fraction",
            "valid_kmers",
            "feature_sum",
            "elapsed_sec",
        ]

        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        w.writeheader()

        for i, r in enumerate(results):
            w.writerow({
                "row_index": i,
                "assembly_accession": r["accession"],
                "final_cohort": r["cohort"],
                "contigs": r["contigs"],
                "total_bases": r["total_bases"],
                "acgt_bases": r["acgt_bases"],
                "ambiguous_bases": r["ambiguous_bases"],
                "ambiguous_fraction": (
                    f"{r['ambiguous_fraction']:.12g}"
                ),
                "valid_kmers": r["valid_kmers"],
                "feature_sum": (
                    f"{r['feature_sum']:.12g}"
                ),
                "elapsed_sec": (
                    f"{r['elapsed_sec']:.6f}"
                ),
            })

    # Vocabulary/order.
    vocab = canonical_vocab()

    vocab_path = outdir / "canonical_k6_vocabulary.tsv"

    with open(
        vocab_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n",
        )

        w.writerow([
            "feature_index",
            "canonical_6mer",
        ])

        for i, kmer in enumerate(vocab):
            w.writerow([i, kmer])

    # Config/provenance.
    metadata = {
        "feature_version": "canonical_k6_v1",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "extractor": str(extractor),
        "extractor_sha256": sha256_file(extractor),
        "n_genomes": n,
        "k": 6,
        "dimension": 2080,
        "strand_policy": (
            "reverse-complement canonical invariant"
        ),
        "sequence_scope": "full assembly; contigs independent",
        "ambiguous_policy": (
            "windows containing non-ACGT are excluded"
        ),
        "normalization": (
            "count / total valid canonical k-mer windows per genome"
        ),
        "dtype": "float32",
        "workers": args.workers,
        "elapsed_sec": elapsed_total,
        "matrix_shape": list(X.shape),
        "max_abs_row_sum_error": max_sum_error,
    }

    with open(
        outdir / "feature_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    # SHA256 provenance of main outputs.
    hashes = []

    for p in (
        matrix_path,
        rows_path,
        qc_path,
        vocab_path,
        outdir / "feature_config.json",
    ):
        hashes.append(
            (p.name, sha256_file(p))
        )

    with open(
        outdir / "SHA256SUMS.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for name, digest in hashes:
            f.write(f"{digest}  {name}\n")

    report_path = outdir / "FEATURE_EXTRACTION_REPORT.txt"

    cohort_counts = {}

    for r in results:
        cohort_counts[r["cohort"]] = (
            cohort_counts.get(r["cohort"], 0) + 1
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("HGV-V3 CANONICAL K=6 FEATURE EXTRACTION\n")
        f.write("=" * 72 + "\n\n")
        f.write("STATUS: PASS\n")
        f.write(f"Genomes: {n}\n")
        f.write("Feature dimension: 2080\n")
        f.write("Feature dtype: float32\n")
        f.write(f"Matrix shape: {X.shape}\n")
        f.write(
            f"Maximum row normalization error: "
            f"{max_sum_error:.12g}\n"
        )
        f.write(f"Elapsed seconds: {elapsed_total:.3f}\n")
        f.write(
            f"Throughput genomes/sec: "
            f"{n / elapsed_total:.6f}\n\n"
        )

        f.write("COHORT COUNTS\n")
        for k in sorted(cohort_counts):
            f.write(f"{k}: {cohort_counts[k]}\n")

        f.write("\nFEATURE_EXTRACTION=PASS\n")

    print()
    print("=" * 72)
    print("ALL FEATURE EXTRACTION GATES PASSED")
    print("=" * 72)
    print(f"Matrix       : {matrix_path}")
    print(f"Shape        : {X.shape}")
    print(f"Max sum err  : {max_sum_error:.3e}")
    print(f"Elapsed      : {elapsed_total:.1f} sec")
    print(f"Output       : {outdir}")
    print()
    print("FEATURE_EXTRACTION=PASS")


if __name__ == "__main__":
    main()
