#!/usr/bin/env python3
import csv
import gzip
import hashlib
import json
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

WORK = ROOT / "21_frontiers_revision_20261112/90_windows_minimal"

W07 = ROOT / (
    "scripts_frontiers_windows/W07_k6_windows_parity_v1.py"
)

W07_STATUS = WORK / (
    "W07_feature_parity/W07_STATUS.txt"
)

W08 = WORK / "W08_E13_MAG_FASTA"

SOURCE = W08 / "W08_ALL_VALIDATED_MAG.tsv"

OUT = WORK / "W09_E13_MAG_FEATURES"

BINS = ("50-70", "70-80", "80-90", "90-100")


def require(condition, message):
    if not condition:
        raise RuntimeError("[STOP] " + message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def uncompressed_sha256(path):
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_tsv(path):
    require(path.is_file(), "Missing TSV: " + str(path))

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        rows = list(reader)

    require(
        bool(columns) and not any(None in r for r in rows),
        "Invalid TSV: " + str(path)
    )
    return rows


def save_tsv(path, rows, fields):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("W09 - REAL MAG CANONICAL K6 FEATURE EXTRACTION")
    print("=" * 72)

    require(W07.is_file(), "Missing validated W07 extractor")
    require(W07_STATUS.is_file(), "Missing W07 parity status")

    status_text = W07_STATUS.read_text(encoding="utf-8")

    require(
        "W07_FEATURE_PARITY=PASS" in status_text,
        "W07 original-genome feature parity not passed"
    )

    require(
        (W08 / "W08_ALL_STATUS.txt").is_file(),
        "Missing W08 status"
    )

    w08_status = (
        W08 / "W08_ALL_STATUS.txt"
    ).read_text(encoding="utf-8")

    require(
        "TARGET=80" in w08_status
        and "PASS=80" in w08_status
        and "STATUS=PASS" in w08_status,
        "W08 full download has not passed"
    )

    require(
        not OUT.exists(),
        "Output already exists; refusing overwrite: " + str(OUT)
    )

    records = load_tsv(SOURCE)

    require(
        len(records) == 80,
        f"Expected 80 MAG records, found {len(records)}"
    )

    accessions = [r["accession"] for r in records]

    require(
        len(set(accessions)) == 80,
        "Duplicate MAG accessions"
    )

    print("[PASS] W07 feature parity and W08 download status")
    print("[PASS] Validated MAG records:", len(records))

    # Import the exact W07 function that passed parity testing.
    extractor = runpy.run_path(str(W07))
    canonical_k6 = extractor["canonical_k6"]

    print("[PASS] Loaded W07 canonical_k6 function")

    vectors = []
    metadata = []
    qc_rows = []

    start = time.perf_counter()

    fasta_directory = (
        W08 / "genomic_fasta_gz"
    ).resolve()

    for index, record in enumerate(records):

        accession = record["accession"]

        require(
            record["download_status"].startswith("PASS_"),
            "Unvalidated MAG: " + accession
        )

        fasta = Path(record["fasta_gz_path"]).resolve()

        require(
            fasta.parent == fasta_directory,
            "Unexpected FASTA directory: " + accession
        )

        require(
            fasta.name == accession + ".fna.gz",
            "FASTA accession mismatch: " + accession
        )

        require(
            fasta.is_file(),
            "FASTA missing: " + accession
        )

        actual_hash = uncompressed_sha256(fasta)

        require(
            actual_hash == record["fna_sha256"],
            "FASTA SHA256 mismatch: " + accession
        )

        stats, vector = canonical_k6(fasta)

        require(
            vector.shape == (2080,),
            "Invalid feature dimension: " + accession
        )

        require(
            vector.dtype == np.float32,
            "Invalid feature dtype: " + accession
        )

        require(
            bool(np.isfinite(vector).all()),
            "Nonfinite feature values: " + accession
        )

        require(
            bool((vector >= 0).all()),
            "Negative feature values: " + accession
        )

        feature_sum = float(
            vector.sum(dtype=np.float64)
        )

        require(
            abs(feature_sum - 1.0) <= 1e-5,
            "Feature normalization failure: " + accession
        )

        require(
            stats["contigs"] >= 1
            and stats["valid_kmers"] >= 1,
            "Invalid sequence statistics: " + accession
        )

        require(
            (
                stats["acgt_bases"]
                + stats["ambiguous_bases"]
                == stats["total_bases"]
            ),
            "Base-count mismatch: " + accession
        )

        vectors.append(vector)

        metadata.append({
            "row_index": index,
            "accession": accession,
            "biosample": record["biosample"],
            "quality_bin": record["quality_bin"],
            "reference_target_level":
                record["reference_target_level"],
            "completeness": record["completeness"],
            "contamination": record["contamination"],
            "gtdb_species": record["gtdb_species"],
            "gtdb_genus": record["gtdb_genus"],
            "gtdb_family": record["gtdb_family"],
            "fna_sha256": actual_hash,
            "fasta_gz_path": str(fasta)
        })

        qc_rows.append({
            "row_index": index,
            "accession": accession,
            "quality_bin": record["quality_bin"],
            "contigs": stats["contigs"],
            "total_bases": stats["total_bases"],
            "acgt_bases": stats["acgt_bases"],
            "ambiguous_bases": stats["ambiguous_bases"],
            "valid_kmers": stats["valid_kmers"],
            "feature_sum": f"{feature_sum:.12g}",
            "feature_status": "PASS"
        })

        if (index + 1) % 10 == 0 or index == 0:
            print(
                f"[{index + 1:02d}/80] PASS "
                f"{accession} "
                f"bin={record['quality_bin']} "
                f"contigs={stats['contigs']} "
                f"valid_kmers={stats['valid_kmers']}",
                flush=True
            )

    X = np.stack(vectors).astype(
        np.float32, copy=False
    )

    require(
        X.shape == (80, 2080),
        "Final matrix shape mismatch"
    )

    require(
        bool(np.isfinite(X).all()),
        "Invalid final matrix"
    )

    bin_counts = Counter(
        r["quality_bin"] for r in metadata
    )

    require(
        all(bin_counts[b] == 20 for b in BINS),
        "Completeness groups changed"
    )

    level_counts = Counter(
        r["reference_target_level"]
        for r in metadata
    )

    require(
        level_counts == Counter({
            "species": 40,
            "genus": 24,
            "family": 16
        }),
        "Taxonomic target counts changed"
    )

    OUT.mkdir(parents=True, exist_ok=False)

    matrix_path = OUT / (
        "E13_MAG_CANONICAL_K6_80x2080.float32.npy"
    )

    np.save(matrix_path, X)

    save_tsv(
        OUT / "E13_MAG_FEATURE_ROWS.tsv",
        metadata,
        list(metadata[0])
    )

    save_tsv(
        OUT / "E13_MAG_FEATURE_QC.tsv",
        qc_rows,
        list(qc_rows[0])
    )

    elapsed = time.perf_counter() - start

    config = {
        "experiment": "E13 real MAG validation",
        "feature_version": "canonical_k6_v1",
        "feature_shape": [80, 2080],
        "feature_dtype": "float32",
        "n_mag": 80,
        "source_mag_manifest_sha256": sha256(SOURCE),
        "validated_w07_script_sha256": sha256(W07),
        "feature_matrix_sha256": sha256(matrix_path),
        "quality_bin_counts": dict(bin_counts),
        "taxonomic_target_counts": dict(level_counts),
        "elapsed_feature_extraction_seconds": elapsed,
        "training_performed": False,
        "model_inference_performed": False,
        "reference_genomes_recomputed": False,
        "reference_near_duplicate_screen": "PENDING",
        "independent_taxonomy_validation": "PENDING"
    }

    (
        OUT / "E13_MAG_FEATURE_CONFIG.json"
    ).write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (
        OUT / "W09_STATUS.txt"
    ).write_text(
        "W09_MAG_FEATURE_EXTRACTION=PASS\n"
        "MAG_COUNT=80\n"
        "FEATURE_DIMENSION=2080\n"
        "DTYPE=float32\n"
        "SOURCE_W07_PARITY=PASS\n"
        "MODEL_TRAINING=NO\n"
        "MODEL_INFERENCE=NOT_YET\n"
        "REFERENCE_DUPLICATE_SCREEN=PENDING\n",
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("W09_MAG_FEATURE_EXTRACTION=PASS")
    print("Matrix:", X.shape, X.dtype)
    print("Completeness bins:", dict(bin_counts))
    print("Taxonomic targets:", dict(level_counts))
    print("Feature matrix SHA256:", sha256(matrix_path))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
