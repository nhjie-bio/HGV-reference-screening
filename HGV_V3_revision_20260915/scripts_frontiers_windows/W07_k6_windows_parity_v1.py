#!/usr/bin/env python3
import csv
import gzip
import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "03_features/canonical_k6_full3154_v1"
MASTER = ROOT / (
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)
GENOMES = ROOT / "02_genomes"
OUT = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W07_feature_parity"
)


def require(ok, detail):
    if not ok:
        raise RuntimeError("STOP: " + detail)


def read_tsv(path):
    require(path.is_file(), "Missing " + str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        rows = list(reader)
    require(
        bool(fields) and not any(None in r for r in rows),
        "Malformed " + str(path)
    )
    return rows


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def revcomp6(x):
    rc = 0
    for _ in range(6):
        rc = (rc << 2) | (3 - (x & 3))
        x >>= 2
    return rc


VOCAB_CODES = [
    x for x in range(4096) if x <= revcomp6(x)
]
require(len(VOCAB_CODES) == 2080, "Vocabulary size")

INDEX = {x: i for i, x in enumerate(VOCAB_CODES)}
CODE_TO_INDEX = [
    INDEX[min(x, revcomp6(x))]
    for x in range(4096)
]

BASE_CODE = [-1] * 256
for base, code in (
    (b"A", 0), (b"C", 1),
    (b"G", 2), (b"T", 3)
):
    BASE_CODE[base[0]] = code
    BASE_CODE[base[0] + 32] = code


def canonical_k6(path):
    with path.open("rb") as f:
        magic = f.read(2)

    opener = gzip.open if magic == b"\x1f\x8b" else open

    counts = [0] * 2080
    contigs = total = acgt = ambiguous = valid = 0
    rolling = valid_run = 0

    with opener(path, "rb") as f:
        for line in f:

            if line.startswith(b">"):
                contigs += 1
                rolling = valid_run = 0
                continue

            if contigs == 0:
                continue

            for char in line:

                if char in (10, 13, 32, 9):
                    continue

                total += 1
                code = BASE_CODE[char]

                if code < 0:
                    ambiguous += 1
                    rolling = valid_run = 0
                    continue

                acgt += 1
                rolling = ((rolling << 2) | code) & 4095
                valid_run += 1

                if valid_run >= 6:
                    counts[CODE_TO_INDEX[rolling]] += 1
                    valid += 1

    require(
        contigs > 0 and valid > 0,
        "Invalid FASTA: " + str(path)
    )

    # Match original C++ setprecision(10).
    vector = np.fromiter(
        (
            float(format(count / valid, ".10g"))
            for count in counts
        ),
        dtype=np.float32,
        count=2080
    )

    stats = {
        "contigs": contigs,
        "total_bases": total,
        "acgt_bases": acgt,
        "ambiguous_bases": ambiguous,
        "valid_kmers": valid
    }

    return stats, vector


def remap_fasta(raw):
    path = str(raw or "").strip().replace("\\", "/")

    if "/02_genomes/" not in path:
        return None

    rest = path.split("/02_genomes/", 1)[1]
    resolved = GENOMES.joinpath(*rest.split("/"))

    return resolved if resolved.is_file() else None


def write_tsv(path, data, fields):
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
        writer.writerows(data)


def main():
    print("=" * 72)
    print("W07 - WINDOWS CANONICAL K6 FEATURE PARITY")
    print("=" * 72)

    for p in (
        FEATURE_DIR / "canonical_k6_rows.tsv",
        FEATURE_DIR / "canonical_k6_extraction_qc.tsv",
        FEATURE_DIR / "canonical_k6_vocabulary.tsv",
        FEATURE_DIR / "canonical_k6_features.float32.npy",
        MASTER
    ):
        require(p.is_file(), "Missing input: " + str(p))

    OUT.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(
        FEATURE_DIR / "canonical_k6_rows.tsv"
    )
    qc = {
        r["assembly_accession"]: r
        for r in read_tsv(
            FEATURE_DIR / "canonical_k6_extraction_qc.tsv"
        )
    }
    manifest = {
        r["assembly_accession_final"]: r
        for r in read_tsv(MASTER)
    }
    vocab = read_tsv(
        FEATURE_DIR / "canonical_k6_vocabulary.tsv"
    )

    X = np.load(
        FEATURE_DIR / "canonical_k6_features.float32.npy",
        mmap_mode="r",
        allow_pickle=False
    )

    require(
        X.shape == (3154, 2080)
        and X.dtype == np.float32,
        "Frozen matrix shape/dtype mismatch"
    )
    require(
        len(rows) == 3154
        and len(manifest) == 3154
        and len(qc) == 3154,
        "Rows or QC count mismatch"
    )
    require(
        len(vocab) == 2080,
        "Stored vocabulary length mismatch"
    )

    for i, rec in enumerate(vocab):
        kmer = "".join(
            "ACGT"[(VOCAB_CODES[i] >> shift) & 3]
            for shift in (10, 8, 6, 4, 2, 0)
        )
        require(
            int(rec["feature_index"]) == i
            and rec["canonical_6mer"] == kmer,
            "Vocabulary mismatch at " + str(i)
        )

    for i, rec in enumerate(rows):
        require(
            int(rec["row_index"]) == i,
            "Feature row index mismatch"
        )
        require(
            rec["assembly_accession"] in manifest,
            "Unknown accession at row " + str(i)
        )

    print(
        "[PASS] Original matrix, vocabulary "
        "and accession-row mapping"
    )

    available = []

    for i, rec in enumerate(rows):
        path = remap_fasta(rec.get("fasta_path", ""))
        if path is not None:
            available.append((i, rec, path))

    if len(available) < 3:
        print("[INFO] Searching relocated FASTA files")

        pattern = re.compile(r"GC[AF]_\d+\.\d+")
        indexed = {}

        for path in GENOMES.rglob("*"):
            if not path.is_file():
                continue

            if not re.search(
                r"\.(?:fa|fna|fasta)(?:\.gz)?$",
                path.name, re.I
            ):
                continue

            for accession in pattern.findall(str(path)):
                if (
                    accession in manifest
                    and accession not in indexed
                ):
                    indexed[accession] = path

        available = [
            (i, rec, indexed[rec["assembly_accession"]])
            for i, rec in enumerate(rows)
            if rec["assembly_accession"] in indexed
        ]

    require(
        len(available) >= 3,
        "Need three accession-matched original FASTA files"
    )

    selected = []
    used = set()

    for cohort in (
        "A", "B1",
        "B2_seen_genus", "B2_unseen_genus"
    ):
        options = [
            item for item in available
            if item[1]["final_cohort"] == cohort
            and item[1]["assembly_accession"] not in used
        ]

        if options and len(selected) < 3:
            item = options[0]
            selected.append(item)
            used.add(item[1]["assembly_accession"])

    for item in available:
        if len(selected) >= 3:
            break

        if item[1]["assembly_accession"] not in used:
            selected.append(item)
            used.add(item[1]["assembly_accession"])

    results = []
    all_pass = True

    for index, row, fasta in selected:
        acc = row["assembly_accession"]

        expected_sha = manifest[acc].get(
            "fasta_file_sha256", ""
        ).strip().lower()

        actual_sha = sha256(fasta)
        hash_ok = (
            not expected_sha or actual_sha == expected_sha
        )

        stats, vector = canonical_k6(fasta)
        cached = np.asarray(X[index])

        different = int(np.count_nonzero(
            vector.view(np.uint32)
            != cached.view(np.uint32)
        ))

        max_delta = float(np.max(np.abs(
            vector.astype(np.float64)
            - cached.astype(np.float64)
        )))

        qc_ok = all(
            int(qc[acc][key]) == stats[key]
            for key in stats
        )

        sum_ok = (
            abs(float(vector.sum(dtype=np.float64)) - 1)
            <= 1e-5
        )

        passed = (
            hash_ok and qc_ok and sum_ok
            and different == 0
        )

        all_pass = all_pass and passed

        results.append({
            "accession": acc,
            "cohort": row["final_cohort"],
            "row_index": index,
            "file_sha256_match":
                "PASS" if hash_ok else "FAIL",
            "original_qc_match":
                "PASS" if qc_ok else "FAIL",
            "bitwise_2080_match":
                "PASS" if different == 0 else "FAIL",
            "differing_float32_features": different,
            "max_abs_feature_delta":
                format(max_delta, ".10g"),
            "valid_kmers": stats["valid_kmers"],
            "result": "PASS" if passed else "FAIL",
            "windows_fasta_path": str(fasta)
        })

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{acc} {row['final_cohort']}: "
            f"SHA={hash_ok}, QC={qc_ok}, "
            f"bitwise_differences={different}, "
            f"max_delta={max_delta:.3g}"
        )

    output = OUT / (
        "W07_ORIGINAL_FASTA_FEATURE_PARITY.tsv"
    )

    write_tsv(
        output,
        results,
        list(results[0])
    )

    status = "PASS" if all_pass else "FAIL"

    (OUT / "W07_STATUS.txt").write_text(
        f"W07_FEATURE_PARITY={status}\n"
        f"Samples={len(results)}\n"
        "No models loaded or modified.\n"
        "No MAG downloaded.\n",
        encoding="utf-8"
    )

    print()
    print("W07_FEATURE_PARITY=" + status)
    print("Output:", output)

    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
