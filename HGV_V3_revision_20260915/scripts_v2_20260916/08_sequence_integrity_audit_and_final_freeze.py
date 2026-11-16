#!/usr/bin/env python3

import csv
import gzip
import hashlib
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE = Path(
    "HGV_V3_revision_20260915"
)

ORIGINAL_FREEZE = (
    BASE
    / "01_manifest"
    / "frozen_pre_download_20260916"
)

REPLACEMENT_DIR = (
    BASE
    / "01_manifest"
    / "replacement_round1_20260916"
)

UPDATED_MANIFEST = (
    REPLACEMENT_DIR
    / "updated_primary_manifest_round1.tsv"
)

REPLACEMENT_PLAN = (
    REPLACEMENT_DIR
    / "replacement_plan_round1.tsv"
)

REPLACEMENT_SHA_TABLE = (
    REPLACEMENT_DIR
    / "replacement_round1_sha256.tsv"
)

GENOME_ROOT = (
    BASE
    / "02_genomes"
)

AUDIT_DIR = (
    BASE
    / "02_genomes"
    / "sequence_audit_round1"
)

FINAL_FREEZE = (
    BASE
    / "01_manifest"
    / "final_sequence_freeze_20260916"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


WORKERS = int(
    os.environ.get(
        "HGV_SEQ_AUDIT_WORKERS",
        "24",
    )
)

if WORKERS < 1:
    WORKERS = 1

if WORKERS > 32:
    WORKERS = 32


# ============================================================
# QC thresholds
# ============================================================

MIN_GENOME_SIZE = 1_000_000
MAX_GENOME_SIZE = 10_000_000

MAX_CONTIGS = 200
MIN_N50 = 20_000

MAX_AMBIGUOUS_FRACTION = 0.005


# ============================================================
# Outputs
# ============================================================

SEQ_AUDIT_OUT = (
    AUDIT_DIR
    / "sequence_integrity_audit.tsv"
)

DUPLICATE_OUT = (
    AUDIT_DIR
    / "sequence_duplicate_groups.tsv"
)

CHECKS_OUT = (
    AUDIT_DIR
    / "sequence_audit_checks.tsv"
)

REPORT_OUT = (
    AUDIT_DIR
    / "sequence_audit_report.txt"
)

MASTER_AUDITED_OUT = (
    AUDIT_DIR
    / "master_manifest_v3_sequence_audited.tsv"
)

FASTA_MAP_OUT = (
    AUDIT_DIR
    / "final_accession_fasta_map.tsv"
)


# ============================================================
# Helpers
# ============================================================

def clean(x):

    if x is None:
        return ""

    x = str(x).strip()

    if x.lower() in {
        "",
        "na",
        "nan",
        "none",
        "null",
    }:
        return ""

    return x


def as_int(x, default=None):

    try:
        return int(float(x))
    except Exception:
        return default


def as_float(x, default=None):

    try:
        return float(x)
    except Exception:
        return default


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def read_tsv(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:

        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def calc_n50(lengths):

    if not lengths:
        return 0

    total = sum(lengths)
    threshold = total / 2.0

    cumulative = 0

    for length in sorted(
        lengths,
        reverse=True,
    ):

        cumulative += length

        if cumulative >= threshold:
            return length

    return 0


# ============================================================
# Sequence normalization
#
# canonical_contigset_sha256 is invariant to:
#
# 1. FASTA contig order
# 2. reverse-complement orientation of each contig
#
# It is intentionally NOT circular-origin/rotation invariant.
# ============================================================

RC_TABLE = str.maketrans(
    "ACGTRYKMSWBDHVN",
    "TGCAYRMKSWVHDBN",
)

VALID_IUPAC = set(
    "ACGTRYKMSWBDHVN"
)


def analyze_fasta(job):

    accession, fasta_path = job

    path = Path(
        fasta_path
    )

    compressed_sha256 = (
        sha256_file(path)
    )

    if str(path).endswith(
        ".gz"
    ):
        opener = gzip.open
    else:
        opener = open


    total_length = 0

    counts = Counter()

    contig_lengths = []

    invalid_count = 0
    empty_contigs = 0

    ordered_contig_tokens = []
    canonical_contig_tokens = []


    def consume_sequence(seq):

        nonlocal total_length
        nonlocal invalid_count
        nonlocal empty_contigs

        seq = (
            "".join(
                seq.split()
            )
            .upper()
        )

        if not seq:

            empty_contigs += 1
            return


        length = len(seq)

        total_length += length

        contig_lengths.append(
            length
        )


        c = Counter(seq)

        counts.update(c)


        invalid_count += sum(
            count
            for base, count
            in c.items()
            if base not in VALID_IUPAC
        )


        seq_bytes = seq.encode(
            "ascii",
            errors="strict",
        )

        forward_hash = (
            hashlib.sha256(
                seq_bytes
            ).hexdigest()
        )


        reverse_seq = (
            seq.translate(
                RC_TABLE
            )[::-1]
        )

        reverse_hash = (
            hashlib.sha256(
                reverse_seq.encode(
                    "ascii",
                    errors="strict",
                )
            ).hexdigest()
        )


        canonical_hash = min(
            forward_hash,
            reverse_hash,
        )


        ordered_contig_tokens.append(
            f"{length}:{forward_hash}"
        )

        canonical_contig_tokens.append(
            f"{length}:{canonical_hash}"
        )


    with opener(
        path,
        "rt",
        encoding="ascii",
        errors="strict",
    ) as handle:

        parts = []
        seen_header = False

        for raw_line in handle:

            if raw_line.startswith(
                ">"
            ):

                if seen_header:

                    consume_sequence(
                        "".join(parts)
                    )

                seen_header = True
                parts = []

            else:

                parts.append(
                    raw_line.strip()
                )


        if seen_header:

            consume_sequence(
                "".join(parts)
            )


    contig_count = len(
        contig_lengths
    )


    ordered_hasher = (
        hashlib.sha256()
    )

    for token in (
        ordered_contig_tokens
    ):

        ordered_hasher.update(
            token.encode(
                "ascii"
            )
        )

        ordered_hasher.update(
            b"\n"
        )


    canonical_hasher = (
        hashlib.sha256()
    )

    for token in sorted(
        canonical_contig_tokens
    ):

        canonical_hasher.update(
            token.encode(
                "ascii"
            )
        )

        canonical_hasher.update(
            b"\n"
        )


    a = counts.get(
        "A",
        0,
    )

    c = counts.get(
        "C",
        0,
    )

    g = counts.get(
        "G",
        0,
    )

    t = counts.get(
        "T",
        0,
    )

    n = counts.get(
        "N",
        0,
    )


    acgt = (
        a + c + g + t
    )

    ambiguous = (
        total_length
        -
        acgt
    )


    if total_length > 0:

        gc_percentage = (
            100.0
            *
            (g + c)
            /
            total_length
        )

        ambiguous_fraction = (
            ambiguous
            /
            total_length
        )

        n_fraction = (
            n
            /
            total_length
        )

    else:

        gc_percentage = 0.0
        ambiguous_fraction = 1.0
        n_fraction = 1.0


    return {
        "assembly_accession":
            accession,

        "fasta_path":
            str(path),

        "fasta_size_bytes":
            path.stat().st_size,

        "fasta_file_sha256":
            compressed_sha256,

        "actual_genome_size":
            total_length,

        "actual_contig_count":
            contig_count,

        "actual_n50_contigs":
            calc_n50(
                contig_lengths
            ),

        "actual_gc_percentage":
            gc_percentage,

        "actual_A":
            a,

        "actual_C":
            c,

        "actual_G":
            g,

        "actual_T":
            t,

        "actual_N":
            n,

        "actual_ambiguous_bases":
            ambiguous,

        "actual_ambiguous_fraction":
            ambiguous_fraction,

        "actual_N_fraction":
            n_fraction,

        "invalid_base_count":
            invalid_count,

        "empty_contigs":
            empty_contigs,

        "sequence_order_sha256":
            ordered_hasher.hexdigest(),

        "canonical_contigset_sha256":
            canonical_hasher.hexdigest(),
    }


# ============================================================
# Audit infrastructure
# ============================================================

checks = []


def add_check(
    name,
    passed,
    expected,
    observed,
    detail="",
):

    checks.append(
        {
            "check":
                name,

            "status":
                "PASS"
                if passed
                else "FAIL",

            "expected":
                str(expected),

            "observed":
                str(observed),

            "detail":
                str(detail),
        }
    )


# ============================================================
# Verify provenance
# ============================================================

def verify_original_freeze():

    checksum_file = (
        ORIGINAL_FREEZE
        / "PRE_DOWNLOAD_FREEZE_SHA256.txt"
    )

    if not checksum_file.exists():

        raise RuntimeError(
            "Missing original freeze checksum file."
        )


    with open(
        checksum_file,
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                None,
                1,
            )

            if len(parts) != 2:

                raise RuntimeError(
                    "Malformed pre-download checksum line: "
                    + line
                )


            expected = parts[0]

            filename = (
                parts[1]
                .strip()
                .lstrip("*")
            )

            path = (
                ORIGINAL_FREEZE
                / filename
            )


            if not path.exists():

                raise RuntimeError(
                    f"Frozen file missing: {path}"
                )


            actual = (
                sha256_file(path)
            )


            if actual != expected:

                raise RuntimeError(
                    "Original metadata freeze changed:\n"
                    f"{path}\n"
                    f"expected={expected}\n"
                    f"actual={actual}"
                )


def verify_replacement_provenance():

    if not (
        REPLACEMENT_SHA_TABLE.exists()
    ):

        raise RuntimeError(
            "Missing replacement SHA table."
        )


    with open(
        REPLACEMENT_SHA_TABLE,
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:

            path = Path(
                row["file"]
            )

            expected = (
                row["sha256"]
            )

            if not path.exists():

                raise RuntimeError(
                    f"Replacement provenance file missing: {path}"
                )


            actual = (
                sha256_file(path)
            )


            if actual != expected:

                raise RuntimeError(
                    "Replacement provenance changed:\n"
                    f"{path}\n"
                    f"expected={expected}\n"
                    f"actual={actual}"
                )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )

    print(
        "HGV-V3 PHASE 2C"
    )

    print(
        "Final sequence integrity audit"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"Workers: {WORKERS}"
    )

    print()


    # --------------------------------------------------------
    # 1. Provenance
    # --------------------------------------------------------

    print(
        "[1/8] Verifying provenance..."
    )


    verify_original_freeze()

    verify_replacement_provenance()


    print(
        "[PASS] Original freeze and replacement provenance unchanged."
    )


    # --------------------------------------------------------
    # 2. Read final provisional manifest
    # --------------------------------------------------------

    print()

    print(
        "[2/8] Reading provisional final manifest..."
    )


    rows = read_tsv(
        UPDATED_MANIFEST
    )


    add_check(
        "final_manifest_rows",
        len(rows) == 3200,
        3200,
        len(rows),
    )


    accession_list = [
        clean(
            r["assembly_accession"]
        )
        for r in rows
    ]


    biosample_list = [
        clean(
            r["biosample"]
        )
        for r in rows
    ]


    add_check(
        "final_accessions_unique",
        len(
            set(
                accession_list
            )
        ) == 3200,
        3200,
        len(
            set(
                accession_list
            )
        ),
    )


    add_check(
        "final_biosamples_unique",
        len(
            set(
                biosample_list
            )
        ) == 3200,
        3200,
        len(
            set(
                biosample_list
            )
        ),
    )


    replacements = [
        r
        for r in rows
        if (
            r.get(
                "replacement_status"
            )
            ==
            "replacement_round1"
        )
    ]


    retained = [
        r
        for r in rows
        if (
            r.get(
                "replacement_status"
            )
            ==
            "original_primary_retained"
        )
    ]


    add_check(
        "round1_replacement_count",
        len(
            replacements
        ) == 9,
        9,
        len(
            replacements
        ),
    )


    add_check(
        "retained_original_count",
        len(
            retained
        ) == 3191,
        3191,
        len(
            retained
        ),
    )


    # --------------------------------------------------------
    # 3. Re-audit final metadata cohort structure
    # --------------------------------------------------------

    A = [
        r
        for r in rows
        if r["cohort"] == "A"
    ]

    B1 = [
        r
        for r in rows
        if r["cohort"] == "B1"
    ]

    B2 = [
        r
        for r in rows
        if r["cohort"] == "B2"
    ]

    B2_SEEN = [
        r
        for r in B2
        if (
            r["cohort_subtype"]
            ==
            "unseen_species_seen_genus"
        )
    ]

    B2_UNSEEN = [
        r
        for r in B2
        if (
            r["cohort_subtype"]
            ==
            "unseen_species_unseen_genus"
        )
    ]


    def sset(x):

        return {
            r[
                "gtdb_species_cluster"
            ]
            for r in x
        }


    def gset(x):

        return {
            r[
                "gtdb_genus"
            ]
            for r in x
        }


    def fset(x):

        return {
            r[
                "gtdb_family"
            ]
            for r in x
        }


    A_SP = sset(A)
    A_GEN = gset(A)

    B1_SP = sset(B1)

    B2_SP = sset(B2)

    B2_SEEN_SP = sset(
        B2_SEEN
    )

    B2_UNSEEN_SP = sset(
        B2_UNSEEN
    )

    B2_SEEN_GEN = gset(
        B2_SEEN
    )

    B2_UNSEEN_GEN = gset(
        B2_UNSEEN
    )


    add_check(
        "A_genomes",
        len(A) == 2400,
        2400,
        len(A),
    )

    add_check(
        "A_species",
        len(A_SP) == 200,
        200,
        len(A_SP),
    )

    add_check(
        "A_genera",
        len(A_GEN) == 100,
        100,
        len(A_GEN),
    )

    add_check(
        "A_families",
        len(
            fset(A)
        ) == 88,
        88,
        len(
            fset(A)
        ),
    )

    add_check(
        "B1_genomes",
        len(B1) == 400,
        400,
        len(B1),
    )

    add_check(
        "B1_species",
        len(B1_SP) == 40,
        40,
        len(B1_SP),
    )

    add_check(
        "B1_species_subset_A",
        B1_SP.issubset(
            A_SP
        ),
        True,
        B1_SP.issubset(
            A_SP
        ),
    )

    add_check(
        "B2_genomes",
        len(B2) == 400,
        400,
        len(B2),
    )

    add_check(
        "B2_species",
        len(B2_SP) == 40,
        40,
        len(B2_SP),
    )

    add_check(
        "A_B2_species_overlap",
        len(
            A_SP & B2_SP
        ) == 0,
        0,
        len(
            A_SP & B2_SP
        ),
    )

    add_check(
        "B2_seen_species",
        len(
            B2_SEEN_SP
        ) == 20,
        20,
        len(
            B2_SEEN_SP
        ),
    )

    add_check(
        "B2_unseen_species",
        len(
            B2_UNSEEN_SP
        ) == 20,
        20,
        len(
            B2_UNSEEN_SP
        ),
    )

    add_check(
        "B2_seen_genera_present_A",
        B2_SEEN_GEN.issubset(
            A_GEN
        ),
        True,
        B2_SEEN_GEN.issubset(
            A_GEN
        ),
    )

    add_check(
        "B2_unseen_genera_absent_A",
        len(
            B2_UNSEEN_GEN
            &
            A_GEN
        ) == 0,
        0,
        len(
            B2_UNSEEN_GEN
            &
            A_GEN
        ),
    )


    # --------------------------------------------------------
    # 4. Locate all FASTA files
    # --------------------------------------------------------

    print()

    print(
        "[3/8] Discovering downloaded FASTA files..."
    )


    fasta_map = defaultdict(
        list
    )


    for pattern in [
        "*_genomic.fna.gz",
        "*_genomic.fna",
    ]:

        for path in (
            GENOME_ROOT.rglob(
                pattern
            )
        ):

            accession = (
                path.parent.name
            )

            fasta_map[
                accession
            ].append(
                path
            )


    selected_accessions = set(
        accession_list
    )


    missing_fasta = sorted(
        acc
        for acc in selected_accessions
        if len(
            fasta_map.get(
                acc,
                [],
            )
        ) == 0
    )


    multiple_fasta = {
        acc: [
            str(x)
            for x in paths
        ]
        for acc, paths
        in fasta_map.items()
        if (
            acc
            in selected_accessions
            and
            len(paths) > 1
        )
    }


    mapped_accessions = {
        acc
        for acc in selected_accessions
        if len(
            fasta_map.get(
                acc,
                [],
            )
        ) == 1
    }


    extra_fasta_accessions = sorted(
        set(
            fasta_map
        )
        -
        selected_accessions
    )


    add_check(
        "selected_accessions_with_exactly_one_FASTA",
        len(
            mapped_accessions
        ) == 3200,
        3200,
        len(
            mapped_accessions
        ),
        (
            f"missing={len(missing_fasta)}, "
            f"multiple={len(multiple_fasta)}"
        ),
    )


    add_check(
        "selected_accessions_missing_FASTA",
        len(
            missing_fasta
        ) == 0,
        0,
        len(
            missing_fasta
        ),
        missing_fasta[:20],
    )


    add_check(
        "selected_accessions_multiple_FASTA",
        len(
            multiple_fasta
        ) == 0,
        0,
        len(
            multiple_fasta
        ),
        list(
            multiple_fasta.items()
        )[:10],
    )


    print(
        f"Selected accessions: {len(selected_accessions)}"
    )

    print(
        f"Exactly one FASTA: {len(mapped_accessions)}"
    )

    print(
        f"Missing FASTA: {len(missing_fasta)}"
    )

    print(
        f"Multiple FASTA: {len(multiple_fasta)}"
    )

    print(
        f"Unselected FASTA accessions present on disk: "
        f"{len(extra_fasta_accessions)}"
    )


    if (
        missing_fasta
        or
        multiple_fasta
    ):

        write_checks_and_report(
            checks,
            "FAIL",
            "FASTA mapping is incomplete or ambiguous.",
        )

        sys.exit(2)


    # --------------------------------------------------------
    # Write FASTA map
    # --------------------------------------------------------

    with open(
        FASTA_MAP_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "assembly_accession",
                "fasta_path",
            ]
        )

        for accession in sorted(
            selected_accessions
        ):

            writer.writerow(
                [
                    accession,
                    str(
                        fasta_map[
                            accession
                        ][0]
                    ),
                ]
            )


    # --------------------------------------------------------
    # 5. Analyze all 3200 FASTA
    # --------------------------------------------------------

    print()

    print(
        "[4/8] Recomputing sequence statistics and hashes..."
    )


    jobs = [
        (
            accession,
            str(
                fasta_map[
                    accession
                ][0]
            ),
        )
        for accession
        in accession_list
    ]


    results = []


    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        for i, result in enumerate(
            executor.map(
                analyze_fasta,
                jobs,
                chunksize=1,
            ),
            start=1,
        ):

            results.append(
                result
            )

            if (
                i % 100 == 0
                or
                i == 3200
            ):

                print(
                    f"Analyzed {i:,}/3,200 genomes"
                )


    result_by_accession = {
        r[
            "assembly_accession"
        ]: r
        for r in results
    }


    # --------------------------------------------------------
    # 6. Sequence QC and metadata comparison
    # --------------------------------------------------------

    print()

    print(
        "[5/8] Auditing sequence QC..."
    )


    audit_rows = []

    hard_fail_genomes = []

    metadata_warning_genomes = []


    for row in rows:

        accession = row[
            "assembly_accession"
        ]

        seq = result_by_accession[
            accession
        ]


        actual_size = seq[
            "actual_genome_size"
        ]

        actual_contigs = seq[
            "actual_contig_count"
        ]

        actual_n50 = seq[
            "actual_n50_contigs"
        ]

        actual_amb = seq[
            "actual_ambiguous_fraction"
        ]


        reasons = []


        if (
            actual_size
            <
            MIN_GENOME_SIZE
            or
            actual_size
            >
            MAX_GENOME_SIZE
        ):

            reasons.append(
                "actual_genome_size"
            )


        if (
            actual_contigs
            >
            MAX_CONTIGS
        ):

            reasons.append(
                "actual_contig_count"
            )


        if (
            actual_n50
            <
            MIN_N50
        ):

            reasons.append(
                "actual_n50"
            )


        if (
            actual_amb
            >
            MAX_AMBIGUOUS_FRACTION
        ):

            reasons.append(
                "actual_ambiguous_fraction"
            )


        if (
            seq[
                "invalid_base_count"
            ]
            > 0
        ):

            reasons.append(
                "invalid_sequence_characters"
            )


        if (
            seq[
                "empty_contigs"
            ]
            > 0
        ):

            reasons.append(
                "empty_contig"
            )


        if (
            actual_size <= 0
        ):

            reasons.append(
                "zero_length"
            )


        meta_size = as_int(
            row.get(
                "genome_size"
            )
        )

        meta_contigs = as_int(
            row.get(
                "contig_count"
            )
        )

        meta_n50 = as_int(
            row.get(
                "n50_contigs"
            )
        )


        if (
            meta_size
            is not None
            and
            meta_size > 0
        ):

            size_delta = (
                actual_size
                -
                meta_size
            )

            size_delta_fraction = (
                size_delta
                /
                meta_size
            )

        else:

            size_delta = ""
            size_delta_fraction = ""


        if (
            meta_contigs
            is not None
        ):

            contig_delta = (
                actual_contigs
                -
                meta_contigs
            )

        else:

            contig_delta = ""


        if (
            meta_n50
            is not None
        ):

            n50_delta = (
                actual_n50
                -
                meta_n50
            )

        else:

            n50_delta = ""


        warnings = []


        if (
            size_delta_fraction
            != ""
            and
            abs(
                size_delta_fraction
            )
            > 0.01
        ):

            warnings.append(
                "metadata_genome_size_delta_gt1pct"
            )


        if (
            contig_delta
            != ""
            and
            contig_delta != 0
        ):

            warnings.append(
                "metadata_contig_count_difference"
            )


        if (
            n50_delta
            != ""
            and
            n50_delta != 0
        ):

            warnings.append(
                "metadata_n50_difference"
            )


        if reasons:

            hard_fail_genomes.append(
                (
                    accession,
                    ";".join(
                        reasons
                    ),
                )
            )


        if warnings:

            metadata_warning_genomes.append(
                (
                    accession,
                    ";".join(
                        warnings
                    ),
                )
            )


        try:

            relative_path = str(
                Path(
                    seq[
                        "fasta_path"
                    ]
                ).relative_to(
                    BASE
                )
            )

        except ValueError:

            relative_path = (
                seq[
                    "fasta_path"
                ]
            )


        audit = {
            "assembly_accession":
                accession,

            "cohort":
                row["cohort"],

            "cohort_subtype":
                row[
                    "cohort_subtype"
                ],

            "gtdb_species_cluster":
                row[
                    "gtdb_species_cluster"
                ],

            "gtdb_species":
                row[
                    "gtdb_species"
                ],

            "gtdb_genus":
                row[
                    "gtdb_genus"
                ],

            "gtdb_family":
                row[
                    "gtdb_family"
                ],

            "replacement_status":
                row[
                    "replacement_status"
                ],

            "original_primary_accession":
                row[
                    "original_primary_accession"
                ],

            "fasta_relpath":
                relative_path,

            "fasta_size_bytes":
                seq[
                    "fasta_size_bytes"
                ],

            "fasta_file_sha256":
                seq[
                    "fasta_file_sha256"
                ],

            "actual_genome_size":
                actual_size,

            "actual_contig_count":
                actual_contigs,

            "actual_n50_contigs":
                actual_n50,

            "actual_gc_percentage":
                f"{seq['actual_gc_percentage']:.8f}",

            "actual_A":
                seq[
                    "actual_A"
                ],

            "actual_C":
                seq[
                    "actual_C"
                ],

            "actual_G":
                seq[
                    "actual_G"
                ],

            "actual_T":
                seq[
                    "actual_T"
                ],

            "actual_N":
                seq[
                    "actual_N"
                ],

            "actual_ambiguous_bases":
                seq[
                    "actual_ambiguous_bases"
                ],

            "actual_ambiguous_fraction":
                f"{seq['actual_ambiguous_fraction']:.10f}",

            "actual_N_fraction":
                f"{seq['actual_N_fraction']:.10f}",

            "invalid_base_count":
                seq[
                    "invalid_base_count"
                ],

            "empty_contigs":
                seq[
                    "empty_contigs"
                ],

            "sequence_order_sha256":
                seq[
                    "sequence_order_sha256"
                ],

            "canonical_contigset_sha256":
                seq[
                    "canonical_contigset_sha256"
                ],

            "metadata_genome_size":
                meta_size
                if meta_size
                is not None
                else "",

            "metadata_contig_count":
                meta_contigs
                if meta_contigs
                is not None
                else "",

            "metadata_n50_contigs":
                meta_n50
                if meta_n50
                is not None
                else "",

            "metadata_size_delta":
                size_delta,

            "metadata_size_delta_fraction":
                (
                    f"{size_delta_fraction:.10f}"
                    if
                    size_delta_fraction
                    != ""
                    else ""
                ),

            "metadata_contig_delta":
                contig_delta,

            "metadata_n50_delta":
                n50_delta,

            "sequence_qc_pass":
                int(
                    len(
                        reasons
                    ) == 0
                ),

            "sequence_qc_fail_reasons":
                ";".join(
                    reasons
                ),

            "metadata_consistency_warnings":
                ";".join(
                    warnings
                ),
        }


        audit_rows.append(
            audit
        )


    add_check(
        "sequence_QC_failures",
        len(
            hard_fail_genomes
        ) == 0,
        0,
        len(
            hard_fail_genomes
        ),
        hard_fail_genomes[:20],
    )


    add_check(
        "invalid_sequence_character_genomes",
        sum(
            1
            for x
            in results
            if x[
                "invalid_base_count"
            ] > 0
        ) == 0,
        0,
        sum(
            1
            for x
            in results
            if x[
                "invalid_base_count"
            ] > 0
        ),
    )


    add_check(
        "empty_contig_genomes",
        sum(
            1
            for x
            in results
            if x[
                "empty_contigs"
            ] > 0
        ) == 0,
        0,
        sum(
            1
            for x
            in results
            if x[
                "empty_contigs"
            ] > 0
        ),
    )


    # --------------------------------------------------------
    # Write sequence audit table
    # --------------------------------------------------------

    AUDIT_FIELDS = list(
        audit_rows[0].keys()
    )


    with open(
        SEQ_AUDIT_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=AUDIT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )


    # --------------------------------------------------------
    # 7. Duplicate sequence hashes
    # --------------------------------------------------------

    print()

    print(
        "[6/8] Checking exact normalized sequence duplicates..."
    )


    by_canonical = defaultdict(
        list
    )

    by_order = defaultdict(
        list
    )


    audit_by_accession = {
        r[
            "assembly_accession"
        ]: r
        for r in audit_rows
    }


    for r in audit_rows:

        by_canonical[
            r[
                "canonical_contigset_sha256"
            ]
        ].append(
            r
        )

        by_order[
            r[
                "sequence_order_sha256"
            ]
        ].append(
            r
        )


    canonical_duplicates = {
        h: members
        for h, members
        in by_canonical.items()
        if len(
            members
        ) > 1
    }


    ordered_duplicates = {
        h: members
        for h, members
        in by_order.items()
        if len(
            members
        ) > 1
    }


    cross_cohort_duplicate_groups = 0


    for h, members in (
        canonical_duplicates.items()
    ):

        cohorts = {
            x[
                "cohort"
            ]
            for x in members
        }

        if len(
            cohorts
        ) > 1:

            cross_cohort_duplicate_groups += 1


    with open(
        DUPLICATE_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "duplicate_type",
                "sha256",
                "n_genomes",
                "accessions",
                "cohorts",
                "species_clusters",
                "genera",
            ]
        )


        for h, members in sorted(
            canonical_duplicates.items()
        ):

            writer.writerow(
                [
                    "canonical_contigset",
                    h,
                    len(
                        members
                    ),
                    ";".join(
                        sorted(
                            x[
                                "assembly_accession"
                            ]
                            for x
                            in members
                        )
                    ),
                    ";".join(
                        sorted(
                            {
                                x[
                                    "cohort"
                                ]
                                for x
                                in members
                            }
                        )
                    ),
                    ";".join(
                        sorted(
                            {
                                x[
                                    "gtdb_species_cluster"
                                ]
                                for x
                                in members
                            }
                        )
                    ),
                    ";".join(
                        sorted(
                            {
                                x[
                                    "gtdb_genus"
                                ]
                                for x
                                in members
                            }
                        )
                    ),
                ]
            )


    add_check(
        "canonical_sequence_duplicate_groups",
        len(
            canonical_duplicates
        ) == 0,
        0,
        len(
            canonical_duplicates
        ),
    )


    add_check(
        "canonical_sequence_hashes_unique",
        len(
            by_canonical
        ) == 3200,
        3200,
        len(
            by_canonical
        ),
    )


    add_check(
        "cross_cohort_sequence_duplicate_groups",
        cross_cohort_duplicate_groups
        == 0,
        0,
        cross_cohort_duplicate_groups,
    )


    print(
        f"Sequence-order duplicate groups: "
        f"{len(ordered_duplicates)}"
    )

    print(
        f"Canonical contig-set duplicate groups: "
        f"{len(canonical_duplicates)}"
    )

    print(
        f"Cross-cohort canonical duplicate groups: "
        f"{cross_cohort_duplicate_groups}"
    )


    # --------------------------------------------------------
    # Build audited master manifest
    # --------------------------------------------------------

    print()

    print(
        "[7/8] Building audited master manifest..."
    )


    master_rows = []


    for row in rows:

        accession = row[
            "assembly_accession"
        ]

        audit = (
            audit_by_accession[
                accession
            ]
        )


        merged = dict(
            row
        )


        for key, value in (
            audit.items()
        ):

            if key in {
                "assembly_accession",
                "cohort",
                "cohort_subtype",
                "gtdb_species_cluster",
                "gtdb_species",
                "gtdb_genus",
                "gtdb_family",
                "replacement_status",
                "original_primary_accession",
            }:
                continue

            merged[
                key
            ] = value


        master_rows.append(
            merged
        )


    master_fields = list(
        master_rows[0].keys()
    )


    with open(
        MASTER_AUDITED_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=master_fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            master_rows
        )


    # --------------------------------------------------------
    # Write checks
    # --------------------------------------------------------

    failed_checks = [
        x
        for x in checks
        if x[
            "status"
        ] == "FAIL"
    ]


    with open(
        CHECKS_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "check",
                "status",
                "expected",
                "observed",
                "detail",
            ],
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            checks
        )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total_actual_bases = sum(
        r[
            "actual_genome_size"
        ]
        for r in results
    )


    report = []

    report.append(
        "============================================================"
    )

    report.append(
        "HGV-V3 FINAL SEQUENCE INTEGRITY AUDIT"
    )

    report.append(
        "============================================================"
    )

    report.append("")

    report.append(
        f"Final logical genomes: {len(rows):,}"
    )

    report.append(
        f"Original primary retained: {len(retained):,}"
    )

    report.append(
        f"Round-1 replacements: {len(replacements):,}"
    )

    report.append("")

    report.append(
        f"FASTA successfully mapped: {len(mapped_accessions):,}"
    )

    report.append(
        f"Total actual sequence bases: {total_actual_bases:,}"
    )

    report.append(
        f"Approx sequence size: {total_actual_bases / 1e9:.3f} Gb"
    )

    report.append("")

    report.append(
        f"Sequence-QC failures: {len(hard_fail_genomes)}"
    )

    report.append(
        f"Metadata consistency warnings: "
        f"{len(metadata_warning_genomes)}"
    )

    report.append("")

    report.append(
        f"Sequence-order duplicate groups: "
        f"{len(ordered_duplicates)}"
    )

    report.append(
        f"Canonical contig-set duplicate groups: "
        f"{len(canonical_duplicates)}"
    )

    report.append(
        f"Cross-cohort canonical duplicate groups: "
        f"{cross_cohort_duplicate_groups}"
    )

    report.append("")

    report.append(
        "Canonical genome hash definition:"
    )

    report.append(
        "Each contig is hashed in forward and reverse-complement "
        "orientation; the smaller hash is retained, contig tokens "
        "are sorted, and the resulting contig set is SHA256-hashed."
    )

    report.append(
        "Therefore the duplicate test is invariant to contig order "
        "and per-contig strand orientation."
    )

    report.append(
        "It is not intended to be invariant to circular origin rotation."
    )

    report.append("")

    report.append(
        f"Audit checks: {len(checks)}"
    )

    report.append(
        f"PASS: {len(checks) - len(failed_checks)}"
    )

    report.append(
        f"FAIL: {len(failed_checks)}"
    )

    report.append("")


    if failed_checks:

        report.append(
            "FINAL SEQUENCE GATE: FAIL"
        )

        report.append("")

        report.append(
            "Do NOT start feature extraction."
        )

        report.append(
            "Any failed or duplicate genome must be handled "
            "through a documented replacement round."
        )

        report.append("")

        report.append(
            "FAILED CHECKS:"
        )

        for item in failed_checks:

            report.append(
                f"{item['check']}: "
                f"expected={item['expected']}; "
                f"observed={item['observed']}; "
                f"{item['detail']}"
            )

    else:

        report.append(
            "FINAL SEQUENCE GATE: PASS"
        )

        report.append("")

        report.append(
            "All 3,200 final genomes have exactly one FASTA, "
            "pass sequence-level QC, and have unique canonical "
            "sequence hashes."
        )

        report.append(
            "The cohort is approved for final sequence-level freeze."
        )


    report.append("")

    report.append(
        "============================================================"
    )


    report_text = (
        "\n".join(
            report
        )
        +
        "\n"
    )


    with open(
        REPORT_OUT,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            report_text
        )


    print(
        report_text
    )


    # --------------------------------------------------------
    # Stop if audit fails
    # --------------------------------------------------------

    if failed_checks:

        print(
            "Sequence audit failed."
        )

        print(
            f"Inspect: {CHECKS_OUT}"
        )

        print(
            f"Duplicates: {DUPLICATE_OUT}"
        )

        sys.exit(2)


    # --------------------------------------------------------
    # 8. Create final immutable sequence-level freeze
    # --------------------------------------------------------

    print(
        "[8/8] Creating final sequence-level freeze..."
    )


    if FINAL_FREEZE.exists():

        raise RuntimeError(
            "Final freeze directory already exists and "
            "will not be overwritten:\n"
            f"{FINAL_FREEZE}"
        )


    FINAL_FREEZE.mkdir(
        parents=True,
        exist_ok=False,
    )


    final_master = (
        FINAL_FREEZE
        / "master_manifest_v3_frozen.tsv"
    )


    shutil.copy2(
        MASTER_AUDITED_OUT,
        final_master,
    )


    shutil.copy2(
        SEQ_AUDIT_OUT,
        FINAL_FREEZE
        / "sequence_integrity_audit.tsv",
    )


    shutil.copy2(
        DUPLICATE_OUT,
        FINAL_FREEZE
        / "sequence_duplicate_groups.tsv",
    )


    shutil.copy2(
        CHECKS_OUT,
        FINAL_FREEZE
        / "sequence_audit_checks.tsv",
    )


    shutil.copy2(
        REPORT_OUT,
        FINAL_FREEZE
        / "sequence_audit_report.txt",
    )


    shutil.copy2(
        REPLACEMENT_PLAN,
        FINAL_FREEZE
        / "replacement_plan_round1.tsv",
    )


    # --------------------------------------------------------
    # Final accession lists
    # --------------------------------------------------------

    final_groups = {
        "development_A_final_accessions.txt":
            [
                r
                for r in rows
                if r["cohort"] == "A"
            ],

        "external_B1_final_accessions.txt":
            [
                r
                for r in rows
                if r["cohort"] == "B1"
            ],

        "external_B2_seen_genus_final_accessions.txt":
            [
                r
                for r in rows
                if (
                    r["cohort"] == "B2"
                    and
                    r["cohort_subtype"]
                    ==
                    "unseen_species_seen_genus"
                )
            ],

        "external_B2_unseen_genus_final_accessions.txt":
            [
                r
                for r in rows
                if (
                    r["cohort"] == "B2"
                    and
                    r["cohort_subtype"]
                    ==
                    "unseen_species_unseen_genus"
                )
            ],

        "external_B2_final_accessions.txt":
            [
                r
                for r in rows
                if r["cohort"] == "B2"
            ],

        "all_final_accessions.txt":
            rows,
    }


    for filename, group_rows in (
        final_groups.items()
    ):

        accessions = sorted(
            {
                r[
                    "assembly_accession"
                ]
                for r in group_rows
            }
        )


        with open(
            FINAL_FREEZE
            / filename,
            "w",
            encoding="utf-8",
        ) as handle:

            for accession in accessions:

                handle.write(
                    accession
                    +
                    "\n"
                )


    # --------------------------------------------------------
    # Freeze info
    # --------------------------------------------------------

    info = f"""HGV-V3 FINAL SEQUENCE-LEVEL FREEZE

Date: 2026-09-16

Final cohort:
A:
  2400 genomes
  200 GTDB species clusters
  100 genera
  88 families
  RefSeq/GCF

B1:
  400 genomes
  40 species
  species seen in A
  GenBank/GCA external cohort

B2:
  400 genomes
  40 species unseen in A

  B2 seen-genus:
    200 genomes
    20 species

  B2 unseen-genus:
    200 genomes
    20 species

Original primary genomes retained:
3191

Round-1 technical replacements:
9

Replacement reason:
NCBI genomic FASTA unavailable for original frozen accession.

Replacement policy:
same cohort
same cohort subtype
same GTDB species cluster
same genus
same family
same source database class
unique accession
unique BioSample
pre-selected frozen reserve only

Final primary total:
3200 genomes

Sequence-level audit:
PASS

Canonical duplicate groups:
0

Cross-cohort canonical duplicate groups:
0

This final freeze must not be changed on the basis of
downstream model performance.

Any future change requires a new documented replacement
round and a new sequence-level freeze.
"""


    with open(
        FINAL_FREEZE
        / "FINAL_FREEZE_INFO.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            info
        )


    # --------------------------------------------------------
    # Final SHA256 manifest
    # --------------------------------------------------------

    freeze_files = sorted(
        [
            path
            for path
            in FINAL_FREEZE.iterdir()
            if (
                path.is_file()
                and
                path.name
                !=
                "FINAL_FREEZE_SHA256.txt"
            )
        ],
        key=lambda p: p.name,
    )


    with open(
        FINAL_FREEZE
        / "FINAL_FREEZE_SHA256.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        for path in freeze_files:

            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )


    # --------------------------------------------------------
    # Make final freeze read-only
    # --------------------------------------------------------

    for path in (
        FINAL_FREEZE.rglob(
            "*"
        )
    ):

        if path.is_file():

            path.chmod(
                path.stat().st_mode
                &
                ~0o222
            )


    FINAL_FREEZE.chmod(
        FINAL_FREEZE.stat().st_mode
        &
        ~0o222
    )


    print()

    print(
        "============================================================"
    )

    print(
        "HGV-V3 FINAL DATA FREEZE: PASS"
    )

    print()

    print(
        "Final master manifest:"
    )

    print(
        final_master
    )

    print()

    print(
        "Final immutable freeze:"
    )

    print(
        FINAL_FREEZE
    )

    print()

    print(
        "The dataset is now approved for feature extraction."
    )

    print(
        "============================================================"
    )


def write_checks_and_report(
    current_checks,
    status,
    reason,
):

    with open(
        CHECKS_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "check",
                "status",
                "expected",
                "observed",
                "detail",
            ],
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            current_checks
        )


    with open(
        REPORT_OUT,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "HGV-V3 FINAL SEQUENCE INTEGRITY AUDIT\n\n"
        )

        handle.write(
            f"FINAL SEQUENCE GATE: {status}\n\n"
        )

        handle.write(
            reason
            +
            "\n"
        )


if __name__ == "__main__":

    main()
