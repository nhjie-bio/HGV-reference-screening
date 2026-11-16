#!/usr/bin/env python3

import csv
import gzip
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

SEL = (
    BASE
    / "01_manifest"
    / "cohort_selection_v2"
)

PRIMARY = (
    SEL
    / "selected_cohorts_primary_v2.tsv"
)

RESERVE = (
    SEL
    / "selected_cohorts_reserve_v2.tsv"
)

SPECIES_DESIGN = (
    SEL
    / "cohort_species_design_v2.tsv"
)

QC_PASS = (
    BASE
    / "01_manifest"
    / "candidate_manifest_qcpass.tsv.gz"
)

AUDIT_DIR = (
    SEL
    / "audit_v2"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_OUT = (
    AUDIT_DIR
    / "selected_cohort_audit_report.txt"
)

CHECKS_OUT = (
    AUDIT_DIR
    / "selected_cohort_audit_checks.tsv"
)

TAXON_OUT = (
    AUDIT_DIR
    / "selected_taxonomic_structure.tsv"
)

FILES_OUT = (
    AUDIT_DIR
    / "selected_file_sha256.tsv"
)


# ============================================================
# Fixed expected design
# ============================================================

EXPECTED_A_GENOMES = 2400
EXPECTED_A_SPECIES = 200
EXPECTED_A_GENERA = 100

EXPECTED_B1_GENOMES = 400
EXPECTED_B1_SPECIES = 40

EXPECTED_B2_GENOMES = 400
EXPECTED_B2_SPECIES = 40

EXPECTED_B2_SEEN_GENOMES = 200
EXPECTED_B2_SEEN_SPECIES = 20

EXPECTED_B2_UNSEEN_GENOMES = 200
EXPECTED_B2_UNSEEN_SPECIES = 20

EXPECTED_TOTAL = 3200

MIN_A_FAMILIES = 20
MIN_ANCHOR_GENERA = 50
MIN_MULTI_GENUS_FAMILIES = 12

A_GENOMES_PER_SPECIES = 12
EXT_GENOMES_PER_SPECIES = 10


# ============================================================
# QC thresholds
# ============================================================

MIN_COMPLETENESS = 95.0
MAX_CONTAMINATION = 5.0

ALLOWED_LEVELS = {
    "complete genome",
    "chromosome",
    "scaffold",
}

MAX_CONTIGS = 200
MIN_N50 = 20_000

MIN_GENOME_SIZE = 1_000_000
MAX_GENOME_SIZE = 10_000_000

MAX_AMBIGUOUS = 0.005


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


def read_accession_list(path):

    values = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            value = line.strip()

            if value:
                values.append(value)

    return values


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


# ============================================================
# Required input files
# ============================================================

required_files = [
    PRIMARY,
    RESERVE,
    SPECIES_DESIGN,
    QC_PASS,

    SEL / "development_A_primary_accessions.txt",
    SEL / "external_B1_primary_accessions.txt",
    SEL / "external_B2_seen_genus_primary_accessions.txt",
    SEL / "external_B2_unseen_genus_primary_accessions.txt",
    SEL / "external_B2_primary_accessions.txt",
    SEL / "all_primary_accessions.txt",
    SEL / "all_reserve_accessions.txt",
]


missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]


if missing_files:

    print(
        "ERROR: missing required files:",
        file=sys.stderr,
    )

    for path in missing_files:
        print(
            path,
            file=sys.stderr,
        )

    sys.exit(1)


# ============================================================
# Read selected manifests
# ============================================================

primary = read_tsv(
    PRIMARY
)

reserve = read_tsv(
    RESERVE
)

species_design = read_tsv(
    SPECIES_DESIGN
)


A = [
    r
    for r in primary
    if r["cohort"] == "A"
]

B1 = [
    r
    for r in primary
    if r["cohort"] == "B1"
]

B2 = [
    r
    for r in primary
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


# ============================================================
# Basic sets
# ============================================================

def aset(rows):

    return {
        clean(
            r["assembly_accession"]
        )
        for r in rows
    }


def bset(rows):

    return {
        clean(
            r["biosample"]
        )
        for r in rows
        if clean(
            r["biosample"]
        )
    }


def sset(rows):

    return {
        clean(
            r["gtdb_species_cluster"]
        )
        for r in rows
    }


def gset(rows):

    return {
        clean(
            r["gtdb_genus"]
        )
        for r in rows
    }


def fset(rows):

    return {
        clean(
            r["gtdb_family"]
        )
        for r in rows
    }


A_ACC = aset(A)
B1_ACC = aset(B1)
B2_ACC = aset(B2)

A_BS = bset(A)
B1_BS = bset(B1)
B2_BS = bset(B2)

A_SP = sset(A)
B1_SP = sset(B1)
B2_SP = sset(B2)

B2_SEEN_SP = sset(B2_SEEN)
B2_UNSEEN_SP = sset(B2_UNSEEN)

A_GEN = gset(A)
B1_GEN = gset(B1)

B2_SEEN_GEN = gset(
    B2_SEEN
)

B2_UNSEEN_GEN = gset(
    B2_UNSEEN
)

A_FAM = fset(A)


primary_acc = aset(
    primary
)

primary_bs = bset(
    primary
)

reserve_acc = aset(
    reserve
)

reserve_bs = bset(
    reserve
)


# ============================================================
# Audit check infrastructure
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
            "check": name,
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
# 1. Exact cohort size checks
# ============================================================

add_check(
    "primary_total_genomes",
    len(primary)
    == EXPECTED_TOTAL,
    EXPECTED_TOTAL,
    len(primary),
)


add_check(
    "A_genomes",
    len(A)
    == EXPECTED_A_GENOMES,
    EXPECTED_A_GENOMES,
    len(A),
)


add_check(
    "B1_genomes",
    len(B1)
    == EXPECTED_B1_GENOMES,
    EXPECTED_B1_GENOMES,
    len(B1),
)


add_check(
    "B2_genomes",
    len(B2)
    == EXPECTED_B2_GENOMES,
    EXPECTED_B2_GENOMES,
    len(B2),
)


add_check(
    "B2_seen_genus_genomes",
    len(B2_SEEN)
    == EXPECTED_B2_SEEN_GENOMES,
    EXPECTED_B2_SEEN_GENOMES,
    len(B2_SEEN),
)


add_check(
    "B2_unseen_genus_genomes",
    len(B2_UNSEEN)
    == EXPECTED_B2_UNSEEN_GENOMES,
    EXPECTED_B2_UNSEEN_GENOMES,
    len(B2_UNSEEN),
)


# ============================================================
# 2. Unique accession / BioSample
# ============================================================

add_check(
    "primary_accessions_unique",
    len(primary_acc)
    == len(primary),
    len(primary),
    len(primary_acc),
)


missing_primary_bs = [
    r["assembly_accession"]
    for r in primary
    if not clean(
        r["biosample"]
    )
]


add_check(
    "primary_biosample_nonmissing",
    len(missing_primary_bs)
    == 0,
    0,
    len(missing_primary_bs),
)


add_check(
    "primary_biosamples_unique",
    len(primary_bs)
    == len(primary),
    len(primary),
    len(primary_bs),
)


# ============================================================
# 3. Source-database rules
# ============================================================

bad_A_source = [
    r["assembly_accession"]
    for r in A
    if (
        r["source_database"]
        != "REFSEQ_GCF"
    )
]


bad_B1_source = [
    r["assembly_accession"]
    for r in B1
    if (
        r["source_database"]
        != "GENBANK_GCA"
    )
]


bad_B2_source = [
    r["assembly_accession"]
    for r in B2
    if (
        r["source_database"]
        != "GENBANK_GCA"
    )
]


add_check(
    "A_all_RefSeq_GCF",
    len(bad_A_source)
    == 0,
    0,
    len(bad_A_source),
)


add_check(
    "B1_all_GenBank_GCA",
    len(bad_B1_source)
    == 0,
    0,
    len(bad_B1_source),
)


add_check(
    "B2_all_GenBank_GCA",
    len(bad_B2_source)
    == 0,
    0,
    len(bad_B2_source),
)


# ============================================================
# 4. Cross-cohort accession overlap
# ============================================================

add_check(
    "A_B1_accession_overlap",
    len(
        A_ACC & B1_ACC
    ) == 0,
    0,
    len(
        A_ACC & B1_ACC
    ),
)


add_check(
    "A_B2_accession_overlap",
    len(
        A_ACC & B2_ACC
    ) == 0,
    0,
    len(
        A_ACC & B2_ACC
    ),
)


add_check(
    "B1_B2_accession_overlap",
    len(
        B1_ACC & B2_ACC
    ) == 0,
    0,
    len(
        B1_ACC & B2_ACC
    ),
)


# ============================================================
# 5. Cross-cohort BioSample overlap
# ============================================================

add_check(
    "A_B1_biosample_overlap",
    len(
        A_BS & B1_BS
    ) == 0,
    0,
    len(
        A_BS & B1_BS
    ),
)


add_check(
    "A_B2_biosample_overlap",
    len(
        A_BS & B2_BS
    ) == 0,
    0,
    len(
        A_BS & B2_BS
    ),
)


add_check(
    "B1_B2_biosample_overlap",
    len(
        B1_BS & B2_BS
    ) == 0,
    0,
    len(
        B1_BS & B2_BS
    ),
)


# ============================================================
# 6. Taxonomic counts
# ============================================================

add_check(
    "A_species_count",
    len(A_SP)
    == EXPECTED_A_SPECIES,
    EXPECTED_A_SPECIES,
    len(A_SP),
)


add_check(
    "A_genus_count",
    len(A_GEN)
    == EXPECTED_A_GENERA,
    EXPECTED_A_GENERA,
    len(A_GEN),
)


add_check(
    "A_family_count",
    len(A_FAM)
    >= MIN_A_FAMILIES,
    f">={MIN_A_FAMILIES}",
    len(A_FAM),
)


add_check(
    "B1_species_count",
    len(B1_SP)
    == EXPECTED_B1_SPECIES,
    EXPECTED_B1_SPECIES,
    len(B1_SP),
)


add_check(
    "B2_species_count",
    len(B2_SP)
    == EXPECTED_B2_SPECIES,
    EXPECTED_B2_SPECIES,
    len(B2_SP),
)


add_check(
    "B2_seen_species_count",
    len(B2_SEEN_SP)
    == EXPECTED_B2_SEEN_SPECIES,
    EXPECTED_B2_SEEN_SPECIES,
    len(B2_SEEN_SP),
)


add_check(
    "B2_unseen_species_count",
    len(B2_UNSEEN_SP)
    == EXPECTED_B2_UNSEEN_SPECIES,
    EXPECTED_B2_UNSEEN_SPECIES,
    len(B2_UNSEEN_SP),
)


# ============================================================
# 7. Species/genus leakage logic
# ============================================================

add_check(
    "B1_species_subset_of_A",
    B1_SP.issubset(
        A_SP
    ),
    "True",
    B1_SP.issubset(
        A_SP
    ),
    f"B1-only species={len(B1_SP - A_SP)}",
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
    "B2_seen_species_unseen_in_A",
    len(
        B2_SEEN_SP & A_SP
    ) == 0,
    0,
    len(
        B2_SEEN_SP & A_SP
    ),
)


add_check(
    "B2_seen_genera_present_in_A",
    B2_SEEN_GEN.issubset(
        A_GEN
    ),
    "True",
    B2_SEEN_GEN.issubset(
        A_GEN
    ),
    (
        "missing genera="
        +
        str(
            sorted(
                B2_SEEN_GEN
                -
                A_GEN
            )
        )
    ),
)


add_check(
    "B2_unseen_species_unseen_in_A",
    len(
        B2_UNSEEN_SP & A_SP
    ) == 0,
    0,
    len(
        B2_UNSEEN_SP & A_SP
    ),
)


add_check(
    "B2_unseen_genera_absent_from_A",
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


add_check(
    "B2_seen_unseen_genus_overlap",
    len(
        B2_SEEN_GEN
        &
        B2_UNSEEN_GEN
    ) == 0,
    0,
    len(
        B2_SEEN_GEN
        &
        B2_UNSEEN_GEN
    ),
)


# ============================================================
# 8. Exact genomes per species
# ============================================================

def species_counts(rows):

    return Counter(
        r["gtdb_species_cluster"]
        for r in rows
    )


A_PER_SPECIES = species_counts(
    A
)

B1_PER_SPECIES = species_counts(
    B1
)

B2_PER_SPECIES = species_counts(
    B2
)


bad_A_species_n = {
    sp: n
    for sp, n
    in A_PER_SPECIES.items()
    if n != A_GENOMES_PER_SPECIES
}


bad_B1_species_n = {
    sp: n
    for sp, n
    in B1_PER_SPECIES.items()
    if n != EXT_GENOMES_PER_SPECIES
}


bad_B2_species_n = {
    sp: n
    for sp, n
    in B2_PER_SPECIES.items()
    if n != EXT_GENOMES_PER_SPECIES
}


add_check(
    "A_exact_12_genomes_per_species",
    len(
        bad_A_species_n
    ) == 0,
    0,
    len(
        bad_A_species_n
    ),
    bad_A_species_n,
)


add_check(
    "B1_exact_10_genomes_per_species",
    len(
        bad_B1_species_n
    ) == 0,
    0,
    len(
        bad_B1_species_n
    ),
    bad_B1_species_n,
)


add_check(
    "B2_exact_10_genomes_per_species",
    len(
        bad_B2_species_n
    ) == 0,
    0,
    len(
        bad_B2_species_n
    ),
    bad_B2_species_n,
)


# ============================================================
# 9. Hierarchical structure of A
# ============================================================

A_GENUS_SPECIES = defaultdict(set)
A_FAMILY_GENERA = defaultdict(set)


for r in A:

    A_GENUS_SPECIES[
        r["gtdb_genus"]
    ].add(
        r["gtdb_species_cluster"]
    )

    A_FAMILY_GENERA[
        r["gtdb_family"]
    ].add(
        r["gtdb_genus"]
    )


anchor_genera = {
    genus
    for genus, species_set
    in A_GENUS_SPECIES.items()
    if len(species_set) >= 3
}


multi_genus_families = {
    family
    for family, genera
    in A_FAMILY_GENERA.items()
    if len(genera) >= 2
}


add_check(
    "A_anchor_genera",
    len(anchor_genera)
    >= MIN_ANCHOR_GENERA,
    f">={MIN_ANCHOR_GENERA}",
    len(anchor_genera),
)


add_check(
    "A_multi_genus_families",
    len(
        multi_genus_families
    )
    >=
    MIN_MULTI_GENUS_FAMILIES,
    f">={MIN_MULTI_GENUS_FAMILIES}",
    len(
        multi_genus_families
    ),
)


# ============================================================
# 10. QC validity of selected genomes
# ============================================================

quality_failures = []


for r in primary:

    comp = as_float(
        r["completeness"]
    )

    cont = as_float(
        r["contamination"]
    )

    contigs = as_int(
        r["contig_count"]
    )

    n50 = as_int(
        r["n50_contigs"]
    )

    genome_size = as_int(
        r["genome_size"]
    )

    ambiguous = as_float(
        r["ambiguous_fraction"]
    )

    level = clean(
        r["assembly_level"]
    ).lower()

    reasons = []

    if (
        comp is None
        or
        comp < MIN_COMPLETENESS
    ):
        reasons.append(
            "completeness"
        )

    if (
        cont is None
        or
        cont > MAX_CONTAMINATION
    ):
        reasons.append(
            "contamination"
        )

    if level not in ALLOWED_LEVELS:
        reasons.append(
            "assembly_level"
        )

    if (
        contigs is None
        or
        contigs > MAX_CONTIGS
    ):
        reasons.append(
            "contig_count"
        )

    if (
        n50 is None
        or
        n50 < MIN_N50
    ):
        reasons.append(
            "n50"
        )

    if (
        genome_size is None
        or
        genome_size
        < MIN_GENOME_SIZE
        or
        genome_size
        > MAX_GENOME_SIZE
    ):
        reasons.append(
            "genome_size"
        )

    if (
        ambiguous is None
        or
        ambiguous
        > MAX_AMBIGUOUS
    ):
        reasons.append(
            "ambiguous"
        )

    if reasons:

        quality_failures.append(
            (
                r[
                    "assembly_accession"
                ],
                ";".join(reasons),
            )
        )


add_check(
    "selected_primary_passes_QC_thresholds",
    len(
        quality_failures
    ) == 0,
    0,
    len(
        quality_failures
    ),
    quality_failures[:10],
)


# ============================================================
# 11. Primary/reserve independence
# ============================================================

add_check(
    "primary_reserve_accession_overlap",
    len(
        primary_acc
        &
        reserve_acc
    ) == 0,
    0,
    len(
        primary_acc
        &
        reserve_acc
    ),
)


add_check(
    "primary_reserve_biosample_overlap",
    len(
        primary_bs
        &
        reserve_bs
    ) == 0,
    0,
    len(
        primary_bs
        &
        reserve_bs
    ),
)


add_check(
    "reserve_accessions_unique",
    len(
        reserve_acc
    ) == len(
        reserve
    ),
    len(
        reserve
    ),
    len(
        reserve_acc
    ),
)


add_check(
    "reserve_biosamples_unique",
    len(
        reserve_bs
    ) == len(
        reserve
    ),
    len(
        reserve
    ),
    len(
        reserve_bs
    ),
)


# ============================================================
# 12. Species design file consistency
# ============================================================

design_A = {
    r["gtdb_species_cluster"]
    for r in species_design
    if r["cohort"] == "A"
}


design_B2_seen = {
    r["gtdb_species_cluster"]
    for r in species_design
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        ==
        "unseen_species_seen_genus"
    )
}


design_B2_unseen = {
    r["gtdb_species_cluster"]
    for r in species_design
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        ==
        "unseen_species_unseen_genus"
    )
}


design_B1 = {
    r["gtdb_species_cluster"]
    for r in species_design
    if (
        r["cohort"] == "A"
        and
        r["is_B1_species"] == "1"
    )
}


add_check(
    "species_design_A_matches_primary",
    design_A == A_SP,
    "exact set match",
    (
        f"missing={len(A_SP - design_A)}, "
        f"extra={len(design_A - A_SP)}"
    ),
)


add_check(
    "species_design_B1_matches_primary",
    design_B1 == B1_SP,
    "exact set match",
    (
        f"missing={len(B1_SP - design_B1)}, "
        f"extra={len(design_B1 - B1_SP)}"
    ),
)


add_check(
    "species_design_B2_seen_matches_primary",
    design_B2_seen
    ==
    B2_SEEN_SP,
    "exact set match",
    (
        f"missing={len(B2_SEEN_SP - design_B2_seen)}, "
        f"extra={len(design_B2_seen - B2_SEEN_SP)}"
    ),
)


add_check(
    "species_design_B2_unseen_matches_primary",
    design_B2_unseen
    ==
    B2_UNSEEN_SP,
    "exact set match",
    (
        f"missing={len(B2_UNSEEN_SP - design_B2_unseen)}, "
        f"extra={len(design_B2_unseen - B2_UNSEEN_SP)}"
    ),
)


# ============================================================
# 13. Accession-list consistency
# ============================================================

ACCESSION_LISTS = {
    "development_A_primary_accessions.txt":
        A_ACC,

    "external_B1_primary_accessions.txt":
        B1_ACC,

    "external_B2_seen_genus_primary_accessions.txt":
        aset(
            B2_SEEN
        ),

    "external_B2_unseen_genus_primary_accessions.txt":
        aset(
            B2_UNSEEN
        ),

    "external_B2_primary_accessions.txt":
        B2_ACC,

    "all_primary_accessions.txt":
        primary_acc,

    "all_reserve_accessions.txt":
        reserve_acc,
}


for filename, expected_set in (
    ACCESSION_LISTS.items()
):

    path = (
        SEL / filename
    )

    values = (
        read_accession_list(
            path
        )
    )

    observed_set = set(
        values
    )

    duplicate_lines = (
        len(values)
        -
        len(observed_set)
    )

    add_check(
        f"{filename}_duplicate_lines",
        duplicate_lines == 0,
        0,
        duplicate_lines,
    )

    add_check(
        f"{filename}_matches_manifest",
        observed_set
        ==
        expected_set,
        "exact set match",
        (
            f"missing={len(expected_set - observed_set)}, "
            f"extra={len(observed_set - expected_set)}"
        ),
    )


# ============================================================
# 14. Independent re-read of QC-pass master manifest
#
# This independently validates:
# - all selected accessions are genuine QC-pass records
# - taxonomy/source/BioSample metadata are consistent
# - external GCA are not paired twins of ANY QC-pass GCF
# - external BioSamples do not occur in ANY QC-pass GCF
# ============================================================

needed_accessions = (
    primary_acc
    |
    reserve_acc
)


found_rows = {}

all_qcpass_gcf_biosamples = set()
all_qcpass_paired_gca = set()


with gzip.open(
    QC_PASS,
    "rt",
    encoding="utf-8",
    errors="replace",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

    for row in reader:

        accession = clean(
            row[
                "assembly_accession"
            ]
        )

        if (
            row[
                "source_database"
            ]
            ==
            "REFSEQ_GCF"
        ):

            bs = clean(
                row["biosample"]
            )

            if bs:
                all_qcpass_gcf_biosamples.add(
                    bs
                )

            paired = clean(
                row[
                    "paired_genbank_assembly"
                ]
            )

            if paired:
                all_qcpass_paired_gca.add(
                    paired
                )

        if accession in needed_accessions:

            if accession in found_rows:

                raise RuntimeError(
                    "Duplicate accession in QC-pass manifest: "
                    + accession
                )

            found_rows[
                accession
            ] = row


missing_from_qcpass = (
    needed_accessions
    -
    set(
        found_rows
    )
)


add_check(
    "selected_primary_and_reserve_exist_in_QCpass_manifest",
    len(
        missing_from_qcpass
    ) == 0,
    0,
    len(
        missing_from_qcpass
    ),
    sorted(
        missing_from_qcpass
    )[:10],
)


# Metadata consistency for primary
metadata_mismatch = []


for r in primary:

    accession = r[
        "assembly_accession"
    ]

    master = found_rows.get(
        accession
    )

    if master is None:
        continue

    for field in [
        "source_database",
        "biosample",
        "gtdb_species_cluster",
        "gtdb_species",
        "gtdb_genus",
        "gtdb_family",
    ]:

        if (
            clean(r[field])
            !=
            clean(master[field])
        ):

            metadata_mismatch.append(
                (
                    accession,
                    field,
                    r[field],
                    master[field],
                )
            )


add_check(
    "selected_primary_metadata_matches_QCpass_master",
    len(
        metadata_mismatch
    ) == 0,
    0,
    len(
        metadata_mismatch
    ),
    metadata_mismatch[:10],
)


# External independence against all QC-pass GCFs
external = (
    B1
    +
    B2
)


external_paired_collisions = [
    r["assembly_accession"]
    for r in external
    if (
        r["assembly_accession"]
        in
        all_qcpass_paired_gca
    )
]


external_biosample_collisions = [
    (
        r["assembly_accession"],
        r["biosample"],
    )
    for r in external
    if (
        clean(
            r["biosample"]
        )
        in
        all_qcpass_gcf_biosamples
    )
]


add_check(
    "external_GCA_not_paired_to_any_QCpass_GCF",
    len(
        external_paired_collisions
    ) == 0,
    0,
    len(
        external_paired_collisions
    ),
    external_paired_collisions[:10],
)


add_check(
    "external_BioSamples_absent_from_all_QCpass_GCF",
    len(
        external_biosample_collisions
    ) == 0,
    0,
    len(
        external_biosample_collisions
    ),
    external_biosample_collisions[:10],
)


# ============================================================
# 15. Taxonomy source-data table
# ============================================================

with open(
    TAXON_OUT,
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
            "cohort",
            "species",
            "genera",
            "families",
            "genomes",
        ]
    )

    for name, rows in [
        ("A", A),
        ("B1", B1),
        ("B2", B2),
        (
            "B2_seen_genus",
            B2_SEEN,
        ),
        (
            "B2_unseen_genus",
            B2_UNSEEN,
        ),
    ]:

        writer.writerow(
            [
                name,
                len(
                    sset(rows)
                ),
                len(
                    gset(rows)
                ),
                len(
                    fset(rows)
                ),
                len(rows),
            ]
        )


# ============================================================
# 16. SHA256 of current pre-freeze files
#
# This does NOT yet mean final sequence freeze.
# It records the exact metadata selection that was audited.
# ============================================================

files_to_hash = [
    PRIMARY,
    RESERVE,
    SPECIES_DESIGN,

    SEL
    / "development_A_primary_accessions.txt",

    SEL
    / "external_B1_primary_accessions.txt",

    SEL
    / "external_B2_seen_genus_primary_accessions.txt",

    SEL
    / "external_B2_unseen_genus_primary_accessions.txt",

    SEL
    / "external_B2_primary_accessions.txt",

    SEL
    / "all_primary_accessions.txt",

    SEL
    / "all_reserve_accessions.txt",
]


with open(
    FILES_OUT,
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
            "sha256",
            "file",
        ]
    )

    for path in files_to_hash:

        writer.writerow(
            [
                sha256_file(
                    path
                ),
                str(path),
            ]
        )


# ============================================================
# Write audit checks TSV
# ============================================================

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


# ============================================================
# Final result
# ============================================================

failed = [
    x
    for x in checks
    if x["status"] == "FAIL"
]


passed_n = (
    len(checks)
    -
    len(failed)
)


report = []

report.append(
    "============================================================"
)

report.append(
    "HGV-V3 SELECTED COHORT FINAL METADATA AUDIT"
)

report.append(
    "============================================================"
)

report.append("")

report.append(
    f"Total audit checks: {len(checks)}"
)

report.append(
    f"PASS: {passed_n}"
)

report.append(
    f"FAIL: {len(failed)}"
)

report.append("")

report.append(
    "PRIMARY COHORT"
)

report.append(
    "--------------"
)

report.append(
    f"A:  {len(A):,} genomes; "
    f"{len(A_SP)} species; "
    f"{len(A_GEN)} genera; "
    f"{len(A_FAM)} families"
)

report.append(
    f"B1: {len(B1):,} genomes; "
    f"{len(B1_SP)} species; "
    f"{len(B1_GEN)} genera; "
    f"{len(fset(B1))} families"
)

report.append(
    f"B2: {len(B2):,} genomes; "
    f"{len(B2_SP)} species; "
    f"{len(gset(B2))} genera; "
    f"{len(fset(B2))} families"
)

report.append("")

report.append(
    f"A anchor genera >=3 species: "
    f"{len(anchor_genera)}"
)

report.append(
    f"A families >=2 genera: "
    f"{len(multi_genus_families)}"
)

report.append(
    f"Reserve genomes: {len(reserve)}"
)

report.append("")

report.append(
    "METADATA LEAKAGE STATUS"
)

report.append(
    "-----------------------"
)

report.append(
    f"A/B1 accession overlap: "
    f"{len(A_ACC & B1_ACC)}"
)

report.append(
    f"A/B2 accession overlap: "
    f"{len(A_ACC & B2_ACC)}"
)

report.append(
    f"B1/B2 accession overlap: "
    f"{len(B1_ACC & B2_ACC)}"
)

report.append(
    f"A/B1 BioSample overlap: "
    f"{len(A_BS & B1_BS)}"
)

report.append(
    f"A/B2 BioSample overlap: "
    f"{len(A_BS & B2_BS)}"
)

report.append(
    f"B1/B2 BioSample overlap: "
    f"{len(B1_BS & B2_BS)}"
)

report.append(
    f"External paired-GCF collisions: "
    f"{len(external_paired_collisions)}"
)

report.append(
    f"External BioSample-vs-all-GCF collisions: "
    f"{len(external_biosample_collisions)}"
)

report.append("")

report.append(
    "SEQUENCE-LEVEL STATUS"
)

report.append(
    "---------------------"
)

report.append(
    "Sequence MD5/SHA256 duplication audit: PENDING"
)

report.append(
    "Reason: FASTA genomes have not yet been downloaded."
)

report.append(
    "Any sequence-level collision found after download "
    "must be replaced using the reserve cohort and "
    "documented before the final frozen master manifest."
)

report.append("")

if failed:

    report.append(
        "FINAL PRE-DOWNLOAD GATE: FAIL"
    )

    report.append("")

    report.append(
        "FAILED CHECKS"
    )

    report.append(
        "-------------"
    )

    for item in failed:

        report.append(
            f"{item['check']}: "
            f"expected={item['expected']}; "
            f"observed={item['observed']}; "
            f"{item['detail']}"
        )

else:

    report.append(
        "FINAL PRE-DOWNLOAD METADATA GATE: PASS"
    )

    report.append("")

    report.append(
        "The 3,200-genome primary cohort is approved "
        "for genome download."
    )

    report.append(
        "The current metadata selection has been "
        "SHA256-recorded but is not yet the final "
        "sequence-level freeze."
    )


report.append("")

report.append(
    "============================================================"
)


report_text = (
    "\n".join(report)
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

print(
    f"Checks TSV: {CHECKS_OUT}"
)

print(
    f"Taxonomy table: {TAXON_OUT}"
)

print(
    f"SHA256 table: {FILES_OUT}"
)

print(
    f"Audit report: {REPORT_OUT}"
)


if failed:
    sys.exit(2)

sys.exit(0)
