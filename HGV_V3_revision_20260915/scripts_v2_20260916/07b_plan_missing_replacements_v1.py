#!/usr/bin/env python3

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

FREEZE = (
    BASE
    / "01_manifest"
    / "frozen_pre_download_20260916"
)

PRIMARY_FILE = (
    FREEZE
    / "selected_cohorts_primary_v2.tsv"
)

RESERVE_FILE = (
    FREEZE
    / "selected_cohorts_reserve_v2.tsv"
)

DOWNLOAD_ROOT = (
    BASE
    / "02_genomes"
)

OUTDIR = (
    BASE
    / "01_manifest"
    / "replacement_round1_20260916"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLAN_OUT = (
    OUTDIR
    / "replacement_plan_round1.tsv"
)

UPDATED_MANIFEST_OUT = (
    OUTDIR
    / "updated_primary_manifest_round1.tsv"
)

REPLACEMENT_ACCESSIONS_OUT = (
    OUTDIR
    / "replacement_accessions_round1.txt"
)

SUMMARY_OUT = (
    OUTDIR
    / "replacement_round1_summary.txt"
)

SHA_OUT = (
    OUTDIR
    / "replacement_round1_sha256.tsv"
)


# ============================================================
# Missing-accession files produced by 07 download audit
# ============================================================

MISSING_FILES = [
    (
        "A",
        "development",
        DOWNLOAD_ROOT
        / "development"
        / "ncbi_download"
        / "audit"
        / "missing_accessions.txt",
    ),
    (
        "B1",
        "matched_species_external",
        DOWNLOAD_ROOT
        / "external_B1"
        / "ncbi_download"
        / "audit"
        / "missing_accessions.txt",
    ),
    (
        "B2",
        "unseen_species_seen_genus",
        DOWNLOAD_ROOT
        / "external_B2"
        / "seen_genus"
        / "ncbi_download"
        / "audit"
        / "missing_accessions.txt",
    ),
    (
        "B2",
        "unseen_species_unseen_genus",
        DOWNLOAD_ROOT
        / "external_B2"
        / "unseen_genus"
        / "ncbi_download"
        / "audit"
        / "missing_accessions.txt",
    ),
]


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


def as_int(x, default=0):

    try:
        return int(float(x))
    except Exception:
        return default


def as_float(x, default=999.0):

    try:
        return float(x)
    except Exception:
        return default


def assembly_rank(level):

    level = clean(level).lower()

    if level == "complete genome":
        return 3

    if level == "chromosome":
        return 2

    if level == "scaffold":
        return 1

    return 0


def quality_key(row):

    return (
        -assembly_rank(
            row["assembly_level"]
        ),
        -as_float(
            row["completeness"],
            0.0,
        ),
        as_float(
            row["contamination"],
            999.0,
        ),
        as_int(
            row["contig_count"],
            999999,
        ),
        -as_int(
            row["n50_contigs"],
            0,
        ),
        as_float(
            row["ambiguous_fraction"],
            999.0,
        ),
        row["assembly_accession"],
    )


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
# Validate required input
# ============================================================

for path in [
    PRIMARY_FILE,
    RESERVE_FILE,
]:

    if not path.exists():

        raise SystemExit(
            f"Missing required frozen file: {path}"
        )


for cohort, subtype, path in MISSING_FILES:

    if not path.exists():

        raise SystemExit(
            f"Missing download audit file: {path}"
        )


# ============================================================
# Read frozen manifests
# ============================================================

primary = read_tsv(
    PRIMARY_FILE
)

reserve = read_tsv(
    RESERVE_FILE
)


primary_by_accession = {
    r["assembly_accession"]: r
    for r in primary
}


# ============================================================
# Parse missing genomes
# ============================================================

missing_records = []


for cohort, subtype, path in MISSING_FILES:

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            accession = line.strip()

            if not accession:
                continue

            if accession not in primary_by_accession:

                raise RuntimeError(
                    "Missing accession is not present "
                    "in frozen primary manifest: "
                    + accession
                )

            row = primary_by_accession[
                accession
            ]

            if row["cohort"] != cohort:

                raise RuntimeError(
                    f"Cohort mismatch for {accession}: "
                    f"expected {cohort}; "
                    f"manifest says {row['cohort']}"
                )

            if row["cohort_subtype"] != subtype:

                raise RuntimeError(
                    f"Subtype mismatch for {accession}: "
                    f"expected {subtype}; "
                    f"manifest says "
                    f"{row['cohort_subtype']}"
                )

            missing_records.append(
                row
            )


missing_accessions = {
    r["assembly_accession"]
    for r in missing_records
}


if len(missing_records) != len(missing_accessions):

    raise RuntimeError(
        "Duplicate missing accession detected."
    )


# ============================================================
# Sanity check expected missing count
# ============================================================

EXPECTED_MISSING = 9


if len(missing_records) != EXPECTED_MISSING:

    raise RuntimeError(
        f"Expected {EXPECTED_MISSING} missing genomes "
        f"from Phase 2B, observed {len(missing_records)}."
    )


# ============================================================
# Build reserve lookup:
# exact same cohort + subtype + species
# ============================================================

reserve_lookup = defaultdict(list)


for row in reserve:

    key = (
        row["cohort"],
        row["cohort_subtype"],
        row["gtdb_species_cluster"],
    )

    reserve_lookup[
        key
    ].append(
        row
    )


for key in reserve_lookup:

    reserve_lookup[
        key
    ].sort(
        key=quality_key
    )


# ============================================================
# Existing primary identity sets
#
# Failed accessions are removed before replacement check.
# ============================================================

retained_primary = [
    r
    for r in primary
    if (
        r["assembly_accession"]
        not in missing_accessions
    )
]


used_accessions = {
    r["assembly_accession"]
    for r in retained_primary
}


used_biosamples = {
    clean(
        r["biosample"]
    )
    for r in retained_primary
    if clean(
        r["biosample"]
    )
}


# ============================================================
# Select replacements
# ============================================================

replacement_plan = []

chosen_replacements = []


for failed in sorted(
    missing_records,
    key=lambda r: (
        r["cohort"],
        r["cohort_subtype"],
        r["gtdb_species_cluster"],
        r["assembly_accession"],
    ),
):

    key = (
        failed["cohort"],
        failed["cohort_subtype"],
        failed["gtdb_species_cluster"],
    )

    candidates = reserve_lookup.get(
        key,
        [],
    )

    valid = []

    for candidate in candidates:

        acc = candidate[
            "assembly_accession"
        ]

        bs = clean(
            candidate["biosample"]
        )

        if not bs:
            continue

        if acc in used_accessions:
            continue

        if bs in used_biosamples:
            continue

        if (
            candidate["gtdb_species_cluster"]
            !=
            failed["gtdb_species_cluster"]
        ):
            continue

        if (
            candidate["gtdb_genus"]
            !=
            failed["gtdb_genus"]
        ):
            continue

        if (
            candidate["gtdb_family"]
            !=
            failed["gtdb_family"]
        ):
            continue

        if (
            candidate["cohort"]
            !=
            failed["cohort"]
        ):
            continue

        if (
            candidate["cohort_subtype"]
            !=
            failed["cohort_subtype"]
        ):
            continue

        if (
            candidate["source_database"]
            !=
            failed["source_database"]
        ):
            continue

        valid.append(
            candidate
        )


    if not valid:

        raise RuntimeError(
            "\nNo valid frozen reserve replacement for:\n"
            f"  failed accession: "
            f"{failed['assembly_accession']}\n"
            f"  cohort: "
            f"{failed['cohort']}\n"
            f"  subtype: "
            f"{failed['cohort_subtype']}\n"
            f"  species: "
            f"{failed['gtdb_species_cluster']}\n"
            "\nDo NOT select a replacement manually.\n"
            "We need to construct a controlled extension "
            "from the QC-pass master pool for this species."
        )


    valid.sort(
        key=quality_key
    )

    replacement = valid[0]

    rep_acc = replacement[
        "assembly_accession"
    ]

    rep_bs = clean(
        replacement["biosample"]
    )


    used_accessions.add(
        rep_acc
    )

    used_biosamples.add(
        rep_bs
    )

    chosen_replacements.append(
        replacement
    )


    replacement_plan.append(
        {
            "replacement_round": "1",

            "reason":
                "NCBI_genomic_FASTA_unavailable",

            "cohort":
                failed["cohort"],

            "cohort_subtype":
                failed["cohort_subtype"],

            "gtdb_species_cluster":
                failed["gtdb_species_cluster"],

            "gtdb_species":
                failed["gtdb_species"],

            "gtdb_genus":
                failed["gtdb_genus"],

            "gtdb_family":
                failed["gtdb_family"],

            "failed_accession":
                failed["assembly_accession"],

            "failed_biosample":
                failed["biosample"],

            "replacement_accession":
                replacement[
                    "assembly_accession"
                ],

            "replacement_biosample":
                replacement[
                    "biosample"
                ],

            "source_database":
                replacement[
                    "source_database"
                ],

            "replacement_assembly_level":
                replacement[
                    "assembly_level"
                ],

            "replacement_completeness":
                replacement[
                    "completeness"
                ],

            "replacement_contamination":
                replacement[
                    "contamination"
                ],

            "replacement_contig_count":
                replacement[
                    "contig_count"
                ],

            "replacement_n50_contigs":
                replacement[
                    "n50_contigs"
                ],
        }
    )


# ============================================================
# Write replacement plan
# ============================================================

PLAN_FIELDS = [
    "replacement_round",
    "reason",

    "cohort",
    "cohort_subtype",

    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",

    "failed_accession",
    "failed_biosample",

    "replacement_accession",
    "replacement_biosample",

    "source_database",

    "replacement_assembly_level",
    "replacement_completeness",
    "replacement_contamination",
    "replacement_contig_count",
    "replacement_n50_contigs",
]


with open(
    PLAN_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=PLAN_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        replacement_plan
    )


# ============================================================
# Replacement accession list
# ============================================================

replacement_accessions = sorted(
    r["assembly_accession"]
    for r in chosen_replacements
)


with open(
    REPLACEMENT_ACCESSIONS_OUT,
    "w",
    encoding="utf-8",
) as handle:

    for accession in replacement_accessions:

        handle.write(
            accession
            + "\n"
        )


# ============================================================
# Build updated provisional primary manifest
#
# Frozen primary is NOT modified.
# ============================================================

replacement_by_failed = {
    row["failed_accession"]:
        row["replacement_accession"]
    for row in replacement_plan
}


replacement_rows_by_accession = {
    r["assembly_accession"]: r
    for r in chosen_replacements
}


updated_rows = []


for original in primary:

    original_acc = original[
        "assembly_accession"
    ]

    if (
        original_acc
        not in replacement_by_failed
    ):

        x = dict(
            original
        )

        x[
            "original_primary_accession"
        ] = original_acc

        x[
            "replacement_status"
        ] = "original_primary_retained"

        x[
            "replacement_reason"
        ] = ""

        updated_rows.append(
            x
        )

    else:

        replacement_acc = (
            replacement_by_failed[
                original_acc
            ]
        )

        replacement = dict(
            replacement_rows_by_accession[
                replacement_acc
            ]
        )

        replacement[
            "selection_role"
        ] = "primary_replacement"

        replacement[
            "original_primary_accession"
        ] = original_acc

        replacement[
            "replacement_status"
        ] = "replacement_round1"

        replacement[
            "replacement_reason"
        ] = "NCBI_genomic_FASTA_unavailable"

        updated_rows.append(
            replacement
        )


# ============================================================
# Validate provisional updated manifest
# ============================================================

if len(updated_rows) != 3200:

    raise RuntimeError(
        f"Updated manifest has {len(updated_rows)} rows, "
        "expected 3200."
    )


updated_accessions = [
    r["assembly_accession"]
    for r in updated_rows
]


updated_biosamples = [
    clean(
        r["biosample"]
    )
    for r in updated_rows
]


if len(
    set(updated_accessions)
) != 3200:

    raise RuntimeError(
        "Updated manifest has duplicate accessions."
    )


if len(
    set(updated_biosamples)
) != 3200:

    raise RuntimeError(
        "Updated manifest has duplicate BioSamples."
    )


# Cohort sizes must remain unchanged
cohort_counts = defaultdict(int)

species_counts = defaultdict(set)


for row in updated_rows:

    cohort_counts[
        (
            row["cohort"],
            row["cohort_subtype"],
        )
    ] += 1

    species_counts[
        (
            row["cohort"],
            row["cohort_subtype"],
        )
    ].add(
        row["gtdb_species_cluster"]
    )


expected_counts = {
    (
        "A",
        "development",
    ): 2400,

    (
        "B1",
        "matched_species_external",
    ): 400,

    (
        "B2",
        "unseen_species_seen_genus",
    ): 200,

    (
        "B2",
        "unseen_species_unseen_genus",
    ): 200,
}


for key, expected in (
    expected_counts.items()
):

    observed = cohort_counts[
        key
    ]

    if observed != expected:

        raise RuntimeError(
            f"Cohort count changed for {key}: "
            f"{observed} != {expected}"
        )


# ============================================================
# Write updated provisional manifest
# ============================================================

if updated_rows:

    fields = list(
        primary[0].keys()
    ) + [
        "original_primary_accession",
        "replacement_status",
        "replacement_reason",
    ]


    with open(
        UPDATED_MANIFEST_OUT,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerows(
            updated_rows
        )


# ============================================================
# SHA256 provenance
# ============================================================

hashed_files = [
    PLAN_OUT,
    REPLACEMENT_ACCESSIONS_OUT,
    UPDATED_MANIFEST_OUT,
]


with open(
    SHA_OUT,
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

    for path in hashed_files:

        writer.writerow(
            [
                sha256_file(
                    path
                ),
                str(path),
            ]
        )


# ============================================================
# Summary
# ============================================================

cohort_missing = defaultdict(int)


for row in missing_records:

    key = (
        row["cohort"],
        row["cohort_subtype"],
    )

    cohort_missing[
        key
    ] += 1


lines = []

lines.append(
    "============================================================"
)

lines.append(
    "HGV-V3 ROUND-1 REPLACEMENT PLAN"
)

lines.append(
    "============================================================"
)

lines.append("")

lines.append(
    f"Unavailable original primary FASTA: "
    f"{len(missing_records)}"
)

lines.append("")

lines.append(
    "Missing by cohort:"
)

lines.append(
    f"A development: "
    f"{cohort_missing[('A', 'development')]}"
)

lines.append(
    f"B1 external: "
    f"{cohort_missing[('B1', 'matched_species_external')]}"
)

lines.append(
    f"B2 seen-genus: "
    f"{cohort_missing[('B2', 'unseen_species_seen_genus')]}"
)

lines.append(
    f"B2 unseen-genus: "
    f"{cohort_missing[('B2', 'unseen_species_unseen_genus')]}"
)

lines.append("")

lines.append(
    f"Frozen reserve replacements selected: "
    f"{len(chosen_replacements)}"
)

lines.append("")

lines.append(
    "Replacement policy:"
)

lines.append(
    "- same cohort"
)

lines.append(
    "- same cohort subtype"
)

lines.append(
    "- same GTDB species cluster"
)

lines.append(
    "- same genus and family"
)

lines.append(
    "- same source database class"
)

lines.append(
    "- unique accession"
)

lines.append(
    "- unique BioSample"
)

lines.append(
    "- replacement selected only from frozen reserve"
)

lines.append("")

lines.append(
    "Original frozen primary manifest modified: NO"
)

lines.append(
    "Updated manifest status: PROVISIONAL"
)

lines.append(
    "Reason: replacement FASTA files have not yet been downloaded."
)

lines.append("")

lines.append(
    "Replacement accessions:"
)

for row in replacement_plan:

    lines.append(
        f"{row['failed_accession']} "
        f"-> "
        f"{row['replacement_accession']} "
        f"[{row['cohort']} / "
        f"{row['cohort_subtype']} / "
        f"{row['gtdb_species_cluster']}]"
    )


lines.append("")

lines.append(
    "NEXT GATE:"
)

lines.append(
    "Download only the replacement accessions."
)

lines.append(
    "Do not perform feature extraction."
)

lines.append(
    "Final freeze is allowed only after sequence-level integrity audit."
)

lines.append("")

lines.append(
    "============================================================"
)


summary = "\n".join(
    lines
) + "\n"


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        summary
    )


print(
    summary
)

print(
    f"Replacement plan: {PLAN_OUT}"
)

print(
    f"Replacement accessions: {REPLACEMENT_ACCESSIONS_OUT}"
)

print(
    f"Provisional updated manifest: {UPDATED_MANIFEST_OUT}"
)

print(
    f"SHA256: {SHA_OUT}"
)
