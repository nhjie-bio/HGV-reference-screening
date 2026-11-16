#!/usr/bin/env python3

import csv
import gzip
import os
import sys
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

META = BASE / "00_metadata" / "bac120_metadata_r232.tsv.gz"
FIELD_DESC = (
    BASE
    / "00_metadata"
    / "auxillary_files"
    / "metadata_field_desc.tsv"
)

OUTDIR = BASE / "01_manifest"
LOGDIR = BASE / "logs"

OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

REPORT = OUTDIR / "gtdb_r232_schema_report.txt"
COL_TSV = OUTDIR / "gtdb_r232_metadata_columns.tsv"


def die(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


if not META.exists():
    die(f"Missing metadata file: {META}")

if not FIELD_DESC.exists():
    die(f"Missing field description file: {FIELD_DESC}")


# ------------------------------------------------------------
# Read GTDB metadata header and first three genome records only.
# No full-table loading is performed here.
# ------------------------------------------------------------

with gzip.open(
    META,
    "rt",
    encoding="utf-8",
    errors="replace",
    newline=""
) as handle:

    reader = csv.reader(
        handle,
        delimiter="\t"
    )

    try:
        header = next(reader)
    except StopIteration:
        die("GTDB metadata file is empty.")

    sample_rows = []

    for _ in range(3):
        try:
            sample_rows.append(
                next(reader)
            )
        except StopIteration:
            break


# ------------------------------------------------------------
# Read field descriptions.
# ------------------------------------------------------------

field_desc_rows = []

with open(
    FIELD_DESC,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as handle:

    reader = csv.reader(
        handle,
        delimiter="\t"
    )

    for row in reader:
        if row:
            field_desc_rows.append(row)


# ------------------------------------------------------------
# Identify fields relevant to HGV-V3.
# ------------------------------------------------------------

KEYWORDS = [
    "accession",
    "taxonomy",
    "representative",
    "species",
    "checkm",
    "completeness",
    "contamination",
    "assembly",
    "genome",
    "contig",
    "n50",
    "gc",
    "ambiguous",
    "biosample",
    "bioproject",
    "category",
    "isolate",
    "strain",
    "source",
]


def is_relevant(name):
    name_lower = name.lower()

    return any(
        keyword in name_lower
        for keyword in KEYWORDS
    )


relevant_columns = [
    col
    for col in header
    if is_relevant(col)
]


# ------------------------------------------------------------
# Build field description map.
#
# GTDB description file is retained raw because its schema itself
# is part of what we want to audit.
# ------------------------------------------------------------

desc_map = {}

for row in field_desc_rows:

    key = row[0].strip()

    if (
        not key
        or key.lower() in {
            "field",
            "field name",
            "field_name",
            "column",
            "column name",
        }
    ):
        continue

    description = " | ".join(
        cell.strip()
        for cell in row[1:]
    )

    desc_map[key] = description


# ------------------------------------------------------------
# Write metadata column table.
# ------------------------------------------------------------

with open(
    COL_TSV,
    "w",
    encoding="utf-8",
    newline=""
) as handle:

    writer = csv.writer(
        handle,
        delimiter="\t"
    )

    writer.writerow(
        [
            "column_number",
            "column_name",
            "hgv_v3_relevant",
            "description",
        ]
    )

    for i, column in enumerate(
        header,
        start=1
    ):

        writer.writerow(
            [
                i,
                column,
                int(is_relevant(column)),
                desc_map.get(column, ""),
            ]
        )


# ------------------------------------------------------------
# Convert first records to dictionaries.
# ------------------------------------------------------------

sample_dicts = []

for row in sample_rows:

    padded = (
        row
        +
        [""] * max(
            0,
            len(header) - len(row)
        )
    )

    sample_dicts.append(
        dict(
            zip(
                header,
                padded
            )
        )
    )


# ------------------------------------------------------------
# Generate human-readable schema report.
# ------------------------------------------------------------

lines = []

lines.append(
    "============================================================"
)

lines.append(
    "GTDB R232 METADATA SCHEMA AUDIT"
)

lines.append(
    "============================================================"
)

lines.append("")

lines.append(
    f"Metadata file: {META}"
)

lines.append(
    f"Field-description file: {FIELD_DESC}"
)

lines.append("")

lines.append(
    f"Total metadata columns: {len(header)}"
)

lines.append("")

lines.append(
    "------------------------------------------------------------"
)

lines.append(
    "ALL METADATA COLUMNS"
)

lines.append(
    "------------------------------------------------------------"
)

for i, column in enumerate(
    header,
    start=1
):
    lines.append(
        f"{i:03d}\t{column}"
    )


lines.append("")
lines.append(
    "------------------------------------------------------------"
)

lines.append(
    "HGV-V3 RELEVANT COLUMN CANDIDATES"
)

lines.append(
    "------------------------------------------------------------"
)

for column in relevant_columns:

    lines.append(
        column
    )


lines.append("")
lines.append(
    "------------------------------------------------------------"
)

lines.append(
    "RELEVANT FIELD DESCRIPTIONS"
)

lines.append(
    "------------------------------------------------------------"
)

for column in relevant_columns:

    description = desc_map.get(
        column,
        ""
    )

    lines.append(
        f"{column}\t{description}"
    )


lines.append("")
lines.append(
    "------------------------------------------------------------"
)

lines.append(
    "FIRST THREE RECORDS: RELEVANT FIELDS ONLY"
)

lines.append(
    "------------------------------------------------------------"
)

for n, record in enumerate(
    sample_dicts,
    start=1
):

    lines.append("")
    lines.append(
        f"[Genome record {n}]"
    )

    for column in relevant_columns:

        value = record.get(
            column,
            ""
        )

        if value != "":
            lines.append(
                f"{column}\t{value}"
            )


lines.append("")
lines.append(
    "------------------------------------------------------------"
)

lines.append(
    "RAW metadata_field_desc.tsv HEADER / FIRST 20 LINES"
)

lines.append(
    "------------------------------------------------------------"
)

for row in field_desc_rows[:20]:

    lines.append(
        "\t".join(row)
    )


lines.append("")
lines.append(
    "============================================================"
)

lines.append(
    "SCHEMA INSPECTION COMPLETE"
)

lines.append(
    "No filtering or cohort selection was performed."
)

lines.append(
    "============================================================"
)


report_text = "\n".join(lines) + "\n"


with open(
    REPORT,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        report_text
    )


print(
    report_text
)

print(
    f"Report written to: {REPORT}"
)

print(
    f"Column table written to: {COL_TSV}"
)
