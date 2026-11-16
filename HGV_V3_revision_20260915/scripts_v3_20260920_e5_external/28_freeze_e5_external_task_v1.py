#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import os
import sys
from datetime import datetime


BASE = Path(
    os.environ.get(
        "HGV_BASE",
        "HGV_V3_revision_20260915"
    )
).resolve()

MANIFEST = (
    BASE
    / "01_manifest"
    / "final_3154_freeze_v1_20260916"
    / "final_master_manifest_3154.tsv"
)

EXTDIR = (
    BASE
    / "09_external_validation"
    / "external_one_shot_evaluation_v1_20260917"
)

QUERY_SOURCE = EXTDIR / "external_query_level_results.tsv"
METRIC_SOURCE = EXTDIR / "external_metrics_seed_level.tsv"

OUT = (
    BASE
    / "09_baselines"
    / "E5_external_unified_v1_20260920"
    / "00_task_freeze"
)


# =============================================================================
# Utility
# =============================================================================

def fail(msg):
    print(f"\nFATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path):
    if not path.exists():
        fail(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames or []

    return fields, rows


def choose_col(fields, candidates, label):
    for c in candidates:
        if c in fields:
            return c

    fail(
        f"Cannot identify {label} column.\n"
        f"Candidates: {candidates}\n"
        f"Available: {fields}"
    )


def is_present(v):
    if v is None:
        return False

    x = str(v).strip()

    return x not in {
        "",
        "nan",
        "NaN",
        "NA",
        "None",
        "null",
    }


# =============================================================================
# Start
# =============================================================================

print("=" * 100)
print("HGV-V3 E5 EXTERNAL FIXED TASK FREEZE v1")
print("=" * 100)
print("BASE         :", BASE)
print("MANIFEST     :", MANIFEST)
print("QUERY SOURCE :", QUERY_SOURCE)
print("OUT          :", OUT)


if OUT.exists():
    fail(
        f"Output directory already exists:\n{OUT}\n"
        "Refusing to overwrite a previous freeze."
    )

OUT.mkdir(parents=True, exist_ok=False)


# =============================================================================
# 1. Read final 3154 manifest
# =============================================================================

mf_fields, mf_rows = read_tsv(MANIFEST)

acc_col = choose_col(
    mf_fields,
    [
        "assembly_accession",
        "accession",
        "genome_accession",
        "gtdb_accession",
    ],
    "accession"
)

cohort_col = choose_col(
    mf_fields,
    [
        "source_cohort",
        "cohort",
        "dataset",
    ],
    "cohort"
)

fasta_col = choose_col(
    mf_fields,
    [
        "fasta_path",
        "genome_fasta",
        "fasta",
        "path",
    ],
    "FASTA path"
)

print()
print("Manifest columns:")
print("  accession :", acc_col)
print("  cohort    :", cohort_col)
print("  fasta     :", fasta_col)


# Build accession map

manifest_by_acc = {}

for r in mf_rows:
    acc = r[acc_col].strip()

    if not acc:
        continue

    if acc in manifest_by_acc:
        fail(f"Duplicate accession in final manifest: {acc}")

    manifest_by_acc[acc] = r


if len(manifest_by_acc) != 3154:
    fail(
        f"Expected 3154 unique manifest accessions; "
        f"found {len(manifest_by_acc)}"
    )


# =============================================================================
# 2. Freeze Dataset A reference database
# =============================================================================

refs = []

for r in mf_rows:
    cohort = str(r[cohort_col]).strip()

    if cohort == "A":
        refs.append(r)


if len(refs) != 2374:
    fail(
        f"Expected exactly 2374 Dataset-A references; "
        f"found {len(refs)}"
    )


# Validate FASTA paths

missing_ref_fasta = []

for r in refs:
    p = Path(r[fasta_col])

    if not p.exists():
        missing_ref_fasta.append(
            (
                r[acc_col],
                str(p)
            )
        )


if missing_ref_fasta:
    print("\nMissing A-reference FASTA examples:")
    for x in missing_ref_fasta[:20]:
        print(" ", x)

    fail(
        f"{len(missing_ref_fasta)} reference FASTA files are missing."
    )


# =============================================================================
# 3. Read existing formal external query results
#
# Use RawK6_BrayCurtis only as a TASK-MEMBERSHIP source.
# We do NOT use its performance values.
# =============================================================================

q_fields, q_rows = read_tsv(QUERY_SOURCE)

required_qcols = [
    "method",
    "cohort",
    "query_accession",
    "query_species",
    "query_genus",
    "query_family",
    "species_first_rank",
    "genus_first_rank",
    "family_first_rank",
]

for c in required_qcols:
    if c not in q_fields:
        fail(
            f"Missing required external query column: {c}"
        )


method_names = sorted({
    r["method"].strip()
    for r in q_rows
    if r["method"].strip()
})

print()
print("Existing formal external methods:")
for m in method_names:
    print(" ", m)


expected_existing_methods = {
    "HGV_Dashing2_tau0p98",
    "RawK6_BrayCurtis",
    "RawK6_Cosine",
    "RawK6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
}

if set(method_names) != expected_existing_methods:
    fail(
        "Existing external method set is not the expected frozen 6-method block.\n"
        f"Observed: {method_names}"
    )


canonical_rows = [
    r for r in q_rows
    if r["method"] == "RawK6_BrayCurtis"
]


# One deterministic row per external query expected.

if len(canonical_rows) != 780:
    fail(
        f"Expected 780 RawK6_BrayCurtis external rows; "
        f"found {len(canonical_rows)}"
    )


# =============================================================================
# 4. Build exact frozen query table
# =============================================================================

query_records = []

seen = set()

for r in canonical_rows:

    cohort = r["cohort"].strip()
    acc = r["query_accession"].strip()

    key = (cohort, acc)

    if key in seen:
        fail(f"Duplicate external query: {key}")

    seen.add(key)

    if acc not in manifest_by_acc:
        fail(
            f"External query accession not found in final manifest: {acc}"
        )

    mr = manifest_by_acc[acc]

    fasta = mr[fasta_col].strip()

    if not Path(fasta).exists():
        fail(
            f"Missing query FASTA for {acc}: {fasta}"
        )

    query_records.append({
        "cohort": cohort,
        "query_accession": acc,
        "query_species": r["query_species"].strip(),
        "query_genus": r["query_genus"].strip(),
        "query_family": r["query_family"].strip(),
        "species_eligible": int(
            is_present(r["species_first_rank"])
        ),
        "genus_eligible": int(
            is_present(r["genus_first_rank"])
        ),
        "family_eligible": int(
            is_present(r["family_first_rank"])
        ),
        "fasta_path": fasta,
    })


# =============================================================================
# 5. Strict frozen-count gates
# =============================================================================

EXPECTED = {
    "B1": {
        "n": 391,
        "species": 391,
        "genus": 391,
        "family": 391,
    },

    "B2_seen_genus": {
        "n": 196,
        "species": 0,
        "genus": 196,
        "family": 196,
    },

    "B2_unseen_genus": {
        "n": 193,
        "species": 0,
        "genus": 0,
        "family": 143,
    },
}


print()
print("Frozen external task counts:")

for cohort, exp in EXPECTED.items():

    x = [
        r for r in query_records
        if r["cohort"] == cohort
    ]

    n = len(x)

    ns = sum(
        r["species_eligible"]
        for r in x
    )

    ng = sum(
        r["genus_eligible"]
        for r in x
    )

    nf = sum(
        r["family_eligible"]
        for r in x
    )

    print(
        f"  {cohort:<16} "
        f"n={n:<3} "
        f"species={ns:<3} "
        f"genus={ng:<3} "
        f"family={nf:<3}"
    )

    observed = {
        "n": n,
        "species": ns,
        "genus": ng,
        "family": nf,
    }

    if observed != exp:
        fail(
            f"Frozen eligibility mismatch for {cohort}.\n"
            f"Expected: {exp}\n"
            f"Observed: {observed}"
        )


if len(query_records) != 780:
    fail(
        f"Expected 780 total external queries; "
        f"found {len(query_records)}"
    )


ref_accessions = {
    r[acc_col].strip()
    for r in refs
}

query_accessions = {
    r["query_accession"]
    for r in query_records
}

overlap = ref_accessions & query_accessions

if overlap:
    fail(
        f"Reference/query accession overlap detected: "
        f"{sorted(overlap)[:20]}"
    )


# =============================================================================
# 6. Write reference manifest
# =============================================================================

ref_tsv = OUT / "REFERENCE_A_2374.tsv"

with ref_tsv.open(
    "w",
    encoding="utf-8",
    newline=""
) as fh:

    w = csv.DictWriter(
        fh,
        fieldnames=mf_fields,
        delimiter="\t"
    )

    w.writeheader()

    for r in sorted(
        refs,
        key=lambda z: z[acc_col]
    ):
        w.writerow(r)


ref_list = OUT / "REFERENCE_A_2374_FASTA.list"

with ref_list.open(
    "w",
    encoding="utf-8"
) as fh:

    for r in sorted(
        refs,
        key=lambda z: z[acc_col]
    ):
        fh.write(
            r[fasta_col].strip()
            + "\n"
        )


# =============================================================================
# 7. Write query manifests/lists
# =============================================================================

q_fields_out = [
    "cohort",
    "query_accession",
    "query_species",
    "query_genus",
    "query_family",
    "species_eligible",
    "genus_eligible",
    "family_eligible",
    "fasta_path",
]


cohort_outputs = {
    "B1": (
        "QUERY_B1_391.tsv",
        "QUERY_B1_391_FASTA.list"
    ),

    "B2_seen_genus": (
        "QUERY_B2_SEEN_196.tsv",
        "QUERY_B2_SEEN_196_FASTA.list"
    ),

    "B2_unseen_genus": (
        "QUERY_B2_UNSEEN_193.tsv",
        "QUERY_B2_UNSEEN_193_FASTA.list"
    ),
}


generated = [
    ref_tsv,
    ref_list,
]


for cohort, (tsv_name, list_name) in cohort_outputs.items():

    rows = sorted(
        [
            r for r in query_records
            if r["cohort"] == cohort
        ],
        key=lambda z: z["query_accession"]
    )

    tsv_path = OUT / tsv_name
    list_path = OUT / list_name

    with tsv_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as fh:

        w = csv.DictWriter(
            fh,
            fieldnames=q_fields_out,
            delimiter="\t"
        )

        w.writeheader()
        w.writerows(rows)

    with list_path.open(
        "w",
        encoding="utf-8"
    ) as fh:

        for r in rows:
            fh.write(
                r["fasta_path"]
                + "\n"
            )

    generated.extend([
        tsv_path,
        list_path,
    ])


# =============================================================================
# 8. Freeze report
# =============================================================================

report = OUT / "EXTERNAL_TASK_FREEZE.txt"

with report.open(
    "w",
    encoding="utf-8"
) as fh:

    fh.write(
        "HGV-V3 E5 EXTERNAL FIXED QUERY/REFERENCE TASK\n"
    )

    fh.write(
        "=" * 80
        + "\n"
    )

    fh.write(
        f"Created: {datetime.now().isoformat(timespec='seconds')}\n"
    )

    fh.write(
        f"Project root: {BASE}\n"
    )

    fh.write(
        f"Reference database: Dataset A = 2374 genomes\n"
    )

    fh.write(
        "External queries:\n"
    )

    fh.write(
        "  B1 = 391\n"
        "  B2_seen_genus = 196\n"
        "  B2_unseen_genus = 193\n"
        "  total = 780\n"
    )

    fh.write(
        "\nEligibility:\n"
    )

    fh.write(
        "  B1: species=391, genus=391, family=391\n"
    )

    fh.write(
        "  B2_seen_genus: species=0, genus=196, family=196\n"
    )

    fh.write(
        "  B2_unseen_genus: species=0, genus=0, family=143\n"
    )

    fh.write(
        "\nExisting completed external method variants:\n"
    )

    for m in method_names:
        fh.write(
            f"  {m}\n"
        )

    fh.write(
        "\nExternal E5 methods still to execute:\n"
        "  Mash native\n"
        "  Dashing2 SetSketch native\n"
        "  FastANI\n"
        "  skani\n"
        "  DNABERT-S\n"
    )

    fh.write(
        "\nRULES:\n"
        "- Do not change A reference membership.\n"
        "- Do not change B1/B2 query membership.\n"
        "- Do not retune HGV or thresholds on external data.\n"
        "- Use the same query/reference task for every new baseline.\n"
        "- Missing FastANI/skani comparisons must remain structurally missing; "
        "do not impute ANI=0.\n"
    )

generated.append(report)


# =============================================================================
# 9. SHA256 freeze
# =============================================================================

sha_file = OUT / "EXTERNAL_TASK_SHA256.txt"

source_files = [
    MANIFEST,
    QUERY_SOURCE,
    METRIC_SOURCE,
]

with sha_file.open(
    "w",
    encoding="utf-8"
) as fh:

    fh.write("# SOURCE FILES\n")

    for p in source_files:

        fh.write(
            f"{sha256(p)}  {p}\n"
        )

    fh.write("\n# GENERATED FREEZE FILES\n")

    for p in generated:

        fh.write(
            f"{sha256(p)}  {p}\n"
        )


# =============================================================================
# Final
# =============================================================================

print()
print("=" * 100)
print("TASK FREEZE PASS")
print("=" * 100)

print("Reference genomes :", len(refs))
print("External queries  :", len(query_records))
print("Overlap           :", len(overlap))

print()
print("Output:")
print(" ", OUT)
print()
print("Files:")
for p in generated:
    print(" ", p.name)

print(" ", sha_file.name)

print()
print("NEXT:")
print(
    "Run Mash / Dashing2 / FastANI / skani / DNABERT-S "
    "on these frozen manifests only."
)
