#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path("HGV_V3_revision_20260915")

MANIFEST = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916"
    / "final_master_manifest_3154.tsv"
)

FEATURE_DIR = (
    BASE
    / "03_features/canonical_k6_full3154_v1"
)

MATRIX = FEATURE_DIR / "canonical_k6_features.float32.npy"
ROWS = FEATURE_DIR / "canonical_k6_rows.tsv"

OUTDIR = (
    BASE
    / "04_splits/preflight_capacity_v1_20260916"
)

EXPECTED_TOTAL = 3154

EXPECTED_COHORTS = {
    "A": 2374,
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193,
}

MIN_A_SPECIES = 120
MIN_A_GENERA = 30
MIN_A_FAMILIES = 12


def die(msg):
    print()
    print("SPLIT_PREFLIGHT=FAIL")
    print(msg)
    sys.exit(2)


def read_tsv(path):
    if not path.is_file():
        die(f"Missing file: {path}")

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    if not fields:
        die(f"No TSV header: {path}")

    return fields, rows


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def write_tsv(path, rows, fields):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fields,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def clean(v):
    return "" if v is None else str(v).strip()


print("=" * 76)
print("HGV-V3 FEATURE + SPLIT CAPACITY PREFLIGHT")
print("=" * 76)

# ------------------------------------------------------------------
# Load manifest
# ------------------------------------------------------------------

manifest_fields, manifest_rows = read_tsv(MANIFEST)

required_manifest = {
    "assembly_accession_final",
    "final_cohort",
    "fasta_path",
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
}

missing_cols = required_manifest - set(manifest_fields)

if missing_cols:
    die(
        "Manifest missing required columns: "
        + ", ".join(sorted(missing_cols))
    )

if len(manifest_rows) != EXPECTED_TOTAL:
    die(
        f"Expected {EXPECTED_TOTAL} manifest rows; "
        f"observed {len(manifest_rows)}"
    )

manifest_accessions = [
    clean(r["assembly_accession_final"])
    for r in manifest_rows
]

if len(set(manifest_accessions)) != EXPECTED_TOTAL:
    die("Manifest accession uniqueness gate failed")

# ------------------------------------------------------------------
# Load feature row mapping
# ------------------------------------------------------------------

row_fields, feature_rows = read_tsv(ROWS)

required_rows = {
    "row_index",
    "assembly_accession",
    "final_cohort",
    "fasta_path",
}

missing_row_cols = required_rows - set(row_fields)

if missing_row_cols:
    die(
        "Feature row map missing columns: "
        + ", ".join(sorted(missing_row_cols))
    )

if len(feature_rows) != EXPECTED_TOTAL:
    die(
        f"Expected {EXPECTED_TOTAL} feature rows; "
        f"observed {len(feature_rows)}"
    )

feature_accessions = [
    clean(r["assembly_accession"])
    for r in feature_rows
]

# Strict row-order mapping
if manifest_accessions != feature_accessions:
    mismatch = []

    for i, (a, b) in enumerate(
        zip(manifest_accessions, feature_accessions)
    ):
        if a != b:
            mismatch.append((i, a, b))
            if len(mismatch) >= 10:
                break

    print("First manifest/feature row mismatches:")
    for x in mismatch:
        print(x)

    die(
        "Manifest and feature matrix row identities/order differ"
    )

print()
print("[Feature identity]")
print(f"Manifest rows       : {len(manifest_rows)}")
print(f"Feature row records : {len(feature_rows)}")
print("Row identity/order  : PASS")

# ------------------------------------------------------------------
# Load feature matrix
# ------------------------------------------------------------------

if not MATRIX.is_file():
    die(f"Missing feature matrix: {MATRIX}")

X = np.load(MATRIX, mmap_mode="r")

print()
print("[Feature matrix]")
print(f"Shape : {X.shape}")
print(f"Dtype : {X.dtype}")

if X.shape != (EXPECTED_TOTAL, 2080):
    die(
        f"Expected matrix shape {(EXPECTED_TOTAL, 2080)}; "
        f"observed {X.shape}"
    )

if X.dtype != np.float32:
    die(f"Expected float32 matrix; observed {X.dtype}")

# Matrix is only ~26 MB, so full finite/row-sum audit is safe.
Xa = np.asarray(X)

if not np.isfinite(Xa).all():
    die("Feature matrix contains NaN or Inf")

row_sums = Xa.sum(axis=1, dtype=np.float64)

max_row_sum_error = float(
    np.max(np.abs(row_sums - 1.0))
)

min_feature = float(np.min(Xa))
max_feature = float(np.max(Xa))

print(f"Finite values       : PASS")
print(f"Minimum feature     : {min_feature:.12g}")
print(f"Maximum feature     : {max_feature:.12g}")
print(f"Max row-sum error   : {max_row_sum_error:.12g}")

if min_feature < 0:
    die("Negative k-mer frequency detected")

if max_row_sum_error > 1e-5:
    die(
        f"Feature normalization failed: "
        f"{max_row_sum_error}"
    )

# ------------------------------------------------------------------
# Cohort audit
# ------------------------------------------------------------------

cohort_counts = Counter(
    clean(r["final_cohort"])
    for r in manifest_rows
)

print()
print("[Final cohort counts]")

for cohort, expected in EXPECTED_COHORTS.items():
    observed = cohort_counts.get(cohort, 0)

    print(
        f"{cohort:18s}: "
        f"{observed:4d} "
        f"(expected {expected})"
    )

    if observed != expected:
        die(
            f"Cohort count mismatch for {cohort}: "
            f"{observed} != {expected}"
        )

if sum(cohort_counts.values()) != EXPECTED_TOTAL:
    die("Final cohort total mismatch")

# ------------------------------------------------------------------
# Taxonomy completeness
# ------------------------------------------------------------------

taxonomy_cols = [
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
]

print()
print("[Taxonomy completeness]")

for col in taxonomy_cols:
    missing = sum(
        1
        for r in manifest_rows
        if not clean(r[col])
    )

    print(f"{col:24s}: missing={missing}")

    if missing != 0:
        die(
            f"Missing taxonomy values in {col}: {missing}"
        )

# ------------------------------------------------------------------
# Separate cohorts
# ------------------------------------------------------------------

A = [
    r for r in manifest_rows
    if clean(r["final_cohort"]) == "A"
]

B1 = [
    r for r in manifest_rows
    if clean(r["final_cohort"]) == "B1"
]

B2_seen = [
    r for r in manifest_rows
    if clean(r["final_cohort"]) == "B2_seen_genus"
]

B2_unseen = [
    r for r in manifest_rows
    if clean(r["final_cohort"]) == "B2_unseen_genus"
]

# ------------------------------------------------------------------
# Development taxonomy diversity
# ------------------------------------------------------------------

A_species_clusters = {
    clean(r["gtdb_species_cluster"])
    for r in A
}

A_species = {
    clean(r["gtdb_species"])
    for r in A
}

A_genera = {
    clean(r["gtdb_genus"])
    for r in A
}

A_families = {
    clean(r["gtdb_family"])
    for r in A
}

print()
print("[Development A taxonomy]")
print(f"Genomes          : {len(A)}")
print(f"Species clusters : {len(A_species_clusters)}")
print(f"Species labels   : {len(A_species)}")
print(f"Genera           : {len(A_genera)}")
print(f"Families         : {len(A_families)}")

if len(A_species_clusters) < MIN_A_SPECIES:
    die(
        f"Development species diversity below "
        f"{MIN_A_SPECIES}"
    )

if len(A_genera) < MIN_A_GENERA:
    die(
        f"Development genus diversity below "
        f"{MIN_A_GENERA}"
    )

if len(A_families) < MIN_A_FAMILIES:
    die(
        f"Development family diversity below "
        f"{MIN_A_FAMILIES}"
    )

# ------------------------------------------------------------------
# Species-level positive-pair capacity
# ------------------------------------------------------------------

species_cluster_counts = Counter(
    clean(r["gtdb_species_cluster"])
    for r in A
)

singleton_species = {
    k: v
    for k, v in species_cluster_counts.items()
    if v < 2
}

print()
print("[Same-species positive-pair capacity]")
print(
    f"Species clusters with >=2 genomes : "
    f"{sum(v >= 2 for v in species_cluster_counts.values())}"
)
print(
    f"Species clusters with <2 genomes  : "
    f"{len(singleton_species)}"
)
print(
    f"Minimum genomes/species cluster   : "
    f"{min(species_cluster_counts.values())}"
)
print(
    f"Maximum genomes/species cluster   : "
    f"{max(species_cluster_counts.values())}"
)

if singleton_species:
    print("Singleton species examples:")
    for k, v in list(singleton_species.items())[:20]:
        print(k, v)

    die(
        "Some Development species clusters cannot "
        "support positive pairs"
    )

# ------------------------------------------------------------------
# Genus structure for hard negatives
# ------------------------------------------------------------------

genus_to_species = defaultdict(set)
genus_to_genomes = Counter()

for r in A:
    g = clean(r["gtdb_genus"])
    s = clean(r["gtdb_species_cluster"])

    genus_to_species[g].add(s)
    genus_to_genomes[g] += 1

genera_ge2_species = sum(
    len(v) >= 2
    for v in genus_to_species.values()
)

genera_ge3_species = sum(
    len(v) >= 3
    for v in genus_to_species.values()
)

print()
print("[Hard-negative taxonomy capacity]")
print(
    f"Genera with >=2 species clusters : "
    f"{genera_ge2_species}/{len(genus_to_species)}"
)
print(
    f"Genera with >=3 species clusters : "
    f"{genera_ge3_species}/{len(genus_to_species)}"
)

# ------------------------------------------------------------------
# Family structure
# ------------------------------------------------------------------

family_to_genera = defaultdict(set)
family_to_species = defaultdict(set)
family_to_genomes = Counter()

for r in A:
    f = clean(r["gtdb_family"])
    g = clean(r["gtdb_genus"])
    s = clean(r["gtdb_species_cluster"])

    family_to_genera[f].add(g)
    family_to_species[f].add(s)
    family_to_genomes[f] += 1

families_ge2_genera = sum(
    len(v) >= 2
    for v in family_to_genera.values()
)

families_ge2_species = sum(
    len(v) >= 2
    for v in family_to_species.values()
)

print()
print("[Family-holdout capacity]")
print(
    f"Families total            : "
    f"{len(family_to_genera)}"
)
print(
    f"Families with >=2 genera  : "
    f"{families_ge2_genera}"
)
print(
    f"Families with >=2 species : "
    f"{families_ge2_species}"
)

# ------------------------------------------------------------------
# External-cohort taxonomic design audit
# ------------------------------------------------------------------

B1_species = {
    clean(r["gtdb_species_cluster"])
    for r in B1
}

B2_seen_species = {
    clean(r["gtdb_species_cluster"])
    for r in B2_seen
}

B2_seen_genera = {
    clean(r["gtdb_genus"])
    for r in B2_seen
}

B2_unseen_species = {
    clean(r["gtdb_species_cluster"])
    for r in B2_unseen
}

B2_unseen_genera = {
    clean(r["gtdb_genus"])
    for r in B2_unseen
}

b1_species_not_in_A = (
    B1_species - A_species_clusters
)

b2_seen_species_overlap_A = (
    B2_seen_species & A_species_clusters
)

b2_seen_genus_not_in_A = (
    B2_seen_genera - A_genera
)

b2_unseen_species_overlap_A = (
    B2_unseen_species & A_species_clusters
)

b2_unseen_genus_overlap_A = (
    B2_unseen_genera & A_genera
)

print()
print("[External taxonomy design]")
print(
    f"B1 species clusters not in A       : "
    f"{len(b1_species_not_in_A)}"
)
print(
    f"B2-seen species overlap A          : "
    f"{len(b2_seen_species_overlap_A)}"
)
print(
    f"B2-seen genera not in A            : "
    f"{len(b2_seen_genus_not_in_A)}"
)
print(
    f"B2-unseen species overlap A        : "
    f"{len(b2_unseen_species_overlap_A)}"
)
print(
    f"B2-unseen genus overlap A          : "
    f"{len(b2_unseen_genus_overlap_A)}"
)

if b1_species_not_in_A:
    die(
        "B1 matched-species design violated: "
        "some B1 species clusters are absent from A"
    )

if b2_seen_species_overlap_A:
    die(
        "B2 seen-genus cohort contains species "
        "clusters already present in A"
    )

if b2_seen_genus_not_in_A:
    die(
        "B2 seen-genus cohort contains genera "
        "not represented in A"
    )

if b2_unseen_species_overlap_A:
    die(
        "B2 unseen-genus cohort contains species "
        "clusters already represented in A"
    )

if b2_unseen_genus_overlap_A:
    die(
        "B2 unseen-genus cohort contains genera "
        "already represented in A"
    )

# ------------------------------------------------------------------
# Only after every gate passes create output directory
# ------------------------------------------------------------------

if OUTDIR.exists():
    die(
        f"Output directory already exists; "
        f"refusing overwrite: {OUTDIR}"
    )

OUTDIR.mkdir(parents=True, exist_ok=False)

# ------------------------------------------------------------------
# Write group-size source data
# ------------------------------------------------------------------

species_rows = []

for s, n in sorted(
    species_cluster_counts.items(),
    key=lambda x: (-x[1], x[0]),
):
    rows_s = [
        r for r in A
        if clean(r["gtdb_species_cluster"]) == s
    ]

    species_rows.append({
        "gtdb_species_cluster": s,
        "gtdb_species": clean(rows_s[0]["gtdb_species"]),
        "gtdb_genus": clean(rows_s[0]["gtdb_genus"]),
        "gtdb_family": clean(rows_s[0]["gtdb_family"]),
        "n_genomes": n,
        "positive_pairs_possible": (
            n * (n - 1) // 2
        ),
    })

write_tsv(
    OUTDIR / "development_species_cluster_sizes.tsv",
    species_rows,
    [
        "gtdb_species_cluster",
        "gtdb_species",
        "gtdb_genus",
        "gtdb_family",
        "n_genomes",
        "positive_pairs_possible",
    ],
)

genus_rows = []

for g in sorted(genus_to_species):
    genus_rows.append({
        "gtdb_genus": g,
        "n_genomes": genus_to_genomes[g],
        "n_species_clusters": len(genus_to_species[g]),
        "supports_hard_negative": (
            1 if len(genus_to_species[g]) >= 2 else 0
        ),
    })

write_tsv(
    OUTDIR / "development_genus_sizes.tsv",
    genus_rows,
    [
        "gtdb_genus",
        "n_genomes",
        "n_species_clusters",
        "supports_hard_negative",
    ],
)

family_rows = []

for f in sorted(family_to_genera):
    family_rows.append({
        "gtdb_family": f,
        "n_genomes": family_to_genomes[f],
        "n_genera": len(family_to_genera[f]),
        "n_species_clusters": len(family_to_species[f]),
    })

write_tsv(
    OUTDIR / "development_family_sizes.tsv",
    family_rows,
    [
        "gtdb_family",
        "n_genomes",
        "n_genera",
        "n_species_clusters",
    ],
)

# ------------------------------------------------------------------
# Compact audit TSV
# ------------------------------------------------------------------

audit_rows = [
    {
        "check": "total_genomes",
        "observed": EXPECTED_TOTAL,
        "expected": EXPECTED_TOTAL,
        "gate": "PASS",
    },
    {
        "check": "feature_matrix_rows",
        "observed": X.shape[0],
        "expected": EXPECTED_TOTAL,
        "gate": "PASS",
    },
    {
        "check": "feature_dimension",
        "observed": X.shape[1],
        "expected": 2080,
        "gate": "PASS",
    },
    {
        "check": "development_A_genomes",
        "observed": len(A),
        "expected": 2374,
        "gate": "PASS",
    },
    {
        "check": "external_total",
        "observed": len(B1) + len(B2_seen) + len(B2_unseen),
        "expected": 780,
        "gate": "PASS",
    },
    {
        "check": "A_species_clusters",
        "observed": len(A_species_clusters),
        "expected": f">={MIN_A_SPECIES}",
        "gate": "PASS",
    },
    {
        "check": "A_genera",
        "observed": len(A_genera),
        "expected": f">={MIN_A_GENERA}",
        "gate": "PASS",
    },
    {
        "check": "A_families",
        "observed": len(A_families),
        "expected": f">={MIN_A_FAMILIES}",
        "gate": "PASS",
    },
    {
        "check": "A_species_singletons",
        "observed": len(singleton_species),
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "B1_species_not_in_A",
        "observed": len(b1_species_not_in_A),
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "B2_seen_species_overlap_A",
        "observed": len(b2_seen_species_overlap_A),
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "B2_seen_genus_not_in_A",
        "observed": len(b2_seen_genus_not_in_A),
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "B2_unseen_species_overlap_A",
        "observed": len(b2_unseen_species_overlap_A),
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "B2_unseen_genus_overlap_A",
        "observed": len(b2_unseen_genus_overlap_A),
        "expected": 0,
        "gate": "PASS",
    },
]

write_tsv(
    OUTDIR / "split_preflight_checks.tsv",
    audit_rows,
    [
        "check",
        "observed",
        "expected",
        "gate",
    ],
)

# ------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------

provenance = {
    "manifest": str(MANIFEST),
    "manifest_sha256": sha256_file(MANIFEST),
    "feature_matrix": str(MATRIX),
    "feature_matrix_sha256": sha256_file(MATRIX),
    "feature_rows": str(ROWS),
    "feature_rows_sha256": sha256_file(ROWS),
    "feature_shape": [
        int(X.shape[0]),
        int(X.shape[1]),
    ],
    "max_abs_row_sum_error": max_row_sum_error,
    "development_genomes": len(A),
    "development_species_clusters": len(A_species_clusters),
    "development_genera": len(A_genera),
    "development_families": len(A_families),
    "external_B1": len(B1),
    "external_B2_seen_genus": len(B2_seen),
    "external_B2_unseen_genus": len(B2_unseen),
}

with open(
    OUTDIR / "split_preflight_provenance.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        provenance,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

# ------------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------------

report = OUTDIR / "SPLIT_PREFLIGHT_REPORT.txt"

with open(
    report,
    "w",
    encoding="utf-8",
) as f:
    f.write("HGV-V3 SPLIT PREFLIGHT REPORT\n")
    f.write("=" * 76 + "\n\n")

    f.write("STATUS: PASS\n\n")

    f.write("FEATURES\n")
    f.write(f"Matrix shape: {X.shape}\n")
    f.write(f"Feature dtype: {X.dtype}\n")
    f.write(
        f"Max row normalization error: "
        f"{max_row_sum_error:.12g}\n"
    )
    f.write("Manifest-feature row identity: PASS\n\n")

    f.write("COHORTS\n")
    for c in (
        "A",
        "B1",
        "B2_seen_genus",
        "B2_unseen_genus",
    ):
        f.write(f"{c}: {cohort_counts[c]}\n")

    f.write("\nDEVELOPMENT TAXONOMY\n")
    f.write(
        f"Species clusters: "
        f"{len(A_species_clusters)}\n"
    )
    f.write(f"Genera: {len(A_genera)}\n")
    f.write(f"Families: {len(A_families)}\n")
    f.write(
        f"Minimum genomes/species cluster: "
        f"{min(species_cluster_counts.values())}\n"
    )

    f.write("\nNEGATIVE-PAIR CAPACITY\n")
    f.write(
        f"Genera with >=2 species: "
        f"{genera_ge2_species}\n"
    )
    f.write(
        f"Genera with >=3 species: "
        f"{genera_ge3_species}\n"
    )
    f.write(
        f"Families with >=2 genera: "
        f"{families_ge2_genera}\n"
    )

    f.write("\nEXTERNAL DESIGN\n")
    f.write(
        f"B1 species not in A: "
        f"{len(b1_species_not_in_A)}\n"
    )
    f.write(
        f"B2-seen species overlap A: "
        f"{len(b2_seen_species_overlap_A)}\n"
    )
    f.write(
        f"B2-seen genera not in A: "
        f"{len(b2_seen_genus_not_in_A)}\n"
    )
    f.write(
        f"B2-unseen species overlap A: "
        f"{len(b2_unseen_species_overlap_A)}\n"
    )
    f.write(
        f"B2-unseen genera overlap A: "
        f"{len(b2_unseen_genus_overlap_A)}\n"
    )

    f.write("\nSPLIT_PREFLIGHT=PASS\n")

# ------------------------------------------------------------------
# Output hashes
# ------------------------------------------------------------------

outputs_to_hash = [
    OUTDIR / "development_species_cluster_sizes.tsv",
    OUTDIR / "development_genus_sizes.tsv",
    OUTDIR / "development_family_sizes.tsv",
    OUTDIR / "split_preflight_checks.tsv",
    OUTDIR / "split_preflight_provenance.json",
    OUTDIR / "SPLIT_PREFLIGHT_REPORT.txt",
]

with open(
    OUTDIR / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in outputs_to_hash:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )

print()
print("=" * 76)
print("ALL SPLIT PREFLIGHT GATES PASSED")
print("=" * 76)
print(f"Development genomes : {len(A)}")
print(
    f"Species clusters    : "
    f"{len(A_species_clusters)}"
)
print(f"Genera              : {len(A_genera)}")
print(f"Families            : {len(A_families)}")
print(
    f"Min genomes/species : "
    f"{min(species_cluster_counts.values())}"
)
print(
    f"Genera >=2 species  : "
    f"{genera_ge2_species}"
)
print(
    f"Families >=2 genera : "
    f"{families_ge2_genera}"
)
print(f"Output              : {OUTDIR}")
print()
print("SPLIT_PREFLIGHT=PASS")
