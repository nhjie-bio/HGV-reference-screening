import csv
import gzip
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent

META = BASE / "00_metadata/bac120_metadata_r232.tsv.gz"

MASTER = BASE / (
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

FEATURES = BASE / (
    "03_features/canonical_k6_full3154_v1/"
    "canonical_k6_features.float32.npy"
)

OUT = BASE / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W06_E13_MAG"
)

SEED = 20261114
TARGET_PER_BIN = 20
RESERVOIR_SIZE = 500

FIELDS = [
    "accession",
    "biosample",
    "completeness",
    "contamination",
    "quality_bin",
    "reference_target_level",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
    "ncbi_genome_category",
    "selection_seed"
]


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def read_tsv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, columns):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def accession_of(row):
    for field in (
        "ncbi_genbank_assembly_accession",
        "accession"
    ):
        text = row.get(field, "") or ""
        match = re.search(
            r"(GC[AF]_\d+\.\d+)", text
        )
        if match:
            return match.group(1)
    return ""


def taxonomy_of(row):
    result = {}
    text = row.get("gtdb_taxonomy", "") or ""

    for token in text.split(";"):
        token = token.strip()

        if len(token) >= 4 and token[1:3] == "__":
            result[token[0]] = token

    return result


def completeness_bin(value):
    if 50 <= value < 70:
        return "50-70"
    if 70 <= value < 80:
        return "70-80"
    if 80 <= value < 90:
        return "80-90"
    if 90 <= value <= 100:
        return "90-100"
    return None


print("=" * 72)
print("W06 - REAL MAG CANDIDATE SELECTION")
print("=" * 72)

for path in (META, MASTER, FEATURES):
    require(path.is_file(), f"Missing: {path}")

# Check the original feature format without loading it into RAM.
X = np.load(
    FEATURES,
    mmap_mode="r",
    allow_pickle=False
)

require(
    X.shape == (3154, 2080),
    f"Unexpected feature dimensions: {X.shape}"
)

require(
    X.dtype == np.float32,
    f"Unexpected feature dtype: {X.dtype}"
)

print("[PASS] Original features:", X.shape, X.dtype)

master = read_tsv(MASTER)

require(
    len(master) == 3154,
    "Master manifest should have 3154 rows"
)

reference = [
    r for r in master
    if r.get("final_cohort", "").strip() == "A"
]

require(
    len(reference) == 2374,
    "A reference set should contain 2374 genomes"
)

ref_taxa = {
    "s": {
        r.get("gtdb_species", "").strip()
        for r in reference
    },
    "g": {
        r.get("gtdb_genus", "").strip()
        for r in reference
    },
    "f": {
        r.get("gtdb_family", "").strip()
        for r in reference
    }
}

existing_accessions = {
    r.get("assembly_accession_final", "").strip()
    for r in master
}

existing_biosamples = {
    r.get("biosample", "").strip()
    for r in master
}

existing_biosamples.discard("")
existing_biosamples.discard("none")

rng = random.Random(SEED)

reservoirs = defaultdict(list)
eligible = Counter()
eligible_levels = Counter()
categories = Counter()
counts = Counter()

print("[INFO] Scanning GTDB MAG metadata...", flush=True)

with gzip.open(
    META,
    "rt",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")
    columns = set(reader.fieldnames or [])

    required = {
        "ncbi_genome_category",
        "checkm2_completeness",
        "checkm2_contamination",
        "gtdb_taxonomy"
    }

    require(
        required.issubset(columns),
        "Missing GTDB columns: "
        + str(sorted(required - columns))
    )

    for row in reader:

        counts["all"] += 1

        category = (
            row.get("ncbi_genome_category") or ""
        ).strip().lower()

        categories[category] += 1

        if category != "derived from metagenome":
            continue

        counts["MAG_category"] += 1

        completeness = numeric(
            row.get("checkm2_completeness")
        )

        contamination = numeric(
            row.get("checkm2_contamination")
        )

        if completeness is None or contamination is None:
            continue

        if contamination < 0 or contamination > 5:
            continue

        group = completeness_bin(completeness)

        if group is None:
            continue

        counts["quality_pass"] += 1

        accession = accession_of(row)

        biosample = (
            row.get("ncbi_biosample") or ""
        ).strip()

        if not accession:
            continue

        if accession in existing_accessions:
            continue

        if biosample and biosample in existing_biosamples:
            continue

        taxonomy = taxonomy_of(row)

        species = taxonomy.get("s", "")
        genus = taxonomy.get("g", "")
        family = taxonomy.get("f", "")

        if species and species in ref_taxa["s"]:
            level = "species"

        elif genus and genus in ref_taxa["g"]:
            level = "genus"

        elif family and family in ref_taxa["f"]:
            level = "family"

        else:
            continue

        counts["reference_comparable"] += 1
        eligible[group] += 1
        eligible_levels[(group, level)] += 1

        item = {
            "accession": accession,
            "biosample": biosample,
            "completeness": f"{completeness:.3f}",
            "contamination": f"{contamination:.3f}",
            "quality_bin": group,
            "reference_target_level": level,
            "gtdb_species": species,
            "gtdb_genus": genus,
            "gtdb_family": family,
            "ncbi_genome_category": category,
            "selection_seed": SEED
        }

        pool = reservoirs[group]

        # Bounded reservoir: do not store the full GTDB.
        if len(pool) < RESERVOIR_SIZE:
            pool.append(item)
        else:
            position = rng.randrange(eligible[group])

            if position < RESERVOIR_SIZE:
                pool[position] = item

# Create a small, taxonomically diversified shortlist.
shortlist = []

quotas = {
    "species": 10,
    "genus": 6,
    "family": 4
}

for group in (
    "50-70",
    "70-80",
    "80-90",
    "90-100"
):

    pool = sorted(
        reservoirs[group],
        key=lambda r: r["accession"]
    )

    rng.shuffle(pool)

    selected = []
    used_accessions = set()
    used_biosamples = set()
    genus_counts = Counter()

    def take(items, limit, max_per_genus=None):

        added = 0

        for item in items:

            if added >= limit:
                break

            if len(selected) >= TARGET_PER_BIN:
                break

            accession = item["accession"]
            biosample = item["biosample"]
            genus = item["gtdb_genus"]

            if accession in used_accessions:
                continue

            if biosample and biosample in used_biosamples:
                continue

            if (
                max_per_genus is not None
                and genus_counts[genus] >= max_per_genus
            ):
                continue

            selected.append(item)
            used_accessions.add(accession)

            if biosample:
                used_biosamples.add(biosample)

            genus_counts[genus] += 1
            added += 1

    for level, quota in quotas.items():

        subset = [
            r for r in pool
            if r["reference_target_level"] == level
        ]

        take(subset, quota, max_per_genus=2)

    take(pool, TARGET_PER_BIN, max_per_genus=2)
    take(pool, TARGET_PER_BIN)

    shortlist.extend(selected)

    print(
        "[MAG BIN]",
        group,
        "eligible=", eligible[group],
        "selected=", len(selected),
        "levels=",
        dict(Counter(
            r["reference_target_level"]
            for r in selected
        )),
        flush=True
    )

OUT.mkdir(parents=True, exist_ok=True)

write_tsv(
    OUT / "W06_E13_MAG_SHORTLIST.tsv",
    shortlist,
    FIELDS
)

summary = []

for group in (
    "50-70",
    "70-80",
    "80-90",
    "90-100"
):
    summary.append({
        "quality_bin": group,
        "eligible_all_levels": eligible[group],
        "eligible_species": eligible_levels[(group, "species")],
        "eligible_genus": eligible_levels[(group, "genus")],
        "eligible_family": eligible_levels[(group, "family")],
        "shortlisted": sum(
            r["quality_bin"] == group
            for r in shortlist
        )
    })

write_tsv(
    OUT / "W06_E13_MAG_POOL_COUNTS.tsv",
    summary,
    [
        "quality_bin",
        "eligible_all_levels",
        "eligible_species",
        "eligible_genus",
        "eligible_family",
        "shortlisted"
    ]
)

status = [
    "W06 E13 REAL MAG CANDIDATE SELECTION",
    "Source: GTDB r232 metadata",
    "Source category: derived from metagenome",
    "Completeness range: 50-100%",
    "Contamination: <=5%",
    "Reference: original A=2374",
    f"Selection seed: {SEED}",
    f"Shortlisted MAGs: {len(shortlist)}",
    "FASTA downloaded: NO",
    "Frozen model inference: NOT YET",
    "Sequence-level duplicate check: PENDING",
    "GTDB taxonomy is not independent ground truth"
]

(OUT / "W06_E13_STATUS.txt").write_text(
    "\n".join(status) + "\n",
    encoding="utf-8"
)

print()
print("Category counts:", categories.most_common(5))
print("Total metadata records:", counts["all"])
print("MAG category records:", counts["MAG_category"])
print("Quality-passing MAGs:", counts["quality_pass"])
print(
    "Reference-comparable candidates:",
    counts["reference_comparable"]
)

print()
print("E13 MAG shortlist:", len(shortlist))
print("FASTA downloads: 0")
print("Model inference: 0")
print("Output:", OUT)
print("W06 COMPLETE")
