#!/usr/bin/env python3

import csv
import gzip
from collections import defaultdict
from pathlib import Path


BASE = Path("HGV_V3_revision_20260915")

MANIFEST = (
    BASE / "01_manifest" /
    "candidate_manifest_qcpass.tsv.gz"
)

SPECIES_FILE = (
    BASE / "01_manifest" /
    "species_independence_availability.tsv"
)

OUT = BASE / "01_manifest"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FINAL COHORT DESIGN
# ============================================================

A_N_SPECIES = 200
A_GENOMES_PER_SPECIES = 12

ANCHOR_GENERA = 50
SPECIES_PER_ANCHOR_GENUS = 3

B1_N_SPECIES = 40
B1_GENOMES_PER_SPECIES = 10

B2_SEEN_N_SPECIES = 20
B2_UNSEEN_N_SPECIES = 20
B2_GENOMES_PER_SPECIES = 10

MIN_DEEP_FAMILIES = 12
MIN_GENERA_PER_DEEP_FAMILY = 3

A_RESERVE_PER_SPECIES = 2
EXT_RESERVE_PER_SPECIES = 2


# ============================================================
# Helpers
# ============================================================

def clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"", "na", "nan", "none", "null"}:
        return ""
    return x


def as_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def as_float(x, default=0.0):
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


def genome_quality_key(row):
    """
    Lower tuple = better genome.
    """
    return (
        -assembly_rank(row["assembly_level"]),
        -as_float(row["completeness"]),
        as_float(row["contamination"], 999),
        as_int(row["contig_count"]),
        -as_int(row["n50_contigs"]),
        as_float(row["ambiguous_fraction"], 999),
        row["assembly_accession"],
    )


# ============================================================
# Read species-level availability
# ============================================================

species = []

with open(
    SPECIES_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.DictReader(handle, delimiter="\t")

    for row in reader:
        for col in [
            "qcpass_gcf_assemblies",
            "unique_gcf_biosamples",
            "qcpass_gca_assemblies",
            "unique_gca_biosamples",
            "genbank_only_gca",
            "genbank_only_gca_unique_biosamples",
            "strict_external_gca",
            "strict_external_unique_biosamples",
            "eligible_A12",
            "eligible_A16",
            "eligible_A20",
            "eligible_external10",
            "eligible_matched_B1",
        ]:
            row[col] = as_int(row[col])

        species.append(row)


by_species = {
    r["gtdb_species_cluster"]: r
    for r in species
}

A12 = [
    r for r in species
    if r["eligible_A12"] == 1
]

EXT10 = [
    r for r in species
    if r["eligible_external10"] == 1
]

MATCHED = {
    r["gtdb_species_cluster"]
    for r in species
    if r["eligible_matched_B1"] == 1
}


a_by_genus = defaultdict(list)
ext_by_genus = defaultdict(list)

for r in A12:
    a_by_genus[r["gtdb_genus"]].append(r)

for r in EXT10:
    ext_by_genus[r["gtdb_genus"]].append(r)


# ============================================================
# Step 1
# Select 20 B2 unseen-genus species first.
#
# Prefer:
# - new families
# - genera with few/no A12 species
# - high external availability
# - distinct genera
# ============================================================

b2_unseen = []
blocked_unseen_genera = set()
used_unseen_families = set()

remaining = EXT10.copy()

while len(b2_unseen) < B2_UNSEEN_N_SPECIES:

    candidates = [
        r for r in remaining
        if r["gtdb_genus"] not in blocked_unseen_genera
    ]

    if not candidates:
        raise RuntimeError(
            "Cannot construct 20 unseen-genus B2 species."
        )

    candidates.sort(
        key=lambda r: (
            -int(r["gtdb_family"] not in used_unseen_families),
            len(a_by_genus[r["gtdb_genus"]]),
            -r["strict_external_unique_biosamples"],
            r["gtdb_species_cluster"],
        )
    )

    chosen = candidates[0]

    b2_unseen.append(chosen)

    blocked_unseen_genera.add(
        chosen["gtdb_genus"]
    )

    used_unseen_families.add(
        chosen["gtdb_family"]
    )

    remaining = [
        r for r in remaining
        if r["gtdb_genus"] != chosen["gtdb_genus"]
    ]


# ============================================================
# Step 2
# Identify genera capable of:
#   - providing >=3 A species
#   - withholding one different B2 external species
#
# These genera are candidates for B2 seen-genus.
# ============================================================

seen_genus_options = {}

for genus, a_rows in a_by_genus.items():

    if genus in blocked_unseen_genera:
        continue

    if len(a_rows) < 3:
        continue

    ext_rows = ext_by_genus.get(genus, [])

    options = []

    a_ids = {
        r["gtdb_species_cluster"]
        for r in a_rows
    }

    for ext in ext_rows:

        reserve_id = ext["gtdb_species_cluster"]

        remaining_a = [
            r for r in a_rows
            if r["gtdb_species_cluster"] != reserve_id
        ]

        if len(remaining_a) >= 3:

            options.append(
                (
                    int(reserve_id not in a_ids),
                    ext["strict_external_unique_biosamples"],
                    ext,
                )
            )

    if options:

        options.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2]["gtdb_species_cluster"],
            )
        )

        seen_genus_options[genus] = options[0][2]


# ============================================================
# Step 3
# Choose 20 B2 seen-genus species / genera.
# Maximize family diversity and robust A capacity.
# ============================================================

selected_anchor_genera = set()
b2_seen_by_genus = {}
used_seen_families = set()

while len(b2_seen_by_genus) < B2_SEEN_N_SPECIES:

    candidates = []

    for genus, b2_species in seen_genus_options.items():

        if genus in selected_anchor_genera:
            continue

        a_rows = [
            r for r in a_by_genus[genus]
            if r["gtdb_species_cluster"]
            != b2_species["gtdb_species_cluster"]
        ]

        matched_n = sum(
            r["gtdb_species_cluster"] in MATCHED
            for r in a_rows
        )

        candidates.append(
            (
                int(
                    b2_species["gtdb_family"]
                    not in used_seen_families
                ),
                matched_n,
                len(a_rows),
                b2_species[
                    "strict_external_unique_biosamples"
                ],
                genus,
                b2_species,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Cannot construct 20 seen-genus B2 species."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            -x[2],
            -x[3],
            x[4],
        )
    )

    _, _, _, _, genus, b2_species = candidates[0]

    selected_anchor_genera.add(genus)
    b2_seen_by_genus[genus] = b2_species
    used_seen_families.add(
        b2_species["gtdb_family"]
    )


# ============================================================
# Step 4
# Ensure >=12 families have >=3 anchor genera.
#
# Anchor-capable genus:
# >=3 available A species after any B2 reservation.
# ============================================================

anchor_candidates_by_family = defaultdict(list)

for genus, a_rows in a_by_genus.items():

    if genus in blocked_unseen_genera:
        continue

    reserved = b2_seen_by_genus.get(genus)

    available = [
        r for r in a_rows
        if (
            reserved is None
            or
            r["gtdb_species_cluster"]
            != reserved["gtdb_species_cluster"]
        )
    ]

    if len(available) < SPECIES_PER_ANCHOR_GENUS:
        continue

    family = available[0]["gtdb_family"]

    matched_n = sum(
        r["gtdb_species_cluster"] in MATCHED
        for r in available
    )

    anchor_candidates_by_family[family].append(
        (
            genus,
            matched_n,
            len(available),
        )
    )


selected_family_counts = defaultdict(int)

for genus in selected_anchor_genera:

    family = a_by_genus[genus][0]["gtdb_family"]
    selected_family_counts[family] += 1


eligible_deep_families = []

for family, genera in anchor_candidates_by_family.items():

    unique_genera = {
        x[0] for x in genera
    }

    if len(unique_genera) >= MIN_GENERA_PER_DEEP_FAMILY:

        current = selected_family_counts[family]

        fill_cost = max(
            0,
            MIN_GENERA_PER_DEEP_FAMILY - current,
        )

        eligible_deep_families.append(
            (
                fill_cost,
                -current,
                -len(unique_genera),
                family,
            )
        )


eligible_deep_families.sort()

deep_families = [
    x[3]
    for x in eligible_deep_families[
        :MIN_DEEP_FAMILIES
    ]
]

if len(deep_families) < MIN_DEEP_FAMILIES:
    raise RuntimeError(
        "Cannot identify 12 families with >=3 anchor-capable genera."
    )


for family in deep_families:

    while (
        selected_family_counts[family]
        < MIN_GENERA_PER_DEEP_FAMILY
    ):

        candidates = []

        for genus, matched_n, available_n \
                in anchor_candidates_by_family[family]:

            if genus in selected_anchor_genera:
                continue

            candidates.append(
                (
                    matched_n,
                    available_n,
                    genus,
                )
            )

        if not candidates:
            raise RuntimeError(
                f"Cannot fill deep family {family}"
            )

        candidates.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2],
            )
        )

        genus = candidates[0][2]

        selected_anchor_genera.add(genus)
        selected_family_counts[family] += 1


# ============================================================
# Step 5
# Fill anchor genera to exactly 50.
# ============================================================

while len(selected_anchor_genera) < ANCHOR_GENERA:

    current_families = {
        a_by_genus[g][0]["gtdb_family"]
        for g in selected_anchor_genera
    }

    candidates = []

    for family, genera in anchor_candidates_by_family.items():

        for genus, matched_n, available_n in genera:

            if genus in selected_anchor_genera:
                continue

            candidates.append(
                (
                    int(family not in current_families),
                    matched_n,
                    available_n,
                    genus,
                )
            )

    if not candidates:
        raise RuntimeError(
            "Cannot fill 50 anchor genera."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            -x[2],
            x[3],
        )
    )

    selected_anchor_genera.add(
        candidates[0][3]
    )


if len(selected_anchor_genera) != ANCHOR_GENERA:
    raise RuntimeError(
        "Anchor genus count is not exactly 50."
    )


# ============================================================
# Step 6
# Select 3 A species from each anchor genus.
#
# Matched-B1 species are prioritized.
# ============================================================

a_selected = {}
species_design = {}


for genus in sorted(selected_anchor_genera):

    reserved = b2_seen_by_genus.get(genus)

    candidates = [
        r for r in a_by_genus[genus]
        if (
            reserved is None
            or
            r["gtdb_species_cluster"]
            != reserved["gtdb_species_cluster"]
        )
    ]

    candidates.sort(
        key=lambda r: (
            -int(
                r["gtdb_species_cluster"]
                in MATCHED
            ),
            -r["unique_gcf_biosamples"],
            -r["strict_external_unique_biosamples"],
            r["gtdb_species_cluster"],
        )
    )

    chosen = candidates[
        :SPECIES_PER_ANCHOR_GENUS
    ]

    if len(chosen) != SPECIES_PER_ANCHOR_GENUS:
        raise RuntimeError(
            f"Anchor genus {genus} has insufficient species."
        )

    for r in chosen:

        sid = r["gtdb_species_cluster"]

        a_selected[sid] = r

        species_design[sid] = {
            "cohort_species_role": "A",
            "A_structure": "anchor_genus",
        }


# ============================================================
# Step 7
# Add 50 breadth species from 50 additional genera.
#
# Priority:
# 1. reach >=40 matched B1-capable A species
# 2. add new families
# 3. high GCF availability
# ============================================================

target_breadth = (
    A_N_SPECIES
    -
    ANCHOR_GENERA * SPECIES_PER_ANCHOR_GENUS
)

if target_breadth != 50:
    raise RuntimeError(
        "Expected breadth target = 50."
    )


selected_a_genera = {
    r["gtdb_genus"]
    for r in a_selected.values()
}

selected_a_families = {
    r["gtdb_family"]
    for r in a_selected.values()
}


for _ in range(target_breadth):

    matched_current = sum(
        sid in MATCHED
        for sid in a_selected
    )

    matched_needed = max(
        0,
        B1_N_SPECIES - matched_current,
    )

    candidates = []

    for r in A12:

        sid = r["gtdb_species_cluster"]
        genus = r["gtdb_genus"]

        if sid in a_selected:
            continue

        if genus in selected_a_genera:
            continue

        if genus in blocked_unseen_genera:
            continue

        if (
            genus in b2_seen_by_genus
            and
            sid
            ==
            b2_seen_by_genus[
                genus
            ]["gtdb_species_cluster"]
        ):
            continue

        candidates.append(
            (
                int(
                    matched_needed > 0
                    and
                    sid in MATCHED
                ),
                int(
                    r["gtdb_family"]
                    not in selected_a_families
                ),
                r["unique_gcf_biosamples"],
                r["strict_external_unique_biosamples"],
                sid,
                r,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Cannot complete A breadth species."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            -x[2],
            -x[3],
            x[4],
        )
    )

    r = candidates[0][5]
    sid = r["gtdb_species_cluster"]

    a_selected[sid] = r

    species_design[sid] = {
        "cohort_species_role": "A",
        "A_structure": "breadth_genus",
    }

    selected_a_genera.add(
        r["gtdb_genus"]
    )

    selected_a_families.add(
        r["gtdb_family"]
    )


if len(a_selected) != A_N_SPECIES:
    raise RuntimeError(
        f"A has {len(a_selected)} species instead of 200."
    )


# ============================================================
# Validate A structure
# ============================================================

a_genera = defaultdict(set)
a_families = defaultdict(set)

for sid, r in a_selected.items():

    a_genera[
        r["gtdb_genus"]
    ].add(sid)

    a_families[
        r["gtdb_family"]
    ].add(
        r["gtdb_genus"]
    )


a_anchor_genera_n = sum(
    len(sids) >= 3
    for sids in a_genera.values()
)

a_deep_families_n = sum(
    len(genera) >= 3
    for genera in a_families.values()
)


if a_anchor_genera_n < ANCHOR_GENERA:
    raise RuntimeError(
        "A does not retain 50 genera with >=3 species."
    )

if a_deep_families_n < MIN_DEEP_FAMILIES:
    raise RuntimeError(
        "A does not retain >=12 families with >=3 genera."
    )


# ============================================================
# Step 8
# Select 40 B1 species from A ∩ matched candidates.
# Maximize genus/family breadth.
# ============================================================

matched_in_A = [
    a_selected[sid]
    for sid in a_selected
    if sid in MATCHED
]

if len(matched_in_A) < B1_N_SPECIES:
    raise RuntimeError(
        f"Only {len(matched_in_A)} matched B1 candidates in A."
    )


b1_species = []
b1_used_genera = set()
b1_used_families = set()

remaining = matched_in_A.copy()

while len(b1_species) < B1_N_SPECIES:

    remaining.sort(
        key=lambda r: (
            -int(
                r["gtdb_family"]
                not in b1_used_families
            ),
            -int(
                r["gtdb_genus"]
                not in b1_used_genera
            ),
            -r["strict_external_unique_biosamples"],
            r["gtdb_species_cluster"],
        )
    )

    chosen = remaining.pop(0)

    b1_species.append(chosen)

    b1_used_genera.add(
        chosen["gtdb_genus"]
    )

    b1_used_families.add(
        chosen["gtdb_family"]
    )


# ============================================================
# B2 lists
# ============================================================

b2_seen = [
    b2_seen_by_genus[g]
    for g in sorted(b2_seen_by_genus)
]

if len(b2_seen) != 20:
    raise RuntimeError(
        "B2 seen-genus species != 20."
    )

if len(b2_unseen) != 20:
    raise RuntimeError(
        "B2 unseen-genus species != 20."
    )


A_IDS = set(a_selected)

B1_IDS = {
    r["gtdb_species_cluster"]
    for r in b1_species
}

B2_SEEN_IDS = {
    r["gtdb_species_cluster"]
    for r in b2_seen
}

B2_UNSEEN_IDS = {
    r["gtdb_species_cluster"]
    for r in b2_unseen
}

B2_IDS = (
    B2_SEEN_IDS
    |
    B2_UNSEEN_IDS
)


if A_IDS & B2_IDS:
    raise RuntimeError(
        "Species leakage: A overlaps B2."
    )


if B2_SEEN_IDS & B2_UNSEEN_IDS:
    raise RuntimeError(
        "B2 seen/unseen overlap."
    )


a_genus_set = {
    r["gtdb_genus"]
    for r in a_selected.values()
}

for r in b2_seen:

    if r["gtdb_genus"] not in a_genus_set:
        raise RuntimeError(
            "B2 seen-genus species has unseen genus."
        )


for r in b2_unseen:

    if r["gtdb_genus"] in a_genus_set:
        raise RuntimeError(
            "B2 unseen-genus species genus occurs in A."
        )


# ============================================================
# Read genome-level manifest.
#
# We reconstruct strict external GCA definition exactly:
# - BioSample exists
# - BioSample absent from ALL QC-pass GCF
# - GCA is not paired counterpart of QC-pass GCF
# ============================================================

all_rows = []

all_gcf_biosamples = set()
paired_gca_from_gcf = set()


with gzip.open(
    MANIFEST,
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

        all_rows.append(row)

        if row["source_database"] == "REFSEQ_GCF":

            bs = clean(row["biosample"])

            if bs:
                all_gcf_biosamples.add(bs)

            paired = clean(
                row["paired_genbank_assembly"]
            )

            if paired:
                paired_gca_from_gcf.add(
                    paired
                )


rows_by_species_gcf = defaultdict(list)
rows_by_species_extgca = defaultdict(list)


for row in all_rows:

    sid = row["gtdb_species_cluster"]

    bs = clean(row["biosample"])

    if row["source_database"] == "REFSEQ_GCF":

        if bs:
            rows_by_species_gcf[sid].append(row)

    elif row["source_database"] == "GENBANK_GCA":

        accession = row["assembly_accession"]

        if not bs:
            continue

        if bs in all_gcf_biosamples:
            continue

        if accession in paired_gca_from_gcf:
            continue

        rows_by_species_extgca[sid].append(row)


# ============================================================
# Select genomes within each species.
#
# First maximize BioProject diversity, then fill by assembly
# quality. BioSamples are globally unique across selected data.
# ============================================================

def choose_genomes(
    rows,
    n_primary,
    n_reserve,
    global_used_biosamples,
):

    # one best assembly per BioSample
    best_by_biosample = {}

    for row in rows:

        bs = clean(row["biosample"])

        if not bs:
            continue

        if bs in global_used_biosamples:
            continue

        old = best_by_biosample.get(bs)

        if (
            old is None
            or
            genome_quality_key(row)
            <
            genome_quality_key(old)
        ):
            best_by_biosample[bs] = row

    candidates = list(
        best_by_biosample.values()
    )

    candidates.sort(
        key=genome_quality_key
    )

    # one best assembly per BioProject first
    best_by_project = {}

    for row in candidates:

        project = clean(
            row["bioproject"]
        )

        if not project:
            project = (
                "__NO_PROJECT__"
                +
                row["assembly_accession"]
            )

        if project not in best_by_project:
            best_by_project[project] = row

    project_first = list(
        best_by_project.values()
    )

    project_first.sort(
        key=genome_quality_key
    )

    selected = []
    selected_accessions = set()

    for row in project_first:

        if len(selected) >= n_primary:
            break

        selected.append(row)

        selected_accessions.add(
            row["assembly_accession"]
        )

    if len(selected) < n_primary:

        for row in candidates:

            if (
                row["assembly_accession"]
                in selected_accessions
            ):
                continue

            selected.append(row)

            selected_accessions.add(
                row["assembly_accession"]
            )

            if len(selected) >= n_primary:
                break

    if len(selected) < n_primary:

        return None, None

    for row in selected:
        global_used_biosamples.add(
            clean(row["biosample"])
        )

    remaining = [
        row
        for row in candidates
        if (
            row["assembly_accession"]
            not in selected_accessions
            and
            clean(row["biosample"])
            not in global_used_biosamples
        )
    ]

    remaining.sort(
        key=genome_quality_key
    )

    reserves = remaining[
        :n_reserve
    ]

    for row in reserves:
        global_used_biosamples.add(
            clean(row["biosample"])
        )

    return selected, reserves


selected_genome_rows = []
reserve_genome_rows = []

global_used_biosamples = set()


def add_selected_rows(
    species_rows,
    row_map,
    cohort,
    subtype,
    n_primary,
    n_reserve,
):

    # scarce species first
    species_rows = sorted(
        species_rows,
        key=lambda r: (
            len(
                row_map[
                    r["gtdb_species_cluster"]
                ]
            ),
            r["gtdb_species_cluster"],
        )
    )

    for srow in species_rows:

        sid = srow["gtdb_species_cluster"]

        primary, reserve = choose_genomes(
            row_map[sid],
            n_primary,
            n_reserve,
            global_used_biosamples,
        )

        if primary is None:

            raise RuntimeError(
                f"Insufficient independent genomes for "
                f"{cohort} {sid}"
            )

        for row in primary:

            x = dict(row)

            x["cohort"] = cohort
            x["cohort_subtype"] = subtype
            x["selection_role"] = "primary"

            selected_genome_rows.append(x)

        for row in reserve:

            x = dict(row)

            x["cohort"] = cohort
            x["cohort_subtype"] = subtype
            x["selection_role"] = "reserve"

            reserve_genome_rows.append(x)


# Dataset A
add_selected_rows(
    list(a_selected.values()),
    rows_by_species_gcf,
    "A",
    "development",
    A_GENOMES_PER_SPECIES,
    A_RESERVE_PER_SPECIES,
)


# External B1
add_selected_rows(
    b1_species,
    rows_by_species_extgca,
    "B1",
    "matched_species_external",
    B1_GENOMES_PER_SPECIES,
    EXT_RESERVE_PER_SPECIES,
)


# External B2 seen genus
add_selected_rows(
    b2_seen,
    rows_by_species_extgca,
    "B2",
    "unseen_species_seen_genus",
    B2_GENOMES_PER_SPECIES,
    EXT_RESERVE_PER_SPECIES,
)


# External B2 unseen genus
add_selected_rows(
    b2_unseen,
    rows_by_species_extgca,
    "B2",
    "unseen_species_unseen_genus",
    B2_GENOMES_PER_SPECIES,
    EXT_RESERVE_PER_SPECIES,
)


# ============================================================
# Save species design
# ============================================================

SPECIES_DESIGN_OUT = (
    OUT / "cohort_species_design.tsv"
)

species_design_rows = []


for sid, r in a_selected.items():

    species_design_rows.append(
        {
            "cohort": "A",
            "cohort_subtype":
                species_design[sid]["A_structure"],
            "gtdb_species_cluster": sid,
            "gtdb_species": r["gtdb_species"],
            "gtdb_genus": r["gtdb_genus"],
            "gtdb_family": r["gtdb_family"],
            "unique_gcf_biosamples":
                r["unique_gcf_biosamples"],
            "strict_external_unique_biosamples":
                r["strict_external_unique_biosamples"],
            "is_B1_species":
                int(sid in B1_IDS),
        }
    )


for r in b2_seen:

    species_design_rows.append(
        {
            "cohort": "B2",
            "cohort_subtype":
                "unseen_species_seen_genus",
            "gtdb_species_cluster":
                r["gtdb_species_cluster"],
            "gtdb_species":
                r["gtdb_species"],
            "gtdb_genus":
                r["gtdb_genus"],
            "gtdb_family":
                r["gtdb_family"],
            "unique_gcf_biosamples":
                r["unique_gcf_biosamples"],
            "strict_external_unique_biosamples":
                r["strict_external_unique_biosamples"],
            "is_B1_species": 0,
        }
    )


for r in b2_unseen:

    species_design_rows.append(
        {
            "cohort": "B2",
            "cohort_subtype":
                "unseen_species_unseen_genus",
            "gtdb_species_cluster":
                r["gtdb_species_cluster"],
            "gtdb_species":
                r["gtdb_species"],
            "gtdb_genus":
                r["gtdb_genus"],
            "gtdb_family":
                r["gtdb_family"],
            "unique_gcf_biosamples":
                r["unique_gcf_biosamples"],
            "strict_external_unique_biosamples":
                r["strict_external_unique_biosamples"],
            "is_B1_species": 0,
        }
    )


with open(
    SPECIES_DESIGN_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    fields = [
        "cohort",
        "cohort_subtype",
        "gtdb_species_cluster",
        "gtdb_species",
        "gtdb_genus",
        "gtdb_family",
        "unique_gcf_biosamples",
        "strict_external_unique_biosamples",
        "is_B1_species",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(species_design_rows)


# ============================================================
# Save selected genome manifests
# ============================================================

GENOME_FIELDS = [
    "cohort",
    "cohort_subtype",
    "selection_role",
    "assembly_accession",
    "gtdb_accession",
    "source_database",
    "biosample",
    "bioproject",
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
    "assembly_level",
    "genome_size",
    "contig_count",
    "n50_contigs",
    "gc_percent",
    "ambiguous_fraction",
    "completeness",
    "contamination",
]


PRIMARY_OUT = (
    OUT / "selected_cohorts_primary.tsv"
)

RESERVE_OUT = (
    OUT / "selected_cohorts_reserve.tsv"
)


def write_genome_manifest(path, rows):

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=GENOME_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        rows = sorted(
            rows,
            key=lambda x: (
                x["cohort"],
                x["cohort_subtype"],
                x["gtdb_species_cluster"],
                x["assembly_accession"],
            )
        )

        writer.writerows(rows)


write_genome_manifest(
    PRIMARY_OUT,
    selected_genome_rows,
)

write_genome_manifest(
    RESERVE_OUT,
    reserve_genome_rows,
)


# ============================================================
# Accession lists
# ============================================================

def accession_list(filename, rows):

    with open(
        OUT / filename,
        "w",
        encoding="utf-8",
    ) as handle:

        for row in sorted(
            rows,
            key=lambda x: x["assembly_accession"],
        ):
            handle.write(
                row["assembly_accession"]
                + "\n"
            )


A_PRIMARY = [
    r for r in selected_genome_rows
    if r["cohort"] == "A"
]

B1_PRIMARY = [
    r for r in selected_genome_rows
    if r["cohort"] == "B1"
]

B2_SEEN_PRIMARY = [
    r for r in selected_genome_rows
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        == "unseen_species_seen_genus"
    )
]

B2_UNSEEN_PRIMARY = [
    r for r in selected_genome_rows
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        == "unseen_species_unseen_genus"
    )
]

B2_PRIMARY = (
    B2_SEEN_PRIMARY
    +
    B2_UNSEEN_PRIMARY
)


accession_list(
    "development_A_primary_accessions.txt",
    A_PRIMARY,
)

accession_list(
    "external_B1_primary_accessions.txt",
    B1_PRIMARY,
)

accession_list(
    "external_B2_seen_genus_primary_accessions.txt",
    B2_SEEN_PRIMARY,
)

accession_list(
    "external_B2_unseen_genus_primary_accessions.txt",
    B2_UNSEEN_PRIMARY,
)

accession_list(
    "external_B2_primary_accessions.txt",
    B2_PRIMARY,
)

accession_list(
    "all_primary_accessions.txt",
    selected_genome_rows,
)

accession_list(
    "all_reserve_accessions.txt",
    reserve_genome_rows,
)


# ============================================================
# Summary
# ============================================================

def taxa_stats(rows):

    return (
        len({
            r["gtdb_species_cluster"]
            for r in rows
        }),
        len({
            r["gtdb_genus"]
            for r in rows
        }),
        len({
            r["gtdb_family"]
            for r in rows
        }),
    )


a_sp, a_ge, a_fa = taxa_stats(A_PRIMARY)
b1_sp, b1_ge, b1_fa = taxa_stats(B1_PRIMARY)
b2_sp, b2_ge, b2_fa = taxa_stats(B2_PRIMARY)


summary = f"""
============================================================
HGV-V3 COHORT SELECTION SUMMARY
============================================================

DATASET A
---------
Primary genomes: {len(A_PRIMARY)}
Species: {a_sp}
Genera: {a_ge}
Families: {a_fa}
Anchor genera with >=3 selected species: {a_anchor_genera_n}
Families with >=3 selected genera: {a_deep_families_n}

Target:
200 species x 12 genomes = 2400

DATASET B1
----------
Primary genomes: {len(B1_PRIMARY)}
Species: {b1_sp}
Genera: {b1_ge}
Families: {b1_fa}

Target:
40 A-matched species x 10 genomes = 400

DATASET B2
----------
Primary genomes: {len(B2_PRIMARY)}
Species: {b2_sp}
Genera: {b2_ge}
Families: {b2_fa}

Seen-genus B2 genomes: {len(B2_SEEN_PRIMARY)}
Unseen-genus B2 genomes: {len(B2_UNSEEN_PRIMARY)}

Target:
20 seen-genus species x 10 = 200
20 unseen-genus species x 10 = 200

TOTAL PRIMARY GENOMES
---------------------
{len(selected_genome_rows)}

TOTAL RESERVE GENOMES
---------------------
{len(reserve_genome_rows)}

PRELIMINARY STRUCTURE CHECKS
----------------------------
A species = 200: {len(A_IDS) == 200}
A genomes = 2400: {len(A_PRIMARY) == 2400}
B1 species = 40: {len(B1_IDS) == 40}
B1 genomes = 400: {len(B1_PRIMARY) == 400}
B2 species = 40: {len(B2_IDS) == 40}
B2 genomes = 400: {len(B2_PRIMARY) == 400}

A/B2 species overlap = {len(A_IDS & B2_IDS)}

B2 seen genera all present in A:
{all(r["gtdb_genus"] in a_genus_set for r in b2_seen)}

B2 unseen genera absent from A:
{all(r["gtdb_genus"] not in a_genus_set for r in b2_unseen)}

IMPORTANT
---------
Cohort selection is NOT yet frozen.

Do NOT download genomes yet.

The next mandatory step is:
05_audit_selected_cohorts.py

It must confirm:
- accession overlap = 0 where prohibited
- BioSample overlap = 0
- GCF/GCA paired-twin overlap = 0
- A/B2 species overlap = 0
- B2 unseen-genus/A genus overlap = 0
- exact cohort sizes
- exact taxonomy structure

Only after that audit passes will accession lists be frozen
and genome FASTA download begin.

============================================================
""".strip() + "\n"


SUMMARY_OUT = (
    OUT / "cohort_selection_summary.txt"
)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(summary)


print(summary)

print("Outputs:")
print(SPECIES_DESIGN_OUT)
print(PRIMARY_OUT)
print(RESERVE_OUT)
print(SUMMARY_OUT)
