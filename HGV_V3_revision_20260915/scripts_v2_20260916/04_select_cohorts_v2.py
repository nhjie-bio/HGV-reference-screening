#!/usr/bin/env python3

import csv
import gzip
from collections import defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

MANIFEST = (
    BASE
    / "01_manifest"
    / "candidate_manifest_qcpass.tsv.gz"
)

SPECIES_FILE = (
    BASE
    / "01_manifest"
    / "species_independence_availability.tsv"
)

OUT = (
    BASE
    / "01_manifest"
    / "cohort_selection_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIXED V3 DESIGN
# ============================================================

A_SPECIES_N = 200
A_GENOMES_PER_SPECIES = 12

ANCHOR_GENERA_N = 50
ANCHOR_SPECIES_PER_GENUS = 3

BREADTH_GENERA_N = 50

MIN_A_FAMILIES = 20

MIN_MULTI_GENUS_FAMILIES = 12
MIN_GENERA_PER_MULTI_FAMILY = 2

B1_SPECIES_N = 40
B1_GENOMES_PER_SPECIES = 10

B2_SEEN_SPECIES_N = 20
B2_UNSEEN_SPECIES_N = 20
B2_GENOMES_PER_SPECIES = 10

A_RESERVE_PER_SPECIES = 2
EXT_RESERVE_PER_SPECIES = 2


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

    return (
        -assembly_rank(
            row["assembly_level"]
        ),
        -as_float(
            row["completeness"]
        ),
        as_float(
            row["contamination"],
            999,
        ),
        as_int(
            row["contig_count"]
        ),
        -as_int(
            row["n50_contigs"]
        ),
        as_float(
            row["ambiguous_fraction"],
            999,
        ),
        row["assembly_accession"],
    )


# ============================================================
# Read species availability
# ============================================================

species_rows = []

with open(
    SPECIES_FILE,
    "r",
    encoding="utf-8",
    newline="",
) as handle:

    reader = csv.DictReader(
        handle,
        delimiter="\t",
    )

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

            row[col] = as_int(
                row[col]
            )

        species_rows.append(
            row
        )


A12 = [
    r
    for r in species_rows
    if r["eligible_A12"] == 1
]

EXT10 = [
    r
    for r in species_rows
    if r["eligible_external10"] == 1
]

MATCHED_IDS = {
    r["gtdb_species_cluster"]
    for r in species_rows
    if r["eligible_matched_B1"] == 1
}


a_by_genus = defaultdict(list)
ext_by_genus = defaultdict(list)


for r in A12:

    a_by_genus[
        r["gtdb_genus"]
    ].append(r)


for r in EXT10:

    ext_by_genus[
        r["gtdb_genus"]
    ].append(r)


# ============================================================
# Identify genera that can support:
#
# A: 3 species
# B2 seen-genus: 1 different external-rich species
#
# ============================================================

seen_b2_options = {}


for genus, a_rows in a_by_genus.items():

    if len(a_rows) < 3:
        continue

    options = []

    for ext in ext_by_genus.get(
        genus,
        [],
    ):

        ext_sid = (
            ext["gtdb_species_cluster"]
        )

        remaining_A = [
            r
            for r in a_rows
            if (
                r["gtdb_species_cluster"]
                != ext_sid
            )
        ]

        if len(remaining_A) < 3:
            continue

        matched_remaining = sum(
            r["gtdb_species_cluster"]
            in MATCHED_IDS
            for r in remaining_A
        )

        options.append(
            (
                matched_remaining,
                ext[
                    "strict_external_unique_biosamples"
                ],
                ext,
            )
        )

    if options:

        options.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2][
                    "gtdb_species_cluster"
                ],
            )
        )

        seen_b2_options[
            genus
        ] = options[0][2]


if len(seen_b2_options) < B2_SEEN_SPECIES_N:

    raise RuntimeError(
        "Insufficient genera for B2 seen-genus design: "
        f"{len(seen_b2_options)}"
    )


# ============================================================
# STEP 1
# Select 20 anchor genera that explicitly support B2 seen-genus.
#
# Maximize family diversity first.
# ============================================================

anchor_genera = set()

reserved_b2_seen = {}

selected_anchor_families = set()


while (
    len(reserved_b2_seen)
    <
    B2_SEEN_SPECIES_N
):

    candidates = []

    for genus, reserved in (
        seen_b2_options.items()
    ):

        if genus in anchor_genera:
            continue

        remaining = [
            r
            for r in a_by_genus[genus]
            if (
                r["gtdb_species_cluster"]
                != reserved[
                    "gtdb_species_cluster"
                ]
            )
        ]

        matched_n = sum(
            r["gtdb_species_cluster"]
            in MATCHED_IDS
            for r in remaining
        )

        family = remaining[0][
            "gtdb_family"
        ]

        candidates.append(
            (
                int(
                    family
                    not in selected_anchor_families
                ),
                matched_n,
                len(remaining),
                reserved[
                    "strict_external_unique_biosamples"
                ],
                genus,
                reserved,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Cannot select 20 B2 seen-genus anchor genera."
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

    chosen = candidates[0]

    genus = chosen[4]
    reserved = chosen[5]

    anchor_genera.add(
        genus
    )

    reserved_b2_seen[
        genus
    ] = reserved

    selected_anchor_families.add(
        a_by_genus[
            genus
        ][0]["gtdb_family"]
    )


# ============================================================
# STEP 2
# Fill remaining anchor genera to 50.
#
# Requirements:
# >=3 A12 species.
#
# Prefer:
# - new family
# - matched B1 capacity
# - more candidate species
# ============================================================

anchor_capable_genera = []


for genus, rows in a_by_genus.items():

    reserved = (
        reserved_b2_seen.get(
            genus
        )
    )

    available = [
        r
        for r in rows
        if (
            reserved is None
            or
            r["gtdb_species_cluster"]
            != reserved[
                "gtdb_species_cluster"
            ]
        )
    ]

    if len(available) < 3:
        continue

    anchor_capable_genera.append(
        (
            genus,
            available,
        )
    )


while len(anchor_genera) < ANCHOR_GENERA_N:

    candidates = []

    for genus, available in (
        anchor_capable_genera
    ):

        if genus in anchor_genera:
            continue

        family = available[0][
            "gtdb_family"
        ]

        matched_n = sum(
            r["gtdb_species_cluster"]
            in MATCHED_IDS
            for r in available
        )

        candidates.append(
            (
                int(
                    family
                    not in selected_anchor_families
                ),
                matched_n,
                len(available),
                genus,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Cannot construct 50 anchor genera."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            -x[2],
            x[3],
        )
    )

    genus = candidates[0][3]

    anchor_genera.add(
        genus
    )

    selected_anchor_families.add(
        a_by_genus[
            genus
        ][0]["gtdb_family"]
    )


# ============================================================
# STEP 3
# Select 3 A species per anchor genus.
#
# Prioritize matched-B1 species.
# ============================================================

A_SELECTED = {}


for genus in sorted(
    anchor_genera
):

    reserved = (
        reserved_b2_seen.get(
            genus
        )
    )

    candidates = [
        r
        for r in a_by_genus[genus]
        if (
            reserved is None
            or
            r["gtdb_species_cluster"]
            != reserved[
                "gtdb_species_cluster"
            ]
        )
    ]

    candidates.sort(
        key=lambda r: (
            -int(
                r["gtdb_species_cluster"]
                in MATCHED_IDS
            ),
            -r[
                "unique_gcf_biosamples"
            ],
            -r[
                "strict_external_unique_biosamples"
            ],
            r[
                "gtdb_species_cluster"
            ],
        )
    )

    chosen = candidates[
        :ANCHOR_SPECIES_PER_GENUS
    ]

    if len(chosen) != 3:

        raise RuntimeError(
            f"Anchor genus {genus} "
            "cannot provide 3 species."
        )

    for r in chosen:

        sid = r[
            "gtdb_species_cluster"
        ]

        A_SELECTED[sid] = {
            **r,
            "A_structure":
                "anchor_genus",
        }


if len(A_SELECTED) != 150:

    raise RuntimeError(
        f"Expected 150 anchor species; "
        f"got {len(A_SELECTED)}"
    )


# ============================================================
# Current selected taxonomy
# ============================================================

def selected_genera():

    return {
        r["gtdb_genus"]
        for r in A_SELECTED.values()
    }


def selected_families():

    return {
        r["gtdb_family"]
        for r in A_SELECTED.values()
    }


def family_to_selected_genera():

    result = defaultdict(set)

    for r in A_SELECTED.values():

        result[
            r["gtdb_family"]
        ].add(
            r["gtdb_genus"]
        )

    return result


# ============================================================
# STEP 4
# Select 50 breadth species, one from each new genus.
#
# First satisfy:
# >=12 families with >=2 selected genera.
#
# Then:
# >=20 total families.
#
# Then ensure >=40 B1-matched A species.
#
# Finally maximize breadth.
# ============================================================

breadth_selected = 0


def candidate_breadth_rows():

    current_genera = (
        selected_genera()
    )

    rows = []

    for genus, genus_rows in (
        a_by_genus.items()
    ):

        if genus in current_genera:
            continue

        # Do not use a species reserved for B2 seen
        reserved = (
            reserved_b2_seen.get(
                genus
            )
        )

        candidates = [
            r
            for r in genus_rows
            if (
                reserved is None
                or
                r["gtdb_species_cluster"]
                != reserved[
                    "gtdb_species_cluster"
                ]
            )
        ]

        if not candidates:
            continue

        candidates.sort(
            key=lambda r: (
                -int(
                    r[
                        "gtdb_species_cluster"
                    ]
                    in MATCHED_IDS
                ),
                -r[
                    "unique_gcf_biosamples"
                ],
                r[
                    "gtdb_species_cluster"
                ],
            )
        )

        rows.append(
            candidates[0]
        )

    return rows


while (
    breadth_selected
    <
    BREADTH_GENERA_N
):

    candidates = (
        candidate_breadth_rows()
    )

    if not candidates:

        raise RuntimeError(
            "Unable to fill 50 breadth genera."
        )

    fam_genera = (
        family_to_selected_genera()
    )

    n_multi_families = sum(
        len(genera)
        >= MIN_GENERA_PER_MULTI_FAMILY
        for genera in fam_genera.values()
    )

    n_families = len(
        selected_families()
    )

    matched_in_A = sum(
        sid in MATCHED_IDS
        for sid in A_SELECTED
    )

    scored = []

    for r in candidates:

        family = r[
            "gtdb_family"
        ]

        current_family_genera = len(
            fam_genera.get(
                family,
                set(),
            )
        )

        # Primary structural needs
        helps_multi_family = (
            n_multi_families
            <
            MIN_MULTI_GENUS_FAMILIES
            and
            current_family_genera == 1
        )

        helps_family_breadth = (
            n_families
            <
            MIN_A_FAMILIES
            and
            family
            not in selected_families()
        )

        helps_B1 = (
            matched_in_A
            <
            B1_SPECIES_N
            and
            r[
                "gtdb_species_cluster"
            ]
            in MATCHED_IDS
        )

        scored.append(
            (
                int(
                    helps_multi_family
                ),
                int(
                    helps_family_breadth
                ),
                int(
                    helps_B1
                ),
                int(
                    family
                    not in selected_families()
                ),
                int(
                    r[
                        "gtdb_species_cluster"
                    ]
                    in MATCHED_IDS
                ),
                r[
                    "unique_gcf_biosamples"
                ],
                r,
            )
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            -x[2],
            -x[3],
            -x[4],
            -x[5],
            x[6][
                "gtdb_species_cluster"
            ],
        )
    )

    r = scored[0][6]

    sid = r[
        "gtdb_species_cluster"
    ]

    A_SELECTED[sid] = {
        **r,
        "A_structure":
            "breadth_genus",
    }

    breadth_selected += 1


# ============================================================
# Validate Dataset A structure
# ============================================================

if len(A_SELECTED) != A_SPECIES_N:

    raise RuntimeError(
        f"A has {len(A_SELECTED)} species, "
        "expected 200."
    )


A_GENERA = {
    r["gtdb_genus"]
    for r in A_SELECTED.values()
}

A_FAMILIES = {
    r["gtdb_family"]
    for r in A_SELECTED.values()
}


fam_genera = (
    family_to_selected_genera()
)


MULTI_FAMILIES = {
    family
    for family, genera
    in fam_genera.items()
    if len(genera) >= 2
}


ANCHOR_COUNT = sum(
    1
    for genus in A_GENERA
    if sum(
        r["gtdb_genus"]
        == genus
        for r in A_SELECTED.values()
    ) >= 3
)


MATCHED_IN_A = {
    sid
    for sid in A_SELECTED
    if sid in MATCHED_IDS
}


if len(A_GENERA) != 100:

    raise RuntimeError(
        f"A genera = {len(A_GENERA)}, "
        "expected exactly 100."
    )


if len(A_FAMILIES) < MIN_A_FAMILIES:

    raise RuntimeError(
        f"A families = {len(A_FAMILIES)}, "
        f"required >= {MIN_A_FAMILIES}."
    )


if (
    len(MULTI_FAMILIES)
    <
    MIN_MULTI_GENUS_FAMILIES
):

    raise RuntimeError(
        "Insufficient families with >=2 selected genera: "
        f"{len(MULTI_FAMILIES)}"
    )


if ANCHOR_COUNT < ANCHOR_GENERA_N:

    raise RuntimeError(
        f"Anchor genera = {ANCHOR_COUNT}"
    )


if len(MATCHED_IN_A) < B1_SPECIES_N:

    raise RuntimeError(
        "Insufficient B1-matched species in A: "
        f"{len(MATCHED_IN_A)}"
    )


# ============================================================
# STEP 5
# B2 seen-genus = the 20 reserved species.
# ============================================================

B2_SEEN = list(
    reserved_b2_seen.values()
)


B2_SEEN_IDS = {
    r["gtdb_species_cluster"]
    for r in B2_SEEN
}


if len(B2_SEEN_IDS) != 20:

    raise RuntimeError(
        "B2 seen-genus species != 20."
    )


for r in B2_SEEN:

    if r["gtdb_species_cluster"] in A_SELECTED:

        raise RuntimeError(
            "B2 seen species overlaps A."
        )

    if r["gtdb_genus"] not in A_GENERA:

        raise RuntimeError(
            "B2 seen genus absent from A."
        )


# ============================================================
# STEP 6
# Select B2 unseen-genus species.
#
# Genus must be completely absent from A.
# One species per genus.
# Maximize family diversity.
# ============================================================

B2_UNSEEN = []

used_b2_unseen_genera = set()
used_b2_unseen_families = set()


while (
    len(B2_UNSEEN)
    <
    B2_UNSEEN_SPECIES_N
):

    candidates = []

    for r in EXT10:

        sid = r[
            "gtdb_species_cluster"
        ]

        genus = r[
            "gtdb_genus"
        ]

        family = r[
            "gtdb_family"
        ]

        if sid in A_SELECTED:
            continue

        if sid in B2_SEEN_IDS:
            continue

        if genus in A_GENERA:
            continue

        if genus in used_b2_unseen_genera:
            continue

        candidates.append(
            (
                int(
                    family
                    not in used_b2_unseen_families
                ),
                r[
                    "strict_external_unique_biosamples"
                ],
                r,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Cannot construct 20 unseen-genus B2 species."
        )

    candidates.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            x[2][
                "gtdb_species_cluster"
            ],
        )
    )

    r = candidates[0][2]

    B2_UNSEEN.append(
        r
    )

    used_b2_unseen_genera.add(
        r["gtdb_genus"]
    )

    used_b2_unseen_families.add(
        r["gtdb_family"]
    )


B2_UNSEEN_IDS = {
    r["gtdb_species_cluster"]
    for r in B2_UNSEEN
}


# ============================================================
# STEP 7
# Select B1 species from matched species already present in A.
#
# Maximize genus/family breadth.
# ============================================================

B1 = []

used_B1_genera = set()
used_B1_families = set()


remaining = [
    A_SELECTED[sid]
    for sid in MATCHED_IN_A
]


while len(B1) < B1_SPECIES_N:

    if not remaining:

        raise RuntimeError(
            "Unable to select 40 B1 species."
        )

    remaining.sort(
        key=lambda r: (
            -int(
                r["gtdb_family"]
                not in used_B1_families
            ),
            -int(
                r["gtdb_genus"]
                not in used_B1_genera
            ),
            -r[
                "strict_external_unique_biosamples"
            ],
            r[
                "gtdb_species_cluster"
            ],
        )
    )

    r = remaining.pop(0)

    B1.append(r)

    used_B1_genera.add(
        r["gtdb_genus"]
    )

    used_B1_families.add(
        r["gtdb_family"]
    )


B1_IDS = {
    r["gtdb_species_cluster"]
    for r in B1
}


# ============================================================
# Species-level leakage checks
# ============================================================

A_IDS = set(
    A_SELECTED
)

B2_IDS = (
    B2_SEEN_IDS
    |
    B2_UNSEEN_IDS
)


if A_IDS & B2_IDS:

    raise RuntimeError(
        "A/B2 species overlap detected."
    )


if B2_SEEN_IDS & B2_UNSEEN_IDS:

    raise RuntimeError(
        "B2 seen/unseen species overlap detected."
    )


for r in B2_UNSEEN:

    if r["gtdb_genus"] in A_GENERA:

        raise RuntimeError(
            "B2 unseen genus occurs in A."
        )


# ============================================================
# Read genome-level manifest
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

        all_rows.append(
            row
        )

        if (
            row["source_database"]
            ==
            "REFSEQ_GCF"
        ):

            bs = clean(
                row["biosample"]
            )

            if bs:
                all_gcf_biosamples.add(
                    bs
                )

            paired = clean(
                row[
                    "paired_genbank_assembly"
                ]
            )

            if paired:
                paired_gca_from_gcf.add(
                    paired
                )


gcf_by_species = defaultdict(list)
extgca_by_species = defaultdict(list)


for row in all_rows:

    sid = row[
        "gtdb_species_cluster"
    ]

    bs = clean(
        row["biosample"]
    )

    if not bs:
        continue

    if (
        row["source_database"]
        ==
        "REFSEQ_GCF"
    ):

        gcf_by_species[
            sid
        ].append(
            row
        )

    elif (
        row["source_database"]
        ==
        "GENBANK_GCA"
    ):

        accession = row[
            "assembly_accession"
        ]

        if bs in all_gcf_biosamples:
            continue

        if accession in paired_gca_from_gcf:
            continue

        extgca_by_species[
            sid
        ].append(
            row
        )


# ============================================================
# Genome selection
#
# Primaries first.
# Reserves only after all primary genomes are fixed.
# ============================================================

selected_primary = []

primary_used_biosamples = set()


def choose_primary_species_set(
    species_list,
    source_map,
    cohort,
    subtype,
    n_per_species,
):

    global selected_primary
    global primary_used_biosamples

    species_list = sorted(
        species_list,
        key=lambda r: (
            len(
                source_map[
                    r["gtdb_species_cluster"]
                ]
            ),
            r[
                "gtdb_species_cluster"
            ],
        )
    )

    for srow in species_list:

        sid = srow[
            "gtdb_species_cluster"
        ]

        # Best assembly per BioSample
        best = {}

        for row in source_map[sid]:

            bs = clean(
                row["biosample"]
            )

            if not bs:
                continue

            if bs in primary_used_biosamples:
                continue

            old = best.get(bs)

            if (
                old is None
                or
                genome_quality_key(row)
                <
                genome_quality_key(old)
            ):

                best[bs] = row

        candidates = list(
            best.values()
        )

        candidates.sort(
            key=genome_quality_key
        )

        # First maximize BioProject diversity
        project_first = []
        used_projects = set()

        for row in candidates:

            project = clean(
                row["bioproject"]
            )

            project_key = (
                project
                if project
                else
                "__NONE__"
                + row[
                    "assembly_accession"
                ]
            )

            if project_key in used_projects:
                continue

            project_first.append(
                row
            )

            used_projects.add(
                project_key
            )

        chosen = []

        chosen_accessions = set()

        for row in project_first:

            if len(chosen) >= n_per_species:
                break

            chosen.append(
                row
            )

            chosen_accessions.add(
                row[
                    "assembly_accession"
                ]
            )

        if len(chosen) < n_per_species:

            for row in candidates:

                if (
                    row[
                        "assembly_accession"
                    ]
                    in chosen_accessions
                ):
                    continue

                chosen.append(
                    row
                )

                chosen_accessions.add(
                    row[
                        "assembly_accession"
                    ]
                )

                if len(chosen) >= n_per_species:
                    break

        if len(chosen) < n_per_species:

            raise RuntimeError(
                f"{cohort} {sid}: "
                f"needed {n_per_species}, "
                f"only selected {len(chosen)} "
                "after BioSample independence."
            )

        for row in chosen:

            x = dict(row)

            x["cohort"] = cohort
            x[
                "cohort_subtype"
            ] = subtype

            x[
                "selection_role"
            ] = "primary"

            selected_primary.append(
                x
            )

            primary_used_biosamples.add(
                clean(
                    row["biosample"]
                )
            )


# Dataset A
choose_primary_species_set(
    list(
        A_SELECTED.values()
    ),
    gcf_by_species,
    "A",
    "development",
    A_GENOMES_PER_SPECIES,
)


# B1
choose_primary_species_set(
    B1,
    extgca_by_species,
    "B1",
    "matched_species_external",
    B1_GENOMES_PER_SPECIES,
)


# B2 seen
choose_primary_species_set(
    B2_SEEN,
    extgca_by_species,
    "B2",
    "unseen_species_seen_genus",
    B2_GENOMES_PER_SPECIES,
)


# B2 unseen
choose_primary_species_set(
    B2_UNSEEN,
    extgca_by_species,
    "B2",
    "unseen_species_unseen_genus",
    B2_GENOMES_PER_SPECIES,
)


# ============================================================
# Select reserves after ALL primaries fixed
# ============================================================

selected_reserve = []

reserve_used_biosamples = set(
    primary_used_biosamples
)


def choose_reserves(
    species_list,
    source_map,
    cohort,
    subtype,
    n_per_species,
):

    for srow in species_list:

        sid = srow[
            "gtdb_species_cluster"
        ]

        candidates = []

        seen_bs = set()

        for row in source_map[sid]:

            bs = clean(
                row["biosample"]
            )

            if not bs:
                continue

            if bs in reserve_used_biosamples:
                continue

            if bs in seen_bs:
                continue

            seen_bs.add(bs)

            candidates.append(
                row
            )

        candidates.sort(
            key=genome_quality_key
        )

        chosen = candidates[
            :n_per_species
        ]

        for row in chosen:

            x = dict(row)

            x["cohort"] = cohort

            x[
                "cohort_subtype"
            ] = subtype

            x[
                "selection_role"
            ] = "reserve"

            selected_reserve.append(
                x
            )

            reserve_used_biosamples.add(
                clean(
                    row["biosample"]
                )
            )


choose_reserves(
    list(
        A_SELECTED.values()
    ),
    gcf_by_species,
    "A",
    "development",
    A_RESERVE_PER_SPECIES,
)


choose_reserves(
    B1,
    extgca_by_species,
    "B1",
    "matched_species_external",
    EXT_RESERVE_PER_SPECIES,
)


choose_reserves(
    B2_SEEN,
    extgca_by_species,
    "B2",
    "unseen_species_seen_genus",
    EXT_RESERVE_PER_SPECIES,
)


choose_reserves(
    B2_UNSEEN,
    extgca_by_species,
    "B2",
    "unseen_species_unseen_genus",
    EXT_RESERVE_PER_SPECIES,
)


# ============================================================
# Write species design
# ============================================================

species_design_rows = []


for sid, r in A_SELECTED.items():

    species_design_rows.append(
        {
            "cohort": "A",
            "cohort_subtype":
                r["A_structure"],
            "gtdb_species_cluster":
                sid,
            "gtdb_species":
                r["gtdb_species"],
            "gtdb_genus":
                r["gtdb_genus"],
            "gtdb_family":
                r["gtdb_family"],
            "is_B1_species":
                int(
                    sid in B1_IDS
                ),
            "is_B2_seen_species":
                0,
        }
    )


for r in B2_SEEN:

    species_design_rows.append(
        {
            "cohort": "B2",
            "cohort_subtype":
                "unseen_species_seen_genus",
            "gtdb_species_cluster":
                r[
                    "gtdb_species_cluster"
                ],
            "gtdb_species":
                r["gtdb_species"],
            "gtdb_genus":
                r["gtdb_genus"],
            "gtdb_family":
                r["gtdb_family"],
            "is_B1_species": 0,
            "is_B2_seen_species": 1,
        }
    )


for r in B2_UNSEEN:

    species_design_rows.append(
        {
            "cohort": "B2",
            "cohort_subtype":
                "unseen_species_unseen_genus",
            "gtdb_species_cluster":
                r[
                    "gtdb_species_cluster"
                ],
            "gtdb_species":
                r["gtdb_species"],
            "gtdb_genus":
                r["gtdb_genus"],
            "gtdb_family":
                r["gtdb_family"],
            "is_B1_species": 0,
            "is_B2_seen_species": 0,
        }
    )


SPECIES_DESIGN_OUT = (
    OUT
    / "cohort_species_design_v2.tsv"
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
        "is_B1_species",
        "is_B2_seen_species",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        species_design_rows
    )


# ============================================================
# Genome manifests
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
    OUT
    / "selected_cohorts_primary_v2.tsv"
)

RESERVE_OUT = (
    OUT
    / "selected_cohorts_reserve_v2.tsv"
)


def write_manifest(
    filename,
    rows,
):

    rows = sorted(
        rows,
        key=lambda r: (
            r["cohort"],
            r["cohort_subtype"],
            r["gtdb_species_cluster"],
            r["assembly_accession"],
        )
    )

    with open(
        filename,
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

        writer.writerows(
            rows
        )


write_manifest(
    PRIMARY_OUT,
    selected_primary,
)

write_manifest(
    RESERVE_OUT,
    selected_reserve,
)


# ============================================================
# Accession lists
# ============================================================

def write_accessions(
    filename,
    rows,
):

    accessions = sorted({
        r["assembly_accession"]
        for r in rows
    })

    with open(
        OUT / filename,
        "w",
        encoding="utf-8",
    ) as handle:

        for accession in accessions:

            handle.write(
                accession
                + "\n"
            )


A_PRIMARY = [
    r
    for r in selected_primary
    if r["cohort"] == "A"
]

B1_PRIMARY = [
    r
    for r in selected_primary
    if r["cohort"] == "B1"
]

B2_SEEN_PRIMARY = [
    r
    for r in selected_primary
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        ==
        "unseen_species_seen_genus"
    )
]

B2_UNSEEN_PRIMARY = [
    r
    for r in selected_primary
    if (
        r["cohort"] == "B2"
        and
        r["cohort_subtype"]
        ==
        "unseen_species_unseen_genus"
    )
]

B2_PRIMARY = (
    B2_SEEN_PRIMARY
    +
    B2_UNSEEN_PRIMARY
)


write_accessions(
    "development_A_primary_accessions.txt",
    A_PRIMARY,
)

write_accessions(
    "external_B1_primary_accessions.txt",
    B1_PRIMARY,
)

write_accessions(
    "external_B2_seen_genus_primary_accessions.txt",
    B2_SEEN_PRIMARY,
)

write_accessions(
    "external_B2_unseen_genus_primary_accessions.txt",
    B2_UNSEEN_PRIMARY,
)

write_accessions(
    "external_B2_primary_accessions.txt",
    B2_PRIMARY,
)

write_accessions(
    "all_primary_accessions.txt",
    selected_primary,
)

write_accessions(
    "all_reserve_accessions.txt",
    selected_reserve,
)


# ============================================================
# Final summary
# ============================================================

def tax_stats(rows):

    species = {
        r["gtdb_species_cluster"]
        for r in rows
    }

    genera = {
        r["gtdb_genus"]
        for r in rows
    }

    families = {
        r["gtdb_family"]
        for r in rows
    }

    return (
        len(species),
        len(genera),
        len(families),
    )


a_sp, a_ge, a_fa = tax_stats(
    A_PRIMARY
)

b1_sp, b1_ge, b1_fa = tax_stats(
    B1_PRIMARY
)

b2_sp, b2_ge, b2_fa = tax_stats(
    B2_PRIMARY
)


summary = f"""
============================================================
HGV-V3 COHORT SELECTION V2 SUMMARY
============================================================

DATASET A
---------
Primary genomes: {len(A_PRIMARY)}
Species: {a_sp}
Genera: {a_ge}
Families: {a_fa}

Anchor genera with >=3 selected species:
{ANCHOR_COUNT}

Families with >=2 selected genera:
{len(MULTI_FAMILIES)}

Matched-B1-capable species inside A:
{len(MATCHED_IN_A)}

Target:
200 species x 12 genomes = 2400
50 anchor genera x 3 species
50 breadth genera x 1 species

DATASET B1
----------
Primary genomes: {len(B1_PRIMARY)}
Species: {b1_sp}
Genera: {b1_ge}
Families: {b1_fa}

Target:
40 A-matched species x 10 independent GCA = 400

DATASET B2
----------
Primary genomes: {len(B2_PRIMARY)}
Species: {b2_sp}
Genera: {b2_ge}
Families: {b2_fa}

Seen-genus genomes:
{len(B2_SEEN_PRIMARY)}

Unseen-genus genomes:
{len(B2_UNSEEN_PRIMARY)}

Target:
20 unseen species / seen genus x 10 = 200
20 unseen species / unseen genus x 10 = 200

TOTAL PRIMARY
-------------
{len(selected_primary)}

TOTAL RESERVE
-------------
{len(selected_reserve)}

STRUCTURAL CHECKS
-----------------
A species == 200:
{len(A_IDS) == 200}

A genera == 100:
{len(A_GENERA) == 100}

A families >= {MIN_A_FAMILIES}:
{len(A_FAMILIES) >= MIN_A_FAMILIES}

Anchor genera >= 50:
{ANCHOR_COUNT >= 50}

Families with >=2 genera >= 12:
{len(MULTI_FAMILIES) >= 12}

A/B2 species overlap:
{len(A_IDS & B2_IDS)}

B2 seen genera all present in A:
{all(r["gtdb_genus"] in A_GENERA for r in B2_SEEN)}

B2 unseen genera all absent from A:
{all(r["gtdb_genus"] not in A_GENERA for r in B2_UNSEEN)}

PRIMARY SIZE CHECKS
-------------------
A == 2400:
{len(A_PRIMARY) == 2400}

B1 == 400:
{len(B1_PRIMARY) == 400}

B2 seen == 200:
{len(B2_SEEN_PRIMARY) == 200}

B2 unseen == 200:
{len(B2_UNSEEN_PRIMARY) == 200}

Total == 3200:
{len(selected_primary) == 3200}

IMPORTANT
---------
This selection is still provisional.

DO NOT DOWNLOAD GENOMES YET.

Next:
05_audit_selected_cohorts_v2.py

Only after the final leakage audit passes will
the cohort files be SHA256-frozen and used for download.

============================================================
""".strip() + "\n"


SUMMARY_OUT = (
    OUT
    / "cohort_selection_v2_summary.txt"
)


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        summary
    )


print(summary)

print()
print("Outputs:")
print(SPECIES_DESIGN_OUT)
print(PRIMARY_OUT)
print(RESERVE_OUT)
print(SUMMARY_OUT)
