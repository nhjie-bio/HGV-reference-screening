#!/usr/bin/env python3

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

MANIFEST = (
    BASE
    / "01_manifest"
    / "candidate_manifest_qcpass.tsv.gz"
)

OUTDIR = BASE / "01_manifest"
OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_OUT = (
    OUTDIR
    / "phase0_independence_design_summary.txt"
)

SPECIES_OUT = (
    OUTDIR
    / "species_independence_availability.tsv"
)

BIOSAMPLE_OUT = (
    OUTDIR
    / "biosample_duplication_audit.tsv"
)

PAIR_OUT = (
    OUTDIR
    / "gcf_gca_pairing_audit.tsv"
)


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


def valid_biosample(x):
    return bool(clean(x))


if not MANIFEST.exists():
    raise SystemExit(
        f"Missing manifest: {MANIFEST}"
    )


# ============================================================
# PASS 1
#
# Establish:
# - all QC-pass GCF BioSamples
# - all GCF -> paired GenBank accession relationships
# - accession uniqueness
# - BioSample multiplicity
# ============================================================

accession_counts = Counter()

biosample_accessions = defaultdict(set)
biosample_sources = defaultdict(set)

all_gcf_biosamples = set()

paired_gca_to_gcf = defaultdict(set)

total_rows = 0
gcf_rows = 0
gca_rows = 0

missing_biosample_gcf = 0
missing_biosample_gca = 0


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

        total_rows += 1

        accession = clean(
            row["assembly_accession"]
        )

        source = clean(
            row["source_database"]
        )

        biosample = clean(
            row["biosample"]
        )

        paired_gca = clean(
            row["paired_genbank_assembly"]
        )

        accession_counts[
            accession
        ] += 1

        if biosample:

            biosample_accessions[
                biosample
            ].add(
                accession
            )

            biosample_sources[
                biosample
            ].add(
                source
            )

        if source == "REFSEQ_GCF":

            gcf_rows += 1

            if biosample:
                all_gcf_biosamples.add(
                    biosample
                )
            else:
                missing_biosample_gcf += 1

            if paired_gca:

                paired_gca_to_gcf[
                    paired_gca
                ].add(
                    accession
                )

        elif source == "GENBANK_GCA":

            gca_rows += 1

            if not biosample:
                missing_biosample_gca += 1


duplicate_accessions = {
    a: n
    for a, n in accession_counts.items()
    if n > 1
}


# ============================================================
# BioSample audit
# ============================================================

with open(
    BIOSAMPLE_OUT,
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
            "biosample",
            "n_qcpass_assemblies",
            "sources",
            "assemblies",
        ]
    )

    for biosample in sorted(
        biosample_accessions
    ):

        accessions = sorted(
            biosample_accessions[
                biosample
            ]
        )

        if len(accessions) <= 1:
            continue

        writer.writerow(
            [
                biosample,
                len(accessions),
                ";".join(
                    sorted(
                        biosample_sources[
                            biosample
                        ]
                    )
                ),
                ";".join(
                    accessions
                ),
            ]
        )


duplicate_biosamples = sum(
    1
    for accessions
    in biosample_accessions.values()
    if len(accessions) > 1
)

cross_source_biosamples = sum(
    1
    for sources
    in biosample_sources.values()
    if (
        "REFSEQ_GCF" in sources
        and
        "GENBANK_GCA" in sources
    )
)


# ============================================================
# PASS 2
#
# Species-specific independent availability.
#
# Definitions:
#
# A GCF candidate:
#   - QC-pass
#   - valid BioSample
#
# GenBank-only external GCA:
#   - QC-pass GCA
#   - valid BioSample
#   - accession is NOT paired to a QC-pass GCF
#
# Strict external GCA:
#   - GenBank-only external GCA
#   - BioSample not present among ANY QC-pass GCF
#
# The last definition is deliberately conservative.
# If it gives enough B1/B2 candidates, we can use it and obtain
# a particularly clean external cohort.
# ============================================================

species_meta = {}

species_gcf_accessions = defaultdict(set)
species_gcf_biosamples = defaultdict(set)

species_gca_accessions = defaultdict(set)
species_gca_biosamples = defaultdict(set)

species_genbank_only_gca = defaultdict(set)
species_genbank_only_gca_biosamples = defaultdict(set)

species_strict_external_gca = defaultdict(set)
species_strict_external_biosamples = defaultdict(set)

paired_collision_rows = []


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

        accession = clean(
            row["assembly_accession"]
        )

        source = clean(
            row["source_database"]
        )

        biosample = clean(
            row["biosample"]
        )

        species = clean(
            row["gtdb_species_cluster"]
        )

        genus = clean(
            row["gtdb_genus"]
        )

        family = clean(
            row["gtdb_family"]
        )

        species_name = clean(
            row["gtdb_species"]
        )

        species_meta[
            species
        ] = {
            "gtdb_species_cluster":
                species,
            "gtdb_species":
                species_name,
            "gtdb_genus":
                genus,
            "gtdb_family":
                family,
        }

        if source == "REFSEQ_GCF":

            species_gcf_accessions[
                species
            ].add(
                accession
            )

            if biosample:

                species_gcf_biosamples[
                    species
                ].add(
                    biosample
                )

        elif source == "GENBANK_GCA":

            species_gca_accessions[
                species
            ].add(
                accession
            )

            if biosample:

                species_gca_biosamples[
                    species
                ].add(
                    biosample
                )

            paired_gcfs = (
                paired_gca_to_gcf.get(
                    accession,
                    set()
                )
            )

            has_refseq_twin = bool(
                paired_gcfs
            )

            if has_refseq_twin:

                paired_collision_rows.append(
                    [
                        accession,
                        biosample,
                        species,
                        ";".join(
                            sorted(
                                paired_gcfs
                            )
                        ),
                    ]
                )

            if (
                biosample
                and
                not has_refseq_twin
            ):

                species_genbank_only_gca[
                    species
                ].add(
                    accession
                )

                species_genbank_only_gca_biosamples[
                    species
                ].add(
                    biosample
                )

                if (
                    biosample
                    not in all_gcf_biosamples
                ):

                    species_strict_external_gca[
                        species
                    ].add(
                        accession
                    )

                    species_strict_external_biosamples[
                        species
                    ].add(
                        biosample
                    )


# ============================================================
# Pairing audit output
# ============================================================

with open(
    PAIR_OUT,
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
            "gca_accession",
            "biosample",
            "gtdb_species_cluster",
            "paired_qcpass_gcf_accession",
        ]
    )

    writer.writerows(
        paired_collision_rows
    )


# ============================================================
# Species availability after independence constraints
# ============================================================

species_rows = []


for species, meta in species_meta.items():

    gcf_n = len(
        species_gcf_accessions[
            species
        ]
    )

    gcf_bs_n = len(
        species_gcf_biosamples[
            species
        ]
    )

    gca_n = len(
        species_gca_accessions[
            species
        ]
    )

    gca_bs_n = len(
        species_gca_biosamples[
            species
        ]
    )

    genbank_only_n = len(
        species_genbank_only_gca[
            species
        ]
    )

    genbank_only_bs_n = len(
        species_genbank_only_gca_biosamples[
            species
        ]
    )

    strict_ext_n = len(
        species_strict_external_gca[
            species
        ]
    )

    strict_ext_bs_n = len(
        species_strict_external_biosamples[
            species
        ]
    )

    species_rows.append(
        {
            **meta,

            "qcpass_gcf_assemblies":
                gcf_n,

            "unique_gcf_biosamples":
                gcf_bs_n,

            "qcpass_gca_assemblies":
                gca_n,

            "unique_gca_biosamples":
                gca_bs_n,

            "genbank_only_gca":
                genbank_only_n,

            "genbank_only_gca_unique_biosamples":
                genbank_only_bs_n,

            "strict_external_gca":
                strict_ext_n,

            "strict_external_unique_biosamples":
                strict_ext_bs_n,

            "eligible_A12":
                int(
                    gcf_bs_n >= 12
                ),

            "eligible_A16":
                int(
                    gcf_bs_n >= 16
                ),

            "eligible_A20":
                int(
                    gcf_bs_n >= 20
                ),

            "eligible_external10":
                int(
                    strict_ext_bs_n >= 10
                ),

            "eligible_matched_B1":
                int(
                    gcf_bs_n >= 12
                    and
                    strict_ext_bs_n >= 10
                ),
        }
    )


species_rows.sort(
    key=lambda x: (
        -x[
            "unique_gcf_biosamples"
        ],
        -x[
            "strict_external_unique_biosamples"
        ],
        x[
            "gtdb_species_cluster"
        ],
    )
)


FIELDS = [
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",

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
]


with open(
    SPECIES_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(
        species_rows
    )


# ============================================================
# Taxonomic structural feasibility
# ============================================================

A12 = [
    r
    for r in species_rows
    if r["eligible_A12"] == 1
]

A16 = [
    r
    for r in species_rows
    if r["eligible_A16"] == 1
]

A20 = [
    r
    for r in species_rows
    if r["eligible_A20"] == 1
]

EXT10 = [
    r
    for r in species_rows
    if r["eligible_external10"] == 1
]

MATCHED = [
    r
    for r in species_rows
    if r["eligible_matched_B1"] == 1
]


genus_A12_species = defaultdict(set)

family_A12_genera = defaultdict(set)
family_A12_species = defaultdict(set)


for r in A12:

    genus_A12_species[
        r["gtdb_genus"]
    ].add(
        r["gtdb_species_cluster"]
    )

    family_A12_genera[
        r["gtdb_family"]
    ].add(
        r["gtdb_genus"]
    )

    family_A12_species[
        r["gtdb_family"]
    ].add(
        r["gtdb_species_cluster"]
    )


genera_with_2_species = sum(
    1
    for s
    in genus_A12_species.values()
    if len(s) >= 2
)

genera_with_3_species = sum(
    1
    for s
    in genus_A12_species.values()
    if len(s) >= 3
)

genera_with_4_species = sum(
    1
    for s
    in genus_A12_species.values()
    if len(s) >= 4
)


families_with_2_genera = sum(
    1
    for s
    in family_A12_genera.values()
    if len(s) >= 2
)

families_with_3_genera = sum(
    1
    for s
    in family_A12_genera.values()
    if len(s) >= 3
)

families_with_4_genera = sum(
    1
    for s
    in family_A12_genera.values()
    if len(s) >= 4
)


# ============================================================
# Potential seen-genus B2 feasibility
#
# A genus is potentially usable if:
# - it contains at least one A12 species
# - and at least one distinct external-rich species.
# ============================================================

genus_a_species = defaultdict(set)
genus_ext_species = defaultdict(set)


for r in A12:

    genus_a_species[
        r["gtdb_genus"]
    ].add(
        r["gtdb_species_cluster"]
    )


for r in EXT10:

    genus_ext_species[
        r["gtdb_genus"]
    ].add(
        r["gtdb_species_cluster"]
    )


seen_genus_b2_candidate_species = set()
seen_genus_b2_candidate_genera = set()


for genus in (
    set(genus_a_species)
    &
    set(genus_ext_species)
):

    for ext_species in (
        genus_ext_species[
            genus
        ]
    ):

        if any(
            a_species
            != ext_species
            for a_species
            in genus_a_species[
                genus
            ]
        ):

            seen_genus_b2_candidate_species.add(
                ext_species
            )

            seen_genus_b2_candidate_genera.add(
                genus
            )


# ============================================================
# Proposed V3 design gates
#
# Preferred design:
# A  = 200 species x 12 GCF = 2400 genomes
# B1 = 40 matched species x 10 GCA = 400
# B2 = 40 unseen species x 10 GCA = 400
#
# B2 later:
#   20 species from seen genera
#   20 species from unseen genera
# ============================================================

design_200x12 = (
    len(A12) >= 200
    and
    len(MATCHED) >= 40
    and
    len(EXT10) >= 80
    and
    genera_with_3_species >= 50
    and
    families_with_3_genera >= 12
    and
    len(
        seen_genus_b2_candidate_species
    ) >= 20
)


# ============================================================
# Summary
# ============================================================

lines = []

lines.append(
    "============================================================"
)
lines.append(
    "HGV-V3 PHASE 0C INDEPENDENCE + DESIGN AUDIT"
)
lines.append(
    "============================================================"
)
lines.append("")

lines.append(
    "QC-PASS INPUT"
)
lines.append(
    "-------------"
)

lines.append(
    f"QC-pass genomes: {total_rows:,}"
)

lines.append(
    f"QC-pass GCF: {gcf_rows:,}"
)

lines.append(
    f"QC-pass GCA: {gca_rows:,}"
)

lines.append("")

lines.append(
    "ACCESSION / BIOSAMPLE AUDIT"
)
lines.append(
    "---------------------------"
)

lines.append(
    f"Duplicate assembly accessions: "
    f"{len(duplicate_accessions):,}"
)

lines.append(
    f"BioSamples represented by >1 QC-pass assembly: "
    f"{duplicate_biosamples:,}"
)

lines.append(
    f"BioSamples represented in both GCF and GCA pools: "
    f"{cross_source_biosamples:,}"
)

lines.append(
    f"GCF rows with missing BioSample: "
    f"{missing_biosample_gcf:,}"
)

lines.append(
    f"GCA rows with missing BioSample: "
    f"{missing_biosample_gca:,}"
)

lines.append("")

lines.append(
    "GCF/GCA PAIRED-TWIN AUDIT"
)
lines.append(
    "-------------------------"
)

lines.append(
    f"Distinct GCA accessions listed as paired counterparts "
    f"of QC-pass GCFs: "
    f"{len(paired_gca_to_gcf):,}"
)

lines.append(
    f"QC-pass GCA rows colliding with a QC-pass GCF pair: "
    f"{len(paired_collision_rows):,}"
)

lines.append("")

lines.append(
    "SPECIES AVAILABILITY AFTER INDEPENDENCE FILTERS"
)
lines.append(
    "----------------------------------------------"
)

lines.append(
    f"Species with >=12 unique GCF BioSamples: "
    f"{len(A12):,}"
)

lines.append(
    f"Species with >=16 unique GCF BioSamples: "
    f"{len(A16):,}"
)

lines.append(
    f"Species with >=20 unique GCF BioSamples: "
    f"{len(A20):,}"
)

lines.append(
    f"Species with >=10 strict-external GCA BioSamples: "
    f"{len(EXT10):,}"
)

lines.append(
    f"Species supporting A + matched B1: "
    f"{len(MATCHED):,}"
)

lines.append("")

lines.append(
    "HIERARCHICAL TAXONOMIC CAPACITY"
)
lines.append(
    "-------------------------------"
)

lines.append(
    f"A12 genera: "
    f"{len(genus_A12_species):,}"
)

lines.append(
    f"A12 families: "
    f"{len(family_A12_genera):,}"
)

lines.append(
    f"Genera with >=2 A12 species: "
    f"{genera_with_2_species:,}"
)

lines.append(
    f"Genera with >=3 A12 species: "
    f"{genera_with_3_species:,}"
)

lines.append(
    f"Genera with >=4 A12 species: "
    f"{genera_with_4_species:,}"
)

lines.append(
    f"Families with >=2 A12 genera: "
    f"{families_with_2_genera:,}"
)

lines.append(
    f"Families with >=3 A12 genera: "
    f"{families_with_3_genera:,}"
)

lines.append(
    f"Families with >=4 A12 genera: "
    f"{families_with_4_genera:,}"
)

lines.append("")

lines.append(
    "B2 STRUCTURAL CAPACITY"
)
lines.append(
    "----------------------"
)

lines.append(
    f"Potential external-rich species usable as "
    f"seen-genus B2 candidates: "
    f"{len(seen_genus_b2_candidate_species):,}"
)

lines.append(
    f"Genera supporting potential seen-genus B2: "
    f"{len(seen_genus_b2_candidate_genera):,}"
)

lines.append("")

lines.append(
    "PREFERRED V3 DESIGN"
)
lines.append(
    "-------------------"
)

lines.append(
    "Dataset A: 200 species x 12 GCF genomes = 2,400"
)

lines.append(
    "Dataset B1: 40 A-matched species x 10 independent GCA = 400"
)

lines.append(
    "Dataset B2: 40 A-unseen species x 10 independent GCA = 400"
)

lines.append(
    "B2 target: 20 seen-genus species + 20 unseen-genus species"
)

lines.append("")

lines.append(
    "PREFERRED 200x12 DESIGN FEASIBILITY GATE: "
    +
    (
        "PASS"
        if design_200x12
        else "FAIL / NEEDS ADJUSTMENT"
    )
)

lines.append("")

lines.append(
    "IMPORTANT:"
)
lines.append(
    "No cohort has been frozen at this stage."
)
lines.append(
    "No genome FASTA has been downloaded."
)
lines.append(
    "Final A/B1/B2 selection will additionally enforce "
    "cross-cohort BioSample and paired-assembly zero overlap."
)

lines.append(
    "Sequence hash duplication will be checked after FASTA download."
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


print(summary)

print(
    f"Species availability: {SPECIES_OUT}"
)

print(
    f"BioSample audit: {BIOSAMPLE_OUT}"
)

print(
    f"GCF/GCA pairing audit: {PAIR_OUT}"
)
