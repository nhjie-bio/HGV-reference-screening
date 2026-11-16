#!/usr/bin/env python3

import csv
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

META = (
    BASE
    / "00_metadata"
    / "bac120_metadata_r232.tsv.gz"
)

QC_FAILED = (
    BASE
    / "00_metadata"
    / "auxillary_files"
    / "qc_failed_r232.tsv"
)

OUTDIR = BASE / "01_manifest"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HGV-V3 PRE-REGISTERED QC THRESHOLDS
# ============================================================

MIN_COMPLETENESS = 95.0
MAX_CONTAMINATION = 5.0

ALLOWED_ASSEMBLY_LEVELS = {
    "complete genome",
    "chromosome",
    "scaffold",
}

MAX_CONTIGS = 200
MIN_N50 = 20_000

MIN_GENOME_SIZE = 1_000_000
MAX_GENOME_SIZE = 10_000_000

MAX_AMBIGUOUS_FRACTION = 0.005

REQUIRED_GENOME_REPRESENTATION = "full"


# ============================================================
# Helpers
# ============================================================

def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def missing(value):
    value = clean(value).lower()
    return value in {
        "",
        "na",
        "nan",
        "none",
        "null",
    }


def to_float(value):
    if missing(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def to_int(value):
    value = to_float(value)

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def strip_gtdb_prefix(accession):
    accession = clean(accession)

    if accession.startswith("RS_"):
        return accession[3:]

    if accession.startswith("GB_"):
        return accession[3:]

    return accession


def source_database(gtdb_accession):
    accession = clean(gtdb_accession)

    if accession.startswith("RS_GCF_"):
        return "REFSEQ_GCF"

    if accession.startswith("GB_GCA_"):
        return "GENBANK_GCA"

    return "OTHER"


def taxonomy_rank(taxonomy, prefix):
    taxonomy = clean(taxonomy)

    for token in taxonomy.split(";"):
        token = token.strip()

        if token.startswith(prefix):
            return token

    return ""


def taxonomy_rank_valid(value, prefix):
    value = clean(value)

    return (
        value != ""
        and
        value != prefix
    )


# ============================================================
# Load GTDB QC-failed accessions for audit.
#
# This is an additional audit flag.
# The main R232 metadata should normally already represent
# genomes available in the release, but we explicitly test it.
# ============================================================

gtdb_qc_failed = set()

if QC_FAILED.exists():

    with open(
        QC_FAILED,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:

        reader = csv.reader(
            handle,
            delimiter="\t",
        )

        first = next(reader, None)

        if first:

            header_lower = [
                x.strip().lower()
                for x in first
            ]

            accession_index = None

            for candidate in [
                "accession",
                "genome",
                "genome_id",
            ]:
                if candidate in header_lower:
                    accession_index = (
                        header_lower.index(candidate)
                    )
                    break

            if accession_index is None:
                accession_index = 0

                value = clean(
                    first[accession_index]
                )

                if (
                    value.startswith("RS_")
                    or
                    value.startswith("GB_")
                    or
                    value.startswith("GCF_")
                    or
                    value.startswith("GCA_")
                ):
                    gtdb_qc_failed.add(
                        strip_gtdb_prefix(value)
                    )

            for row in reader:

                if len(row) <= accession_index:
                    continue

                value = clean(
                    row[accession_index]
                )

                if value:
                    gtdb_qc_failed.add(
                        strip_gtdb_prefix(value)
                    )


# ============================================================
# Output files
# ============================================================

ALL_OUT = (
    OUTDIR
    / "candidate_manifest_all_qc.tsv.gz"
)

PASS_OUT = (
    OUTDIR
    / "candidate_manifest_qcpass.tsv.gz"
)

ATTRITION_OUT = (
    OUTDIR
    / "qc_attrition.tsv"
)

REASONS_OUT = (
    OUTDIR
    / "qc_reason_counts.tsv"
)

SPECIES_OUT = (
    OUTDIR
    / "species_availability.tsv"
)

GENUS_OUT = (
    OUTDIR
    / "genus_availability.tsv"
)

FAMILY_OUT = (
    OUTDIR
    / "family_availability.tsv"
)

SUMMARY_OUT = (
    OUTDIR
    / "phase0_candidate_summary.txt"
)


# ============================================================
# Manifest columns
# ============================================================

OUT_FIELDS = [
    "gtdb_accession",
    "assembly_accession",
    "source_database",

    "paired_genbank_assembly",
    "biosample",
    "bioproject",

    "ncbi_organism_name",
    "ncbi_species_taxid",
    "ncbi_isolate",
    "ncbi_strain_identifiers",
    "ncbi_isolation_source",

    "gtdb_species_cluster",
    "gtdb_domain",
    "gtdb_phylum",
    "gtdb_class",
    "gtdb_order",
    "gtdb_family",
    "gtdb_genus",
    "gtdb_species",

    "gtdb_representative",

    "assembly_level",
    "assembly_type",
    "refseq_category",
    "genome_category",
    "genome_representation",

    "genome_size",
    "contig_count",
    "n50_contigs",
    "gc_percent",
    "ambiguous_bases",
    "ambiguous_fraction",

    "completeness",
    "contamination",
    "quality_source",

    "listed_in_gtdb_qc_failed",

    "qc_accession",
    "qc_taxonomy",
    "qc_quality",
    "qc_assembly_level",
    "qc_isolate_like",
    "qc_full_representation",
    "qc_contig_count",
    "qc_n50",
    "qc_genome_size",
    "qc_ambiguous",

    "qc_pass",
    "exclusion_reasons",
]


# ============================================================
# Counters
# ============================================================

total_rows = 0
pass_rows = 0

source_counts_all = Counter()
source_counts_pass = Counter()

reason_counts = Counter()

attrition = Counter()

species_counts = defaultdict(
    lambda: Counter()
)

species_info = {}

genus_counts = defaultdict(
    lambda: Counter()
)

family_counts = defaultdict(
    lambda: Counter()
)


# ============================================================
# Main processing
# ============================================================

if not META.exists():
    print(
        f"ERROR: missing metadata: {META}",
        file=sys.stderr,
    )
    sys.exit(1)


print(
    "============================================================"
)
print(
    "HGV-V3 Phase 0B: candidate manifest construction"
)
print(
    "============================================================"
)
print()
print(
    f"Input metadata: {META}"
)
print()
print(
    "QC thresholds:"
)
print(
    f"  completeness >= {MIN_COMPLETENESS}"
)
print(
    f"  contamination <= {MAX_CONTAMINATION}"
)
print(
    "  assembly level = Complete Genome / Chromosome / Scaffold"
)
print(
    "  isolate-like: ncbi_genome_category == none"
)
print(
    "  genome representation == full"
)
print(
    f"  contigs <= {MAX_CONTIGS}"
)
print(
    f"  N50 >= {MIN_N50}"
)
print(
    f"  genome size = {MIN_GENOME_SIZE}-{MAX_GENOME_SIZE}"
)
print(
    f"  ambiguous fraction <= {MAX_AMBIGUOUS_FRACTION}"
)
print()
print(
    "Quality source policy:"
)
print(
    "  CheckM2 primary; CheckM1 fallback only if CheckM2 missing."
)
print()


with gzip.open(
    META,
    "rt",
    encoding="utf-8",
    errors="replace",
    newline="",
) as source_handle, \
     gzip.open(
         ALL_OUT,
         "wt",
         encoding="utf-8",
         newline="",
     ) as all_handle, \
     gzip.open(
         PASS_OUT,
         "wt",
         encoding="utf-8",
         newline="",
     ) as pass_handle:

    reader = csv.DictReader(
        source_handle,
        delimiter="\t",
    )

    all_writer = csv.DictWriter(
        all_handle,
        fieldnames=OUT_FIELDS,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )

    pass_writer = csv.DictWriter(
        pass_handle,
        fieldnames=OUT_FIELDS,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )

    all_writer.writeheader()
    pass_writer.writeheader()

    for row in reader:

        total_rows += 1

        if total_rows % 100_000 == 0:
            print(
                f"Processed {total_rows:,} genomes..."
            )

        gtdb_accession = clean(
            row.get("accession")
        )

        assembly_accession = (
            strip_gtdb_prefix(
                gtdb_accession
            )
        )

        src = source_database(
            gtdb_accession
        )

        source_counts_all[src] += 1

        taxonomy = clean(
            row.get("gtdb_taxonomy")
        )

        domain = taxonomy_rank(
            taxonomy,
            "d__",
        )

        phylum = taxonomy_rank(
            taxonomy,
            "p__",
        )

        class_name = taxonomy_rank(
            taxonomy,
            "c__",
        )

        order = taxonomy_rank(
            taxonomy,
            "o__",
        )

        family = taxonomy_rank(
            taxonomy,
            "f__",
        )

        genus = taxonomy_rank(
            taxonomy,
            "g__",
        )

        species = taxonomy_rank(
            taxonomy,
            "s__",
        )

        species_cluster = (
            strip_gtdb_prefix(
                row.get(
                    "gtdb_genome_representative"
                )
            )
        )

        paired_gca = clean(
            row.get(
                "ncbi_genbank_assembly_accession"
            )
        )

        # ----------------------------------------------------
        # Quality values:
        # CheckM2 first, CheckM1 fallback.
        # ----------------------------------------------------

        checkm2_comp = to_float(
            row.get(
                "checkm2_completeness"
            )
        )

        checkm2_cont = to_float(
            row.get(
                "checkm2_contamination"
            )
        )

        checkm1_comp = to_float(
            row.get(
                "checkm_completeness"
            )
        )

        checkm1_cont = to_float(
            row.get(
                "checkm_contamination"
            )
        )

        if (
            checkm2_comp is not None
            and
            checkm2_cont is not None
        ):
            completeness = checkm2_comp
            contamination = checkm2_cont
            quality_source = "CheckM2"

        elif (
            checkm1_comp is not None
            and
            checkm1_cont is not None
        ):
            completeness = checkm1_comp
            contamination = checkm1_cont
            quality_source = "CheckM1_fallback"

        else:
            completeness = None
            contamination = None
            quality_source = "missing"

        genome_size = to_int(
            row.get("genome_size")
        )

        contig_count = to_int(
            row.get("contig_count")
        )

        n50 = to_int(
            row.get("n50_contigs")
        )

        gc_percent = to_float(
            row.get("gc_percentage")
        )

        ambiguous_bases = to_int(
            row.get("ambiguous_bases")
        )

        if (
            genome_size is not None
            and
            genome_size > 0
            and
            ambiguous_bases is not None
        ):
            ambiguous_fraction = (
                ambiguous_bases
                /
                genome_size
            )
        else:
            ambiguous_fraction = None

        assembly_level = clean(
            row.get(
                "ncbi_assembly_level"
            )
        )

        genome_category = clean(
            row.get(
                "ncbi_genome_category"
            )
        )

        genome_representation = clean(
            row.get(
                "ncbi_genome_representation"
            )
        )

        # ----------------------------------------------------
        # QC flags
        # ----------------------------------------------------

        qc_accession = (
            src in {
                "REFSEQ_GCF",
                "GENBANK_GCA",
            }
            and
            assembly_accession.startswith(
                ("GCF_", "GCA_")
            )
            and
            assembly_accession
            not in gtdb_qc_failed
        )

        qc_taxonomy = (
            taxonomy_rank_valid(
                domain,
                "d__",
            )
            and
            taxonomy_rank_valid(
                family,
                "f__",
            )
            and
            taxonomy_rank_valid(
                genus,
                "g__",
            )
            and
            taxonomy_rank_valid(
                species,
                "s__",
            )
            and
            species_cluster != ""
        )

        qc_quality = (
            completeness is not None
            and
            contamination is not None
            and
            completeness
            >= MIN_COMPLETENESS
            and
            contamination
            <= MAX_CONTAMINATION
        )

        qc_assembly_level = (
            assembly_level.lower()
            in ALLOWED_ASSEMBLY_LEVELS
        )

        # GTDB description:
        # ncbi_genome_category indicates whether genome is
        # single-cell, metagenome, or environmental-sample derived.
        #
        # "none" therefore forms our strict isolate-like main pool.
        qc_isolate_like = (
            genome_category.lower()
            == "none"
        )

        qc_full_representation = (
            genome_representation.lower()
            ==
            REQUIRED_GENOME_REPRESENTATION
        )

        qc_contig_count = (
            contig_count is not None
            and
            contig_count <= MAX_CONTIGS
        )

        qc_n50 = (
            n50 is not None
            and
            n50 >= MIN_N50
        )

        qc_genome_size = (
            genome_size is not None
            and
            MIN_GENOME_SIZE
            <= genome_size
            <= MAX_GENOME_SIZE
        )

        qc_ambiguous = (
            ambiguous_fraction
            is not None
            and
            ambiguous_fraction
            <= MAX_AMBIGUOUS_FRACTION
        )

        flags = {
            "accession": qc_accession,
            "taxonomy": qc_taxonomy,
            "quality": qc_quality,
            "assembly_level":
                qc_assembly_level,
            "isolate_like":
                qc_isolate_like,
            "full_representation":
                qc_full_representation,
            "contig_count":
                qc_contig_count,
            "n50":
                qc_n50,
            "genome_size":
                qc_genome_size,
            "ambiguous":
                qc_ambiguous,
        }

        exclusion_reasons = [
            name
            for name, passed
            in flags.items()
            if not passed
        ]

        qc_pass = (
            len(exclusion_reasons)
            == 0
        )

        for reason in exclusion_reasons:
            reason_counts[reason] += 1

        # ----------------------------------------------------
        # Cumulative attrition.
        # ----------------------------------------------------

        cumulative = True

        attrition["00_total"] += 1

        for key, label in [
            (
                "accession",
                "01_accession"
            ),
            (
                "taxonomy",
                "02_taxonomy"
            ),
            (
                "quality",
                "03_quality"
            ),
            (
                "assembly_level",
                "04_assembly_level"
            ),
            (
                "isolate_like",
                "05_isolate_like"
            ),
            (
                "full_representation",
                "06_full_representation"
            ),
            (
                "contig_count",
                "07_contig_count"
            ),
            (
                "n50",
                "08_n50"
            ),
            (
                "genome_size",
                "09_genome_size"
            ),
            (
                "ambiguous",
                "10_ambiguous"
            ),
        ]:

            cumulative = (
                cumulative
                and
                flags[key]
            )

            if cumulative:
                attrition[label] += 1

        # ----------------------------------------------------
        # Output manifest row.
        # ----------------------------------------------------

        out = {
            "gtdb_accession":
                gtdb_accession,

            "assembly_accession":
                assembly_accession,

            "source_database":
                src,

            "paired_genbank_assembly":
                paired_gca,

            "biosample":
                clean(
                    row.get(
                        "ncbi_biosample"
                    )
                ),

            "bioproject":
                clean(
                    row.get(
                        "ncbi_bioproject"
                    )
                ),

            "ncbi_organism_name":
                clean(
                    row.get(
                        "ncbi_organism_name"
                    )
                ),

            "ncbi_species_taxid":
                clean(
                    row.get(
                        "ncbi_species_taxid"
                    )
                ),

            "ncbi_isolate":
                clean(
                    row.get(
                        "ncbi_isolate"
                    )
                ),

            "ncbi_strain_identifiers":
                clean(
                    row.get(
                        "ncbi_strain_identifiers"
                    )
                ),

            "ncbi_isolation_source":
                clean(
                    row.get(
                        "ncbi_isolation_source"
                    )
                ),

            "gtdb_species_cluster":
                species_cluster,

            "gtdb_domain":
                domain,

            "gtdb_phylum":
                phylum,

            "gtdb_class":
                class_name,

            "gtdb_order":
                order,

            "gtdb_family":
                family,

            "gtdb_genus":
                genus,

            "gtdb_species":
                species,

            "gtdb_representative":
                clean(
                    row.get(
                        "gtdb_representative"
                    )
                ),

            "assembly_level":
                assembly_level,

            "assembly_type":
                clean(
                    row.get(
                        "ncbi_assembly_type"
                    )
                ),

            "refseq_category":
                clean(
                    row.get(
                        "ncbi_refseq_category"
                    )
                ),

            "genome_category":
                genome_category,

            "genome_representation":
                genome_representation,

            "genome_size":
                genome_size
                if genome_size is not None
                else "",

            "contig_count":
                contig_count
                if contig_count is not None
                else "",

            "n50_contigs":
                n50
                if n50 is not None
                else "",

            "gc_percent":
                gc_percent
                if gc_percent is not None
                else "",

            "ambiguous_bases":
                ambiguous_bases
                if ambiguous_bases
                is not None
                else "",

            "ambiguous_fraction":
                (
                    f"{ambiguous_fraction:.8f}"
                    if
                    ambiguous_fraction
                    is not None
                    else ""
                ),

            "completeness":
                completeness
                if completeness
                is not None
                else "",

            "contamination":
                contamination
                if contamination
                is not None
                else "",

            "quality_source":
                quality_source,

            "listed_in_gtdb_qc_failed":
                int(
                    assembly_accession
                    in gtdb_qc_failed
                ),

            "qc_accession":
                int(qc_accession),

            "qc_taxonomy":
                int(qc_taxonomy),

            "qc_quality":
                int(qc_quality),

            "qc_assembly_level":
                int(qc_assembly_level),

            "qc_isolate_like":
                int(qc_isolate_like),

            "qc_full_representation":
                int(qc_full_representation),

            "qc_contig_count":
                int(qc_contig_count),

            "qc_n50":
                int(qc_n50),

            "qc_genome_size":
                int(qc_genome_size),

            "qc_ambiguous":
                int(qc_ambiguous),

            "qc_pass":
                int(qc_pass),

            "exclusion_reasons":
                ";".join(
                    exclusion_reasons
                ),
        }

        all_writer.writerow(out)

        if qc_pass:

            pass_rows += 1

            source_counts_pass[src] += 1

            pass_writer.writerow(out)

            cluster = species_cluster

            species_counts[
                cluster
            ][src] += 1

            species_counts[
                cluster
            ]["TOTAL"] += 1

            species_info[
                cluster
            ] = {
                "gtdb_species_cluster":
                    cluster,

                "gtdb_family":
                    family,

                "gtdb_genus":
                    genus,

                "gtdb_species":
                    species,
            }

            genus_counts[
                genus
            ][src] += 1

            genus_counts[
                genus
            ]["TOTAL"] += 1

            family_counts[
                family
            ][src] += 1

            family_counts[
                family
            ]["TOTAL"] += 1


# ============================================================
# Write attrition table
# ============================================================

ATTRITION_LABELS = [
    (
        "00_total",
        "All GTDB bacterial metadata rows"
    ),
    (
        "01_accession",
        "Valid GCF/GCA accession and not QC-failed"
    ),
    (
        "02_taxonomy",
        "Usable GTDB species/genus/family taxonomy"
    ),
    (
        "03_quality",
        "Completeness >=95 and contamination <=5"
    ),
    (
        "04_assembly_level",
        "Complete/Chromosome/Scaffold"
    ),
    (
        "05_isolate_like",
        "Strict isolate-like genome category"
    ),
    (
        "06_full_representation",
        "NCBI genome representation = full"
    ),
    (
        "07_contig_count",
        "Contig count <=200"
    ),
    (
        "08_n50",
        "Contig N50 >=20 kb"
    ),
    (
        "09_genome_size",
        "Genome size 1-10 Mb"
    ),
    (
        "10_ambiguous",
        "Ambiguous bases <=0.5%"
    ),
]


with open(
    ATTRITION_OUT,
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
            "stage",
            "description",
            "remaining_genomes",
            "fraction_of_total",
        ]
    )

    for stage, description in ATTRITION_LABELS:

        count = attrition[stage]

        fraction = (
            count / total_rows
            if total_rows
            else 0
        )

        writer.writerow(
            [
                stage,
                description,
                count,
                f"{fraction:.6f}",
            ]
        )


# ============================================================
# Write independent exclusion reason counts
# ============================================================

with open(
    REASONS_OUT,
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
            "reason",
            "genome_count",
            "fraction_of_total",
        ]
    )

    for reason, count in sorted(
        reason_counts.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        writer.writerow(
            [
                reason,
                count,
                f"{count / total_rows:.6f}",
            ]
        )


# ============================================================
# Write species availability
# ============================================================

species_rows = []

for cluster, counts in species_counts.items():

    info = species_info[cluster]

    gcf = counts["REFSEQ_GCF"]
    gca = counts["GENBANK_GCA"]
    total = counts["TOTAL"]

    species_rows.append(
        {
            **info,
            "qcpass_gcf": gcf,
            "qcpass_gca": gca,
            "qcpass_total": total,
            "eligible_A_ge12_gcf":
                int(gcf >= 12),
            "eligible_A_ge18_gcf":
                int(gcf >= 18),
            "eligible_B1_ge10_gca":
                int(gca >= 10),
            "eligible_external_ge10_gca":
                int(gca >= 10),
        }
    )


species_rows.sort(
    key=lambda x: (
        -x["qcpass_gcf"],
        -x["qcpass_gca"],
        x["gtdb_species_cluster"],
    )
)


with open(
    SPECIES_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    fields = [
        "gtdb_species_cluster",
        "gtdb_family",
        "gtdb_genus",
        "gtdb_species",
        "qcpass_gcf",
        "qcpass_gca",
        "qcpass_total",
        "eligible_A_ge12_gcf",
        "eligible_A_ge18_gcf",
        "eligible_B1_ge10_gca",
        "eligible_external_ge10_gca",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(
        species_rows
    )


# ============================================================
# Genus availability
# ============================================================

with open(
    GENUS_OUT,
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
            "gtdb_genus",
            "qcpass_gcf",
            "qcpass_gca",
            "qcpass_total",
            "qcpass_species_clusters",
        ]
    )

    genus_species = defaultdict(set)

    for row in species_rows:
        genus_species[
            row["gtdb_genus"]
        ].add(
            row["gtdb_species_cluster"]
        )

    for genus in sorted(
        genus_counts
    ):
        counts = genus_counts[genus]

        writer.writerow(
            [
                genus,
                counts["REFSEQ_GCF"],
                counts["GENBANK_GCA"],
                counts["TOTAL"],
                len(
                    genus_species[genus]
                ),
            ]
        )


# ============================================================
# Family availability
# ============================================================

with open(
    FAMILY_OUT,
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
            "gtdb_family",
            "qcpass_gcf",
            "qcpass_gca",
            "qcpass_total",
            "qcpass_genera",
            "qcpass_species_clusters",
        ]
    )

    family_genera = defaultdict(set)
    family_species = defaultdict(set)

    for row in species_rows:

        family_genera[
            row["gtdb_family"]
        ].add(
            row["gtdb_genus"]
        )

        family_species[
            row["gtdb_family"]
        ].add(
            row["gtdb_species_cluster"]
        )

    for family in sorted(
        family_counts
    ):

        counts = family_counts[family]

        writer.writerow(
            [
                family,
                counts["REFSEQ_GCF"],
                counts["GENBANK_GCA"],
                counts["TOTAL"],
                len(
                    family_genera[
                        family
                    ]
                ),
                len(
                    family_species[
                        family
                    ]
                ),
            ]
        )


# ============================================================
# Feasibility analysis
# ============================================================

A12 = [
    row
    for row in species_rows
    if row["qcpass_gcf"] >= 12
]

A18 = [
    row
    for row in species_rows
    if row["qcpass_gcf"] >= 18
]

B1 = [
    row
    for row in species_rows
    if (
        row["qcpass_gcf"] >= 12
        and
        row["qcpass_gca"] >= 10
    )
]

GCA10 = [
    row
    for row in species_rows
    if row["qcpass_gca"] >= 10
]


a12_genera = {
    row["gtdb_genus"]
    for row in A12
}

a12_families = {
    row["gtdb_family"]
    for row in A12
}

genus_to_a_species = defaultdict(set)

family_to_a_genera = defaultdict(set)

for row in A12:

    genus_to_a_species[
        row["gtdb_genus"]
    ].add(
        row["gtdb_species_cluster"]
    )

    family_to_a_genera[
        row["gtdb_family"]
    ].add(
        row["gtdb_genus"]
    )


genera_ge3_species = sum(
    1
    for species_set
    in genus_to_a_species.values()
    if len(species_set) >= 3
)

families_ge2_genera = sum(
    1
    for genus_set
    in family_to_a_genera.values()
    if len(genus_set) >= 2
)

families_ge3_genera = sum(
    1
    for genus_set
    in family_to_a_genera.values()
    if len(genus_set) >= 3
)


max_A_with_cap20 = sum(
    min(
        row["qcpass_gcf"],
        20,
    )
    for row in A12
)

max_B1_with_cap10 = sum(
    min(
        row["qcpass_gca"],
        10,
    )
    for row in B1
)


# ============================================================
# Summary
# ============================================================

summary = []

summary.append(
    "============================================================"
)
summary.append(
    "HGV-V3 PHASE 0B CANDIDATE POOL SUMMARY"
)
summary.append(
    "============================================================"
)
summary.append("")

summary.append(
    f"Total GTDB metadata rows: {total_rows:,}"
)

summary.append(
    f"Final strict-QC pass genomes: {pass_rows:,}"
)

summary.append(
    f"Final QC pass fraction: {pass_rows / total_rows:.4%}"
)

summary.append("")

summary.append(
    "SOURCE DATABASE COUNTS"
)
summary.append(
    "----------------------"
)

for src in sorted(
    source_counts_all
):

    summary.append(
        f"{src}: "
        f"all={source_counts_all[src]:,}; "
        f"QCpass={source_counts_pass[src]:,}"
    )


summary.append("")
summary.append(
    "TAXONOMIC AVAILABILITY AFTER STRICT QC"
)
summary.append(
    "--------------------------------------"
)

summary.append(
    f"Species clusters with >=1 QC-pass genome: "
    f"{len(species_rows):,}"
)

summary.append(
    f"Species eligible for Dataset A (>=12 GCF): "
    f"{len(A12):,}"
)

summary.append(
    f"Species with >=18 GCF: "
    f"{len(A18):,}"
)

summary.append(
    f"Genera represented among A-eligible species: "
    f"{len(a12_genera):,}"
)

summary.append(
    f"Families represented among A-eligible species: "
    f"{len(a12_families):,}"
)

summary.append(
    f"Genera containing >=3 A-eligible species: "
    f"{genera_ge3_species:,}"
)

summary.append(
    f"Families containing >=2 A-eligible genera: "
    f"{families_ge2_genera:,}"
)

summary.append(
    f"Families containing >=3 A-eligible genera: "
    f"{families_ge3_genera:,}"
)

summary.append("")
summary.append(
    "EXTERNAL COHORT AVAILABILITY"
)
summary.append(
    "----------------------------"
)

summary.append(
    f"Species supporting A + matched B1 "
    f"(>=12 GCF and >=10 GCA): "
    f"{len(B1):,}"
)

summary.append(
    f"Species with >=10 QC-pass GCA genomes: "
    f"{len(GCA10):,}"
)

summary.append("")
summary.append(
    "CAPACITY CHECK"
)
summary.append(
    "--------------"
)

summary.append(
    f"Maximum A genomes using <=20/species "
    f"from A-eligible species: "
    f"{max_A_with_cap20:,}"
)

summary.append(
    f"Maximum B1 genomes using <=10/species "
    f"from matched eligible species: "
    f"{max_B1_with_cap10:,}"
)

summary.append("")

basic_gate = (
    len(A12) >= 120
    and
    len(a12_genera) >= 30
    and
    len(a12_families) >= 12
    and
    max_A_with_cap20 >= 2400
    and
    len(B1) >= 40
    and
    max_B1_with_cap10 >= 400
)

summary.append(
    "V3 BASIC DATASET FEASIBILITY GATE: "
    +
    (
        "PASS"
        if basic_gate
        else "FAIL / REQUIRES DESIGN ADJUSTMENT"
    )
)

summary.append("")
summary.append(
    "NOTE:"
)
summary.append(
    "No Dataset A/B1/B2 cohort has been selected yet."
)
summary.append(
    "No genome FASTA has been downloaded."
)
summary.append(
    "This stage only constructs and audits the strict-QC candidate pool."
)
summary.append(
    "B1/B2 accession, BioSample, paired-assembly and sequence-hash "
    "independence will be enforced during cohort selection/finalization."
)

summary.append("")
summary.append(
    "============================================================"
)


summary_text = "\n".join(summary) + "\n"


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        summary_text
    )


print()
print(summary_text)

print(
    f"All-QC manifest: {ALL_OUT}"
)

print(
    f"QC-pass manifest: {PASS_OUT}"
)

print(
    f"Attrition table: {ATTRITION_OUT}"
)

print(
    f"Reason counts: {REASONS_OUT}"
)

print(
    f"Species availability: {SPECIES_OUT}"
)

print(
    f"Summary: {SUMMARY_OUT}"
)
