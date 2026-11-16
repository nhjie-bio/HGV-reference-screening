#!/usr/bin/env python3

import csv
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

AUDIT_DIR = (
    BASE
    / "02_genomes"
    / "sequence_audit_round1"
)

SEQ_AUDIT = (
    AUDIT_DIR
    / "sequence_integrity_audit.tsv"
)

DUP_FILE = (
    AUDIT_DIR
    / "sequence_duplicate_groups.tsv"
)

UPDATED_MANIFEST = (
    BASE
    / "01_manifest"
    / "replacement_round1_20260916"
    / "updated_primary_manifest_round1.tsv"
)

FROZEN_RESERVE = (
    BASE
    / "01_manifest"
    / "frozen_pre_download_20260916"
    / "selected_cohorts_reserve_v2.tsv"
)

OUTDIR = (
    BASE
    / "02_genomes"
    / "sequence_audit_diagnosis_round1"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

FAIL_OUT = (
    OUTDIR
    / "sequence_qc_failures_detailed.tsv"
)

DUP_OUT = (
    OUTDIR
    / "duplicate_members_detailed.tsv"
)

TARGET_OUT = (
    OUTDIR
    / "round2_replacement_targets.tsv"
)

RESERVE_OUT = (
    OUTDIR
    / "round2_reserve_feasibility.tsv"
)

SUMMARY_OUT = (
    OUTDIR
    / "round2_diagnosis_summary.txt"
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


def as_float(x, default=None):

    try:
        return float(x)
    except Exception:
        return default


def as_int(x, default=None):

    try:
        return int(float(x))
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


for path in [
    SEQ_AUDIT,
    DUP_FILE,
    UPDATED_MANIFEST,
    FROZEN_RESERVE,
]:

    if not path.exists():

        raise SystemExit(
            f"Missing required file: {path}"
        )


seq_rows = read_tsv(
    SEQ_AUDIT
)

manifest_rows = read_tsv(
    UPDATED_MANIFEST
)

reserve_rows = read_tsv(
    FROZEN_RESERVE
)


seq_by_acc = {
    r["assembly_accession"]: r
    for r in seq_rows
}

manifest_by_acc = {
    r["assembly_accession"]: r
    for r in manifest_rows
}


# ============================================================
# 1. Sequence-QC failures
# ============================================================

fail_rows = [
    r
    for r in seq_rows
    if r["sequence_qc_pass"] == "0"
]


fail_reason_counts = Counter()

fail_combo_counts = Counter()


for row in fail_rows:

    reasons = [
        x
        for x in row[
            "sequence_qc_fail_reasons"
        ].split(";")
        if x
    ]

    fail_combo_counts[
        ";".join(reasons)
    ] += 1

    for reason in reasons:
        fail_reason_counts[
            reason
        ] += 1


FAIL_FIELDS = [
    "assembly_accession",
    "cohort",
    "cohort_subtype",

    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",

    "replacement_status",

    "sequence_qc_fail_reasons",

    "actual_genome_size",
    "actual_contig_count",
    "actual_n50_contigs",

    "actual_ambiguous_bases",
    "actual_ambiguous_fraction",
    "actual_N",
    "actual_N_fraction",

    "actual_nonN_ambiguous_bases",
    "actual_nonN_ambiguous_fraction",

    "metadata_ambiguous_bases",
    "metadata_ambiguous_fraction",

    "ambiguous_fraction_delta",
]


failure_detailed = []


for seq in fail_rows:

    acc = seq[
        "assembly_accession"
    ]

    meta = manifest_by_acc[
        acc
    ]


    actual_size = as_int(
        seq[
            "actual_genome_size"
        ],
        0,
    )

    actual_amb = as_int(
        seq[
            "actual_ambiguous_bases"
        ],
        0,
    )

    actual_n = as_int(
        seq[
            "actual_N"
        ],
        0,
    )


    non_n_amb = max(
        0,
        actual_amb - actual_n,
    )


    non_n_frac = (
        non_n_amb / actual_size
        if actual_size > 0
        else 0.0
    )


    actual_frac = as_float(
        seq[
            "actual_ambiguous_fraction"
        ],
        0.0,
    )

    metadata_frac = as_float(
        meta.get(
            "ambiguous_fraction"
        ),
        None,
    )


    if metadata_frac is None:

        frac_delta = ""

    else:

        frac_delta = (
            actual_frac
            -
            metadata_frac
        )


    failure_detailed.append(
        {
            "assembly_accession":
                acc,

            "cohort":
                seq["cohort"],

            "cohort_subtype":
                seq[
                    "cohort_subtype"
                ],

            "gtdb_species_cluster":
                seq[
                    "gtdb_species_cluster"
                ],

            "gtdb_species":
                seq[
                    "gtdb_species"
                ],

            "gtdb_genus":
                seq[
                    "gtdb_genus"
                ],

            "gtdb_family":
                seq[
                    "gtdb_family"
                ],

            "replacement_status":
                seq[
                    "replacement_status"
                ],

            "sequence_qc_fail_reasons":
                seq[
                    "sequence_qc_fail_reasons"
                ],

            "actual_genome_size":
                actual_size,

            "actual_contig_count":
                seq[
                    "actual_contig_count"
                ],

            "actual_n50_contigs":
                seq[
                    "actual_n50_contigs"
                ],

            "actual_ambiguous_bases":
                actual_amb,

            "actual_ambiguous_fraction":
                f"{actual_frac:.10f}",

            "actual_N":
                actual_n,

            "actual_N_fraction":
                seq[
                    "actual_N_fraction"
                ],

            "actual_nonN_ambiguous_bases":
                non_n_amb,

            "actual_nonN_ambiguous_fraction":
                f"{non_n_frac:.10f}",

            "metadata_ambiguous_bases":
                meta.get(
                    "ambiguous_bases",
                    "",
                ),

            "metadata_ambiguous_fraction":
                meta.get(
                    "ambiguous_fraction",
                    "",
                ),

            "ambiguous_fraction_delta":
                (
                    f"{frac_delta:.10f}"
                    if frac_delta != ""
                    else ""
                ),
        }
    )


with open(
    FAIL_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=FAIL_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        sorted(
            failure_detailed,
            key=lambda r: (
                r["cohort"],
                r["cohort_subtype"],
                r["gtdb_species_cluster"],
                r["assembly_accession"],
            ),
        )
    )


# ============================================================
# 2. Duplicate groups
# ============================================================

duplicate_groups = defaultdict(list)


for row in seq_rows:

    duplicate_groups[
        row[
            "canonical_contigset_sha256"
        ]
    ].append(
        row
    )


duplicate_groups = {
    h: members
    for h, members
    in duplicate_groups.items()
    if len(members) > 1
}


duplicate_member_rows = []

duplicate_target_accessions = set()

duplicate_keeper_accessions = set()

duplicate_cross_species_groups = 0

duplicate_cross_genus_groups = 0

duplicate_cross_cohort_groups = 0


for group_index, (
    seq_hash,
    members,
) in enumerate(
    sorted(
        duplicate_groups.items()
    ),
    start=1,
):

    species = {
        x[
            "gtdb_species_cluster"
        ]
        for x in members
    }

    genera = {
        x[
            "gtdb_genus"
        ]
        for x in members
    }

    cohorts = {
        x[
            "cohort"
        ]
        for x in members
    }


    if len(species) > 1:
        duplicate_cross_species_groups += 1

    if len(genera) > 1:
        duplicate_cross_genus_groups += 1

    if len(cohorts) > 1:
        duplicate_cross_cohort_groups += 1


    passing_members = [
        x
        for x in members
        if x[
            "sequence_qc_pass"
        ] == "1"
    ]


    # --------------------------------------------------------
    # Minimal-intervention deterministic rule:
    #
    # If at least one duplicate member passes sequence QC:
    # keep exactly one.
    #
    # Preference:
    # 1. original primary retained
    # 2. lexicographically smallest accession
    #
    # All other passing duplicate members become replacement
    # targets.
    #
    # QC-failing members are already replacement targets.
    # --------------------------------------------------------

    keeper = None


    if passing_members:

        passing_members.sort(
            key=lambda x: (
                0
                if x[
                    "replacement_status"
                ]
                ==
                "original_primary_retained"
                else 1,

                x[
                    "assembly_accession"
                ],
            )
        )

        keeper = passing_members[0]

        duplicate_keeper_accessions.add(
            keeper[
                "assembly_accession"
            ]
        )


        for member in passing_members[1:]:

            duplicate_target_accessions.add(
                member[
                    "assembly_accession"
                ]
            )


    for member in members:

        acc = member[
            "assembly_accession"
        ]

        if (
            member[
                "sequence_qc_pass"
            ]
            == "0"
        ):

            action = (
                "replace_sequence_qc_failure"
            )

        elif (
            keeper is not None
            and
            acc
            ==
            keeper[
                "assembly_accession"
            ]
        ):

            action = (
                "keep_duplicate_representative"
            )

        else:

            action = (
                "replace_exact_sequence_duplicate"
            )


        duplicate_member_rows.append(
            {
                "duplicate_group":
                    f"DUP{group_index:03d}",

                "canonical_contigset_sha256":
                    seq_hash,

                "group_size":
                    len(members),

                "assembly_accession":
                    acc,

                "cohort":
                    member[
                        "cohort"
                    ],

                "cohort_subtype":
                    member[
                        "cohort_subtype"
                    ],

                "gtdb_species_cluster":
                    member[
                        "gtdb_species_cluster"
                    ],

                "gtdb_species":
                    member[
                        "gtdb_species"
                    ],

                "gtdb_genus":
                    member[
                        "gtdb_genus"
                    ],

                "replacement_status":
                    member[
                        "replacement_status"
                    ],

                "sequence_qc_pass":
                    member[
                        "sequence_qc_pass"
                    ],

                "sequence_qc_fail_reasons":
                    member[
                        "sequence_qc_fail_reasons"
                    ],

                "action":
                    action,
            }
        )


DUP_FIELDS = [
    "duplicate_group",
    "canonical_contigset_sha256",
    "group_size",

    "assembly_accession",
    "cohort",
    "cohort_subtype",

    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",

    "replacement_status",

    "sequence_qc_pass",
    "sequence_qc_fail_reasons",

    "action",
]


with open(
    DUP_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=DUP_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        duplicate_member_rows
    )


# ============================================================
# 3. Union of round-2 targets
# ============================================================

qc_fail_accessions = {
    r[
        "assembly_accession"
    ]
    for r in fail_rows
}


all_target_accessions = (
    qc_fail_accessions
    |
    duplicate_target_accessions
)


target_reason_map = defaultdict(
    list
)


for acc in qc_fail_accessions:

    target_reason_map[
        acc
    ].append(
        "sequence_qc_failure"
    )


for acc in duplicate_target_accessions:

    target_reason_map[
        acc
    ].append(
        "exact_canonical_sequence_duplicate"
    )


targets = []


for acc in sorted(
    all_target_accessions
):

    seq = seq_by_acc[
        acc
    ]

    meta = manifest_by_acc[
        acc
    ]


    targets.append(
        {
            "assembly_accession":
                acc,

            "cohort":
                seq[
                    "cohort"
                ],

            "cohort_subtype":
                seq[
                    "cohort_subtype"
                ],

            "gtdb_species_cluster":
                seq[
                    "gtdb_species_cluster"
                ],

            "gtdb_species":
                seq[
                    "gtdb_species"
                ],

            "gtdb_genus":
                seq[
                    "gtdb_genus"
                ],

            "gtdb_family":
                seq[
                    "gtdb_family"
                ],

            "biosample":
                meta[
                    "biosample"
                ],

            "source_database":
                meta[
                    "source_database"
                ],

            "replacement_status":
                seq[
                    "replacement_status"
                ],

            "target_reasons":
                ";".join(
                    target_reason_map[
                        acc
                    ]
                ),

            "sequence_qc_fail_reasons":
                seq[
                    "sequence_qc_fail_reasons"
                ],

            "actual_ambiguous_fraction":
                seq[
                    "actual_ambiguous_fraction"
                ],

            "canonical_contigset_sha256":
                seq[
                    "canonical_contigset_sha256"
                ],
        }
    )


TARGET_FIELDS = list(
    targets[0].keys()
)


with open(
    TARGET_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=TARGET_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        targets
    )


# ============================================================
# 4. Frozen reserve feasibility
# ============================================================

current_primary_accessions = {
    r[
        "assembly_accession"
    ]
    for r in manifest_rows
}

current_primary_biosamples = {
    clean(
        r[
            "biosample"
        ]
    )
    for r in manifest_rows
    if clean(
        r[
            "biosample"
        ]
    )
}


reserve_lookup = defaultdict(
    list
)


for row in reserve_rows:

    key = (
        row[
            "cohort"
        ],
        row[
            "cohort_subtype"
        ],
        row[
            "gtdb_species_cluster"
        ],
    )

    reserve_lookup[
        key
    ].append(
        row
    )


targets_by_key = defaultdict(
    list
)


for target in targets:

    key = (
        target[
            "cohort"
        ],
        target[
            "cohort_subtype"
        ],
        target[
            "gtdb_species_cluster"
        ],
    )

    targets_by_key[
        key
    ].append(
        target
    )


reserve_feasibility_rows = []


for key, species_targets in sorted(
    targets_by_key.items()
):

    cohort, subtype, species = key


    candidates = []


    for row in reserve_lookup.get(
        key,
        [],
    ):

        acc = row[
            "assembly_accession"
        ]

        bs = clean(
            row[
                "biosample"
            ]
        )


        if not bs:
            continue

        if acc in current_primary_accessions:
            continue

        if bs in current_primary_biosamples:
            continue


        candidates.append(
            row
        )


    needed = len(
        species_targets
    )

    available = len(
        candidates
    )


    reserve_feasibility_rows.append(
        {
            "cohort":
                cohort,

            "cohort_subtype":
                subtype,

            "gtdb_species_cluster":
                species,

            "gtdb_species":
                species_targets[0][
                    "gtdb_species"
                ],

            "gtdb_genus":
                species_targets[0][
                    "gtdb_genus"
                ],

            "targets_needing_replacement":
                needed,

            "unused_frozen_reserve_candidates":
                available,

            "reserve_deficit":
                max(
                    0,
                    needed - available,
                ),

            "frozen_reserve_sufficient":
                int(
                    available >= needed
                ),

            "target_accessions":
                ";".join(
                    sorted(
                        x[
                            "assembly_accession"
                        ]
                        for x
                        in species_targets
                    )
                ),

            "candidate_reserve_accessions":
                ";".join(
                    sorted(
                        x[
                            "assembly_accession"
                        ]
                        for x
                        in candidates
                    )
                ),
        }
    )


RESERVE_FIELDS = list(
    reserve_feasibility_rows[0].keys()
)


with open(
    RESERVE_OUT,
    "w",
    encoding="utf-8",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=RESERVE_FIELDS,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        reserve_feasibility_rows
    )


deficit_species = [
    r
    for r in reserve_feasibility_rows
    if r[
        "reserve_deficit"
    ] > 0
]


total_reserve_deficit = sum(
    r[
        "reserve_deficit"
    ]
    for r in reserve_feasibility_rows
)


# ============================================================
# 5. Summary
# ============================================================

overlap_qc_duplicate = (
    qc_fail_accessions
    &
    duplicate_target_accessions
)


duplicate_member_count = sum(
    len(members)
    for members
    in duplicate_groups.values()
)


duplicate_redundancy = sum(
    len(members) - 1
    for members
    in duplicate_groups.values()
)


replacement_status_counts = Counter(
    manifest_by_acc[
        x[
            "assembly_accession"
        ]
    ][
        "replacement_status"
    ]
    for x in targets
)


target_cohort_counts = Counter(
    (
        x["cohort"],
        x["cohort_subtype"],
    )
    for x in targets
)


lines = []

lines.append(
    "============================================================"
)

lines.append(
    "HGV-V3 SEQUENCE GATE FAILURE DIAGNOSIS"
)

lines.append(
    "============================================================"
)

lines.append("")

lines.append(
    "SEQUENCE-QC FAILURES"
)

lines.append(
    "--------------------"
)

lines.append(
    f"Total sequence-QC failing genomes: "
    f"{len(fail_rows)}"
)

lines.append("")

lines.append(
    "Failure reasons:"
)

for reason, count in sorted(
    fail_reason_counts.items()
):

    lines.append(
        f"  {reason}: {count}"
    )


lines.append("")

lines.append(
    "Failure-reason combinations:"
)

for combo, count in sorted(
    fail_combo_counts.items()
):

    lines.append(
        f"  {combo}: {count}"
    )


lines.append("")

lines.append(
    "EXACT SEQUENCE DUPLICATES"
)

lines.append(
    "-------------------------"
)

lines.append(
    f"Canonical duplicate groups: "
    f"{len(duplicate_groups)}"
)

lines.append(
    f"Genome members involved: "
    f"{duplicate_member_count}"
)

lines.append(
    f"Redundant genomes beyond one/group: "
    f"{duplicate_redundancy}"
)

lines.append(
    f"Cross-cohort duplicate groups: "
    f"{duplicate_cross_cohort_groups}"
)

lines.append(
    f"Cross-species duplicate groups: "
    f"{duplicate_cross_species_groups}"
)

lines.append(
    f"Cross-genus duplicate groups: "
    f"{duplicate_cross_genus_groups}"
)

lines.append("")

lines.append(
    "ROUND-2 REPLACEMENT TARGETS"
)

lines.append(
    "---------------------------"
)

lines.append(
    f"QC-failure targets: "
    f"{len(qc_fail_accessions)}"
)

lines.append(
    f"Additional duplicate targets: "
    f"{len(duplicate_target_accessions)}"
)

lines.append(
    f"Overlap between QC and duplicate-target sets: "
    f"{len(overlap_qc_duplicate)}"
)

lines.append(
    f"Unique replacement targets: "
    f"{len(all_target_accessions)}"
)

lines.append("")

lines.append(
    "Targets by cohort:"
)

for key, count in sorted(
    target_cohort_counts.items()
):

    lines.append(
        f"  {key[0]} / {key[1]}: {count}"
    )


lines.append("")

lines.append(
    "Replacement provenance among targets:"
)

for key, count in sorted(
    replacement_status_counts.items()
):

    lines.append(
        f"  {key}: {count}"
    )


lines.append("")

lines.append(
    "FROZEN RESERVE FEASIBILITY"
)

lines.append(
    "--------------------------"
)

lines.append(
    f"Species/subtype groups requiring replacement: "
    f"{len(reserve_feasibility_rows)}"
)

lines.append(
    f"Groups with insufficient frozen reserve: "
    f"{len(deficit_species)}"
)

lines.append(
    f"Total additional reserve genomes required: "
    f"{total_reserve_deficit}"
)


if deficit_species:

    lines.append("")

    lines.append(
        "Controlled reserve extension required for:"
    )

    for row in deficit_species:

        lines.append(
            "  "
            +
            f"{row['cohort']} / "
            f"{row['cohort_subtype']} / "
            f"{row['gtdb_species_cluster']} / "
            f"{row['gtdb_species']}: "
            f"need={row['targets_needing_replacement']}, "
            f"frozen_available="
            f"{row['unused_frozen_reserve_candidates']}, "
            f"deficit={row['reserve_deficit']}"
        )


lines.append("")

lines.append(
    "INTERPRETATION"
)

lines.append(
    "--------------"
)

lines.append(
    "The 955 metadata consistency warnings are not "
    "automatically replacement targets."
)

lines.append(
    "Only hard sequence-QC failures and redundant exact "
    "sequence duplicates are included in the round-2 target set."
)

lines.append(
    "No cohort or taxonomic design is changed at this stage."
)

lines.append("")

lines.append(
    "NEXT:"
)

lines.append(
    "Review this diagnosis before any round-2 replacement selection."
)

lines.append(
    "Do not start feature extraction."
)

lines.append("")

lines.append(
    "============================================================"
)


summary = (
    "\n".join(
        lines
    )
    +
    "\n"
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

print(
    f"QC failures: {FAIL_OUT}"
)

print(
    f"Duplicate members: {DUP_OUT}"
)

print(
    f"Replacement targets: {TARGET_OUT}"
)

print(
    f"Reserve feasibility: {RESERVE_OUT}"
)

print(
    f"Summary: {SUMMARY_OUT}"
)
