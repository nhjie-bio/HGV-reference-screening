#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# HGV-V3
# G1-G4 frozen split generation
#
# Input:
#   final_master_manifest_3154.tsv
#   canonical_k6_rows.tsv
#
# Development cohort only:
#   A = 2374 genomes
#
# Protocols:
#   G1 random genome        70/15/15
#   G2 species holdout      70/15/15
#   G3 genus holdout        70/15/15
#   G4 family holdout       70/15/15
#
# G1:
#   seed 20260915
#
# G2-G4:
#   split seeds:
#       20260915
#       20260916
#       20260917
#
# Existing outputs are NEVER overwritten.
# ============================================================


BASE = Path("HGV_V3_revision_20260915")

MANIFEST = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916"
    / "final_master_manifest_3154.tsv"
)

FEATURE_ROWS = (
    BASE
    / "03_features/canonical_k6_full3154_v1"
    / "canonical_k6_rows.tsv"
)

PREFLIGHT_REPORT = (
    BASE
    / "04_splits/preflight_capacity_v1_20260916"
    / "SPLIT_PREFLIGHT_REPORT.txt"
)

OUTDIR = (
    BASE
    / "04_splits/g1_g4_v1_20260916"
)

ROLE_ORDER = ["train", "val", "test"]

RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

EXPECTED_A = 2374
EXPECTED_EXTERNAL = 780

G1_SEED = 20260915

TAXONOMY_SEEDS = [
    20260915,
    20260916,
    20260917,
]

PROTOCOLS = {
    "G2": {
        "name": "species_holdout",
        "group_field": "gtdb_species_cluster",
    },
    "G3": {
        "name": "genus_holdout",
        "group_field": "gtdb_genus",
    },
    "G4": {
        "name": "family_holdout",
        "group_field": "gtdb_family",
    },
}

N_GREEDY_ATTEMPTS = 2000


# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

def die(msg):
    print()
    print("SPLIT_GENERATION=FAIL")
    print(msg)
    sys.exit(2)


def clean(x):
    if x is None:
        return ""
    return str(x).strip()


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def read_tsv(path):
    if not path.is_file():
        die(f"Missing required file: {path}")

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        die(f"No TSV header: {path}")

    return fields, rows


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
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for r in rows:
            writer.writerow(
                {k: r.get(k, "") for k in fields}
            )


def comb2(n):
    if n < 2:
        return 0
    return n * (n - 1) // 2


def target_counts(n):
    """
    Largest-remainder allocation for exact total n.
    For 2374:
      train = 1662
      val   = 356
      test  = 356
    """

    raw = {
        role: n * RATIOS[role]
        for role in ROLE_ORDER
    }

    out = {
        role: int(math.floor(raw[role]))
        for role in ROLE_ORDER
    }

    remaining = n - sum(out.values())

    ranked = sorted(
        ROLE_ORDER,
        key=lambda r: (
            -(raw[r] - math.floor(raw[r])),
            ROLE_ORDER.index(r),
        ),
    )

    for role in ranked[:remaining]:
        out[role] += 1

    assert sum(out.values()) == n

    return out


def detect_optional_field(fields, exact_names, substrings=()):
    lower_map = {
        x.lower(): x
        for x in fields
    }

    for name in exact_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in fields:
        lc = col.lower()

        if all(
            token.lower() in lc
            for token in substrings
        ):
            return col

    return None


# ------------------------------------------------------------
# Load frozen inputs
# ------------------------------------------------------------

print("=" * 78)
print("HGV-V3 G1-G4 SPLIT GENERATION")
print("=" * 78)

manifest_fields, manifest_rows = read_tsv(MANIFEST)
feature_fields, feature_rows = read_tsv(FEATURE_ROWS)

required_manifest = {
    "assembly_accession_final",
    "final_cohort",
    "fasta_path",
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
}

missing = required_manifest - set(manifest_fields)

if missing:
    die(
        "Manifest missing required columns: "
        + ", ".join(sorted(missing))
    )

if len(manifest_rows) != 3154:
    die(
        f"Expected 3154 final manifest rows; "
        f"observed {len(manifest_rows)}"
    )

if "assembly_accession" not in feature_fields:
    die(
        "canonical_k6_rows.tsv missing assembly_accession"
    )

if "row_index" not in feature_fields:
    die(
        "canonical_k6_rows.tsv missing row_index"
    )

feature_index = {}

for r in feature_rows:
    acc = clean(r["assembly_accession"])

    if acc in feature_index:
        die(f"Duplicate feature accession: {acc}")

    feature_index[acc] = int(r["row_index"])

if len(feature_index) != 3154:
    die(
        f"Expected 3154 feature identities; "
        f"observed {len(feature_index)}"
    )

A_rows = [
    r
    for r in manifest_rows
    if clean(r["final_cohort"]) == "A"
]

external_rows = [
    r
    for r in manifest_rows
    if clean(r["final_cohort"]) != "A"
]

if len(A_rows) != EXPECTED_A:
    die(
        f"Expected A={EXPECTED_A}; "
        f"observed {len(A_rows)}"
    )

if len(external_rows) != EXPECTED_EXTERNAL:
    die(
        f"Expected external={EXPECTED_EXTERNAL}; "
        f"observed {len(external_rows)}"
    )

A_accessions = [
    clean(r["assembly_accession_final"])
    for r in A_rows
]

if len(set(A_accessions)) != EXPECTED_A:
    die("Duplicate accessions in Development A")

missing_features = [
    x
    for x in A_accessions
    if x not in feature_index
]

if missing_features:
    die(
        f"Missing feature rows for "
        f"{len(missing_features)} A genomes"
    )

if not PREFLIGHT_REPORT.is_file():
    die(
        f"Missing preflight report: {PREFLIGHT_REPORT}"
    )

preflight_text = PREFLIGHT_REPORT.read_text(
    encoding="utf-8"
)

if "SPLIT_PREFLIGHT=PASS" not in preflight_text:
    die(
        "Previous split preflight did not contain "
        "SPLIT_PREFLIGHT=PASS"
    )

print()
print("[Frozen inputs]")
print(f"Final manifest  : {len(manifest_rows)}")
print(f"Development A   : {len(A_rows)}")
print(f"External heldout: {len(external_rows)}")
print(f"Feature rows    : {len(feature_index)}")
print("Preflight       : PASS")


# ------------------------------------------------------------
# Detect provenance / leakage columns
# ------------------------------------------------------------

biosample_col = detect_optional_field(
    manifest_fields,
    [
        "biosample",
        "bio_sample",
        "ncbi_biosample",
    ],
)

paired_col = detect_optional_field(
    manifest_fields,
    [
        "paired_assembly",
        "paired_assembly_accession",
        "paired_accession",
    ],
)

md5_col = detect_optional_field(
    manifest_fields,
    [
        "md5",
        "fasta_md5",
        "sequence_md5",
    ],
)

canonical_hash_col = detect_optional_field(
    manifest_fields,
    [
        "canonical_sequence_hash",
        "canonical_contigset_sha256",
    ],
)

if canonical_hash_col is None:
    for col in manifest_fields:
        lc = col.lower()

        if (
            "canonical" in lc
            and ("sha" in lc or "hash" in lc)
        ):
            canonical_hash_col = col
            break

print()
print("[Leakage/provenance columns]")
print(f"BioSample       : {biosample_col}")
print(f"Paired assembly : {paired_col}")
print(f"MD5             : {md5_col}")
print(f"Canonical hash  : {canonical_hash_col}")


# ------------------------------------------------------------
# Pair-capacity audit
# ------------------------------------------------------------

def pair_capacity(rows):
    """
    Exact combinatorial capacity within one partition.

    Positive:
        same GTDB species cluster

    Hard negative:
        same genus, different species

    Medium negative:
        same family, different genus

    Easy negative:
        different family
    """

    species_counts = Counter(
        clean(r["gtdb_species_cluster"])
        for r in rows
    )

    genus_counts = Counter(
        clean(r["gtdb_genus"])
        for r in rows
    )

    family_counts = Counter(
        clean(r["gtdb_family"])
        for r in rows
    )

    total_pairs = comb2(len(rows))

    positive = sum(
        comb2(n)
        for n in species_counts.values()
    )

    within_genus = sum(
        comb2(n)
        for n in genus_counts.values()
    )

    within_family = sum(
        comb2(n)
        for n in family_counts.values()
    )

    hard = within_genus - positive
    medium = within_family - within_genus
    easy = total_pairs - within_family

    different = hard + medium + easy

    if positive + different != total_pairs:
        raise RuntimeError(
            "Pair-capacity decomposition failed"
        )

    return {
        "n_genomes": len(rows),
        "n_species": len(species_counts),
        "n_genera": len(genus_counts),
        "n_families": len(family_counts),
        "positive": positive,
        "hard": hard,
        "medium": medium,
        "easy": easy,
        "different": different,
        "total_pairs": total_pairs,
    }


def val_test_mix_capacity(role, cap):
    """
    Desired validation/test pair design.

    Validation:
        up to 20k unique positives + 1:1 negatives

    Test:
        up to 30k unique positives + 1:1 negatives

    Negative target:
        50% hard
        30% medium
        20% easy

    Deficits are NOT silently fixed here.
    They are only reported; the next pair-building stage
    will apply the pre-registered redistribution rule.
    """

    if role == "val":
        positive_limit = 20000
    elif role == "test":
        positive_limit = 30000
    else:
        return {
            "desired_positive": "",
            "desired_negative": "",
            "hard_needed": "",
            "medium_needed": "",
            "easy_needed": "",
            "hard_deficit": "",
            "medium_deficit": "",
            "easy_deficit": "",
        }

    desired_positive = min(
        positive_limit,
        cap["positive"],
    )

    desired_negative = desired_positive

    hard_needed = int(
        math.floor(desired_negative * 0.50)
    )

    medium_needed = int(
        math.floor(desired_negative * 0.30)
    )

    easy_needed = (
        desired_negative
        - hard_needed
        - medium_needed
    )

    return {
        "desired_positive": desired_positive,
        "desired_negative": desired_negative,
        "hard_needed": hard_needed,
        "medium_needed": medium_needed,
        "easy_needed": easy_needed,
        "hard_deficit": max(
            0,
            hard_needed - cap["hard"],
        ),
        "medium_deficit": max(
            0,
            medium_needed - cap["medium"],
        ),
        "easy_deficit": max(
            0,
            easy_needed - cap["easy"],
        ),
    }


# ------------------------------------------------------------
# G1 random genome split
# ------------------------------------------------------------

def make_g1(rows, seed):
    rng = random.Random(seed)

    shuffled = list(rows)
    rng.shuffle(shuffled)

    targets = target_counts(len(shuffled))

    role_map = {}

    i0 = 0

    for role in ROLE_ORDER:
        n = targets[role]

        subset = shuffled[i0:i0 + n]

        for r in subset:
            acc = clean(
                r["assembly_accession_final"]
            )

            role_map[acc] = role

        i0 += n

    if len(role_map) != len(rows):
        raise RuntimeError(
            "G1 assignment size mismatch"
        )

    return role_map


# ------------------------------------------------------------
# G2-G4 group-aware greedy split
# ------------------------------------------------------------

def rows_by_group(rows, group_field):
    groups = defaultdict(list)

    for r in rows:
        g = clean(r[group_field])

        if not g:
            raise RuntimeError(
                f"Empty {group_field}"
            )

        groups[g].append(r)

    return dict(groups)


def assignment_to_role_rows(
    groups,
    group_assignment,
):
    out = {
        role: []
        for role in ROLE_ORDER
    }

    for group, role in group_assignment.items():
        out[role].extend(groups[group])

    return out


def candidate_final_score(
    role_rows,
    group_assignment,
    genome_targets,
    group_targets,
):
    """
    Lower is better.

    Primary:
        genome count ratio balance.

    Secondary:
        number of taxonomic groups per role.

    A large penalty discourages a split that completely
    destroys a negative-difficulty stratum in val/test.

    This uses taxonomy only, never model outcomes.
    """

    total_genomes = sum(
        len(x)
        for x in role_rows.values()
    )

    group_counts = Counter(
        group_assignment.values()
    )

    genome_dev = 0.0

    for role in ROLE_ORDER:
        genome_dev += (
            abs(
                len(role_rows[role])
                - genome_targets[role]
            )
            / total_genomes
        )

    total_groups = len(group_assignment)

    group_dev = 0.0

    for role in ROLE_ORDER:
        group_dev += (
            abs(
                group_counts[role]
                - group_targets[role]
            )
            / total_groups
        )

    zero_strata = 0

    for role in ("val", "test"):
        cap = pair_capacity(role_rows[role])

        for key in (
            "positive",
            "hard",
            "medium",
            "easy",
        ):
            if cap[key] == 0:
                zero_strata += 1

    return (
        genome_dev
        + 0.25 * group_dev
        + 10.0 * zero_strata
    )


def make_group_split(
    rows,
    group_field,
    seed,
    attempts=N_GREEDY_ATTEMPTS,
):
    groups = rows_by_group(
        rows,
        group_field,
    )

    group_keys = list(groups)

    genome_targets = target_counts(
        len(rows)
    )

    group_targets = target_counts(
        len(group_keys)
    )

    best = None
    best_score = None

    for attempt in range(attempts):
        rng = random.Random(
            seed
            + attempt * 104729
        )

        order = list(group_keys)
        rng.shuffle(order)

        genome_load = {
            role: 0
            for role in ROLE_ORDER
        }

        group_load = {
            role: 0
            for role in ROLE_ORDER
        }

        assignment = {}

        for group in order:
            size = len(groups[group])

            scored_roles = []

            for role in ROLE_ORDER:
                projected_genomes = (
                    genome_load[role]
                    + size
                )

                projected_groups = (
                    group_load[role]
                    + 1
                )

                genome_fill = (
                    projected_genomes
                    / max(
                        1,
                        genome_targets[role],
                    )
                )

                group_fill = (
                    projected_groups
                    / max(
                        1,
                        group_targets[role],
                    )
                )

                overflow = max(
                    0,
                    projected_genomes
                    - genome_targets[role],
                ) / max(
                    1,
                    genome_targets[role],
                )

                # Seeded tiny tie-break.
                score = (
                    genome_fill
                    + 0.08 * group_fill
                    + 4.0 * overflow
                    + rng.random() * 1e-10
                )

                scored_roles.append(
                    (score, role)
                )

            scored_roles.sort()

            role = scored_roles[0][1]

            assignment[group] = role

            genome_load[role] += size
            group_load[role] += 1

        role_rows = assignment_to_role_rows(
            groups,
            assignment,
        )

        if any(
            len(role_rows[r]) == 0
            for r in ROLE_ORDER
        ):
            continue

        final_score = candidate_final_score(
            role_rows,
            assignment,
            genome_targets,
            group_targets,
        )

        if (
            best is None
            or final_score < best_score
        ):
            best = {
                "groups": groups,
                "assignment": assignment,
                "role_rows": role_rows,
                "genome_targets": genome_targets,
                "group_targets": group_targets,
                "score": final_score,
            }

            best_score = final_score

    if best is None:
        raise RuntimeError(
            f"Could not generate split for "
            f"{group_field}"
        )

    role_map = {}

    for group, role in best["assignment"].items():
        for r in best["groups"][group]:
            acc = clean(
                r["assembly_accession_final"]
            )

            if acc in role_map:
                raise RuntimeError(
                    f"Duplicate assignment: {acc}"
                )

            role_map[acc] = role

    if len(role_map) != len(rows):
        raise RuntimeError(
            "Group assignment did not cover "
            "all Development genomes"
        )

    return role_map, best


# ------------------------------------------------------------
# Leakage audit
# ------------------------------------------------------------

def split_rows_from_role_map(rows, role_map):
    out = {
        role: []
        for role in ROLE_ORDER
    }

    for r in rows:
        acc = clean(
            r["assembly_accession_final"]
        )

        role = role_map.get(acc)

        if role not in ROLE_ORDER:
            raise RuntimeError(
                f"Missing/invalid role for {acc}"
            )

        out[role].append(r)

    return out


def nonempty_set(rows, field):
    if field is None:
        return set()

    return {
        clean(r.get(field, ""))
        for r in rows
        if clean(r.get(field, ""))
    }


def overlap_counts(role_rows, field):
    sets = {
        role: nonempty_set(
            role_rows[role],
            field,
        )
        for role in ROLE_ORDER
    }

    return {
        "train_val": len(
            sets["train"]
            & sets["val"]
        ),
        "train_test": len(
            sets["train"]
            & sets["test"]
        ),
        "val_test": len(
            sets["val"]
            & sets["test"]
        ),
    }


def accession_overlap(role_rows):
    return overlap_counts(
        role_rows,
        "assembly_accession_final",
    )


def paired_identifier_overlap(role_rows):
    """
    Detect GCA/GCF paired-assembly leakage more robustly
    than comparing only paired_assembly values.

    Each role identifier set contains:
        actual accession
        +
        paired assembly accession, when present
    """

    if paired_col is None:
        return {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }

    role_ids = {}

    for role in ROLE_ORDER:
        s = set()

        for r in role_rows[role]:
            acc = clean(
                r["assembly_accession_final"]
            )

            paired = clean(
                r.get(paired_col, "")
            )

            if acc:
                s.add(acc)

            if paired:
                s.add(paired)

        role_ids[role] = s

    return {
        "train_val": len(
            role_ids["train"]
            & role_ids["val"]
        ),
        "train_test": len(
            role_ids["train"]
            & role_ids["test"]
        ),
        "val_test": len(
            role_ids["val"]
            & role_ids["test"]
        ),
    }


def total_overlap(d):
    return sum(d.values())


def validate_one_split(
    protocol,
    group_field,
    seed,
    role_map,
):
    role_rows = split_rows_from_role_map(
        A_rows,
        role_map,
    )

    if (
        sum(len(x) for x in role_rows.values())
        != EXPECTED_A
    ):
        raise RuntimeError(
            f"{protocol} seed {seed}: "
            "genome count mismatch"
        )

    for role in ROLE_ORDER:
        if not role_rows[role]:
            raise RuntimeError(
                f"{protocol} seed {seed}: "
                f"empty role {role}"
            )

    # Accessions must never overlap.
    acc_ov = accession_overlap(
        role_rows
    )

    if total_overlap(acc_ov) != 0:
        raise RuntimeError(
            f"{protocol} seed {seed}: "
            "accession leakage"
        )

    # Group leakage is forbidden G2-G4.
    if protocol != "G1":
        group_ov = overlap_counts(
            role_rows,
            group_field,
        )

        if total_overlap(group_ov) != 0:
            raise RuntimeError(
                f"{protocol} seed {seed}: "
                f"{group_field} leakage"
            )
    else:
        group_ov = {
            "train_val": "",
            "train_test": "",
            "val_test": "",
        }

    # Exact canonical sequence identities.
    if canonical_hash_col:
        hash_ov = overlap_counts(
            role_rows,
            canonical_hash_col,
        )

        if total_overlap(hash_ov) != 0:
            raise RuntimeError(
                f"{protocol} seed {seed}: "
                "canonical sequence leakage"
            )
    else:
        hash_ov = {
            "train_val": "",
            "train_test": "",
            "val_test": "",
        }

    # BioSample.
    if biosample_col:
        biosample_ov = overlap_counts(
            role_rows,
            biosample_col,
        )

        if total_overlap(biosample_ov) != 0:
            raise RuntimeError(
                f"{protocol} seed {seed}: "
                "BioSample leakage"
            )
    else:
        biosample_ov = {
            "train_val": "",
            "train_test": "",
            "val_test": "",
        }

    # MD5 if available.
    if md5_col:
        md5_ov = overlap_counts(
            role_rows,
            md5_col,
        )

        if total_overlap(md5_ov) != 0:
            raise RuntimeError(
                f"{protocol} seed {seed}: "
                "MD5 leakage"
            )
    else:
        md5_ov = {
            "train_val": "",
            "train_test": "",
            "val_test": "",
        }

    # Paired GCF/GCA identifiers.
    paired_ov = paired_identifier_overlap(
        role_rows
    )

    if total_overlap(paired_ov) != 0:
        raise RuntimeError(
            f"{protocol} seed {seed}: "
            "paired-assembly identifier leakage"
        )

    return {
        "role_rows": role_rows,
        "accession_overlap": acc_ov,
        "group_overlap": group_ov,
        "hash_overlap": hash_ov,
        "biosample_overlap": biosample_ov,
        "md5_overlap": md5_ov,
        "paired_overlap": paired_ov,
    }


# ------------------------------------------------------------
# Generate all frozen split runs in memory first
# ------------------------------------------------------------

all_runs = []

print()
print("[Generating G1]")

g1_role_map = make_g1(
    A_rows,
    G1_SEED,
)

g1_valid = validate_one_split(
    "G1",
    "assembly_accession_final",
    G1_SEED,
    g1_role_map,
)

all_runs.append({
    "protocol": "G1",
    "protocol_name": "random_genome",
    "group_field": "assembly_accession_final",
    "seed": G1_SEED,
    "role_map": g1_role_map,
    "validation": g1_valid,
    "greedy_info": None,
})

print(
    f"G1 seed={G1_SEED}: PASS"
)

for protocol, cfg in PROTOCOLS.items():
    print()
    print(
        f"[Generating {protocol}: "
        f"{cfg['name']}]"
    )

    for seed in TAXONOMY_SEEDS:
        role_map, info = make_group_split(
            A_rows,
            cfg["group_field"],
            seed,
        )

        valid = validate_one_split(
            protocol,
            cfg["group_field"],
            seed,
            role_map,
        )

        all_runs.append({
            "protocol": protocol,
            "protocol_name": cfg["name"],
            "group_field": cfg["group_field"],
            "seed": seed,
            "role_map": role_map,
            "validation": valid,
            "greedy_info": info,
        })

        counts = Counter(
            role_map.values()
        )

        print(
            f"{protocol} seed={seed}: "
            f"train={counts['train']} "
            f"val={counts['val']} "
            f"test={counts['test']} "
            f"PASS"
        )


# ------------------------------------------------------------
# Make sure the three taxonomy seeds are actually different
# ------------------------------------------------------------

seed_diversity_rows = []

for protocol in ("G2", "G3", "G4"):
    runs = [
        r
        for r in all_runs
        if r["protocol"] == protocol
    ]

    runs.sort(
        key=lambda x: x["seed"]
    )

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a = runs[i]
            b = runs[j]

            changed = sum(
                a["role_map"][acc]
                != b["role_map"][acc]
                for acc in A_accessions
            )

            fraction = (
                changed
                / len(A_accessions)
            )

            gate = (
                "PASS"
                if changed > 0
                else "FAIL"
            )

            seed_diversity_rows.append({
                "protocol": protocol,
                "seed_a": a["seed"],
                "seed_b": b["seed"],
                "genomes_changed_role": changed,
                "fraction_changed_role": (
                    f"{fraction:.12g}"
                ),
                "gate": gate,
            })

            if changed == 0:
                die(
                    f"{protocol}: split seeds "
                    f"{a['seed']} and "
                    f"{b['seed']} produced "
                    "identical assignments"
                )


# ------------------------------------------------------------
# Build audit/source-data before writing
# ------------------------------------------------------------

summary_rows = []
capacity_rows = []
leakage_rows = []
group_assignment_rows = []

for run in all_runs:
    protocol = run["protocol"]
    protocol_name = run["protocol_name"]
    seed = run["seed"]
    group_field = run["group_field"]
    role_map = run["role_map"]
    valid = run["validation"]

    role_rows = valid["role_rows"]

    split_id = (
        f"{protocol}_seed{seed}"
    )

    genome_targets = target_counts(
        EXPECTED_A
    )

    if protocol == "G1":
        group_targets = target_counts(
            EXPECTED_A
        )
    else:
        n_groups = len({
            clean(r[group_field])
            for r in A_rows
        })

        group_targets = target_counts(
            n_groups
        )

    for role in ROLE_ORDER:
        rows_role = role_rows[role]

        group_count = len({
            clean(r[group_field])
            for r in rows_role
        })

        cap = pair_capacity(
            rows_role
        )

        mix = val_test_mix_capacity(
            role,
            cap,
        )

        summary_rows.append({
            "split_id": split_id,
            "protocol": protocol,
            "protocol_name": protocol_name,
            "seed": seed,
            "group_unit": group_field,
            "role": role,
            "n_genomes": len(rows_role),
            "target_genomes": genome_targets[role],
            "delta_genomes": (
                len(rows_role)
                - genome_targets[role]
            ),
            "genome_fraction": (
                f"{len(rows_role) / EXPECTED_A:.12g}"
            ),
            "n_groups": group_count,
            "target_groups": group_targets[role],
            "n_species_clusters": cap["n_species"],
            "n_genera": cap["n_genera"],
            "n_families": cap["n_families"],
        })

        capacity_rows.append({
            "split_id": split_id,
            "protocol": protocol,
            "seed": seed,
            "role": role,
            "n_genomes": cap["n_genomes"],
            "positive_unique_capacity": cap["positive"],
            "hard_negative_capacity": cap["hard"],
            "medium_negative_capacity": cap["medium"],
            "easy_negative_capacity": cap["easy"],
            "different_species_capacity": cap["different"],
            "all_pair_capacity": cap["total_pairs"],
            **mix,
        })

    def add_leakage(
        check_name,
        ov,
        expected,
    ):
        for comparison in (
            "train_val",
            "train_test",
            "val_test",
        ):
            observed = ov[comparison]

            if observed == "":
                gate = "NA"
            else:
                gate = (
                    "PASS"
                    if int(observed) == expected
                    else "FAIL"
                )

            leakage_rows.append({
                "split_id": split_id,
                "protocol": protocol,
                "seed": seed,
                "check": check_name,
                "comparison": comparison,
                "observed_overlap": observed,
                "expected_overlap": (
                    "" if observed == ""
                    else expected
                ),
                "gate": gate,
            })

    add_leakage(
        "accession",
        valid["accession_overlap"],
        0,
    )

    add_leakage(
        "group_unit",
        valid["group_overlap"],
        0,
    )

    add_leakage(
        "canonical_sequence_hash",
        valid["hash_overlap"],
        0,
    )

    add_leakage(
        "biosample",
        valid["biosample_overlap"],
        0,
    )

    add_leakage(
        "md5",
        valid["md5_overlap"],
        0,
    )

    add_leakage(
        "paired_assembly_identifier",
        valid["paired_overlap"],
        0,
    )

    if protocol != "G1":
        info = run["greedy_info"]

        for group, role in sorted(
            info["assignment"].items()
        ):
            rows_group = info["groups"][group]

            group_assignment_rows.append({
                "split_id": split_id,
                "protocol": protocol,
                "seed": seed,
                "group_unit": group_field,
                "group_value": group,
                "role": role,
                "n_genomes": len(rows_group),
                "n_species_clusters": len({
                    clean(r["gtdb_species_cluster"])
                    for r in rows_group
                }),
                "n_genera": len({
                    clean(r["gtdb_genus"])
                    for r in rows_group
                }),
                "n_families": len({
                    clean(r["gtdb_family"])
                    for r in rows_group
                }),
            })


# ------------------------------------------------------------
# Final leakage gate before touching output directory
# ------------------------------------------------------------

failed_leakage = [
    r
    for r in leakage_rows
    if r["gate"] == "FAIL"
]

if failed_leakage:
    for r in failed_leakage[:20]:
        print(r)

    die(
        f"Leakage audit failed: "
        f"{len(failed_leakage)} checks"
    )


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

if OUTDIR.exists():
    die(
        f"Output directory already exists: "
        f"{OUTDIR}\n"
        "Refusing to overwrite frozen splits."
    )

MANIFEST_DIR = OUTDIR / "manifests"
AUDIT_DIR = OUTDIR / "audits"

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ------------------------------------------------------------
# Write each split manifest
# ------------------------------------------------------------

prepend_fields = [
    "split_id",
    "protocol",
    "protocol_name",
    "split_seed",
    "role",
    "group_unit",
    "group_value",
    "feature_row_index",
]

split_manifest_paths = []

role_rank = {
    "train": 0,
    "val": 1,
    "test": 2,
}

for run in all_runs:
    protocol = run["protocol"]
    protocol_name = run["protocol_name"]
    seed = run["seed"]
    group_field = run["group_field"]
    role_map = run["role_map"]

    split_id = (
        f"{protocol}_seed{seed}"
    )

    out_rows = []

    for original in A_rows:
        acc = clean(
            original["assembly_accession_final"]
        )

        role = role_map[acc]

        row = dict(original)

        row["split_id"] = split_id
        row["protocol"] = protocol
        row["protocol_name"] = protocol_name
        row["split_seed"] = seed
        row["role"] = role
        row["group_unit"] = group_field
        row["group_value"] = clean(
            original[group_field]
        )
        row["feature_row_index"] = (
            feature_index[acc]
        )

        out_rows.append(row)

    out_rows.sort(
        key=lambda r: (
            role_rank[r["role"]],
            r["group_value"],
            r["assembly_accession_final"],
        )
    )

    fields = list(prepend_fields)

    for f in manifest_fields:
        if f not in fields:
            fields.append(f)

    path = (
        MANIFEST_DIR
        / f"{split_id}.tsv"
    )

    write_tsv(
        path,
        out_rows,
        fields,
    )

    split_manifest_paths.append(path)


# ------------------------------------------------------------
# Freeze external 780 separately
# ------------------------------------------------------------

external_output_rows = []

external_role_map = {
    "B1": "CD1_external_matched_species",
    "B2_seen_genus": "CD2a_external_unseen_species_seen_genus",
    "B2_unseen_genus": "CD2b_external_unseen_species_unseen_genus",
}

for original in external_rows:
    acc = clean(
        original["assembly_accession_final"]
    )

    cohort = clean(
        original["final_cohort"]
    )

    if cohort not in external_role_map:
        die(
            f"Unexpected external cohort: "
            f"{cohort}"
        )

    row = dict(original)

    row["external_protocol"] = (
        external_role_map[cohort]
    )

    row["role"] = "external"

    row["feature_row_index"] = (
        feature_index[acc]
    )

    external_output_rows.append(row)

external_output_rows.sort(
    key=lambda r: (
        r["final_cohort"],
        r["assembly_accession_final"],
    )
)

external_fields = [
    "external_protocol",
    "role",
    "feature_row_index",
]

for f in manifest_fields:
    if f not in external_fields:
        external_fields.append(f)

external_path = (
    MANIFEST_DIR
    / "external_holdout_780.tsv"
)

write_tsv(
    external_path,
    external_output_rows,
    external_fields,
)


# ------------------------------------------------------------
# Write audit tables
# ------------------------------------------------------------

write_tsv(
    AUDIT_DIR / "split_summary.tsv",
    summary_rows,
    [
        "split_id",
        "protocol",
        "protocol_name",
        "seed",
        "group_unit",
        "role",
        "n_genomes",
        "target_genomes",
        "delta_genomes",
        "genome_fraction",
        "n_groups",
        "target_groups",
        "n_species_clusters",
        "n_genera",
        "n_families",
    ],
)

write_tsv(
    AUDIT_DIR / "pair_capacity_by_partition.tsv",
    capacity_rows,
    [
        "split_id",
        "protocol",
        "seed",
        "role",
        "n_genomes",
        "positive_unique_capacity",
        "hard_negative_capacity",
        "medium_negative_capacity",
        "easy_negative_capacity",
        "different_species_capacity",
        "all_pair_capacity",
        "desired_positive",
        "desired_negative",
        "hard_needed",
        "medium_needed",
        "easy_needed",
        "hard_deficit",
        "medium_deficit",
        "easy_deficit",
    ],
)

write_tsv(
    AUDIT_DIR / "leakage_audit.tsv",
    leakage_rows,
    [
        "split_id",
        "protocol",
        "seed",
        "check",
        "comparison",
        "observed_overlap",
        "expected_overlap",
        "gate",
    ],
)

write_tsv(
    AUDIT_DIR / "group_assignments.tsv",
    group_assignment_rows,
    [
        "split_id",
        "protocol",
        "seed",
        "group_unit",
        "group_value",
        "role",
        "n_genomes",
        "n_species_clusters",
        "n_genera",
        "n_families",
    ],
)

write_tsv(
    AUDIT_DIR / "seed_diversity.tsv",
    seed_diversity_rows,
    [
        "protocol",
        "seed_a",
        "seed_b",
        "genomes_changed_role",
        "fraction_changed_role",
        "gate",
    ],
)


# ------------------------------------------------------------
# Configuration / provenance
# ------------------------------------------------------------

config = {
    "split_version": "g1_g4_v1_20260916",
    "manifest": str(MANIFEST),
    "manifest_sha256": sha256_file(MANIFEST),
    "feature_rows": str(FEATURE_ROWS),
    "feature_rows_sha256": sha256_file(FEATURE_ROWS),
    "development_cohort": "A",
    "development_genomes": EXPECTED_A,
    "external_genomes": EXPECTED_EXTERNAL,
    "ratios": RATIOS,
    "G1": {
        "group_unit": "genome",
        "seed": G1_SEED,
    },
    "G2": {
        "group_unit": "gtdb_species_cluster",
        "seeds": TAXONOMY_SEEDS,
    },
    "G3": {
        "group_unit": "gtdb_genus",
        "seeds": TAXONOMY_SEEDS,
    },
    "G4": {
        "group_unit": "gtdb_family",
        "seeds": TAXONOMY_SEEDS,
    },
    "group_split_algorithm": (
        "seeded random group order + greedy proportional "
        "genome/group balancing; best of "
        f"{N_GREEDY_ATTEMPTS} deterministic attempts"
    ),
    "pair_capacity_definition": {
        "positive": (
            "same GTDB species cluster"
        ),
        "hard_negative": (
            "same genus, different species cluster"
        ),
        "medium_negative": (
            "same family, different genus"
        ),
        "easy_negative": (
            "different family"
        ),
    },
    "pair_partition_rule": (
        "future train/val/test pairs must be constructed "
        "within the same partition only"
    ),
    "external_policy": (
        "B1/B2 remain external and are not used for "
        "split selection/model tuning"
    ),
}

config_path = (
    OUTDIR / "split_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

report_path = (
    OUTDIR
    / "SPLIT_GENERATION_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "HGV-V3 G1-G4 SPLIT GENERATION REPORT\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write("STATUS: PASS\n\n")

    f.write("INPUT\n")
    f.write(
        f"Development A genomes: "
        f"{EXPECTED_A}\n"
    )
    f.write(
        f"External held-out genomes: "
        f"{EXPECTED_EXTERNAL}\n"
    )
    f.write(
        "Train/Val/Test target: 70/15/15\n\n"
    )

    f.write("PROTOCOLS\n")
    f.write(
        f"G1 random genome: seed {G1_SEED}\n"
    )
    f.write(
        "G2 species holdout: "
        + ",".join(map(str, TAXONOMY_SEEDS))
        + "\n"
    )
    f.write(
        "G3 genus holdout: "
        + ",".join(map(str, TAXONOMY_SEEDS))
        + "\n"
    )
    f.write(
        "G4 family holdout: "
        + ",".join(map(str, TAXONOMY_SEEDS))
        + "\n\n"
    )

    f.write("SPLIT COUNTS\n")

    for row in summary_rows:
        f.write(
            f"{row['split_id']} "
            f"{row['role']}: "
            f"n={row['n_genomes']} "
            f"target={row['target_genomes']} "
            f"groups={row['n_groups']}\n"
        )

    f.write("\nLEAKAGE\n")
    f.write(
        "All required accession/group/hash/"
        "BioSample/paired-assembly checks: PASS\n"
    )

    f.write("\nSEED DIVERSITY\n")

    for row in seed_diversity_rows:
        f.write(
            f"{row['protocol']} "
            f"{row['seed_a']} vs "
            f"{row['seed_b']}: "
            f"{row['genomes_changed_role']} "
            f"genomes changed role "
            f"({row['fraction_changed_role']})\n"
        )

    f.write("\nEXTERNAL\n")
    f.write(
        "B1/B2 external genomes remain fully held out: "
        f"{len(external_output_rows)}\n"
    )

    f.write(
        "\nSPLIT_GENERATION=PASS\n"
    )


# ------------------------------------------------------------
# SHA256 freeze
# ------------------------------------------------------------

files_to_hash = []

files_to_hash.extend(
    split_manifest_paths
)

files_to_hash.extend([
    external_path,
    AUDIT_DIR / "split_summary.tsv",
    AUDIT_DIR / "pair_capacity_by_partition.tsv",
    AUDIT_DIR / "leakage_audit.tsv",
    AUDIT_DIR / "group_assignments.tsv",
    AUDIT_DIR / "seed_diversity.tsv",
    config_path,
    report_path,
])

sha_path = (
    OUTDIR / "SHA256SUMS.txt"
)

with open(
    sha_path,
    "w",
    encoding="utf-8",
) as f:
    for path in files_to_hash:
        relative = path.relative_to(
            OUTDIR
        )

        f.write(
            f"{sha256_file(path)}  "
            f"{relative}\n"
        )


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print()
print("=" * 78)
print("ALL G1-G4 SPLIT GATES PASSED")
print("=" * 78)

for row in summary_rows:
    print(
        f"{row['split_id']:22s} "
        f"{row['role']:5s} "
        f"n={row['n_genomes']:4d} "
        f"groups={row['n_groups']:4d}"
    )

print()
print(
    f"External held out : "
    f"{len(external_output_rows)}"
)

print(
    f"Output            : "
    f"{OUTDIR}"
)

print()
print("SPLIT_GENERATION=PASS")
