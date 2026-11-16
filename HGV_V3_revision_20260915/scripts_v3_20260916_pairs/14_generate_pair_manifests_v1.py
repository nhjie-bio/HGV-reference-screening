#!/usr/bin/env python3

import csv
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("HGV_V3_revision_20260915")

SPLIT_ROOT = (
    BASE
    / "04_splits/g1_g4_v1_20260916"
)

SPLIT_MANIFEST_DIR = (
    SPLIT_ROOT
    / "manifests"
)

CAPACITY_FILE = (
    SPLIT_ROOT
    / "audits/pair_capacity_by_partition.tsv"
)

REAUDIT_REPORT = (
    BASE
    / "04_splits/g1_g4_provenance_reaudit_v1_20260916"
    / "PROVENANCE_REAUDIT_REPORT.txt"
)

OUTDIR = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
)

TRAIN_EPOCH_EXAMPLES = 100000
TRAIN_POS_PER_EPOCH = 50000
TRAIN_NEG_PER_EPOCH = 50000

VAL_POS_CAP = 20000
TEST_POS_CAP = 30000

NEG_WEIGHTS = {
    "hard": 0.50,
    "medium": 0.30,
    "easy": 0.20,
}

NEG_ORDER = [
    "hard",
    "medium",
    "easy",
]

PROTOCOL_OFFSET = {
    "G1": 1000,
    "G2": 2000,
    "G3": 3000,
    "G4": 4000,
}

ROLE_OFFSET = {
    "train": 11,
    "val": 22,
    "test": 33,
}


def die(msg):
    print()
    print("PAIR_GENERATION=FAIL")
    print(msg)
    sys.exit(2)


def clean(x):
    return "" if x is None else str(x).strip()


def read_tsv(path):
    if not path.is_file():
        die(f"Missing file: {path}")

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        die(f"No header: {path}")

    return fields, rows


def write_tsv(path, rows, fields):
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )

        w.writeheader()

        for row in rows:
            w.writerow(
                {k: row.get(k, "") for k in fields}
            )


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def comb2(n):
    return n * (n - 1) // 2 if n >= 2 else 0


def stable_pair_id(split_id, role, a, b):
    if b < a:
        a, b = b, a

    raw = (
        f"{split_id}\t{role}\t{a}\t{b}"
    ).encode()

    return hashlib.sha256(raw).hexdigest()[:24]


def normalize_record(r):
    return {
        "accession": clean(
            r["assembly_accession_final"]
        ),
        "feature_row": int(
            r["feature_row_index"]
        ),
        "species": clean(
            r["gtdb_species_cluster"]
        ),
        "genus": clean(
            r["gtdb_genus"]
        ),
        "family": clean(
            r["gtdb_family"]
        ),
        "role": clean(
            r["role"]
        ),
    }


def classify_pair(a, b):
    if a["species"] == b["species"]:
        return "positive"

    if a["genus"] == b["genus"]:
        return "hard"

    if a["family"] == b["family"]:
        return "medium"

    return "easy"


def capacity(records):
    species = Counter(
        r["species"]
        for r in records
    )

    genus = Counter(
        r["genus"]
        for r in records
    )

    family = Counter(
        r["family"]
        for r in records
    )

    positive = sum(
        comb2(n)
        for n in species.values()
    )

    within_genus = sum(
        comb2(n)
        for n in genus.values()
    )

    within_family = sum(
        comb2(n)
        for n in family.values()
    )

    total = comb2(len(records))

    hard = within_genus - positive
    medium = within_family - within_genus
    easy = total - within_family

    if min(
        positive,
        hard,
        medium,
        easy,
    ) < 0:
        raise RuntimeError(
            "Negative capacity detected"
        )

    if (
        positive
        + hard
        + medium
        + easy
        != total
    ):
        raise RuntimeError(
            "Pair capacity decomposition failed"
        )

    return {
        "positive": positive,
        "hard": hard,
        "medium": medium,
        "easy": easy,
        "total": total,
    }


def target_negative_mix(n):
    raw = {
        k: n * NEG_WEIGHTS[k]
        for k in NEG_ORDER
    }

    out = {
        k: int(math.floor(raw[k]))
        for k in NEG_ORDER
    }

    remaining = n - sum(out.values())

    ranked = sorted(
        NEG_ORDER,
        key=lambda k: (
            -(raw[k] - math.floor(raw[k])),
            NEG_ORDER.index(k),
        ),
    )

    for k in ranked[:remaining]:
        out[k] += 1

    if sum(out.values()) != n:
        raise RuntimeError(
            "Negative target allocation failed"
        )

    return out


def capacity_aware_negative_allocation(
    n,
    caps,
):
    """
    1. Start from 50/30/20 H/M/E target.
    2. Clamp each stratum to actual unique capacity.
    3. Redistribute any deficit deterministically,
       prioritizing harder available negatives:
           hard -> medium -> easy.
    4. Never duplicate a pair merely to hit a target.
    """

    target = target_negative_mix(n)

    actual = {
        k: min(
            target[k],
            caps[k],
        )
        for k in NEG_ORDER
    }

    remaining = n - sum(actual.values())

    while remaining > 0:
        progressed = False

        for k in NEG_ORDER:
            spare = (
                caps[k]
                - actual[k]
            )

            if spare <= 0:
                continue

            take = min(
                remaining,
                spare,
            )

            actual[k] += take
            remaining -= take
            progressed = True

            if remaining == 0:
                break

        if not progressed:
            break

    if remaining != 0:
        raise RuntimeError(
            f"Insufficient unique negative "
            f"capacity for n={n}; "
            f"remaining={remaining}"
        )

    return target, actual


def all_positive_pairs(records):
    groups = defaultdict(list)

    for r in records:
        groups[r["species"]].append(r)

    pairs = []

    for species in sorted(groups):
        members = sorted(
            groups[species],
            key=lambda x: x["accession"],
        )

        for a, b in itertools.combinations(
            members,
            2,
        ):
            pairs.append((a, b))

    return pairs


def sample_positive_pairs(
    records,
    target,
    seed,
):
    pairs = all_positive_pairs(records)

    if len(pairs) < target:
        raise RuntimeError(
            f"Positive target={target}, "
            f"capacity={len(pairs)}"
        )

    if len(pairs) == target:
        return pairs

    rng = random.Random(seed)

    chosen = rng.sample(
        pairs,
        target,
    )

    chosen.sort(
        key=lambda p: (
            p[0]["accession"],
            p[1]["accession"],
        )
    )

    return chosen


def reservoir_negative_sample(
    records,
    target_by_type,
    seed,
):
    rng = random.Random(seed)

    reservoir = {
        k: []
        for k in NEG_ORDER
    }

    seen = Counter()

    n = len(records)

    for i in range(n):
        a = records[i]

        for j in range(i + 1, n):
            b = records[j]

            kind = classify_pair(a, b)

            if kind == "positive":
                continue

            seen[kind] += 1

            k = target_by_type[kind]

            if k <= 0:
                continue

            if len(reservoir[kind]) < k:
                reservoir[kind].append(
                    (a, b)
                )
                continue

            x = rng.randrange(
                seen[kind]
            )

            if x < k:
                reservoir[kind][x] = (
                    a,
                    b,
                )

    for kind in NEG_ORDER:
        if len(reservoir[kind]) != target_by_type[kind]:
            raise RuntimeError(
                f"{kind}: requested "
                f"{target_by_type[kind]}, "
                f"sampled {len(reservoir[kind])}"
            )

    return reservoir, dict(seen)


def pair_row(
    split_id,
    protocol,
    split_seed,
    role,
    pair_seed,
    a,
    b,
    pair_type,
):
    if b["accession"] < a["accession"]:
        a, b = b, a

    label = (
        1
        if pair_type == "positive"
        else 0
    )

    return {
        "pair_id": stable_pair_id(
            split_id,
            role,
            a["accession"],
            b["accession"],
        ),
        "split_id": split_id,
        "protocol": protocol,
        "split_seed": split_seed,
        "role": role,
        "pair_sampling_seed": pair_seed,
        "accession_a": a["accession"],
        "accession_b": b["accession"],
        "feature_row_a": a["feature_row"],
        "feature_row_b": b["feature_row"],
        "label": label,
        "pair_type": pair_type,
        "teacher_allowed": (
            1 if role == "train" else 0
        ),
        "species_cluster_a": a["species"],
        "species_cluster_b": b["species"],
        "genus_a": a["genus"],
        "genus_b": b["genus"],
        "family_a": a["family"],
        "family_b": b["family"],
    }


PAIR_FIELDS = [
    "pair_id",
    "split_id",
    "protocol",
    "split_seed",
    "role",
    "pair_sampling_seed",
    "accession_a",
    "accession_b",
    "feature_row_a",
    "feature_row_b",
    "label",
    "pair_type",
    "teacher_allowed",
    "species_cluster_a",
    "species_cluster_b",
    "genus_a",
    "genus_b",
    "family_a",
    "family_b",
]


def audit_pair_rows(
    rows,
    expected_role,
):
    seen_pairs = set()
    seen_ids = set()

    self_pairs = 0
    duplicate_pairs = 0
    duplicate_ids = 0
    label_errors = 0
    type_errors = 0
    role_errors = 0
    teacher_errors = 0

    type_counts = Counter()

    for r in rows:
        a = clean(r["accession_a"])
        b = clean(r["accession_b"])

        if a == b:
            self_pairs += 1

        key = tuple(sorted((a, b)))

        if key in seen_pairs:
            duplicate_pairs += 1

        seen_pairs.add(key)

        pid = clean(r["pair_id"])

        if pid in seen_ids:
            duplicate_ids += 1

        seen_ids.add(pid)

        if clean(r["role"]) != expected_role:
            role_errors += 1

        species_same = (
            clean(r["species_cluster_a"])
            ==
            clean(r["species_cluster_b"])
        )

        genus_same = (
            clean(r["genus_a"])
            ==
            clean(r["genus_b"])
        )

        family_same = (
            clean(r["family_a"])
            ==
            clean(r["family_b"])
        )

        label = int(r["label"])
        pair_type = clean(r["pair_type"])

        if species_same:
            expected_type = "positive"
            expected_label = 1
        elif genus_same:
            expected_type = "hard"
            expected_label = 0
        elif family_same:
            expected_type = "medium"
            expected_label = 0
        else:
            expected_type = "easy"
            expected_label = 0

        if label != expected_label:
            label_errors += 1

        if pair_type != expected_type:
            type_errors += 1

        if expected_role == "train":
            if int(r["teacher_allowed"]) != 1:
                teacher_errors += 1
        else:
            if int(r["teacher_allowed"]) != 0:
                teacher_errors += 1

        type_counts[pair_type] += 1

    return {
        "n_pairs": len(rows),
        "self_pairs": self_pairs,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_pair_ids": duplicate_ids,
        "label_errors": label_errors,
        "type_errors": type_errors,
        "role_errors": role_errors,
        "teacher_scope_errors": teacher_errors,
        "positive": type_counts["positive"],
        "hard": type_counts["hard"],
        "medium": type_counts["medium"],
        "easy": type_counts["easy"],
    }


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

print("=" * 78)
print("HGV-V3 PAIR MANIFEST GENERATION")
print("=" * 78)

if not REAUDIT_REPORT.is_file():
    die(
        f"Missing provenance re-audit: "
        f"{REAUDIT_REPORT}"
    )

report_text = REAUDIT_REPORT.read_text(
    encoding="utf-8"
)

if "PROVENANCE_REAUDIT=PASS" not in report_text:
    die(
        "Provenance re-audit did not PASS"
    )

split_paths = sorted(
    SPLIT_MANIFEST_DIR.glob(
        "G*_seed*.tsv"
    )
)

if len(split_paths) != 10:
    die(
        f"Expected 10 split manifests; "
        f"observed {len(split_paths)}"
    )

print()
print(f"Split manifests : {len(split_paths)}")
print("Provenance audit: PASS")


# ------------------------------------------------------------
# Load previously audited capacities
# ------------------------------------------------------------

cap_fields, cap_rows = read_tsv(
    CAPACITY_FILE
)

capacity_reference = {}

for r in cap_rows:
    key = (
        clean(r["split_id"]),
        clean(r["role"]),
    )

    capacity_reference[key] = {
        "positive": int(
            r["positive_unique_capacity"]
        ),
        "hard": int(
            r["hard_negative_capacity"]
        ),
        "medium": int(
            r["medium_negative_capacity"]
        ),
        "easy": int(
            r["easy_negative_capacity"]
        ),
    }


# ------------------------------------------------------------
# Generate all artifacts in memory
# ------------------------------------------------------------

artifacts = {}
summary_rows = []
redistribution_rows = []
integrity_rows = []
capacity_crosscheck_rows = []

for split_path in split_paths:

    fields, rows = read_tsv(
        split_path
    )

    required = {
        "split_id",
        "protocol",
        "split_seed",
        "role",
        "assembly_accession_final",
        "feature_row_index",
        "gtdb_species_cluster",
        "gtdb_genus",
        "gtdb_family",
    }

    missing = required - set(fields)

    if missing:
        die(
            f"{split_path.name} missing: "
            + ", ".join(sorted(missing))
        )

    split_ids = {
        clean(r["split_id"])
        for r in rows
    }

    protocols = {
        clean(r["protocol"])
        for r in rows
    }

    seeds = {
        int(r["split_seed"])
        for r in rows
    }

    if (
        len(split_ids) != 1
        or len(protocols) != 1
        or len(seeds) != 1
    ):
        die(
            f"Inconsistent metadata in "
            f"{split_path.name}"
        )

    split_id = next(iter(split_ids))
    protocol = next(iter(protocols))
    split_seed = next(iter(seeds))

    if protocol not in PROTOCOL_OFFSET:
        die(
            f"Unknown protocol: {protocol}"
        )

    print()
    print(f"[{split_id}]")

    artifacts[split_id] = {
        "protocol": protocol,
        "split_seed": split_seed,
        "roles": {},
    }

    for role in (
        "train",
        "val",
        "test",
    ):

        raw_role_rows = [
            r
            for r in rows
            if clean(r["role"]) == role
        ]

        records = [
            normalize_record(r)
            for r in raw_role_rows
        ]

        records.sort(
            key=lambda x: x["accession"]
        )

        if not records:
            die(
                f"{split_id} {role}: "
                f"empty partition"
            )

        current_cap = capacity(records)

        ref_key = (
            split_id,
            role,
        )

        if ref_key not in capacity_reference:
            die(
                f"Capacity reference missing "
                f"{ref_key}"
            )

        ref = capacity_reference[
            ref_key
        ]

        mismatches = []

        for kind in (
            "positive",
            "hard",
            "medium",
            "easy",
        ):
            if current_cap[kind] != ref[kind]:
                mismatches.append(
                    (
                        kind,
                        current_cap[kind],
                        ref[kind],
                    )
                )

        capacity_crosscheck_rows.append({
            "split_id": split_id,
            "role": role,
            "positive_recomputed": current_cap["positive"],
            "positive_reference": ref["positive"],
            "hard_recomputed": current_cap["hard"],
            "hard_reference": ref["hard"],
            "medium_recomputed": current_cap["medium"],
            "medium_reference": ref["medium"],
            "easy_recomputed": current_cap["easy"],
            "easy_reference": ref["easy"],
            "gate": (
                "PASS"
                if not mismatches
                else "FAIL"
            ),
        })

        if mismatches:
            die(
                f"{split_id} {role}: "
                f"capacity mismatch "
                f"{mismatches}"
            )

        if role == "train":
            positive_target = (
                current_cap["positive"]
            )

        elif role == "val":
            positive_target = min(
                current_cap["positive"],
                VAL_POS_CAP,
            )

        elif role == "test":
            positive_target = min(
                current_cap["positive"],
                TEST_POS_CAP,
            )

        else:
            raise RuntimeError(role)

        if positive_target <= 0:
            die(
                f"{split_id} {role}: "
                "zero positive capacity"
            )

        negative_target = (
            positive_target
        )

        neg_caps = {
            k: current_cap[k]
            for k in NEG_ORDER
        }

        planned_mix, actual_mix = (
            capacity_aware_negative_allocation(
                negative_target,
                neg_caps,
            )
        )

        pair_seed = (
            split_seed
            + PROTOCOL_OFFSET[protocol]
            + ROLE_OFFSET[role]
        )

        positive_pairs = (
            sample_positive_pairs(
                records,
                positive_target,
                pair_seed + 1,
            )
        )

        negative_reservoir, seen_caps = (
            reservoir_negative_sample(
                records,
                actual_mix,
                pair_seed + 2,
            )
        )

        for kind in NEG_ORDER:
            if (
                seen_caps.get(kind, 0)
                != current_cap[kind]
            ):
                die(
                    f"{split_id} {role}: "
                    f"{kind} enumeration "
                    f"capacity mismatch: "
                    f"{seen_caps.get(kind,0)} "
                    f"vs {current_cap[kind]}"
                )

        pair_rows = []

        for a, b in positive_pairs:
            pair_rows.append(
                pair_row(
                    split_id,
                    protocol,
                    split_seed,
                    role,
                    pair_seed,
                    a,
                    b,
                    "positive",
                )
            )

        for kind in NEG_ORDER:
            for a, b in negative_reservoir[kind]:
                pair_rows.append(
                    pair_row(
                        split_id,
                        protocol,
                        split_seed,
                        role,
                        pair_seed,
                        a,
                        b,
                        kind,
                    )
                )

        pair_rows.sort(
            key=lambda r: r["pair_id"]
        )

        audit = audit_pair_rows(
            pair_rows,
            role,
        )

        bad_fields = [
            "self_pairs",
            "duplicate_pairs",
            "duplicate_pair_ids",
            "label_errors",
            "type_errors",
            "role_errors",
            "teacher_scope_errors",
        ]

        bad_total = sum(
            audit[x]
            for x in bad_fields
        )

        if bad_total != 0:
            die(
                f"{split_id} {role}: "
                f"pair integrity failed "
                f"{audit}"
            )

        expected_total = (
            positive_target
            + negative_target
        )

        if audit["n_pairs"] != expected_total:
            die(
                f"{split_id} {role}: "
                f"expected {expected_total} "
                f"pairs, observed "
                f"{audit['n_pairs']}"
            )

        if audit["positive"] != positive_target:
            die(
                f"{split_id} {role}: "
                "positive count mismatch"
            )

        for kind in NEG_ORDER:
            if audit[kind] != actual_mix[kind]:
                die(
                    f"{split_id} {role}: "
                    f"{kind} count mismatch"
                )

        artifacts[split_id]["roles"][
            role
        ] = {
            "rows": pair_rows,
            "pair_seed": pair_seed,
            "positive_pool": positive_target,
            "negative_pool": negative_target,
            "planned_mix": planned_mix,
            "actual_mix": actual_mix,
        }

        summary_rows.append({
            "split_id": split_id,
            "protocol": protocol,
            "split_seed": split_seed,
            "role": role,
            "n_genomes": len(records),
            "positive_capacity": current_cap["positive"],
            "positive_used": positive_target,
            "negative_used": negative_target,
            "total_pairs_written": len(pair_rows),
            "hard_capacity": current_cap["hard"],
            "medium_capacity": current_cap["medium"],
            "easy_capacity": current_cap["easy"],
            "hard_used": actual_mix["hard"],
            "medium_used": actual_mix["medium"],
            "easy_used": actual_mix["easy"],
            "pair_sampling_seed": pair_seed,
        })

        for kind in NEG_ORDER:
            redistribution_rows.append({
                "split_id": split_id,
                "protocol": protocol,
                "role": role,
                "negative_type": kind,
                "capacity": current_cap[kind],
                "planned_count": planned_mix[kind],
                "actual_count": actual_mix[kind],
                "delta_from_plan": (
                    actual_mix[kind]
                    - planned_mix[kind]
                ),
                "actual_fraction_of_negatives": (
                    f"{actual_mix[kind] / negative_target:.12g}"
                ),
            })

        integrity_rows.append({
            "split_id": split_id,
            "protocol": protocol,
            "role": role,
            **audit,
            "gate": "PASS",
        })

        print(
            f"{role:5s} "
            f"genomes={len(records):4d} "
            f"pos={positive_target:5d} "
            f"neg={negative_target:5d} "
            f"H/M/E="
            f"{actual_mix['hard']}/"
            f"{actual_mix['medium']}/"
            f"{actual_mix['easy']} "
            f"PASS"
        )


# ------------------------------------------------------------
# Cross-role pair leakage audit
# ------------------------------------------------------------

cross_role_rows = []

for split_id, payload in artifacts.items():

    pair_sets = {}

    for role in (
        "train",
        "val",
        "test",
    ):
        pair_sets[role] = {
            (
                r["accession_a"],
                r["accession_b"],
            )
            for r in payload["roles"][
                role
            ]["rows"]
        }

    for a, b in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = (
            pair_sets[a]
            & pair_sets[b]
        )

        cross_role_rows.append({
            "split_id": split_id,
            "role_a": a,
            "role_b": b,
            "pair_overlap": len(overlap),
            "expected": 0,
            "gate": (
                "PASS"
                if len(overlap) == 0
                else "FAIL"
            ),
        })

        if overlap:
            die(
                f"{split_id}: pair overlap "
                f"between {a} and {b}"
            )


# ------------------------------------------------------------
# Only now create outputs
# ------------------------------------------------------------

if OUTDIR.exists():
    die(
        f"Output directory exists: "
        f"{OUTDIR}"
    )

AUDIT_DIR = OUTDIR / "audits"

OUTDIR.mkdir(
    parents=True,
    exist_ok=False,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

output_files = []

for split_id in sorted(artifacts):

    payload = artifacts[split_id]

    split_out = (
        OUTDIR / split_id
    )

    split_out.mkdir()

    # Train unique frozen pool
    train_path = (
        split_out
        / "train_unique_pair_pool.tsv"
    )

    write_tsv(
        train_path,
        payload["roles"]["train"]["rows"],
        PAIR_FIELDS,
    )

    output_files.append(train_path)

    # Frozen validation
    val_path = (
        split_out
        / "val_pairs.tsv"
    )

    write_tsv(
        val_path,
        payload["roles"]["val"]["rows"],
        PAIR_FIELDS,
    )

    output_files.append(val_path)

    # Frozen test
    test_path = (
        split_out
        / "test_pairs.tsv"
    )

    write_tsv(
        test_path,
        payload["roles"]["test"]["rows"],
        PAIR_FIELDS,
    )

    output_files.append(test_path)

    train_meta = (
        payload["roles"]["train"]
    )

    sampler_config = {
        "split_id": split_id,
        "protocol": payload["protocol"],
        "split_seed": payload["split_seed"],
        "train_epoch_examples": (
            TRAIN_EPOCH_EXAMPLES
        ),
        "positive_examples_per_epoch": (
            TRAIN_POS_PER_EPOCH
        ),
        "negative_examples_per_epoch": (
            TRAIN_NEG_PER_EPOCH
        ),
        "positive_unique_pool_size": (
            train_meta["positive_pool"]
        ),
        "negative_unique_pool_size": (
            train_meta["negative_pool"]
        ),
        "positive_sampling": (
            "uniform_with_replacement_from_"
            "frozen_unique_positive_pool"
        ),
        "negative_sampling": (
            "stratified_with_replacement_from_"
            "frozen_unique_negative_pool"
        ),
        "negative_target_weights": (
            NEG_WEIGHTS
        ),
        "negative_actual_unique_pool_counts": (
            train_meta["actual_mix"]
        ),
        "teacher_scope": (
            "train_unique_pair_pool_only"
        ),
        "validation_teacher_allowed": False,
        "test_teacher_allowed": False,
        "note": (
            "100000 examples/epoch are generated "
            "from frozen unique pools; "
            "100000 does not mean 100000 unique pairs."
        ),
    }

    config_path = (
        split_out
        / "train_sampler_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sampler_config,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    output_files.append(config_path)


# ------------------------------------------------------------
# Global audits
# ------------------------------------------------------------

summary_path = (
    AUDIT_DIR
    / "pair_summary.tsv"
)

write_tsv(
    summary_path,
    summary_rows,
    [
        "split_id",
        "protocol",
        "split_seed",
        "role",
        "n_genomes",
        "positive_capacity",
        "positive_used",
        "negative_used",
        "total_pairs_written",
        "hard_capacity",
        "medium_capacity",
        "easy_capacity",
        "hard_used",
        "medium_used",
        "easy_used",
        "pair_sampling_seed",
    ],
)

output_files.append(summary_path)

redistribution_path = (
    AUDIT_DIR
    / "negative_redistribution.tsv"
)

write_tsv(
    redistribution_path,
    redistribution_rows,
    [
        "split_id",
        "protocol",
        "role",
        "negative_type",
        "capacity",
        "planned_count",
        "actual_count",
        "delta_from_plan",
        "actual_fraction_of_negatives",
    ],
)

output_files.append(
    redistribution_path
)

integrity_path = (
    AUDIT_DIR
    / "pair_integrity_audit.tsv"
)

write_tsv(
    integrity_path,
    integrity_rows,
    [
        "split_id",
        "protocol",
        "role",
        "n_pairs",
        "self_pairs",
        "duplicate_pairs",
        "duplicate_pair_ids",
        "label_errors",
        "type_errors",
        "role_errors",
        "teacher_scope_errors",
        "positive",
        "hard",
        "medium",
        "easy",
        "gate",
    ],
)

output_files.append(
    integrity_path
)

cross_role_path = (
    AUDIT_DIR
    / "cross_role_pair_overlap.tsv"
)

write_tsv(
    cross_role_path,
    cross_role_rows,
    [
        "split_id",
        "role_a",
        "role_b",
        "pair_overlap",
        "expected",
        "gate",
    ],
)

output_files.append(
    cross_role_path
)

capacity_path = (
    AUDIT_DIR
    / "capacity_crosscheck.tsv"
)

write_tsv(
    capacity_path,
    capacity_crosscheck_rows,
    [
        "split_id",
        "role",
        "positive_recomputed",
        "positive_reference",
        "hard_recomputed",
        "hard_reference",
        "medium_recomputed",
        "medium_reference",
        "easy_recomputed",
        "easy_reference",
        "gate",
    ],
)

output_files.append(
    capacity_path
)


# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

global_config = {
    "pair_version": (
        "pair_manifests_v1_20260916"
    ),
    "split_root": str(SPLIT_ROOT),
    "split_root_report_sha256": (
        sha256_file(
            SPLIT_ROOT
            / "SPLIT_GENERATION_REPORT.txt"
        )
    ),
    "provenance_reaudit": (
        str(REAUDIT_REPORT)
    ),
    "provenance_reaudit_sha256": (
        sha256_file(
            REAUDIT_REPORT
        )
    ),
    "positive_definition": (
        "same GTDB species cluster"
    ),
    "hard_negative_definition": (
        "same genus, different species cluster"
    ),
    "medium_negative_definition": (
        "same family, different genus"
    ),
    "easy_negative_definition": (
        "different family"
    ),
    "negative_target_mix": (
        NEG_WEIGHTS
    ),
    "negative_shortage_policy": (
        "clamp each target to unique capacity, "
        "then redistribute remaining deficit "
        "to available strata in hardness order "
        "hard -> medium -> easy; "
        "all realized counts are audited"
    ),
    "train_policy": {
        "unique_positive_pool": (
            "all available unique positives"
        ),
        "unique_negative_pool_size": (
            "equal to unique positive pool"
        ),
        "examples_per_epoch": (
            TRAIN_EPOCH_EXAMPLES
        ),
        "positive_per_epoch": (
            TRAIN_POS_PER_EPOCH
        ),
        "negative_per_epoch": (
            TRAIN_NEG_PER_EPOCH
        ),
        "sampling_with_replacement": True,
    },
    "validation_policy": {
        "positive_cap": VAL_POS_CAP,
        "negative_ratio": "1:1",
        "teacher_allowed": False,
    },
    "test_policy": {
        "positive_cap": TEST_POS_CAP,
        "negative_ratio": "1:1",
        "teacher_allowed": False,
    },
    "teacher_policy": (
        "teacher supervision is permitted "
        "only for train_unique_pair_pool.tsv"
    ),
    "external_policy": (
        "B1/B2 external pair generation is deferred "
        "until model and threshold are frozen"
    ),
}

config_path = (
    OUTDIR / "pair_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        global_config,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

output_files.append(
    config_path
)


# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

report_path = (
    OUTDIR
    / "PAIR_GENERATION_REPORT.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "HGV-V3 PAIR MANIFEST GENERATION\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write("STATUS: PASS\n\n")

    f.write("PAIR DEFINITIONS\n")
    f.write(
        "positive = same GTDB species cluster\n"
    )
    f.write(
        "hard = same genus, different species\n"
    )
    f.write(
        "medium = same family, different genus\n"
    )
    f.write(
        "easy = different family\n\n"
    )

    f.write("TRAINING\n")
    f.write(
        "Unique positive pool: all available\n"
    )
    f.write(
        "Unique negative pool: 1:1 with positives\n"
    )
    f.write(
        "Training examples/epoch: 100000\n"
    )
    f.write(
        "Positive/negative per epoch: 50000/50000\n"
    )
    f.write(
        "Epoch sampling uses replacement from "
        "the frozen unique pools.\n\n"
    )

    f.write("VALIDATION/TEST\n")
    f.write(
        "Validation positive cap: 20000\n"
    )
    f.write(
        "Test positive cap: 30000\n"
    )
    f.write(
        "All current split capacities are below "
        "these caps; actual data determine counts.\n\n"
    )

    f.write("NEGATIVE MIX\n")
    f.write(
        "Target = 50% hard / 30% medium / 20% easy.\n"
    )
    f.write(
        "When unique capacity is insufficient, "
        "the deficit is redistributed without "
        "duplicating evaluation pairs; "
        "actual mix is recorded.\n\n"
    )

    f.write("SUMMARY\n")

    for r in summary_rows:
        f.write(
            f"{r['split_id']} "
            f"{r['role']}: "
            f"pos={r['positive_used']} "
            f"neg={r['negative_used']} "
            f"H/M/E="
            f"{r['hard_used']}/"
            f"{r['medium_used']}/"
            f"{r['easy_used']} "
            f"total={r['total_pairs_written']}\n"
        )

    f.write(
        "\nExternal B1/B2 remain untouched "
        "at this stage.\n"
    )

    f.write(
        "\nPAIR_GENERATION=PASS\n"
    )

output_files.append(
    report_path
)


# ------------------------------------------------------------
# Freeze hashes
# ------------------------------------------------------------

sha_path = (
    OUTDIR / "SHA256SUMS.txt"
)

with open(
    sha_path,
    "w",
    encoding="utf-8",
) as f:

    for path in sorted(
        output_files,
        key=lambda p: str(p),
    ):
        rel = path.relative_to(
            OUTDIR
        )

        f.write(
            f"{sha256_file(path)}  "
            f"{rel}\n"
        )


# ------------------------------------------------------------
# Console
# ------------------------------------------------------------

print()
print("=" * 78)
print("ALL PAIR GENERATION GATES PASSED")
print("=" * 78)

for r in summary_rows:
    print(
        f"{r['split_id']:22s} "
        f"{r['role']:5s} "
        f"pos={r['positive_used']:5d} "
        f"neg={r['negative_used']:5d} "
        f"H/M/E="
        f"{r['hard_used']}/"
        f"{r['medium_used']}/"
        f"{r['easy_used']}"
    )

print()
print(f"Output: {OUTDIR}")
print()
print("PAIR_GENERATION=PASS")
