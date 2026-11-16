#!/usr/bin/env python3

import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# HGV-V3 final 3154 freeze
#
# READ-ONLY INPUTS
# - updated_primary_manifest_round1.tsv
# - master_manifest_v3_sequence_audited.tsv
# - final_accession_fasta_map.tsv
# - round2_replacement_targets.tsv
# - sequence_qc_failures_detailed.tsv
# - duplicate_members_detailed.tsv
#
# NEW OUTPUT ONLY
# - 01_manifest/final_3154_freeze_v1_20260916/
#
# Existing files are NEVER overwritten.
# ============================================================

BASE = Path("HGV_V3_revision_20260915")

PRIMARY = BASE / "01_manifest/replacement_round1_20260916/updated_primary_manifest_round1.tsv"

AUDITED = BASE / "02_genomes/sequence_audit_round1/master_manifest_v3_sequence_audited.tsv"

FASTA_MAP = BASE / "02_genomes/sequence_audit_round1/final_accession_fasta_map.tsv"

TARGETS = BASE / "02_genomes/sequence_audit_diagnosis_round1/round2_replacement_targets.tsv"

QC_FAIL = BASE / "02_genomes/sequence_audit_diagnosis_round1/sequence_qc_failures_detailed.tsv"

DUP_MEMBERS = BASE / "02_genomes/sequence_audit_diagnosis_round1/duplicate_members_detailed.tsv"

OUTDIR = BASE / "01_manifest/final_3154_freeze_v1_20260916"

EXPECTED_INPUT = 3200
EXPECTED_TARGETS = 46
EXPECTED_RETAINED = 3154

EXPECTED_PRE = {
    "A": 2400,
    "B1": 400,
    "B2_seen_genus": 200,
    "B2_unseen_genus": 200,
}

EXPECTED_FINAL = {
    "A": 2374,
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193,
}

EXPECTED_QC_FAILURES = 36
EXPECTED_DUPLICATE_EXCLUSIONS = 11
EXPECTED_QC_DUP_OVERLAP = 1
EXPECTED_ADDITIONAL_DUPLICATE_EXCLUSIONS = 10
EXPECTED_DUPLICATE_GROUPS_BEFORE = 11
EXPECTED_DUPLICATE_MEMBERS_BEFORE = 22


def die(msg):
    print()
    print("FINAL_FREEZE=FAIL")
    print(msg)
    sys.exit(2)


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_tsv(path):
    if not path.is_file():
        die(f"Missing required input: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            die(f"No TSV header detected: {path}")
        rows = list(reader)

    return list(reader.fieldnames), rows


def normalize_accession(value):
    if value is None:
        return ""
    value = str(value).strip()
    m = re.search(r"(GC[AF]_[0-9]+\.[0-9]+)", value)
    return m.group(1) if m else ""


def detect_accession_col(fields, rows, label):
    preferred = [
        "assembly_accession",
        "accession",
        "ncbi_accession",
        "gtdb_accession",
        "replacement_accession",
        "target_accession",
        "current_accession",
        "failed_accession",
    ]

    lower_map = {x.lower(): x for x in fields}

    for p in preferred:
        if p in lower_map:
            col = lower_map[p]
            n = sum(bool(normalize_accession(r.get(col, ""))) for r in rows[:500])
            if n > 0:
                return col

    best_col = None
    best_score = -1

    for col in fields:
        score = sum(bool(normalize_accession(r.get(col, ""))) for r in rows[:500])
        if score > best_score:
            best_score = score
            best_col = col

    if best_col is None or best_score <= 0:
        die(f"Cannot detect accession column in {label}")

    return best_col


def build_unique_index(rows, acc_col, label):
    out = {}
    duplicates = []

    for r in rows:
        acc = normalize_accession(r.get(acc_col, ""))
        if not acc:
            die(f"Invalid accession in {label}: {r.get(acc_col, '')}")

        if acc in out:
            duplicates.append(acc)
        else:
            out[acc] = r

    if duplicates:
        die(
            f"Duplicate accessions in {label}: "
            f"{len(set(duplicates))}; examples={sorted(set(duplicates))[:10]}"
        )

    return out


def build_accession_set(rows, acc_col):
    s = set()
    for r in rows:
        acc = normalize_accession(r.get(acc_col, ""))
        if acc:
            s.add(acc)
    return s


def detect_path_col(fields, rows):
    preferred = [
        "fasta_path",
        "genomic_fasta_path",
        "genome_fasta_path",
        "resolved_fasta_path",
        "local_fasta_path",
        "path",
    ]

    lower_map = {x.lower(): x for x in fields}

    for p in preferred:
        if p in lower_map:
            col = lower_map[p]
            if any(str(r.get(col, "")).strip() for r in rows[:100]):
                return col

    candidates = []
    for col in fields:
        lc = col.lower()
        if "fasta" in lc and "path" in lc:
            candidates.append(col)

    if candidates:
        return candidates[0]

    return None


def detect_canonical_hash_col(fields, rows):
    candidates = []

    for col in fields:
        lc = col.lower()
        score = 0

        if "canonical" in lc and "hash" in lc:
            score += 100
        elif "canonical" in lc and "sha" in lc:
            score += 90
        elif "sequence" in lc and "canonical" in lc:
            score += 80
        elif "sequence_hash" in lc:
            score += 40

        if score > 0:
            nonempty = sum(bool(str(r.get(col, "")).strip()) for r in rows)
            candidates.append((score, nonempty, col))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][2]


def merge_row(base_row, other_row, other_acc_col, prefix):
    if other_row is None:
        return

    for k, v in other_row.items():
        if k == other_acc_col:
            continue

        v = "" if v is None else str(v)

        if k not in base_row or str(base_row.get(k, "")).strip() == "":
            base_row[k] = v
            continue

        current = str(base_row.get(k, ""))

        if not v.strip() or current == v:
            continue

        alt = prefix + k

        if alt not in base_row or not str(base_row.get(alt, "")).strip():
            base_row[alt] = v


def cohort_from_row(row):
    values = []

    for k, v in row.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            values.append((str(k).lower(), s.lower()))

    joined = " | ".join(v for _, v in values)

    # B2 unseen genus must be checked before generic "seen genus"
    if (
        "b2_unseen" in joined
        or "b2-unseen" in joined
        or "b2 unseen" in joined
        or "unseen_genus" in joined
        or "unseen-genus" in joined
        or "unseen genus" in joined
    ):
        return "B2_unseen_genus"

    if (
        "b2_seen" in joined
        or "b2-seen" in joined
        or "b2 seen" in joined
        or "seen_genus" in joined
        or "seen-genus" in joined
        or "genus seen" in joined
    ):
        return "B2_seen_genus"

    # Generic B1
    if (
        re.search(r"(^|[^a-z0-9])b1([^a-z0-9]|$)", joined)
        or "external_b1" in joined
        or "external-b1" in joined
        or "external b1" in joined
    ):
        return "B1"

    # Generic B2 rows: try dedicated subtype fields
    if (
        re.search(r"(^|[^a-z0-9])b2([^a-z0-9]|$)", joined)
        or "external_b2" in joined
        or "external-b2" in joined
        or "external b2" in joined
    ):
        for k, v in values:
            if any(x in k for x in ("subtype", "stratum", "genus_status", "b2")):
                if "unseen" in v:
                    return "B2_unseen_genus"
                if "seen" in v:
                    return "B2_seen_genus"
        return "B2_UNKNOWN"

    # Development A
    for k, v in values:
        vv = v.replace("-", "_").replace(" ", "_")

        if vv in {
            "a",
            "dataset_a",
            "development",
            "development_a",
            "development_cohort",
            "refseq_development",
        }:
            return "A"

        if (
            ("cohort" in k or "dataset" in k or "source" in k)
            and "development" in vv
        ):
            return "A"

    return "UNKNOWN"


def broad_cohort(c):
    if c == "A":
        return "A"
    if c == "B1":
        return "B1"
    if c.startswith("B2_"):
        return "B2"
    return c


def detect_tax_col(fields, rank):
    prefs = {
        "species_cluster": [
            "gtdb_species_cluster",
            "species_cluster",
            "gtdb_species_cluster_id",
            "species_cluster_id",
        ],
        "species": [
            "gtdb_species",
            "species",
        ],
        "genus": [
            "gtdb_genus",
            "genus",
        ],
        "family": [
            "gtdb_family",
            "family",
        ],
    }

    lower_map = {x.lower(): x for x in fields}

    for p in prefs[rank]:
        if p in lower_map:
            return lower_map[p]

    if rank == "species_cluster":
        for col in fields:
            lc = col.lower()
            if "species" in lc and "cluster" in lc:
                return col

    for col in fields:
        lc = col.lower()
        if rank in lc and "ncbi" not in lc:
            return col

    return None


def collect_fieldnames(rows):
    fields = []
    seen = set()

    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    return fields


def write_tsv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def target_reason_text(row, acc_col):
    pieces = []

    for k, v in row.items():
        if k == acc_col:
            continue

        if v is None or not str(v).strip():
            continue

        lk = k.lower()

        if any(
            token in lk
            for token in (
                "reason",
                "failure",
                "issue",
                "category",
                "action",
                "status",
                "qc",
                "duplicate",
            )
        ):
            pieces.append(f"{k}={str(v).strip()}")

    return "; ".join(pieces)


print("=" * 72)
print("HGV-V3 FINAL 3154 FREEZE")
print("=" * 72)

# ------------------------------------------------------------
# 1. Load inputs
# ------------------------------------------------------------

primary_fields, primary_rows = read_tsv(PRIMARY)
audit_fields, audit_rows = read_tsv(AUDITED)
map_fields, map_rows = read_tsv(FASTA_MAP)
target_fields, target_rows = read_tsv(TARGETS)
qc_fields, qc_rows = read_tsv(QC_FAIL)
dup_fields, dup_rows = read_tsv(DUP_MEMBERS)

primary_acc_col = detect_accession_col(primary_fields, primary_rows, "PRIMARY")
audit_acc_col = detect_accession_col(audit_fields, audit_rows, "AUDITED")
map_acc_col = detect_accession_col(map_fields, map_rows, "FASTA_MAP")
target_acc_col = detect_accession_col(target_fields, target_rows, "TARGETS")
qc_acc_col = detect_accession_col(qc_fields, qc_rows, "QC_FAIL")
dup_acc_col = detect_accession_col(dup_fields, dup_rows, "DUP_MEMBERS")

print()
print("[Detected accession columns]")
print(f"PRIMARY      : {primary_acc_col}")
print(f"AUDITED      : {audit_acc_col}")
print(f"FASTA_MAP    : {map_acc_col}")
print(f"TARGETS      : {target_acc_col}")
print(f"QC_FAIL      : {qc_acc_col}")
print(f"DUP_MEMBERS  : {dup_acc_col}")

primary_idx = build_unique_index(primary_rows, primary_acc_col, "PRIMARY")
audit_idx = build_unique_index(audit_rows, audit_acc_col, "AUDITED")
map_idx = build_unique_index(map_rows, map_acc_col, "FASTA_MAP")
target_idx = build_unique_index(target_rows, target_acc_col, "TARGETS")

qc_set = build_accession_set(qc_rows, qc_acc_col)
dup_member_set = build_accession_set(dup_rows, dup_acc_col)

# ------------------------------------------------------------
# 2. Core input gates
# ------------------------------------------------------------

print()
print("[Input counts]")
print(f"Primary manifest          : {len(primary_idx)}")
print(f"Sequence audited manifest : {len(audit_idx)}")
print(f"FASTA map                 : {len(map_idx)}")
print(f"Final exclusion targets   : {len(target_idx)}")
print(f"Sequence-QC failures      : {len(qc_set)}")
print(f"Duplicate member records  : {len(dup_member_set)}")

if len(primary_idx) != EXPECTED_INPUT:
    die(f"Expected {EXPECTED_INPUT} primary genomes; observed {len(primary_idx)}")

if len(audit_idx) != EXPECTED_INPUT:
    die(f"Expected {EXPECTED_INPUT} audited genomes; observed {len(audit_idx)}")

if len(map_idx) != EXPECTED_INPUT:
    die(f"Expected {EXPECTED_INPUT} FASTA map entries; observed {len(map_idx)}")

if len(target_idx) != EXPECTED_TARGETS:
    die(f"Expected {EXPECTED_TARGETS} exclusion targets; observed {len(target_idx)}")

primary_set = set(primary_idx)
audit_set = set(audit_idx)
map_set = set(map_idx)
target_set = set(target_idx)

if primary_set != audit_set:
    print(f"PRIMARY-only accessions: {len(primary_set - audit_set)}")
    print(f"AUDIT-only accessions  : {len(audit_set - primary_set)}")
    die("PRIMARY and sequence-audited accession sets differ")

if primary_set != map_set:
    print(f"PRIMARY-only accessions: {len(primary_set - map_set)}")
    print(f"MAP-only accessions    : {len(map_set - primary_set)}")
    die("PRIMARY and FASTA-map accession sets differ")

if not target_set.issubset(primary_set):
    missing = sorted(target_set - primary_set)
    die(f"Exclusion targets not found in primary manifest: {missing[:20]}")

# ------------------------------------------------------------
# 3. Reconstruct QC / redundancy accounting
# ------------------------------------------------------------

qc_targets = target_set & qc_set
dup_targets = target_set & dup_member_set
qc_dup_overlap = qc_targets & dup_targets
additional_dup_targets = dup_targets - qc_targets

print()
print("[Exclusion accounting]")
print(f"QC target exclusions          : {len(qc_targets)}")
print(f"Duplicate target exclusions   : {len(dup_targets)}")
print(f"QC + duplicate overlap        : {len(qc_dup_overlap)}")
print(f"Additional duplicate excludes : {len(additional_dup_targets)}")
print(f"Unique final excludes         : {len(target_set)}")

if len(qc_targets) != EXPECTED_QC_FAILURES:
    die(
        f"Expected {EXPECTED_QC_FAILURES} QC-failure targets; "
        f"observed {len(qc_targets)}"
    )

if len(dup_targets) != EXPECTED_DUPLICATE_EXCLUSIONS:
    die(
        f"Expected {EXPECTED_DUPLICATE_EXCLUSIONS} duplicate exclusions; "
        f"observed {len(dup_targets)}"
    )

if len(qc_dup_overlap) != EXPECTED_QC_DUP_OVERLAP:
    die(
        f"Expected QC/duplicate overlap={EXPECTED_QC_DUP_OVERLAP}; "
        f"observed {len(qc_dup_overlap)}"
    )

if len(additional_dup_targets) != EXPECTED_ADDITIONAL_DUPLICATE_EXCLUSIONS:
    die(
        f"Expected additional duplicate exclusions="
        f"{EXPECTED_ADDITIONAL_DUPLICATE_EXCLUSIONS}; "
        f"observed {len(additional_dup_targets)}"
    )

# ------------------------------------------------------------
# 4. Merge primary + sequence-audit information
# ------------------------------------------------------------

merged = {}

for acc in sorted(primary_idx):
    row = dict(primary_idx[acc])

    row["assembly_accession_final"] = acc

    merge_row(
        row,
        audit_idx.get(acc),
        audit_acc_col,
        "audit__",
    )

    merge_row(
        row,
        map_idx.get(acc),
        map_acc_col,
        "map__",
    )

    merged[acc] = row

# ------------------------------------------------------------
# 5. Standardize FASTA path
# ------------------------------------------------------------

map_path_col = detect_path_col(map_fields, map_rows)

if map_path_col is None:
    die("Could not detect FASTA path column in final_accession_fasta_map.tsv")

print()
print(f"[Detected FASTA path column] {map_path_col}")

missing_fasta = []
fasta_paths = []

for acc, row in merged.items():
    p = str(map_idx[acc].get(map_path_col, "")).strip()

    if not p:
        missing_fasta.append(acc)
        continue

    pp = Path(p)

    if not pp.is_absolute():
        pp = BASE / pp

    pp = pp.resolve()

    row["fasta_path"] = str(pp)
    fasta_paths.append(str(pp))

    if not pp.is_file():
        missing_fasta.append(acc)

print(f"Missing/nonexistent FASTA : {len(missing_fasta)}")

if missing_fasta:
    print("Examples:")
    for x in missing_fasta[:20]:
        print("  ", x)
    die("One or more logical genomes do not map to an existing FASTA")

if len(set(fasta_paths)) != EXPECTED_INPUT:
    die(
        f"Expected {EXPECTED_INPUT} unique FASTA paths; "
        f"observed {len(set(fasta_paths))}"
    )

# ------------------------------------------------------------
# 6. Infer logical cohort / subtype
# ------------------------------------------------------------

unknown_cohort = []

for acc, row in merged.items():
    c = cohort_from_row(row)
    row["final_cohort"] = c

    if c in ("UNKNOWN", "B2_UNKNOWN"):
        unknown_cohort.append(acc)

if unknown_cohort:
    print()
    print("Could not infer cohort/subtype for:")
    for acc in unknown_cohort[:30]:
        print("  ", acc)
    die(
        "Cohort inference failed. "
        "No output freeze was created."
    )

pre_counts = Counter(row["final_cohort"] for row in merged.values())

print()
print("[Pre-exclusion cohort counts]")

for c in ("A", "B1", "B2_seen_genus", "B2_unseen_genus"):
    print(f"{c:18s}: {pre_counts.get(c, 0)}")

for c, expected in EXPECTED_PRE.items():
    if pre_counts.get(c, 0) != expected:
        die(
            f"Pre-exclusion cohort gate failed for {c}: "
            f"expected {expected}, observed {pre_counts.get(c, 0)}"
        )

# ------------------------------------------------------------
# 7. Canonical sequence hash gate
# ------------------------------------------------------------

all_fields_after_merge = collect_fieldnames(list(merged.values()))
canonical_hash_col = detect_canonical_hash_col(
    all_fields_after_merge,
    list(merged.values()),
)

if canonical_hash_col is None:
    die(
        "Could not detect canonical sequence hash column in merged audit data. "
        "Final freeze requires the canonical sequence hash."
    )

print()
print(f"[Detected canonical hash column] {canonical_hash_col}")

for row in merged.values():
    row["canonical_sequence_hash"] = str(
        row.get(canonical_hash_col, "")
    ).strip()

missing_hash = [
    acc
    for acc, row in merged.items()
    if not row["canonical_sequence_hash"]
]

if missing_hash:
    die(
        f"Missing canonical sequence hash for {len(missing_hash)} genomes"
    )

hash_groups_before = defaultdict(list)

for acc, row in merged.items():
    hash_groups_before[row["canonical_sequence_hash"]].append(acc)

duplicate_groups_before = {
    h: members
    for h, members in hash_groups_before.items()
    if len(members) > 1
}

duplicate_members_before = sum(
    len(v) for v in duplicate_groups_before.values()
)

cross_cohort_duplicate_groups_before = 0

for h, members in duplicate_groups_before.items():
    cohorts = {
        broad_cohort(merged[a]["final_cohort"])
        for a in members
    }

    if len(cohorts) > 1:
        cross_cohort_duplicate_groups_before += 1

print()
print("[Canonical duplicate audit: before final exclusion]")
print(f"Duplicate groups       : {len(duplicate_groups_before)}")
print(f"Duplicate members      : {duplicate_members_before}")
print(f"Cross-cohort groups    : {cross_cohort_duplicate_groups_before}")

if len(duplicate_groups_before) != EXPECTED_DUPLICATE_GROUPS_BEFORE:
    die(
        f"Expected {EXPECTED_DUPLICATE_GROUPS_BEFORE} canonical duplicate groups; "
        f"observed {len(duplicate_groups_before)}"
    )

if duplicate_members_before != EXPECTED_DUPLICATE_MEMBERS_BEFORE:
    die(
        f"Expected {EXPECTED_DUPLICATE_MEMBERS_BEFORE} duplicate members; "
        f"observed {duplicate_members_before}"
    )

if cross_cohort_duplicate_groups_before != 0:
    die(
        f"Expected 0 cross-cohort canonical duplicate groups; "
        f"observed {cross_cohort_duplicate_groups_before}"
    )

# ------------------------------------------------------------
# 8. Apply FINAL exclusion targets
# ------------------------------------------------------------

retained = []
excluded = []

for acc in sorted(merged):
    row = dict(merged[acc])

    if acc in target_set:
        row["final_status"] = "excluded"

        reasons = []

        if acc in qc_targets:
            reasons.append("sequence_qc_failure")

        if acc in dup_targets:
            reasons.append("duplicate_redundancy_target")

        if not reasons:
            reasons.append("final_exclusion_target")

        row["final_exclusion_reason"] = ";".join(reasons)

        tr = target_idx[acc]

        detail = target_reason_text(tr, target_acc_col)
        row["target_reason_detail"] = detail

        for k, v in tr.items():
            if k == target_acc_col:
                continue
            row["target__" + k] = "" if v is None else str(v)

        excluded.append(row)

    else:
        row["final_status"] = "retained"
        row["final_exclusion_reason"] = ""
        row["target_reason_detail"] = ""
        retained.append(row)

print()
print("[Final retention]")
print(f"Retained : {len(retained)}")
print(f"Excluded : {len(excluded)}")

if len(retained) != EXPECTED_RETAINED:
    die(
        f"Expected {EXPECTED_RETAINED} retained genomes; "
        f"observed {len(retained)}"
    )

if len(excluded) != EXPECTED_TARGETS:
    die(
        f"Expected {EXPECTED_TARGETS} excluded genomes; "
        f"observed {len(excluded)}"
    )

# ------------------------------------------------------------
# 9. Final cohort gates
# ------------------------------------------------------------

final_counts = Counter(r["final_cohort"] for r in retained)

print()
print("[Final cohort counts]")

for c in ("A", "B1", "B2_seen_genus", "B2_unseen_genus"):
    print(f"{c:18s}: {final_counts.get(c, 0)}")

for c, expected in EXPECTED_FINAL.items():
    if final_counts.get(c, 0) != expected:
        die(
            f"Final cohort gate failed for {c}: "
            f"expected {expected}, observed {final_counts.get(c, 0)}"
        )

external_total = (
    final_counts["B1"]
    + final_counts["B2_seen_genus"]
    + final_counts["B2_unseen_genus"]
)

print(f"{'External total':18s}: {external_total}")

if external_total != 780:
    die(f"Expected external total 780; observed {external_total}")

# ------------------------------------------------------------
# 10. Final retained duplicate/hash leakage gate
# ------------------------------------------------------------

retained_hash_groups = defaultdict(list)

for row in retained:
    retained_hash_groups[
        row["canonical_sequence_hash"]
    ].append(row)

retained_duplicate_groups = {
    h: rows
    for h, rows in retained_hash_groups.items()
    if len(rows) > 1
}

retained_cross_cohort_duplicate_groups = 0

for h, rows in retained_duplicate_groups.items():
    cohorts = {
        broad_cohort(r["final_cohort"])
        for r in rows
    }
    if len(cohorts) > 1:
        retained_cross_cohort_duplicate_groups += 1

print()
print("[Canonical duplicate audit: retained 3154]")
print(f"Duplicate groups    : {len(retained_duplicate_groups)}")
print(f"Cross-cohort groups : {retained_cross_cohort_duplicate_groups}")

if retained_duplicate_groups:
    examples = list(retained_duplicate_groups.items())[:5]
    for h, rows in examples:
        print(h, [x["assembly_accession_final"] for x in rows])
    die("Canonical sequence duplicates remain after final exclusion")

if retained_cross_cohort_duplicate_groups != 0:
    die("Cross-cohort duplicate leakage remains")

# ------------------------------------------------------------
# 11. Final FASTA uniqueness gate
# ------------------------------------------------------------

retained_paths = [r["fasta_path"] for r in retained]

if len(set(retained_paths)) != EXPECTED_RETAINED:
    die(
        f"Retained FASTA paths are not unique: "
        f"{len(set(retained_paths))}/{EXPECTED_RETAINED}"
    )

# ------------------------------------------------------------
# 12. Detect taxonomy columns for final summary
# ------------------------------------------------------------

retained_fields = collect_fieldnames(retained)

species_cluster_col = detect_tax_col(
    retained_fields,
    "species_cluster",
)
species_col = detect_tax_col(
    retained_fields,
    "species",
)
genus_col = detect_tax_col(
    retained_fields,
    "genus",
)
family_col = detect_tax_col(
    retained_fields,
    "family",
)

print()
print("[Detected taxonomy columns]")
print(f"species_cluster : {species_cluster_col}")
print(f"species         : {species_col}")
print(f"genus           : {genus_col}")
print(f"family          : {family_col}")


def unique_nonempty(rows, col):
    if col is None:
        return set()

    return {
        str(r.get(col, "")).strip()
        for r in rows
        if str(r.get(col, "")).strip()
    }


all_species_clusters = unique_nonempty(
    retained,
    species_cluster_col,
)
all_species = unique_nonempty(
    retained,
    species_col,
)
all_genera = unique_nonempty(
    retained,
    genus_col,
)
all_families = unique_nonempty(
    retained,
    family_col,
)

A_rows = [
    r for r in retained
    if r["final_cohort"] == "A"
]

A_species_clusters = unique_nonempty(
    A_rows,
    species_cluster_col,
)
A_species = unique_nonempty(
    A_rows,
    species_col,
)
A_genera = unique_nonempty(
    A_rows,
    genus_col,
)
A_families = unique_nonempty(
    A_rows,
    family_col,
)

# ------------------------------------------------------------
# 13. ALL GATES PASSED -- now create NEW output directory
# ------------------------------------------------------------

if OUTDIR.exists():
    die(
        f"Output directory already exists: {OUTDIR}\n"
        "Refusing to overwrite a frozen result."
    )

OUTDIR.mkdir(parents=True, exist_ok=False)

# ------------------------------------------------------------
# 14. Write final master
# ------------------------------------------------------------

final_master = OUTDIR / "final_master_manifest_3154.tsv"

master_fields = collect_fieldnames(retained)

# Put key final fields at front where possible
preferred_front = [
    "assembly_accession_final",
    "final_cohort",
    "final_status",
    "fasta_path",
    "canonical_sequence_hash",
    "final_exclusion_reason",
]

ordered_master_fields = []

for x in preferred_front:
    if x in master_fields and x not in ordered_master_fields:
        ordered_master_fields.append(x)

for x in master_fields:
    if x not in ordered_master_fields:
        ordered_master_fields.append(x)

write_tsv(
    final_master,
    retained,
    ordered_master_fields,
)

# ------------------------------------------------------------
# 15. Exclusion ledger
# ------------------------------------------------------------

exclusion_ledger = OUTDIR / "final_exclusion_ledger_46.tsv"

excluded.sort(
    key=lambda r: (
        r.get("final_cohort", ""),
        r.get("assembly_accession_final", ""),
    )
)

write_tsv(
    exclusion_ledger,
    excluded,
)

# ------------------------------------------------------------
# 16. Cohort counts
# ------------------------------------------------------------

cohort_rows = []

for c in ("A", "B1", "B2_seen_genus", "B2_unseen_genus"):
    cohort_rows.append({
        "cohort": c,
        "pre_exclusion": pre_counts[c],
        "excluded": pre_counts[c] - final_counts[c],
        "final_retained": final_counts[c],
        "expected_final": EXPECTED_FINAL[c],
        "gate": "PASS" if final_counts[c] == EXPECTED_FINAL[c] else "FAIL",
    })

cohort_rows.append({
    "cohort": "External_total",
    "pre_exclusion": (
        pre_counts["B1"]
        + pre_counts["B2_seen_genus"]
        + pre_counts["B2_unseen_genus"]
    ),
    "excluded": 800 - external_total,
    "final_retained": external_total,
    "expected_final": 780,
    "gate": "PASS" if external_total == 780 else "FAIL",
})

cohort_rows.append({
    "cohort": "TOTAL",
    "pre_exclusion": EXPECTED_INPUT,
    "excluded": EXPECTED_TARGETS,
    "final_retained": EXPECTED_RETAINED,
    "expected_final": EXPECTED_RETAINED,
    "gate": "PASS",
})

write_tsv(
    OUTDIR / "final_cohort_counts.tsv",
    cohort_rows,
)

# ------------------------------------------------------------
# 17. Taxonomy summary
# ------------------------------------------------------------

taxonomy_rows = [
    {
        "scope": "all_retained",
        "genomes": len(retained),
        "species_clusters": len(all_species_clusters),
        "species_labels": len(all_species),
        "genera": len(all_genera),
        "families": len(all_families),
    },
    {
        "scope": "development_A",
        "genomes": len(A_rows),
        "species_clusters": len(A_species_clusters),
        "species_labels": len(A_species),
        "genera": len(A_genera),
        "families": len(A_families),
    },
]

write_tsv(
    OUTDIR / "final_taxonomy_counts.tsv",
    taxonomy_rows,
)

# ------------------------------------------------------------
# 18. Sequence/QC summary
# ------------------------------------------------------------

qc_summary_rows = [
    {
        "metric": "downloaded_unique_accessions",
        "value": EXPECTED_INPUT,
    },
    {
        "metric": "sequence_qc_failure_exclusions",
        "value": len(qc_targets),
    },
    {
        "metric": "duplicate_exclusion_targets_total",
        "value": len(dup_targets),
    },
    {
        "metric": "qc_duplicate_overlap",
        "value": len(qc_dup_overlap),
    },
    {
        "metric": "additional_duplicate_exclusions",
        "value": len(additional_dup_targets),
    },
    {
        "metric": "unique_final_exclusions",
        "value": len(target_set),
    },
    {
        "metric": "canonical_duplicate_groups_before",
        "value": len(duplicate_groups_before),
    },
    {
        "metric": "canonical_duplicate_members_before",
        "value": duplicate_members_before,
    },
    {
        "metric": "cross_cohort_duplicate_groups_before",
        "value": cross_cohort_duplicate_groups_before,
    },
    {
        "metric": "canonical_duplicate_groups_after",
        "value": len(retained_duplicate_groups),
    },
    {
        "metric": "cross_cohort_duplicate_groups_after",
        "value": retained_cross_cohort_duplicate_groups,
    },
    {
        "metric": "final_retained",
        "value": len(retained),
    },
]

write_tsv(
    OUTDIR / "final_sequence_qc_summary.tsv",
    qc_summary_rows,
)

# ------------------------------------------------------------
# 19. Leakage audit
# ------------------------------------------------------------

accession_duplicate_count = (
    len(retained)
    - len({
        r["assembly_accession_final"]
        for r in retained
    })
)

fasta_path_duplicate_count = (
    len(retained)
    - len(set(retained_paths))
)

leakage_rows = [
    {
        "check": "retained_accession_duplicates",
        "observed": accession_duplicate_count,
        "expected": 0,
        "gate": "PASS" if accession_duplicate_count == 0 else "FAIL",
    },
    {
        "check": "retained_missing_fasta",
        "observed": 0,
        "expected": 0,
        "gate": "PASS",
    },
    {
        "check": "retained_duplicate_fasta_paths",
        "observed": fasta_path_duplicate_count,
        "expected": 0,
        "gate": "PASS" if fasta_path_duplicate_count == 0 else "FAIL",
    },
    {
        "check": "retained_canonical_duplicate_groups",
        "observed": len(retained_duplicate_groups),
        "expected": 0,
        "gate": "PASS" if len(retained_duplicate_groups) == 0 else "FAIL",
    },
    {
        "check": "retained_cross_cohort_canonical_duplicate_groups",
        "observed": retained_cross_cohort_duplicate_groups,
        "expected": 0,
        "gate": "PASS" if retained_cross_cohort_duplicate_groups == 0 else "FAIL",
    },
]

write_tsv(
    OUTDIR / "final_leakage_audit.tsv",
    leakage_rows,
)

# ------------------------------------------------------------
# 20. Retained accession list
# ------------------------------------------------------------

with open(
    OUTDIR / "final_retained_accessions_3154.txt",
    "w",
    encoding="utf-8",
) as f:
    for r in retained:
        f.write(r["assembly_accession_final"] + "\n")

# ------------------------------------------------------------
# 21. Input provenance hashes
# ------------------------------------------------------------

input_files = [
    PRIMARY,
    AUDITED,
    FASTA_MAP,
    TARGETS,
    QC_FAIL,
    DUP_MEMBERS,
]

input_hash_rows = []

for p in input_files:
    input_hash_rows.append({
        "file": str(p),
        "sha256": sha256_file(p),
    })

write_tsv(
    OUTDIR / "final_freeze_input_sha256.tsv",
    input_hash_rows,
)

# ------------------------------------------------------------
# 22. Final manifest SHA256
# ------------------------------------------------------------

master_sha256 = sha256_file(final_master)

with open(
    OUTDIR / "final_master_manifest_3154.tsv.sha256",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        f"{master_sha256}  {final_master.name}\n"
    )

# ------------------------------------------------------------
# 23. Human-readable report
# ------------------------------------------------------------

report = OUTDIR / "FINAL_FREEZE_REPORT.txt"

with open(report, "w", encoding="utf-8") as f:
    f.write("HGV-V3 FINAL 3154 FREEZE REPORT\n")
    f.write("=" * 72 + "\n\n")

    f.write("STATUS: PASS\n")
    f.write("Freeze version: final_3154_freeze_v1_20260916\n\n")

    f.write("INPUT / EXCLUSION\n")
    f.write(f"Logical input genomes: {EXPECTED_INPUT}\n")
    f.write(f"Sequence-QC failures: {len(qc_targets)}\n")
    f.write(f"Duplicate exclusion targets: {len(dup_targets)}\n")
    f.write(f"QC/duplicate overlap: {len(qc_dup_overlap)}\n")
    f.write(
        "Additional duplicate exclusions after overlap: "
        f"{len(additional_dup_targets)}\n"
    )
    f.write(f"Unique exclusions: {len(target_set)}\n")
    f.write(f"Final retained genomes: {len(retained)}\n\n")

    f.write("FINAL COHORTS\n")
    f.write(f"A: {final_counts['A']}\n")
    f.write(f"B1: {final_counts['B1']}\n")
    f.write(f"B2 seen-genus: {final_counts['B2_seen_genus']}\n")
    f.write(f"B2 unseen-genus: {final_counts['B2_unseen_genus']}\n")
    f.write(f"External total: {external_total}\n\n")

    f.write("CANONICAL SEQUENCE AUDIT\n")
    f.write(
        f"Duplicate groups before exclusion: "
        f"{len(duplicate_groups_before)}\n"
    )
    f.write(
        f"Duplicate members before exclusion: "
        f"{duplicate_members_before}\n"
    )
    f.write(
        f"Cross-cohort duplicate groups before exclusion: "
        f"{cross_cohort_duplicate_groups_before}\n"
    )
    f.write(
        f"Duplicate groups after exclusion: "
        f"{len(retained_duplicate_groups)}\n"
    )
    f.write(
        f"Cross-cohort duplicate groups after exclusion: "
        f"{retained_cross_cohort_duplicate_groups}\n\n"
    )

    f.write("FASTA\n")
    f.write(f"Existing mapped FASTA: {len(retained)}\n")
    f.write(f"Missing FASTA: 0\n")
    f.write(f"Duplicate retained FASTA paths: 0\n\n")

    f.write("TAXONOMY SUMMARY\n")
    f.write(f"All retained species clusters: {len(all_species_clusters)}\n")
    f.write(f"All retained genera: {len(all_genera)}\n")
    f.write(f"All retained families: {len(all_families)}\n")
    f.write(f"Development A species clusters: {len(A_species_clusters)}\n")
    f.write(f"Development A genera: {len(A_genera)}\n")
    f.write(f"Development A families: {len(A_families)}\n\n")

    f.write("DETECTED COLUMNS\n")
    f.write(f"Primary accession: {primary_acc_col}\n")
    f.write(f"Audit accession: {audit_acc_col}\n")
    f.write(f"FASTA map accession: {map_acc_col}\n")
    f.write(f"FASTA path: {map_path_col}\n")
    f.write(f"Canonical sequence hash: {canonical_hash_col}\n")
    f.write(f"Species cluster: {species_cluster_col}\n")
    f.write(f"Species: {species_col}\n")
    f.write(f"Genus: {genus_col}\n")
    f.write(f"Family: {family_col}\n\n")

    f.write("FINAL MASTER SHA256\n")
    f.write(master_sha256 + "\n\n")

    f.write("FINAL_FREEZE=PASS\n")

print()
print("=" * 72)
print("ALL FINAL FREEZE GATES PASSED")
print("=" * 72)
print(f"Output directory : {OUTDIR}")
print(f"Final master     : {final_master}")
print(f"Retained genomes : {len(retained)}")
print(f"Excluded genomes : {len(excluded)}")
print(f"External total   : {external_total}")
print(f"SHA256           : {master_sha256}")
print()
print("FINAL_FREEZE=PASS")
