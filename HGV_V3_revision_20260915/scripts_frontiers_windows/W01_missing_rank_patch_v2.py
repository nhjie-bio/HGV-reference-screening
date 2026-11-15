#!/usr/bin/env python3

import csv
import subprocess
import sys

from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = ROOT / "scripts_frontiers_windows"

SOURCE = SCRIPTS / "W01_existing_prediction_audit_v1.py"
TARGET = SCRIPTS / "W01_existing_prediction_audit_v2.py"

PRED = ROOT / (
    "09_external_validation/"
    "external_one_shot_evaluation_v1_20260917/"
    "external_query_level_results.tsv"
)

OUT = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W01_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

LEVELS = {
    "B1": "species",
    "B2_seen_genus": "genus",
    "B2_unseen_genus": "family",
}

EMPTY = {"", "na", "n/a", "nan", "none", "null", "-"}
FALSE = {"false", "0", "no", "n"}


def is_missing(value):
    return str(value or "").strip().lower() in EMPTY


def is_no_hit(row):
    flag = str(
        row.get("any_reference_above_threshold") or ""
    ).strip().lower()

    count = str(
        row.get("references_above_threshold") or ""
    ).strip()

    try:
        zero_count = bool(count) and float(count) == 0.0
    except ValueError:
        zero_count = False

    return flag in FALSE or zero_count


print("=" * 76)
print("HGV WINDOWS W01-v2 - MISSING RANK AUDIT")
print("=" * 76)

if not SOURCE.is_file() or not PRED.is_file():
    raise SystemExit("[STOP] Original script or TSV missing")

issues = []
counts = Counter()

with PRED.open("r", encoding="utf-8-sig", newline="") as f:

    reader = csv.DictReader(f, delimiter="\t")

    required = {
        "cohort",
        "method",
        "model_seed",
        "query_accession",
        "any_reference_above_threshold",
        "references_above_threshold",
    }

    for level in LEVELS.values():
        required.update({
            f"{level}_first_rank",
            f"query_{level}",
            f"top1_{level}",
        })

    if not required.issubset(set(reader.fieldnames or [])):
        raise SystemExit(
            "[STOP] Missing columns: "
            + repr(sorted(required - set(reader.fieldnames or [])))
        )

    for row in reader:

        cohort = row["cohort"].strip()

        if cohort not in LEVELS:
            continue

        level = LEVELS[cohort]
        raw = row[f"{level}_first_rank"]

        if not is_missing(raw):
            continue

        truth = str(
            row.get(f"query_{level}") or ""
        ).strip()

        top = str(
            row.get(f"top1_{level}") or ""
        ).strip()

        if is_missing(truth):
            status = "UNRESOLVED_QUERY_TAXONOMY"

        elif not is_missing(top):

            if top != truth:
                status = "VERIFIED_TOP1_WRONG"
            else:
                status = "CONTRADICTION_TOP1_CORRECT"

        elif is_no_hit(row):
            status = "VERIFIED_NO_HIT"

        else:
            status = "UNRESOLVED_NO_TOP1"

        counts[status] += 1

        issues.append({
            "cohort": cohort,
            "method": row["method"],
            "model_seed": row["model_seed"],
            "query_accession": row["query_accession"],
            "rank_level": level,
            "raw_rank": raw,
            "query_taxon": truth,
            "top1_taxon": top,
            "any_hit": row["any_reference_above_threshold"],
            "references_above_threshold":
                row["references_above_threshold"],
            "status": status,
        })


# =====================================================
# 1. Save missing-rank audit
# =====================================================

AUDIT = OUT / "W01_MISSING_RANK_AUDIT.tsv"

fields = [
    "cohort",
    "method",
    "model_seed",
    "query_accession",
    "rank_level",
    "raw_rank",
    "query_taxon",
    "top1_taxon",
    "any_hit",
    "references_above_threshold",
    "status",
]

with AUDIT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=fields, delimiter="\t"
    )
    writer.writeheader()
    writer.writerows(issues)

print()
print("[1] MISSING RANK SUMMARY")
print("Missing rank records:", len(issues))

for status, n in sorted(counts.items()):
    print(f"{status}: {n}")

print("Audit saved:", AUDIT)


# =====================================================
# 2. Strict validation
# =====================================================

unsafe = [
    row for row in issues
    if not row["status"].startswith("VERIFIED_")
]

if unsafe:

    print()
    print("[STOP] Ambiguous or contradictory ranks")

    for row in unsafe[:10]:
        print(
            row["cohort"],
            row["method"],
            row["query_accession"],
            row["status"],
            "truth=", row["query_taxon"],
            "top1=", row["top1_taxon"],
        )

    raise SystemExit(2)

print("[PASS] Missing ranks classified using evidence")


# =====================================================
# 3. Patch v1 to create separate v2
# =====================================================

source = SOURCE.read_text(encoding="utf-8-sig")

start = source.find("def rank1(")
end = source.find("\ndef pct(", start)

if start < 0 or end < 0:
    raise SystemExit("[STOP] Cannot locate rank1 function")

old_out = '"90_windows_minimal/W01"'

if source.count(old_out) != 1:
    raise SystemExit("[STOP] Unexpected output path structure")

replacement = '''
def rank1(row, level="species"):

    value = str(
        row.get(level + "_first_rank") or ""
    ).strip()

    missing_tokens = {
        "", "na", "n/a", "nan", "none", "null", "-"
    }

    if value.lower() in missing_tokens:

        truth = str(
            row.get("query_" + level) or ""
        ).strip()

        top = str(
            row.get("top1_" + level) or ""
        ).strip()

        flag = str(
            row.get("any_reference_above_threshold") or ""
        ).strip().lower()

        count = str(
            row.get("references_above_threshold") or ""
        ).strip()

        try:
            zero_count = bool(count) and float(count) == 0.0
        except ValueError:
            zero_count = False

        if truth and truth.lower() not in missing_tokens:

            if top and top.lower() not in missing_tokens:

                if top != truth:
                    return 0

            elif (
                flag in {"false", "0", "no", "n"}
                or zero_count
            ):
                return 0

        raise RuntimeError(
            "Ambiguous or contradictory missing rank: "
            + str(row.get("query_accession"))
            + " level=" + level
        )

    try:
        x = float(value)
    except ValueError:
        raise RuntimeError(
            "Invalid rank: " + repr(value)
        )

    require(math.isfinite(x), "Nonfinite rank")

    return int(x == 1.0)

'''

patched = (
    source[:start]
    + replacement
    + source[end + 1:]
)

patched = patched.replace(
    old_out,
    '"90_windows_minimal/W01_v2"',
    1
)

if TARGET.exists():

    if TARGET.read_text(encoding="utf-8-sig") != patched:
        raise SystemExit(
            "[STOP] Existing v2 differs; not overwriting"
        )

else:
    TARGET.write_text(patched, encoding="utf-8")

print()
print("[2] PATCH")
print("[PASS] W01-v2 created")
print("Original W01-v1 preserved")
print("Original external TSV unchanged")


# =====================================================
# 4. Execute revised analysis
# =====================================================

print()
print("[3] EXECUTE W01-v2")
print("=" * 76, flush=True)

result = subprocess.run(
    [sys.executable, str(TARGET)],
    cwd=str(ROOT),
)

raise SystemExit(result.returncode)
