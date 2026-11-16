#!/usr/bin/env python3

import csv
import hashlib
import itertools
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE = Path("HGV_V3_revision_20260915")

FINAL_MANIFEST = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916"
    / "final_master_manifest_3154.tsv"
)

SPLIT_DIR = (
    BASE
    / "04_splits/g1_g4_v1_20260916"
)

MANIFEST_DIR = SPLIT_DIR / "manifests"

OUTDIR = (
    BASE
    / "04_splits/g1_g4_provenance_reaudit_v1_20260916"
)

WORKERS = 32

ACC_RE = re.compile(r"GC[AF]_[0-9]+\.[0-9]+")


def die(msg):
    print()
    print("PROVENANCE_REAUDIT=FAIL")
    print(msg)
    sys.exit(2)


def clean(x):
    return "" if x is None else str(x).strip()


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
        w = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def md5_file(path, block=1024 * 1024):
    """
    MD5 of the actual on-disk FASTA file.
    For .fna.gz this is the compressed-file MD5.

    Exact biological sequence identity is independently
    audited with canonical_sequence_hash.
    """
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def md5_one(item):
    acc, path = item
    p = Path(path)

    if not p.is_file():
        return acc, "", f"missing:{p}"

    try:
        return acc, md5_file(p), ""
    except Exception as e:
        return acc, "", repr(e)


def cross_role_values(rows, field):
    value_roles = defaultdict(set)

    for r in rows:
        value = clean(r.get(field, ""))
        role = clean(r.get("role", ""))

        if value and role:
            value_roles[value].add(role)

    bad = {
        value: roles
        for value, roles in value_roles.items()
        if len(roles) > 1
    }

    return bad


def cross_partition_pair_edges(role_of, edges):
    bad = []

    for a, b in edges:
        if a not in role_of or b not in role_of:
            continue

        ra = role_of[a]
        rb = role_of[b]

        if ra != rb:
            bad.append((a, b, ra, rb))

    return bad


print("=" * 78)
print("HGV-V3 SPLIT PROVENANCE RE-AUDIT")
print("=" * 78)

# ------------------------------------------------------------------
# 1. Load frozen final manifest
# ------------------------------------------------------------------

fields, rows = read_tsv(FINAL_MANIFEST)

required = {
    "assembly_accession_final",
    "final_cohort",
    "biosample",
    "canonical_sequence_hash",
    "fasta_path",
    "gtdb_species_cluster",
    "gtdb_genus",
    "gtdb_family",
}

missing = required - set(fields)

if missing:
    die(
        "Final manifest missing required fields: "
        + ", ".join(sorted(missing))
    )

if len(rows) != 3154:
    die(f"Expected 3154 final genomes, observed {len(rows)}")

by_acc = {}

for r in rows:
    acc = clean(r["assembly_accession_final"])

    if not acc:
        die("Empty assembly accession")

    if acc in by_acc:
        die(f"Duplicate final accession: {acc}")

    by_acc[acc] = r

A = {
    acc
    for acc, r in by_acc.items()
    if clean(r["final_cohort"]) == "A"
}

EXTERNAL = set(by_acc) - A

print()
print("[Frozen cohort]")
print(f"Total    : {len(by_acc)}")
print(f"A        : {len(A)}")
print(f"External : {len(EXTERNAL)}")

if len(A) != 2374:
    die(f"Expected actual A=2374, observed {len(A)}")

if len(EXTERNAL) != 780:
    die(f"Expected actual external=780, observed {len(EXTERNAL)}")

# ------------------------------------------------------------------
# 2. Compute real file MD5
# ------------------------------------------------------------------

print()
print(f"[Computing FASTA file MD5 with {WORKERS} workers]")

tasks = [
    (
        acc,
        clean(r["fasta_path"]),
    )
    for acc, r in by_acc.items()
]

md5_map = {}
md5_fail = []

done = 0

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {
        ex.submit(md5_one, item): item[0]
        for item in tasks
    }

    for fut in as_completed(futures):
        acc, digest, error = fut.result()

        done += 1

        if error:
            md5_fail.append((acc, error))
        else:
            md5_map[acc] = digest

        if done % 250 == 0 or done == len(tasks):
            print(
                f"[{done:4d}/{len(tasks):4d}] "
                f"fail={len(md5_fail)}"
            )

if md5_fail:
    for x in md5_fail[:20]:
        print(x)

    die(
        f"FASTA MD5 failed for {len(md5_fail)} genomes"
    )

if len(md5_map) != 3154:
    die(
        f"Expected 3154 MD5 values, observed {len(md5_map)}"
    )

# ------------------------------------------------------------------
# 3. Global exact duplicate checks
# ------------------------------------------------------------------

md5_groups = defaultdict(list)

for acc, digest in md5_map.items():
    md5_groups[digest].append(acc)

duplicate_md5_groups = {
    h: members
    for h, members in md5_groups.items()
    if len(members) > 1
}

canonical_groups = defaultdict(list)

for acc, r in by_acc.items():
    h = clean(r["canonical_sequence_hash"])
    canonical_groups[h].append(acc)

duplicate_canonical_groups = {
    h: members
    for h, members in canonical_groups.items()
    if len(members) > 1
}

print()
print("[Global exact duplicate audit]")
print(
    f"Duplicate file-MD5 groups       : "
    f"{len(duplicate_md5_groups)}"
)
print(
    f"Duplicate canonical-hash groups : "
    f"{len(duplicate_canonical_groups)}"
)

if duplicate_canonical_groups:
    die(
        "Canonical sequence duplicates remain in final 3154"
    )

# ------------------------------------------------------------------
# 4. A vs external BioSample/hash/MD5 leakage
# ------------------------------------------------------------------

def values_for(accessions, field):
    return {
        clean(by_acc[a].get(field, ""))
        for a in accessions
        if clean(by_acc[a].get(field, ""))
    }


A_biosample = values_for(A, "biosample")
E_biosample = values_for(EXTERNAL, "biosample")

A_hash = values_for(A, "canonical_sequence_hash")
E_hash = values_for(EXTERNAL, "canonical_sequence_hash")

A_md5 = {md5_map[a] for a in A}
E_md5 = {md5_map[a] for a in EXTERNAL}

external_biosample_overlap = A_biosample & E_biosample
external_hash_overlap = A_hash & E_hash
external_md5_overlap = A_md5 & E_md5

print()
print("[A vs external leakage]")
print(
    f"BioSample overlap      : "
    f"{len(external_biosample_overlap)}"
)
print(
    f"Canonical hash overlap : "
    f"{len(external_hash_overlap)}"
)
print(
    f"FASTA MD5 overlap      : "
    f"{len(external_md5_overlap)}"
)

if external_biosample_overlap:
    die("A/external BioSample leakage detected")

if external_hash_overlap:
    die("A/external canonical-sequence leakage detected")

if external_md5_overlap:
    die("A/external FASTA MD5 leakage detected")

# ------------------------------------------------------------------
# 5. Parse GCF/GCA pairing audit files
# ------------------------------------------------------------------

pairing_files = sorted(
    set(BASE.glob("01_manifest/**/gcf_gca_pairing_audit.tsv"))
)

if not pairing_files:
    die(
        "Could not locate gcf_gca_pairing_audit.tsv"
    )

pair_edges = set()
pair_source_rows = []

for p in pairing_files:
    pf, pr = read_tsv(p)

    multi_accession_rows = 0

    for r in pr:
        found = set()

        for value in r.values():
            if value is None:
                continue

            found.update(
                ACC_RE.findall(str(value))
            )

        if len(found) >= 2:
            multi_accession_rows += 1

            for a, b in itertools.combinations(
                sorted(found),
                2,
            ):
                pair_edges.add((a, b))

    pair_source_rows.append({
        "file": str(p),
        "rows": len(pr),
        "multi_accession_rows": multi_accession_rows,
        "parsed_pair_edges_total_after_file": len(pair_edges),
    })

print()
print("[Paired-assembly audit source]")
for r in pair_source_rows:
    print(
        f"{r['file']} "
        f"rows={r['rows']} "
        f"multi_accession_rows="
        f"{r['multi_accession_rows']}"
    )

print(f"Parsed unique pair edges: {len(pair_edges)}")

# Final cohort paired leakage.
A_external_pair_violations = []

for a, b in pair_edges:
    if (
        (a in A and b in EXTERNAL)
        or
        (b in A and a in EXTERNAL)
    ):
        A_external_pair_violations.append(
            (a, b)
        )

print(
    f"A/external paired-assembly edges: "
    f"{len(A_external_pair_violations)}"
)

if A_external_pair_violations:
    print(A_external_pair_violations[:20])
    die(
        "Paired GCF/GCA leakage exists between "
        "Development and external cohort"
    )

# ------------------------------------------------------------------
# 6. Re-audit every frozen split
# ------------------------------------------------------------------

split_paths = sorted(
    MANIFEST_DIR.glob("G*_seed*.tsv")
)

if len(split_paths) != 10:
    die(
        f"Expected 10 G1-G4 split manifests; "
        f"observed {len(split_paths)}"
    )

audit_rows = []
pair_violation_rows = []

for path in split_paths:
    sf, sr = read_tsv(path)

    required_split = {
        "assembly_accession_final",
        "role",
        "protocol",
        "biosample",
        "canonical_sequence_hash",
        "gtdb_species_cluster",
        "gtdb_genus",
        "gtdb_family",
    }

    miss = required_split - set(sf)

    if miss:
        die(
            f"{path.name} missing fields: "
            + ", ".join(sorted(miss))
        )

    split_id = clean(sr[0]["split_id"])
    protocol = clean(sr[0]["protocol"])

    # All split rows must be Development A.
    split_acc = {
        clean(r["assembly_accession_final"])
        for r in sr
    }

    if split_acc != A:
        die(
            f"{split_id}: split accession set "
            "does not equal frozen Development A"
        )

    # Add true file MD5 for audit.
    for r in sr:
        acc = clean(r["assembly_accession_final"])
        r["_file_md5"] = md5_map[acc]

    checks = {}

    checks["accession"] = cross_role_values(
        sr,
        "assembly_accession_final",
    )

    checks["biosample"] = cross_role_values(
        sr,
        "biosample",
    )

    checks["canonical_sequence_hash"] = cross_role_values(
        sr,
        "canonical_sequence_hash",
    )

    checks["fasta_file_md5"] = cross_role_values(
        sr,
        "_file_md5",
    )

    if protocol == "G2":
        checks["required_group"] = cross_role_values(
            sr,
            "gtdb_species_cluster",
        )

    elif protocol == "G3":
        checks["required_group"] = cross_role_values(
            sr,
            "gtdb_genus",
        )

    elif protocol == "G4":
        checks["required_group"] = cross_role_values(
            sr,
            "gtdb_family",
        )

    else:
        checks["required_group"] = {}

    role_of = {
        clean(r["assembly_accession_final"]):
        clean(r["role"])
        for r in sr
    }

    paired_bad = cross_partition_pair_edges(
        role_of,
        pair_edges,
    )

    for a, b, ra, rb in paired_bad:
        pair_violation_rows.append({
            "split_id": split_id,
            "accession_a": a,
            "accession_b": b,
            "role_a": ra,
            "role_b": rb,
        })

    checks["paired_assembly"] = {
        f"{a}|{b}": {ra, rb}
        for a, b, ra, rb in paired_bad
    }

    print()
    print(f"[{split_id}]")

    for check, bad in checks.items():
        n = len(bad)

        print(f"{check:26s}: overlap={n}")

        audit_rows.append({
            "split_id": split_id,
            "protocol": protocol,
            "check": check,
            "observed_overlap": n,
            "expected_overlap": 0,
            "gate": "PASS" if n == 0 else "FAIL",
        })

# ------------------------------------------------------------------
# 7. Final gate
# ------------------------------------------------------------------

failures = [
    r
    for r in audit_rows
    if r["gate"] != "PASS"
]

if failures:
    print()
    for r in failures:
        print(r)

    die(
        f"{len(failures)} split provenance "
        "checks failed"
    )

# ------------------------------------------------------------------
# 8. Write NEW audit directory only after PASS
# ------------------------------------------------------------------

if OUTDIR.exists():
    die(
        f"Output already exists: {OUTDIR}"
    )

OUTDIR.mkdir(
    parents=True,
    exist_ok=False,
)

md5_rows = [
    {
        "assembly_accession": acc,
        "fasta_path": clean(
            by_acc[acc]["fasta_path"]
        ),
        "fasta_file_md5": md5_map[acc],
        "final_cohort": clean(
            by_acc[acc]["final_cohort"]
        ),
    }
    for acc in sorted(by_acc)
]

write_tsv(
    OUTDIR / "final3154_fasta_file_md5.tsv",
    md5_rows,
    [
        "assembly_accession",
        "fasta_path",
        "fasta_file_md5",
        "final_cohort",
    ],
)

write_tsv(
    OUTDIR / "paired_audit_sources.tsv",
    pair_source_rows,
    [
        "file",
        "rows",
        "multi_accession_rows",
        "parsed_pair_edges_total_after_file",
    ],
)

write_tsv(
    OUTDIR / "split_provenance_reaudit.tsv",
    audit_rows,
    [
        "split_id",
        "protocol",
        "check",
        "observed_overlap",
        "expected_overlap",
        "gate",
    ],
)

write_tsv(
    OUTDIR / "paired_cross_role_violations.tsv",
    pair_violation_rows,
    [
        "split_id",
        "accession_a",
        "accession_b",
        "role_a",
        "role_b",
    ],
)

split_hash_rows = []

for p in split_paths:
    split_hash_rows.append({
        "split_manifest": str(p),
        "sha256": sha256_file(p),
    })

write_tsv(
    OUTDIR / "audited_split_sha256.tsv",
    split_hash_rows,
    [
        "split_manifest",
        "sha256",
    ],
)

summary = {
    "status": "PASS",
    "final_genomes": len(by_acc),
    "development_A": len(A),
    "external": len(EXTERNAL),
    "duplicate_file_md5_groups": len(
        duplicate_md5_groups
    ),
    "duplicate_canonical_hash_groups": len(
        duplicate_canonical_groups
    ),
    "A_external_biosample_overlap": len(
        external_biosample_overlap
    ),
    "A_external_canonical_hash_overlap": len(
        external_hash_overlap
    ),
    "A_external_file_md5_overlap": len(
        external_md5_overlap
    ),
    "parsed_paired_assembly_edges": len(
        pair_edges
    ),
    "A_external_paired_assembly_edges": len(
        A_external_pair_violations
    ),
    "split_manifests_audited": len(
        split_paths
    ),
    "split_failed_checks": len(
        failures
    ),
}

with open(
    OUTDIR / "provenance_reaudit.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")

report = OUTDIR / "PROVENANCE_REAUDIT_REPORT.txt"

with open(
    report,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "HGV-V3 SPLIT PROVENANCE RE-AUDIT\n"
    )
    f.write("=" * 78 + "\n\n")
    f.write("STATUS: PASS\n\n")
    f.write(f"Final genomes: {len(by_acc)}\n")
    f.write(f"Development A: {len(A)}\n")
    f.write(f"External: {len(EXTERNAL)}\n\n")

    f.write("GLOBAL DUPLICATION\n")
    f.write(
        "Duplicate canonical-hash groups: "
        f"{len(duplicate_canonical_groups)}\n"
    )
    f.write(
        "Duplicate FASTA-file MD5 groups: "
        f"{len(duplicate_md5_groups)}\n\n"
    )

    f.write("A VS EXTERNAL\n")
    f.write(
        f"BioSample overlap: "
        f"{len(external_biosample_overlap)}\n"
    )
    f.write(
        f"Canonical hash overlap: "
        f"{len(external_hash_overlap)}\n"
    )
    f.write(
        f"FASTA MD5 overlap: "
        f"{len(external_md5_overlap)}\n"
    )
    f.write(
        f"Paired assembly edges: "
        f"{len(A_external_pair_violations)}\n\n"
    )

    f.write(
        f"Split manifests audited: "
        f"{len(split_paths)}\n"
    )
    f.write(
        f"Failed split checks: "
        f"{len(failures)}\n\n"
    )

    f.write("PROVENANCE_REAUDIT=PASS\n")

files_to_hash = [
    OUTDIR / "final3154_fasta_file_md5.tsv",
    OUTDIR / "paired_audit_sources.tsv",
    OUTDIR / "split_provenance_reaudit.tsv",
    OUTDIR / "paired_cross_role_violations.tsv",
    OUTDIR / "audited_split_sha256.tsv",
    OUTDIR / "provenance_reaudit.json",
    report,
]

with open(
    OUTDIR / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in files_to_hash:
        f.write(
            f"{sha256_file(p)}  {p.name}\n"
        )

print()
print("=" * 78)
print("ALL PROVENANCE RE-AUDIT GATES PASSED")
print("=" * 78)
print(f"Split manifests       : {len(split_paths)}")
print(
    f"A/external BioSample : "
    f"{len(external_biosample_overlap)}"
)
print(
    f"A/external hash      : "
    f"{len(external_hash_overlap)}"
)
print(
    f"A/external MD5       : "
    f"{len(external_md5_overlap)}"
)
print(
    f"A/external paired    : "
    f"{len(A_external_pair_violations)}"
)
print(f"Output                : {OUTDIR}")
print()
print("PROVENANCE_REAUDIT=PASS")
