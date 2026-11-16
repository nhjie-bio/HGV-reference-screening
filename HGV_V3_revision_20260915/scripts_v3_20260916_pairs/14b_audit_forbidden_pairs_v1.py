#!/usr/bin/env python3

import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("HGV_V3_revision_20260915")

MASTER = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916/"
      "final_master_manifest_3154.tsv"
)

PAIR_ROOT = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
)

PAIRING_AUDIT = (
    BASE
    / "01_manifest/gcf_gca_pairing_audit.tsv"
)

OUT = (
    BASE
    / "05_pairs/pair_forbidden_reaudit_v1_20260916"
)


def die(msg):
    print(f"ERROR: {msg}")
    print("FORBIDDEN_PAIR_REAUDIT=FAIL")
    sys.exit(2)


def clean(x):
    return "" if x is None else str(x).strip()


def read_tsv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    return fields, rows


def write_tsv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def pair_key(a, b):
    a = clean(a)
    b = clean(b)

    return tuple(sorted((a, b)))


print("=" * 78)
print("HGV-V3 FORBIDDEN WITHIN-PAIR REAUDIT")
print("=" * 78)

if OUT.exists():
    die(f"Output already exists: {OUT}")

if not MASTER.is_file():
    die(f"Master manifest missing: {MASTER}")

if not PAIR_ROOT.is_dir():
    die(f"Pair root missing: {PAIR_ROOT}")

OUT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1. Master manifest
# ----------------------------------------------------------------------

print()
print("[1] LOAD FINAL MASTER MANIFEST")

fields, rows = read_tsv(MASTER)

acc_col = "assembly_accession_final"

if acc_col not in fields:
    die(f"Missing accession column: {acc_col}")

biosample_candidates = [
    "biosample",
    "biosample_accession",
    "BioSample",
    "biosample_acc",
    "biosample_id",
]

biosample_col = next(
    (
        c
        for c in biosample_candidates
        if c in fields
    ),
    None,
)

if biosample_col is None:
    print("Available columns:")
    print("\n".join(fields))
    die("Could not detect BioSample column")

hash_candidates = [
    "canonical_sequence_hash",
    "canonical_hash",
    "sequence_hash",
    "canonical_md5",
]

hash_col = next(
    (
        c
        for c in hash_candidates
        if c in fields
    ),
    None,
)

acc_to_biosample = {}
acc_to_hash = {}

for r in rows:
    acc = clean(r[acc_col])

    if not acc:
        die("Empty assembly accession in master")

    if acc in acc_to_biosample:
        die(f"Duplicate accession in master: {acc}")

    acc_to_biosample[acc] = clean(
        r.get(biosample_col, "")
    )

    if hash_col:
        acc_to_hash[acc] = clean(
            r.get(hash_col, "")
        )

print(f"Master genomes    : {len(acc_to_biosample)}")
print(f"BioSample column  : {biosample_col}")
print(f"Hash column       : {hash_col or 'not_detected'}")

if len(acc_to_biosample) != 3154:
    die(
        f"Expected final 3154 genomes; "
        f"observed {len(acc_to_biosample)}"
    )

missing_bs = [
    acc
    for acc, bs in acc_to_biosample.items()
    if not bs
]

print(f"Missing BioSample : {len(missing_bs)}")

# ----------------------------------------------------------------------
# 2. Global duplicate BioSample audit
# ----------------------------------------------------------------------

print()
print("[2] GLOBAL BIOSAMPLE DUPLICATION")

bs_groups = defaultdict(list)

for acc, bs in acc_to_biosample.items():
    if bs:
        bs_groups[bs].append(acc)

duplicate_bs = {
    bs: sorted(accs)
    for bs, accs in bs_groups.items()
    if len(accs) > 1
}

print(
    "Duplicate BioSample groups : "
    f"{len(duplicate_bs)}"
)

print(
    "Assemblies in duplicate "
    f"BioSample groups : "
    f"{sum(len(x) for x in duplicate_bs.values())}"
)

dup_rows = []

for bs, accs in sorted(
    duplicate_bs.items()
):
    dup_rows.append({
        "biosample": bs,
        "n_assemblies": len(accs),
        "assemblies": ",".join(accs),
    })

write_tsv(
    OUT / "duplicate_biosample_groups.tsv",
    dup_rows,
    [
        "biosample",
        "n_assemblies",
        "assemblies",
    ],
)

# ----------------------------------------------------------------------
# 3. Recover known GCF/GCA pair edges if audit has any
# ----------------------------------------------------------------------

print()
print("[3] KNOWN GCF/GCA PAIR EDGES")

known_paired_edges = set()

if (
    PAIRING_AUDIT.is_file()
    and PAIRING_AUDIT.stat().st_size > 0
):
    pf, pr = read_tsv(
        PAIRING_AUDIT
    )

    accession_re = re.compile(
        r"\bGC[FA]_\d+\.\d+\b"
    )

    for r in pr:
        accessions = []

        for value in r.values():
            accessions.extend(
                accession_re.findall(
                    clean(value)
                )
            )

        accessions = sorted(
            set(accessions)
        )

        for i in range(len(accessions)):
            for j in range(i + 1, len(accessions)):
                a = accessions[i]
                b = accessions[j]

                if (
                    a.startswith("GCF_")
                    and b.startswith("GCA_")
                ) or (
                    a.startswith("GCA_")
                    and b.startswith("GCF_")
                ):
                    known_paired_edges.add(
                        pair_key(a, b)
                    )

print(
    "Known paired-assembly edges: "
    f"{len(known_paired_edges)}"
)

# ----------------------------------------------------------------------
# 4. Audit all generated pair files
# ----------------------------------------------------------------------

print()
print("[4] AUDIT GENERATED PAIRS")

pair_files = []

for split_dir in sorted(
    p
    for p in PAIR_ROOT.iterdir()
    if p.is_dir()
):
    for role, name in [
        (
            "train",
            "train_unique_pair_pool.tsv",
        ),
        (
            "val",
            "val_pairs.tsv",
        ),
        (
            "test",
            "test_pairs.tsv",
        ),
    ]:
        p = split_dir / name

        if p.is_file():
            pair_files.append(
                (
                    split_dir.name,
                    role,
                    p,
                )
            )

if len(pair_files) != 30:
    die(
        "Expected 30 pair files "
        f"(10 splits x 3 roles); "
        f"observed {len(pair_files)}"
    )

violations = []
summary = []

total_pairs = 0
total_same_bs = 0
total_known_pair = 0
total_same_hash = 0
total_unknown_acc = 0

for split_id, role, path in pair_files:

    pf, pr = read_tsv(path)

    required = {
        "accession_a",
        "accession_b",
    }

    if not required.issubset(pf):
        die(
            f"Missing pair accession fields: "
            f"{path}"
        )

    c = Counter()

    for row_number, r in enumerate(
        pr,
        start=2,
    ):
        a = clean(r["accession_a"])
        b = clean(r["accession_b"])

        if (
            a not in acc_to_biosample
            or b not in acc_to_biosample
        ):
            c["unknown_accession"] += 1

            violations.append({
                "split_id": split_id,
                "role": role,
                "row_number": row_number,
                "accession_a": a,
                "accession_b": b,
                "biosample_a": (
                    acc_to_biosample.get(a, "")
                ),
                "biosample_b": (
                    acc_to_biosample.get(b, "")
                ),
                "violation_type": (
                    "unknown_accession"
                ),
            })

            continue

        bs_a = acc_to_biosample[a]
        bs_b = acc_to_biosample[b]

        if (
            a != b
            and bs_a
            and bs_a == bs_b
        ):
            c["same_biosample"] += 1

            violations.append({
                "split_id": split_id,
                "role": role,
                "row_number": row_number,
                "accession_a": a,
                "accession_b": b,
                "biosample_a": bs_a,
                "biosample_b": bs_b,
                "violation_type": (
                    "same_biosample_different_assembly"
                ),
            })

        if (
            pair_key(a, b)
            in known_paired_edges
        ):
            c["known_paired_assembly"] += 1

            violations.append({
                "split_id": split_id,
                "role": role,
                "row_number": row_number,
                "accession_a": a,
                "accession_b": b,
                "biosample_a": bs_a,
                "biosample_b": bs_b,
                "violation_type": (
                    "known_gcf_gca_pair"
                ),
            })

        if hash_col:
            ha = acc_to_hash.get(a, "")
            hb = acc_to_hash.get(b, "")

            if (
                a != b
                and ha
                and ha == hb
            ):
                c["same_canonical_hash"] += 1

                violations.append({
                    "split_id": split_id,
                    "role": role,
                    "row_number": row_number,
                    "accession_a": a,
                    "accession_b": b,
                    "biosample_a": bs_a,
                    "biosample_b": bs_b,
                    "violation_type": (
                        "same_canonical_sequence_hash"
                    ),
                })

    n = len(pr)

    total_pairs += n
    total_same_bs += c[
        "same_biosample"
    ]
    total_known_pair += c[
        "known_paired_assembly"
    ]
    total_same_hash += c[
        "same_canonical_hash"
    ]
    total_unknown_acc += c[
        "unknown_accession"
    ]

    gate = (
        "PASS"
        if sum(c.values()) == 0
        else "FAIL"
    )

    summary.append({
        "split_id": split_id,
        "role": role,
        "n_pairs": n,
        "same_biosample": c[
            "same_biosample"
        ],
        "known_paired_assembly": c[
            "known_paired_assembly"
        ],
        "same_canonical_hash": c[
            "same_canonical_hash"
        ],
        "unknown_accession": c[
            "unknown_accession"
        ],
        "gate": gate,
    })

    print(
        f"{split_id:22s} "
        f"{role:5s} "
        f"pairs={n:6d} "
        f"same_bs={c['same_biosample']:3d} "
        f"paired={c['known_paired_assembly']:3d} "
        f"hash={c['same_canonical_hash']:3d} "
        f"unknown={c['unknown_accession']:3d} "
        f"{gate}"
    )

# ----------------------------------------------------------------------
# 5. Output
# ----------------------------------------------------------------------

write_tsv(
    OUT / "pair_forbidden_summary.tsv",
    summary,
    [
        "split_id",
        "role",
        "n_pairs",
        "same_biosample",
        "known_paired_assembly",
        "same_canonical_hash",
        "unknown_accession",
        "gate",
    ],
)

write_tsv(
    OUT / "pair_forbidden_violations.tsv",
    violations,
    [
        "split_id",
        "role",
        "row_number",
        "accession_a",
        "accession_b",
        "biosample_a",
        "biosample_b",
        "violation_type",
    ],
)

overall_pass = (
    len(missing_bs) == 0
    and total_same_bs == 0
    and total_known_pair == 0
    and total_same_hash == 0
    and total_unknown_acc == 0
)

with open(
    OUT / "FORBIDDEN_PAIR_REAUDIT_REPORT.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "HGV-V3 FORBIDDEN WITHIN-PAIR REAUDIT\n"
    )
    f.write("=" * 78 + "\n\n")

    f.write(
        f"STATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}\n\n"
    )

    f.write(
        f"Final genomes: "
        f"{len(acc_to_biosample)}\n"
    )

    f.write(
        f"BioSample field: "
        f"{biosample_col}\n"
    )

    f.write(
        f"Missing BioSample: "
        f"{len(missing_bs)}\n"
    )

    f.write(
        f"Global duplicate BioSample groups: "
        f"{len(duplicate_bs)}\n"
    )

    f.write(
        f"Known paired-assembly edges: "
        f"{len(known_paired_edges)}\n"
    )

    f.write(
        f"Pair manifests audited: "
        f"{len(pair_files)}\n"
    )

    f.write(
        f"Pairs audited: "
        f"{total_pairs}\n\n"
    )

    f.write(
        f"same BioSample pair violations: "
        f"{total_same_bs}\n"
    )

    f.write(
        f"known GCF/GCA pair violations: "
        f"{total_known_pair}\n"
    )

    f.write(
        f"same canonical hash violations: "
        f"{total_same_hash}\n"
    )

    f.write(
        f"unknown accession violations: "
        f"{total_unknown_acc}\n\n"
    )

    f.write(
        "FORBIDDEN_PAIR_REAUDIT="
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
        + "\n"
    )

# checksums

checksum_files = [
    OUT / "duplicate_biosample_groups.tsv",
    OUT / "pair_forbidden_summary.tsv",
    OUT / "pair_forbidden_violations.tsv",
    OUT / "FORBIDDEN_PAIR_REAUDIT_REPORT.txt",
]

with open(
    OUT / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:
    for p in checksum_files:
        f.write(
            f"{sha256(p)}  {p.name}\n"
        )

print()
print("=" * 78)

if overall_pass:
    print("ALL FORBIDDEN-PAIR GATES PASSED")
    print("=" * 78)
    print(f"Output: {OUT}")
    print()
    print("FORBIDDEN_PAIR_REAUDIT=PASS")
else:
    print("FORBIDDEN-PAIR AUDIT FAILED")
    print("=" * 78)
    print(f"Output: {OUT}")
    print()
    print("FORBIDDEN_PAIR_REAUDIT=FAIL")
    sys.exit(1)
