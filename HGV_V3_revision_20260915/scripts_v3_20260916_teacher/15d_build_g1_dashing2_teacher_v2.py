#!/usr/bin/env python3

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


BASE = Path(
    "HGV_V3_revision_20260915"
)

PAIR_ROOT = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
    / "G1_seed20260915"
)

CACHE_ROOTS = [
    BASE / "06_teacher/teacher_cache_v1_20260916",
    BASE / "06_teacher/teacher_cache_v2_20260916",
]

OUT = (
    BASE
    / "06_teacher/g1_teacher_augmentation_v2_20260917"
)

TAU = 0.98

ACCESSION_RE = re.compile(
    r"(GC[AF]_\d+\.\d+)"
)


def clean(x):
    return "" if x is None else str(x).strip()


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def read_tsv(path):

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

        return (
            list(
                reader.fieldnames or []
            ),
            list(reader),
        )


def write_tsv(
    path,
    rows,
    fields,
):

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


def accession_from_text(text):

    m = ACCESSION_RE.search(
        str(text)
    )

    if not m:

        raise RuntimeError(
            f"Cannot extract accession from: {text}"
        )

    return m.group(1)


def canonical_pair(a, b):

    if a == b:
        raise RuntimeError(
            f"Self pair encountered: {a}"
        )

    return (
        (a, b)
        if a < b
        else (b, a)
    )


# ==============================================================
# 0. PRECHECK
# ==============================================================

print("=" * 78)
print("HGV-V3 G1 DASHING2 TEACHER AUGMENTATION")
print("=" * 78)

print("test_used=False")
print("external_used=False")
print("dashing2_rerun=False")

if OUT.exists():

    raise SystemExit(
        f"ERROR: output already exists: {OUT}"
    )

train_path = (
    PAIR_ROOT
    / "train_unique_pair_pool.tsv"
)

val_path = (
    PAIR_ROOT
    / "val_pairs.tsv"
)

for p in [
    train_path,
    val_path,
]:

    if not p.is_file():

        raise RuntimeError(
            f"Missing pair manifest: {p}"
        )

# ==============================================================
# 1. LOAD G1 TRAIN + VAL ONLY
# ==============================================================

print()
print("[1] LOAD G1 TRAIN / VALIDATION PAIRS")

pair_sets = {}

wanted = set()

for role, path in [
    ("train", train_path),
    ("val", val_path),
]:

    fields, rows = read_tsv(
        path
    )

    required = [
        "accession_a",
        "accession_b",
        "label",
    ]

    for c in required:

        if c not in fields:

            raise RuntimeError(
                f"Missing column {c} in {path}"
            )

    for row in rows:

        a = clean(
            row[
                "accession_a"
            ]
        )

        b = clean(
            row[
                "accession_b"
            ]
        )

        wanted.add(
            canonical_pair(
                a,
                b,
            )
        )

    pair_sets[
        role
    ] = {
        "fields": fields,
        "rows": rows,
        "path": path,
    }

    print(
        f"{role:5s} pairs="
        f"{len(rows)}"
    )

print(
    f"unique_wanted_pairs={len(wanted)}"
)

# ==============================================================
# 2. AUTO-DISCOVER PRODUCTION DASHING2 MATRIX
# ==============================================================

print()
print("[2] DISCOVER FULL-A DASHING2 SETSKETCH MATRIX")

candidates = []

for root in CACHE_ROOTS:

    if not root.is_dir():
        continue

    for p in root.rglob("*"):

        if not p.is_file():
            continue

        try:

            with open(
                p,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:

                line1 = f.readline().rstrip(
                    "\n"
                )

                if (
                    line1
                    != "#Dashing2 Symmetric pairwise Output"
                ):
                    continue

                line2 = f.readline().rstrip(
                    "\n"
                )

                line3 = f.readline().rstrip(
                    "\n"
                )

        except Exception:
            continue

        if not line3.startswith(
            "#Sources\t"
        ):
            continue

        sources = line3.split(
            "\t"
        )[1:]

        n_sources = len(
            sources
        )

        if (
            n_sources == 2374
            and "k:31" in line2
            and "sketchsize:1024" in line2
            and "onepermsetsketch" in line2
        ):

            candidates.append(
                (
                    p,
                    line2,
                    sources,
                )
            )

if len(candidates) != 1:

    print(
        "Production-matrix candidates:"
    )

    for p, _, sources in candidates:

        print(
            f"  {p} "
            f"sources={len(sources)}"
        )

    raise RuntimeError(
        "Expected exactly one 2374-source "
        "Dashing2 SetSketch production matrix; "
        f"found {len(candidates)}"
    )

matrix_path, options_line, sources = (
    candidates[0]
)

print(
    f"matrix={matrix_path}"
)

print(
    f"sources={len(sources)}"
)

print(
    f"options={options_line}"
)

source_accessions = [
    accession_from_text(x)
    for x in sources
]

if (
    len(source_accessions)
    != len(
        set(source_accessions)
    )
):

    raise RuntimeError(
        "Duplicate accessions in Dashing2 #Sources."
    )

if len(source_accessions) != 2374:

    raise RuntimeError(
        "Unexpected Dashing2 source count."
    )

acc_to_matrix_idx = {
    acc: i
    for i, acc
    in enumerate(
        source_accessions
    )
}

# ==============================================================
# 3. MAP WANTED PAIRS TO UPPER-TRIANGLE CELLS
# ==============================================================

print()
print("[3] MAP G1 PAIRS TO DASHING2 MATRIX CELLS")

wanted_by_row = defaultdict(
    list
)

unknown = set()

for key in wanted:

    a, b = key

    if a not in acc_to_matrix_idx:

        unknown.add(a)

    if b not in acc_to_matrix_idx:

        unknown.add(b)

if unknown:

    raise RuntimeError(
        "G1 accessions missing from Dashing2 matrix: "
        + ",".join(
            sorted(unknown)[:20]
        )
    )

for key in wanted:

    a, b = key

    ia = acc_to_matrix_idx[
        a
    ]

    ib = acc_to_matrix_idx[
        b
    ]

    if ia < ib:

        i, j = ia, ib

    else:

        i, j = ib, ia

    wanted_by_row[
        i
    ].append(
        (
            j,
            key,
        )
    )

print(
    f"matrix_rows_needed="
    f"{len(wanted_by_row)}"
)

# ==============================================================
# 4. PARSE ONLY REQUIRED MATRIX CELLS
# ==============================================================

print()
print("[4] EXTRACT G1 DASHING2 DISTANCES")

distance = {}

with open(
    matrix_path,
    "r",
    encoding="utf-8",
    errors="strict",
) as f:

    header1 = f.readline()
    header2 = f.readline()
    header3 = f.readline()

    row_idx = 0

    for raw in f:

        raw = raw.rstrip(
            "\n"
        )

        if not raw:
            continue

        fields = raw.split(
            "\t"
        )

        if len(fields) != (
            1 + len(source_accessions)
        ):

            raise RuntimeError(
                f"Malformed Dashing2 row {row_idx}: "
                f"{len(fields)} fields; "
                f"expected {1 + len(source_accessions)}"
            )

        row_acc = accession_from_text(
            fields[0]
        )

        expected_acc = (
            source_accessions[
                row_idx
            ]
        )

        if row_acc != expected_acc:

            raise RuntimeError(
                f"Dashing2 row-order mismatch at "
                f"{row_idx}: "
                f"{row_acc} != {expected_acc}"
            )

        if row_idx in wanted_by_row:

            for col_idx, key in (
                wanted_by_row[
                    row_idx
                ]
            ):

                cell = fields[
                    col_idx + 1
                ]

                if cell == "-":

                    raise RuntimeError(
                        f"Required upper-triangle cell is '-' "
                        f"for {key}"
                    )

                # Dashing2 production matrix stores SIMILARITY.
                # HGV teacher target is DISTANCE.
                sim = float(
                    cell
                )

                if not (
                    0.0 <= sim <= 1.0
                ):

                    raise RuntimeError(
                        f"Invalid Dashing2 similarity {sim} "
                        f"for {key}"
                    )

                d = (
                    1.0
                    - sim
                )

                if key in distance:

                    raise RuntimeError(
                        f"Duplicate distance for {key}"
                    )

                distance[
                    key
                ] = d

        row_idx += 1

        if (
            row_idx % 250
            == 0
        ):

            print(
                f"rows_parsed="
                f"{row_idx}/2374 "
                f"wanted_found="
                f"{len(distance)}/"
                f"{len(wanted)}"
            )

if row_idx != 2374:

    raise RuntimeError(
        f"Expected 2374 Dashing2 rows; got {row_idx}"
    )

if len(distance) != len(wanted):

    missing = (
        wanted
        - set(
            distance.keys()
        )
    )

    raise RuntimeError(
        f"Missing wanted Dashing2 pairs: "
        f"{len(missing)}; "
        f"examples={list(sorted(missing))[:10]}"
    )

print(
    f"wanted_found="
    f"{len(distance)}/{len(wanted)}"
)

# ==============================================================
# 5. WRITE G1 TEACHER-AUGMENTED TRAIN / VAL
# ==============================================================

print()
print("[5] WRITE G1 TEACHER TABLES")

OUT.mkdir(
    parents=True,
)

teacher_dir = (
    OUT
    / "teacher_augmented"
    / "G1_seed20260915"
)

teacher_dir.mkdir(
    parents=True,
)

audit_rows = []

for role in [
    "train",
    "val",
]:

    fields = pair_sets[
        role
    ][
        "fields"
    ]

    rows = pair_sets[
        role
    ][
        "rows"
    ]

    out_rows = []

    pos_n = 0
    neg_n = 0
    informative_neg_n = 0

    for row in rows:

        a = clean(
            row[
                "accession_a"
            ]
        )

        b = clean(
            row[
                "accession_b"
            ]
        )

        key = canonical_pair(
            a,
            b,
        )

        d = distance[
            key
        ]

        y = int(
            float(
                row[
                    "label"
                ]
            )
        )

        if y == 1:

            pos_n += 1

        elif y == 0:

            neg_n += 1

            if d < TAU:

                informative_neg_n += 1

        else:

            raise RuntimeError(
                f"Invalid label: {y}"
            )

        new_row = dict(
            row
        )

        new_row[
            "dashing2_set_distance"
        ] = f"{d:.12g}"

        out_rows.append(
            new_row
        )

    out_fields = list(
        fields
    )

    if (
        "dashing2_set_distance"
        not in out_fields
    ):

        out_fields.append(
            "dashing2_set_distance"
        )

    out_path = (
        teacher_dir
        / (
            "train_teacher.tsv"
            if role == "train"
            else "val_teacher.tsv"
        )
    )

    write_tsv(
        out_path,
        out_rows,
        out_fields,
    )

    informative_fraction = (
        informative_neg_n
        / neg_n
        if neg_n
        else 0.0
    )

    audit_rows.append({
        "split_id":
            "G1_seed20260915",

        "role":
            role,

        "n_pairs":
            len(rows),

        "positive_n":
            pos_n,

        "negative_n":
            neg_n,

        "tau":
            TAU,

        "informative_negative_n":
            informative_neg_n,

        "informative_negative_fraction":
            informative_fraction,

        "teacher_missing":
            0,

        "output":
            str(
                out_path
            ),
    })

    print(
        f"{role:5s} "
        f"n={len(rows)} "
        f"positive={pos_n} "
        f"negative={neg_n} "
        f"D2_tau0.98_neg="
        f"{informative_neg_n}/"
        f"{neg_n} "
        f"({informative_fraction:.6f}) "
        f"PASS"
    )

audit_dir = (
    OUT
    / "audits"
)

audit_dir.mkdir()

write_tsv(
    audit_dir
    / "g1_teacher_join_audit.tsv",
    audit_rows,
    [
        "split_id",
        "role",
        "n_pairs",
        "positive_n",
        "negative_n",
        "tau",
        "informative_negative_n",
        "informative_negative_fraction",
        "teacher_missing",
        "output",
    ],
)

# ==============================================================
# 6. PROVENANCE
# ==============================================================

provenance = {
    "status":
        "PASS",

    "split_id":
        "G1_seed20260915",

    "teacher":
        "Dashing2 SetSketch",

    "dashing2_k":
        31,

    "dashing2_sketch_size":
        1024,

    "dashing2_mode":
        "onepermsetsketch",

    "tau":
        TAU,

    "matrix_path":
        str(
            matrix_path
        ),

    "matrix_sha256":
        sha256(
            matrix_path
        ),

    "train_pair_sha256":
        sha256(
            train_path
        ),

    "val_pair_sha256":
        sha256(
            val_path
        ),

    "test_used":
        False,

    "external_used":
        False,

    "matrix_value_semantics":
        "similarity",

    "distance_transform":
        "1 - Dashing2 similarity",

    "dashing2_rerun":
        False,

    "wanted_pairs":
        len(
            wanted
        ),

    "wanted_pairs_found":
        len(
            distance
        ),
}

(
    OUT
    / "g1_teacher_provenance.json"
).write_text(
    json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

# ==============================================================
# 7. REPORT + SHA256
# ==============================================================

report = (
    OUT
    / "G1_TEACHER_AUGMENTATION_REPORT.txt"
)

with open(
    report,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "HGV-V3 G1 DASHING2 TEACHER AUGMENTATION\n"
    )

    f.write(
        "=" * 78
        + "\n\n"
    )

    f.write(
        "STATUS: PASS\n\n"
    )

    f.write(
        "Split: G1_seed20260915\n"
    )

    f.write(
        "Teacher: Dashing2 SetSketch\n"
    )

    f.write(
        "k: 31\n"
    )

    f.write(
        "sketch size: 1024\n"
    )

    f.write(
        "tau: 0.98\n"
    )

    f.write(
        "Matrix value semantics: similarity\n"
    )

    f.write(
        "Teacher distance transform: 1 - similarity\n\n"
    )

    f.write(
        f"Production matrix: {matrix_path}\n"
    )

    f.write(
        f"Production sources: "
        f"{len(source_accessions)}\n"
    )

    f.write(
        f"Wanted unique pairs: "
        f"{len(wanted)}\n"
    )

    f.write(
        f"Wanted pairs found: "
        f"{len(distance)}\n\n"
    )

    for r in audit_rows:

        f.write(
            f"{r['role']}: "
            f"pairs={r['n_pairs']}; "
            f"positive={r['positive_n']}; "
            f"negative={r['negative_n']}; "
            f"informative-negative="
            f"{r['informative_negative_n']}/"
            f"{r['negative_n']} "
            f"({r['informative_negative_fraction']:.6f})\n"
        )

    f.write("\n")
    f.write(
        "G1 test used: NO\n"
    )

    f.write(
        "External B1/B2 used: NO\n"
    )

    f.write(
        "Dashing2 recomputed: NO\n\n"
    )

    f.write(
        "G1_TEACHER_AUGMENTATION=PASS\n"
    )

files = sorted([
    p
    for p in OUT.rglob("*")
    if (
        p.is_file()
        and p.name
        != "SHA256SUMS.txt"
    )
])

with open(
    OUT
    / "SHA256SUMS.txt",
    "w",
    encoding="utf-8",
) as f:

    for p in files:

        f.write(
            f"{sha256(p)}  "
            f"{p.relative_to(OUT)}\n"
        )

print()
print("=" * 78)
print("G1 TEACHER AUGMENTATION COMPLETE")
print("=" * 78)

print(
    f"Output: {OUT}"
)

print(
    "test_used=False"
)

print(
    "external_used=False"
)

print(
    "dashing2_rerun=False"
)

print(
    "G1_TEACHER_AUGMENTATION=PASS"
)
