#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("HGV_V3_revision_20260915")

PAIR_ROOT = (
    BASE
    / "05_pairs/pair_manifests_v1_20260916"
)

SPLIT_ROOT = (
    BASE
    / "04_splits/g1_g4_v1_20260916/manifests"
)

CACHE1 = (
    BASE
    / "06_teacher/teacher_cache_v1_20260916"
)

CACHE2 = (
    BASE
    / "06_teacher/teacher_cache_v2_20260916"
)

A_MANIFEST = (
    CACHE1
    / "development_A_manifest.tsv"
)

MASH_EDGES = (
    CACHE1
    / "mash/mash_triangle_edges.tsv"
)

DASHING_SET = (
    CACHE1
    / "dashing2_set/dashing2_set_similarity.tsv"
)

BAG_STATUS = (
    CACHE2
    / "dashing2_bagminhash_pilot/bagminhash_status.tsv"
)

SKANI_DB = (
    CACHE2
    / "skani/development_A_skani_db"
)

OUTDIR = (
    BASE
    / "06_teacher/teacher_geometry_audit_v1_20260916"
)

THREADS = 40

CANONICAL_SPLIT = "G2_seed20260915"

TAUS = [
    0.90,
    0.95,
    0.98,
]

MAIN_PROTOCOLS = {
    "G2",
    "G3",
    "G4",
}


def die(msg):
    print()
    print("TEACHER_GEOMETRY_AUDIT=FAIL")
    print(msg)
    sys.exit(2)


def clean(x):
    return "" if x is None else str(x).strip()


def pair_key(a, b):
    a = clean(a)
    b = clean(b)

    if a <= b:
        return (a, b)

    return (b, a)


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

        for r in rows:
            writer.writerow({
                k: r.get(k, "")
                for k in fields
            })


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(block)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def safe_float(x):
    x = clean(x)

    if not x:
        return None

    try:
        v = float(x)
    except Exception:
        return None

    if not math.isfinite(v):
        return None

    return v


def percentile(values, q):
    vals = sorted(
        float(x)
        for x in values
        if x is not None
        and math.isfinite(float(x))
    )

    if not vals:
        return None

    if len(vals) == 1:
        return vals[0]

    pos = (
        (len(vals) - 1)
        * q
    )

    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return vals[lo]

    frac = pos - lo

    return (
        vals[lo] * (1.0 - frac)
        + vals[hi] * frac
    )


def average_ranks(values):
    n = len(values)

    order = sorted(
        range(n),
        key=lambda i: values[i],
    )

    ranks = [0.0] * n

    i = 0

    while i < n:
        j = i + 1

        while (
            j < n
            and values[order[j]]
            == values[order[i]]
        ):
            j += 1

        avg_rank = (
            (i + 1)
            + j
        ) / 2.0

        for k in range(i, j):
            ranks[order[k]] = avg_rank

        i = j

    return ranks


def pearson(x, y):
    if len(x) != len(y):
        raise ValueError(
            "Pearson length mismatch"
        )

    n = len(x)

    if n < 3:
        return None

    mx = sum(x) / n
    my = sum(y) / n

    dx = [
        a - mx
        for a in x
    ]

    dy = [
        b - my
        for b in y
    ]

    num = sum(
        a * b
        for a, b in zip(dx, dy)
    )

    denx = math.sqrt(
        sum(a * a for a in dx)
    )

    deny = math.sqrt(
        sum(b * b for b in dy)
    )

    if denx == 0 or deny == 0:
        return None

    return num / (denx * deny)


def spearman(x, y):
    if len(x) < 3:
        return None

    return pearson(
        average_ranks(x),
        average_ranks(y),
    )


def roc_auc(labels, scores):
    if len(labels) != len(scores):
        raise ValueError("AUC mismatch")

    pos = sum(labels)
    neg = len(labels) - pos

    if pos == 0 or neg == 0:
        return None

    ranks = average_ranks(scores)

    rank_sum_pos = sum(
        ranks[i]
        for i, y in enumerate(labels)
        if y == 1
    )

    return (
        rank_sum_pos
        - pos * (pos + 1) / 2.0
    ) / (pos * neg)


def average_precision(labels, scores):
    """
    Tie-stable threshold-group AP.
    """

    if len(labels) != len(scores):
        raise ValueError("AP mismatch")

    total_pos = sum(labels)

    if total_pos == 0:
        return None

    groups = defaultdict(
        lambda: [0, 0]
    )

    for y, s in zip(
        labels,
        scores,
    ):
        groups[s][0] += int(y == 1)
        groups[s][1] += int(y == 0)

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    for score in sorted(
        groups,
        reverse=True,
    ):
        gpos, gneg = groups[score]

        tp += gpos
        fp += gneg

        recall = tp / total_pos
        precision = tp / (tp + fp)

        ap += (
            recall - prev_recall
        ) * precision

        prev_recall = recall

    return ap


# ----------------------------------------------------------------------
# Input gates
# ----------------------------------------------------------------------

print("=" * 78)
print("HGV-V3 TEACHER GEOMETRY / FIDELITY AUDIT")
print("=" * 78)

required_files = [
    A_MANIFEST,
    MASH_EDGES,
    DASHING_SET,
    BAG_STATUS,
]

for p in required_files:
    if not p.is_file():
        die(f"Missing prerequisite: {p}")

if not SKANI_DB.exists():
    die(
        f"Missing skani DB: {SKANI_DB}"
    )

if OUTDIR.exists():
    die(
        f"Output exists: {OUTDIR}"
    )

tmpdir = Path(
    str(OUTDIR)
    + f".tmp_{os.getpid()}"
)

if tmpdir.exists():
    shutil.rmtree(tmpdir)

tmpdir.mkdir(
    parents=True,
    exist_ok=False,
)

try:

    # ------------------------------------------------------------------
    # Development-A path/accession map
    # ------------------------------------------------------------------

    print()
    print("[1] LOAD DEVELOPMENT-A MAP")

    a_fields, a_rows = read_tsv(
        A_MANIFEST
    )

    required_a = {
        "assembly_accession",
        "fasta_path",
    }

    missing = required_a - set(a_fields)

    if missing:
        die(
            "Development manifest missing: "
            + ",".join(sorted(missing))
        )

    acc_to_path = {}
    path_to_acc = {}

    for r in a_rows:
        acc = clean(
            r["assembly_accession"]
        )

        path = clean(
            r["fasta_path"]
        )

        if not acc or not path:
            die(
                "Empty accession/path "
                "in Development manifest"
            )

        if acc in acc_to_path:
            die(
                f"Duplicate accession: {acc}"
            )

        if path in path_to_acc:
            die(
                f"Duplicate FASTA path: {path}"
            )

        acc_to_path[acc] = path
        path_to_acc[path] = acc

    n_a = len(acc_to_path)

    expected_all_pairs = (
        n_a * (n_a - 1) // 2
    )

    print(f"Development A     : {n_a}")
    print(
        f"Expected all-pairs: "
        f"{expected_all_pairs}"
    )

    if n_a != 2374:
        die(
            f"Expected frozen A=2374; "
            f"observed {n_a}"
        )

    # ------------------------------------------------------------------
    # Locate G2/G3/G4 train + val pair manifests
    # ------------------------------------------------------------------

    print()
    print("[2] COLLECT MAIN TRAIN/VALIDATION PAIR KEYS")

    pair_sources = []

    for split_dir in sorted(
        p
        for p in PAIR_ROOT.iterdir()
        if p.is_dir()
    ):
        split_id = split_dir.name

        protocol = split_id.split("_", 1)[0]

        if protocol not in MAIN_PROTOCOLS:
            continue

        for role, filename in [
            (
                "train",
                "train_unique_pair_pool.tsv",
            ),
            (
                "val",
                "val_pairs.tsv",
            ),
        ]:
            p = split_dir / filename

            if not p.is_file():
                die(
                    f"Missing pair manifest: "
                    f"{p}"
                )

            pair_sources.append(
                (
                    split_id,
                    protocol,
                    role,
                    p,
                )
            )

    if len(pair_sources) != 18:
        die(
            f"Expected 18 train/val pair manifests; "
            f"observed {len(pair_sources)}"
        )

    wanted = set()

    source_counts = []

    for (
        split_id,
        protocol,
        role,
        p,
    ) in pair_sources:

        fields, rows = read_tsv(p)

        for r in rows:
            key = pair_key(
                r["accession_a"],
                r["accession_b"],
            )

            if key[0] == key[1]:
                die(
                    f"Self-pair found: "
                    f"{split_id} {role}"
                )

            wanted.add(key)

        source_counts.append({
            "split_id": split_id,
            "protocol": protocol,
            "role": role,
            "n_pairs": len(rows),
        })

    print(
        f"Input pair manifests : "
        f"{len(pair_sources)}"
    )
    print(
        f"Unique wanted pairs  : "
        f"{len(wanted)}"
    )

    # ------------------------------------------------------------------
    # Parse Mash full-A edge list
    # ------------------------------------------------------------------

    print()
    print("[3] PARSE MASH FULL-A GEOMETRY")

    mash_map = {}

    mash_total = 0
    mash_unknown_path = 0

    with open(
        MASH_EDGES,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            parts = line.rstrip(
                "\n"
            ).split("\t")

            if len(parts) < 5:
                die(
                    "Malformed Mash edge line"
                )

            p1 = parts[0]
            p2 = parts[1]

            if (
                p1 not in path_to_acc
                or
                p2 not in path_to_acc
            ):
                mash_unknown_path += 1
                continue

            a = path_to_acc[p1]
            b = path_to_acc[p2]

            key = pair_key(a, b)

            d = safe_float(parts[2])
            pv = safe_float(parts[3])

            if d is None:
                die(
                    f"Invalid Mash distance: "
                    f"{parts[2]}"
                )

            shared = clean(parts[4])

            shared_num = ""
            shared_den = ""

            if "/" in shared:
                left, right = (
                    shared.split("/", 1)
                )

                shared_num = left
                shared_den = right

            mash_total += 1

            if key in wanted:
                if key in mash_map:
                    die(
                        f"Duplicate Mash edge: "
                        f"{key}"
                    )

                mash_map[key] = {
                    "mash_distance": d,
                    "mash_similarity": (
                        max(
                            0.0,
                            min(
                                1.0,
                                1.0 - d,
                            ),
                        )
                    ),
                    "mash_pvalue": pv,
                    "mash_shared": shared,
                    "mash_shared_num": (
                        shared_num
                    ),
                    "mash_shared_den": (
                        shared_den
                    ),
                }

            if (
                mash_total % 500000
                == 0
            ):
                print(
                    f"Mash edges parsed: "
                    f"{mash_total}"
                )

    print(
        f"Mash total edges : {mash_total}"
    )
    print(
        f"Mash wanted found: "
        f"{len(mash_map)}"
    )

    if mash_unknown_path:
        die(
            f"Mash unknown paths: "
            f"{mash_unknown_path}"
        )

    if (
        mash_total
        != expected_all_pairs
    ):
        die(
            "Mash full-pair count mismatch: "
            f"{mash_total} vs "
            f"{expected_all_pairs}"
        )

    if len(mash_map) != len(wanted):
        missing_keys = (
            wanted - set(mash_map)
        )

        print(
            "Example missing Mash keys:",
            list(missing_keys)[:10],
        )

        die(
            f"Mash teacher missing "
            f"{len(missing_keys)} wanted pairs"
        )

    # ------------------------------------------------------------------
    # Parse Dashing2 SetSketch full-A upper-triangular matrix
    # ------------------------------------------------------------------

    print()
    print("[4] PARSE DASHING2 SETSKETCH GEOMETRY")

    d2_map = {}

    sources = None
    d2_total = 0
    d2_rows = 0

    with open(
        DASHING_SET,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.rstrip("\n")

            if not line:
                continue

            if line.startswith(
                "#Sources\t"
            ):
                sources = (
                    line.split("\t")[1:]
                )

                print(
                    f"Dashing2 sources: "
                    f"{len(sources)}"
                )

                if len(sources) != n_a:
                    die(
                        "Dashing2 source count "
                        f"{len(sources)} != {n_a}"
                    )

                for p in sources:
                    if p not in path_to_acc:
                        die(
                            "Dashing2 unknown source "
                            f"path: {p}"
                        )

                continue

            if line.startswith("#"):
                continue

            if sources is None:
                die(
                    "Dashing2 #Sources "
                    "header not found"
                )

            parts = line.split("\t")

            if len(parts) != n_a + 1:
                die(
                    "Unexpected Dashing2 row "
                    f"width={len(parts)} "
                    f"expected={n_a+1}"
                )

            row_path = parts[0]

            if row_path not in path_to_acc:
                die(
                    f"Unknown Dashing2 row path: "
                    f"{row_path}"
                )

            a = path_to_acc[
                row_path
            ]

            values = parts[1:]

            for j, token in enumerate(
                values
            ):
                token = clean(token)

                if (
                    not token
                    or token == "-"
                ):
                    continue

                sim = safe_float(token)

                if sim is None:
                    die(
                        f"Invalid Dashing2 value: "
                        f"{token}"
                    )

                b_path = sources[j]
                b = path_to_acc[
                    b_path
                ]

                if a == b:
                    continue

                key = pair_key(a, b)

                d2_total += 1

                if key in wanted:
                    if key in d2_map:
                        die(
                            "Duplicate Dashing2 "
                            f"edge: {key}"
                        )

                    d2_map[key] = {
                        "dashing2_set_similarity": (
                            sim
                        ),
                        "dashing2_set_distance": (
                            max(
                                0.0,
                                min(
                                    1.0,
                                    1.0 - sim,
                                ),
                            )
                        ),
                    }

            d2_rows += 1

            if d2_rows % 250 == 0:
                print(
                    f"Dashing2 rows parsed: "
                    f"{d2_rows}/{n_a}"
                )

    print(
        f"Dashing2 rows       : "
        f"{d2_rows}"
    )
    print(
        f"Dashing2 total edges: "
        f"{d2_total}"
    )
    print(
        f"Dashing2 wanted     : "
        f"{len(d2_map)}"
    )

    if d2_rows != n_a:
        die(
            f"Dashing2 row count "
            f"{d2_rows} != {n_a}"
        )

    if d2_total != expected_all_pairs:
        die(
            "Dashing2 full-pair count "
            f"{d2_total} != "
            f"{expected_all_pairs}"
        )

    if len(d2_map) != len(wanted):
        missing_keys = (
            wanted - set(d2_map)
        )

        print(
            "Example missing D2 keys:",
            list(missing_keys)[:10],
        )

        die(
            f"Dashing2 missing "
            f"{len(missing_keys)} wanted pairs"
        )

    # ------------------------------------------------------------------
    # Write teacher-augmented train/val manifests
    # ------------------------------------------------------------------

    print()
    print("[5] JOIN TEACHERS TO TRAIN/VALIDATION PAIRS")

    teacher_fields = [
        "mash_distance",
        "mash_similarity",
        "mash_pvalue",
        "mash_shared",
        "mash_shared_num",
        "mash_shared_den",
        "dashing2_set_similarity",
        "dashing2_set_distance",
    ]

    joined_counts = []

    canonical_val_rows = []

    for (
        split_id,
        protocol,
        role,
        p,
    ) in pair_sources:

        fields, rows = read_tsv(p)

        out_rows = []

        for r in rows:

            key = pair_key(
                r["accession_a"],
                r["accession_b"],
            )

            mr = mash_map[key]
            dr = d2_map[key]

            out = dict(r)

            out.update(mr)
            out.update(dr)

            out_rows.append(out)

            if (
                split_id
                == CANONICAL_SPLIT
                and role == "val"
            ):
                canonical_val_rows.append(
                    out
                )

        outpath = (
            tmpdir
            / "teacher_augmented"
            / split_id
            / (
                "train_teacher.tsv"
                if role == "train"
                else "val_teacher.tsv"
            )
        )

        write_tsv(
            outpath,
            out_rows,
            fields + teacher_fields,
        )

        joined_counts.append({
            "split_id": split_id,
            "protocol": protocol,
            "role": role,
            "n_pairs": len(rows),
            "mash_missing": 0,
            "dashing2_missing": 0,
            "gate": "PASS",
        })

        print(
            f"{split_id:22s} "
            f"{role:5s} "
            f"n={len(rows):5d} PASS"
        )

    if not canonical_val_rows:
        die(
            "Canonical G2 validation "
            "rows not found"
        )

    # ------------------------------------------------------------------
    # Build canonical G2 validation FASTA list from SPLIT manifest
    # ------------------------------------------------------------------

    print()
    print("[6] BUILD CANONICAL G2 VALIDATION GENOME LIST")

    split_manifest = (
        SPLIT_ROOT
        / f"{CANONICAL_SPLIT}.tsv"
    )

    sf, sr = read_tsv(
        split_manifest
    )

    val_accs = sorted({
        clean(
            r["assembly_accession_final"]
        )
        for r in sr
        if clean(
            r["role"]
        ) == "val"
    })

    if not val_accs:
        die(
            "No canonical G2 val genomes"
        )

    val_fasta = []

    for acc in val_accs:
        if acc not in acc_to_path:
            die(
                f"Val accession missing "
                f"from A manifest: {acc}"
            )

        p = acc_to_path[acc]

        if not Path(p).is_file():
            die(
                f"Missing val FASTA: {p}"
            )

        val_fasta.append(p)

    print(
        f"Canonical G2 val genomes: "
        f"{len(val_accs)}"
    )

    val_list_path = (
        tmpdir
        / "canonical_g2_val_fasta.list"
    )

    with open(
        val_list_path,
        "w",
        encoding="utf-8",
    ) as f:
        for p in val_fasta:
            f.write(p + "\n")

    # ------------------------------------------------------------------
    # skani triangle only on canonical G2 validation genomes
    # ------------------------------------------------------------------

    print()
    print("[7] SKANI CANONICAL G2 VALIDATION ANI/AF CONTROL")

    skani_edges_path = (
        tmpdir
        / "skani_g2val_edges.tsv"
    )

    skani_stderr = (
        tmpdir
        / "skani_g2val.stderr.txt"
    )

    cmd = [
        "skani",
        "triangle",
        "-t",
        str(THREADS),
        "-E",
    ] + val_fasta

    print(
        f"skani triangle genomes="
        f"{len(val_fasta)} "
        f"threads={THREADS}"
    )

    with open(
        skani_edges_path,
        "w",
        encoding="utf-8",
    ) as fo, open(
        skani_stderr,
        "w",
        encoding="utf-8",
    ) as fe:

        proc = subprocess.run(
            cmd,
            stdout=fo,
            stderr=fe,
            text=True,
        )

    if proc.returncode != 0:
        print(
            skani_stderr.read_text(
                encoding="utf-8",
            )
        )

        die(
            f"skani triangle failed: "
            f"{proc.returncode}"
        )

    if not skani_edges_path.is_file():
        die(
            "skani edge file missing"
        )

    # ------------------------------------------------------------------
    # Parse skani output
    # ------------------------------------------------------------------

    skani_map = {}

    skani_total_edges = 0

    with open(
        skani_edges_path,
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(
                "Ref_file\t"
            ):
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")

            if len(parts) < 5:
                continue

            ref_path = parts[0]
            query_path = parts[1]

            if (
                ref_path not in path_to_acc
                or
                query_path not in path_to_acc
            ):
                die(
                    "Unknown path in skani "
                    f"output: {ref_path} / "
                    f"{query_path}"
                )

            ani = safe_float(parts[2])
            af_ref = safe_float(parts[3])
            af_query = safe_float(parts[4])

            if (
                ani is None
                or af_ref is None
                or af_query is None
            ):
                continue

            key = pair_key(
                path_to_acc[ref_path],
                path_to_acc[query_path],
            )

            skani_total_edges += 1

            if key not in skani_map:
                skani_map[key] = {
                    "skani_ani": ani,
                    "skani_af_ref": af_ref,
                    "skani_af_query": (
                        af_query
                    ),
                    "skani_af_mean": (
                        (af_ref + af_query)
                        / 2.0
                    ),
                }

    print(
        f"skani reported edges: "
        f"{skani_total_edges}"
    )

    # ------------------------------------------------------------------
    # Canonical G2 validation join
    # ------------------------------------------------------------------

    print()
    print("[8] CANONICAL G2 VALIDATION FIDELITY TABLE")

    canonical_full = []

    coverage_by_type = Counter()
    total_by_type = Counter()

    for r in canonical_val_rows:

        key = pair_key(
            r["accession_a"],
            r["accession_b"],
        )

        out = dict(r)

        pair_type = clean(
            r["pair_type"]
        )

        total_by_type[
            pair_type
        ] += 1

        if key in skani_map:
            s = skani_map[key]

            out.update(s)
            out[
                "skani_reported"
            ] = 1

            coverage_by_type[
                pair_type
            ] += 1

        else:
            out[
                "skani_ani"
            ] = ""

            out[
                "skani_af_ref"
            ] = ""

            out[
                "skani_af_query"
            ] = ""

            out[
                "skani_af_mean"
            ] = ""

            out[
                "skani_reported"
            ] = 0

        canonical_full.append(out)

    canonical_fields = (
        list(
            canonical_val_rows[0].keys()
        )
        + [
            "skani_ani",
            "skani_af_ref",
            "skani_af_query",
            "skani_af_mean",
            "skani_reported",
        ]
    )

    write_tsv(
        tmpdir
        / "canonical_g2_val_teacher_skani.tsv",
        canonical_full,
        canonical_fields,
    )

    # ------------------------------------------------------------------
    # Dynamic range audit
    # ------------------------------------------------------------------

    print()
    print("[9] TEACHER DYNAMIC-RANGE AUDIT")

    dynamic_rows = []

    strata = [
        "all",
        "positive",
        "negative",
        "hard",
        "medium",
        "easy",
    ]

    teacher_defs = [
        (
            "Mash",
            "mash_distance",
        ),
        (
            "Dashing2_SetSketch",
            "dashing2_set_distance",
        ),
    ]

    for teacher, field in teacher_defs:

        for stratum in strata:

            vals = []

            for r in canonical_full:

                pt = clean(
                    r["pair_type"]
                )

                keep = False

                if stratum == "all":
                    keep = True

                elif stratum == "positive":
                    keep = (
                        int(r["label"])
                        == 1
                    )

                elif stratum == "negative":
                    keep = (
                        int(r["label"])
                        == 0
                    )

                else:
                    keep = (
                        pt == stratum
                    )

                if not keep:
                    continue

                v = safe_float(
                    r[field]
                )

                if v is not None:
                    vals.append(v)

            if not vals:
                continue

            saturated = sum(
                1
                for v in vals
                if v >= 1.0 - 1e-12
            )

            zero = sum(
                1
                for v in vals
                if v <= 1e-12
            )

            dynamic_rows.append({
                "teacher": teacher,
                "stratum": stratum,
                "n": len(vals),
                "n_unique_values": (
                    len(set(vals))
                ),
                "mean_distance": (
                    sum(vals) / len(vals)
                ),
                "q00": percentile(
                    vals,
                    0.00,
                ),
                "q05": percentile(
                    vals,
                    0.05,
                ),
                "q25": percentile(
                    vals,
                    0.25,
                ),
                "q50": percentile(
                    vals,
                    0.50,
                ),
                "q75": percentile(
                    vals,
                    0.75,
                ),
                "q95": percentile(
                    vals,
                    0.95,
                ),
                "q100": percentile(
                    vals,
                    1.00,
                ),
                "distance_eq_1_n": (
                    saturated
                ),
                "distance_eq_1_fraction": (
                    saturated / len(vals)
                ),
                "distance_eq_0_n": zero,
                "distance_eq_0_fraction": (
                    zero / len(vals)
                ),
            })

    write_tsv(
        tmpdir
        / "audits/teacher_dynamic_range.tsv",
        dynamic_rows,
        [
            "teacher",
            "stratum",
            "n",
            "n_unique_values",
            "mean_distance",
            "q00",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "q100",
            "distance_eq_1_n",
            "distance_eq_1_fraction",
            "distance_eq_0_n",
            "distance_eq_0_fraction",
        ],
    )

    # ------------------------------------------------------------------
    # tau informative-negative coverage
    # ------------------------------------------------------------------

    tau_rows = []

    negative_rows = [
        r
        for r in canonical_full
        if int(r["label"]) == 0
    ]

    for teacher, field in teacher_defs:

        distances = [
            safe_float(r[field])
            for r in negative_rows
        ]

        distances = [
            x
            for x in distances
            if x is not None
        ]

        for tau in TAUS:

            included = sum(
                1
                for d in distances
                if d < tau
            )

            tau_rows.append({
                "teacher": teacher,
                "tau": tau,
                "negative_n": (
                    len(distances)
                ),
                "informative_n": (
                    included
                ),
                "informative_fraction": (
                    included
                    / len(distances)
                ),
                "saturated_or_excluded_n": (
                    len(distances)
                    - included
                ),
            })

    write_tsv(
        tmpdir
        / "audits/tau_candidate_coverage.tsv",
        tau_rows,
        [
            "teacher",
            "tau",
            "negative_n",
            "informative_n",
            "informative_fraction",
            "saturated_or_excluded_n",
        ],
    )

    # ------------------------------------------------------------------
    # skani coverage by taxonomy stratum
    # ------------------------------------------------------------------

    coverage_rows = []

    for pt in [
        "positive",
        "hard",
        "medium",
        "easy",
    ]:

        total = total_by_type[
            pt
        ]

        reported = coverage_by_type[
            pt
        ]

        coverage_rows.append({
            "pair_type": pt,
            "n_pairs": total,
            "skani_reported": reported,
            "skani_missing": (
                total - reported
            ),
            "coverage_fraction": (
                reported / total
                if total
                else ""
            ),
        })

    total_pairs = len(
        canonical_full
    )

    total_reported = sum(
        coverage_by_type.values()
    )

    coverage_rows.append({
        "pair_type": "all",
        "n_pairs": total_pairs,
        "skani_reported": (
            total_reported
        ),
        "skani_missing": (
            total_pairs
            - total_reported
        ),
        "coverage_fraction": (
            total_reported
            / total_pairs
        ),
    })

    write_tsv(
        tmpdir
        / "audits/skani_coverage.tsv",
        coverage_rows,
        [
            "pair_type",
            "n_pairs",
            "skani_reported",
            "skani_missing",
            "coverage_fraction",
        ],
    )

    # ------------------------------------------------------------------
    # Teacher vs skani fidelity
    # ------------------------------------------------------------------

    print()
    print("[10] TEACHER vs SKANI ANI/AF FIDELITY")

    fidelity_rows = []

    covered = [
        r
        for r in canonical_full
        if int(
            r["skani_reported"]
        ) == 1
    ]

    covered_85_100 = [
        r
        for r in covered
        if (
            safe_float(
                r["skani_ani"]
            )
            is not None
            and
            safe_float(
                r["skani_ani"]
            ) >= 85.0
        )
    ]

    for subset_name, subset in [
        (
            "all_skani_reported",
            covered,
        ),
        (
            "ANI_85_100",
            covered_85_100,
        ),
    ]:

        for (
            teacher,
            dist_field,
            sim_field,
        ) in [
            (
                "Mash",
                "mash_distance",
                "mash_similarity",
            ),
            (
                "Dashing2_SetSketch",
                "dashing2_set_distance",
                "dashing2_set_similarity",
            ),
        ]:

            sim = []
            dist = []
            ani = []
            af = []

            for r in subset:

                s = safe_float(
                    r[sim_field]
                )

                d = safe_float(
                    r[dist_field]
                )

                a = safe_float(
                    r["skani_ani"]
                )

                m = safe_float(
                    r["skani_af_mean"]
                )

                if None in (
                    s,
                    d,
                    a,
                    m,
                ):
                    continue

                sim.append(s)
                dist.append(d)
                ani.append(a)
                af.append(m)

            fidelity_rows.append({
                "subset": subset_name,
                "teacher": teacher,
                "n": len(sim),
                "pearson_similarity_vs_ANI": (
                    pearson(sim, ani)
                ),
                "spearman_similarity_vs_ANI": (
                    spearman(sim, ani)
                ),
                "pearson_distance_vs_ANI": (
                    pearson(dist, ani)
                ),
                "spearman_distance_vs_ANI": (
                    spearman(dist, ani)
                ),
                "spearman_similarity_vs_AFmean": (
                    spearman(sim, af)
                ),
            })

    write_tsv(
        tmpdir
        / "audits/skani_teacher_fidelity.tsv",
        fidelity_rows,
        [
            "subset",
            "teacher",
            "n",
            "pearson_similarity_vs_ANI",
            "spearman_similarity_vs_ANI",
            "pearson_distance_vs_ANI",
            "spearman_distance_vs_ANI",
            "spearman_similarity_vs_AFmean",
        ],
    )

    # ------------------------------------------------------------------
    # Direct teacher discrimination on canonical G2 validation
    # ------------------------------------------------------------------

    print()
    print("[11] DIRECT TEACHER VALIDATION DISCRIMINATION")

    labels = [
        int(r["label"])
        for r in canonical_full
    ]

    candidate_rows = []

    for (
        teacher,
        distance_field,
        similarity_field,
    ) in [
        (
            "Mash",
            "mash_distance",
            "mash_similarity",
        ),
        (
            "Dashing2_SetSketch",
            "dashing2_set_distance",
            "dashing2_set_similarity",
        ),
    ]:

        scores = [
            float(
                r[similarity_field]
            )
            for r in canonical_full
        ]

        distances = [
            float(
                r[distance_field]
            )
            for r in canonical_full
        ]

        neg_dist = [
            distances[i]
            for i, y in enumerate(
                labels
            )
            if y == 0
        ]

        sat_neg = sum(
            1
            for x in neg_dist
            if x >= 1.0 - 1e-12
        )

        skani_match = next(
            (
                x
                for x in fidelity_rows
                if (
                    x["subset"]
                    == "ANI_85_100"
                    and
                    x["teacher"]
                    == teacher
                )
            ),
            None,
        )

        candidate_rows.append({
            "teacher": teacher,
            "canonical_split": (
                CANONICAL_SPLIT
            ),
            "val_pairs": len(labels),
            "direct_val_AUROC": (
                roc_auc(
                    labels,
                    scores,
                )
            ),
            "direct_val_AUPRC": (
                average_precision(
                    labels,
                    scores,
                )
            ),
            "negative_saturation_n": (
                sat_neg
            ),
            "negative_saturation_fraction": (
                sat_neg
                / len(neg_dist)
            ),
            "ANI85_100_n": (
                skani_match["n"]
                if skani_match
                else ""
            ),
            "ANI85_100_spearman_similarity_vs_ANI": (
                skani_match[
                    "spearman_similarity_vs_ANI"
                ]
                if skani_match
                else ""
            ),
            "primary_teacher_status": (
                "NOT_FROZEN_PENDING_HGV_VALIDATION"
            ),
        })

    write_tsv(
        tmpdir
        / "teacher_candidate_summary.tsv",
        candidate_rows,
        [
            "teacher",
            "canonical_split",
            "val_pairs",
            "direct_val_AUROC",
            "direct_val_AUPRC",
            "negative_saturation_n",
            "negative_saturation_fraction",
            "ANI85_100_n",
            "ANI85_100_spearman_similarity_vs_ANI",
            "primary_teacher_status",
        ],
    )

    for r in candidate_rows:
        print(
            f"{r['teacher']:22s} "
            f"AUROC={r['direct_val_AUROC']:.6f} "
            f"AUPRC={r['direct_val_AUPRC']:.6f} "
            f"neg_sat="
            f"{r['negative_saturation_fraction']:.4f} "
            f"ANI_rho="
            f"{r['ANI85_100_spearman_similarity_vs_ANI']}"
        )

    # ------------------------------------------------------------------
    # BagMinHash feasibility provenance
    # ------------------------------------------------------------------

    bf, br = read_tsv(
        BAG_STATUS
    )

    bag_status = {
        clean(r["metric"]):
        clean(r["value"])
        for r in br
    }

    # ------------------------------------------------------------------
    # Audit summaries
    # ------------------------------------------------------------------

    write_tsv(
        tmpdir
        / "audits/teacher_join_coverage.tsv",
        joined_counts,
        [
            "split_id",
            "protocol",
            "role",
            "n_pairs",
            "mash_missing",
            "dashing2_missing",
            "gate",
        ],
    )

    write_tsv(
        tmpdir
        / "audits/input_pair_counts.tsv",
        source_counts,
        [
            "split_id",
            "protocol",
            "role",
            "n_pairs",
        ],
    )

    config = {
        "status": "PASS",
        "development_genomes": n_a,
        "canonical_teacher_selection_split": (
            CANONICAL_SPLIT
        ),
        "teacher_selection_data_scope": (
            "Development train/validation only"
        ),
        "test_pairs_read": False,
        "external_genomes_used": False,
        "mash": {
            "version": "2.3",
            "k": 21,
            "sketch_size": 1000,
            "distance_field": (
                "native Mash distance"
            ),
        },
        "dashing2_set": {
            "version": "2.1.20",
            "k": 31,
            "sketch_size": 1024,
            "mode": (
                "one-permutation SetSketch"
            ),
            "distance_definition": (
                "1 - native similarity"
            ),
        },
        "dashing2_bagminhash": {
            "status": (
                bag_status.get(
                    "classification",
                    "UNKNOWN",
                )
            ),
            "full_p40": (
                bag_status.get(
                    "full2374_p40_status",
                    "",
                )
            ),
            "pilot_p1_exit": (
                bag_status.get(
                    "pilot_p1_exit",
                    "",
                )
            ),
            "pilot_p8_exit": (
                bag_status.get(
                    "pilot_p8_exit",
                    "",
                )
            ),
            "used_for_primary_teacher_selection": (
                False
            ),
        },
        "skani": {
            "version": "0.3.2",
            "scope": (
                "canonical G2 validation "
                "ANI/AF control only"
            ),
            "missing_policy": (
                "preserve missing; never "
                "impute ANI"
            ),
        },
        "tau_candidates": TAUS,
        "primary_teacher_frozen": False,
        "next_gate": (
            "compare downstream HGV validation "
            "AUC/AUPRC for Mash vs Dashing2 "
            "before primary-teacher freeze"
        ),
    }

    with open(
        tmpdir
        / "teacher_geometry_config.json",
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

    # ------------------------------------------------------------------
    # Human report
    # ------------------------------------------------------------------

    report_path = (
        tmpdir
        / "TEACHER_GEOMETRY_AUDIT_REPORT.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HGV-V3 TEACHER GEOMETRY / "
            "FIDELITY AUDIT\n"
        )

        f.write("=" * 78 + "\n\n")

        f.write("STATUS: PASS\n\n")

        f.write(
            f"Development A genomes: "
            f"{n_a}\n"
        )

        f.write(
            f"Full Mash edges: "
            f"{mash_total}\n"
        )

        f.write(
            f"Full Dashing2 SetSketch edges: "
            f"{d2_total}\n"
        )

        f.write(
            "Test pair manifests read: NO\n"
        )

        f.write(
            "External genomes used: NO\n\n"
        )

        f.write(
            "Dashing2 BagMinHash:\n"
        )

        f.write(
            "  full p40: "
            + bag_status.get(
                "full2374_p40_status",
                "",
            )
            + "\n"
        )

        f.write(
            "  pilot p1 exit: "
            + bag_status.get(
                "pilot_p1_exit",
                "",
            )
            + "\n"
        )

        f.write(
            "  pilot p8 exit: "
            + bag_status.get(
                "pilot_p8_exit",
                "",
            )
            + "\n"
        )

        f.write(
            "  classification: "
            + bag_status.get(
                "classification",
                "",
            )
            + "\n"
        )

        f.write(
            "  primary candidate: NO\n\n"
        )

        f.write(
            "Canonical teacher-selection "
            f"validation: {CANONICAL_SPLIT}\n"
        )

        f.write(
            f"Validation pairs: "
            f"{len(canonical_full)}\n"
        )

        f.write(
            f"skani reported on frozen "
            f"validation pairs: "
            f"{total_reported}/"
            f"{total_pairs} "
            f"({total_reported/total_pairs:.6f})\n\n"
        )

        f.write(
            "Teacher candidates:\n"
        )

        for r in candidate_rows:
            f.write(
                f"  {r['teacher']}: "
                f"direct AUROC="
                f"{r['direct_val_AUROC']:.6f}; "
                f"AUPRC="
                f"{r['direct_val_AUPRC']:.6f}; "
                f"negative saturation="
                f"{r['negative_saturation_fraction']:.6f}; "
                f"ANI85-100 Spearman="
                f"{r['ANI85_100_spearman_similarity_vs_ANI']}\n"
            )

        f.write("\n")
        f.write(
            "PRIMARY TEACHER STATUS: "
            "NOT FROZEN\n"
        )

        f.write(
            "Reason: V3 requires teacher "
            "saturation + ANI monotonicity + "
            "downstream HGV validation "
            "AUC/AUPRC before freezing T1/T2.\n"
        )

        f.write(
            "Test/external remain untouched.\n\n"
        )

        f.write(
            "TEACHER_GEOMETRY_AUDIT=PASS\n"
        )

    # ------------------------------------------------------------------
    # Environment/provenance
    # ------------------------------------------------------------------

    env_path = (
        tmpdir
        / "environment.txt"
    )

    with open(
        env_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"python={sys.version}\n"
        )

        f.write(
            f"conda_env="
            f"{os.environ.get('CONDA_DEFAULT_ENV','')}\n"
        )

        f.write(
            f"threads={THREADS}\n"
        )

        for cmd0 in [
            ["mash", "--version"],
            ["skani", "--version"],
        ]:
            try:
                p = subprocess.run(
                    cmd0,
                    capture_output=True,
                    text=True,
                )

                f.write(
                    " ".join(cmd0)
                    + ": "
                    + (
                        p.stdout.strip()
                        or p.stderr.strip()
                    )
                    + "\n"
                )
            except Exception as e:
                f.write(
                    f"{cmd0[0]}: "
                    f"{repr(e)}\n"
                )

        f.write(
            "dashing2=2.1.20\n"
        )

    # ------------------------------------------------------------------
    # SHA256 everything except checksum itself
    # ------------------------------------------------------------------

    output_files = sorted([
        p
        for p in tmpdir.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.txt"
    ])

    with open(
        tmpdir / "SHA256SUMS.txt",
        "w",
        encoding="utf-8",
    ) as f:

        for p in output_files:

            rel = p.relative_to(
                tmpdir
            )

            f.write(
                f"{sha256_file(p)}  "
                f"{rel}\n"
            )

    # ------------------------------------------------------------------
    # Atomic freeze
    # ------------------------------------------------------------------

    tmpdir.rename(
        OUTDIR
    )

except Exception:
    if tmpdir.exists():
        print()
        print(
            "Temporary output preserved for "
            "diagnosis:"
        )
        print(tmpdir)

    raise


print()
print("=" * 78)
print("TEACHER GEOMETRY / FIDELITY AUDIT COMPLETE")
print("=" * 78)
print(f"Output: {OUTDIR}")
print()
print("TEACHER_GEOMETRY_AUDIT=PASS")
