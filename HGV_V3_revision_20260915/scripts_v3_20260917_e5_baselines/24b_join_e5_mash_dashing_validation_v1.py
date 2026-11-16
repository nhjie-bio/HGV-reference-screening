#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import pandas as pd


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

MANIFEST = (
    BASE /
    "01_manifest/"
    "final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

TASKROOT = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917/"
    "00_native_validation/tasks"
)

MASH = (
    BASE /
    "06_teacher/"
    "teacher_cache_v1_20260916/"
    "mash/"
    "mash_triangle_edges.tsv"
)

DASH = (
    BASE /
    "06_teacher/"
    "teacher_cache_v1_20260916/"
    "dashing2_set/"
    "dashing2_set_similarity.tsv"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

MASH_OUT = (
    E5 /
    "02_mash/"
    "validation_cache_join_v1"
)

DASH_OUT = (
    E5 /
    "03_dashing2/"
    "validation_cache_join_v1"
)

AUDIT_OUT = (
    E5 /
    "00_native_validation/"
    "cache_join_audit_v1"
)

for p in [
    MASH_OUT,
    DASH_OUT,
    AUDIT_OUT,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )


EXPECTED = {
    ("G2", "20260915"): 3874,
    ("G2", "20260916"): 3872,
    ("G2", "20260917"): 3874,

    ("G3", "20260915"): 3764,
    ("G3", "20260916"): 3764,
    ("G3", "20260917"): 3830,

    ("G4", "20260915"): 3916,
    ("G4", "20260916"): 3786,
    ("G4", "20260917"): 3832,
}

EXPECTED_TOTAL = 34512
EXPECTED_A_GENOMES = 2374

EXPECTED_MASH_EDGES = (
    EXPECTED_A_GENOMES *
    (EXPECTED_A_GENOMES - 1) //
    2
)


def canon_pair(a, b):

    a = str(a)
    b = str(b)

    if a == b:
        raise RuntimeError(
            f"Self-pair detected: {a}"
        )

    return (
        (a, b)
        if a < b
        else (b, a)
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


print("=" * 90)
print(
    "E5 MASH + DASHING2 "
    "VALIDATION CACHE JOIN v1"
)
print("=" * 90)


# ================================================================
# Frozen master manifest
# ================================================================

manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
    dtype=str,
)

for c in [
    "assembly_accession_final",
    "fasta_path",
]:

    if c not in manifest.columns:
        raise RuntimeError(
            f"Manifest missing: {c}"
        )


path_to_acc = dict(
    zip(
        manifest["fasta_path"],
        manifest[
            "assembly_accession_final"
        ],
    )
)

acc_to_path = dict(
    zip(
        manifest[
            "assembly_accession_final"
        ],
        manifest["fasta_path"],
    )
)


print()
print(
    "manifest genomes =",
    len(manifest)
)


# ================================================================
# Load 9 validation tasks
# ================================================================

print()
print(
    "===== LOAD VALIDATION TASKS ====="
)

split_frames = {}

target_keys = set()

total_occurrences = 0


for protocol in [
    "G2",
    "G3",
    "G4",
]:

    for seed in [
        "20260915",
        "20260916",
        "20260917",
    ]:

        src = (
            TASKROOT /
            f"{protocol}_seed{seed}" /
            "native_validation_pairs.tsv"
        )

        if not src.is_file():
            raise FileNotFoundError(
                src
            )

        df = pd.read_csv(
            src,
            sep="\t",
            dtype=str,
        )

        expected = EXPECTED[
            (protocol, seed)
        ]

        if len(df) != expected:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"expected {expected}, "
                f"got {len(df)}"
            )

        keys = [
            canon_pair(a, b)
            for a, b in zip(
                df["accession_a"],
                df["accession_b"],
            )
        ]

        df = df.copy()

        df["_pair_key"] = keys

        split_frames[
            (protocol, seed)
        ] = df

        target_keys.update(
            keys
        )

        total_occurrences += len(df)

        print(
            f"{protocol} seed{seed}: "
            f"{len(df)}"
        )


if total_occurrences != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} "
        f"validation pair occurrences, "
        f"got {total_occurrences}"
    )


print()
print(
    "validation occurrences =",
    total_occurrences
)

print(
    "unique unordered pairs =",
    len(target_keys)
)


# ================================================================
# Mash: scan complete A triangle once
# ================================================================

print()
print(
    "===== SCAN MASH CACHE ====="
)

mash_lookup = {}

mash_lines = 0

mash_unknown_paths = 0

mash_sha = hashlib.sha256()


with open(MASH, "rb") as f:

    for raw in f:

        mash_sha.update(raw)

        mash_lines += 1

        parts = (
            raw.decode("utf-8")
            .rstrip("\n")
            .split("\t")
        )

        if len(parts) != 5:
            raise RuntimeError(
                f"Mash line {mash_lines}: "
                f"{len(parts)} fields"
            )

        p1 = parts[0]
        p2 = parts[1]

        a1 = path_to_acc.get(p1)
        a2 = path_to_acc.get(p2)

        if (
            a1 is None
            or a2 is None
        ):
            mash_unknown_paths += 1
            continue

        key = canon_pair(
            a1,
            a2,
        )

        if key not in target_keys:
            continue

        if key in mash_lookup:
            raise RuntimeError(
                f"Duplicate Mash target "
                f"pair: {key}"
            )

        distance = float(
            parts[2]
        )

        pvalue = float(
            parts[3]
        )

        shared = parts[4]

        numerator, denominator = (
            shared.split("/", 1)
        )

        mash_lookup[key] = {
            "mash_distance":
                distance,

            "mash_similarity":
                1.0 - distance,

            "mash_pvalue":
                pvalue,

            "mash_shared_hashes":
                shared,

            "mash_shared_numerator":
                int(numerator),

            "mash_shared_denominator":
                int(denominator),
        }


print(
    "Mash lines       =",
    mash_lines
)

print(
    "Mash targets     =",
    len(mash_lookup)
)


if mash_lines != EXPECTED_MASH_EDGES:
    raise RuntimeError(
        "Mash cache edge count "
        "does not equal complete "
        "2374-genome triangle"
    )

if mash_unknown_paths:
    raise RuntimeError(
        f"Mash unknown paths: "
        f"{mash_unknown_paths}"
    )


mash_missing = (
    target_keys -
    set(mash_lookup)
)

if mash_missing:
    raise RuntimeError(
        f"Mash missing "
        f"{len(mash_missing)} "
        "validation pair(s)"
    )


print(
    "MASH VALIDATION COVERAGE = 100%"
)


# ================================================================
# Dashing2 upper triangular matrix
# ================================================================

print()
print(
    "===== PARSE DASHING2 CACHE ====="
)

dash_lookup = {}

dash_sha = hashlib.sha256()


with open(DASH, "rb") as f:

    raw1 = f.readline()
    raw2 = f.readline()
    raw3 = f.readline()

    dash_sha.update(raw1)
    dash_sha.update(raw2)
    dash_sha.update(raw3)

    source_fields = (
        raw3.decode("utf-8")
        .rstrip("\n")
        .split("\t")
    )

    if (
        not source_fields
        or source_fields[0]
        != "#Sources"
    ):
        raise RuntimeError(
            "Invalid Dashing2 "
            "#Sources header"
        )

    sources = (
        source_fields[1:]
    )

    if (
        len(sources)
        != EXPECTED_A_GENOMES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_A_GENOMES} "
            f"Dashing2 sources, "
            f"got {len(sources)}"
        )

    source_accs = []

    for p in sources:

        acc = path_to_acc.get(p)

        if acc is None:
            raise RuntimeError(
                f"Unknown Dashing2 "
                f"path: {p}"
            )

        source_accs.append(acc)


    source_index = {
        acc: i
        for i, acc in
        enumerate(source_accs)
    }


    needed_by_row = {}

    for key in target_keys:

        a, b = key

        if (
            a not in source_index
            or b not in source_index
        ):
            raise RuntimeError(
                f"Target absent from "
                f"Dashing2 source list: "
                f"{key}"
            )

        ia = source_index[a]
        ib = source_index[b]

        if ia < ib:
            r = ia
            c = ib
        else:
            r = ib
            c = ia

        needed_by_row.setdefault(
            r,
            {}
        )[c] = key


    row_count = 0

    for row_i in range(
        EXPECTED_A_GENOMES
    ):

        raw = f.readline()

        if not raw:
            raise RuntimeError(
                "Dashing2 matrix "
                "ended early"
            )

        dash_sha.update(raw)

        row_count += 1

        parts = (
            raw.decode("utf-8")
            .rstrip("\n")
            .split("\t")
        )

        if len(parts) != (
            EXPECTED_A_GENOMES + 1
        ):
            raise RuntimeError(
                f"Dashing2 row "
                f"{row_i}: "
                f"{len(parts)} fields"
            )

        if parts[0] != sources[
            row_i
        ]:
            raise RuntimeError(
                f"Dashing2 row order "
                f"mismatch: {row_i}"
            )

        needed = (
            needed_by_row.get(
                row_i
            )
        )

        if not needed:
            continue

        for col_i, key in (
            needed.items()
        ):

            token = parts[
                col_i + 1
            ]

            if token == "-":
                raise RuntimeError(
                    f"Required Dashing2 "
                    f"value missing: "
                    f"{key}"
                )

            similarity = float(
                token
            )

            if not (
                0.0
                <= similarity
                <= 1.0
            ):
                raise RuntimeError(
                    f"Bad similarity: "
                    f"{similarity}"
                )

            dash_lookup[key] = {
                "dashing2_similarity":
                    similarity,

                "dashing2_distance":
                    1.0 - similarity,
            }


    extra_nonempty = 0

    for raw in f:

        dash_sha.update(raw)

        if raw.strip():
            extra_nonempty += 1


    if extra_nonempty:
        raise RuntimeError(
            "Extra Dashing2 "
            "matrix rows detected"
        )


print(
    "Dashing2 rows    =",
    row_count
)

print(
    "Dashing targets  =",
    len(dash_lookup)
)


dash_missing = (
    target_keys -
    set(dash_lookup)
)

if dash_missing:
    raise RuntimeError(
        f"Dashing2 missing "
        f"{len(dash_missing)} "
        "validation pair(s)"
    )


print(
    "DASHING2 VALIDATION COVERAGE = 100%"
)


# ================================================================
# Join back to nine validation manifests
# ================================================================

print()
print(
    "===== WRITE VALIDATION RESULTS ====="
)

coverage_rows = []

score_rows = []

sha_rows = []


for protocol in [
    "G2",
    "G3",
    "G4",
]:

    for seed in [
        "20260915",
        "20260916",
        "20260917",
    ]:

        df = split_frames[
            (protocol, seed)
        ].copy()

        mash_extra = pd.DataFrame(
            [
                mash_lookup[k]
                for k in
                df["_pair_key"]
            ]
        )

        dash_extra = pd.DataFrame(
            [
                dash_lookup[k]
                for k in
                df["_pair_key"]
            ]
        )

        base_df = (
            df.drop(
                columns=[
                    "_pair_key"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        mash_result = pd.concat(
            [
                base_df,
                mash_extra.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        dash_result = pd.concat(
            [
                base_df,
                dash_extra.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        expected = EXPECTED[
            (protocol, seed)
        ]

        if (
            len(mash_result)
            != expected
        ):
            raise RuntimeError(
                "Mash validation "
                "row-count mismatch"
            )

        if (
            len(dash_result)
            != expected
        ):
            raise RuntimeError(
                "Dashing2 validation "
                "row-count mismatch"
            )


        mash_dir = (
            MASH_OUT /
            "results" /
            f"{protocol}_seed{seed}"
        )

        dash_dir = (
            DASH_OUT /
            "results" /
            f"{protocol}_seed{seed}"
        )

        mash_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dash_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        mash_file = (
            mash_dir /
            "mash_validation_results.tsv"
        )

        dash_file = (
            dash_dir /
            "dashing2_validation_results.tsv"
        )


        mash_result.to_csv(
            mash_file,
            sep="\t",
            index=False,
        )

        dash_result.to_csv(
            dash_file,
            sep="\t",
            index=False,
        )


        for method in [
            "Mash",
            "Dashing2_SetSketch",
        ]:

            coverage_rows.append({
                "method":
                    method,

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "n_expected":
                    expected,

                "n_reported":
                    expected,

                "n_missing":
                    0,

                "coverage":
                    1.0,
            })


        for label, g in (
            mash_result.groupby(
                "label"
            )
        ):

            score_rows.append({
                "method":
                    "Mash",

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "label":
                    label,

                "n":
                    len(g),

                "score_column":
                    "mash_similarity",

                "score_min":
                    g[
                        "mash_similarity"
                    ].min(),

                "score_median":
                    g[
                        "mash_similarity"
                    ].median(),

                "score_mean":
                    g[
                        "mash_similarity"
                    ].mean(),

                "score_max":
                    g[
                        "mash_similarity"
                    ].max(),
            })


        for label, g in (
            dash_result.groupby(
                "label"
            )
        ):

            score_rows.append({
                "method":
                    "Dashing2_SetSketch",

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "label":
                    label,

                "n":
                    len(g),

                "score_column":
                    "dashing2_similarity",

                "score_min":
                    g[
                        "dashing2_similarity"
                    ].min(),

                "score_median":
                    g[
                        "dashing2_similarity"
                    ].median(),

                "score_mean":
                    g[
                        "dashing2_similarity"
                    ].mean(),

                "score_max":
                    g[
                        "dashing2_similarity"
                    ].max(),
            })


        sha_rows.extend([
            {
                "method":
                    "Mash",

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "file":
                    str(mash_file),

                "sha256":
                    sha256_file(
                        mash_file
                    ),
            },

            {
                "method":
                    "Dashing2_SetSketch",

                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "file":
                    str(dash_file),

                "sha256":
                    sha256_file(
                        dash_file
                    ),
            },
        ])


        print(
            f"{protocol} seed{seed}: "
            f"Mash={expected}/{expected}; "
            f"Dashing2="
            f"{expected}/{expected}"
        )


coverage_df = pd.DataFrame(
    coverage_rows
)

score_df = pd.DataFrame(
    score_rows
)

sha_df = pd.DataFrame(
    sha_rows
)


coverage_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_validation_coverage.tsv",
    sep="\t",
    index=False,
)

score_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_validation_score_audit.tsv",
    sep="\t",
    index=False,
)

sha_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_validation_sha256.tsv",
    sep="\t",
    index=False,
)


audit = {
    "validation_pair_occurrences":
        total_occurrences,

    "unique_validation_pairs":
        len(target_keys),

    "mash_source":
        str(MASH),

    "mash_source_sha256":
        mash_sha.hexdigest(),

    "mash_edges":
        mash_lines,

    "mash_targets_found":
        len(mash_lookup),

    "dashing2_source":
        str(DASH),

    "dashing2_source_sha256":
        dash_sha.hexdigest(),

    "dashing2_sources":
        EXPECTED_A_GENOMES,

    "dashing2_rows":
        row_count,

    "dashing2_targets_found":
        len(dash_lookup),

    "mash_validation_coverage":
        1.0,

    "dashing2_validation_coverage":
        1.0,

    "status":
        "PASS",
}


with open(
    AUDIT_OUT /
    "VALIDATION_CACHE_JOIN_AUDIT.json",
    "w",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
    )


report = (
    AUDIT_OUT /
    "MASH_DASHING_VALIDATION_CACHE_JOIN_REPORT.txt"
)

with open(report, "w") as f:

    f.write(
        "E5 Mash + Dashing2 "
        "Validation Cache Join v1\n"
    )

    f.write(
        "=" * 72 +
        "\n\n"
    )

    f.write(
        f"Validation pair occurrences: "
        f"{total_occurrences}\n"
    )

    f.write(
        f"Unique unordered pairs: "
        f"{len(target_keys)}\n\n"
    )

    f.write(
        f"Mash coverage: 100%\n"
    )

    f.write(
        f"Dashing2 coverage: 100%\n\n"
    )

    f.write(
        "Mash unified score = "
        "1 - native Mash distance\n"
    )

    f.write(
        "Dashing2 unified score = "
        "native SetSketch similarity\n\n"
    )

    f.write(
        "STATUS: PASS\n"
    )


print()
print("=" * 90)
print(
    "FINAL VALIDATION CACHE-JOIN AUDIT"
)
print("=" * 90)

print(
    "validation occurrences :",
    total_occurrences
)

print(
    "unique validation pairs:",
    len(target_keys)
)

print(
    "Mash target coverage   : "
    f"{len(mash_lookup)}/"
    f"{len(target_keys)}"
)

print(
    "Dashing target coverage: "
    f"{len(dash_lookup)}/"
    f"{len(target_keys)}"
)

print()
print(
    "Validation score audit:"
)

print(
    score_df.to_string(
        index=False
    )
)

print()
print("Report:")
print(report)

print()
print("STATUS = PASS")
