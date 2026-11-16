#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import os

import pandas as pd


BASE = Path("HGV_V3_revision_20260915")

MANIFEST = (
    BASE /
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

PAIRROOT = (
    BASE /
    "05_pairs/pair_manifests_v1_20260916"
)

MASH = (
    BASE /
    "06_teacher/teacher_cache_v1_20260916/"
    "mash/mash_triangle_edges.tsv"
)

DASH = (
    BASE /
    "06_teacher/teacher_cache_v1_20260916/"
    "dashing2_set/dashing2_set_similarity.tsv"
)

E5 = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917"
)

MASH_OUT = (
    E5 /
    "02_mash/internal_cache_join_v1"
)

DASH_OUT = (
    E5 /
    "03_dashing2/internal_cache_join_v1"
)

AUDIT_OUT = (
    E5 /
    "07_unified_internal/cache_join_audit_v1"
)

for p in [
    MASH_OUT,
    DASH_OUT,
    AUDIT_OUT,
]:
    p.mkdir(parents=True, exist_ok=True)


EXPECTED = {
    ("G2", "20260915"): 3872,
    ("G2", "20260916"): 3874,
    ("G2", "20260917"): 3874,

    ("G3", "20260915"): 3916,
    ("G3", "20260916"): 3852,
    ("G3", "20260917"): 3894,

    ("G4", "20260915"): 3832,
    ("G4", "20260916"): 3894,
    ("G4", "20260917"): 3874,
}

EXPECTED_TOTAL_OCCURRENCES = 34882

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
            f"Unexpected self-pair: {a}"
        )

    if a < b:
        return (a, b)

    return (b, a)


def file_sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


print("=" * 90)
print("E5 MASH + DASHING2 INTERNAL CACHE JOIN v1")
print("=" * 90)

# ------------------------------------------------------------------
# 1. Frozen manifest -> exact FASTA path/accession mapping
# ------------------------------------------------------------------

print()
print("===== LOAD FROZEN MANIFEST =====")

manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
    dtype=str,
)

required_manifest = [
    "assembly_accession_final",
    "fasta_path",
]

for c in required_manifest:
    if c not in manifest.columns:
        raise RuntimeError(
            f"Manifest missing column: {c}"
        )

if manifest[
    "assembly_accession_final"
].duplicated().any():
    raise RuntimeError(
        "Duplicate assembly_accession_final "
        "in frozen manifest"
    )

if manifest[
    "fasta_path"
].duplicated().any():
    raise RuntimeError(
        "Duplicate fasta_path in frozen manifest"
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

print(
    "manifest genomes        =",
    len(manifest)
)

print(
    "path mappings           =",
    len(path_to_acc)
)


# ------------------------------------------------------------------
# 2. Load all nine frozen test-pair manifests.
# ------------------------------------------------------------------

print()
print("===== LOAD 9 FROZEN TEST MANIFESTS =====")

split_frames = {}
target_keys = set()

total_occurrences = 0

for protocol in ["G2", "G3", "G4"]:

    for seed in [
        "20260915",
        "20260916",
        "20260917",
    ]:

        pair_file = (
            PAIRROOT /
            f"{protocol}_seed{seed}" /
            "test_pairs.tsv"
        )

        if not pair_file.is_file():
            raise FileNotFoundError(
                pair_file
            )

        df = pd.read_csv(
            pair_file,
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
                f"found {len(df)}"
            )

        for c in [
            "pair_id",
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
        ]:
            if c not in df.columns:
                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    f"missing {c}"
                )

        # Ensure every accession exists in frozen master manifest.
        unknown_a = (
            set(df["accession_a"]) -
            set(acc_to_path)
        )

        unknown_b = (
            set(df["accession_b"]) -
            set(acc_to_path)
        )

        if unknown_a or unknown_b:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "unknown accession(s)"
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

        target_keys.update(keys)

        total_occurrences += len(df)

        print(
            f"{protocol} seed{seed}: "
            f"{len(df)} pairs"
        )


if (
    total_occurrences
    != EXPECTED_TOTAL_OCCURRENCES
):
    raise RuntimeError(
        "Total frozen pair occurrence "
        "count mismatch"
    )

print()
print(
    "total pair occurrences  =",
    total_occurrences
)

print(
    "unique unordered pairs  =",
    len(target_keys)
)


# ------------------------------------------------------------------
# 3. Stream Mash complete triangle ONCE.
#
# Standard mash dist fields:
# reference/query path
# distance
# p-value
# shared hashes
# ------------------------------------------------------------------

print()
print("===== SCAN MASH ALL-PAIR CACHE =====")

mash_lookup = {}

mash_line_count = 0
mash_unknown_paths = 0
mash_target_duplicates = 0

mash_hash = hashlib.sha256()

with open(MASH, "rb") as f:

    for raw in f:

        mash_hash.update(raw)

        mash_line_count += 1

        line = raw.decode(
            "utf-8"
        ).rstrip("\n")

        parts = line.split("\t")

        if len(parts) != 5:
            raise RuntimeError(
                f"Mash line "
                f"{mash_line_count} "
                f"has {len(parts)} fields"
            )

        path1 = parts[0]
        path2 = parts[1]

        acc1 = path_to_acc.get(path1)
        acc2 = path_to_acc.get(path2)

        if acc1 is None or acc2 is None:
            mash_unknown_paths += 1
            continue

        key = canon_pair(
            acc1,
            acc2,
        )

        if key not in target_keys:
            continue

        if key in mash_lookup:
            mash_target_duplicates += 1
            raise RuntimeError(
                f"Duplicate Mash target pair: "
                f"{key}"
            )

        distance = float(parts[2])
        pvalue = float(parts[3])
        shared = parts[4]

        try:
            shared_num_s, shared_den_s = (
                shared.split("/", 1)
            )

            shared_num = int(
                shared_num_s
            )

            shared_den = int(
                shared_den_s
            )

        except Exception:
            raise RuntimeError(
                f"Malformed Mash shared "
                f"hash field: {shared}"
            )

        if not (
            0.0 <= distance <= 1.0
        ):
            raise RuntimeError(
                f"Mash distance outside "
                f"[0,1]: {distance}"
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
                shared_num,

            "mash_shared_denominator":
                shared_den,
        }


print(
    "Mash lines             =",
    mash_line_count
)

print(
    "Expected triangle      =",
    EXPECTED_MASH_EDGES
)

print(
    "Matched target pairs   =",
    len(mash_lookup)
)

print(
    "Unknown source paths   =",
    mash_unknown_paths
)


if mash_line_count != EXPECTED_MASH_EDGES:
    raise RuntimeError(
        "Mash cache is not the expected "
        "complete 2374-genome triangle"
    )

if mash_unknown_paths != 0:
    raise RuntimeError(
        "Mash contains paths absent "
        "from frozen manifest"
    )

mash_missing_keys = (
    target_keys -
    set(mash_lookup)
)

if mash_missing_keys:
    raise RuntimeError(
        f"Mash missing "
        f"{len(mash_missing_keys)} "
        "required unique pair(s). "
        f"Examples: "
        f"{list(mash_missing_keys)[:10]}"
    )

print("MASH TARGET COVERAGE = 100%")


# ------------------------------------------------------------------
# 4. Parse Dashing2 symmetric upper-triangle matrix ONCE.
#
# Format:
# line 1 comment
# line 2 options
# line 3 "#Sources" + 2374 paths
# line 4..2377:
# row path + 2374 cells
#
# diagonal/lower triangle = "-"
# upper triangle = native similarity
# ------------------------------------------------------------------

print()
print(
    "===== PARSE DASHING2 "
    "SIMILARITY MATRIX ====="
)

dash_lookup = {}

dash_hash = hashlib.sha256()

with open(DASH, "rb") as f:

    raw1 = f.readline()
    raw2 = f.readline()
    raw3 = f.readline()

    dash_hash.update(raw1)
    dash_hash.update(raw2)
    dash_hash.update(raw3)

    line1 = raw1.decode(
        "utf-8"
    ).rstrip("\n")

    line2 = raw2.decode(
        "utf-8"
    ).rstrip("\n")

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
            "Dashing2 source header "
            "not found"
        )

    sources = source_fields[1:]

    n_sources = len(sources)

    print(
        "Dashing2 sources       =",
        n_sources
    )

    if (
        n_sources
        != EXPECTED_A_GENOMES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_A_GENOMES} "
            f"Dashing2 sources, "
            f"found {n_sources}"
        )

    source_accs = []

    for p in sources:

        acc = path_to_acc.get(p)

        if acc is None:
            raise RuntimeError(
                "Dashing2 source path "
                "absent from frozen "
                f"manifest: {p}"
            )

        source_accs.append(acc)

    if (
        len(set(source_accs))
        != n_sources
    ):
        raise RuntimeError(
            "Duplicate accession in "
            "Dashing2 #Sources"
        )

    source_index = {
        acc: i
        for i, acc
        in enumerate(source_accs)
    }

    # Every frozen target accession must
    # be represented in Dashing2 cache.

    target_accessions = set()

    for a, b in target_keys:
        target_accessions.add(a)
        target_accessions.add(b)

    absent = (
        target_accessions -
        set(source_index)
    )

    if absent:
        raise RuntimeError(
            f"{len(absent)} target "
            "accessions missing from "
            "Dashing2 sources. "
            f"Examples: {list(absent)[:10]}"
        )

    # For each unordered pair, select
    # the upper-triangle cell.

    needs_by_row = {}

    for key in target_keys:

        a, b = key

        ia = source_index[a]
        ib = source_index[b]

        if ia == ib:
            raise RuntimeError(
                f"Unexpected same matrix "
                f"index for {key}"
            )

        if ia < ib:
            row_i = ia
            col_j = ib
        else:
            row_i = ib
            col_j = ia

        needs_by_row.setdefault(
            row_i,
            {}
        )[col_j] = key

    row_count = 0

    for row_i in range(
        n_sources
    ):

        raw = f.readline()

        if not raw:
            raise RuntimeError(
                "Dashing2 matrix ended "
                "before all source rows"
            )

        dash_hash.update(raw)

        row_count += 1

        parts = (
            raw.decode("utf-8")
            .rstrip("\n")
            .split("\t")
        )

        if len(parts) != (
            n_sources + 1
        ):
            raise RuntimeError(
                f"Dashing2 row {row_i}: "
                f"expected "
                f"{n_sources + 1} fields, "
                f"found {len(parts)}"
            )

        row_path = parts[0]

        if row_path != sources[row_i]:
            raise RuntimeError(
                f"Dashing2 row/source "
                f"order mismatch at "
                f"index {row_i}"
            )

        needed = needs_by_row.get(
            row_i
        )

        if not needed:
            continue

        for col_j, key in (
            needed.items()
        ):

            token = parts[
                col_j + 1
            ]

            if token == "-":
                raise RuntimeError(
                    "Required Dashing2 "
                    "upper-triangle value "
                    f"is '-': {key}"
                )

            similarity = float(token)

            if not (
                0.0
                <= similarity
                <= 1.0
            ):
                raise RuntimeError(
                    "Dashing2 similarity "
                    "outside [0,1]: "
                    f"{similarity}"
                )

            if key in dash_lookup:
                raise RuntimeError(
                    "Duplicate Dashing2 "
                    f"target pair: {key}"
                )

            dash_lookup[key] = {
                "dashing2_similarity":
                    similarity,

                "dashing2_distance":
                    1.0 - similarity,
            }

    # Hash any remaining bytes, while also
    # rejecting unexpected extra nonempty rows.

    extra_nonempty = 0

    for raw in f:

        dash_hash.update(raw)

        if raw.strip():
            extra_nonempty += 1

    if extra_nonempty:
        raise RuntimeError(
            "Unexpected extra Dashing2 "
            f"matrix rows: {extra_nonempty}"
        )


print(
    "Dashing2 matrix rows   =",
    row_count
)

print(
    "Matched target pairs   =",
    len(dash_lookup)
)

dash_missing_keys = (
    target_keys -
    set(dash_lookup)
)

if dash_missing_keys:
    raise RuntimeError(
        f"Dashing2 missing "
        f"{len(dash_missing_keys)} "
        "required pair(s). "
        f"Examples: "
        f"{list(dash_missing_keys)[:10]}"
    )

print(
    "DASHING2 TARGET COVERAGE = 100%"
)


# ------------------------------------------------------------------
# 5. Join caches back onto every frozen test-pair occurrence.
# ------------------------------------------------------------------

print()
print(
    "===== WRITE 9 SPLIT-LEVEL "
    "RESULT SETS ====="
)

coverage_rows = []
score_audit_rows = []
sha_rows = []

for protocol in ["G2", "G3", "G4"]:

    for seed in [
        "20260915",
        "20260916",
        "20260917",
    ]:

        df = split_frames[
            (protocol, seed)
        ].copy()

        mash_rows = []
        dash_rows = []

        for key in df[
            "_pair_key"
        ]:

            mash_rows.append(
                mash_lookup[key]
            )

            dash_rows.append(
                dash_lookup[key]
            )

        mash_extra = pd.DataFrame(
            mash_rows
        )

        dash_extra = pd.DataFrame(
            dash_rows
        )

        mash_result = pd.concat(
            [
                df.drop(
                    columns=[
                        "_pair_key"
                    ]
                ).reset_index(
                    drop=True
                ),
                mash_extra.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        dash_result = pd.concat(
            [
                df.drop(
                    columns=[
                        "_pair_key"
                    ]
                ).reset_index(
                    drop=True
                ),
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
                f"{protocol} {seed}: "
                "Mash output row "
                "count mismatch"
            )

        if (
            len(dash_result)
            != expected
        ):
            raise RuntimeError(
                f"{protocol} {seed}: "
                "Dashing output row "
                "count mismatch"
            )

        if mash_result[
            "mash_distance"
        ].isna().any():
            raise RuntimeError(
                f"{protocol} {seed}: "
                "Mash NA score"
            )

        if dash_result[
            "dashing2_similarity"
        ].isna().any():
            raise RuntimeError(
                f"{protocol} {seed}: "
                "Dashing2 NA score"
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
            "mash_fixed_pair_results.tsv"
        )

        dash_file = (
            dash_dir /
            "dashing2_fixed_pair_results.tsv"
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

        # Complete coverage by construction.
        coverage_rows.extend([
            {
                "method":
                    "Mash",

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
            },
            {
                "method":
                    "Dashing2_SetSketch",

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
            },
        ])

        # Simple score-distribution audit only.
        # Formal AUROC/AUPRC is deferred to
        # unified evaluator.

        for label, g in (
            mash_result.groupby(
                "label"
            )
        ):
            score_audit_rows.append({
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
            score_audit_rows.append({
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
                    file_sha256(
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
                    file_sha256(
                        dash_file
                    ),
            },
        ])

        print(
            f"{protocol} seed{seed}: "
            f"Mash={expected}/{expected}; "
            f"Dashing2={expected}/{expected}"
        )


coverage_df = pd.DataFrame(
    coverage_rows
)

score_audit_df = pd.DataFrame(
    score_audit_rows
)

sha_df = pd.DataFrame(
    sha_rows
)


coverage_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_internal_coverage.tsv",
    sep="\t",
    index=False,
)

score_audit_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_score_distribution_audit.tsv",
    sep="\t",
    index=False,
)

sha_df.to_csv(
    AUDIT_OUT /
    "mash_dashing_result_sha256.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------------
# 6. Provenance/final audit.
# ------------------------------------------------------------------

source_audit = {
    "manifest":
        str(MANIFEST),

    "mash_source":
        str(MASH),

    "mash_source_sha256":
        mash_hash.hexdigest(),

    "mash_line_count":
        mash_line_count,

    "mash_expected_complete_edges":
        EXPECTED_MASH_EDGES,

    "dashing2_source":
        str(DASH),

    "dashing2_source_sha256":
        dash_hash.hexdigest(),

    "dashing2_source_count":
        n_sources,

    "dashing2_matrix_rows":
        row_count,

    "frozen_pair_occurrences":
        total_occurrences,

    "unique_target_pairs":
        len(target_keys),

    "mash_target_pairs_found":
        len(mash_lookup),

    "dashing2_target_pairs_found":
        len(dash_lookup),

    "mash_missing_target_pairs":
        0,

    "dashing2_missing_target_pairs":
        0,

    "mash_score_semantics":
        "native distance; unified similarity = 1 - distance",

    "dashing2_score_semantics":
        "native SetSketch similarity; unified similarity = native similarity",

    "status":
        "PASS",
}

with open(
    AUDIT_OUT /
    "CACHE_JOIN_AUDIT.json",
    "w",
) as f:
    json.dump(
        source_audit,
        f,
        indent=2,
    )


report_file = (
    AUDIT_OUT /
    "MASH_DASHING_INTERNAL_CACHE_JOIN_REPORT.txt"
)

with open(
    report_file,
    "w",
) as f:

    f.write(
        "E5 Mash + Dashing2 Internal "
        "Cache Join v1\n"
    )

    f.write(
        "=" * 72 +
        "\n\n"
    )

    f.write(
        f"Frozen pair occurrences: "
        f"{total_occurrences}\n"
    )

    f.write(
        f"Unique unordered target pairs: "
        f"{len(target_keys)}\n\n"
    )

    f.write(
        f"Mash source edges: "
        f"{mash_line_count}\n"
    )

    f.write(
        f"Expected Mash complete triangle: "
        f"{EXPECTED_MASH_EDGES}\n"
    )

    f.write(
        f"Mash required-pair coverage: "
        f"100%\n\n"
    )

    f.write(
        f"Dashing2 sources: "
        f"{n_sources}\n"
    )

    f.write(
        f"Dashing2 matrix rows: "
        f"{row_count}\n"
    )

    f.write(
        f"Dashing2 required-pair coverage: "
        f"100%\n\n"
    )

    f.write(
        "Mash semantics:\n"
    )

    f.write(
        "  native = distance\n"
    )

    f.write(
        "  unified similarity = "
        "1 - distance\n\n"
    )

    f.write(
        "Dashing2 semantics:\n"
    )

    f.write(
        "  native = SetSketch similarity\n"
    )

    f.write(
        "  unified similarity = "
        "native similarity\n\n"
    )

    f.write(
        "Formal classification metrics "
        "are intentionally deferred to "
        "the E5 unified evaluator.\n\n"
    )

    f.write(
        "STATUS: PASS\n"
    )


print()
print("=" * 90)
print("FINAL CACHE-JOIN AUDIT")
print("=" * 90)

print(
    f"Frozen pair occurrences : "
    f"{total_occurrences}"
)

print(
    f"Unique target pairs      : "
    f"{len(target_keys)}"
)

print(
    f"Mash edges scanned       : "
    f"{mash_line_count}"
)

print(
    f"Mash target coverage     : "
    f"{len(mash_lookup)}/"
    f"{len(target_keys)}"
)

print(
    f"Dashing2 sources         : "
    f"{n_sources}"
)

print(
    f"Dashing2 target coverage : "
    f"{len(dash_lookup)}/"
    f"{len(target_keys)}"
)

print()
print("Score distribution audit:")
print(
    score_audit_df.to_string(
        index=False
    )
)

print()
print("Report:")
print(report_file)

print()
print("STATUS = PASS")
