#!/usr/bin/env python3

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


BASE = Path(
    "HGV_V3_revision_20260915"
)

ROOT = (
    BASE /
    "09_baselines/E5_external_unified_v1_20260920"
)

E5 = (
    ROOT /
    "02_native_external_full_v1"
)

RAW = (
    E5 /
    "raw/dashing2/dashing2_external_rectangular.tsv"
)

OLD_ALL11 = (
    E5 /
    "source_data/E5_EXTERNAL_ALL11_QUERY_LEVEL.tsv"
)

OUT = (
    ROOT /
    "04_dashing2_external_repair_v2_20260922"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

METHOD = "Dashing2_SetSketch_k31_S1024"

ACC_RE = re.compile(
    r"(GC[AF]_\d+\.\d+)"
)


# ============================================================
# Helpers
# ============================================================

def fail(msg):
    raise RuntimeError(msg)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def read_list(path):
    return [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def acc_from_path(x):
    m = ACC_RE.search(str(x))

    if m is None:
        fail(
            f"Cannot extract accession from path: {x}"
        )

    return m.group(1)


def locate_one(name):
    hits = list(
        ROOT.rglob(name)
    )

    if len(hits) != 1:
        fail(
            f"Expected exactly one {name}; "
            f"found {len(hits)}: {hits}"
        )

    return hits[0]


def find_acc_col(df):
    preferred = [
        "query_accession",
        "assembly_accession",
        "assembly_accession_final",
        "accession",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    for c in df.columns:
        vals = (
            df[c]
            .astype(str)
            .head(100)
            .tolist()
        )

        n = sum(
            1
            for x in vals
            if ACC_RE.search(x)
        )

        if n >= min(20, len(vals)):
            return c

    fail(
        f"Cannot identify accession column: "
        f"{list(df.columns)}"
    )


def find_fasta_col(df):
    preferred = [
        "fasta_path",
        "genome_fasta",
        "assembly_fasta",
    ]

    for c in preferred:
        if c in df.columns:
            return c

    for c in df.columns:
        vals = (
            df[c]
            .astype(str)
            .head(50)
            .tolist()
        )

        n = sum(
            x.endswith(
                (
                    ".fna",
                    ".fna.gz",
                    ".fa",
                    ".fa.gz",
                    ".fasta",
                    ".fasta.gz",
                )
            )
            for x in vals
        )

        if n >= min(10, len(vals)):
            return c

    fail(
        f"Cannot identify FASTA column: "
        f"{list(df.columns)}"
    )


def clean_value(x):
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.lower() in {
        "",
        "nan",
        "none",
        "na",
        "n/a",
    }:
        return ""

    return s


def candidate_tax_cols(df, rank):
    rank = rank.lower()

    cols = []

    for c in df.columns:
        lc = c.lower()

        if rank == "species":
            ok = (
                "species" in lc
                or "sp_cluster" in lc
                or "species_cluster" in lc
            )

        elif rank == "genus":
            ok = "genus" in lc

        elif rank == "family":
            ok = "family" in lc

        else:
            fail(
                f"Unsupported rank: {rank}"
            )

        if ok:
            cols.append(c)

    return cols


def detect_tax_pair(
    ref,
    query,
    rank,
):
    """
    Pick query/reference taxonomy columns by maximum
    value overlap.

    This avoids assuming exact historical column names.
    """

    rcands = candidate_tax_cols(
        ref,
        rank,
    )

    qcands = candidate_tax_cols(
        query,
        rank,
    )

    if not rcands or not qcands:
        fail(
            f"{rank}: taxonomy candidate columns missing.\n"
            f"ref={rcands}\n"
            f"query={qcands}"
        )

    best = None

    for qc in qcands:
        qvals = {
            clean_value(x)
            for x in query[qc]
        }
        qvals.discard("")

        for rc in rcands:
            rvals = {
                clean_value(x)
                for x in ref[rc]
            }
            rvals.discard("")

            overlap = len(
                qvals & rvals
            )

            item = (
                overlap,
                qc,
                rc,
            )

            if (
                best is None
                or item[0] > best[0]
            ):
                best = item

    if best is None or best[0] == 0:
        fail(
            f"{rank}: could not identify matching "
            f"taxonomy columns."
        )

    return (
        best[1],
        best[2],
        best[0],
    )


def find_col(
    df,
    preferred,
    contains=None,
):
    for c in preferred:
        if c in df.columns:
            return c

    if contains:
        hits = []

        for c in df.columns:
            lc = c.lower()

            if all(
                token.lower() in lc
                for token in contains
            ):
                hits.append(c)

        if len(hits) == 1:
            return hits[0]

    fail(
        f"Cannot identify column. "
        f"preferred={preferred}; "
        f"columns={list(df.columns)}"
    )


# ============================================================
# 1. Locate frozen external task artifacts
# ============================================================

REF_TSV = locate_one(
    "REFERENCE_A_2374.tsv"
)

REF_LIST = locate_one(
    "REFERENCE_A_2374_FASTA.list"
)

QUERY_FILES = [
    (
        "B1",
        "species",
        locate_one(
            "QUERY_B1_391.tsv"
        ),
        locate_one(
            "QUERY_B1_391_FASTA.list"
        ),
    ),
    (
        "B2_seen_genus",
        "genus",
        locate_one(
            "QUERY_B2_SEEN_196.tsv"
        ),
        locate_one(
            "QUERY_B2_SEEN_196_FASTA.list"
        ),
    ),
    (
        "B2_unseen_genus",
        "family",
        locate_one(
            "QUERY_B2_UNSEEN_193.tsv"
        ),
        locate_one(
            "QUERY_B2_UNSEEN_193_FASTA.list"
        ),
    ),
]


if not RAW.is_file():
    raise FileNotFoundError(
        RAW
    )

if not OLD_ALL11.is_file():
    raise FileNotFoundError(
        OLD_ALL11
    )


# ============================================================
# 2. Frozen reference/query inventory
# ============================================================

ref = pd.read_csv(
    REF_TSV,
    sep="\t",
    dtype=str,
)

ref_acc_col = find_acc_col(
    ref
)

ref_fasta_col = find_fasta_col(
    ref
)

ref_acc = (
    ref[ref_acc_col]
    .astype(str)
    .tolist()
)

ref_fasta = read_list(
    REF_LIST
)

if len(ref) != 2374:
    fail(
        f"Reference TSV !=2374: {len(ref)}"
    )

if len(ref_fasta) != 2374:
    fail(
        f"Reference FASTA list !=2374: "
        f"{len(ref_fasta)}"
    )

if len(set(ref_acc)) != 2374:
    fail(
        "Reference accessions not unique"
    )

query_frames = []
query_fasta = []

for endpoint, rank, tsv, flist in QUERY_FILES:

    q = pd.read_csv(
        tsv,
        sep="\t",
        dtype=str,
    )

    q_acc_col = find_acc_col(
        q
    )

    q_fasta_col = find_fasta_col(
        q
    )

    q = q.copy()

    q["__query_accession"] = (
        q[q_acc_col]
        .astype(str)
    )

    q["__endpoint"] = endpoint
    q["__target_rank"] = rank

    qlist = read_list(
        flist
    )

    if len(q) != len(qlist):
        fail(
            f"{endpoint}: TSV/list count mismatch "
            f"{len(q)} vs {len(qlist)}"
        )

    # Ensure list order can be identified by accession.
    list_acc = [
        acc_from_path(x)
        for x in qlist
    ]

    q_acc_set = set(
        q["__query_accession"]
    )

    if set(list_acc) != q_acc_set:
        fail(
            f"{endpoint}: FASTA/accession set mismatch"
        )

    query_frames.append(
        q
    )

    query_fasta.extend(
        qlist
    )


query = pd.concat(
    query_frames,
    ignore_index=True,
)

if len(query) != 780:
    fail(
        f"Query count !=780: {len(query)}"
    )

if (
    query["__query_accession"]
    .nunique()
    != 780
):
    fail(
        "Query accessions not unique"
    )

query_acc = [
    acc_from_path(x)
    for x in query_fasta
]

if set(query_acc) != set(
    query["__query_accession"]
):
    fail(
        "Combined query FASTA order/set invalid"
    )

if set(query_acc) & set(ref_acc):
    fail(
        "Query/reference accession overlap"
    )


# ============================================================
# 3. Parse exact Dashing2 rectangular format
#
# Observed production format:
#
# line 1 comment
# line 2 options
# line 3 #Sources + 3154 paths
#
# next 2374 rows:
#   reference_path + 780 similarity values
#
# Therefore the physical matrix is:
#   reference x query
#
# Retrieval requires:
#   query x reference
#
# so transpose is mandatory.
# ============================================================

with RAW.open(
    "r",
    encoding="utf-8",
) as f:

    line1 = f.readline().rstrip(
        "\n"
    )

    line2 = f.readline().rstrip(
        "\n"
    )

    source_fields = (
        f.readline()
        .rstrip("\n")
        .split("\t")
    )

    if (
        not source_fields
        or source_fields[0]
        != "#Sources"
    ):
        fail(
            "Dashing2 #Sources header missing"
        )

    sources = source_fields[1:]

    if len(sources) != 3154:
        fail(
            f"Expected 3154 Dashing2 sources; "
            f"found {len(sources)}"
        )

    source_ref = sources[:2374]
    source_query = sources[2374:]

    if source_ref != ref_fasta:
        fail(
            "Dashing2 first 2374 #Sources are not "
            "identical to frozen reference order"
        )

    if source_query != query_fasta:
        fail(
            "Dashing2 final 780 #Sources are not "
            "identical to frozen query order"
        )

    M_ref_query = np.empty(
        (
            2374,
            780,
        ),
        dtype=np.float32,
    )

    row_acc = []

    for i in range(2374):

        line = f.readline()

        if not line:
            fail(
                f"Dashing2 ended early at row {i}"
            )

        parts = (
            line.rstrip("\n")
            .split("\t")
        )

        if len(parts) != 781:
            fail(
                f"Dashing2 row {i}: "
                f"expected 781 fields, "
                f"found {len(parts)}"
            )

        row_path = parts[0]

        if row_path != ref_fasta[i]:
            fail(
                f"Dashing2 row/reference order "
                f"mismatch at {i}"
            )

        row_acc.append(
            acc_from_path(
                row_path
            )
        )

        try:
            vals = np.asarray(
                parts[1:],
                dtype=np.float32,
            )
        except Exception as e:
            fail(
                f"Dashing2 numeric parse failed "
                f"at row {i}: {e}"
            )

        if not np.isfinite(
            vals
        ).all():
            fail(
                f"Non-finite Dashing2 value row {i}"
            )

        if (
            vals.min() < -1e-7
            or vals.max() > 1.000001
        ):
            fail(
                f"Dashing2 values outside [0,1] "
                f"at row {i}"
            )

        M_ref_query[
            i,
            :
        ] = vals

    extra = [
        x
        for x in f
        if x.strip()
    ]

    if extra:
        fail(
            f"Unexpected extra Dashing2 rows: "
            f"{len(extra)}"
        )


if row_acc != ref_acc:
    # REF_TSV ordering may differ from FASTA list even though
    # the set is identical. Rebuild metadata in matrix order.
    ref_by_acc = (
        ref.set_index(
            ref_acc_col,
            drop=False,
        )
    )

    missing = [
        a
        for a in row_acc
        if a not in ref_by_acc.index
    ]

    if missing:
        fail(
            f"Reference metadata missing: "
            f"{missing[:10]}"
        )

    ref = (
        ref_by_acc
        .loc[row_acc]
        .reset_index(drop=True)
    )

    ref_acc = row_acc


# Critical correction:
M_query_ref = (
    M_ref_query.T.copy()
)

if M_query_ref.shape != (
    780,
    2374,
):
    fail(
        f"Corrected matrix shape invalid: "
        f"{M_query_ref.shape}"
    )


# ============================================================
# 4. Build endpoint/query metadata in exact query matrix order
# ============================================================

q_by_acc = {}

for _, r in query.iterrows():

    a = str(
        r["__query_accession"]
    )

    if a in q_by_acc:
        fail(
            f"Duplicate query metadata: {a}"
        )

    q_by_acc[a] = r


ordered_query_rows = []

for a in query_acc:

    if a not in q_by_acc:
        fail(
            f"Query metadata missing: {a}"
        )

    ordered_query_rows.append(
        q_by_acc[a]
    )


# ============================================================
# 5. Detect correct taxonomic columns endpoint-by-endpoint
# ============================================================

tax_info = {}

for endpoint, rank, tsv, flist in QUERY_FILES:

    q = pd.read_csv(
        tsv,
        sep="\t",
        dtype=str,
    )

    q_acc_col = find_acc_col(
        q
    )

    qc, rc, overlap = (
        detect_tax_pair(
            ref,
            q,
            rank,
        )
    )

    tax_info[endpoint] = {
        "rank": rank,
        "query_column": qc,
        "reference_column": rc,
        "overlap_values": overlap,
        "query_accession_column":
            q_acc_col,
        "query_frame": q,
    }

    print(
        f"TAXONOMY {endpoint}: "
        f"rank={rank} "
        f"query_col={qc} "
        f"ref_col={rc} "
        f"value_overlap={overlap}"
    )


# ============================================================
# 6. Correct retrieval evaluation
# ============================================================

repaired_rows = []

expected_eligible = {
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 143,
}

for qi, qacc in enumerate(
    query_acc
):

    qr = q_by_acc[
        qacc
    ]

    endpoint = str(
        qr["__endpoint"]
    )

    rank = str(
        qr["__target_rank"]
    )

    info = tax_info[
        endpoint
    ]

    qdf = info[
        "query_frame"
    ]

    qacol = info[
        "query_accession_column"
    ]

    qhit = qdf[
        qdf[qacol].astype(str)
        == qacc
    ]

    if len(qhit) != 1:
        fail(
            f"{endpoint}/{qacc}: "
            f"metadata rows={len(qhit)}"
        )

    qtax = clean_value(
        qhit.iloc[0][
            info["query_column"]
        ]
    )

    if not qtax:
        fail(
            f"{endpoint}/{qacc}: "
            f"empty {rank}"
        )

    ref_tax = np.asarray(
        [
            clean_value(x)
            for x in ref[
                info["reference_column"]
            ]
        ],
        dtype=object,
    )

    relevant = (
        ref_tax == qtax
    )

    eligible = bool(
        relevant.any()
    )

    scores = (
        M_query_ref[
            qi,
            :
        ]
        .astype(
            np.float64
        )
    )

    if not np.isfinite(
        scores
    ).all():
        fail(
            f"{qacc}: non-finite score"
        )

    # Stable descending ranking.
    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    top_index = int(
        order[0]
    )

    top1_reference = (
        ref_acc[
            top_index
        ]
    )

    top1_score = float(
        scores[
            top_index
        ]
    )

    if eligible:

        rel_ordered = relevant[
            order
        ]

        hit = np.flatnonzero(
            rel_ordered
        )

        if len(hit) == 0:
            fail(
                f"{qacc}: eligible but no ranked hit"
            )

        first_rank = int(
            hit[0] + 1
        )

    else:
        first_rank = np.nan

    repaired_rows.append(
        {
            "method_type":
                "native_external",

            "method":
                METHOD,

            "model_seed":
                "NA",

            "endpoint":
                endpoint,

            "query_accession":
                qacc,

            "target_rank":
                rank,

            "eligible":
                int(
                    eligible
                ),

            "n_references":
                2374,

            "query_covered":
                1,

            "pair_coverage_query":
                1.0,

            "top1_reference":
                top1_reference,

            "top1_score":
                top1_score,

            "first_relevant_rank":
                first_rank,
        }
    )


rep = pd.DataFrame(
    repaired_rows
)

if len(rep) != 780:
    fail(
        f"Repaired query rows !=780: {len(rep)}"
    )

if (
    rep["query_accession"]
    .nunique()
    != 780
):
    fail(
        "Repaired query accessions not unique"
    )


# ============================================================
# 7. Exact eligibility gates + metrics
# ============================================================

summary_rows = []

for endpoint, rank in [
    (
        "B1",
        "species",
    ),
    (
        "B2_seen_genus",
        "genus",
    ),
    (
        "B2_unseen_genus",
        "family",
    ),
]:

    g = rep[
        rep["endpoint"]
        == endpoint
    ].copy()

    elig = g[
        g["eligible"] == 1
    ].copy()

    n_elig = len(
        elig
    )

    expected = expected_eligible[
        endpoint
    ]

    if n_elig != expected:
        fail(
            f"{endpoint}: expected eligible={expected}, "
            f"found {n_elig}"
        )

    ranks = (
        pd.to_numeric(
            elig[
                "first_relevant_rank"
            ],
            errors="coerce",
        )
    )

    if ranks.isna().any():
        fail(
            f"{endpoint}: eligible ranks contain NA"
        )

    def recall_at(k):
        return float(
            (ranks <= k)
            .sum()
            / n_elig
        )

    mrr = float(
        (1.0 / ranks)
        .sum()
        / n_elig
    )

    row = {
        "method":
            METHOD,

        "endpoint":
            endpoint,

        "target_rank":
            rank,

        "n_queries":
            len(g),

        "n_references":
            2374,

        "n_eligible":
            n_elig,

        "query_coverage":
            float(
                g[
                    "query_covered"
                ].mean()
            ),

        "pair_coverage_mean":
            float(
                g[
                    "pair_coverage_query"
                ].mean()
            ),

        "R1":
            recall_at(1),

        "R5":
            recall_at(5),

        "R10":
            recall_at(10),

        "MRR":
            mrr,
    }

    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 8. Preserve original ALL11 table, replace ONLY Dashing2 rows
#
# Existing E5 external query-level schema:
#
#   cohort       = B1 / B2_seen_genus / B2_unseen_genus
#   endpoint     = species / genus / family
#   n_candidates = 2374
#
# Internal repair dataframe currently uses:
#
#   endpoint      = cohort
#   target_rank   = endpoint
#   n_references  = n_candidates
#
# Convert explicitly before merging.
# ============================================================

old = pd.read_csv(
    OLD_ALL11,
    sep="\t",
    dtype=str,
)

required_old = [
    "method_type",
    "method",
    "model_seed",
    "cohort",
    "query_accession",
    "endpoint",
    "eligible",
    "n_candidates",
    "query_covered",
    "pair_coverage_query",
    "top1_reference",
    "top1_score",
    "first_relevant_rank",
]

missing_cols = [
    c
    for c in required_old
    if c not in old.columns
]

if missing_cols:
    fail(
        f"Old ALL11 query table missing columns: "
        f"{missing_cols}\n"
        f"columns={list(old.columns)}"
    )


# Convert repaired rows to EXACT historical E5 schema.
rep_export = rep.rename(
    columns={
        "endpoint":
            "cohort",
        "target_rank":
            "endpoint",
        "n_references":
            "n_candidates",
    }
).copy()


expected_rep_cols = [
    "method_type",
    "method",
    "model_seed",
    "cohort",
    "query_accession",
    "endpoint",
    "eligible",
    "n_candidates",
    "query_covered",
    "pair_coverage_query",
    "top1_reference",
    "top1_score",
    "first_relevant_rank",
]

missing_rep = [
    c
    for c in expected_rep_cols
    if c not in rep_export.columns
]

if missing_rep:
    fail(
        f"Repaired export missing columns: "
        f"{missing_rep}\n"
        f"columns={list(rep_export.columns)}"
    )

rep_export = rep_export[
    expected_rep_cols
].copy()


if len(rep_export) != 780:
    fail(
        f"Expected 780 repaired Dashing2 rows; "
        f"found {len(rep_export)}"
    )

if (
    rep_export["query_accession"]
    .nunique()
    != 780
):
    fail(
        "Repaired Dashing2 query accessions "
        "are not unique"
    )


dmask = (
    old["method"]
    == METHOD
)

if int(
    dmask.sum()
) != 780:
    fail(
        f"Old ALL11 expected 780 Dashing2 rows; "
        f"found {int(dmask.sum())}"
    )


# Strict key:
# cohort + query_accession
rep_lookup = {}

for _, r in rep_export.iterrows():

    key = (
        str(r["cohort"]),
        str(r["query_accession"]),
    )

    if key in rep_lookup:
        fail(
            f"Duplicate repaired key: {key}"
        )

    rep_lookup[key] = r


merged = old.copy()

replace_cols = [
    "endpoint",
    "eligible",
    "n_candidates",
    "query_covered",
    "pair_coverage_query",
    "top1_reference",
    "top1_score",
    "first_relevant_rank",
]


for idx in merged.index[
    dmask
]:

    cohort = str(
        merged.at[
            idx,
            "cohort",
        ]
    )

    qacc = str(
        merged.at[
            idx,
            "query_accession",
        ]
    )

    key = (
        cohort,
        qacc,
    )

    if key not in rep_lookup:
        fail(
            f"Missing repaired Dashing2 row: "
            f"{key}"
        )

    r = rep_lookup[
        key
    ]

    for c in replace_cols:

        merged.at[
            idx,
            c,
        ] = r[c]


# ------------------------------------------------------------
# Strict post-merge gates
# ------------------------------------------------------------

new_d2 = merged[
    merged["method"]
    == METHOD
].copy()

if len(new_d2) != 780:
    fail(
        f"Post-merge Dashing2 rows !=780: "
        f"{len(new_d2)}"
    )

if (
    new_d2["query_accession"]
    .nunique()
    != 780
):
    fail(
        "Post-merge Dashing2 query accessions "
        "are not unique"
    )


expected_cohort_counts = {
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193,
}

observed_cohort_counts = (
    new_d2["cohort"]
    .value_counts()
    .to_dict()
)

if observed_cohort_counts != expected_cohort_counts:
    fail(
        "Post-merge cohort counts mismatch.\n"
        f"expected={expected_cohort_counts}\n"
        f"observed={observed_cohort_counts}"
    )


expected_endpoints = {
    "B1": "species",
    "B2_seen_genus": "genus",
    "B2_unseen_genus": "family",
}

for cohort, endpoint in expected_endpoints.items():

    vals = set(
        new_d2.loc[
            new_d2["cohort"] == cohort,
            "endpoint",
        ].astype(str)
    )

    if vals != {endpoint}:
        fail(
            f"{cohort}: endpoint mismatch: {vals}"
        )


# Verify all non-Dashing2 rows remain byte-for-byte-equivalent
# after dataframe parsing/string normalization.
old_non = (
    old[
        old["method"] != METHOD
    ]
    .reset_index(drop=True)
)

new_non = (
    merged[
        merged["method"] != METHOD
    ]
    .reset_index(drop=True)
)

if not old_non.equals(
    new_non
):
    fail(
        "Non-Dashing2 rows changed unexpectedly"
    )


print()
print(
    "ALL11 SCHEMA MERGE GATE=PASS"
)

print(
    "Dashing2 rows:",
    len(new_d2),
)

print(
    "Non-Dashing2 rows preserved:",
    len(new_non),
)


# ============================================================
# 9. Write new files ONLY
# ============================================================

rep_path = (
    OUT /
    "DASHING2_REPAIRED_QUERY_LEVEL.tsv"
)

summary_path = (
    OUT /
    "DASHING2_REPAIRED_SUMMARY.tsv"
)

merged_path = (
    OUT /
    "E5_EXTERNAL_ALL11_QUERY_LEVEL_REPAIRED.tsv"
)

matrix_path = (
    OUT /
    "DASHING2_QUERY_X_REFERENCE_FLOAT32.npy"
)


rep_export.to_csv(
    rep_path,
    sep="\t",
    index=False,
)

summary.to_csv(
    summary_path,
    sep="\t",
    index=False,
)

merged.to_csv(
    merged_path,
    sep="\t",
    index=False,
)

np.save(
    matrix_path,
    M_query_ref.astype(
        np.float32,
        copy=False,
    ),
)


# ============================================================
# 10. Provenance + report
# ============================================================

sha_rows = []

for label, path in [
    (
        "raw_rectangular",
        RAW,
    ),
    (
        "old_all11_query_level",
        OLD_ALL11,
    ),
    (
        "repaired_query_level",
        rep_path,
    ),
    (
        "repaired_summary",
        summary_path,
    ),
    (
        "merged_all11_query_level",
        merged_path,
    ),
    (
        "corrected_matrix",
        matrix_path,
    ),
]:

    sha_rows.append(
        {
            "artifact":
                label,
            "path":
                str(path),
            "sha256":
                sha256_file(
                    path
                ),
        }
    )


pd.DataFrame(
    sha_rows
).to_csv(
    OUT /
    "SHA256.tsv",
    sep="\t",
    index=False,
)


report = (
    OUT /
    "DASHING2_EXTERNAL_REPAIR_REPORT.txt"
)

with report.open(
    "w"
) as f:

    f.write(
        "HGV-V3 E5 EXTERNAL DASHING2 REPAIR v1\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        f"raw={RAW}\n"
    )

    f.write(
        f"raw_sha256={sha256_file(RAW)}\n\n"
    )

    f.write(
        "Observed Dashing2 rectangular orientation:\n"
    )

    f.write(
        "  physical rows    = 2374 references\n"
    )

    f.write(
        "  physical columns = 780 external queries\n"
    )

    f.write(
        "  retrieval matrix = transpose(raw matrix)\n"
    )

    f.write(
        "  final shape      = 780 x 2374\n\n"
    )

    f.write(
        "No Dashing2 recomputation was performed.\n"
    )

    f.write(
        "No other E5 method was recomputed.\n"
    )

    f.write(
        "Original E5 source-data were not overwritten.\n\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
        "STRUCTURAL_REPAIR_GATE=PASS\n"
    )


print()
print("=" * 100)
print("DASHING2 EXTERNAL REPAIR COMPLETE")
print("=" * 100)

print(
    "Physical raw matrix :",
    M_ref_query.shape,
    "(reference x query)"
)

print(
    "Corrected matrix    :",
    M_query_ref.shape,
    "(query x reference)"
)

print()
print(summary.to_string(index=False))

print()
print(
    "Output:",
    OUT,
)

print()
print(
    "STRUCTURAL_REPAIR_GATE=PASS"
)
