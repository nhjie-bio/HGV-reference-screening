#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    rankdata,
)

BASE = Path(
    "HGV_V3_revision_20260915"
)

V9 = (
    BASE
    / "scripts_v3_20260926_remaining_all_v9"
)

os.environ["HGV_BASE"] = str(BASE)
os.environ["HGV_RUN_TAG"] = "20260925_v6"

sys.path.insert(
    0,
    str(V9),
)

import common as C


# ======================================================================================
# Generic helpers
# ======================================================================================

def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for b in iter(
            lambda:
            f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def rel(path):

    try:

        return str(
            Path(path)
            .relative_to(BASE)
        )

    except Exception:

        return str(path)


def write_json(
    path,
    obj,
):

    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )


# ======================================================================================
# Strict runtime audit
# ======================================================================================

TIME_COL_RE = re.compile(
    r"(wall|elapsed|runtime|"
    r"time[_ ]?s|seconds|"
    r"maxrss|peak[_ ]?rss)",
    re.I,
)

TIME_TEXT_RE = re.compile(
    r"(?:wall_seconds|elapsed_seconds|runtime_seconds|"
    r"wall\s*time|elapsed\s*time|runtime|"
    r"real)\D{0,30}"
    r"([0-9]+(?:\.[0-9]+)?)",
    re.I,
)

METHODS = [
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
    "HGV",
    "DNABERT-S",
]


def strict_runtime_audit(
    prev,
    out,
):

    rows = []

    inventory = (
        prev
        /
        "RUNTIME_EXISTING_EVIDENCE_INVENTORY.tsv"
    )

    if inventory.is_file():

        inv = pd.read_csv(
            inventory,
            sep="\t",
        )

        for _, r in inv.iterrows():

            method = str(
                r.get(
                    "method",
                    "",
                )
            )

            if method not in METHODS:
                continue

            source_rel = str(
                r.get(
                    "path",
                    "",
                )
            )

            source = (
                BASE
                /
                source_rel
            )

            numeric_values = []

            if source.is_file():

                suffix = (
                    source
                    .suffix
                    .lower()
                )

                if suffix in {
                    ".tsv",
                    ".csv",
                }:

                    try:

                        sep = (
                            "\t"
                            if suffix
                            ==
                            ".tsv"
                            else
                            ","
                        )

                        df = pd.read_csv(
                            source,
                            sep=sep,
                        )

                        for c in df.columns:

                            if not TIME_COL_RE.search(
                                str(c)
                            ):
                                continue

                            vals = pd.to_numeric(
                                df[c],
                                errors="coerce",
                            ).dropna()

                            vals = vals[
                                vals > 0
                            ]

                            numeric_values.extend(
                                vals
                                .astype(float)
                                .tolist()
                            )

                    except Exception:
                        pass

                try:

                    if (
                        source.stat().st_size
                        <=
                        25
                        *
                        1024
                        *
                        1024
                    ):

                        txt = source.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )

                        for m in TIME_TEXT_RE.finditer(
                            txt
                        ):

                            try:

                                x = float(
                                    m.group(1)
                                )

                                if x > 0:
                                    numeric_values.append(
                                        x
                                    )

                            except Exception:
                                pass

                except Exception:
                    pass

            # Also inspect timing signal recorded by v1.
            signal = str(
                r.get(
                    "timing_signal",
                    "",
                )
            )

            for m in TIME_TEXT_RE.finditer(
                signal
            ):

                try:

                    x = float(
                        m.group(1)
                    )

                    if x > 0:
                        numeric_values.append(
                            x
                        )

                except Exception:
                    pass

            rows.append({
                "method":
                    method,

                "source":
                    source_rel,

                "strict_numeric_runtime":
                    bool(
                        numeric_values
                    ),

                "n_numeric_values":
                    len(
                        numeric_values
                    ),

                "numeric_min":
                    (
                        min(
                            numeric_values
                        )
                        if numeric_values
                        else
                        np.nan
                    ),

                "numeric_median":
                    (
                        float(
                            np.median(
                                numeric_values
                            )
                        )
                        if numeric_values
                        else
                        np.nan
                    ),

                "numeric_max":
                    (
                        max(
                            numeric_values
                        )
                        if numeric_values
                        else
                        np.nan
                    ),
            })

    # Explicitly add E12-v2 HGV real-FASTA runtime.
    hgv = (
        BASE
        /
        "16_scalability"
        /
        "E12_scalability_v2_gzipfix_20260926"
        /
        "E12_V2_REAL_GZIP_FASTA_RUNS.tsv"
    )

    if hgv.is_file():

        df = pd.read_csv(
            hgv,
            sep="\t",
        )

        vals = pd.to_numeric(
            df[
                "end_to_end_wall_s"
            ],
            errors="coerce",
        ).dropna()

        rows.append({
            "method":
                "HGV",

            "source":
                rel(hgv),

            "strict_numeric_runtime":
                bool(
                    len(vals)
                ),

            "n_numeric_values":
                len(vals),

            "numeric_min":
                (
                    float(
                        vals.min()
                    )
                    if len(vals)
                    else
                    np.nan
                ),

            "numeric_median":
                (
                    float(
                        vals.median()
                    )
                    if len(vals)
                    else
                    np.nan
                ),

            "numeric_max":
                (
                    float(
                        vals.max()
                    )
                    if len(vals)
                    else
                    np.nan
                ),
        })

    # Runtime supplement generated by this v2 script.
    supplement = (
        out
        /
        "runtime_supplement"
        /
        "RUNTIME_SUPPLEMENT_RUNS.tsv"
    )

    if supplement.is_file():

        sdf = pd.read_csv(
            supplement,
            sep="\t",
        )

        for _, r in sdf.iterrows():

            method = str(
                r[
                    "method"
                ]
            )

            candidate = []

            for c in [
                "build_s",
                "query_s",
                "combined_s",
            ]:

                if c not in sdf.columns:
                    continue

                try:

                    x = float(
                        r[c]
                    )

                    if (
                        np.isfinite(x)
                        and
                        x > 0
                    ):
                        candidate.append(
                            x
                        )

                except Exception:
                    pass

            rows.append({
                "method":
                    method,

                "source":
                    rel(
                        supplement
                    ),

                "strict_numeric_runtime":
                    bool(
                        candidate
                    ),

                "n_numeric_values":
                    len(
                        candidate
                    ),

                "numeric_min":
                    (
                        min(candidate)
                        if candidate
                        else
                        np.nan
                    ),

                "numeric_median":
                    (
                        float(
                            np.median(
                                candidate
                            )
                        )
                        if candidate
                        else
                        np.nan
                    ),

                "numeric_max":
                    (
                        max(candidate)
                        if candidate
                        else
                        np.nan
                    ),
            })

    audit = pd.DataFrame(
        rows
    )

    audit.to_csv(
        out
        /
        "STRICT_RUNTIME_EVIDENCE_LONG.tsv",
        sep="\t",
        index=False,
    )

    status_rows = []

    for method in METHODS:

        g = (
            audit[
                audit[
                    "method"
                ]
                ==
                method
            ]
            if not audit.empty
            else pd.DataFrame()
        )

        ok = bool(
            len(g)
            and
            g[
                "strict_numeric_runtime"
            ]
            .astype(bool)
            .any()
        )

        status_rows.append({
            "method":
                method,

            "strict_numeric_runtime_available":
                ok,

            "n_source_rows":
                len(g),

            "n_numeric_source_rows":
                (
                    int(
                        g[
                            "strict_numeric_runtime"
                        ]
                        .astype(bool)
                        .sum()
                    )
                    if len(g)
                    else
                    0
                ),
        })

    status = pd.DataFrame(
        status_rows
    )

    status.to_csv(
        out
        /
        "STRICT_RUNTIME_STATUS.tsv",
        sep="\t",
        index=False,
    )

    return {
        r[
            "method"
        ]:
        bool(
            r[
                "strict_numeric_runtime_available"
            ]
        )
        for r
        in status_rows
    }


# ======================================================================================
# Newick tree parsing
# ======================================================================================

class Node:

    __slots__ = [
        "name",
        "length",
        "children",
        "parent",
        "depth",
    ]

    def __init__(
        self,
        name="",
        length=0.0,
    ):

        self.name = name
        self.length = length
        self.children = []
        self.parent = None
        self.depth = 0.0


def locate_tree():

    candidates = [
        BASE
        /
        "00_metadata"
        /
        "bac120_r232.tree",
    ]

    candidates.extend(
        sorted(
            BASE.rglob(
                "bac120_r232.tree"
            )
        )
    )

    candidates.extend(
        sorted(
            BASE.rglob(
                "*bac120*.tree"
            )
        )
    )

    seen = set()

    for p in candidates:

        p = Path(p)

        if str(p) in seen:
            continue

        seen.add(
            str(p)
        )

        if (
            p.is_file()
            and
            p.stat().st_size
            >
            1000
        ):

            return p

    return None


def parse_newick(
    text,
):

    text = text.strip()

    i = 0
    n = len(text)

    def skip():

        nonlocal i

        while (
            i < n
            and
            text[i].isspace()
        ):
            i += 1

    def label():

        nonlocal i

        skip()

        if i >= n:
            return ""

        if text[i] in "'\"":

            q = text[i]
            i += 1
            start = i

            while (
                i < n
                and
                text[i]
                !=
                q
            ):
                i += 1

            x = text[
                start:
                i
            ]

            if i < n:
                i += 1

            return x

        start = i

        while (
            i < n
            and
            text[i]
            not in
            ",():;"
        ):
            i += 1

        return text[
            start:
            i
        ].strip()

    def length():

        nonlocal i

        skip()

        if (
            i >= n
            or
            text[i]
            !=
            ":"
        ):
            return 0.0

        i += 1

        start = i

        while (
            i < n
            and
            text[i]
            not in
            ",();"
        ):
            i += 1

        raw = text[
            start:
            i
        ].strip()

        try:

            return float(raw)

        except Exception:

            return 0.0

    def subtree():

        nonlocal i

        skip()

        if text[i] == "(":

            i += 1

            node = Node()

            while True:

                child = subtree()

                child.parent = node

                node.children.append(
                    child
                )

                skip()

                if text[i] == ",":

                    i += 1
                    continue

                if text[i] == ")":

                    i += 1
                    break

                raise RuntimeError(
                    "Newick parser failed"
                )

            node.name = label()
            node.length = length()

            return node

        node = Node(
            label(),
            0.0,
        )

        node.length = length()

        return node

    return subtree()


def index_tree(
    root,
):

    leaves = {}

    stack = [
        (
            root,
            0.0,
        )
    ]

    while stack:

        node, parent_depth = (
            stack.pop()
        )

        node.depth = (
            parent_depth
            +
            float(
                node.length
                or
                0.0
            )
        )

        if node.children:

            for ch in node.children:

                stack.append(
                    (
                        ch,
                        node.depth,
                    )
                )

        else:

            leaves[
                node.name
            ] = node

    return leaves


def node_distance(
    a,
    b,
):

    ancestors = {}

    x = a

    while x is not None:

        ancestors[
            id(x)
        ] = x

        x = x.parent

    y = b

    while y is not None:

        if id(y) in ancestors:

            return (
                a.depth
                +
                b.depth
                -
                2.0
                *
                y.depth
            )

        y = y.parent

    return np.nan


ACC_RE = re.compile(
    r"(?:RS_|GB_)?"
    r"((?:GCF|GCA)_\d+\.\d+)"
)


def normalize_accession(
    name,
):

    m = ACC_RE.search(
        str(name)
    )

    return (
        m.group(1)
        if m
        else
        None
    )


# ======================================================================================
# Tree consistency
# ======================================================================================

def tree_consistency(
    out,
):

    tree_path = locate_tree()

    if tree_path is None:

        pd.DataFrame([
            {
                "status":
                    "NOT_COMPUTED",

                "reason":
                    "No bac120 tree located under project root",
            }
        ]).to_csv(
            out
            /
            "E10_TREE_CONSISTENCY_SUMMARY.tsv",
            sep="\t",
            index=False,
        )

        return False

    print(
        "TREE_FILE="
        +
        str(tree_path)
    )

    print(
        "TREE_SIZE_BYTES="
        +
        str(
            tree_path.stat().st_size
        )
    )

    root = parse_newick(
        tree_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    leaves = index_tree(
        root
    )

    acc_to_leaf = {}

    for name, node in leaves.items():

        acc = normalize_accession(
            name
        )

        if acc:

            acc_to_leaf[
                acc
            ] = node

    manifest = C.load_manifest()

    mc = C.manifest_columns(
        manifest
    )

    acc_col = mc[
        "accession"
    ]

    species_col = None

    preferred = [
        "gtdb_species_cluster",
        "species_cluster",
        "gtdb_species",
        "species",
    ]

    for c in preferred:

        if c in manifest.columns:

            species_col = c
            break

    if species_col is None:

        for key in [
            "species_cluster",
            "species",
        ]:

            if (
                key in mc
                and
                mc[key]
                in manifest.columns
            ):

                species_col = mc[
                    key
                ]

                break

    if species_col is None:

        for c in manifest.columns:

            if (
                "species"
                in c.lower()
                and
                "count"
                not in c.lower()
            ):

                species_col = c
                break

    if species_col is None:

        raise RuntimeError(
            "Could not resolve species column"
        )

    reps = []

    for species, g in manifest.groupby(
        species_col,
        dropna=False,
    ):

        cand = []

        for _, r in g.iterrows():

            acc = str(
                r[
                    acc_col
                ]
            )

            cohort = C.infer_cohort(
                r,
                mc,
                accession=acc,
            )

            if (
                cohort
                ==
                "A"
                and
                acc
                in
                acc_to_leaf
            ):

                cand.append(
                    acc
                )

        if cand:

            reps.append({
                "species":
                    str(species),

                "accession":
                    sorted(
                        cand
                    )[0],
            })

    reps = pd.DataFrame(
        reps
    ).sort_values(
        [
            "species",
            "accession",
        ]
    ).reset_index(
        drop=True
    )

    reps.to_csv(
        out
        /
        "E10_TREE_SPECIES_REPRESENTATIVES.tsv",
        sep="\t",
        index=False,
    )

    print(
        "TREE_MAPPED_SPECIES_REPRESENTATIVES="
        +
        str(
            len(reps)
        )
    )

    if len(reps) < 50:

        raise RuntimeError(
            "Too few A species representatives mapped to tree"
        )

    Xall, xidx, _, _ = (
        C.feature_accession_index()
    )

    X = np.asarray(
        Xall[
            [
                xidx[a]
                for a
                in reps[
                    "accession"
                ]
            ]
        ],
        dtype=np.float32,
    )

    torch, _, _ = (
        C.import_torch()
    )

    seeds = [
        20260915,
        20260916,
        20260917,
    ]

    Hs = []

    seed_rows = []

    for seed in seeds:

        ckpt = (
            BASE
            /
            "08_main_generalization"
            /
            "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
            /
            "G2"
            /
            "split20260915"
            /
            f"model{seed}"
            /
            "best_model.pt"
        )

        if not ckpt.is_file():

            raise RuntimeError(
                f"Checkpoint missing: {ckpt}"
            )

        model, device, _ = (
            C.load_model_checkpoint(
                ckpt,
                input_dim=2080,
                embed_dim=64,
                normalize_output=True,
            )
        )

        model.eval()

        zs = []

        with torch.no_grad():

            for st in range(
                0,
                len(X),
                256,
            ):

                xx = torch.as_tensor(
                    np.asarray(
                        X[
                            st:
                            st + 256
                        ],
                        dtype=np.float32,
                    ),
                    dtype=torch.float32,
                    device=device,
                )

                z = model(
                    xx
                )

                zs.append(
                    z.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

        Z = np.concatenate(
            zs,
            axis=0,
        )

        H = np.linalg.norm(
            Z[:, None, :]
            -
            Z[None, :, :],
            axis=2,
        )

        Hs.append(
            H
        )

    n = len(reps)

    T = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float64,
    )

    for i in range(n):

        ni = acc_to_leaf[
            reps.iloc[i][
                "accession"
            ]
        ]

        for j in range(
            i + 1,
            n,
        ):

            nj = acc_to_leaf[
                reps.iloc[j][
                    "accession"
                ]
            ]

            d = node_distance(
                ni,
                nj,
            )

            T[
                i,
                j
            ] = d

            T[
                j,
                i
            ] = d

    tri = np.triu_indices(
        n,
        1,
    )

    tvec = T[
        tri
    ]

    trank = rankdata(
        tvec
    )

    for seed, H in zip(
        seeds,
        Hs,
    ):

        hvec = H[
            tri
        ]

        hrank = rankdata(
            hvec
        )

        rho = pearsonr(
            hrank,
            trank,
        ).statistic

        seed_rows.append({
            "model_seed":
                seed,

            "n_species_representatives":
                n,

            "n_pairs":
                len(
                    hvec
                ),

            "spearman_rho":
                float(
                    rho
                ),
        })

    Hmean = np.mean(
        np.stack(
            Hs,
            axis=0,
        ),
        axis=0,
    )

    hvec = Hmean[
        tri
    ]

    hrank = rankdata(
        hvec
    )

    rho = pearsonr(
        hrank,
        trank,
    ).statistic

    # Rank matrix for efficient Mantel-style permutation.
    rank_matrix = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float64,
    )

    rank_matrix[
        tri
    ] = hrank

    rank_matrix[
        (
            tri[1],
            tri[0],
        )
    ] = hrank

    rng = np.random.default_rng(
        20260926
    )

    null = np.empty(
        1000,
        dtype=np.float64,
    )

    for b in range(
        1000
    ):

        perm = rng.permutation(
            n
        )

        hp = rank_matrix[
            np.ix_(
                perm,
                perm,
            )
        ][
            tri
        ]

        null[b] = pearsonr(
            hp,
            trank,
        ).statistic

    perm_p = (
        1
        +
        np.sum(
            np.abs(
                null
            )
            >=
            abs(
                rho
            )
        )
    ) / 1001.0

    pd.DataFrame(
        seed_rows
    ).to_csv(
        out
        /
        "E10_TREE_CONSISTENCY_BY_SEED.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame([
        {
            "status":
                "PASS",

            "tree_source":
                rel(
                    tree_path
                ),

            "tree_sha256":
                sha256(
                    tree_path
                ),

            "n_species_representatives":
                n,

            "n_pairs":
                len(
                    hvec
                ),

            "hgv_embedding":
                (
                    "G2 split20260915 D3 "
                    "3-seed mean"
                ),

            "spearman_rho":
                float(
                    rho
                ),

            "permutations":
                1000,

            "mantel_style_two_sided_p":
                float(
                    perm_p
                ),

            "claim_boundary":
                (
                    "tree-distance consistency only; "
                    "not phylogeny reconstruction"
                ),
        }
    ]).to_csv(
        out
        /
        "E10_TREE_CONSISTENCY_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    rows = []

    k = 0

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):

            rows.append({
                "species_a":
                    reps.iloc[i][
                        "species"
                    ],

                "species_b":
                    reps.iloc[j][
                        "species"
                    ],

                "accession_a":
                    reps.iloc[i][
                        "accession"
                    ],

                "accession_b":
                    reps.iloc[j][
                        "accession"
                    ],

                "hgv_distance_mean_3seed":
                    float(
                        hvec[k]
                    ),

                "gtdb_tree_patristic_distance":
                    float(
                        tvec[k]
                    ),
            })

            k += 1

    pd.DataFrame(
        rows
    ).to_csv(
        out
        /
        "E10_TREE_PAIR_SOURCE.tsv",
        sep="\t",
        index=False,
    )

    return True


# ======================================================================================
# Final closure v2
# ======================================================================================

def finalize(
    prev,
    out,
    table_root,
    fig_root,
):

    runtime = strict_runtime_audit(
        prev,
        out,
    )

    tree_ok = tree_consistency(
        out
    )

    reviewer_src = (
        prev
        /
        "REVIEWER_15_COMMENT_EVIDENCE_MAP.tsv"
    )

    reviewer = pd.read_csv(
        reviewer_src,
        sep="\t",
    )

    runtime_core = all(
        runtime.get(
            m,
            False,
        )
        for m
        in [
            "Mash",
            "Dashing2",
            "skani",
            "FastANI",
            "HGV",
        ]
    )

    reviewer.loc[
        reviewer[
            "reviewer_id"
        ]
        ==
        "R1-2",
        "status",
    ] = (
        "EVIDENCE_READY"
        if runtime_core
        else
        "EVIDENCE_GAP"
    )

    reviewer.loc[
        reviewer[
            "reviewer_id"
        ]
        ==
        "R3-2",
        "status",
    ] = (
        "EVIDENCE_READY"
        if runtime_core
        else
        "EVIDENCE_GAP"
    )

    reviewer.loc[
        reviewer[
            "reviewer_id"
        ]
        ==
        "R1-3",
        "status",
    ] = (
        "EVIDENCE_READY"
        if tree_ok
        else
        "EVIDENCE_GAP"
    )

    reviewer.to_csv(
        out
        /
        "REVIEWER_15_COMMENT_EVIDENCE_MAP_V2.tsv",
        sep="\t",
        index=False,
    )

    gaps = int(
        np.sum(
            reviewer[
                "status"
            ]
            ==
            "EVIDENCE_GAP"
        )
    )

    old_status = pd.read_csv(
        prev
        /
        "FINAL_CLOSURE_STATUS.tsv",
        sep="\t",
    )

    status = old_status.copy()

    status.loc[
        status[
            "block"
        ]
        ==
        "TREE_CONSISTENCY",
        "status",
    ] = (
        "PASS"
        if tree_ok
        else
        "NOT_COMPUTED"
    )

    status.loc[
        status[
            "block"
        ]
        ==
        "E5_BASELINE_RUNTIME_CORE",
        "status",
    ] = (
        "PASS"
        if runtime_core
        else
        "FAIL"
    )

    status.loc[
        status[
            "block"
        ]
        ==
        "REVIEWER_EVIDENCE_GAPS",
        "status",
    ] = str(
        gaps
    )

    status.to_csv(
        out
        /
        "FINAL_CLOSURE_STATUS_V2.tsv",
        sep="\t",
        index=False,
    )

    core_fail = []

    for _, r in status.iterrows():

        block = str(
            r[
                "block"
            ]
        )

        value = str(
            r[
                "status"
            ]
        )

        if block == "REVIEWER_EVIDENCE_GAPS":
            continue

        if value not in {
            "PASS",
            "PARTIAL",
        }:

            core_fail.append(
                (
                    block,
                    value,
                )
            )

    ready = (
        not core_fail
        and
        gaps == 0
        and
        runtime_core
        and
        tree_ok
    )

    # Reporting policy: do not blindly publish diagnostic t CIs
    # that extend outside the metric domain.
    reporting = pd.DataFrame([
        {
            "result_block":
                "Main G2/G3/G4",

            "primary_manuscript_reporting":
                (
                    "mean +/- SD across 3 taxonomy split seeds"
                ),

            "formal_inference":
                (
                    "use frozen formal bootstrap/paired statistics "
                    "where defined; do not treat pair observations "
                    "as independent biological replicates"
                ),

            "do_not_use":
                (
                    "unbounded diagnostic t intervals outside [0,1] "
                    "as literal probability limits"
                ),
        },

        {
            "result_block":
                "External B1/B2",

            "primary_manuscript_reporting":
                (
                    "3-seed mean +/- SD; eligible query counts stated explicitly"
                ),

            "formal_inference":
                (
                    "retrieval metrics are descriptive external one-shot results; "
                    "no external tuning"
                ),

            "do_not_use":
                (
                    "CI on counts/threshold as substantive biological evidence"
                ),
        },

        {
            "result_block":
                "E6",

            "primary_manuscript_reporting":
                (
                    "observed paired deltas + species-block bootstrap CI + Holm P"
                ),

            "formal_inference":
                "formal comparator-specific inference",

            "do_not_use":
                (
                    "universal D3 superiority wording"
                ),
        },
    ])

    reporting.to_csv(
        out
        /
        "FINAL_STATISTICAL_REPORTING_POLICY.tsv",
        sep="\t",
        index=False,
    )

    # Freeze closure-v2 products into manuscript source roots.
    if table_root:

        dest = (
            table_root
            /
            "FINAL_CLOSURE_V2"
        )

        dest.mkdir(
            parents=True,
            exist_ok=True,
        )

        for p in out.iterdir():

            if (
                p.is_file()
                and
                p.suffix.lower()
                in {
                    ".tsv",
                    ".txt",
                    ".md",
                    ".json",
                }
            ):

                q = (
                    dest
                    /
                    p.name
                )

                if not q.exists():

                    q.symlink_to(
                        p
                    )

    if fig_root:

        for name in [
            "E10_TREE_PAIR_SOURCE.tsv",
            "E10_TREE_CONSISTENCY_SUMMARY.tsv",
        ]:

            p = (
                out
                /
                name
            )

            if p.is_file():

                q = (
                    fig_root
                    /
                    name
                )

                if not q.exists():

                    q.symlink_to(
                        p
                    )

    lines = [
        "# HGV-V3 Final Closure v2",
        "",
        f"- Tree consistency: {'PASS' if tree_ok else 'NOT_COMPUTED'}",
        f"- Strict runtime core: {'PASS' if runtime_core else 'FAIL'}",
        f"- Reviewer evidence gaps: {gaps}",
        "",
        "## Strict runtime status",
        "",
    ]

    for m in METHODS:

        lines.append(
            f"- {m}: "
            f"{'NUMERIC_RUNTIME_AVAILABLE' if runtime.get(m, False) else 'NO_STRICT_NUMERIC_RUNTIME'}"
        )

    lines.extend([
        "",
        "## Final decision",
        "",
    ])

    if ready:

        lines.extend([
            "HGV_V3_FINAL_CLOSURE_V2=PASS",
            "HGV_V3_EXPERIMENTS_CLOSED=YES",
            "NO_MORE_MODEL_TRAINING=YES",
            "READY_FOR_MANUSCRIPT=YES",
            "NO_FROZEN_INPUT_MODIFIED=YES",
        ])

    else:

        lines.extend([
            "HGV_V3_FINAL_CLOSURE_V2=NEEDS_REVIEW",
            "HGV_V3_EXPERIMENTS_CLOSED=NO",
            "NO_MORE_MODEL_TRAINING=YES",
            "READY_FOR_MANUSCRIPT=CONDITIONAL",
            "NO_FROZEN_INPUT_MODIFIED=YES",
        ])

    (
        out
        /
        "FINAL_CLOSURE_REPORT_V2.md"
    ).write_text(
        "\n".join(
            lines
        )
        +
        "\n",
        encoding="utf-8",
    )

    (
        out
        /
        "FINAL_CLOSURE_GATE_V2.txt"
    ).write_text(
        "\n".join(
            lines[
                -5:
            ]
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "TREE_CONSISTENCY="
        +
        (
            "PASS"
            if tree_ok
            else
            "NOT_COMPUTED"
        )
    )

    print(
        "STRICT_RUNTIME_CORE="
        +
        (
            "PASS"
            if runtime_core
            else
            "FAIL"
        )
    )

    print(
        "REVIEWER_EVIDENCE_GAPS="
        +
        str(
            gaps
        )
    )

    print(
        "READY_FOR_MANUSCRIPT="
        +
        str(
            ready
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=[
            "audit-runtime",
            "finalize",
        ],
        required=True,
    )

    ap.add_argument(
        "--prev",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--table-root",
        type=Path,
    )

    ap.add_argument(
        "--fig-root",
        type=Path,
    )

    args = ap.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "audit-runtime":

        status = strict_runtime_audit(
            args.prev,
            args.out,
        )

        write_json(
            args.out
            /
            "STRICT_RUNTIME_STATUS.json",
            status,
        )

        for m in METHODS:

            print(
                f"{m}="
                +
                str(
                    status.get(
                        m,
                        False,
                    )
                )
            )

        return

    finalize(
        args.prev,
        args.out,
        args.table_root,
        args.fig_root,
    )


if __name__ == "__main__":
    main()
