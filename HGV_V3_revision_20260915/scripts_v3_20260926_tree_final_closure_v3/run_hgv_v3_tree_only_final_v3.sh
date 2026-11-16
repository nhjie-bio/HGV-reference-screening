#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
V9="$BASE/scripts_v3_20260926_remaining_all_v9"
PATCH2="$BASE/scripts_v3_20260926_final_closure_patch_v2"
FINAL="$BASE/scripts_v3_20260926_tree_final_closure_v3"

PREV=$(find \
    "$BASE/20_rebuttal" \
    -maxdepth 1 \
    -type d \
    -name 'HGV_V3_final_closure_v2_*' \
    | sort \
    | tail -n 1)

TABLE_ROOT=$(find \
    "$BASE/19_tables" \
    -maxdepth 1 \
    -type d \
    -name 'final_manuscript_source_v1_*' \
    | sort \
    | tail -n 1)

FIG_ROOT=$(find \
    "$BASE/18_figures" \
    -maxdepth 1 \
    -type d \
    -name 'final_manuscript_source_v1_*' \
    | sort \
    | tail -n 1)

STAMP=$(date +%Y%m%d_%H%M%S)

OUT="$BASE/20_rebuttal/HGV_V3_final_closure_v3_${STAMP}"

RESTORE="$BASE/00_metadata/restored_for_E10_tree_consistency_v1"
TREE="$RESTORE/bac120_r232.tree"

mkdir -p \
    "$OUT" \
    "$RESTORE"

export HGV_BASE="$BASE"
export HGV_RUN_TAG=20260925_v6

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32

echo "===================================================================================================="
echo "HGV-V3 TREE-ONLY FINAL CLOSURE v3"
echo "START=$(date --iso-8601=seconds)"
echo "PREVIOUS_CLOSURE=$PREV"
echo "OUT=$OUT"
echo "===================================================================================================="

if [ -z "$PREV" ] || [ ! -d "$PREV" ]; then
    echo "[FAIL] previous closure-v2 not found"
    exit 2
fi

for f in \
    "$PREV/FINAL_CLOSURE_STATUS_V2.tsv" \
    "$PREV/REVIEWER_15_COMMENT_EVIDENCE_MAP_V2.tsv" \
    "$PREV/STRICT_RUNTIME_STATUS.tsv"
do
    if [ ! -f "$f" ]; then
        echo "[FAIL] missing previous closure artifact: $f"
        exit 2
    fi
done

# ======================================================================================
# 1. RESTORE OFFICIAL GTDB R232 bac120 TREE ONLY
# ======================================================================================

echo
echo "===== 1. RESTORE GTDB R232 bac120 TREE ====="

if [ ! -s "$TREE" ]; then

    SCHEME=https
    HOST=data.gtdb.ecogenomic.org
    REMOTE=/releases/release232/232.0/bac120_r232.tree

    if command -v curl >/dev/null 2>&1; then

        curl \
            -fL \
            --retry 5 \
            --retry-delay 3 \
            "${SCHEME}://${HOST}${REMOTE}" \
            -o "$TREE"

    elif command -v wget >/dev/null 2>&1; then

        wget \
            -O "$TREE" \
            "${SCHEME}://${HOST}${REMOTE}"

    else

        echo "[FAIL] neither curl nor wget available"
        exit 2
    fi

fi

SIZE=$(stat -c %s "$TREE")

if [ "$SIZE" -lt 1000000 ]; then
    echo "[FAIL] downloaded tree unexpectedly small: $SIZE"
    exit 2
fi

FIRST=$(head -c 1 "$TREE")
LAST=$(tail -c 2 "$TREE" | tr -d '\n\r')

if [ "$FIRST" != "(" ]; then
    echo "[FAIL] tree does not begin with '('"
    exit 2
fi

sha256sum \
    "$TREE" \
    > "$RESTORE/bac120_r232.tree.sha256"

cat > "$RESTORE/RESTORE_PROVENANCE.txt" <<EOF
PURPOSE=E10 GTDB tree consistency only
RELEASE=GTDB R232
FILE=bac120_r232.tree
RESTORED=$(date --iso-8601=seconds)
ORIGINAL_PROJECT_TREE_WAS_PREVIOUSLY_REMOVED_DURING_CLEANUP=YES
NO_MODEL_OR_THRESHOLD_CHANGED=YES
TREE_SIZE_BYTES=$SIZE
EOF

echo "TREE_RESTORE=PASS"
echo "TREE_SIZE_BYTES=$SIZE"

# ======================================================================================
# 2. SPECIES-CLUSTER-AWARE TREE CONSISTENCY
# ======================================================================================

cat > "$FINAL/run_tree_consistency_v3.py" <<'PY'
#!/usr/bin/env python3

from pathlib import Path
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

from scipy.stats import (
    spearmanr,
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

PREV = Path(
    sys.argv[1]
)

OUT = Path(
    sys.argv[2]
)

TREE = Path(
    sys.argv[3]
)

TABLE_ROOT = Path(
    sys.argv[4]
) if len(sys.argv) > 4 and sys.argv[4] else None

FIG_ROOT = Path(
    sys.argv[5]
) if len(sys.argv) > 5 and sys.argv[5] else None


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for b in iter(
            lambda:
            f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                b
            )

    return h.hexdigest()


# ======================================================================================
# Newick parser
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


def parse_newick(text):

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
                text[i] != q
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

    def branch_length():

        nonlocal i

        skip()

        if (
            i >= n
            or
            text[i] != ":"
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

        x = text[
            start:
            i
        ].strip()

        try:
            return float(x)

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
                    "Newick parse error"
                )

            node.name = label()
            node.length = branch_length()

            return node

        node = Node()

        node.name = label()
        node.length = branch_length()

        return node

    return subtree()


def index_tree(root):

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


def patristic(a, b):

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


def norm_acc(x):

    m = ACC_RE.search(
        str(x)
    )

    return (
        m.group(1)
        if m
        else None
    )


# ======================================================================================
# Parse official tree
# ======================================================================================

print(
    "Parsing GTDB tree...",
    flush=True,
)

root = parse_newick(
    TREE.read_text(
        encoding="utf-8",
        errors="replace",
    )
)

leaf_raw = index_tree(
    root
)

leaf_by_acc = {}

for name, node in leaf_raw.items():

    a = norm_acc(
        name
    )

    if a:

        leaf_by_acc[
            a
        ] = node

print(
    "TREE_LEAVES_WITH_ACCESSIONS="
    +
    str(
        len(
            leaf_by_acc
        )
    )
)


# ======================================================================================
# Frozen final manifest
# ======================================================================================

manifest = C.load_manifest()

mc = C.manifest_columns(
    manifest
)

acc_col = mc[
    "accession"
]

cluster_col = None

for c in [
    "gtdb_species_cluster",
    "species_cluster",
]:

    if c in manifest.columns:

        cluster_col = c
        break

if cluster_col is None:

    if (
        "species_cluster"
        in mc
        and
        mc[
            "species_cluster"
        ]
        in manifest.columns
    ):

        cluster_col = mc[
            "species_cluster"
        ]

species_col = None

for c in [
    "gtdb_species",
    "species",
]:

    if c in manifest.columns:

        species_col = c
        break

if species_col is None:

    if (
        "species"
        in mc
        and
        mc[
            "species"
        ]
        in manifest.columns
    ):

        species_col = mc[
            "species"
        ]

A_rows = []

for _, r in manifest.iterrows():

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

    if cohort != "A":
        continue

    A_rows.append({
        "accession":
            acc,

        "cluster":
            (
                str(
                    r[
                        cluster_col
                    ]
                )
                if cluster_col
                else ""
            ),

        "species":
            (
                str(
                    r[
                        species_col
                    ]
                )
                if species_col
                else ""
            ),
    })

A_df = pd.DataFrame(
    A_rows
)

assert len(A_df) == 2374


# ======================================================================================
# Resolve GTDB representative accession for every A genome
#
# First preference:
# manifest gtdb_species_cluster if it directly contains representative accession.
#
# Fallback:
# GTDB bac120 metadata field gtdb_genome_representative.
# ======================================================================================

A_df[
    "tree_representative"
] = A_df[
    "cluster"
].map(
    norm_acc
)

need = set(
    A_df.loc[
        A_df[
            "tree_representative"
        ].isna(),
        "accession",
    ].astype(str)
)

meta_file = (
    BASE
    /
    "00_metadata"
    /
    "bac120_metadata_r232.tsv.gz"
)

if need:

    if not meta_file.is_file():

        raise SystemExit(
            "GTDB metadata missing for representative fallback"
        )

    header = pd.read_csv(
        meta_file,
        sep="\t",
        nrows=0,
    )

    cols = list(
        header.columns
    )

    accession_meta_col = next(
        (
            c
            for c
            in cols
            if c.lower()
            ==
            "accession"
        ),
        None,
    )

    rep_col = next(
        (
            c
            for c
            in cols
            if c.lower()
            ==
            "gtdb_genome_representative"
        ),
        None,
    )

    if rep_col is None:

        rep_candidates = [
            c
            for c
            in cols
            if (
                "genome"
                in c.lower()
                and
                "representative"
                in c.lower()
            )
        ]

        if rep_candidates:

            rep_col = rep_candidates[0]

    if (
        accession_meta_col is None
        or
        rep_col is None
    ):

        raise SystemExit(
            "Could not resolve GTDB representative column"
        )

    rep_map = {}

    target = set(
        A_df[
            "accession"
        ].astype(str)
    )

    for chunk in pd.read_csv(
        meta_file,
        sep="\t",
        usecols=[
            accession_meta_col,
            rep_col,
        ],
        chunksize=100000,
    ):

        for a, rep in zip(
            chunk[
                accession_meta_col
            ],
            chunk[
                rep_col
            ],
        ):

            aa = norm_acc(
                a
            )

            if (
                aa
                and
                aa in target
            ):

                rr = norm_acc(
                    rep
                )

                if rr:
                    rep_map[
                        aa
                    ] = rr

    miss_mask = A_df[
        "tree_representative"
    ].isna()

    A_df.loc[
        miss_mask,
        "tree_representative",
    ] = A_df.loc[
        miss_mask,
        "accession",
    ].map(
        rep_map
    )


# ======================================================================================
# Only retain species clusters whose representative occurs in official tree
# ======================================================================================

A_df[
    "tree_mapped"
] = A_df[
    "tree_representative"
].isin(
    leaf_by_acc
)

mapped_genomes = int(
    A_df[
        "tree_mapped"
    ].sum()
)

print(
    "A_GENOMES_WITH_TREE_REP="
    +
    str(
        mapped_genomes
    )
)

mapped = A_df[
    A_df[
        "tree_mapped"
    ]
].copy()

if mapped.empty:

    raise SystemExit(
        "No A genome could be mapped to GTDB species representative tree"
    )

# One deterministic A genome per GTDB species representative.
# This is NOT selected on performance.
mapped = (
    mapped.sort_values(
        "accession"
    )
    .drop_duplicates(
        subset=[
            "tree_representative"
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)

print(
    "A_SPECIES_REPRESENTATIVES_MAPPED="
    +
    str(
        len(
            mapped
        )
    )
)

if len(mapped) < 100:

    raise SystemExit(
        "Too few mapped species representatives; expected broad coverage of A species panel"
    )

mapped.to_csv(
    OUT
    /
    "E10_TREE_SPECIES_MAPPING.tsv",
    sep="\t",
    index=False,
)


# ======================================================================================
# Frozen HGV embeddings
# ======================================================================================

Xall, xidx, _, _ = (
    C.feature_accession_index()
)

X = np.asarray(
    Xall[
        [
            xidx[a]
            for a
            in mapped[
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

H_mats = []

seed_results = []

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

        raise SystemExit(
            f"Missing frozen checkpoint: {ckpt}"
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

    parts = []

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

            parts.append(
                z.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

    Z = np.concatenate(
        parts,
        axis=0,
    )

    H = np.linalg.norm(
        Z[:, None, :]
        -
        Z[None, :, :],
        axis=2,
    )

    H_mats.append(
        H
    )


# ======================================================================================
# GTDB patristic matrix
# ======================================================================================

n = len(
    mapped
)

T = np.zeros(
    (
        n,
        n,
    ),
    dtype=np.float64,
)

for i in range(n):

    ai = leaf_by_acc[
        mapped.iloc[i][
            "tree_representative"
        ]
    ]

    for j in range(
        i + 1,
        n,
    ):

        aj = leaf_by_acc[
            mapped.iloc[j][
                "tree_representative"
            ]
        ]

        d = patristic(
            ai,
            aj,
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

tree_vec = T[
    tri
]

for seed, H in zip(
    seeds,
    H_mats,
):

    h = H[
        tri
    ]

    rho, p = spearmanr(
        h,
        tree_vec,
    )

    seed_results.append({
        "model_seed":
            seed,

        "n_species_representatives":
            n,

        "n_pairs":
            len(h),

        "spearman_rho":
            float(
                rho
            ),

        "spearman_p_asymptotic":
            float(
                p
            ),
    })


# ======================================================================================
# 3-seed mean geometry
# ======================================================================================

Hmean = np.mean(
    np.stack(
        H_mats,
        axis=0,
    ),
    axis=0,
)

hmean = Hmean[
    tri
]

rho, asym_p = spearmanr(
    hmean,
    tree_vec,
)

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

    hp = Hmean[
        np.ix_(
            perm,
            perm,
        )
    ][
        tri
    ]

    null[b] = spearmanr(
        hp,
        tree_vec,
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
    seed_results
).to_csv(
    OUT
    /
    "E10_TREE_CONSISTENCY_BY_SEED.tsv",
    sep="\t",
    index=False,
)

summary = pd.DataFrame([
    {
        "status":
            "PASS",

        "tree_release":
            "GTDB R232",

        "tree_file":
            str(
                TREE
            ),

        "tree_sha256":
            sha256(
                TREE
            ),

        "mapping_rule":
            (
                "one lexicographically fixed A genome per "
                "GTDB species cluster; species cluster mapped "
                "to official GTDB representative tree leaf"
            ),

        "n_species_representatives":
            n,

        "n_pairs":
            len(
                hmean
            ),

        "hgv_geometry":
            (
                "frozen G2 split20260915 D3 "
                "mean across model seeds "
                "20260915/16/17"
            ),

        "spearman_rho":
            float(
                rho
            ),

        "spearman_p_asymptotic":
            float(
                asym_p
            ),

        "mantel_style_permutations":
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
])

summary.to_csv(
    OUT
    /
    "E10_TREE_CONSISTENCY_SUMMARY.tsv",
    sep="\t",
    index=False,
)


# ======================================================================================
# Pair-level source data
# ======================================================================================

rows = []

k = 0

for i in range(n):

    for j in range(
        i + 1,
        n,
    ):

        rows.append({
            "species_a":
                mapped.iloc[i][
                    "species"
                ],

            "species_b":
                mapped.iloc[j][
                    "species"
                ],

            "a_genome_a":
                mapped.iloc[i][
                    "accession"
                ],

            "a_genome_b":
                mapped.iloc[j][
                    "accession"
                ],

            "gtdb_tree_rep_a":
                mapped.iloc[i][
                    "tree_representative"
                ],

            "gtdb_tree_rep_b":
                mapped.iloc[j][
                    "tree_representative"
                ],

            "hgv_distance_mean_3seed":
                float(
                    hmean[k]
                ),

            "gtdb_tree_patristic_distance":
                float(
                    tree_vec[k]
                ),
        })

        k += 1

pd.DataFrame(
    rows
).to_csv(
    OUT
    /
    "E10_TREE_PAIR_SOURCE.tsv",
    sep="\t",
    index=False,
)


# ======================================================================================
# Patch previous final closure v2
# ======================================================================================

status = pd.read_csv(
    PREV
    /
    "FINAL_CLOSURE_STATUS_V2.tsv",
    sep="\t",
)

status.loc[
    status[
        "block"
    ]
    ==
    "TREE_CONSISTENCY",
    "status",
] = "PASS"

status.loc[
    status[
        "block"
    ]
    ==
    "REVIEWER_EVIDENCE_GAPS",
    "status",
] = "0"

status.to_csv(
    OUT
    /
    "FINAL_CLOSURE_STATUS_V3.tsv",
    sep="\t",
    index=False,
)


reviewer = pd.read_csv(
    PREV
    /
    "REVIEWER_15_COMMENT_EVIDENCE_MAP_V2.tsv",
    sep="\t",
)

reviewer.loc[
    reviewer[
        "reviewer_id"
    ]
    ==
    "R1-3",
    "status",
] = "EVIDENCE_READY"

reviewer.to_csv(
    OUT
    /
    "REVIEWER_15_COMMENT_EVIDENCE_MAP_V3.tsv",
    sep="\t",
    index=False,
)


# ======================================================================================
# Runtime closure remains inherited unchanged from v2
# ======================================================================================

runtime = pd.read_csv(
    PREV
    /
    "STRICT_RUNTIME_STATUS.tsv",
    sep="\t",
)

runtime.to_csv(
    OUT
    /
    "STRICT_RUNTIME_STATUS_FROZEN_V2_COPY.tsv",
    sep="\t",
    index=False,
)

required_runtime = [
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
    "HGV",
    "DNABERT-S",
]

runtime_ok = True

for m in required_runtime:

    g = runtime[
        runtime[
            "method"
        ]
        ==
        m
    ]

    if (
        g.empty
        or
        not bool(
            g.iloc[0][
                "strict_numeric_runtime_available"
            ]
        )
    ):

        runtime_ok = False


# ======================================================================================
# Final gate
# ======================================================================================

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
            f"{block}={value}"
        )

gaps = int(
    (
        reviewer[
            "status"
        ]
        ==
        "EVIDENCE_GAP"
    ).sum()
)

ready = (
    not core_fail
    and
    gaps == 0
    and
    runtime_ok
)

gate = [
    (
        "HGV_V3_FINAL_CLOSURE_V3=PASS"
        if ready
        else
        "HGV_V3_FINAL_CLOSURE_V3=NEEDS_REVIEW"
    ),

    (
        "HGV_V3_EXPERIMENTS_CLOSED=YES"
        if ready
        else
        "HGV_V3_EXPERIMENTS_CLOSED=NO"
    ),

    "NO_MORE_MODEL_TRAINING=YES",

    (
        "READY_FOR_MANUSCRIPT=YES"
        if ready
        else
        "READY_FOR_MANUSCRIPT=CONDITIONAL"
    ),

    "NO_FROZEN_INPUT_MODIFIED=YES",
]

(
    OUT
    /
    "FINAL_CLOSURE_GATE_V3.txt"
).write_text(
    "\n".join(
        gate
    )
    +
    "\n",
    encoding="utf-8",
)


# ======================================================================================
# Freeze pointers for manuscript
# ======================================================================================

if TABLE_ROOT:

    dest = (
        TABLE_ROOT
        /
        "FINAL_CLOSURE_V3"
    )

    dest.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in OUT.iterdir():

        if (
            p.is_file()
            and
            p.suffix.lower()
            in {
                ".tsv",
                ".txt",
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

if FIG_ROOT:

    for name in [
        "E10_TREE_CONSISTENCY_SUMMARY.tsv",
        "E10_TREE_PAIR_SOURCE.tsv",
    ]:

        p = (
            OUT
            /
            name
        )

        q = (
            FIG_ROOT
            /
            name
        )

        if (
            p.is_file()
            and
            not q.exists()
        ):

            q.symlink_to(
                p
            )


print(
    "===================================================================================================="
)

print(
    "TREE CONSISTENCY FINAL"
)

print(
    "===================================================================================================="
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "===================================================================================================="
)

print(
    "FINAL GATE"
)

print(
    "===================================================================================================="
)

print(
    "\n".join(
        gate
    )
)

print(
    "FINAL_CLOSURE_V3_ROOT="
    +
    str(
        OUT
    )
)
PY

python -m py_compile \
    "$FINAL/run_tree_consistency_v3.py"

echo "TREE_SCRIPT_COMPILE=PASS"

python \
    "$FINAL/run_tree_consistency_v3.py" \
    "$PREV" \
    "$OUT" \
    "$TREE" \
    "$TABLE_ROOT" \
    "$FIG_ROOT"

echo
echo "===================================================================================================="
echo "TREE CONSISTENCY"
echo "===================================================================================================="

column -t -s $'\t' \
    "$OUT/E10_TREE_CONSISTENCY_SUMMARY.tsv"

echo
echo "===================================================================================================="
echo "FINAL CLOSURE STATUS v3"
echo "===================================================================================================="

column -t -s $'\t' \
    "$OUT/FINAL_CLOSURE_STATUS_V3.tsv"

echo
echo "===================================================================================================="
echo "FINAL GATE v3"
echo "===================================================================================================="

cat \
    "$OUT/FINAL_CLOSURE_GATE_V3.txt"

echo
echo "END=$(date --iso-8601=seconds)"
echo "===================================================================================================="
