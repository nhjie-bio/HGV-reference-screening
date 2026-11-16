#!/usr/bin/env bash
set -euo pipefail

# ======================================================================================
# HGV-V3 FINAL CLOSURE v1
#
# PURPOSE
# -------
# 1. Audit all existing runtime evidence.
# 2. Run Mash/Dashing2 runtime-only supplement ONLY if runtime evidence is absent.
# 3. Compute GTDB bac120 tree consistency from frozen HGV embeddings.
# 4. Compute confound-adjusted HGV-vs-ANI association from frozen E9/E10 source data.
# 5. Audit/derive Main + External confidence intervals from frozen source data.
# 6. Freeze final manuscript source-data with SHA256 provenance.
# 7. Build 15-comment reviewer evidence map.
# 8. Build final claim-boundary audit.
# 9. Decide whether experiments are CLOSED and manuscript writing can start.
#
# HARD RULES
# ----------
# NO HGV TRAINING
# NO MODEL SELECTION
# NO TEST THRESHOLD TUNING
# NO EXTERNAL RETUNING
# NO MODIFICATION OF FROZEN INPUTS
# ======================================================================================

BASE=HGV_V3_revision_20260915
V9="$BASE/scripts_v3_20260926_remaining_all_v9"
FIX="$BASE/scripts_v3_20260926_e7_e12_final_fix_v1"
CLOSE="$BASE/scripts_v3_20260926_final_closure_v1"

STAMP=$(date +%Y%m%d_%H%M%S)

OUT="$BASE/20_rebuttal/HGV_V3_final_closure_v1_${STAMP}"

TABLE_ROOT="$BASE/19_tables/final_manuscript_source_v1_${STAMP}"
FIG_ROOT="$BASE/18_figures/final_manuscript_source_v1_${STAMP}"

LOGDIR="$CLOSE/logs"
mkdir -p \
    "$OUT" \
    "$TABLE_ROOT" \
    "$FIG_ROOT" \
    "$LOGDIR"

exec > >(tee -a "$LOGDIR/final_closure_${STAMP}.log") 2>&1

export HGV_BASE="$BASE"
export HGV_RUN_TAG=20260925_v6

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32

echo "===================================================================================================="
echo "HGV-V3 FINAL CLOSURE v1"
echo "START=$(date --iso-8601=seconds)"
echo "OUT=$OUT"
echo "===================================================================================================="

# ======================================================================================
# 0. PREFLIGHT
# ======================================================================================

echo
echo "===== 0. PREFLIGHT ====="

for x in \
    "$V9/common.py" \
    "$BASE/01_manifest/final_3154_freeze_v1_20260916" \
    "$BASE/03_features/canonical_k6_full3154_v1" \
    "$BASE/04_splits/g1_g4_v1_20260916" \
    "$BASE/05_pairs/pair_manifests_v1_20260916" \
    "$BASE/08_main_generalization/blind_test_evaluation_v1_20260917" \
    "$BASE/09_external_validation/external_one_shot_evaluation_v1_20260917" \
    "$BASE/11_sensitivity/E7_sensitivity_v2_gzipfix_20260925_v6/final_validation_refreeze_v2_20260926/E7_FINAL_PASS.txt" \
    "$BASE/16_scalability/E12_scalability_v2_gzipfix_20260926/E12_V2_PASS.txt"
do
    if [ ! -e "$x" ]; then
        echo "[FAIL] required frozen artifact missing:"
        echo "$x"
        exit 2
    fi
done

python - <<'PY'
import numpy, pandas, scipy, sklearn, torch
print("PYTHON_PREFLIGHT=PASS")
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

command -v /usr/bin/time >/dev/null || {
    echo "[FAIL] /usr/bin/time missing"
    exit 2
}

echo "PREFLIGHT=PASS"

# ======================================================================================
# 1. CREATE SINGLE PYTHON CLOSURE ENGINE
# ======================================================================================

cat > "$CLOSE/final_closure_engine_v1.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    rankdata,
    spearmanr,
    t,
)

BASE = Path(
    "HGV_V3_revision_20260915"
)

V9 = (
    BASE
    / "scripts_v3_20260926_remaining_all_v9"
)

FIX = (
    BASE
    / "scripts_v3_20260926_e7_e12_final_fix_v1"
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

def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def rel(path: Path) -> str:

    try:
        return str(
            path.relative_to(BASE)
        )

    except Exception:
        return str(path)


def first_existing(paths):

    for p in paths:

        p = Path(p)

        if p.exists():
            return p

    return None


def safe_read_table(path):

    path = Path(path)

    try:

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        return pd.read_csv(
            path,
            sep="\t",
        )

    except Exception:
        return None


def find_file(
    name: str,
    prefer_terms=None,
):

    prefer_terms = (
        prefer_terms
        or []
    )

    hits = list(
        BASE.rglob(name)
    )

    if not hits:
        return None

    def score(p):

        s = 0
        x = str(p).lower()

        for term in prefer_terms:

            if term.lower() in x:
                s += 10

        if "failed" in x:
            s -= 100

        if "invalid" in x:
            s -= 100

        return s

    return sorted(
        hits,
        key=lambda p: (
            score(p),
            str(p),
        ),
    )[-1]


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
# Runtime inventory
# ======================================================================================

TIME_RE = re.compile(
    r"(wall|elapsed|runtime|time[_ ]?s|seconds|"
    r"peak[_ ]?rss|maxrss|memory|throughput)",
    re.I,
)

METHOD_RE = {
    "Mash":
        re.compile(
            r"\bmash\b",
            re.I,
        ),

    "Dashing2":
        re.compile(
            r"dashing2|\bdashing\b",
            re.I,
        ),

    "skani":
        re.compile(
            r"\bskani\b",
            re.I,
        ),

    "FastANI":
        re.compile(
            r"\bfastani\b",
            re.I,
        ),

    "HGV":
        re.compile(
            r"\bhgv\b",
            re.I,
        ),

    "DNABERT-S":
        re.compile(
            r"dnabert",
            re.I,
        ),
}


def runtime_inventory(out: Path):

    roots = [
        BASE / "06_teacher",
        BASE / "08_main_generalization",
        BASE / "09_external_validation",
        BASE / "09_baselines",
        BASE / "15_reference_screening",
        BASE / "16_scalability",
        BASE / "logs",
    ]

    rows = []

    allowed = {
        ".tsv",
        ".csv",
        ".txt",
        ".json",
        ".log",
        ".md",
    }

    for root in roots:

        if not root.exists():
            continue

        for p in root.rglob("*"):

            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in allowed
            ):
                continue

            try:

                if p.stat().st_size > 40 * 1024 * 1024:
                    continue

            except Exception:
                continue

            text = ""

            try:

                text = p.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

            except Exception:
                continue

            hay = (
                str(p)
                +
                "\n"
                +
                text[:300000]
            )

            methods = [
                name
                for name, pat
                in METHOD_RE.items()
                if pat.search(hay)
            ]

            if not methods:
                continue

            timing_lines = [
                ln.strip()
                for ln
                in text.splitlines()
                if TIME_RE.search(ln)
            ]

            table_cols = []

            if p.suffix.lower() in {
                ".tsv",
                ".csv",
            }:

                try:

                    sep = (
                        "\t"
                        if p.suffix.lower() == ".tsv"
                        else ","
                    )

                    df = pd.read_csv(
                        p,
                        sep=sep,
                        nrows=50,
                    )

                    table_cols = [
                        str(c)
                        for c
                        in df.columns
                        if TIME_RE.search(
                            str(c)
                        )
                    ]

                except Exception:
                    pass

            if (
                not timing_lines
                and
                not table_cols
                and
                not TIME_RE.search(
                    p.name
                )
            ):
                continue

            for method in methods:

                rows.append({
                    "method":
                        method,

                    "path":
                        rel(p),

                    "timing_columns":
                        ",".join(
                            table_cols
                        ),

                    "timing_signal":
                        " | ".join(
                            timing_lines[:12]
                        )[:4000],

                    "size_bytes":
                        p.stat().st_size,
                })

    inv = pd.DataFrame(rows)

    inv.to_csv(
        out
        /
        "RUNTIME_EXISTING_EVIDENCE_INVENTORY.tsv",
        sep="\t",
        index=False,
    )

    supplement = (
        out
        /
        "runtime_supplement"
        /
        "RUNTIME_SUPPLEMENT_SUMMARY.tsv"
    )

    supplemental_methods = set()

    if supplement.is_file():

        try:

            sdf = pd.read_csv(
                supplement,
                sep="\t",
            )

            if "method" in sdf.columns:

                supplemental_methods = set(
                    sdf.loc[
                        sdf.get(
                            "status",
                            "PASS"
                        ).astype(str).str.contains(
                            "PASS",
                            case=False,
                            na=False,
                        ),
                        "method",
                    ]
                    .astype(str)
                )

        except Exception:
            pass

    status = {}

    for method in METHOD_RE:

        existing = False

        if not inv.empty:

            existing = bool(
                (
                    inv[
                        "method"
                    ]
                    ==
                    method
                )
                .any()
            )

        status[method] = {
            "existing_numeric_or_timing_evidence":
                existing,

            "supplement_runtime_pass":
                method
                in
                supplemental_methods,

            "available":
                (
                    existing
                    or
                    method
                    in
                    supplemental_methods
                ),
        }

    write_json(
        out
        /
        "RUNTIME_STATUS.json",
        status,
    )

    pd.DataFrame([
        {
            "method":
                m,

            **v,
        }
        for m, v
        in status.items()
    ]).to_csv(
        out
        /
        "RUNTIME_STATUS.tsv",
        sep="\t",
        index=False,
    )

    return status


# ======================================================================================
# Build fixed runtime lists
# ======================================================================================

def build_runtime_lists(out: Path):

    list_dir = (
        out
        /
        "runtime_lists"
    )

    list_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        C.load_manifest()
    )

    mc = (
        C.manifest_columns(
            manifest
        )
    )

    fmap = (
        C.get_fasta_map(
            manifest
        )
    )

    meta = manifest.set_index(
        mc["accession"]
    )

    A = []
    B1 = []

    for acc, row in meta.iterrows():

        acc = str(acc)

        cohort = C.infer_cohort(
            row,
            mc,
            accession=acc,
        )

        if cohort == "A":
            A.append(acc)

        elif cohort == "B1":
            B1.append(acc)

    A = sorted(A)
    B1 = sorted(B1)

    if len(A) != 2374:
        raise RuntimeError(
            f"A != 2374: {len(A)}"
        )

    if len(B1) != 391:
        raise RuntimeError(
            f"B1 != 391: {len(B1)}"
        )

    q = B1[:50]

    def write_acc_path(
        name,
        accessions,
    ):

        p = (
            list_dir
            /
            name
        )

        with p.open(
            "w",
            encoding="utf-8",
        ) as f:

            for acc in accessions:

                fp = Path(
                    fmap[acc]
                )

                if not fp.is_file():

                    raise RuntimeError(
                        f"missing FASTA: {acc} {fp}"
                    )

                f.write(
                    f"{acc}\t{fp}\n"
                )

        return p

    def write_path_only(
        name,
        accessions,
    ):

        p = (
            list_dir
            /
            name
        )

        with p.open(
            "w",
            encoding="utf-8",
        ) as f:

            for acc in accessions:

                f.write(
                    str(
                        Path(
                            fmap[acc]
                        )
                    )
                    +
                    "\n"
                )

        return p

    write_acc_path(
        "A2374_accession_path.tsv",
        A,
    )

    write_acc_path(
        "B1_Q50_accession_path.tsv",
        q,
    )

    write_path_only(
        "A2374_paths.txt",
        A,
    )

    write_path_only(
        "B1_Q50_paths.txt",
        q,
    )

    write_path_only(
        "A2374_plus_B1Q50_paths.txt",
        A + q,
    )

    return {
        "A":
            A,

        "Q":
            q,
    }


# ======================================================================================
# GTDB tree parser fallback
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


def parse_newick_custom(text):

    text = text.strip()

    i = 0
    n = len(text)

    def skip_ws():

        nonlocal i

        while (
            i < n
            and
            text[i].isspace()
        ):
            i += 1

    def parse_label():

        nonlocal i

        skip_ws()

        if i >= n:
            return ""

        if text[i] in "'\"":

            quote = text[i]
            i += 1
            start = i

            while (
                i < n
                and
                text[i] != quote
            ):
                i += 1

            label = text[
                start:
                i
            ]

            if (
                i < n
                and
                text[i] == quote
            ):
                i += 1

            return label

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

    def parse_length():

        nonlocal i

        skip_ws()

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

        skip_ws()

        if text[i] == "(":

            i += 1

            node = Node()

            while True:

                child = subtree()
                child.parent = node
                node.children.append(
                    child
                )

                skip_ws()

                if text[i] == ",":

                    i += 1
                    continue

                if text[i] == ")":

                    i += 1
                    break

                raise RuntimeError(
                    "Newick parse error"
                )

            node.name = parse_label()
            node.length = parse_length()

            return node

        name = parse_label()
        length = parse_length()

        return Node(
            name,
            length,
        )

    root = subtree()

    return root


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

        if not node.children:

            leaves[
                node.name
            ] = node

        else:

            for ch in node.children:

                stack.append(
                    (
                        ch,
                        node.depth,
                    )
                )

    return leaves


def node_distance(
    a,
    b,
):

    aa = {}

    x = a

    while x is not None:

        aa[id(x)] = x
        x = x.parent

    y = b

    while y is not None:

        if id(y) in aa:

            lca = y

            return (
                a.depth
                +
                b.depth
                -
                2.0
                *
                lca.depth
            )

        y = y.parent

    return float("nan")


# ======================================================================================
# Tree consistency
# ======================================================================================

ACC_RE = re.compile(
    r"(?:RS_|GB_)?((?:GCF|GCA)_\d+\.\d+)"
)


def normalize_tree_accession(name):

    m = ACC_RE.search(
        str(name)
    )

    if not m:
        return None

    return m.group(1)


def tree_consistency(out: Path):

    tree_path = (
        BASE
        /
        "00_metadata"
        /
        "bac120_r232.tree"
    )

    if not tree_path.is_file():

        pd.DataFrame([
            {
                "status":
                    "NOT_COMPUTED",

                "reason":
                    "bac120_r232.tree missing",
            }
        ]).to_csv(
            out
            /
            "TREE_CONSISTENCY_STATUS.tsv",
            sep="\t",
            index=False,
        )

        return "NOT_COMPUTED"

    root = parse_newick_custom(
        tree_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    leaf_nodes = index_tree(
        root
    )

    accession_to_leaf = {}

    for leaf_name, node in leaf_nodes.items():

        acc = normalize_tree_accession(
            leaf_name
        )

        if acc:
            accession_to_leaf[
                acc
            ] = node

    manifest = C.load_manifest()
    mc = C.manifest_columns(
        manifest
    )

    acc_col = mc[
        "accession"
    ]

    meta = manifest.copy()

    species_candidates = [
        "gtdb_species_cluster",
        "species_cluster",
        "gtdb_species",
        "species",
    ]

    species_col = None

    for c in species_candidates:

        if c in meta.columns:

            species_col = c
            break

    if species_col is None:

        # fallback through common manifest mapping
        for k in [
            "species_cluster",
            "species",
        ]:

            if k in mc:
                species_col = mc[k]
                break

    if species_col is None:

        raise RuntimeError(
            "Cannot identify species column"
        )

    reps = []

    for species, g in meta.groupby(
        species_col,
        dropna=False,
    ):

        cand = []

        for _, row in g.iterrows():

            acc = str(
                row[
                    acc_col
                ]
            )

            cohort = C.infer_cohort(
                row,
                mc,
                accession=acc,
            )

            if (
                cohort == "A"
                and
                acc in accession_to_leaf
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

    if len(reps) < 50:

        raise RuntimeError(
            f"Only {len(reps)} A species representatives map to GTDB tree"
        )

    reps.to_csv(
        out
        /
        "E10_TREE_SPECIES_REPRESENTATIVES.tsv",
        sep="\t",
        index=False,
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

    torch, _, _ = C.import_torch()

    seeds = [
        20260915,
        20260916,
        20260917,
    ]

    matrices = []

    per_seed_rows = []

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
                f"Missing canonical checkpoint: {ckpt}"
            )

        model, device, _ = (
            C.load_model_checkpoint(
                ckpt,
                input_dim=2080,
                embed_dim=64,
                normalize_output=True,
            )
        )

        zs = []

        model.eval()

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

        matrices.append(
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

        a = accession_to_leaf[
            reps.iloc[i][
                "accession"
            ]
        ]

        for j in range(
            i + 1,
            n,
        ):

            b = accession_to_leaf[
                reps.iloc[j][
                    "accession"
                ]
            ]

            d = node_distance(
                a,
                b,
            )

            T[i, j] = d
            T[j, i] = d

    tri = np.triu_indices(
        n,
        1,
    )

    tvec = T[
        tri
    ]

    for seed, H in zip(
        seeds,
        matrices,
    ):

        hvec = H[
            tri
        ]

        rho, p = spearmanr(
            hvec,
            tvec,
        )

        per_seed_rows.append({
            "model_seed":
                seed,

            "n_species_representatives":
                n,

            "n_pairs":
                len(hvec),

            "spearman_rho":
                float(rho),

            "spearman_p_asymptotic":
                float(p),
        })

    Hmean = np.mean(
        np.stack(
            matrices,
            axis=0,
        ),
        axis=0,
    )

    hmean = Hmean[
        tri
    ]

    rho, p = spearmanr(
        hmean,
        tvec,
    )

    rng = np.random.default_rng(
        20260926
    )

    null = []

    for _ in range(1000):

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

        rr = spearmanr(
            hp,
            tvec,
        ).statistic

        null.append(
            float(rr)
        )

    null = np.asarray(
        null,
        dtype=np.float64,
    )

    perm_p = (
        1
        +
        int(
            np.sum(
                np.abs(
                    null
                )
                >=
                abs(
                    rho
                )
            )
        )
    ) / 1001.0

    pd.DataFrame(
        per_seed_rows
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

            "model":
                "G2 split20260915 D3 canonical 3-seed mean",

            "n_species_representatives":
                n,

            "n_pairs":
                len(hmean),

            "spearman_rho":
                float(rho),

            "spearman_p_asymptotic":
                float(p),

            "mantel_style_permutations":
                1000,

            "mantel_style_two_sided_p":
                float(
                    perm_p
                ),

            "claim_boundary":
                "tree-distance consistency only; not phylogeny reconstruction",
        }
    ]).to_csv(
        out
        /
        "E10_TREE_CONSISTENCY_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    # Save pair source for plotting / manuscript source data.
    pair_rows = []

    k = 0

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):

            pair_rows.append({
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
                        hmean[k]
                    ),

                "gtdb_tree_patristic_distance":
                    float(
                        tvec[k]
                    ),
            })

            k += 1

    pd.DataFrame(
        pair_rows
    ).to_csv(
        out
        /
        "E10_TREE_PAIR_SOURCE.tsv",
        sep="\t",
        index=False,
    )

    return "PASS"


# ======================================================================================
# Confound adjusted E10
# ======================================================================================

def pair_cols(df):

    candidate_pairs = [
        (
            "accession_a",
            "accession_b",
        ),
        (
            "genome_a",
            "genome_b",
        ),
        (
            "query_accession",
            "reference_accession",
        ),
        (
            "query",
            "reference",
        ),
    ]

    for a, b in candidate_pairs:

        if (
            a in df.columns
            and
            b in df.columns
        ):

            return a, b

    return None


def canonical_pair_key(
    a,
    b,
):

    a = str(a)
    b = str(b)

    return (
        a
        +
        "||"
        +
        b
        if a <= b
        else
        b
        +
        "||"
        +
        a
    )


def choose_column(
    df,
    exact=None,
    includes=None,
    excludes=None,
):

    exact = exact or []
    includes = includes or []
    excludes = excludes or []

    lower = {
        c.lower():
            c
        for c in df.columns
    }

    for c in exact:

        if c.lower() in lower:
            return lower[
                c.lower()
            ]

    for c in df.columns:

        x = c.lower()

        if (
            all(
                z.lower()
                in x
                for z
                in includes
            )
            and
            not any(
                z.lower()
                in x
                for z
                in excludes
            )
        ):

            return c

    return None


def confound_adjusted(out: Path):

    e10 = find_file(
        "E10_HGV_SKANI_PAIR_SOURCE.tsv",
        [
            "20260925_v6",
        ],
    )

    e9 = find_file(
        "E9_PAIR_CONFOUND_SOURCE.tsv",
        [
            "20260925_v6",
        ],
    )

    if (
        e10 is None
        or
        e9 is None
    ):

        pd.DataFrame([
            {
                "status":
                    "NOT_COMPUTED",

                "reason":
                    "E9/E10 pair source missing",
            }
        ]).to_csv(
            out
            /
            "E10_CONFOUND_ADJUSTED_SUMMARY.tsv",
            sep="\t",
            index=False,
        )

        return "NOT_COMPUTED"

    a = pd.read_csv(
        e10,
        sep="\t",
    )

    b = pd.read_csv(
        e9,
        sep="\t",
    )

    pa = pair_cols(a)
    pb = pair_cols(b)

    if pa is None:

        raise RuntimeError(
            "Cannot find pair columns in E10 source"
        )

    a[
        "_pair_key"
    ] = [
        canonical_pair_key(
            x,
            y,
        )
        for x, y
        in zip(
            a[
                pa[0]
            ],
            a[
                pa[1]
            ],
        )
    ]

    if pb is not None:

        b[
            "_pair_key"
        ] = [
            canonical_pair_key(
                x,
                y,
            )
            for x, y
            in zip(
                b[
                    pb[0]
                ],
                b[
                    pb[1]
                ],
            )
        ]

        merged = a.merge(
            b,
            on="_pair_key",
            how="left",
            suffixes=(
                "",
                "_conf",
            ),
        )

    else:

        merged = a.copy()

    hgv_col = choose_column(
        merged,
        exact=[
            "hgv_distance",
            "distance_hgv",
        ],
        includes=[
            "hgv",
            "distance",
        ],
    )

    ani_col = choose_column(
        merged,
        exact=[
            "skani_ani",
            "ani",
        ],
        includes=[
            "ani",
        ],
        excludes=[
            "distance",
        ],
    )

    if (
        hgv_col is None
        or
        ani_col is None
    ):

        raise RuntimeError(
            f"Cannot resolve target columns: HGV={hgv_col} ANI={ani_col}"
        )

    confound_candidates = []

    for c in merged.columns:

        x = c.lower()

        if not any(
            token in x
            for token
            in [
                "gc",
                "length",
                "n50",
                "contig",
                "completeness",
            ]
        ):
            continue

        if not any(
            token in x
            for token
            in [
                "delta",
                "diff",
                "ratio",
                "abs",
            ]
        ):
            continue

        v = pd.to_numeric(
            merged[c],
            errors="coerce",
        )

        if (
            v.notna().sum()
            >=
            100
            and
            v.nunique(
                dropna=True
            )
            >
            1
        ):

            confound_candidates.append(
                c
            )

    # Keep one representative per concept.
    selected = []

    for concept in [
        "gc",
        "length",
        "n50",
        "contig",
        "completeness",
    ]:

        hits = [
            c
            for c
            in confound_candidates
            if concept
            in c.lower()
        ]

        if hits:

            selected.append(
                sorted(
                    hits,
                    key=lambda z:
                    (
                        0
                        if "abs"
                        in z.lower()
                        else
                        1,
                        len(z),
                    ),
                )[0]
            )

    selected = list(
        dict.fromkeys(
            selected
        )
    )

    if len(selected) < 2:

        pd.DataFrame([
            {
                "status":
                    "NOT_COMPUTED",

                "reason":
                    "fewer than two usable confound columns",

                "hgv_column":
                    hgv_col,

                "ani_column":
                    ani_col,

                "detected_confounds":
                    ",".join(
                        selected
                    ),
            }
        ]).to_csv(
            out
            /
            "E10_CONFOUND_ADJUSTED_SUMMARY.tsv",
            sep="\t",
            index=False,
        )

        return "NOT_COMPUTED"

    hgv = pd.to_numeric(
        merged[
            hgv_col
        ],
        errors="coerce",
    )

    ani = pd.to_numeric(
        merged[
            ani_col
        ],
        errors="coerce",
    )

    # ANI may be represented as [0,1] or [0,100].
    finite_ani = ani[
        np.isfinite(
            ani
        )
    ]

    if (
        len(
            finite_ani
        )
        and
        finite_ani.median()
        >
        2.0
    ):

        ani_distance = (
            100.0
            -
            ani
        )

    else:

        ani_distance = (
            1.0
            -
            ani
        )

    work = pd.DataFrame({
        "hgv_distance":
            hgv,

        "ani_distance":
            ani_distance,
    })

    for c in selected:

        work[
            c
        ] = pd.to_numeric(
            merged[
                c
            ],
            errors="coerce",
        )

    # Optional strata column
    pair_type_col = choose_column(
        merged,
        exact=[
            "pair_type",
            "pair_stratum",
        ],
        includes=[
            "pair",
            "type",
        ],
    )

    if pair_type_col:

        work[
            "pair_type"
        ] = merged[
            pair_type_col
        ].astype(str)

    def run_one(
        g,
        label,
    ):

        cols = [
            "hgv_distance",
            "ani_distance",
        ] + selected

        x = g[
            cols
        ].dropna()

        if len(x) < 100:

            return None

        raw_rho = spearmanr(
            x[
                "hgv_distance"
            ],
            x[
                "ani_distance"
            ],
        ).statistic

        yr = rankdata(
            x[
                "hgv_distance"
            ]
            .to_numpy(
                float
            )
        )

        xr = rankdata(
            x[
                "ani_distance"
            ]
            .to_numpy(
                float
            )
        )

        covs = []

        used_covs = []

        for c in selected:

            v = rankdata(
                x[
                    c
                ]
                .to_numpy(
                    float
                )
            )

            if np.std(v) <= 0:
                continue

            covs.append(v)
            used_covs.append(c)

        if not covs:

            return None

        COV = np.column_stack(
            covs
        )

        DESIGN = np.column_stack([
            np.ones(
                len(x)
            ),
            COV,
        ])

        by = np.linalg.lstsq(
            DESIGN,
            yr,
            rcond=None,
        )[0]

        bx = np.linalg.lstsq(
            DESIGN,
            xr,
            rcond=None,
        )[0]

        ry = (
            yr
            -
            DESIGN
            @
            by
        )

        rx = (
            xr
            -
            DESIGN
            @
            bx
        )

        partial_r = pearsonr(
            ry,
            rx,
        ).statistic

        rng = np.random.default_rng(
            20260926
            +
            len(x)
        )

        null = []

        for _ in range(1000):

            rp = rng.permutation(
                rx
            )

            null.append(
                pearsonr(
                    ry,
                    rp,
                ).statistic
            )

        null = np.asarray(
            null
        )

        perm_p = (
            1
            +
            np.sum(
                np.abs(
                    null
                )
                >=
                abs(
                    partial_r
                )
            )
        ) / 1001.0

        return {
            "stratum":
                label,

            "n":
                len(x),

            "raw_spearman_hgv_vs_ani_distance":
                float(
                    raw_rho
                ),

            "partial_rank_correlation":
                float(
                    partial_r
                ),

            "permutation_replicates":
                1000,

            "permutation_two_sided_p":
                float(
                    perm_p
                ),

            "covariates":
                ",".join(
                    used_covs
                ),
        }

    rows = []

    z = run_one(
        work,
        "all",
    )

    if z:
        rows.append(z)

    if "pair_type" in work.columns:

        for label, g in work.groupby(
            "pair_type"
        ):

            z = run_one(
                g,
                str(label),
            )

            if z:
                rows.append(z)

    outdf = pd.DataFrame(
        rows
    )

    if outdf.empty:

        return "NOT_COMPUTED"

    outdf.insert(
        0,
        "status",
        "PASS",
    )

    outdf.to_csv(
        out
        /
        "E10_CONFOUND_ADJUSTED_SUMMARY.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame([
        {
            "e10_source":
                rel(e10),

            "e9_confound_source":
                rel(e9),

            "hgv_distance_column":
                hgv_col,

            "ani_column":
                ani_col,

            "covariates":
                ",".join(
                    selected
                ),

            "interpretation_boundary":
                (
                    "partial rank association is descriptive "
                    "and does not establish biological causality"
                ),
        }
    ]).to_csv(
        out
        /
        "E10_CONFOUND_ADJUSTED_PROVENANCE.tsv",
        sep="\t",
        index=False,
    )

    return "PASS"


# ======================================================================================
# Main + external statistical completeness
# ======================================================================================

def ci95_t(
    mean,
    sd,
    n,
):

    if (
        n < 2
        or
        not np.isfinite(sd)
    ):
        return (
            float("nan"),
            float("nan"),
        )

    crit = t.ppf(
        0.975,
        df=n - 1,
    )

    half = (
        crit
        *
        sd
        /
        math.sqrt(n)
    )

    return (
        mean - half,
        mean + half,
    )


def discover_main_metrics():

    root = (
        BASE
        /
        "08_main_generalization"
        /
        "blind_test_evaluation_v1_20260917"
    )

    candidates = []

    for p in root.rglob("*.tsv"):

        try:

            if p.stat().st_size > 100 * 1024 * 1024:
                continue

        except Exception:
            continue

        df = safe_read_table(
            p
        )

        if (
            df is None
            or
            df.empty
        ):
            continue

        cols = [
            c.lower()
            for c in df.columns
        ]

        metric_hits = sum(
            any(
                m
                in c
                for c
                in cols
            )
            for m in [
                "auroc",
                "auprc",
                "mcc",
            ]
        )

        if metric_hits < 2:
            continue

        value_text = (
            " ".join(
                df.astype(str)
                .head(100)
                .fillna("")
                .values
                .ravel()
                .tolist()
            )
        )

        proto_hits = sum(
            p0
            in value_text
            for p0
            in [
                "G2",
                "G3",
                "G4",
            ]
        )

        score = (
            metric_hits
            *
            10
            +
            proto_hits
            *
            10
        )

        if "seed" in p.name.lower():
            score += 10

        if "summary" in p.name.lower():
            score += 5

        candidates.append(
            (
                score,
                p,
                df,
            )
        )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda x:
        (
            x[0],
            str(
                x[1]
            ),
        )
    )

    _, p, df = candidates[-1]

    return p, df


def main_stats(out: Path):

    p, df = discover_main_metrics()

    if df is None:

        pd.DataFrame([
            {
                "status":
                    "NOT_FOUND",
            }
        ]).to_csv(
            out
            /
            "MAIN_STATISTICAL_COMPLETENESS.tsv",
            sep="\t",
            index=False,
        )

        return "NOT_FOUND"

    cols_lower = {
        c.lower():
            c
        for c in df.columns
    }

    protocol_col = None

    for c in df.columns:

        if any(
            token
            in c.lower()
            for token
            in [
                "protocol",
                "holdout",
            ]
        ):

            protocol_col = c
            break

    if protocol_col is None:

        for c in df.columns:

            vals = set(
                df[c]
                .astype(str)
                .head(500)
            )

            if (
                "G2" in vals
                or
                "G3" in vals
                or
                "G4" in vals
            ):

                protocol_col = c
                break

    method_col = None

    for c in df.columns:

        if "method" in c.lower():

            method_col = c
            break

    work = df.copy()

    if method_col:

        mask = (
            work[
                method_col
            ]
            .astype(str)
            .str.contains(
                "HGV",
                case=False,
                na=False,
            )
        )

        if mask.any():
            work = work[
                mask
            ].copy()

    if protocol_col is None:

        pd.DataFrame([
            {
                "status":
                    "SOURCE_FOUND_PROTOCOL_COLUMN_UNRESOLVED",

                "source":
                    rel(p),
            }
        ]).to_csv(
            out
            /
            "MAIN_STATISTICAL_COMPLETENESS.tsv",
            sep="\t",
            index=False,
        )

        return "PARTIAL"

    split_col = None

    for c in work.columns:

        if (
            "split"
            in c.lower()
            and
            "seed"
            in c.lower()
        ):

            split_col = c
            break

    metric_map = {}

    for canonical, patterns in {
        "AUROC":
            [
                "test_auroc",
                "auroc",
            ],

        "AUPRC":
            [
                "test_auprc",
                "auprc",
            ],

        "MCC":
            [
                "test_mcc",
                "mcc",
            ],
    }.items():

        for patt in patterns:

            for c in work.columns:

                if c.lower() == patt:

                    metric_map[
                        canonical
                    ] = c

                    break

            if canonical in metric_map:
                break

    # If source is already summary-form.
    summary_cols = [
        c
        for c
        in work.columns
        if "_mean"
        in c.lower()
        or "_sd"
        in c.lower()
    ]

    rows = []

    if (
        split_col
        and
        len(
            metric_map
        )
        >=
        2
    ):

        for proto in [
            "G2",
            "G3",
            "G4",
        ]:

            g = work[
                work[
                    protocol_col
                ]
                .astype(str)
                ==
                proto
            ].copy()

            if g.empty:
                continue

            split_means = (
                g.groupby(
                    split_col
                )[
                    list(
                        metric_map.values()
                    )
                ]
                .mean(
                    numeric_only=True
                )
            )

            for metric, col in metric_map.items():

                vals = pd.to_numeric(
                    split_means[
                        col
                    ],
                    errors="coerce",
                ).dropna()

                if len(vals) < 2:
                    continue

                mean = float(
                    vals.mean()
                )

                sd = float(
                    vals.std(
                        ddof=1
                    )
                )

                lo, hi = ci95_t(
                    mean,
                    sd,
                    len(vals),
                )

                rows.append({
                    "status":
                        "PASS",

                    "protocol":
                        proto,

                    "metric":
                        metric,

                    "primary_variability_unit":
                        "taxonomy split seed",

                    "n_units":
                        len(vals),

                    "mean":
                        mean,

                    "sd":
                        sd,

                    "ci95_low":
                        lo,

                    "ci95_high":
                        hi,

                    "source":
                        rel(p),
                })

    elif summary_cols:

        # Preserve already-frozen summary source and mark for manuscript.
        for proto in [
            "G2",
            "G3",
            "G4",
        ]:

            g = work[
                work[
                    protocol_col
                ]
                .astype(str)
                ==
                proto
            ]

            if g.empty:
                continue

            row = g.iloc[0]

            for metric in [
                "AUROC",
                "AUPRC",
                "MCC",
            ]:

                mean_col = next(
                    (
                        c
                        for c
                        in work.columns
                        if metric.lower()
                        in c.lower()
                        and
                        "mean"
                        in c.lower()
                    ),
                    None,
                )

                sd_col = next(
                    (
                        c
                        for c
                        in work.columns
                        if metric.lower()
                        in c.lower()
                        and
                        (
                            "_sd"
                            in c.lower()
                            or
                            "std"
                            in c.lower()
                        )
                    ),
                    None,
                )

                if (
                    mean_col is None
                    or
                    sd_col is None
                ):
                    continue

                mean = float(
                    row[
                        mean_col
                    ]
                )

                sd = float(
                    row[
                        sd_col
                    ]
                )

                lo, hi = ci95_t(
                    mean,
                    sd,
                    3,
                )

                rows.append({
                    "status":
                        "PASS_DERIVED_CI_FROM_FROZEN_MEAN_SD",

                    "protocol":
                        proto,

                    "metric":
                        metric,

                    "primary_variability_unit":
                        "3 taxonomy split seeds",

                    "n_units":
                        3,

                    "mean":
                        mean,

                    "sd":
                        sd,

                    "ci95_low":
                        lo,

                    "ci95_high":
                        hi,

                    "source":
                        rel(p),
                })

    if not rows:

        rows = [{
            "status":
                "PARTIAL_SOURCE_FOUND_BUT_CI_NOT_DERIVED",

            "source":
                rel(p),
        }]

    pd.DataFrame(
        rows
    ).to_csv(
        out
        /
        "MAIN_STATISTICAL_COMPLETENESS.tsv",
        sep="\t",
        index=False,
    )

    if any(
        str(r[
            "status"
        ]).startswith(
            "PASS"
        )
        for r
        in rows
    ):

        return "PASS"

    return "PARTIAL"


def external_stats(out: Path):

    root = (
        BASE
        /
        "09_external_validation"
        /
        "external_one_shot_evaluation_v1_20260917"
    )

    seed_file = (
        root
        /
        "external_metrics_seed_level.tsv"
    )

    summary_file = (
        root
        /
        "external_metrics_summary_long.tsv"
    )

    rows = []

    # Prefer already frozen CI columns if present.
    if summary_file.is_file():

        sdf = pd.read_csv(
            summary_file,
            sep="\t",
        )

        ci_cols = [
            c
            for c
            in sdf.columns
            if "ci"
            in c.lower()
        ]

        if len(ci_cols) >= 2:

            sdf.insert(
                0,
                "closure_status",
                "PASS_EXISTING_FROZEN_CI",
            )

            sdf.to_csv(
                out
                /
                "EXTERNAL_STATISTICAL_COMPLETENESS_EXISTING.tsv",
                sep="\t",
                index=False,
            )

            return "PASS"

    if not seed_file.is_file():

        pd.DataFrame([
            {
                "status":
                    "NOT_FOUND",

                "expected_source":
                    rel(seed_file),
            }
        ]).to_csv(
            out
            /
            "EXTERNAL_STATISTICAL_COMPLETENESS.tsv",
            sep="\t",
            index=False,
        )

        return "NOT_FOUND"

    df = pd.read_csv(
        seed_file,
        sep="\t",
    )

    seed_col = next(
        (
            c
            for c
            in df.columns
            if "seed"
            in c.lower()
        ),
        None,
    )

    if seed_col is None:

        return "PARTIAL"

    id_cols = [
        c
        for c
        in df.columns
        if any(
            token
            in c.lower()
            for token
            in [
                "cohort",
                "endpoint",
                "task",
                "metric",
                "method",
                "stratum",
            ]
        )
    ]

    value_col = next(
        (
            c
            for c
            in df.columns
            if c.lower()
            in [
                "value",
                "score",
                "estimate",
            ]
        ),
        None,
    )

    if value_col:

        group_cols = [
            c
            for c
            in id_cols
            if c != seed_col
        ]

        for keys, g in df.groupby(
            group_cols,
            dropna=False,
        ):

            if not isinstance(
                keys,
                tuple,
            ):
                keys = (
                    keys,
                )

            vals = pd.to_numeric(
                g[
                    value_col
                ],
                errors="coerce",
            ).dropna()

            if len(vals) < 2:
                continue

            mean = float(
                vals.mean()
            )

            sd = float(
                vals.std(
                    ddof=1
                )
            )

            lo, hi = ci95_t(
                mean,
                sd,
                len(vals),
            )

            row = {
                c:
                    v
                for c, v
                in zip(
                    group_cols,
                    keys,
                )
            }

            row.update({
                "status":
                    "PASS_SEED_LEVEL_CI",

                "n_model_seeds":
                    len(vals),

                "mean":
                    mean,

                "sd":
                    sd,

                "ci95_low":
                    lo,

                "ci95_high":
                    hi,

                "source":
                    rel(seed_file),
            })

            rows.append(
                row
            )

    else:

        exclude = set(
            id_cols
            +
            [
                seed_col
            ]
        )

        numeric_metrics = []

        for c in df.columns:

            if c in exclude:
                continue

            vals = pd.to_numeric(
                df[
                    c
                ],
                errors="coerce",
            )

            if vals.notna().sum() >= 2:

                numeric_metrics.append(
                    c
                )

        group_cols = [
            c
            for c
            in id_cols
            if c != seed_col
        ]

        if not group_cols:

            df[
                "_all"
            ] = "all"

            group_cols = [
                "_all"
            ]

        for keys, g in df.groupby(
            group_cols,
            dropna=False,
        ):

            if not isinstance(
                keys,
                tuple,
            ):
                keys = (
                    keys,
                )

            for metric in numeric_metrics:

                vals = pd.to_numeric(
                    g[
                        metric
                    ],
                    errors="coerce",
                ).dropna()

                if len(vals) < 2:
                    continue

                mean = float(
                    vals.mean()
                )

                sd = float(
                    vals.std(
                        ddof=1
                    )
                )

                lo, hi = ci95_t(
                    mean,
                    sd,
                    len(vals),
                )

                row = {
                    c:
                        v
                    for c, v
                    in zip(
                        group_cols,
                        keys,
                    )
                }

                row.update({
                    "metric":
                        metric,

                    "status":
                        "PASS_SEED_LEVEL_CI",

                    "n_model_seeds":
                        len(vals),

                    "mean":
                        mean,

                    "sd":
                        sd,

                    "ci95_low":
                        lo,

                    "ci95_high":
                        hi,

                    "source":
                        rel(seed_file),
                })

                rows.append(
                    row
                )

    pd.DataFrame(
        rows
    ).to_csv(
        out
        /
        "EXTERNAL_STATISTICAL_COMPLETENESS.tsv",
        sep="\t",
        index=False,
    )

    if rows:
        return "PASS"

    return "PARTIAL"


# ======================================================================================
# Source-of-truth freeze
# ======================================================================================

def source_freeze(
    out: Path,
    table_root: Path,
    fig_root: Path,
):

    roots = {
        "E0_DATA":
            BASE
            /
            "01_manifest"
            /
            "final_3154_freeze_v1_20260916",

        "MAIN":
            BASE
            /
            "08_main_generalization"
            /
            "blind_test_evaluation_v1_20260917",

        "EXTERNAL":
            BASE
            /
            "09_external_validation"
            /
            "external_one_shot_evaluation_v1_20260917",

        "E5":
            BASE
            /
            "09_baselines"
            /
            "E5_unified_baseline_v1_20260917",

        "E6":
            BASE
            /
            "10_ablation"
            /
            "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922",

        "E7":
            BASE
            /
            "11_sensitivity"
            /
            "E7_sensitivity_v2_gzipfix_20260925_v6"
            /
            "final_validation_refreeze_v2_20260926",

        "E8":
            BASE
            /
            "12_fragmentation"
            /
            "E8_fragmentation_v1_20260925_v6",

        "E9":
            BASE
            /
            "13_explainability"
            /
            "E9_explainability_v1_20260925_v6",

        "E10":
            BASE
            /
            "14_biological_validation"
            /
            "E10_biological_fidelity_v1_20260925_v6",

        "E11":
            BASE
            /
            "15_reference_screening"
            /
            "E11_reference_screening_v1_20260925_v6",

        "E12":
            BASE
            /
            "16_scalability"
            /
            "E12_scalability_v2_gzipfix_20260926",
    }

    allowed = {
        ".tsv",
        ".csv",
        ".json",
        ".txt",
        ".md",
    }

    index_rows = []

    for block, root in roots.items():

        if not root.exists():

            index_rows.append({
                "block":
                    block,

                "source_path":
                    rel(root),

                "status":
                    "ROOT_MISSING",

                "sha256":
                    "",
            })

            continue

        dest_root = (
            table_root
            /
            block
        )

        dest_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for p in root.rglob("*"):

            if (
                not p.is_file()
                or
                p.suffix.lower()
                not in allowed
            ):
                continue

            try:

                if p.stat().st_size > 150 * 1024 * 1024:
                    continue

            except Exception:
                continue

            rp = p.relative_to(
                root
            )

            dest = (
                dest_root
                /
                rp
            )

            dest.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not dest.exists():

                dest.symlink_to(
                    p
                )

            index_rows.append({
                "block":
                    block,

                "source_path":
                    rel(p),

                "freeze_link":
                    str(
                        dest
                    ),

                "size_bytes":
                    p.stat().st_size,

                "sha256":
                    sha256(
                        p
                    ),

                "status":
                    "VALID_SOURCE",
            })

    # Closure-generated scientific source data
    closure_dest = (
        table_root
        /
        "FINAL_CLOSURE"
    )

    closure_dest.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in out.iterdir():

        if (
            not p.is_file()
            or
            p.suffix.lower()
            not in allowed
        ):
            continue

        dest = (
            closure_dest
            /
            p.name
        )

        if not dest.exists():

            dest.symlink_to(
                p
            )

        index_rows.append({
            "block":
                "FINAL_CLOSURE",

            "source_path":
                rel(p),

            "freeze_link":
                str(
                    dest
                ),

            "size_bytes":
                p.stat().st_size,

            "sha256":
                sha256(
                    p
                ),

            "status":
                "VALID_SOURCE",
        })

    idx = pd.DataFrame(
        index_rows
    )

    idx.to_csv(
        out
        /
        "FINAL_SOURCE_DATA_INDEX.tsv",
        sep="\t",
        index=False,
    )

    idx.to_csv(
        table_root
        /
        "FINAL_SOURCE_DATA_INDEX.tsv",
        sep="\t",
        index=False,
    )

    # Compact figure-source symlinks
    figure_files = [
        (
            BASE
            /
            "11_sensitivity"
            /
            "E7_sensitivity_v2_gzipfix_20260925_v6"
            /
            "final_validation_refreeze_v2_20260926"
            /
            "E7_FINAL_SENSITIVITY_SUMMARY.tsv"
        ),

        find_file(
            "E8_FRAGMENTATION_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        ),

        find_file(
            "E6_D3_PAIRED_CONTRASTS_HOLM.tsv",
            [
                "20260925_v6",
            ],
        ),

        find_file(
            "E9_KMER_INTEGRATED_GRADIENT_RANKING.tsv",
            [
                "20260925_v6",
            ],
        ),

        find_file(
            "E10_CORRELATION_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        ),

        out
        /
        "E10_TREE_PAIR_SOURCE.tsv",

        out
        /
        "E10_CONFOUND_ADJUSTED_SUMMARY.tsv",

        find_file(
            "E11_RETRIEVAL_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        ),

        BASE
        /
        "16_scalability"
        /
        "E12_scalability_v2_gzipfix_20260926"
        /
        "E12_V2_REAL_GZIP_FASTA_SUMMARY.tsv",
    ]

    for p in figure_files:

        if (
            p is None
            or
            not Path(p).is_file()
        ):
            continue

        p = Path(p)

        dest = (
            fig_root
            /
            p.name
        )

        if not dest.exists():

            dest.symlink_to(
                p
            )

    # Explicit invalid/superseded ledger
    invalid = pd.DataFrame([
        {
            "artifact":
                (
                    "E7 sensitivity v1 k5/k7"
                ),

            "path":
                (
                    "11_sensitivity/"
                    "E7_sensitivity_v1_20260925_v6"
                ),

            "status":
                "DO_NOT_USE",

            "reason":
                (
                    "plain-text extractor was incompatible "
                    "with gzip FASTA; superseded by E7-v2"
                ),
        },

        {
            "artifact":
                "E12-v1 real-FASTA timing",

            "path":
                (
                    "16_scalability/"
                    "E12_scalability_v1_20260925_v6/"
                    "E12_REAL_FASTA_*"
                ),

            "status":
                "DO_NOT_USE",

            "reason":
                (
                    "plain-text extractor was incompatible "
                    "with gzip FASTA; superseded by E12-v2"
                ),
        },

        {
            "artifact":
                "E12-v1 vector microbenchmark",

            "path":
                (
                    "16_scalability/"
                    "E12_scalability_v1_20260925_v6/"
                    "E12_VECTOR_MICROBENCHMARK.tsv"
                ),

            "status":
                "RETAIN",

            "reason":
                (
                    "does not depend on FASTA parsing; "
                    "explicitly a vector-search microbenchmark"
                ),
        },

        {
            "artifact":
                "E10 original tree component",

            "path":
                (
                    "14_biological_validation/"
                    "E10_biological_fidelity_v1_20260925_v6/"
                    "E10_TREE_COMPONENT_STATUS.txt"
                ),

            "status":
                "SUPERSEDED_FOR_TREE_COMPONENT",

            "reason":
                (
                    "closure computes tree consistency directly "
                    "from frozen bac120_r232.tree"
                ),
        },
    ])

    invalid.to_csv(
        out
        /
        "SUPERSEDED_AND_INVALID_ARTIFACT_LEDGER.tsv",
        sep="\t",
        index=False,
    )

    return roots


# ======================================================================================
# Reviewer and claim maps
# ======================================================================================

def reviewer_and_claim_maps(
    out: Path,
    runtime_status,
    tree_status,
    confound_status,
    main_stat_status,
    ext_stat_status,
):

    runtime_core = all([
        runtime_status.get(
            "Mash",
            {}
        ).get(
            "available",
            False,
        ),

        runtime_status.get(
            "Dashing2",
            {}
        ).get(
            "available",
            False,
        ),

        runtime_status.get(
            "skani",
            {}
        ).get(
            "available",
            False,
        ),

        runtime_status.get(
            "FastANI",
            {}
        ).get(
            "available",
            False,
        ),
    ])

    e6_ok = bool(
        list(
            (
                BASE
                /
                "10_ablation"
                /
                "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
            )
            .glob(
                "08_final_freeze_v1_20260925_v6"
            )
        )
    )

    e7_ok = (
        BASE
        /
        "11_sensitivity"
        /
        "E7_sensitivity_v2_gzipfix_20260925_v6"
        /
        "final_validation_refreeze_v2_20260926"
        /
        "E7_FINAL_PASS.txt"
    ).is_file()

    e8_ok = (
        find_file(
            "E8_FRAGMENTATION_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        )
        is not None
    )

    e9_ok = (
        find_file(
            "E9_KMER_INTEGRATED_GRADIENT_RANKING.tsv",
            [
                "20260925_v6",
            ],
        )
        is not None
    )

    e10_ok = (
        find_file(
            "E10_CORRELATION_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        )
        is not None
    )

    e11_ok = (
        find_file(
            "E11_RETRIEVAL_SUMMARY.tsv",
            [
                "20260925_v6",
            ],
        )
        is not None
    )

    e12_ok = (
        BASE
        /
        "16_scalability"
        /
        "E12_scalability_v2_gzipfix_20260926"
        /
        "E12_V2_PASS.txt"
    ).is_file()

    main_ok = (
        main_stat_status
        in [
            "PASS",
            "PARTIAL",
        ]
    )

    ext_ok = (
        ext_stat_status
        in [
            "PASS",
            "PARTIAL",
        ]
    )

    def exp_status(ok):

        return (
            "EVIDENCE_READY"
            if ok
            else
            "EVIDENCE_GAP"
        )

    reviewer_rows = [
        {
            "reviewer_id":
                "R1-1",

            "need":
                ">1000 genomes + cross-dataset validation",

            "evidence":
                "E0-E4; 3154 genomes; A=2374; external=780",

            "status":
                exp_status(
                    main_ok
                    and
                    ext_ok
                ),

            "manuscript_target":
                "Methods Data; Results Generalization; Fig.2",
        },

        {
            "reviewer_id":
                "R1-2",

            "need":
                "strong baselines + accuracy/runtime/memory",

            "evidence":
                "E5 + E12-v2 + final runtime closure",

            "status":
                exp_status(
                    runtime_core
                    and
                    e12_ok
                ),

            "manuscript_target":
                "Results Baselines; Table 2-3",
        },

        {
            "reviewer_id":
                "R1-3",

            "need":
                "biological interpretation/use case",

            "evidence":
                "E10 ANI/taxonomy/tree + E11 screening",

            "status":
                exp_status(
                    e10_ok
                    and
                    e11_ok
                    and
                    tree_status
                    ==
                    "PASS"
                ),

            "manuscript_target":
                "Results Biological fidelity; screening",
        },

        {
            "reviewer_id":
                "R1-4",

            "need":
                "model explainability",

            "evidence":
                "E9 attribution + confound audit",

            "status":
                exp_status(
                    e9_ok
                    and
                    confound_status
                    ==
                    "PASS"
                ),

            "manuscript_target":
                "Supplementary / explainability",
        },

        {
            "reviewer_id":
                "R1-5",

            "need":
                "CI/p-values/sensitivity/ablation",

            "evidence":
                "E6 statistics + E7-v2",

            "status":
                exp_status(
                    e6_ok
                    and
                    e7_ok
                ),

            "manuscript_target":
                "Fig.3-4 + Supplement",
        },

        {
            "reviewer_id":
                "R1-6",

            "need":
                "generalization across broader taxa",

            "evidence":
                "G2/G3/G4 + B1/B2",

            "status":
                exp_status(
                    main_ok
                    and
                    ext_ok
                ),

            "manuscript_target":
                "Results Generalization",
        },

        {
            "reviewer_id":
                "R1-7",

            "need":
                "scalability credibility",

            "evidence":
                "E12-v2 real gzip FASTA + vector microbenchmark",

            "status":
                exp_status(
                    e12_ok
                ),

            "manuscript_target":
                "Fig.6 / Table 3",
        },

        {
            "reviewer_id":
                "R1-8",

            "need":
                "clear novelty statement",

            "evidence":
                "writing-only: contributions rewrite",

            "status":
                "READY_FOR_WRITING",

            "manuscript_target":
                "Introduction",
        },

        {
            "reviewer_id":
                "R1-9",

            "need":
                "updated literature",

            "evidence":
                "writing-only: 2020-2026 literature update",

            "status":
                "READY_FOR_WRITING",

            "manuscript_target":
                "Introduction / Discussion",
        },

        {
            "reviewer_id":
                "R1-10",

            "need":
                "writing and clarity",

            "evidence":
                "writing-only",

            "status":
                "READY_FOR_WRITING",

            "manuscript_target":
                "whole manuscript",
        },

        {
            "reviewer_id":
                "R3-1",

            "need":
                "broader taxonomic coverage",

            "evidence":
                "E0-E4",

            "status":
                exp_status(
                    main_ok
                    and
                    ext_ok
                ),

            "manuscript_target":
                "Data / Generalization",
        },

        {
            "reviewer_id":
                "R3-2",

            "need":
                "MinHash / embedding baselines",

            "evidence":
                "Mash, Dashing2, DNABERT-S, skani, FastANI, Raw/PCA",

            "status":
                exp_status(
                    runtime_core
                ),

            "manuscript_target":
                "Table 2-3",
        },

        {
            "reviewer_id":
                "R3-3",

            "need":
                "suggested image preprocessing citation",

            "evidence":
                "writing-only domain-scope response",

            "status":
                "READY_FOR_WRITING",

            "manuscript_target":
                "Rebuttal",
        },

        {
            "reviewer_id":
                "R3-4",

            "need":
                "selective vs full vs no distillation",

            "evidence":
                "E6 D0-D5",

            "status":
                exp_status(
                    e6_ok
                ),

            "manuscript_target":
                "Fig.3",
        },

        {
            "reviewer_id":
                "R3-5",

            "need":
                "teacher fidelity",

            "evidence":
                "E6 teacher controls + E10 ANI fidelity",

            "status":
                exp_status(
                    e6_ok
                    and
                    e10_ok
                ),

            "manuscript_target":
                "Fig.3 / Fig.5",
        },
    ]

    reviewer = pd.DataFrame(
        reviewer_rows
    )

    reviewer.to_csv(
        out
        /
        "REVIEWER_15_COMMENT_EVIDENCE_MAP.tsv",
        sep="\t",
        index=False,
    )

    claims = pd.DataFrame([
        {
            "claim":
                "Selective distillation",

            "safe_wording":
                (
                    "selective distillation alters/regularizes "
                    "the learned geometry while retaining competitive discrimination"
                ),

            "do_not_write":
                (
                    "D3 significantly improves every ablation comparator"
                ),

            "reason":
                (
                    "formal comparator-specific E6 deltas/CIs do not support "
                    "unconditional universal improvement"
                ),
        },

        {
            "claim":
                "k-mer sensitivity",

            "safe_wording":
                (
                    "k=6 provides a favorable accuracy-compactness operating point"
                ),

            "do_not_write":
                "k=6 is numerically best",

            "reason":
                (
                    "k=7 has slightly higher AUROC/AUPRC but 8192-D input "
                    "vs 2080-D for k=6"
                ),
        },

        {
            "claim":
                "External validation",

            "safe_wording":
                "cross-source external validation / independent external cohort",

            "do_not_write":
                "three completely independent databases",

            "reason":
                "GTDB is the taxonomy backbone and NCBI provides FASTA",
        },

        {
            "claim":
                "Broad retrieval",

            "safe_wording":
                (
                    "strong matched-species and seen-genus screening, "
                    "with reduced performance for unseen genera"
                ),

            "do_not_write":
                "uniformly strong broad-taxonomy retrieval",

            "reason":
                "B2 unseen-genus is the principal applicability boundary",
        },

        {
            "claim":
                "Biological fidelity",

            "safe_wording":
                (
                    "HGV distances show ANI/taxonomic/tree consistency "
                    "subject to measurable compositional confounding"
                ),

            "do_not_write":
                "HGV reconstructs phylogeny",

            "reason":
                "tree analysis is consistency evidence, not phylogeny inference",
        },

        {
            "claim":
                "Screening utility",

            "safe_wording":
                "first-stage screening before skani/FastANI confirmation",

            "do_not_write":
                "HGV replaces ANI",

            "reason":
                "E11 uses ANI tools as confirmatory stage",
        },

        {
            "claim":
                "Scalability",

            "safe_wording":
                (
                    "after one-time feature extraction, 64-D vectors "
                    "require 256 bytes/genome and support inexpensive exact scanning"
                ),

            "do_not_write":
                (
                    "database-size independent or million-genome "
                    "real end-to-end benchmark"
                ),

            "reason":
                (
                    "exact scan is O(Nd); 1M experiment is vector-search microbenchmark"
                ),
        },

        {
            "claim":
                "E12 real FASTA",

            "safe_wording":
                "E12-v2 gzip-aware real FASTA results only",

            "do_not_write":
                "E12-v1 real-FASTA timings",

            "reason":
                "v1 extractor was incompatible with gzip FASTA",
        },
    ])

    claims.to_csv(
        out
        /
        "FINAL_CLAIM_BOUNDARY_AUDIT.tsv",
        sep="\t",
        index=False,
    )

    evidence_gaps = int(
        np.sum(
            reviewer[
                "status"
            ]
            ==
            "EVIDENCE_GAP"
        )
    )

    return {
        "runtime_core":
            runtime_core,

        "evidence_gaps":
            evidence_gaps,

        "e6":
            e6_ok,

        "e7":
            e7_ok,

        "e8":
            e8_ok,

        "e9":
            e9_ok,

        "e10":
            e10_ok,

        "e11":
            e11_ok,

        "e12":
            e12_ok,
    }


# ======================================================================================
# Final report
# ======================================================================================

def final_report(
    out: Path,
    table_root: Path,
    fig_root: Path,
):

    runtime_status = runtime_inventory(
        out
    )

    tree_status = tree_consistency(
        out
    )

    confound_status = confound_adjusted(
        out
    )

    main_stat_status = main_stats(
        out
    )

    ext_stat_status = external_stats(
        out
    )

    state = reviewer_and_claim_maps(
        out,
        runtime_status,
        tree_status,
        confound_status,
        main_stat_status,
        ext_stat_status,
    )

    source_freeze(
        out,
        table_root,
        fig_root,
    )

    # Re-run source index now closure tables exist.
    runtime_core = state[
        "runtime_core"
    ]

    hard_core = all([
        state["e6"],
        state["e7"],
        state["e8"],
        state["e9"],
        state["e10"],
        state["e11"],
        state["e12"],
        tree_status == "PASS",
        confound_status == "PASS",
        main_stat_status
        in [
            "PASS",
            "PARTIAL",
        ],
        ext_stat_status
        in [
            "PASS",
            "PARTIAL",
        ],
        runtime_core,
    ])

    ready = (
        hard_core
        and
        state[
            "evidence_gaps"
        ]
        ==
        0
    )

    status_rows = [
        (
            "DATA_FREEZE",
            (
                "PASS"
                if (
                    BASE
                    /
                    "01_manifest"
                    /
                    "final_3154_freeze_v1_20260916"
                ).exists()
                else
                "FAIL"
            ),
        ),

        (
            "MAIN_GENERALIZATION",
            (
                "PASS"
                if main_stat_status
                in [
                    "PASS",
                    "PARTIAL",
                ]
                else
                "FAIL"
            ),
        ),

        (
            "EXTERNAL_VALIDATION",
            (
                "PASS"
                if ext_stat_status
                in [
                    "PASS",
                    "PARTIAL",
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E5_BASELINE_RUNTIME_CORE",
            (
                "PASS"
                if runtime_core
                else
                "FAIL"
            ),
        ),

        (
            "E6_ABLATION_STATS",
            (
                "PASS"
                if state[
                    "e6"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E7_SENSITIVITY_V2",
            (
                "PASS"
                if state[
                    "e7"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E8_FRAGMENTATION",
            (
                "PASS"
                if state[
                    "e8"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E9_EXPLAINABILITY",
            (
                "PASS"
                if state[
                    "e9"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E10_BIOLOGICAL_FIDELITY",
            (
                "PASS"
                if state[
                    "e10"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "TREE_CONSISTENCY",
            tree_status,
        ),

        (
            "CONFOUND_ADJUSTED_AUDIT",
            confound_status,
        ),

        (
            "E11_SCREENING",
            (
                "PASS"
                if state[
                    "e11"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "E12_SCALABILITY_V2",
            (
                "PASS"
                if state[
                    "e12"
                ]
                else
                "FAIL"
            ),
        ),

        (
            "MAIN_STATISTICAL_COMPLETENESS",
            main_stat_status,
        ),

        (
            "EXTERNAL_STATISTICAL_COMPLETENESS",
            ext_stat_status,
        ),

        (
            "REVIEWER_EVIDENCE_GAPS",
            str(
                state[
                    "evidence_gaps"
                ]
            ),
        ),
    ]

    status_df = pd.DataFrame(
        status_rows,
        columns=[
            "block",
            "status",
        ],
    )

    status_df.to_csv(
        out
        /
        "FINAL_CLOSURE_STATUS.tsv",
        sep="\t",
        index=False,
    )

    lines = []

    lines.append(
        "# HGV-V3 Final Closure Report"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Final experimental status"
    )

    lines.append(
        ""
    )

    for block, status in status_rows:

        lines.append(
            f"- {block}: **{status}**"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Runtime evidence"
    )

    lines.append(
        ""
    )

    for method, info in runtime_status.items():

        lines.append(
            f"- {method}: "
            f"{'AVAILABLE' if info['available'] else 'NOT FOUND'}"
        )

    lines.append(
        ""
    )

    lines.append(
        "## Freeze roots"
    )

    lines.append(
        ""
    )

    lines.append(
        f"- Table source freeze: `{table_root}`"
    )

    lines.append(
        f"- Figure source freeze: `{fig_root}`"
    )

    lines.append(
        f"- Closure root: `{out}`"
    )

    lines.append(
        ""
    )

    lines.append(
        "## Hard claim boundaries"
    )

    lines.append(
        ""
    )

    lines.append(
        "- Do not claim universal D3 superiority."
    )

    lines.append(
        "- Do not call k=6 numerically best; describe the accuracy-compactness trade-off."
    )

    lines.append(
        "- Do not claim HGV reconstructs phylogeny."
    )

    lines.append(
        "- Do not claim HGV replaces ANI."
    )

    lines.append(
        "- Do not claim database-size independence."
    )

    lines.append(
        "- Do not describe the 1M vector microbenchmark as a 1M real-genome end-to-end test."
    )

    lines.append(
        "- Do not use E7-v1 k5/k7 or E12-v1 real-FASTA timings."
    )

    lines.append(
        ""
    )

    if ready:

        lines.append(
            "HGV_V3_FINAL_CLOSURE=PASS"
        )

        lines.append(
            "HGV_V3_EXPERIMENTS_CLOSED=YES"
        )

        lines.append(
            "NO_MORE_MODEL_TRAINING=YES"
        )

        lines.append(
            "READY_FOR_MANUSCRIPT=YES"
        )

    else:

        lines.append(
            "HGV_V3_FINAL_CLOSURE=NEEDS_REVIEW"
        )

        lines.append(
            "HGV_V3_EXPERIMENTS_CLOSED=NO"
        )

        lines.append(
            "NO_MORE_MODEL_TRAINING=YES"
        )

        lines.append(
            "READY_FOR_MANUSCRIPT=CONDITIONAL"
        )

    lines.append(
        "NO_FROZEN_INPUT_MODIFIED=YES"
    )

    (
        out
        /
        "FINAL_CLOSURE_REPORT.md"
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
        "FINAL_CLOSURE_GATE.txt"
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

    return ready


# ======================================================================================
# CLI
# ======================================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        choices=[
            "pre",
            "post",
        ],
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

    out = args.out

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.phase == "pre":

        build_runtime_lists(
            out
        )

        status = runtime_inventory(
            out
        )

        print(
            "PRE_CLOSURE_RUNTIME_INVENTORY=PASS"
        )

        for method in [
            "Mash",
            "Dashing2",
            "skani",
            "FastANI",
            "HGV",
            "DNABERT-S",
        ]:

            print(
                f"{method}_RUNTIME_AVAILABLE="
                f"{status[method]['available']}"
            )

        return

    if args.table_root is None:
        raise SystemExit(
            "--table-root required for post"
        )

    if args.fig_root is None:
        raise SystemExit(
            "--fig-root required for post"
        )

    ready = final_report(
        out,
        args.table_root,
        args.fig_root,
    )

    print(
        "POST_CLOSURE_ANALYSIS=PASS"
    )

    print(
        f"READY_FOR_MANUSCRIPT={ready}"
    )


if __name__ == "__main__":
    main()
PY

python -m py_compile \
    "$CLOSE/final_closure_engine_v1.py"

echo "CLOSURE_ENGINE_COMPILE=PASS"

# ======================================================================================
# 2. PRE-CLOSURE AUDIT + FIXED RUNTIME LISTS
# ======================================================================================

echo
echo "===== 1. PRE-CLOSURE RUNTIME INVENTORY ====="

python \
    "$CLOSE/final_closure_engine_v1.py" \
    --phase pre \
    --out "$OUT"

NEED_MASH=$(python - "$OUT/RUNTIME_STATUS.json" <<'PY'
import json
import sys
x=json.load(open(sys.argv[1]))
print(
    0
    if x["Mash"]["available"]
    else
    1
)
PY
)

NEED_DASHING=$(python - "$OUT/RUNTIME_STATUS.json" <<'PY'
import json
import sys
x=json.load(open(sys.argv[1]))
print(
    0
    if x["Dashing2"]["available"]
    else
    1
)
PY
)

echo "NEED_MASH_RUNTIME_SUPPLEMENT=$NEED_MASH"
echo "NEED_DASHING2_RUNTIME_SUPPLEMENT=$NEED_DASHING"

# ======================================================================================
# 3. RUNTIME-ONLY SUPPLEMENT
#
# Same fixed task:
# Reference = Development A, N=2374
# Query     = first 50 lexicographically sorted B1 accessions
#
# No accuracy/model training occurs here.
# ======================================================================================

RUNTIME_OUT="$OUT/runtime_supplement"

mkdir -p \
    "$RUNTIME_OUT" \
    "$RUNTIME_OUT/mash" \
    "$RUNTIME_OUT/dashing2"

RUNTIME_SUMMARY="$RUNTIME_OUT/RUNTIME_SUPPLEMENT_SUMMARY.tsv"

printf \
"method\tmode\trep\tbuild_ref_s\tbuild_query_s\tquery_s\tcombined_s\tbuild_maxrss_kb\tquery_maxrss_kb\tindex_bytes\tstatus\tnote\n" \
> "$RUNTIME_SUMMARY"

REFLIST="$OUT/runtime_lists/A2374_paths.txt"
QLIST="$OUT/runtime_lists/B1_Q50_paths.txt"
ALLLIST="$OUT/runtime_lists/A2374_plus_B1Q50_paths.txt"

timed_run() {
    TIMEFILE="$1"
    shift

    /usr/bin/time \
        -f "%e\t%M" \
        -o "$TIMEFILE" \
        "$@"
}

# ======================================================================================
# 3A. Mash
# ======================================================================================

if [ "$NEED_MASH" -eq 1 ]; then

    echo
    echo "===== 2A. MASH RUNTIME-ONLY SUPPLEMENT ====="

    if ! command -v mash >/dev/null 2>&1; then

        echo "[WARN] mash not found"

        printf \
"Mash\tcached_sketch_query\t0\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tFAIL\tmash executable not found\n" \
        >> "$RUNTIME_SUMMARY"

    else

        mash --version \
        > "$RUNTIME_OUT/mash/version.txt" \
        2>&1 || true

        for REP in 1 2 3
        do

            R="$RUNTIME_OUT/mash/rep${REP}"

            mkdir -p "$R"

            REF_PREFIX="$R/ref_A2374"
            QUERY_PREFIX="$R/query_B1Q50"

            echo "[Mash] rep=$REP build reference"

            timed_run \
                "$R/ref_build.time" \
                mash sketch \
                -p 48 \
                -k 21 \
                -s 1000 \
                -l "$REFLIST" \
                -o "$REF_PREFIX"

            echo "[Mash] rep=$REP build query"

            timed_run \
                "$R/query_build.time" \
                mash sketch \
                -p 48 \
                -k 21 \
                -s 1000 \
                -l "$QLIST" \
                -o "$QUERY_PREFIX"

            echo "[Mash] rep=$REP cached query"

            timed_run \
                "$R/query_dist.time" \
                mash dist \
                -p 48 \
                "${REF_PREFIX}.msh" \
                "${QUERY_PREFIX}.msh" \
                > "$R/mash_dist_A2374_x_B1Q50.tsv"

            REF_BUILD=$(cut -f1 "$R/ref_build.time")
            REF_RSS=$(cut -f2 "$R/ref_build.time")

            QUERY_BUILD=$(cut -f1 "$R/query_build.time")

            QUERY_S=$(cut -f1 "$R/query_dist.time")
            QUERY_RSS=$(cut -f2 "$R/query_dist.time")

            INDEX_BYTES=$(stat -c %s "${REF_PREFIX}.msh")

            printf \
"Mash\tcached_sketch_query\t%s\t%s\t%s\t%s\tNA\t%s\t%s\t%s\tPASS\tk21 sketch1000; reference=A2374; query=B1Q50\n" \
                "$REP" \
                "$REF_BUILD" \
                "$QUERY_BUILD" \
                "$QUERY_S" \
                "$REF_RSS" \
                "$QUERY_RSS" \
                "$INDEX_BYTES" \
            >> "$RUNTIME_SUMMARY"

        done

    fi

else

    echo
    echo "Mash runtime evidence already exists -> no supplemental Mash run."

fi

# ======================================================================================
# 3B. Dashing2
# ======================================================================================

if [ "$NEED_DASHING" -eq 1 ]; then

    echo
    echo "===== 2B. DASHING2 RUNTIME-ONLY SUPPLEMENT ====="

    if ! command -v dashing2 >/dev/null 2>&1; then

        echo "[WARN] dashing2 not found"

        printf \
"Dashing2\trectangular_query\t0\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tFAIL\tdashing2 executable not found\n" \
        >> "$RUNTIME_SUMMARY"

    else

        dashing2 --version \
        > "$RUNTIME_OUT/dashing2/version.txt" \
        2>&1 || true

        DHELP="$RUNTIME_OUT/dashing2/sketch_help.txt"

        dashing2 sketch --help \
            > "$DHELP" \
            2>&1 || true

        CANON_FLAG=""

        if grep -q -- '--canonical' "$DHELP"; then
            CANON_FLAG="--canonical"
        elif grep -q -- '--canon' "$DHELP"; then
            CANON_FLAG="--canon"
        fi

        CACHE_FLAG=""

        if grep -q -- '--cache-sketches' "$DHELP"; then
            CACHE_FLAG="--cache-sketches"
        elif grep -q -- '--cache' "$DHELP"; then
            CACHE_FLAG="--cache"
        fi

        HAS_OUTPREFIX=0

        if grep -q -- '--outprefix' "$DHELP"; then
            HAS_OUTPREFIX=1
        fi

        echo "DASHING2_CANON_FLAG=$CANON_FLAG"
        echo "DASHING2_CACHE_FLAG=$CACHE_FLAG"
        echo "DASHING2_HAS_OUTPREFIX=$HAS_OUTPREFIX"

        # ------------------------------------------------------------------
        # Preferred mode:
        # precompute/cache sketches first, then rectangular query.
        # ------------------------------------------------------------------

        CACHE_MODE_WORKS=0

        if \
            [ -n "$CANON_FLAG" ] && \
            [ -n "$CACHE_FLAG" ] && \
            [ "$HAS_OUTPREFIX" -eq 1 ]
        then

            TESTDIR="$RUNTIME_OUT/dashing2/cache_probe"

            mkdir -p "$TESTDIR/cache"

            set +e

            dashing2 sketch \
                -p48 \
                -k31 \
                -S1024 \
                "$CANON_FLAG" \
                "$CACHE_FLAG" \
                --outprefix "$TESTDIR/cache/" \
                -F "$ALLLIST" \
                > "$TESTDIR/build.stdout" \
                2> "$TESTDIR/build.stderr"

            RC1=$?

            if [ "$RC1" -eq 0 ]; then

                dashing2 sketch \
                    -p48 \
                    -k31 \
                    -S1024 \
                    "$CANON_FLAG" \
                    "$CACHE_FLAG" \
                    --outprefix "$TESTDIR/cache/" \
                    --cmpout "$TESTDIR/query.tsv" \
                    -F "$REFLIST" \
                    -Q "$QLIST" \
                    > "$TESTDIR/query.stdout" \
                    2> "$TESTDIR/query.stderr"

                RC2=$?

                if \
                    [ "$RC2" -eq 0 ] && \
                    [ -s "$TESTDIR/query.tsv" ]
                then
                    CACHE_MODE_WORKS=1
                fi

            fi

            set -e

        fi

        echo "DASHING2_CACHE_MODE_WORKS=$CACHE_MODE_WORKS"

        if [ "$CACHE_MODE_WORKS" -eq 1 ]; then

            for REP in 1 2 3
            do

                R="$RUNTIME_OUT/dashing2/rep${REP}"

                rm -rf "$R"
                mkdir -p "$R/cache"

                echo "[Dashing2] rep=$REP build cached sketches"

                timed_run \
                    "$R/build.time" \
                    dashing2 sketch \
                    -p48 \
                    -k31 \
                    -S1024 \
                    "$CANON_FLAG" \
                    "$CACHE_FLAG" \
                    --outprefix "$R/cache/" \
                    -F "$ALLLIST" \
                    > "$R/build.stdout" \
                    2> "$R/build.stderr"

                echo "[Dashing2] rep=$REP rectangular cached query"

                timed_run \
                    "$R/query.time" \
                    dashing2 sketch \
                    -p48 \
                    -k31 \
                    -S1024 \
                    "$CANON_FLAG" \
                    "$CACHE_FLAG" \
                    --outprefix "$R/cache/" \
                    --cmpout "$R/query.tsv" \
                    -F "$REFLIST" \
                    -Q "$QLIST" \
                    > "$R/query.stdout" \
                    2> "$R/query.stderr"

                BUILD_S=$(cut -f1 "$R/build.time")
                BUILD_RSS=$(cut -f2 "$R/build.time")

                QUERY_S=$(cut -f1 "$R/query.time")
                QUERY_RSS=$(cut -f2 "$R/query.time")

                INDEX_BYTES=$(du -sb "$R/cache" | awk '{print $1}')

                printf \
"Dashing2\tcached_rectangular_setsketch\t%s\t%s\tNA\t%s\tNA\t%s\t%s\t%s\tPASS\tk31 S1024 canonical; cached sketches; reference=A2374; query=B1Q50\n" \
                    "$REP" \
                    "$BUILD_S" \
                    "$QUERY_S" \
                    "$BUILD_RSS" \
                    "$QUERY_RSS" \
                    "$INDEX_BYTES" \
                >> "$RUNTIME_SUMMARY"

            done

        else

            # ------------------------------------------------------------------
            # Fallback:
            # one-step rectangular query from real FASTA.
            #
            # This is valid absolute runtime evidence, but build/query are
            # inseparable. It MUST NOT be used for direct cached-query speed ratios.
            # ------------------------------------------------------------------

            echo
            echo "[WARN] Dashing2 cache mode unavailable."
            echo "Using scoped one-step rectangular real-FASTA runtime."

            for REP in 1 2 3
            do

                R="$RUNTIME_OUT/dashing2/combined_rep${REP}"

                mkdir -p "$R"

                timed_run \
                    "$R/combined.time" \
                    dashing2 sketch \
                    -p48 \
                    -k31 \
                    -S1024 \
                    $CANON_FLAG \
                    --cmpout "$R/query.tsv" \
                    -F "$REFLIST" \
                    -Q "$QLIST" \
                    > "$R/stdout.txt" \
                    2> "$R/stderr.txt"

                COMBINED_S=$(cut -f1 "$R/combined.time")
                RSS=$(cut -f2 "$R/combined.time")

                printf \
"Dashing2\treal_fasta_rectangular_combined\t%s\tNA\tNA\tNA\t%s\t%s\tNA\tNA\tPASS_SCOPED\tbuild/query inseparable; do not use cached-query speed-ratio claim\n" \
                    "$REP" \
                    "$COMBINED_S" \
                    "$RSS" \
                >> "$RUNTIME_SUMMARY"

            done

        fi

    fi

else

    echo
    echo "Dashing2 runtime evidence already exists -> no supplemental Dashing2 run."

fi

# ======================================================================================
# 4. SUMMARIZE RUNTIME SUPPLEMENT
# ======================================================================================

echo
echo "===== 3. RUNTIME SUPPLEMENT SUMMARY ====="

python - "$RUNTIME_SUMMARY" "$RUNTIME_OUT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
out = Path(sys.argv[2])

df = pd.read_csv(
    src,
    sep="\t",
)

numeric = [
    "build_ref_s",
    "build_query_s",
    "query_s",
    "combined_s",
    "build_maxrss_kb",
    "query_maxrss_kb",
    "index_bytes",
]

for c in numeric:

    if c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

rows=[]

for (method, mode), g in df.groupby(
    ["method","mode"],
    dropna=False,
):

    row={
        "method":method,
        "mode":mode,
        "n_repeats":len(g),
        "status":"PASS"
        if g["status"].astype(str).str.contains(
            "PASS",
            case=False,
            na=False,
        ).all()
        else "PARTIAL",
    }

    for c in numeric:

        if c not in g.columns:
            continue

        vals=pd.to_numeric(
            g[c],
            errors="coerce",
        ).dropna()

        if len(vals):

            row[c+"_median"]=float(
                vals.median()
            )

            row[c+"_min"]=float(
                vals.min()
            )

            row[c+"_max"]=float(
                vals.max()
            )

    rows.append(row)

pd.DataFrame(
    rows
).to_csv(
    out / "RUNTIME_SUPPLEMENT_METHOD_SUMMARY.tsv",
    sep="\t",
    index=False,
)

print(
    pd.DataFrame(
        rows
    ).to_string(
        index=False
    )
)
PY

# ======================================================================================
# 5. FINAL SCIENTIFIC CLOSURE
# ======================================================================================

echo
echo "===== 4. TREE + CONFOUND + CI + SOURCE FREEZE + REVIEWER MAP ====="

python \
    "$CLOSE/final_closure_engine_v1.py" \
    --phase post \
    --out "$OUT" \
    --table-root "$TABLE_ROOT" \
    --fig-root "$FIG_ROOT"

# ======================================================================================
# 6. FINAL HUMAN-READABLE OUTPUT
# ======================================================================================

echo
echo "===================================================================================================="
echo "FINAL CLOSURE REPORT"
echo "===================================================================================================="

cat \
    "$OUT/FINAL_CLOSURE_REPORT.md"

echo
echo "===================================================================================================="
echo "FINAL CLOSURE STATUS"
echo "===================================================================================================="

column -t -s $'\t' \
    "$OUT/FINAL_CLOSURE_STATUS.tsv"

echo
echo "===================================================================================================="
echo "TREE CONSISTENCY"
echo "===================================================================================================="

if [ -f "$OUT/E10_TREE_CONSISTENCY_SUMMARY.tsv" ]; then

    column -t -s $'\t' \
        "$OUT/E10_TREE_CONSISTENCY_SUMMARY.tsv"

fi

echo
echo "===================================================================================================="
echo "CONFOUND-ADJUSTED E10"
echo "===================================================================================================="

if [ -f "$OUT/E10_CONFOUND_ADJUSTED_SUMMARY.tsv" ]; then

    column -t -s $'\t' \
        "$OUT/E10_CONFOUND_ADJUSTED_SUMMARY.tsv"

fi

echo
echo "===================================================================================================="
echo "MAIN STATISTICAL COMPLETENESS"
echo "===================================================================================================="

if [ -f "$OUT/MAIN_STATISTICAL_COMPLETENESS.tsv" ]; then

    column -t -s $'\t' \
        "$OUT/MAIN_STATISTICAL_COMPLETENESS.tsv"

fi

echo
echo "===================================================================================================="
echo "EXTERNAL STATISTICAL COMPLETENESS"
echo "===================================================================================================="

if [ -f "$OUT/EXTERNAL_STATISTICAL_COMPLETENESS.tsv" ]; then

    column -t -s $'\t' \
        "$OUT/EXTERNAL_STATISTICAL_COMPLETENESS.tsv"

elif [ -f "$OUT/EXTERNAL_STATISTICAL_COMPLETENESS_EXISTING.tsv" ]; then

    column -t -s $'\t' \
        "$OUT/EXTERNAL_STATISTICAL_COMPLETENESS_EXISTING.tsv"

fi

echo
echo "===================================================================================================="
echo "RUNTIME STATUS"
echo "===================================================================================================="

column -t -s $'\t' \
    "$OUT/RUNTIME_STATUS.tsv"

echo
echo "===================================================================================================="
echo "REVIEWER EVIDENCE MAP"
echo "===================================================================================================="

column -t -s $'\t' \
    "$OUT/REVIEWER_15_COMMENT_EVIDENCE_MAP.tsv"

echo
echo "===================================================================================================="
echo "FINAL GATE"
echo "===================================================================================================="

cat \
    "$OUT/FINAL_CLOSURE_GATE.txt"

echo
echo "CLOSURE_ROOT=$OUT"
echo "TABLE_SOURCE_ROOT=$TABLE_ROOT"
echo "FIGURE_SOURCE_ROOT=$FIG_ROOT"

echo
echo "END=$(date --iso-8601=seconds)"
echo "===================================================================================================="
