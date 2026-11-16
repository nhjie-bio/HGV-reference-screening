#!/usr/bin/env bash

set -u
set -o pipefail

BASE="${HGV_BASE:-HGV_V3_revision_20260915}"

FREEZE="$BASE/09_baselines/E5_external_unified_v1_20260920/00_task_freeze"

E5="$BASE/09_baselines/E5_unified_baseline_v1_20260917"

DNABERT="$E5/06_dnaberts"

TEACHER="$BASE/06_teacher/teacher_cache_v1_20260916"

OUTROOT="$BASE/09_baselines/E5_external_unified_v1_20260920/01_execution_preflight"

STAMP="$(date +%Y%m%d_%H%M%S)"

OUT="$OUTROOT/run_$STAMP"

mkdir -p "$OUT"

REPORT="$OUT/E5_EXTERNAL_EXECUTION_PREFLIGHT.txt"


exec > >(tee "$REPORT") 2>&1


echo "===================================================================================================="
echo "HGV-V3 E5 EXTERNAL NATIVE BASELINE EXECUTION PREFLIGHT v1"
echo "===================================================================================================="
echo
echo "DATE      : $(date -Is)"
echo "BASE      : $BASE"
echo "FREEZE    : $FREEZE"
echo "E5        : $E5"
echo "DNABERT   : $DNABERT"
echo "OUT       : $OUT"
echo


# --------------------------------------------------------------------------------------------------
# 1. Frozen task gate
# --------------------------------------------------------------------------------------------------

echo "===================================================================================================="
echo "1. FROZEN EXTERNAL TASK"
echo "===================================================================================================="

REQUIRED_FREEZE_FILES=(
    "$FREEZE/REFERENCE_A_2374.tsv"
    "$FREEZE/REFERENCE_A_2374_FASTA.list"
    "$FREEZE/QUERY_B1_391.tsv"
    "$FREEZE/QUERY_B1_391_FASTA.list"
    "$FREEZE/QUERY_B2_SEEN_196.tsv"
    "$FREEZE/QUERY_B2_SEEN_196_FASTA.list"
    "$FREEZE/QUERY_B2_UNSEEN_193.tsv"
    "$FREEZE/QUERY_B2_UNSEEN_193_FASTA.list"
    "$FREEZE/EXTERNAL_TASK_FREEZE.txt"
    "$FREEZE/EXTERNAL_TASK_SHA256.txt"
)

FREEZE_FAIL=0

for f in "${REQUIRED_FREEZE_FILES[@]}"
do
    if [[ -s "$f" ]]
    then
        echo "PASS  $f"
    else
        echo "FAIL  $f"
        FREEZE_FAIL=1
    fi
done

echo

if [[ "$FREEZE_FAIL" -ne 0 ]]
then
    echo "FATAL: frozen task is incomplete."
    exit 2
fi


echo "Line counts:"

for f in \
    "$FREEZE/REFERENCE_A_2374.tsv" \
    "$FREEZE/QUERY_B1_391.tsv" \
    "$FREEZE/QUERY_B2_SEEN_196.tsv" \
    "$FREEZE/QUERY_B2_UNSEEN_193.tsv"
do
    printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
done

echo


echo "SHA256 verification:"

(
    cd "$FREEZE" || exit 1
    sha256sum -c EXTERNAL_TASK_SHA256.txt
)

SHA_RC=$?

echo
echo "SHA256_CHECK_RC=$SHA_RC"

if [[ "$SHA_RC" -ne 0 ]]
then
    echo "FATAL: external task SHA256 verification failed."
    exit 3
fi


# --------------------------------------------------------------------------------------------------
# 2. Tool availability
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "2. NATIVE TOOL AVAILABILITY"
echo "===================================================================================================="

TOOLS=(
    mash
    dashing2
    fastANI
    skani
)

for tool in "${TOOLS[@]}"
do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "TOOL: $tool"
    echo "--------------------------------------------------------------------------------"

    if command -v "$tool" >/dev/null 2>&1
    then
        echo "ACTIVE_ENV_PATH=$(command -v "$tool")"

        "$tool" --version 2>&1 | head -n 5 || \
        "$tool" -V 2>&1 | head -n 5 || \
        "$tool" -h 2>&1 | head -n 5 || true
    else
        echo "ACTIVE_ENV_PATH=NOT_FOUND"
    fi

    if command -v conda >/dev/null 2>&1
    then
        echo
        echo "[hgv_env]"

        conda run -n hgv_env \
            bash -c "command -v $tool || true" \
            2>/dev/null || true

        conda run -n hgv_env \
            "$tool" --version \
            2>&1 | head -n 5 || true
    fi
done


# --------------------------------------------------------------------------------------------------
# 3. Known Mash / Dashing2 production caches
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "3. EXISTING PRODUCTION TEACHER CACHES"
echo "===================================================================================================="

KNOWN_CACHE_FILES=(
    "$TEACHER/mash/mash_triangle_edges.tsv"
    "$TEACHER/dashing2_set/dashing2_set_similarity.tsv"
)

for f in "${KNOWN_CACHE_FILES[@]}"
do
    echo

    if [[ -f "$f" ]]
    then
        echo "FOUND $f"
        ls -lh "$f"

        echo "HEADER:"
        head -n 3 "$f"
    else
        echo "MISSING $f"
    fi
done


echo
echo "NOTE:"
echo "These teacher caches are expected to describe Dataset A."
echo "They do NOT by themselves prove that B1/B2 external query sketches have been generated."


# --------------------------------------------------------------------------------------------------
# 4. Find exact existing native-baseline scripts / command syntax
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "4. EXISTING TOOL INVOCATIONS IN FROZEN/USED SCRIPTS"
echo "===================================================================================================="

SEARCH_SCRIPT_ROOTS=(
    "$BASE/scripts_v3_20260917_e5_baselines"
    "$BASE/scripts_v3_20260916_teacher"
    "$BASE/scripts_v3_20260916_hgv"
)

for root in "${SEARCH_SCRIPT_ROOTS[@]}"
do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "SCRIPT ROOT: $root"
    echo "--------------------------------------------------------------------------------"

    if [[ ! -d "$root" ]]
    then
        echo "MISSING"
        continue
    fi

    grep -RInE \
        '(^|[^A-Za-z0-9_])(mash|dashing2|fastANI|skani)([^A-Za-z0-9_]|$)' \
        "$root" \
        --include='*.py' \
        --include='*.sh' \
        2>/dev/null \
        | head -n 160 || true
done


# --------------------------------------------------------------------------------------------------
# 5. Existing E5 native baseline result structure
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "5. E5 INTERNAL NATIVE BASELINE FILE INVENTORY"
echo "===================================================================================================="

if [[ -d "$E5" ]]
then
    find "$E5" \
        -maxdepth 4 \
        -type f \
        \( \
            -iname '*mash*' -o \
            -iname '*dashing*' -o \
            -iname '*fastani*' -o \
            -iname '*skani*' -o \
            -iname '*threshold*' -o \
            -iname '*coverage*' -o \
            -iname '*native*' \
        \) \
        -printf '%s\t%p\n' \
        2>/dev/null \
        | sort -n \
        | tail -n 180
else
    echo "MISSING E5 ROOT: $E5"
fi


# --------------------------------------------------------------------------------------------------
# 6. DNABERT-S production artifact inventory
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "6. DNABERT-S PRODUCTION ARTIFACTS"
echo "===================================================================================================="

if [[ ! -d "$DNABERT" ]]
then
    echo "FATAL: DNABERT root missing:"
    echo "$DNABERT"
    exit 4
fi


echo
echo "[Candidate embedding / matrix files]"

find "$DNABERT" \
    -maxdepth 5 \
    -type f \
    \( \
        -iname '*.npy' -o \
        -iname '*.npz' -o \
        -iname '*embedding*' -o \
        -iname '*vector*' \
    \) \
    -printf '%s\t%p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 120


echo
echo "[Candidate row-order / accession / manifest files]"

find "$DNABERT" \
    -maxdepth 5 \
    -type f \
    \( \
        -iname '*manifest*' -o \
        -iname '*accession*' -o \
        -iname '*row*id*' -o \
        -iname '*row*order*' -o \
        -iname '*index*' -o \
        -iname '*metadata*' -o \
        -iname '*production*protocol*' \
    \) \
    -printf '%s\t%p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 160


# --------------------------------------------------------------------------------------------------
# 7. Inspect NumPy matrix shapes without loading matrices into RAM
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "7. DNABERT-S NUMPY MATRIX SHAPE AUDIT"
echo "===================================================================================================="

export HGV_DNABERT_ROOT="$DNABERT"

python3 - <<'PY'
from pathlib import Path
import os
import numpy as np

root = Path(os.environ["HGV_DNABERT_ROOT"])

files = []

for p in root.rglob("*.npy"):
    if any(
        x in p.name.lower()
        for x in [
            "embed",
            "vector",
            "genome",
            "l2",
            "raw",
        ]
    ):
        files.append(p)

print(f"candidate_npy_files={len(files)}")

for p in sorted(files):
    try:
        x = np.load(p, mmap_mode="r")
        print(
            f"NPY\tshape={x.shape}\tdtype={x.dtype}\t"
            f"bytes={p.stat().st_size}\t{p}"
        )
    except Exception as e:
        print(f"NPY_ERROR\t{p}\t{type(e).__name__}: {e}")

for p in sorted(root.rglob("*.npz")):
    if not any(
        x in p.name.lower()
        for x in [
            "embed",
            "vector",
            "genome",
            "l2",
            "raw",
        ]
    ):
        continue

    try:
        z = np.load(p, mmap_mode="r")
        desc = []
        for k in z.files:
            a = z[k]
            desc.append(f"{k}:{a.shape}:{a.dtype}")
        print(
            f"NPZ\t{';'.join(desc)}\t"
            f"bytes={p.stat().st_size}\t{p}"
        )
    except Exception as e:
        print(f"NPZ_ERROR\t{p}\t{type(e).__name__}: {e}")
PY


# --------------------------------------------------------------------------------------------------
# 8. Locate DNABERT production scripts and frozen hashes
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "8. DNABERT-S PRODUCTION SCRIPT / HASH EVIDENCE"
echo "===================================================================================================="

find "$DNABERT" \
    -maxdepth 6 \
    -type f \
    \( \
        -iname '*sha256*' -o \
        -iname '*freeze*' -o \
        -iname '*production*' -o \
        -iname '*strict32*' -o \
        -iname '*blind*test*' \
    \) \
    -printf '%s\t%p\n' \
    2>/dev/null \
    | sort -n \
    | tail -n 180


echo
echo "[DNABERT external/retrieval-related code references]"

grep -RInE \
    '(retrieval|external|query|reference|cosine|euclidean|embedding)' \
    "$BASE/scripts_v3_20260917_e5_baselines" \
    "$DNABERT" \
    --include='*.py' \
    --include='*.sh' \
    --include='*.txt' \
    2>/dev/null \
    | grep -i 'dnabert\|embedding\|06_dnaberts' \
    | head -n 180 || true


# --------------------------------------------------------------------------------------------------
# 9. Check exact query/reference lists
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "9. FASTA LIST FILE CHECK"
echo "===================================================================================================="

python3 - <<'PY'
from pathlib import Path

base = Path(
    "HGV_V3_revision_20260915/"
    "09_baselines/E5_external_unified_v1_20260920/00_task_freeze"
)

items = {
    "A": base / "REFERENCE_A_2374_FASTA.list",
    "B1": base / "QUERY_B1_391_FASTA.list",
    "B2_seen": base / "QUERY_B2_SEEN_196_FASTA.list",
    "B2_unseen": base / "QUERY_B2_UNSEEN_193_FASTA.list",
}

for name, p in items.items():

    lines = [
        x.strip()
        for x in p.read_text().splitlines()
        if x.strip()
    ]

    missing = [
        x for x in lines
        if not Path(x).is_file()
    ]

    print(
        f"{name}\t"
        f"n={len(lines)}\t"
        f"unique={len(set(lines))}\t"
        f"missing={len(missing)}"
    )

    for x in missing[:5]:
        print("  MISSING", x)
PY


# --------------------------------------------------------------------------------------------------
# 10. Condensed decision block
# --------------------------------------------------------------------------------------------------

echo
echo "===================================================================================================="
echo "10. PREFLIGHT COMPLETE"
echo "===================================================================================================="
echo
echo "This report is execution provenance only."
echo
echo "No Mash/Dashing2/FastANI/skani/DNABERT external experiment was executed."
echo "No frozen result was modified."
echo
echo "Use this report to build the final 29a-29e external baseline runners."
echo
echo "REPORT:"
echo "$REPORT"
echo
echo "===================================================================================================="
