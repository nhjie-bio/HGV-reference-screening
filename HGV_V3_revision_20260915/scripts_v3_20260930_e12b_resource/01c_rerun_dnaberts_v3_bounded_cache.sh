#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SDIR="$BASE/scripts_v3_20260930_e12b_resource"
OUT="$BASE/16_scalability/E12B_synchronized_resource_v1_20260930"

source "$OUT/02_preflight/E12B_RESOLVED_ENV.sh"

RUNROOT="$OUT/03_runs"
RUNV3="$RUNROOT/DNABERT-S_v3_bounded_cache_20261001"
LOGS="$OUT/logs"

SCRIPT="$SDIR/e12b_dnaberts_v3_bounded_cache.py"

OLD="$RUNROOT/DNABERT-S"

mkdir -p "$LOGS"

# Never overwrite a partial v3 production run.
if [[ -d "$RUNV3/rep1" || -d "$RUNV3/rep2" || -d "$RUNV3/rep3" ]]; then
    echo "[FAIL] v3 production run directory already contains repeat data:"
    echo "  $RUNV3"
    echo "Do not overwrite it."
    exit 2
fi

mkdir -p "$RUNV3/validate"

echo "======================================================================"
echo "HGV-V3 E12B DNABERT-S BOUNDED-CACHE RESOURCE RERUN v3"
echo "======================================================================"
echo
echo "Only DNABERT-S is rerun."
echo "All previous E12B methods remain read-only."
echo

DNACOMB="$OUT/01_manifests/E12B_DNABERT_COMBINED_2424.tsv"

if [[ ! -s "$DNACOMB" ]]; then
python3 - \
"$E12B_REFS_TSV" \
"$E12B_QUERIES_TSV" \
"$DNACOMB" <<'PY'
import sys
import pandas as pd

r = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
q = pd.read_csv(sys.argv[2], sep="\t", dtype=str)

x = pd.concat([r, q], ignore_index=True)

assert len(r) == 2374
assert len(q) == 50
assert len(x) == 2424
assert x["assembly_accession"].nunique() == 2424

x.to_csv(sys.argv[3], sep="\t", index=False)

print("COMBINED MANIFEST PASS:", len(x))
PY
fi

# ----------------------------------------------------------------------
# Exact frozen sequence validation
# ----------------------------------------------------------------------

echo
echo "======================================================================"
echo "STEP V3-1 - STRICT32 SEQUENCE VALIDATION"
echo "======================================================================"

conda run -n dnabert_s_e5 \
python "$SCRIPT" \
  --mode validate \
  --strict32 "$E12B_DNABERT_STRICT32" \
  --subset "$E12B_QUERIES_TSV" \
  --snapshot "$E12B_DNABERT_SNAPSHOT" \
  --outdir "$RUNV3/validate" \
  2>&1 | tee "$LOGS/10_dnabert_v3_validation.log"

grep -q 'VALIDATION PASS' \
"$LOGS/10_dnabert_v3_validation.log" || {
    echo "[FAIL] v3 strict32 validation"
    exit 2
}

echo "[PASS] exact STRICT32 sequence validation"

# ----------------------------------------------------------------------
# Resource monitor
# ----------------------------------------------------------------------

run_mon () {
    local name="$1"
    local od="$2"
    shift 2

    mkdir -p "$od"

    python "$SDIR/e12b_monitor.py" \
      --name "$name" \
      --out "$od" \
      -- "$@"
}

# ----------------------------------------------------------------------
# 3 independent repeats
# ----------------------------------------------------------------------

for rep in 1 2 3; do

    echo
    echo "======================================================================"
    echo "DNABERT-S v3 REPEAT $rep/3"
    echo "======================================================================"

    MROOT="$RUNV3/rep${rep}"

    mkdir -p \
      "$MROOT/build" \
      "$MROOT/query" \
      "$MROOT/full"

    # BUILD: 2374 references
    run_mon \
      "DNABERT-S-v3.build.rep${rep}" \
      "$MROOT/build/resource" \
      conda run -n dnabert_s_e5 \
      python "$SCRIPT" \
        --mode build \
        --strict32 "$E12B_DNABERT_STRICT32" \
        --subset "$E12B_REFS_TSV" \
        --snapshot "$E12B_DNABERT_SNAPSHOT" \
        --outdir "$MROOT/build/output"

    INDEX="$MROOT/build/output/reference_dnaberts_l2.float32.npy"

    [[ -s "$INDEX" ]] || {
        echo "[FAIL] missing v3 reference vectors"
        exit 2
    }

    # WARM QUERY: 50 queries
    run_mon \
      "DNABERT-S-v3.query.rep${rep}" \
      "$MROOT/query/resource" \
      conda run -n dnabert_s_e5 \
      python "$SCRIPT" \
        --mode query \
        --strict32 "$E12B_DNABERT_STRICT32" \
        --subset "$E12B_QUERIES_TSV" \
        --snapshot "$E12B_DNABERT_SNAPSHOT" \
        --index-in "$INDEX" \
        --outdir "$MROOT/query/output"

    # ------------------------------------------------------------------
    # After rep1, prove bounded-cache v3 is numerically equivalent
    # to the already completed v2 implementation.
    # ------------------------------------------------------------------

    if [[ "$rep" -eq 1 ]]; then

        echo
        echo "===== v2 vs v3 REPRESENTATION EQUIVALENCE ====="

        python3 - \
          "$OLD/rep1/build/output/reference_dnaberts_l2.float32.npy" \
          "$MROOT/build/output/reference_dnaberts_l2.float32.npy" \
          "$OLD/rep1/query/output/top1_predictions.tsv" \
          "$MROOT/query/output/top1_predictions.tsv" \
          "$RUNV3/V2_V3_EQUIVALENCE_AUDIT.txt" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

old_vec = np.load(sys.argv[1]).astype(np.float64)
new_vec = np.load(sys.argv[2]).astype(np.float64)

if old_vec.shape != new_vec.shape:
    raise RuntimeError(
        f"shape mismatch: {old_vec.shape} vs {new_vec.shape}"
    )

max_abs = float(np.max(np.abs(old_vec - new_vec)))

num = np.sum(old_vec * new_vec, axis=1)
den = (
    np.linalg.norm(old_vec, axis=1)
    *
    np.linalg.norm(new_vec, axis=1)
)

cos = num / np.maximum(den, 1e-15)

min_cos = float(np.min(cos))
mean_cos = float(np.mean(cos))

old_q = pd.read_csv(sys.argv[3], sep="\t")
new_q = pd.read_csv(sys.argv[4], sep="\t")

same_top1 = int(
    (
        old_q["top1_index"].astype(int).to_numpy()
        ==
        new_q["top1_index"].astype(int).to_numpy()
    ).sum()
)

nq = len(old_q)

status = (
    max_abs <= 1e-4
    and min_cos >= 0.999999
    and same_top1 == nq
)

lines = [
    "HGV-V3 E12B DNABERT-S v2/v3 equivalence audit",
    f"shape={old_vec.shape}",
    f"max_abs_diff={max_abs:.12g}",
    f"min_cosine={min_cos:.12g}",
    f"mean_cosine={mean_cos:.12g}",
    f"query_top1_identical={same_top1}/{nq}",
    f"STATUS={'PASS' if status else 'FAIL'}",
]

Path(sys.argv[5]).write_text(
    "\n".join(lines) + "\n"
)

print("\n".join(lines))

if not status:
    raise SystemExit(2)
PY

        grep -q '^STATUS=PASS$' \
          "$RUNV3/V2_V3_EQUIVALENCE_AUDIT.txt" || {
            echo "[FAIL] v2/v3 equivalence gate"
            exit 2
        }

        echo "[PASS] v2/v3 numerical equivalence"
    fi

    # COMPLETE FIRST-USE: 2374 refs + 50 queries
    run_mon \
      "DNABERT-S-v3.full.rep${rep}" \
      "$MROOT/full/resource" \
      conda run -n dnabert_s_e5 \
      python "$SCRIPT" \
        --mode full \
        --strict32 "$E12B_DNABERT_STRICT32" \
        --subset "$DNACOMB" \
        --refs "$E12B_REFS_TSV" \
        --queries "$E12B_QUERIES_TSV" \
        --snapshot "$E12B_DNABERT_SNAPSHOT" \
        --outdir "$MROOT/full/output"

done

echo
echo "======================================================================"
echo "DNABERT-S v3 BOUNDED-CACHE RUN COMPLETE"
echo "======================================================================"

cat "$RUNV3/V2_V3_EQUIVALENCE_AUDIT.txt"
