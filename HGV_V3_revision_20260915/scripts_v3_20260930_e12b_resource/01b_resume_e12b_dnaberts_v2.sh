#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SDIR="$BASE/scripts_v3_20260930_e12b_resource"
OUT="$BASE/16_scalability/E12B_synchronized_resource_v1_20260930"

ENVFILE="$OUT/02_preflight/E12B_RESOLVED_ENV.sh"
RUNROOT="$OUT/03_runs"
LOGS="$OUT/logs"

source "$ENVFILE"

REPEATS=3

echo "======================================================================"
echo "HGV-V3 E12B RESUME - DNABERT-S ONLY v2"
echo "======================================================================"
echo
echo "Already completed and NOT rerun:"
echo "  HGV"
echo "  Raw6_BrayCurtis"
echo "  Raw6_Cosine"
echo "  Raw6_Euclidean"
echo "  PCA64_Euclidean"
echo "  PCA64_L2"
echo "  Mash"
echo "  Dashing2"
echo "  skani"
echo "  FastANI"
echo
echo "Remaining:"
echo "  DNABERT-S + final aggregation"
echo "======================================================================"

# ------------------------------------------------------------
# Archive the failed v1 DNABERT validation only.
# ------------------------------------------------------------

if [[ -d "$RUNROOT/DNABERT-S" ]]; then
    if [[ ! -e "$RUNROOT/DNABERT-S_failed_v1_20261001" ]]; then
        mv \
          "$RUNROOT/DNABERT-S" \
          "$RUNROOT/DNABERT-S_failed_v1_20261001"

        echo "[PASS] failed DNABERT-S v1 provenance archived"
    fi
fi

mkdir -p "$RUNROOT/DNABERT-S"

# ------------------------------------------------------------
# Combined 2374 + 50 manifest
# ------------------------------------------------------------

DNACOMB="$OUT/01_manifests/E12B_DNABERT_COMBINED_2424.tsv"

python3 - \
"$E12B_REFS_TSV" \
"$E12B_QUERIES_TSV" \
"$DNACOMB" <<'PY'
import sys
import pandas as pd

r = pd.read_csv(
    sys.argv[1],
    sep="\t",
    dtype=str,
)

q = pd.read_csv(
    sys.argv[2],
    sep="\t",
    dtype=str,
)

x = pd.concat(
    [r, q],
    ignore_index=True,
)

assert len(r) == 2374
assert len(q) == 50
assert len(x) == 2424
assert x["assembly_accession"].nunique() == 2424

x.to_csv(
    sys.argv[3],
    sep="\t",
    index=False,
)

print("COMBINED MANIFEST PASS:", len(x))
PY

# ------------------------------------------------------------
# Mandatory validation gate
# ------------------------------------------------------------

echo
echo "======================================================================"
echo "STEP 7B - DNABERT-S STRICT32 EXACT-COORDINATE VALIDATION"
echo "======================================================================"

conda run -n dnabert_s_e5 \
python "$SDIR/e12b_dnaberts_v2.py" \
  --mode validate \
  --strict32 "$E12B_DNABERT_STRICT32" \
  --subset "$E12B_QUERIES_TSV" \
  --snapshot "$E12B_DNABERT_SNAPSHOT" \
  --outdir "$RUNROOT/DNABERT-S/validate" \
  2>&1 | tee "$LOGS/07b_dnabert_validation_v2.log"

grep -q 'VALIDATION PASS' \
"$LOGS/07b_dnabert_validation_v2.log" || {
    echo "[FAIL] DNABERT strict32 validation did not pass"
    exit 2
}

echo "[PASS] frozen STRICT32 sequence recovery validated"

# ------------------------------------------------------------
# Monitor helper
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Three repeats
# ------------------------------------------------------------

echo
echo "======================================================================"
echo "STEP 8B - DNABERT-S RESOURCE BENCHMARK"
echo "======================================================================"

for rep in 1 2 3; do

    echo
    echo "----- DNABERT-S REPEAT $rep/3 -----"

    MROOT="$RUNROOT/DNABERT-S/rep${rep}"

    mkdir -p \
      "$MROOT/build" \
      "$MROOT/query" \
      "$MROOT/full"

    # ---------------- BUILD ----------------

    run_mon \
      "DNABERT-S.build.rep${rep}" \
      "$MROOT/build/resource" \
      conda run -n dnabert_s_e5 \
      python "$SDIR/e12b_dnaberts_v2.py" \
        --mode build \
        --strict32 "$E12B_DNABERT_STRICT32" \
        --subset "$E12B_REFS_TSV" \
        --snapshot "$E12B_DNABERT_SNAPSHOT" \
        --outdir "$MROOT/build/output"

    DNAINDEX="$MROOT/build/output/reference_dnaberts_l2.float32.npy"

    [[ -s "$DNAINDEX" ]] || {
        echo "[FAIL] missing DNABERT reference vectors"
        exit 2
    }

    # ---------------- WARM QUERY ----------------

    run_mon \
      "DNABERT-S.query.rep${rep}" \
      "$MROOT/query/resource" \
      conda run -n dnabert_s_e5 \
      python "$SDIR/e12b_dnaberts_v2.py" \
        --mode query \
        --strict32 "$E12B_DNABERT_STRICT32" \
        --subset "$E12B_QUERIES_TSV" \
        --snapshot "$E12B_DNABERT_SNAPSHOT" \
        --index-in "$DNAINDEX" \
        --outdir "$MROOT/query/output"

    # ---------------- COMPLETE ----------------

    run_mon \
      "DNABERT-S.full.rep${rep}" \
      "$MROOT/full/resource" \
      conda run -n dnabert_s_e5 \
      python "$SDIR/e12b_dnaberts_v2.py" \
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
echo "STEP 9B - FINALIZE EXISTING 11-METHOD E12B"
echo "======================================================================"

python "$SDIR/e12b_finalize.py" \
  2>&1 | tee "$LOGS/09b_finalize_e12b_after_dnabert_v2.log"

echo
echo "======================================================================"
echo "E12B RESUME COMPLETE"
echo "======================================================================"

cat "$OUT/05_final_freeze/E12B_FINAL_STATUS.txt"

echo
echo "TABLE:"
echo "$OUT/04_summary/TABLE7_SOURCE_DATA.tsv"
