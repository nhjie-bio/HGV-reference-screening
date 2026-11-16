#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
PKG="$BASE/18_figures/HGV_V3_FINAL_MANUSCRIPT_PACKAGE_v6_20260926"

if [[ -e "$PKG" ]]; then
    echo "[STOP] v6 package already exists:"
    echo "$PKG"
    echo "Nothing overwritten."
    exit 1
fi

SRC="$PKG/source_data"
mkdir -p \
    "$SRC" \
    "$PKG/main_figures" \
    "$PKG/supplementary_figures" \
    "$PKG/tables" \
    "$PKG/audit" \
    "$PKG/captions"

INT="$BASE/09_baselines/E5_unified_baseline_v1_20260917/07_unified_internal/e5_internal_11method_final_v1_20260918"

EXT="$BASE/09_baselines/E5_external_unified_v1_20260920/06_external_final_freeze_v2_20260922"

OLD="$BASE/11_manuscript/HGV_V3_final_manuscript_figures_v3_20260926/00_source_snapshot/source_data"

copy_required () {
    SRCFILE="$1"
    DSTNAME="$2"

    if [[ ! -s "$SRCFILE" ]]; then
        echo "[FAIL] Missing: $SRCFILE"
        exit 1
    fi

    cp -a "$SRCFILE" "$SRC/$DSTNAME"
    echo "[PASS] $DSTNAME"
}

echo "============================================================"
echo "HGV-V3 FINAL SOURCE FREEZE v6"
echo "============================================================"

# ------------------------------------------------------------
# E5 final 11-method blocks
# ------------------------------------------------------------

copy_required \
"$INT/E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv" \
"E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv"

copy_required \
"$INT/E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv" \
"E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv"

copy_required \
"$INT/E5_INTERNAL_11METHOD_METHOD_CATALOG.tsv" \
"E5_INTERNAL_11METHOD_METHOD_CATALOG.tsv"

copy_required \
"$EXT/E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv" \
"E5_EXTERNAL_ALL11_METHOD_SUMMARY_FINAL.tsv"

copy_required \
"$EXT/E5_EXTERNAL_ALL11_SEED_LEVEL_FINAL.tsv" \
"E5_EXTERNAL_ALL11_SEED_LEVEL_FINAL.tsv"

copy_required \
"$EXT/E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv" \
"E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv"

# ------------------------------------------------------------
# E6-E12 canonical frozen result blocks
# ------------------------------------------------------------

for F in \
    e6_modes.tsv \
    e6_contrasts.tsv \
    e7.tsv \
    e8.tsv \
    e9_ig.tsv \
    e10_corr.tsv \
    e10_tax.tsv \
    e10_confound.tsv \
    e10_tree_summary.tsv \
    e10_tree_pairs.tsv \
    e11.tsv \
    e12.tsv
do
    copy_required "$OLD/$F" "$F"
done

# ------------------------------------------------------------
# Runtime provenance status
# ------------------------------------------------------------

RUNTIME="$BASE/20_rebuttal/HGV_V3_final_closure_v3_20260926_185328/STRICT_RUNTIME_STATUS_FROZEN_V2_COPY.tsv"

if [[ -s "$RUNTIME" ]]; then
    cp -a "$RUNTIME" "$SRC/STRICT_RUNTIME_STATUS.tsv"
fi

# ------------------------------------------------------------
# SHA freeze
# ------------------------------------------------------------

(
    cd "$SRC"
    sha256sum * | sort
) > "$PKG/audit/SOURCE_SHA256.tsv"

find "$SRC" -maxdepth 1 -type f -printf '%f\n' \
    | sort \
    > "$PKG/audit/SOURCE_FILES.txt"

echo
echo "Files frozen:"
wc -l "$PKG/audit/SOURCE_FILES.txt"

echo
echo "============================================================"
echo "SOURCE_FREEZE_V6=PASS"
echo "PKG=$PKG"
echo "============================================================"
