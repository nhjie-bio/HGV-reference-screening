#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
TAG=20260926

OUT="$BASE/11_manuscript/HGV_V3_final_manuscript_figures_v3_${TAG}"

if [[ -e "$OUT" ]]; then
    echo "[STOP] Output directory already exists:"
    echo "$OUT"
    echo "Nothing was overwritten."
    exit 1
fi

mkdir -p \
    "$OUT/00_source_snapshot" \
    "$OUT/01_normalized_source_data" \
    "$OUT/02_figures/main" \
    "$OUT/02_figures/supplementary" \
    "$OUT/03_audit" \
    "$OUT/04_tables"

echo "============================================================"
echo "HGV-V3 FINAL MANUSCRIPT FIGURE PACKAGE v3"
echo "READ-ONLY SOURCE COLLECTION"
echo "============================================================"

# ------------------------------------------------------------
# 1. Prefer fix2; otherwise use the already-audited fix1 package
# ------------------------------------------------------------

SRC=$(find "$BASE" -type d \
    -name 'HGV_V3_manuscript_figure_package_v2_fix2_20260926' \
    -print -quit)

if [[ -z "${SRC:-}" ]]; then
    SRC=$(find "$BASE" -type d \
        -name 'HGV_V3_manuscript_figure_package_v2_fix1_20260926' \
        -print -quit)
fi

if [[ -z "${SRC:-}" ]]; then
    echo "[FAIL] Cannot locate fix2 or fix1 manuscript figure package."
    exit 1
fi

echo "[SOURCE PACKAGE]"
echo "$SRC"
echo "$SRC" > "$OUT/03_audit/SOURCE_PACKAGE.txt"

# ------------------------------------------------------------
# 2. Copy data/text sources only.
#    No PNG/PDF from the old package is reused.
# ------------------------------------------------------------

cd "$SRC"

find . -type f \
    \( -name '*.tsv' \
       -o -name '*.csv' \
       -o -name '*.json' \
       -o -name '*.txt' \) \
    -print0 |
while IFS= read -r -d '' F; do
    mkdir -p "$OUT/00_source_snapshot/$(dirname "$F")"
    cp -a "$F" "$OUT/00_source_snapshot/$F"
done

# ------------------------------------------------------------
# 3. Add frozen final manifest
# ------------------------------------------------------------

MANIFEST="$BASE/01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv"

if [[ -s "$MANIFEST" ]]; then
    cp -a "$MANIFEST" \
        "$OUT/00_source_snapshot/final_master_manifest_3154.tsv"
else
    echo "[FAIL] Frozen 3154 manifest not found:"
    echo "$MANIFEST"
    exit 1
fi

# ------------------------------------------------------------
# 4. Build inventory
# ------------------------------------------------------------

find "$OUT/00_source_snapshot" -type f \
    | sort \
    > "$OUT/03_audit/SOURCE_FILE_LIST.txt"

(
    cd "$OUT/00_source_snapshot"
    find . -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "$OUT/03_audit/SOURCE_SHA256.txt"

echo
echo "[SOURCE FILE COUNT]"
find "$OUT/00_source_snapshot" -type f | wc -l

echo
echo "[TABLE INVENTORY]"
find "$OUT/00_source_snapshot" -type f \
    \( -name '*.tsv' -o -name '*.csv' \) \
    | sort

echo
echo "============================================================"
echo "SOURCE SNAPSHOT COMPLETE"
echo "============================================================"
echo "OUT=$OUT"
