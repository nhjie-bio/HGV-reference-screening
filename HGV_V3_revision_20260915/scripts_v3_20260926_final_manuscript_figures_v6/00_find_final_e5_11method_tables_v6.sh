#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
OUT="$BASE/18_figures/HGV_V3_FINAL_MANUSCRIPT_FIGURES_v6_20260926"

mkdir -p "$OUT/audit"

REPORT="$OUT/audit/E5_11METHOD_TABLE_INVENTORY.txt"

{
echo "======================================================================"
echo "HGV-V3 E5 FINAL 11-METHOD TABLE INVENTORY"
echo "READ ONLY"
echo "======================================================================"
echo

echo "### Search areas"
echo

for D in \
    "$BASE/09_baselines" \
    "$BASE/09_external_validation" \
    "$BASE/20_rebuttal" \
    "$BASE/18_figures"
do
    echo
    echo "######################################################################"
    echo "DIRECTORY: $D"
    echo "######################################################################"

    if [[ ! -d "$D" ]]; then
        echo "[MISSING DIRECTORY]"
        continue
    fi

    find "$D" -type f \
        \( -iname '*.tsv' -o -iname '*.csv' \) \
        \( \
          -iname '*11method*' -o \
          -iname '*11_method*' -o \
          -iname '*unified*' -o \
          -iname '*baseline*summary*' -o \
          -iname '*primary*summary*' -o \
          -iname '*external*final*' -o \
          -iname '*runtime*summary*' \
        \) \
        -printf '%s\t%p\n' \
        | sort -nr
done

echo
echo "======================================================================"
echo "COLUMN PREVIEW"
echo "======================================================================"

find \
    "$BASE/09_baselines" \
    "$BASE/09_external_validation" \
    "$BASE/20_rebuttal" \
    -type f \
    \( -iname '*.tsv' -o -iname '*.csv' \) \
    2>/dev/null \
    | while read -r F
do
    HEADER=$(head -n 1 "$F" 2>/dev/null || true)

    if echo "$HEADER" | grep -Eqi \
       'method.*AUROC|AUROC.*method|method.*MRR|method.*runtime'
    then
        echo
        echo "FILE=$F"
        echo "HEADER=$HEADER"
        echo "ROWS=$(wc -l < "$F")"
        head -n 4 "$F"
    fi
done

} | tee "$REPORT"

echo
echo "[PASS] $REPORT"
