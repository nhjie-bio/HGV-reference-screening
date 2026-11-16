#!/usr/bin/env bash
set -Eeuo pipefail

BASE="HGV_V3_revision_20260915"
META="$BASE/00_metadata"
AUX="$META/auxillary_files"
LOG="$BASE/logs/00_fetch_gtdb_r232_metadata.log"

ROOT_URL="https://data.gtdb.ecogenomic.org/releases/release232/232.0"
AUX_URL="$ROOT_URL/auxillary_files"

mkdir -p "$META"
mkdir -p "$AUX"
mkdir -p "$BASE/logs"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "HGV-V3 Phase 0A: GTDB R232 metadata download"
echo "Start: $(date '+%F %T')"
echo "Base : $BASE"
echo "============================================================"

echo
echo "[1/6] Checking required commands..."

for cmd in wget gzip md5sum awk grep; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $cmd"
        exit 1
    fi
done

echo "Required commands: OK"

echo
echo "[2/6] Checking project directories..."

if [[ ! -d "$BASE/scripts" ]]; then
    echo "ERROR: project directory is not initialized correctly:"
    echo "$BASE"
    exit 1
fi

df -h "$BASE"

download_file() {
    local url="$1"
    local destdir="$2"
    local filename="${url##*/}"

    echo
    echo "------------------------------------------------------------"
    echo "File: $filename"
    echo "URL : $url"
    echo "Dest: $destdir/$filename"
    echo "------------------------------------------------------------"

    (
        cd "$destdir"

        wget \
            -c \
            --tries=10 \
            --timeout=60 \
            --waitretry=5 \
            --retry-connrefused \
            "$url"
    )

    if [[ ! -s "$destdir/$filename" ]]; then
        echo "ERROR: file is missing or empty:"
        echo "$destdir/$filename"
        exit 1
    fi
}

echo
echo "[3/6] Downloading GTDB release information..."

download_file "$ROOT_URL/VERSION.txt" "$META"
download_file "$ROOT_URL/RELEASE_NOTES.txt" "$META"
download_file "$ROOT_URL/FILE_DESCRIPTIONS.txt" "$META"
download_file "$ROOT_URL/METHODS.txt" "$META"
download_file "$ROOT_URL/MD5SUM.txt" "$META"

echo
echo "[4/6] Downloading bacterial metadata/taxonomy/tree..."

download_file "$ROOT_URL/bac120_metadata_r232.tsv.gz" "$META"
download_file "$ROOT_URL/bac120_taxonomy_r232.tsv.gz" "$META"
download_file "$ROOT_URL/bac120_r232.tree" "$META"

echo
echo "[5/6] Downloading required auxiliary files..."

download_file "$AUX_URL/bac120_r232.sp_labels.tree" "$AUX"
download_file "$AUX_URL/sp_clusters_r232.tsv" "$AUX"
download_file "$AUX_URL/synonyms.bac120_r232.tsv" "$AUX"
download_file "$AUX_URL/metadata_field_desc.tsv" "$AUX"
download_file "$AUX_URL/qc_failed_r232.tsv" "$AUX"

echo
echo "[6/6] Validating downloaded files..."

cd "$META"

echo
echo "Checking gzip integrity..."

gzip -t bac120_metadata_r232.tsv.gz
echo "PASS: bac120_metadata_r232.tsv.gz"

gzip -t bac120_taxonomy_r232.tsv.gz
echo "PASS: bac120_taxonomy_r232.tsv.gz"

echo
echo "Building official MD5 target list..."

TARGETS=(
    "VERSION.txt"
    "RELEASE_NOTES.txt"
    "FILE_DESCRIPTIONS.txt"
    "METHODS.txt"
    "bac120_metadata_r232.tsv.gz"
    "bac120_taxonomy_r232.tsv.gz"
    "bac120_r232.tree"
    "auxillary_files/bac120_r232.sp_labels.tree"
    "auxillary_files/sp_clusters_r232.tsv"
    "auxillary_files/synonyms.bac120_r232.tsv"
    "auxillary_files/metadata_field_desc.tsv"
    "auxillary_files/qc_failed_r232.tsv"
)

: > MD5SUM.targets.txt

for target in "${TARGETS[@]}"; do
    if ! grep -F "  ./$target" MD5SUM.txt >> MD5SUM.targets.txt; then
        echo "ERROR: official MD5 entry not found for:"
        echo "$target"
        exit 1
    fi
done

echo
echo "Official checksum entries:"
cat MD5SUM.targets.txt

echo
echo "Running MD5 verification..."

md5sum -c MD5SUM.targets.txt

echo
echo "============================================================"
echo "GTDB R232 METADATA DOWNLOAD COMPLETE"
echo "End: $(date '+%F %T')"
echo "============================================================"

echo
echo "Release:"
cat VERSION.txt

echo
echo "Root metadata files:"
ls -lh \
    VERSION.txt \
    RELEASE_NOTES.txt \
    FILE_DESCRIPTIONS.txt \
    METHODS.txt \
    MD5SUM.txt \
    bac120_metadata_r232.tsv.gz \
    bac120_taxonomy_r232.tsv.gz \
    bac120_r232.tree

echo
echo "Auxiliary files:"
ls -lh \
    auxillary_files/bac120_r232.sp_labels.tree \
    auxillary_files/sp_clusters_r232.tsv \
    auxillary_files/synonyms.bac120_r232.tsv \
    auxillary_files/metadata_field_desc.tsv \
    auxillary_files/qc_failed_r232.tsv

echo
echo "ALL CHECKS PASSED."
