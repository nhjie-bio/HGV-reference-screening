#!/usr/bin/env bash
set -Eeuo pipefail

BASE="HGV_V3_revision_20260915"

FREEZE="$BASE/01_manifest/frozen_pre_download_20260916"

LOGDIR="$BASE/logs"
LOG="$LOGDIR/07_download_genomes_ncbi_v2.log"

DOWNLOAD_ROOT="$BASE/02_genomes"

SUMMARY="$DOWNLOAD_ROOT/download_reconciliation_summary.tsv"
PROVENANCE="$DOWNLOAD_ROOT/DOWNLOAD_PROVENANCE.txt"

WORKERS="${HGV_REHYDRATE_WORKERS:-20}"
GZIP_CHECK_WORKERS="${HGV_GZIP_CHECK_WORKERS:-8}"

mkdir -p "$LOGDIR"
mkdir -p "$DOWNLOAD_ROOT"

exec > >(tee -a "$LOG") 2>&1


# ============================================================
# Basic checks
# ============================================================

echo "============================================================"
echo "HGV-V3 PHASE 2B"
echo "NCBI genome download"
echo "============================================================"
echo
echo "Start: $(date '+%F %T')"
echo "BASE: $BASE"
echo


if [[ ! -d "$FREEZE" ]]; then
    echo "ERROR: frozen cohort directory missing:"
    echo "$FREEZE"
    exit 1
fi


if [[ ! -s "$FREEZE/PRE_DOWNLOAD_FREEZE_SHA256.txt" ]]; then
    echo "ERROR: freeze checksum file missing."
    exit 1
fi


echo "[CHECK] Verifying frozen cohort..."

(
    cd "$FREEZE"
    sha256sum -c PRE_DOWNLOAD_FREEZE_SHA256.txt
)

echo "[PASS] Frozen metadata cohort is unchanged."


# ============================================================
# Software
# ============================================================

for cmd in \
    datasets \
    unzip \
    gzip \
    sha256sum \
    sort \
    comm \
    find \
    awk \
    grep \
    wc \
    basename \
    dirname
do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: required command missing: $cmd"
        exit 1
    fi
done


if (( WORKERS < 1 || WORKERS > 30 )); then
    echo "ERROR: HGV_REHYDRATE_WORKERS must be 1-30."
    exit 1
fi


echo
echo "[INFO] NCBI Datasets:"
datasets --version

echo
echo "[INFO] Rehydrate workers: $WORKERS"
echo "[INFO] gzip integrity workers: $GZIP_CHECK_WORKERS"


# ============================================================
# Provenance
# ============================================================

{
    echo "HGV-V3 genome download provenance"
    echo "Date: $(date '+%F %T')"
    echo
    echo "Datasets CLI:"
    datasets --version
    echo
    echo "Rehydrate workers: $WORKERS"
    echo "gzip check workers: $GZIP_CHECK_WORKERS"
    echo
    if [[ -n "${NCBI_API_KEY:-}" ]]; then
        echo "NCBI API key configured: yes"
    else
        echo "NCBI API key configured: no"
    fi
    echo
    echo "Download strategy:"
    echo "datasets download genome accession --inputfile ... --dehydrated --include genome"
    echo "unzip"
    echo "datasets rehydrate --gzip"
    echo
    echo "Frozen accession source:"
    echo "$FREEZE"
} > "$PROVENANCE"


# ============================================================
# Summary table
# ============================================================

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "cohort" \
    "expected" \
    "requested_unique" \
    "fasta_files" \
    "downloaded_accessions" \
    "missing" \
    "extra" \
    "status" \
    > "$SUMMARY"


GLOBAL_AUDIT_FAIL=0


# ============================================================
# Download / audit function
# ============================================================

download_cohort() {

    local LABEL="$1"
    local ACCESSION_LIST="$2"
    local EXPECTED="$3"
    local OUTDIR="$4"

    local AUDIT="$OUTDIR/audit"
    local ZIP="$OUTDIR/${LABEL}_dehydrated.zip"
    local PKG="$OUTDIR/package"
    local COMPLETE_FLAG="$OUTDIR/REHYDRATE_COMPLETE.flag"

    local REQUESTED_SORTED="$AUDIT/requested_accessions.sorted.txt"
    local DOWNLOADED_SORTED="$AUDIT/downloaded_accessions.sorted.txt"

    local FASTA_LIST="$AUDIT/genomic_fasta_files.txt"
    local MISSING="$AUDIT/missing_accessions.txt"
    local EXTRA="$AUDIT/extra_accessions.txt"

    mkdir -p "$OUTDIR"
    mkdir -p "$AUDIT"
    mkdir -p "$PKG"

    echo
    echo "============================================================"
    echo "COHORT: $LABEL"
    echo "Expected genomes: $EXPECTED"
    echo "Input list: $ACCESSION_LIST"
    echo "Output: $OUTDIR"
    echo "============================================================"


    # --------------------------------------------------------
    # Validate requested accession list
    # --------------------------------------------------------

    if [[ ! -s "$ACCESSION_LIST" ]]; then
        echo "ERROR: accession list missing or empty:"
        echo "$ACCESSION_LIST"
        exit 1
    fi


    grep -v '^[[:space:]]*$' "$ACCESSION_LIST" \
        | sort -u \
        > "$REQUESTED_SORTED"


    local REQUESTED_N

    REQUESTED_N=$(
        wc -l < "$REQUESTED_SORTED"
    )

    echo
    echo "Requested unique accessions: $REQUESTED_N"


    if [[ "$REQUESTED_N" -ne "$EXPECTED" ]]; then

        echo "ERROR: requested accession count mismatch."
        echo "Expected: $EXPECTED"
        echo "Observed: $REQUESTED_N"

        exit 1
    fi


    # --------------------------------------------------------
    # Dehydrated package
    # --------------------------------------------------------

    echo
    echo "[1/4] Preparing dehydrated NCBI package..."


    if [[ -f "$ZIP" ]]; then

        echo "Existing dehydrated archive found."

        if unzip -tq "$ZIP" >/dev/null 2>&1; then

            echo "[PASS] Existing archive passes unzip integrity test."

        else

            echo "WARNING: existing archive is corrupt."
            echo "Moving it aside."

            mv \
                "$ZIP" \
                "${ZIP}.corrupt.$(date '+%Y%m%d_%H%M%S')"
        fi
    fi


    if [[ ! -f "$ZIP" ]]; then

        echo "Downloading dehydrated package..."

        datasets download genome accession \
            --inputfile "$ACCESSION_LIST" \
            --dehydrated \
            --include genome \
            --filename "$ZIP"

        echo "Dehydrated archive downloaded."
    else
        echo "Skipping dehydrated download."
    fi


    if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
        echo "ERROR: dehydrated zip validation failed:"
        echo "$ZIP"
        exit 1
    fi

    echo "[PASS] Dehydrated zip integrity OK."


    # --------------------------------------------------------
    # Extract metadata package
    #
    # Safe on reruns: does not delete previously rehydrated
    # genome files.
    # --------------------------------------------------------

    echo
    echo "[2/4] Extracting / refreshing package metadata..."

    unzip -oq \
        "$ZIP" \
        -d "$PKG"


    local FETCH_FILE

    FETCH_FILE=$(
        find "$PKG" \
            -type f \
            -name "fetch.txt" \
            -print \
            -quit
    )


    if [[ -z "$FETCH_FILE" ]]; then

        echo "ERROR: fetch.txt not found after dehydrated extraction."
        echo "Package: $PKG"

        exit 1
    fi


    echo "fetch.txt:"
    echo "$FETCH_FILE"

    echo "Files scheduled by fetch.txt:"
    wc -l "$FETCH_FILE"


    # --------------------------------------------------------
    # Rehydrate
    # --------------------------------------------------------

    echo
    echo "[3/4] Rehydrating genomic FASTA files..."


    if [[ ! -f "$COMPLETE_FLAG" ]]; then

        datasets rehydrate \
            --directory "$PKG" \
            --gzip \
            --max-workers "$WORKERS"

        {
            echo "Rehydrate completed successfully."
            echo "Date: $(date '+%F %T')"
            echo "Datasets CLI: $(datasets --version)"
            echo "Workers: $WORKERS"
        } > "$COMPLETE_FLAG"

    else

        echo "Existing successful rehydrate flag found:"
        echo "$COMPLETE_FLAG"
        echo
        echo "Skipping rehydrate."

    fi


    # --------------------------------------------------------
    # Locate package data directory
    # --------------------------------------------------------

    local DATA_DIR

    DATA_DIR=$(
        find "$PKG" \
            -type d \
            -path "*/ncbi_dataset/data" \
            -print \
            -quit
    )


    if [[ -z "$DATA_DIR" ]]; then

        echo "ERROR: ncbi_dataset/data directory not found."

        exit 1
    fi


    echo
    echo "NCBI data directory:"
    echo "$DATA_DIR"


    # --------------------------------------------------------
    # Find FASTA
    # --------------------------------------------------------

    find "$DATA_DIR" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        \( \
            -name "*_genomic.fna.gz" \
            -o \
            -name "*_genomic.fna" \
        \) \
        -print \
        | sort \
        > "$FASTA_LIST"


    local FASTA_N

    FASTA_N=$(
        wc -l < "$FASTA_LIST"
    )


    echo
    echo "Genomic FASTA files found: $FASTA_N"


    # --------------------------------------------------------
    # gzip integrity
    # --------------------------------------------------------

    echo
    echo "Testing compressed FASTA integrity..."


    local GZIP_OK=1

    local GZ_LIST="$AUDIT/gzipped_fasta_files.txt"

    grep '\.gz$' "$FASTA_LIST" \
        > "$GZ_LIST" \
        || true


    if [[ -s "$GZ_LIST" ]]; then

        if ! xargs \
            -r \
            -P "$GZIP_CHECK_WORKERS" \
            -n 1 \
            gzip -t \
            < "$GZ_LIST"
        then
            GZIP_OK=0
        fi
    fi


    if [[ "$GZIP_OK" -eq 1 ]]; then
        echo "[PASS] gzip integrity test."
    else
        echo "[FAIL] One or more gzip FASTA files are corrupt."
    fi


    # --------------------------------------------------------
    # Recover downloaded accessions from directory names
    # --------------------------------------------------------

    : > "$DOWNLOADED_SORTED"


    while IFS= read -r fasta; do

        [[ -z "$fasta" ]] && continue

        basename \
            "$(dirname "$fasta")"

    done < "$FASTA_LIST" \
        | sort -u \
        > "$DOWNLOADED_SORTED"


    local DOWNLOADED_N

    DOWNLOADED_N=$(
        wc -l < "$DOWNLOADED_SORTED"
    )


    echo "Unique accession directories with FASTA: $DOWNLOADED_N"


    # --------------------------------------------------------
    # Requested vs downloaded
    # --------------------------------------------------------

    comm -23 \
        "$REQUESTED_SORTED" \
        "$DOWNLOADED_SORTED" \
        > "$MISSING"


    comm -13 \
        "$REQUESTED_SORTED" \
        "$DOWNLOADED_SORTED" \
        > "$EXTRA"


    local MISSING_N
    local EXTRA_N

    MISSING_N=$(
        wc -l < "$MISSING"
    )

    EXTRA_N=$(
        wc -l < "$EXTRA"
    )


    echo
    echo "Missing accessions: $MISSING_N"
    echo "Extra accessions:   $EXTRA_N"


    if [[ "$MISSING_N" -gt 0 ]]; then

        echo
        echo "First missing accessions:"

        head -n 20 "$MISSING"
    fi


    if [[ "$EXTRA_N" -gt 0 ]]; then

        echo
        echo "First unexpected accessions:"

        head -n 20 "$EXTRA"
    fi


    # --------------------------------------------------------
    # Data report
    # --------------------------------------------------------

    local REPORT_FILE

    REPORT_FILE="$DATA_DIR/assembly_data_report.jsonl"


    if [[ -s "$REPORT_FILE" ]]; then

        echo
        echo "NCBI assembly-data report records:"
        wc -l "$REPORT_FILE"

    else

        echo
        echo "WARNING: assembly_data_report.jsonl not found."
    fi


    # --------------------------------------------------------
    # Cohort audit status
    # --------------------------------------------------------

    local STATUS="PASS"


    if [[ "$REQUESTED_N" -ne "$EXPECTED" ]]; then
        STATUS="FAIL"
    fi

    if [[ "$FASTA_N" -ne "$EXPECTED" ]]; then
        STATUS="FAIL"
    fi

    if [[ "$DOWNLOADED_N" -ne "$EXPECTED" ]]; then
        STATUS="FAIL"
    fi

    if [[ "$MISSING_N" -ne 0 ]]; then
        STATUS="FAIL"
    fi

    if [[ "$EXTRA_N" -ne 0 ]]; then
        STATUS="FAIL"
    fi

    if [[ "$GZIP_OK" -ne 1 ]]; then
        STATUS="FAIL"
    fi


    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$LABEL" \
        "$EXPECTED" \
        "$REQUESTED_N" \
        "$FASTA_N" \
        "$DOWNLOADED_N" \
        "$MISSING_N" \
        "$EXTRA_N" \
        "$STATUS" \
        >> "$SUMMARY"


    echo
    echo "----------------------------------------"
    echo "$LABEL download audit: $STATUS"
    echo "----------------------------------------"


    if [[ "$STATUS" != "PASS" ]]; then
        GLOBAL_AUDIT_FAIL=1
    fi
}


# ============================================================
# Cohort accession lists from immutable freeze
# ============================================================

A_LIST="$FREEZE/development_A_primary_accessions.txt"

B1_LIST="$FREEZE/external_B1_primary_accessions.txt"

B2_SEEN_LIST="$FREEZE/external_B2_seen_genus_primary_accessions.txt"

B2_UNSEEN_LIST="$FREEZE/external_B2_unseen_genus_primary_accessions.txt"


# ============================================================
# Sequential downloads
#
# Keep cohorts separate for provenance and replacement audits.
# ============================================================

download_cohort \
    "A_development" \
    "$A_LIST" \
    2400 \
    "$DOWNLOAD_ROOT/development/ncbi_download"


download_cohort \
    "B1_external" \
    "$B1_LIST" \
    400 \
    "$DOWNLOAD_ROOT/external_B1/ncbi_download"


download_cohort \
    "B2_seen_genus" \
    "$B2_SEEN_LIST" \
    200 \
    "$DOWNLOAD_ROOT/external_B2/seen_genus/ncbi_download"


download_cohort \
    "B2_unseen_genus" \
    "$B2_UNSEEN_LIST" \
    200 \
    "$DOWNLOAD_ROOT/external_B2/unseen_genus/ncbi_download"


# ============================================================
# Final download report
# ============================================================

echo
echo "============================================================"
echo "HGV-V3 DOWNLOAD RECONCILIATION"
echo "============================================================"
echo

column -t -s $'\t' "$SUMMARY" || cat "$SUMMARY"

echo


if [[ "$GLOBAL_AUDIT_FAIL" -ne 0 ]]; then

    echo "============================================================"
    echo "PHASE 2B DOWNLOAD GATE: INCOMPLETE"
    echo
    echo "At least one cohort has missing, extra, or corrupt FASTA."
    echo
    echo "DO NOT START FEATURE EXTRACTION."
    echo
    echo "Next action:"
    echo "inspect cohort audit/missing_accessions.txt files"
    echo "and replace only failed accessions from the frozen reserve."
    echo "============================================================"

    exit 2

fi


echo "============================================================"
echo "PHASE 2B DOWNLOAD GATE: PASS"
echo
echo "All 3,200 requested primary assemblies have genomic FASTA."
echo "Requested/downloaded accession sets match exactly."
echo "Compressed FASTA integrity checks passed."
echo
echo "Next mandatory stage:"
echo "sequence-level normalization + SHA256 duplicate audit."
echo "============================================================"

exit 0
