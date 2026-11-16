#!/usr/bin/env bash
set -Eeuo pipefail

BASE="HGV_V3_revision_20260915"

R="$BASE/01_manifest/replacement_round1_20260916"

LIST="$R/replacement_accessions_round1.txt"
PLAN="$R/replacement_plan_round1.tsv"
UPDATED="$R/updated_primary_manifest_round1.tsv"
SHA_TABLE="$R/replacement_round1_sha256.tsv"

OUT="$BASE/02_genomes/replacement_round1"
PKG="$OUT/package"
AUDIT="$OUT/audit"
ZIP="$OUT/replacement_round1_dehydrated.zip"

LOG="$BASE/logs/07c_download_replacements_round1.log"

WORKERS="${HGV_REHYDRATE_WORKERS:-9}"

mkdir -p "$OUT"
mkdir -p "$PKG"
mkdir -p "$AUDIT"

exec > >(tee -a "$LOG") 2>&1


echo "============================================================"
echo "HGV-V3 REPLACEMENT ROUND 1 DOWNLOAD"
echo "============================================================"
echo
echo "Start: $(date '+%F %T')"
echo


# ============================================================
# 1. Verify replacement-plan provenance
# ============================================================

for f in \
    "$LIST" \
    "$PLAN" \
    "$UPDATED" \
    "$SHA_TABLE"
do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing replacement input:"
        echo "$f"
        exit 1
    fi
done


echo "[1/7] Verifying replacement-plan SHA256..."

while IFS=$'\t' read -r expected file; do

    [[ "$expected" == "sha256" ]] && continue
    [[ -z "$expected" ]] && continue

    if [[ ! -f "$file" ]]; then
        echo "ERROR: recorded file missing:"
        echo "$file"
        exit 1
    fi

    actual=$(
        sha256sum "$file" |
        awk '{print $1}'
    )

    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: replacement-plan file changed:"
        echo "$file"
        echo "Expected: $expected"
        echo "Observed: $actual"
        exit 1
    fi

done < "$SHA_TABLE"

echo "[PASS] Replacement-plan provenance unchanged."


# ============================================================
# 2. Validate accession list
# ============================================================

echo
echo "[2/7] Validating replacement accession list..."

sort -u "$LIST" \
    > "$AUDIT/requested_accessions.sorted.txt"

REQUESTED_N=$(
    wc -l \
    < "$AUDIT/requested_accessions.sorted.txt"
)

if [[ "$REQUESTED_N" -ne 9 ]]; then
    echo "ERROR: expected 9 replacement accessions."
    echo "Observed: $REQUESTED_N"
    exit 1
fi

echo "[PASS] 9 unique replacement accessions."


# ============================================================
# 3. Software check
# ============================================================

echo
echo "[3/7] Checking software..."

for cmd in \
    datasets \
    unzip \
    gzip \
    find \
    sort \
    comm \
    sha256sum \
    awk
do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: missing command: $cmd"
        exit 1
    fi
done

datasets --version


# ============================================================
# 4. Download dehydrated replacement package
# ============================================================

echo
echo "[4/7] Preparing dehydrated package..."

if [[ -f "$ZIP" ]]; then

    if unzip -tq "$ZIP" >/dev/null 2>&1; then
        echo "Existing replacement ZIP is valid."
    else
        echo "Existing ZIP is corrupt; moving aside."

        mv \
            "$ZIP" \
            "${ZIP}.corrupt.$(date '+%Y%m%d_%H%M%S')"
    fi
fi


if [[ ! -f "$ZIP" ]]; then

    datasets download genome accession \
        --inputfile "$LIST" \
        --dehydrated \
        --include genome \
        --filename "$ZIP"
fi


unzip -tq "$ZIP"

echo "[PASS] Dehydrated replacement ZIP valid."


# ============================================================
# 5. Extract and rehydrate
# ============================================================

echo
echo "[5/7] Rehydrating replacement genomes..."

unzip -oq \
    "$ZIP" \
    -d "$PKG"


FETCH=$(
    find "$PKG" \
        -type f \
        -name "fetch.txt" \
        -print \
        -quit
)


if [[ -z "$FETCH" ]]; then
    echo "ERROR: fetch.txt not found."
    exit 1
fi


echo "Files scheduled by NCBI:"
wc -l "$FETCH"


datasets rehydrate \
    --directory "$PKG" \
    --gzip \
    --max-workers "$WORKERS"


# ============================================================
# 6. FASTA reconciliation
# ============================================================

echo
echo "[6/7] Reconciling FASTA files..."

DATA_DIR=$(
    find "$PKG" \
        -type d \
        -path "*/ncbi_dataset/data" \
        -print \
        -quit
)


if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: NCBI data directory not found."
    exit 1
fi


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
    > "$AUDIT/genomic_fasta_files.txt"


FASTA_N=$(
    wc -l \
    < "$AUDIT/genomic_fasta_files.txt"
)


: > "$AUDIT/downloaded_accessions.sorted.txt"


while IFS= read -r fasta; do

    [[ -z "$fasta" ]] && continue

    basename \
        "$(dirname "$fasta")"

done < "$AUDIT/genomic_fasta_files.txt" \
    | sort -u \
    > "$AUDIT/downloaded_accessions.sorted.txt"


DOWNLOADED_N=$(
    wc -l \
    < "$AUDIT/downloaded_accessions.sorted.txt"
)


comm -23 \
    "$AUDIT/requested_accessions.sorted.txt" \
    "$AUDIT/downloaded_accessions.sorted.txt" \
    > "$AUDIT/missing_accessions.txt"


comm -13 \
    "$AUDIT/requested_accessions.sorted.txt" \
    "$AUDIT/downloaded_accessions.sorted.txt" \
    > "$AUDIT/extra_accessions.txt"


MISSING_N=$(
    wc -l \
    < "$AUDIT/missing_accessions.txt"
)

EXTRA_N=$(
    wc -l \
    < "$AUDIT/extra_accessions.txt"
)


echo
echo "Requested:  $REQUESTED_N"
echo "FASTA:      $FASTA_N"
echo "Downloaded: $DOWNLOADED_N"
echo "Missing:    $MISSING_N"
echo "Extra:      $EXTRA_N"


# ============================================================
# gzip integrity
# ============================================================

echo
echo "Testing gzip integrity..."

GZ_N=0
GZIP_FAIL=0


while IFS= read -r fasta; do

    [[ -z "$fasta" ]] && continue

    if [[ "$fasta" == *.gz ]]; then

        GZ_N=$((GZ_N + 1))

        if ! gzip -t "$fasta"; then
            echo "CORRUPT: $fasta"
            GZIP_FAIL=$((GZIP_FAIL + 1))
        fi
    fi

done < "$AUDIT/genomic_fasta_files.txt"


echo "Compressed FASTA checked: $GZ_N"
echo "gzip failures: $GZIP_FAIL"


# ============================================================
# 7. Final gate
# ============================================================

echo
echo "[7/7] Replacement download gate..."


STATUS="PASS"


if [[ "$REQUESTED_N" -ne 9 ]]; then
    STATUS="FAIL"
fi

if [[ "$FASTA_N" -ne 9 ]]; then
    STATUS="FAIL"
fi

if [[ "$DOWNLOADED_N" -ne 9 ]]; then
    STATUS="FAIL"
fi

if [[ "$MISSING_N" -ne 0 ]]; then
    STATUS="FAIL"
fi

if [[ "$EXTRA_N" -ne 0 ]]; then
    STATUS="FAIL"
fi

if [[ "$GZIP_FAIL" -ne 0 ]]; then
    STATUS="FAIL"
fi


cat > "$AUDIT/replacement_download_summary.txt" <<EOF
HGV-V3 replacement round 1 download

Requested replacement accessions: $REQUESTED_N
Genomic FASTA files: $FASTA_N
Downloaded unique accessions: $DOWNLOADED_N
Missing accessions: $MISSING_N
Extra accessions: $EXTRA_N
gzip integrity failures: $GZIP_FAIL

Status: $STATUS
EOF


echo
cat "$AUDIT/replacement_download_summary.txt"


if [[ "$STATUS" != "PASS" ]]; then

    echo
    echo "============================================================"
    echo "REPLACEMENT ROUND 1 DOWNLOAD GATE: INCOMPLETE"
    echo "============================================================"

    if [[ "$MISSING_N" -gt 0 ]]; then
        echo
        echo "Missing replacement accessions:"
        cat "$AUDIT/missing_accessions.txt"
    fi

    echo
    echo "Do NOT modify the original frozen cohort."
    echo "Do NOT start feature extraction."
    echo "A round-2 reserve replacement must be planned."

    exit 2
fi


sha256sum \
    "$AUDIT/genomic_fasta_files.txt" \
    "$AUDIT/downloaded_accessions.sorted.txt" \
    > "$AUDIT/replacement_download_audit_sha256.txt"


echo
echo "============================================================"
echo "REPLACEMENT ROUND 1 DOWNLOAD GATE: PASS"
echo
echo "All 9 replacement genomes have genomic FASTA."
echo "No missing or extra replacement accession."
echo "FASTA compression integrity passed."
echo
echo "Logical primary cohort can now return to:"
echo "3191 original available + 9 replacements = 3200"
echo
echo "NEXT:"
echo "sequence-level integrity and normalized SHA256 audit"
echo "============================================================"
