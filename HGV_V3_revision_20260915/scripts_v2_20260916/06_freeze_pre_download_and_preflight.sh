#!/usr/bin/env bash
set -Eeuo pipefail

BASE="HGV_V3_revision_20260915"

SEL="$BASE/01_manifest/cohort_selection_v2"
AUDIT="$SEL/audit_v2"

FREEZE="$BASE/01_manifest/frozen_pre_download_20260916"

PRIMARY="$SEL/selected_cohorts_primary_v2.tsv"

echo "============================================================"
echo "HGV-V3 PHASE 2A"
echo "Pre-download freeze + environment preflight"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Verify final audit
# ------------------------------------------------------------

REPORT="$AUDIT/selected_cohort_audit_report.txt"
CHECKS="$AUDIT/selected_cohort_audit_checks.tsv"
SHA_TABLE="$AUDIT/selected_file_sha256.tsv"

for f in "$REPORT" "$CHECKS" "$SHA_TABLE"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: missing audit file:"
        echo "$f"
        exit 1
    fi
done

if ! grep -q \
    "FINAL PRE-DOWNLOAD METADATA GATE: PASS" \
    "$REPORT"; then

    echo "ERROR: metadata gate is not PASS."
    exit 1
fi

FAIL_N=$(
    awk -F'\t' '
        NR > 1 && $2 == "FAIL" {n++}
        END {print n+0}
    ' "$CHECKS"
)

if [[ "$FAIL_N" -ne 0 ]]; then
    echo "ERROR: audit contains $FAIL_N failed checks."
    exit 1
fi

echo "[PASS] 64-check metadata audit is PASS."


# ------------------------------------------------------------
# 2. Re-check SHA256 of audited selection
# ------------------------------------------------------------

echo
echo "[CHECK] Re-verifying audited selection SHA256..."

while IFS=$'\t' read -r expected file; do

    [[ "$expected" == "sha256" ]] && continue
    [[ -z "$expected" ]] && continue

    if [[ ! -f "$file" ]]; then
        echo "ERROR: audited file is missing:"
        echo "$file"
        exit 1
    fi

    actual=$(
        sha256sum "$file" |
        awk '{print $1}'
    )

    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: SHA256 changed:"
        echo "File     : $file"
        echo "Expected : $expected"
        echo "Observed : $actual"
        exit 1
    fi

done < "$SHA_TABLE"

echo "[PASS] All audited files retain their recorded SHA256."


# ------------------------------------------------------------
# 3. Exact accession counts
# ------------------------------------------------------------

A_LIST="$SEL/development_A_primary_accessions.txt"
B1_LIST="$SEL/external_B1_primary_accessions.txt"
B2S_LIST="$SEL/external_B2_seen_genus_primary_accessions.txt"
B2U_LIST="$SEL/external_B2_unseen_genus_primary_accessions.txt"
ALL_LIST="$SEL/all_primary_accessions.txt"

check_count() {

    local file="$1"
    local expected="$2"
    local observed

    observed=$(grep -cve '^[[:space:]]*$' "$file")

    if [[ "$observed" -ne "$expected" ]]; then

        echo "ERROR: accession count mismatch"
        echo "$file"
        echo "Expected: $expected"
        echo "Observed: $observed"

        exit 1
    fi
}

check_count "$A_LIST"   2400
check_count "$B1_LIST"   400
check_count "$B2S_LIST"  200
check_count "$B2U_LIST"  200
check_count "$ALL_LIST" 3200

echo "[PASS] Accession counts = 2400 + 400 + 200 + 200 = 3200."


# ------------------------------------------------------------
# 4. Verify required download software
# ------------------------------------------------------------

echo
echo "[CHECK] Required software..."

for cmd in datasets unzip sha256sum awk grep find sort comm; do

    if ! command -v "$cmd" >/dev/null 2>&1; then

        echo "ERROR: command not found: $cmd"

        if [[ "$cmd" == "datasets" ]]; then
            echo
            echo "Install into hgv_env with:"
            echo "conda install -n hgv_env -c conda-forge ncbi-datasets-cli -y"
        fi

        exit 1
    fi

done

echo
echo "NCBI Datasets version:"
datasets --version

echo
echo "[PASS] Required software available."


# ------------------------------------------------------------
# 5. Report expected raw genome volume from manifest
# ------------------------------------------------------------

echo
echo "[INFO] Selected genome-size total..."

awk -F'\t' '
NR==1 {
    for(i=1;i<=NF;i++) {
        if($i=="genome_size") c=i
    }
    next
}
{
    sum += $c
}
END {
    printf("Total selected bases: %.0f\n", sum)
    printf("Approx raw FASTA sequence: %.2f GB\n", sum/1e9)
}
' "$PRIMARY"


# ------------------------------------------------------------
# 6. Disk information
# ------------------------------------------------------------

echo
echo "[INFO] Disk space:"
df -h "$BASE"


# ------------------------------------------------------------
# 7. Create immutable pre-download freeze
# ------------------------------------------------------------

echo
echo "[FREEZE] Creating pre-download selection snapshot..."

if [[ -e "$FREEZE" ]]; then

    echo "ERROR: freeze directory already exists:"
    echo "$FREEZE"
    echo
    echo "It will NOT be overwritten automatically."
    exit 2
fi

mkdir -p "$FREEZE"

cp -a \
    "$SEL/selected_cohorts_primary_v2.tsv" \
    "$SEL/selected_cohorts_reserve_v2.tsv" \
    "$SEL/cohort_species_design_v2.tsv" \
    "$SEL/development_A_primary_accessions.txt" \
    "$SEL/external_B1_primary_accessions.txt" \
    "$SEL/external_B2_seen_genus_primary_accessions.txt" \
    "$SEL/external_B2_unseen_genus_primary_accessions.txt" \
    "$SEL/external_B2_primary_accessions.txt" \
    "$SEL/all_primary_accessions.txt" \
    "$SEL/all_reserve_accessions.txt" \
    "$FREEZE/"

cp -a \
    "$AUDIT/selected_cohort_audit_report.txt" \
    "$AUDIT/selected_cohort_audit_checks.tsv" \
    "$AUDIT/selected_taxonomic_structure.tsv" \
    "$AUDIT/selected_file_sha256.tsv" \
    "$FREEZE/"

cat > "$FREEZE/FREEZE_INFO.txt" <<EOF
Project: HGV-V3 Major Revision
Freeze type: pre-download metadata freeze
Date: 2026-09-16

Primary cohort:
A  = 2400 GCF genomes / 200 species
B1 = 400 independent GCA genomes / 40 A-seen species
B2 = 400 independent GCA genomes / 40 A-unseen species
     200 seen-genus
     200 unseen-genus

Total primary genomes: 3200

Metadata audit:
64 PASS
0 FAIL

Important:
This freeze defines the originally selected metadata cohort.
It must not be modified based on downstream model performance.

Sequence-level duplicate checking is still pending.

If a genome is unavailable or fails sequence-level integrity checks,
replacement must come from the pre-selected reserve cohort and must
be documented in the replacement ledger.
EOF

(
    cd "$FREEZE"

    sha256sum \
        selected_cohorts_primary_v2.tsv \
        selected_cohorts_reserve_v2.tsv \
        cohort_species_design_v2.tsv \
        development_A_primary_accessions.txt \
        external_B1_primary_accessions.txt \
        external_B2_seen_genus_primary_accessions.txt \
        external_B2_unseen_genus_primary_accessions.txt \
        external_B2_primary_accessions.txt \
        all_primary_accessions.txt \
        all_reserve_accessions.txt \
        > PRE_DOWNLOAD_FREEZE_SHA256.txt
)

chmod -R a-w "$FREEZE"

echo
echo "[PASS] Immutable pre-download freeze created:"
echo "$FREEZE"


# ------------------------------------------------------------
# 8. Prepare download directories
# ------------------------------------------------------------

mkdir -p \
    "$BASE/02_genomes/development" \
    "$BASE/02_genomes/external_B1" \
    "$BASE/02_genomes/external_B2/seen_genus" \
    "$BASE/02_genomes/external_B2/unseen_genus"

echo
echo "============================================================"
echo "PHASE 2A PRE-DOWNLOAD GATE: PASS"
echo
echo "Metadata selection is frozen."
echo "Genome download may begin."
echo "============================================================"
