#!/usr/bin/env bash
set -uo pipefail

BASE=HGV_V3_revision_20260915

MANIFEST="$BASE/01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv"

OUT="$BASE/06_teacher/tool_preflight_v1_20260916"

if [ -e "$OUT" ]; then
    echo "ERROR: output already exists:"
    echo "$OUT"
    echo "Refusing to overwrite."
    exit 2
fi

mkdir -p "$OUT"

exec > >(tee "$OUT/teacher_tool_preflight.log") 2>&1

echo "=============================================================================="
echo "HGV-V3 TEACHER TOOL PREFLIGHT"
echo "=============================================================================="
echo

# ----------------------------------------------------------------------
# 1. Basic environment
# ----------------------------------------------------------------------

echo "[1] ENVIRONMENT"

{
    echo "DATE"
    date -Is
    echo
    echo "HOST"
    hostname
    echo
    echo "KERNEL"
    uname -a
    echo
    echo "CPU"
    lscpu
    echo
    echo "MEMORY"
    free -h
    echo
    echo "NPROC"
    nproc
    echo
    echo "CONDA_PREFIX"
    echo "${CONDA_PREFIX:-}"
    echo
    echo "CONDA_DEFAULT_ENV"
    echo "${CONDA_DEFAULT_ENV:-}"
} > "$OUT/environment.txt"

echo "environment.txt written"

# ----------------------------------------------------------------------
# 2. Tool existence
# ----------------------------------------------------------------------

echo
echo "[2] TOOL AVAILABILITY"

FAIL=0

for TOOL in mash dashing2 skani; do
    if command -v "$TOOL" >/dev/null 2>&1; then
        echo "FOUND $TOOL: $(command -v "$TOOL")"
    else
        echo "MISSING $TOOL"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo
    echo "TEACHER_TOOL_PREFLIGHT=FAIL"
    echo "One or more required tools are missing."
    exit 3
fi

# ----------------------------------------------------------------------
# 3. Versions
# ----------------------------------------------------------------------

echo
echo "[3] TOOL VERSIONS"

{
    echo "===== MASH ====="
    mash --version 2>&1 || mash -v 2>&1 || true

    echo
    echo "===== DASHING2 ====="
    dashing2 --version 2>&1 || dashing2 -V 2>&1 || true

    echo
    echo "===== SKANI ====="
    skani --version 2>&1 || skani -V 2>&1 || true
} | tee "$OUT/tool_versions.txt"

# ----------------------------------------------------------------------
# 4. Capture actual help from THIS installation
# ----------------------------------------------------------------------

echo
echo "[4] CAPTURE LOCAL HELP"

mash sketch -h > "$OUT/mash_sketch_help.txt" 2>&1 || true
mash dist -h > "$OUT/mash_dist_help.txt" 2>&1 || true

dashing2 --help > "$OUT/dashing2_help.txt" 2>&1 || true
dashing2 sketch --help > "$OUT/dashing2_sketch_help.txt" 2>&1 || true

skani dist --help > "$OUT/skani_dist_help.txt" 2>&1 || true
skani sketch --help > "$OUT/skani_sketch_help.txt" 2>&1 || true

echo "Help captured."

echo
echo "===== DASHING2 RELEVANT OPTIONS ====="

grep -iE \
'canonical|multiset|bagminhash|setsketch|set sketch|cache|cmpout|sketch.size|threads|^-p| -p|--size| -S' \
"$OUT/dashing2_sketch_help.txt" \
> "$OUT/dashing2_relevant_options.txt" || true

cat "$OUT/dashing2_relevant_options.txt" || true

# ----------------------------------------------------------------------
# 5. Select three REAL Development-A genomes
#
# genome_1 + genome_2: same GTDB species cluster
# genome_3: preferably different family from genome_1
# ----------------------------------------------------------------------

echo
echo "[5] SELECT REAL SMOKE-TEST GENOMES"

python3 - "$MANIFEST" "$OUT/smoke_manifest.tsv" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])

with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = [
        r for r in reader
        if r.get("final_cohort", "").strip() == "A"
    ]

if not rows:
    raise SystemExit("No Development A genomes found")

groups = defaultdict(list)

for r in rows:
    sp = r["gtdb_species_cluster"].strip()
    groups[sp].append(r)

same = None

for sp in sorted(groups):
    members = sorted(
        groups[sp],
        key=lambda r: r["assembly_accession_final"]
    )
    if len(members) >= 2:
        same = members[:2]
        break

if same is None:
    raise SystemExit("Could not find same-species pair")

a, b = same

# Prefer different family for a clearly distant smoke-control genome.
c = None

for r in sorted(rows, key=lambda x: x["assembly_accession_final"]):
    if r["gtdb_family"].strip() != a["gtdb_family"].strip():
        c = r
        break

# Fallback: different genus.
if c is None:
    for r in sorted(rows, key=lambda x: x["assembly_accession_final"]):
        if r["gtdb_genus"].strip() != a["gtdb_genus"].strip():
            c = r
            break

# Final fallback: different species.
if c is None:
    for r in sorted(rows, key=lambda x: x["assembly_accession_final"]):
        if r["gtdb_species_cluster"].strip() != a["gtdb_species_cluster"].strip():
            c = r
            break

if c is None:
    raise SystemExit("Could not find cross-taxonomic genome")

selected = [
    ("genome_1", a, "reference"),
    ("genome_2", b, "same_species_as_genome_1"),
    ("genome_3", c, "taxonomically_distant_from_genome_1"),
]

fields = [
    "smoke_id",
    "assembly_accession",
    "gtdb_species_cluster",
    "gtdb_species",
    "gtdb_genus",
    "gtdb_family",
    "relationship",
    "fasta_path",
]

with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        delimiter="\t",
        fieldnames=fields,
        lineterminator="\n",
    )
    w.writeheader()

    for smoke_id, r, relationship in selected:
        w.writerow({
            "smoke_id": smoke_id,
            "assembly_accession": r["assembly_accession_final"].strip(),
            "gtdb_species_cluster": r["gtdb_species_cluster"].strip(),
            "gtdb_species": r["gtdb_species"].strip(),
            "gtdb_genus": r["gtdb_genus"].strip(),
            "gtdb_family": r["gtdb_family"].strip(),
            "relationship": relationship,
            "fasta_path": r["fasta_path"].strip(),
        })

print("SMOKE_SELECTION=PASS")

for smoke_id, r, relationship in selected:
    print(
        smoke_id,
        r["assembly_accession_final"],
        r["gtdb_species_cluster"],
        r["gtdb_genus"],
        r["gtdb_family"],
        relationship,
        sep="\t",
    )
PY

mapfile -t FASTAS < <(
    awk -F'\t' 'NR > 1 {print $8}' \
    "$OUT/smoke_manifest.tsv"
)

if [ "${#FASTAS[@]}" -ne 3 ]; then
    echo "ERROR: expected 3 FASTA paths"
    exit 4
fi

for F in "${FASTAS[@]}"; do
    if [ ! -s "$F" ]; then
        echo "ERROR: missing FASTA: $F"
        exit 5
    fi
done

# ----------------------------------------------------------------------
# 6. GZIP integrity of selected FASTAs
# ----------------------------------------------------------------------

echo
echo "[6] SMOKE FASTA GZIP TEST"

GZIP_FAIL=0

for F in "${FASTAS[@]}"; do
    if gzip -t "$F" 2>/dev/null; then
        echo "PASS $F"
    else
        echo "FAIL $F"
        GZIP_FAIL=1
    fi
done

if [ "$GZIP_FAIL" -ne 0 ]; then
    echo "TEACHER_TOOL_PREFLIGHT=FAIL"
    exit 6
fi

# ----------------------------------------------------------------------
# 7. Mash smoke test
#
# Production preregistration:
#   k = 21
#   sketch size = 1000
# ----------------------------------------------------------------------

echo
echo "[7] MASH SMOKE TEST"

MASH_STATUS=0

mash sketch \
    -k 21 \
    -s 1000 \
    -o "$OUT/mash_smoke" \
    "${FASTAS[@]}" \
    > "$OUT/mash_sketch.stdout.txt" \
    2> "$OUT/mash_sketch.stderr.txt" \
    || MASH_STATUS=$?

if [ "$MASH_STATUS" -eq 0 ] && [ -s "$OUT/mash_smoke.msh" ]; then

    mash info "$OUT/mash_smoke.msh" \
        > "$OUT/mash_info.txt" \
        2>&1 || true

    mash dist \
        "$OUT/mash_smoke.msh" \
        "$OUT/mash_smoke.msh" \
        > "$OUT/mash_smoke_dist.tsv" \
        2> "$OUT/mash_dist.stderr.txt" \
        || MASH_STATUS=$?
fi

echo "MASH_STATUS=$MASH_STATUS"

if [ -s "$OUT/mash_smoke_dist.tsv" ]; then
    echo "Mash output preview:"
    head -n 12 "$OUT/mash_smoke_dist.tsv"
fi

# ----------------------------------------------------------------------
# 8. Dashing2 smoke test
#
# IMPORTANT:
# This is ONLY a CLI smoke test.
# Do not treat this output as the frozen production teacher.
#
# Production parameters will be frozen only AFTER inspecting
# the installed binary's canonicalization/cache/multiset options.
# ----------------------------------------------------------------------

echo
echo "[8] DASHING2 SMOKE TEST"

DASHING2_STATUS=0

dashing2 sketch \
    -k31 \
    -S1024 \
    --cmpout "$OUT/dashing2_smoke.out" \
    "${FASTAS[@]}" \
    > "$OUT/dashing2_smoke.stdout.txt" \
    2> "$OUT/dashing2_smoke.stderr.txt" \
    || DASHING2_STATUS=$?

echo "DASHING2_STATUS=$DASHING2_STATUS"

if [ -s "$OUT/dashing2_smoke.out" ]; then
    echo "Dashing2 output preview:"
    head -n 20 "$OUT/dashing2_smoke.out" || true
else
    echo "Dashing2 comparison output is empty or was not created."
fi

# ----------------------------------------------------------------------
# 9. skani smoke test
#
# Use temporary uncompressed FASTA copies so this test does not
# depend on whether the installed skani build accepts .gz directly.
# ----------------------------------------------------------------------

echo
echo "[9] SKANI SMOKE TEST"

mkdir -p "$OUT/skani_smoke_fasta"

gzip -cd "${FASTAS[0]}" \
    > "$OUT/skani_smoke_fasta/genome_1.fna"

gzip -cd "${FASTAS[1]}" \
    > "$OUT/skani_smoke_fasta/genome_2.fna"

gzip -cd "${FASTAS[2]}" \
    > "$OUT/skani_smoke_fasta/genome_3.fna"

SKANI_SAME_STATUS=0
SKANI_DISTANT_STATUS=0

skani dist \
    -t 4 \
    "$OUT/skani_smoke_fasta/genome_1.fna" \
    "$OUT/skani_smoke_fasta/genome_2.fna" \
    > "$OUT/skani_same_species.tsv" \
    2> "$OUT/skani_same_species.stderr.txt" \
    || SKANI_SAME_STATUS=$?

skani dist \
    -t 4 \
    "$OUT/skani_smoke_fasta/genome_1.fna" \
    "$OUT/skani_smoke_fasta/genome_3.fna" \
    > "$OUT/skani_distant.tsv" \
    2> "$OUT/skani_distant.stderr.txt" \
    || SKANI_DISTANT_STATUS=$?

echo "SKANI_SAME_STATUS=$SKANI_SAME_STATUS"
echo "SKANI_DISTANT_STATUS=$SKANI_DISTANT_STATUS"

echo
echo "Same-species skani output:"
cat "$OUT/skani_same_species.tsv" || true

echo
echo "Distant skani output:"
cat "$OUT/skani_distant.tsv" || true

# No-output for a distant pair is allowed:
# skani may suppress sufficiently low-AF / low-ANI comparisons.

# ----------------------------------------------------------------------
# 10. Summary
# ----------------------------------------------------------------------

echo
echo "[10] PREFLIGHT SUMMARY"

{
    echo -e "component\tstatus"
    echo -e "mash\t$MASH_STATUS"
    echo -e "dashing2\t$DASHING2_STATUS"
    echo -e "skani_same_species\t$SKANI_SAME_STATUS"
    echo -e "skani_distant\t$SKANI_DISTANT_STATUS"
} > "$OUT/preflight_status.tsv"

column -t -s $'\t' "$OUT/preflight_status.tsv" || \
cat "$OUT/preflight_status.tsv"

if [ "$MASH_STATUS" -ne 0 ] \
   || [ "$DASHING2_STATUS" -ne 0 ] \
   || [ "$SKANI_SAME_STATUS" -ne 0 ] \
   || [ "$SKANI_DISTANT_STATUS" -ne 0 ]; then

    echo
    echo "TEACHER_TOOL_PREFLIGHT=FAIL"
    echo "Inspect stderr/help files under:"
    echo "$OUT"
    exit 7
fi

# ----------------------------------------------------------------------
# 11. Freeze checksums
# ----------------------------------------------------------------------

(
    cd "$OUT"

    sha256sum \
        smoke_manifest.tsv \
        tool_versions.txt \
        mash_smoke.msh \
        mash_smoke_dist.tsv \
        dashing2_smoke.out \
        skani_same_species.tsv \
        skani_distant.tsv \
        > SHA256SUMS.txt
)

echo
echo "=============================================================================="
echo "ALL TEACHER TOOL PREFLIGHT GATES PASSED"
echo "=============================================================================="
echo "Output: $OUT"
echo
echo "TEACHER_TOOL_PREFLIGHT=PASS"
