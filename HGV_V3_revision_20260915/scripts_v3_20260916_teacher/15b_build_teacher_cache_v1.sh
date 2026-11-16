#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915

MANIFEST="$BASE/01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv"

OUT="$BASE/06_teacher/teacher_cache_v1_20260916"

THREADS=40

if [ -e "$OUT" ]; then
    echo "ERROR: output exists:"
    echo "$OUT"
    echo "Refusing overwrite."
    exit 2
fi

mkdir -p \
    "$OUT/mash" \
    "$OUT/dashing2_set" \
    "$OUT/dashing2_multiset" \
    "$OUT/skani"

exec > >(tee "$OUT/15b_build_teacher_cache.log") 2>&1

echo "=============================================================================="
echo "HGV-V3 TEACHER CACHE BUILD"
echo "=============================================================================="
echo

echo "[1] ENVIRONMENT"

{
    echo "date=$(date -Is)"
    echo "host=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV:-}"
    echo "conda_prefix=${CONDA_PREFIX:-}"
    echo "threads=$THREADS"
    echo
    echo "===== mash ====="
    mash --version
    echo
    echo "===== dashing2 ====="
    dashing2 sketch --help 2>&1 | head -n 3
    echo
    echo "===== skani ====="
    skani --version
} > "$OUT/environment.txt"

cat "$OUT/environment.txt"

# ----------------------------------------------------------------------
# 2. Freeze real Development-A FASTA list from final manifest
# ----------------------------------------------------------------------

echo
echo "[2] BUILD DEVELOPMENT-A FASTA LIST"

python3 - "$MANIFEST" "$OUT/development_A_manifest.tsv" "$OUT/development_A_fasta.list" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_manifest = Path(sys.argv[2])
out_list = Path(sys.argv[3])

with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = [
        r for r in reader
        if r.get("final_cohort", "").strip() == "A"
    ]

rows.sort(
    key=lambda r: r["assembly_accession_final"].strip()
)

if len(rows) != 2374:
    raise SystemExit(
        f"Expected actual frozen A=2374; observed {len(rows)}"
    )

seen_acc = set()
seen_path = set()

fields = [
    "row_index",
    "assembly_accession",
    "gtdb_species_cluster",
    "gtdb_genus",
    "gtdb_family",
    "fasta_path",
]

with open(out_manifest, "w", encoding="utf-8", newline="") as fo, \
     open(out_list, "w", encoding="utf-8") as fl:

    w = csv.DictWriter(
        fo,
        delimiter="\t",
        fieldnames=fields,
        lineterminator="\n",
    )
    w.writeheader()

    for i, r in enumerate(rows):
        acc = r["assembly_accession_final"].strip()
        path = r["fasta_path"].strip()

        if not acc:
            raise SystemExit("Empty accession")

        if acc in seen_acc:
            raise SystemExit(f"Duplicate accession: {acc}")

        if path in seen_path:
            raise SystemExit(f"Duplicate FASTA path: {path}")

        if not Path(path).is_file():
            raise SystemExit(f"Missing FASTA: {path}")

        seen_acc.add(acc)
        seen_path.add(path)

        w.writerow({
            "row_index": i,
            "assembly_accession": acc,
            "gtdb_species_cluster": r["gtdb_species_cluster"].strip(),
            "gtdb_genus": r["gtdb_genus"].strip(),
            "gtdb_family": r["gtdb_family"].strip(),
            "fasta_path": path,
        })

        fl.write(path + "\n")

print(f"DEVELOPMENT_A={len(rows)}")
print("A_LIST=PASS")
PY

echo
echo "FASTA count:"
wc -l "$OUT/development_A_fasta.list"

# ----------------------------------------------------------------------
# 3. gzip integrity before expensive work
# ----------------------------------------------------------------------

echo
echo "[3] GZIP INTEGRITY"

GZIP_BAD=0
N=0

while IFS= read -r F; do
    N=$((N+1))

    if ! gzip -t "$F" 2>/dev/null; then
        echo "BAD_GZIP $F"
        GZIP_BAD=$((GZIP_BAD+1))
    fi

    if [ $((N % 250)) -eq 0 ]; then
        echo "checked=$N bad=$GZIP_BAD"
    fi
done < "$OUT/development_A_fasta.list"

echo "checked=$N bad=$GZIP_BAD"

if [ "$GZIP_BAD" -ne 0 ]; then
    echo "TEACHER_CACHE_BUILD=FAIL"
    exit 3
fi

# ----------------------------------------------------------------------
# 4. Mash
#
# k=21
# sketch size=1000
# canonical DNA is Mash default
# triangle -E avoids symmetric duplicate computation
# ----------------------------------------------------------------------

echo
echo "[4] MASH k=21 s=1000"

MASH_PREFIX="$OUT/mash/development_A_k21_s1000"

/usr/bin/time -v \
    -o "$OUT/mash/mash_sketch.time.txt" \
    mash sketch \
        -p "$THREADS" \
        -k 21 \
        -s 1000 \
        -l "$OUT/development_A_fasta.list" \
        -o "$MASH_PREFIX" \
        > "$OUT/mash/mash_sketch.stdout.txt" \
        2> "$OUT/mash/mash_sketch.stderr.txt"

if [ ! -s "${MASH_PREFIX}.msh" ]; then
    echo "ERROR: Mash sketch missing"
    exit 4
fi

echo "Mash sketch created:"
ls -lh "${MASH_PREFIX}.msh"

/usr/bin/time -v \
    -o "$OUT/mash/mash_triangle.time.txt" \
    mash triangle \
        -p "$THREADS" \
        -E \
        "${MASH_PREFIX}.msh" \
        > "$OUT/mash/mash_triangle_edges.tsv" \
        2> "$OUT/mash/mash_triangle.stderr.txt"

if [ ! -s "$OUT/mash/mash_triangle_edges.tsv" ]; then
    echo "ERROR: Mash triangle output empty"
    exit 5
fi

echo "Mash edge preview:"
head -n 5 "$OUT/mash/mash_triangle_edges.tsv"

echo
echo "Mash edges:"
wc -l "$OUT/mash/mash_triangle_edges.tsv"

# ----------------------------------------------------------------------
# 5. Dashing2 SetSketch
#
# Installed version confirmed:
# - canonical DNA by default
# - One-Permutation SetSketch default
# - similarity output
# ----------------------------------------------------------------------

echo
echo "[5] DASHING2 SETSKETCH k=31 S=1024"

/usr/bin/time -v \
    -o "$OUT/dashing2_set/dashing2_set.time.txt" \
    dashing2 sketch \
        -p "$THREADS" \
        -k31 \
        -S1024 \
        --cmpout "$OUT/dashing2_set/dashing2_set_similarity.tsv" \
        -F "$OUT/development_A_fasta.list" \
        > "$OUT/dashing2_set/dashing2_set.stdout.txt" \
        2> "$OUT/dashing2_set/dashing2_set.stderr.txt"

if [ ! -s "$OUT/dashing2_set/dashing2_set_similarity.tsv" ]; then
    echo "ERROR: Dashing2 SetSketch output empty"
    exit 6
fi

echo "Dashing2 SetSketch preview:"
head -n 8 "$OUT/dashing2_set/dashing2_set_similarity.tsv"

# ----------------------------------------------------------------------
# 6. Dashing2 BagMinHash multiset sensitivity
#
# This is not yet the primary teacher.
# It is built because the V3 plan preregistered evaluation
# of multiset feasibility.
# ----------------------------------------------------------------------

echo
echo "[6] DASHING2 BAGMINHASH k=31 S=1024"

/usr/bin/time -v \
    -o "$OUT/dashing2_multiset/dashing2_multiset.time.txt" \
    dashing2 sketch \
        -p "$THREADS" \
        -k31 \
        -S1024 \
        -B \
        --cmpout "$OUT/dashing2_multiset/dashing2_multiset_similarity.tsv" \
        -F "$OUT/development_A_fasta.list" \
        > "$OUT/dashing2_multiset/dashing2_multiset.stdout.txt" \
        2> "$OUT/dashing2_multiset/dashing2_multiset.stderr.txt"

if [ ! -s "$OUT/dashing2_multiset/dashing2_multiset_similarity.tsv" ]; then
    echo "ERROR: Dashing2 multiset output empty"
    exit 7
fi

echo "Dashing2 multiset preview:"
head -n 8 "$OUT/dashing2_multiset/dashing2_multiset_similarity.tsv"

# ----------------------------------------------------------------------
# 7. skani reusable Development-A database
#
# No external genomes.
# No teacher selection here.
# ----------------------------------------------------------------------

echo
echo "[7] SKANI DEVELOPMENT-A SKETCH DATABASE"

SKANI_DB="$OUT/skani/development_A_skani_db"

/usr/bin/time -v \
    -o "$OUT/skani/skani_sketch.time.txt" \
    skani sketch \
        -t "$THREADS" \
        -l "$OUT/development_A_fasta.list" \
        -o "$SKANI_DB" \
        > "$OUT/skani/skani_sketch.stdout.txt" \
        2> "$OUT/skani/skani_sketch.stderr.txt"

if [ ! -d "$SKANI_DB" ]; then
    echo "ERROR: skani database directory missing"
    exit 8
fi

N_SKETCH=$(find "$SKANI_DB" -type f -name '*.sketch' | wc -l)

echo "skani .sketch files: $N_SKETCH"

if [ "$N_SKETCH" -ne 2374 ]; then
    echo "ERROR: expected 2374 skani sketches"
    exit 9
fi

# ----------------------------------------------------------------------
# 8. Basic output accounting
# ----------------------------------------------------------------------

echo
echo "[8] OUTPUT ACCOUNTING"

du -sh \
    "$OUT/mash" \
    "$OUT/dashing2_set" \
    "$OUT/dashing2_multiset" \
    "$OUT/skani"

{
    echo -e "artifact\tbytes"

    echo -e "mash_sketch\t$(stat -c %s "${MASH_PREFIX}.msh")"
    echo -e "mash_edges\t$(stat -c %s "$OUT/mash/mash_triangle_edges.tsv")"
    echo -e "dashing2_set_matrix\t$(stat -c %s "$OUT/dashing2_set/dashing2_set_similarity.tsv")"
    echo -e "dashing2_multiset_matrix\t$(stat -c %s "$OUT/dashing2_multiset/dashing2_multiset_similarity.tsv")"

    SKANI_BYTES=$(du -sb "$SKANI_DB" | awk '{print $1}')
    echo -e "skani_database\t$SKANI_BYTES"

} > "$OUT/artifact_sizes.tsv"

column -t -s $'\t' "$OUT/artifact_sizes.tsv"

# ----------------------------------------------------------------------
# 9. Tool/version/config freeze
# ----------------------------------------------------------------------

echo
echo "[9] CONFIG FREEZE"

cat > "$OUT/teacher_cache_config.tsv" <<EOF
parameter	value
development_genomes	2374
threads	$THREADS
mash_version	$(mash --version | head -n 1)
mash_k	21
mash_sketch_size	1000
mash_canonical	default
dashing2_version	2.1.20
dashing2_k	31
dashing2_sketch_size	1024
dashing2_canonical	default_enabled
dashing2_set_mode	onepermsetsketch
dashing2_multiset_mode	BagMinHash
skani_version	$(skani --version | head -n 1)
external_genomes_used	0
teacher_selection_performed	0
EOF

cat "$OUT/teacher_cache_config.tsv"

# ----------------------------------------------------------------------
# 10. SHA256
# ----------------------------------------------------------------------

echo
echo "[10] SHA256"

(
    cd "$OUT"

    sha256sum \
        development_A_manifest.tsv \
        development_A_fasta.list \
        environment.txt \
        teacher_cache_config.tsv \
        artifact_sizes.tsv \
        mash/development_A_k21_s1000.msh \
        mash/mash_triangle_edges.tsv \
        dashing2_set/dashing2_set_similarity.tsv \
        dashing2_multiset/dashing2_multiset_similarity.tsv \
        > SHA256SUMS.txt
)

cd "$OUT"
sha256sum -c SHA256SUMS.txt

# ----------------------------------------------------------------------
# 11. Final report
# ----------------------------------------------------------------------

cat > "$OUT/TEACHER_CACHE_REPORT.txt" <<EOF
HGV-V3 TEACHER CACHE BUILD
==============================================================================

STATUS: PASS

Development genomes: 2374
External genomes used: 0

Mash:
  version: $(mash --version | head -n 1)
  k: 21
  sketch size: 1000
  canonical DNA: default
  sketch: mash/development_A_k21_s1000.msh
  pairwise edge list: mash/mash_triangle_edges.tsv

Dashing2 SetSketch:
  version: 2.1.20
  k: 31
  sketch size: 1024
  canonical DNA: enabled by default
  mode: One-Permutation SetSketch
  output: dashing2_set/dashing2_set_similarity.tsv

Dashing2 multiset:
  version: 2.1.20
  k: 31
  sketch size: 1024
  canonical DNA: enabled
  mode: BagMinHash
  output: dashing2_multiset/dashing2_multiset_similarity.tsv

skani:
  version: $(skani --version | head -n 1)
  reusable Development-A sketch database:
  skani/development_A_skani_db

IMPORTANT:
No B1/B2 external genome was used.
No primary teacher was selected in this step.
No test/external result was used for hyperparameter selection.

TEACHER_CACHE_BUILD=PASS
EOF

echo
echo "=============================================================================="
echo "ALL TEACHER CACHE BUILD GATES PASSED"
echo "=============================================================================="
echo "Output: $OUT"
echo
echo "TEACHER_CACHE_BUILD=PASS"
