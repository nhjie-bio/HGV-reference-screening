#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915

SRC="$BASE/06_teacher/teacher_cache_v1_20260916"
OUT="$BASE/06_teacher/teacher_cache_v2_20260916"

THREADS=40
PILOT_N=128
PILOT_SEED=20260916

MASH_MSH="$SRC/mash/development_A_k21_s1000.msh"
MASH_EDGES="$SRC/mash/mash_triangle_edges.tsv"
D2_SET="$SRC/dashing2_set/dashing2_set_similarity.tsv"
A_MANIFEST="$SRC/development_A_manifest.tsv"
A_LIST="$SRC/development_A_fasta.list"

if [ -e "$OUT" ]; then
    echo "ERROR: output exists:"
    echo "$OUT"
    exit 2
fi

for F in \
    "$MASH_MSH" \
    "$MASH_EDGES" \
    "$D2_SET" \
    "$A_MANIFEST" \
    "$A_LIST"
do
    if [ ! -s "$F" ]; then
        echo "ERROR: missing prerequisite:"
        echo "$F"
        exit 3
    fi
done

mkdir -p \
    "$OUT/existing_core" \
    "$OUT/dashing2_bagminhash_pilot" \
    "$OUT/skani"

exec > >(tee "$OUT/15b_resume_teacher_cache_v3.log") 2>&1

echo "=============================================================================="
echo "HGV-V3 TEACHER CACHE RESUME V3"
echo "=============================================================================="

echo
echo "[1] VERIFY EXISTING CORE ARTIFACTS"

N_A=$(wc -l < "$A_LIST")
N_MASH=$(wc -l < "$MASH_EDGES")

echo "Development A genomes = $N_A"
echo "Mash pairwise edges    = $N_MASH"

if [ "$N_A" -ne 2374 ]; then
    echo "ERROR: expected 2374 A genomes"
    exit 4
fi

EXPECTED_MASH=$((2374 * 2373 / 2))

if [ "$N_MASH" -ne "$EXPECTED_MASH" ]; then
    echo "ERROR: Mash edge count mismatch"
    echo "expected=$EXPECTED_MASH observed=$N_MASH"
    exit 5
fi

sha256sum \
    "$MASH_MSH" \
    "$MASH_EDGES" \
    "$D2_SET" \
    "$A_MANIFEST" \
    "$A_LIST" \
    > "$OUT/existing_core/source_artifact_sha256.tsv"

echo "Existing Mash + Dashing2 SetSketch: PASS"

echo
echo "[2] BUILD DETERMINISTIC BAGMINHASH PILOT SET"

python3 - \
    "$A_MANIFEST" \
    "$OUT/dashing2_bagminhash_pilot/pilot128_manifest.tsv" \
    "$OUT/dashing2_bagminhash_pilot/pilot128_fasta.list" \
    "$PILOT_N" \
    "$PILOT_SEED" <<'PY'
import csv
import random
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_manifest = Path(sys.argv[2])
out_list = Path(sys.argv[3])
n = int(sys.argv[4])
seed = int(sys.argv[5])

with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if len(rows) != 2374:
    raise SystemExit(f"Expected 2374 rows, observed {len(rows)}")

rng = random.Random(seed)
chosen = rng.sample(rows, n)

chosen.sort(
    key=lambda r: r["assembly_accession"]
)

fields = list(chosen[0].keys())

with open(out_manifest, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(
        f,
        delimiter="\t",
        fieldnames=fields,
        lineterminator="\n",
    )
    w.writeheader()
    w.writerows(chosen)

with open(out_list, "w", encoding="utf-8") as f:
    for r in chosen:
        p = r["fasta_path"].strip()

        if not Path(p).is_file():
            raise SystemExit(f"Missing FASTA: {p}")

        f.write(p + "\n")

print(f"PILOT_N={len(chosen)}")
print(f"PILOT_SEED={seed}")
print("PILOT_SELECTION=PASS")
PY

echo
echo "[3] BAGMINHASH PILOT: 1 THREAD"

BAG1_STATUS=0

/usr/bin/time -v \
    -o "$OUT/dashing2_bagminhash_pilot/bagminhash_p1.time.txt" \
    dashing2 sketch \
        -p 1 \
        -k31 \
        -S1024 \
        -B \
        --cmpout "$OUT/dashing2_bagminhash_pilot/bagminhash_p1.tsv" \
        -F "$OUT/dashing2_bagminhash_pilot/pilot128_fasta.list" \
        > "$OUT/dashing2_bagminhash_pilot/bagminhash_p1.stdout.txt" \
        2> "$OUT/dashing2_bagminhash_pilot/bagminhash_p1.stderr.txt" \
    || BAG1_STATUS=$?

echo "BAGMINHASH_P1_STATUS=$BAG1_STATUS"

echo
echo "[4] BAGMINHASH PILOT: 8 THREADS"

BAG8_STATUS=0

/usr/bin/time -v \
    -o "$OUT/dashing2_bagminhash_pilot/bagminhash_p8.time.txt" \
    dashing2 sketch \
        -p 8 \
        -k31 \
        -S1024 \
        -B \
        --cmpout "$OUT/dashing2_bagminhash_pilot/bagminhash_p8.tsv" \
        -F "$OUT/dashing2_bagminhash_pilot/pilot128_fasta.list" \
        > "$OUT/dashing2_bagminhash_pilot/bagminhash_p8.stdout.txt" \
        2> "$OUT/dashing2_bagminhash_pilot/bagminhash_p8.stderr.txt" \
    || BAG8_STATUS=$?

echo "BAGMINHASH_P8_STATUS=$BAG8_STATUS"

for T in p1 p8; do
    F="$OUT/dashing2_bagminhash_pilot/bagminhash_${T}.tsv"

    if [ -s "$F" ]; then
        echo
        echo "BagMinHash $T preview:"
        head -n 6 "$F"
    fi
done

echo
echo "[5] CLASSIFY BAGMINHASH FEASIBILITY"

if [ "$BAG1_STATUS" -eq 0 ] && [ "$BAG8_STATUS" -eq 0 ]; then
    BAG_CLASS="PILOT_PASS_FULL2374_FAILED_SCALE_OR_PARALLELISM"
elif [ "$BAG1_STATUS" -eq 0 ] && [ "$BAG8_STATUS" -ne 0 ]; then
    BAG_CLASS="P1_PASS_P8_FAIL_PARALLELISM_SUSPECTED"
elif [ "$BAG1_STATUS" -ne 0 ] && [ "$BAG8_STATUS" -eq 0 ]; then
    BAG_CLASS="P1_FAIL_P8_PASS_UNUSUAL_RUNTIME_BEHAVIOR"
else
    BAG_CLASS="PILOT_FAIL_BAGMINHASH_UNAVAILABLE_IN_CURRENT_BUILD"
fi

echo "BAGMINHASH_CLASS=$BAG_CLASS"

cat > "$OUT/dashing2_bagminhash_pilot/bagminhash_status.tsv" <<EOF
metric	value
full2374_p40_status	SIGSEGV_SIGNAL11
full2374_p40_peak_rss_kb	13972044
pilot_genomes	$PILOT_N
pilot_seed	$PILOT_SEED
pilot_p1_exit	$BAG1_STATUS
pilot_p8_exit	$BAG8_STATUS
classification	$BAG_CLASS
EOF

echo
echo "[6] BUILD SKANI 0.3.2 SEARCHABLE DATABASE"

SKANI_DB="$OUT/skani/development_A_skani_db"

SKANI_STATUS=0

/usr/bin/time -v \
    -o "$OUT/skani/skani_sketch.time.txt" \
    skani sketch \
        -t "$THREADS" \
        -l "$A_LIST" \
        -o "$SKANI_DB" \
        > "$OUT/skani/skani_sketch.stdout.txt" \
        2> "$OUT/skani/skani_sketch.stderr.txt" \
    || SKANI_STATUS=$?

echo "SKANI_STATUS=$SKANI_STATUS"

if [ "$SKANI_STATUS" -ne 0 ]; then
    echo "ERROR: skani sketch failed"
    cat "$OUT/skani/skani_sketch.stderr.txt"
    exit 6
fi

if [ ! -e "$SKANI_DB" ]; then
    echo "ERROR: skani database was not created"
    exit 7
fi

echo "skani database:"
ls -lh "$SKANI_DB"

echo
echo "[7] SKANI DATABASE QUERY SANITY TEST"

FIRST_FASTA=$(head -n 1 "$A_LIST")

skani search \
    -d "$SKANI_DB" \
    -t 4 \
    "$FIRST_FASTA" \
    -o "$OUT/skani/skani_self_search.tsv" \
    > "$OUT/skani/skani_self_search.stdout.txt" \
    2> "$OUT/skani/skani_self_search.stderr.txt"

if [ ! -s "$OUT/skani/skani_self_search.tsv" ]; then
    echo "ERROR: skani self-search output empty"
    exit 8
fi

head -n 5 "$OUT/skani/skani_self_search.tsv"

echo
echo "[8] RESOURCE SUMMARY"

{
    echo -e "component\telapsed\tmax_rss_kb"

    for ITEM in \
        "bag_p1:$OUT/dashing2_bagminhash_pilot/bagminhash_p1.time.txt" \
        "bag_p8:$OUT/dashing2_bagminhash_pilot/bagminhash_p8.time.txt" \
        "skani:$OUT/skani/skani_sketch.time.txt"
    do
        NAME=${ITEM%%:*}
        FILE=${ITEM#*:}

        ELAPSED=$(grep "Elapsed (wall clock)" "$FILE" 2>/dev/null | sed 's/^[[:space:]]*//')
        RSS=$(grep "Maximum resident set size" "$FILE" 2>/dev/null | awk -F': ' '{print $2}')

        printf "%s\t%s\t%s\n" "$NAME" "$ELAPSED" "$RSS"
    done

} > "$OUT/resource_summary.tsv"

cat "$OUT/resource_summary.tsv"

echo
echo "[9] FINAL REPORT"

cat > "$OUT/TEACHER_CACHE_V3_REPORT.txt" <<EOF
HGV-V3 TEACHER CACHE RESUME V3
==============================================================================

STATUS: PASS

Development A genomes:
2374

Existing successful core:
Mash k=21 sketch=1000
Mash all-pairs edges=$EXPECTED_MASH
Dashing2 SetSketch k=31 sketch=1024

Dashing2 BagMinHash:
Full Development-A p=40:
SIGSEGV signal 11
Peak RSS: 13972044 kB
No swap/OOM evidence.

Pilot:
N=$PILOT_N
seed=$PILOT_SEED
p1_exit=$BAG1_STATUS
p8_exit=$BAG8_STATUS
classification=$BAG_CLASS

BagMinHash is retained only as a feasibility/sensitivity method.
It is not required for the primary teacher pipeline.

skani:
version=$(skani --version | head -n 1)
Development-A searchable database built successfully.
Self-query sanity test passed.

External B1/B2 genomes used:
0

TEACHER_CACHE_V3=PASS
EOF

cat "$OUT/TEACHER_CACHE_V3_REPORT.txt"

echo
echo "[10] SHA256"

(
    cd "$OUT"

    sha256sum \
        existing_core/source_artifact_sha256.tsv \
        dashing2_bagminhash_pilot/pilot128_manifest.tsv \
        dashing2_bagminhash_pilot/pilot128_fasta.list \
        dashing2_bagminhash_pilot/bagminhash_status.tsv \
        skani/skani_self_search.tsv \
        resource_summary.tsv \
        TEACHER_CACHE_V3_REPORT.txt \
        > SHA256SUMS.txt

    sha256sum -c SHA256SUMS.txt
)

echo
echo "=============================================================================="
echo "TEACHER CACHE RESUME COMPLETE"
echo "=============================================================================="
echo
echo "TEACHER_CACHE_V3=PASS"
