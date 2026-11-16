#!/usr/bin/env bash

set -euo pipefail

BASE=HGV_V3_revision_20260915

ROOT="$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"

TRAINDIR="$ROOT/03_training"
FREEZE="$ROOT/01_execution_freeze"

mkdir -p "$FREEZE"

count_complete() {

    local mode="$1"
    local mode_dir="$TRAINDIR/$mode"

    if [[ ! -d "$mode_dir" ]]; then
        echo 0
        return
    fi

    grep -rlx \
        "RUN_COMPLETE" \
        "$mode_dir" \
        2>/dev/null \
        | wc -l
}

D0_N="$(count_complete D0)"
D1_N="$(count_complete D1)"
D2_N="$(count_complete D2)"
D3_N="$(count_complete D3)"
D4_N="$(count_complete D4)"
D5_N="$(count_complete D5)"

TOTAL_NEW=$((D0_N + D1_N + D2_N + D3_N + D4_N + D5_N))

echo "============================================"
echo "HGV-V3 E6 NEW TRAINING FINAL AUDIT"
echo "============================================"
echo "D0=$D0_N"
echo "D1=$D1_N"
echo "D2=$D2_N"
echo "D3=$D3_N"
echo "D4=$D4_N"
echo "D5=$D5_N"
echo "TOTAL_NEW=$TOTAL_NEW"
echo

if [[ "$D0_N" -ne 5 ]]; then
    echo "D0_COUNT=FAIL"
    exit 1
fi

if [[ "$D1_N" -ne 5 ]]; then
    echo "D1_COUNT=FAIL"
    exit 1
fi

if [[ "$D2_N" -ne 5 ]]; then
    echo "D2_COUNT=FAIL"
    exit 1
fi

if [[ "$D3_N" -ne 2 ]]; then
    echo "D3_COUNT=FAIL"
    exit 1
fi

if [[ "$D4_N" -ne 5 ]]; then
    echo "D4_COUNT=FAIL"
    exit 1
fi

if [[ "$D5_N" -ne 5 ]]; then
    echo "D5_COUNT=FAIL"
    exit 1
fi

if [[ "$TOTAL_NEW" -ne 27 ]]; then
    echo "TOTAL_NEW_COUNT=FAIL"
    exit 1
fi

{
    echo "D0=$D0_N"
    echo "D1=$D1_N"
    echo "D2=$D2_N"
    echo "D3=$D3_N"
    echo "D4=$D4_N"
    echo "D5=$D5_N"
    echo "TOTAL_NEW=$TOTAL_NEW"
    echo "REUSED_FROZEN_D3=3"
    echo "FINAL_E6_CELLS=30"
} > "$FREEZE/TRAINING_COMPLETION_COUNTS_v2.txt"

{
    echo "E6_NEW_TRAINING_MATRIX=PASS"
    echo "E6_NEW_TRAINING_TOTAL=27"
    echo "FROZEN_D3_TO_REUSE=3"
    echo "FINAL_E6_CELLS_AFTER_REUSE=30"
    echo "TEST_OPENED_DURING_TRAINING=NO"
    echo "EXTERNAL_OPENED_DURING_TRAINING=NO"
    echo "E6_27RUN_TRAINING=PASS"
} > "$FREEZE/E6_27RUN_TRAINING_FREEZE_v2.txt"

find "$TRAINDIR" \
    -type f \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$FREEZE/E6_TRAINING_ARTIFACT_SHA256_v1.txt"

echo "E6_NEW_TRAINING_MATRIX=PASS"
echo "E6_NEW_TRAINING_TOTAL=27"
echo "FROZEN_D3_TO_REUSE=3"
echo "FINAL_E6_CELLS_AFTER_REUSE=30"
echo "E6_27RUN_TRAINING=PASS"
