#!/usr/bin/env bash

set -euo pipefail

BASE=HGV_V3_revision_20260915

SCRIPT_DIR="$BASE/scripts_v3_20260922_e6_final"

TRAINER="$SCRIPT_DIR/03_train_e6_d0_d5_from_17b_v1.py"

ROOT="$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"

PREFLIGHT="$ROOT/00_input_freeze/E6_PREFLIGHT_REPORT_v1.txt"

FREEZE="$ROOT/01_execution_freeze"
LOGDIR="$ROOT/02_training_logs"
TRAINDIR="$ROOT/03_training"

EXPECTED_PARENT_SHA="97185a30f8651867e0399a5082a9f45d14232fe4df9d449c152d3ee1191ad356"

EXPECTED_TRAINER_SHA="ed84a7dd2dfe4daa9945128f0a2418fa325117ca39439591a79664b673fb829b"

mkdir -p \
    "$FREEZE" \
    "$LOGDIR" \
    "$TRAINDIR"


echo "===================================================================================================="
echo "HGV-V3 FORMAL E6 TRAINING"
echo "===================================================================================================="
echo "ROOT=$ROOT"
echo "TRAINER=$TRAINER"
echo "PROTOCOL=G2"
echo "SPLIT_SEED=20260915"
echo
echo "NEW RUN MATRIX:"
echo "  D0 = 5"
echo "  D1 = 5"
echo "  D2 = 5"
echo "  D3 = 2  [20260918, 20260919 only]"
echo "  D4 = 5"
echo "  D5 = 5"
echo "  TOTAL NEW = 27"
echo
echo "FROZEN D3 REUSE:"
echo "  20260915"
echo "  20260916"
echo "  20260917"
echo "===================================================================================================="


# ============================================================
# Gate 1: formal preflight
# ============================================================

if [[ ! -f "$PREFLIGHT" ]]; then
    echo "ERROR: missing preflight report:"
    echo "$PREFLIGHT"
    exit 1
fi

if ! grep -qx "E6_PREFLIGHT=PASS" "$PREFLIGHT"; then
    echo "ERROR: E6_PREFLIGHT=PASS not found"
    exit 1
fi

if ! grep -qx "TEST_OPENED=NO" "$PREFLIGHT"; then
    echo "ERROR: preflight test gate failed"
    exit 1
fi

if ! grep -qx "EXTERNAL_OPENED=NO" "$PREFLIGHT"; then
    echo "ERROR: preflight external gate failed"
    exit 1
fi

echo "PREFLIGHT_GATE=PASS"


# ============================================================
# Gate 2: trainer existence + SHA256
# ============================================================

if [[ ! -f "$TRAINER" ]]; then
    echo "ERROR: missing trainer:"
    echo "$TRAINER"
    exit 1
fi

TRAINER_SHA="$(
    sha256sum "$TRAINER" |
    awk '{print $1}'
)"

if [[ "$TRAINER_SHA" != "$EXPECTED_TRAINER_SHA" ]]; then
    echo "ERROR: trainer SHA256 mismatch"
    echo "expected=$EXPECTED_TRAINER_SHA"
    echo "observed=$TRAINER_SHA"
    exit 1
fi

PARENT="$SCRIPT_DIR/17b_train_main_hierarchical_v1.ORIGINAL.py"

PARENT_SHA="$(
    sha256sum "$PARENT" |
    awk '{print $1}'
)"

if [[ "$PARENT_SHA" != "$EXPECTED_PARENT_SHA" ]]; then
    echo "ERROR: parent 17b SHA256 mismatch"
    echo "expected=$EXPECTED_PARENT_SHA"
    echo "observed=$PARENT_SHA"
    exit 1
fi

echo "TRAINER_SHA256_GATE=PASS"
echo "PARENT_SHA256_GATE=PASS"


# ============================================================
# Freeze execution environment before training
# ============================================================

date -Is > "$FREEZE/START_TIME.txt"

sha256sum \
    "$PARENT" \
    "$TRAINER" \
    "$PREFLIGHT" \
    > "$FREEZE/INPUT_SHA256.txt"

{
    echo "hostname=$(hostname)"
    echo "date=$(date -Is)"
    echo "pwd=$(pwd)"
    echo "python=$(command -v python3)"
    python3 --version
    echo
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-NA}"
    echo
    python3 - <<'PY'
import sys
import numpy
import sklearn
import torch

print("python", sys.version.replace("\n", " "))
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
} > "$FREEZE/PYTHON_ENVIRONMENT.txt" 2>&1

{
    echo "===== NVIDIA-SMI ====="
    nvidia-smi || true

    echo
    echo "===== CPU ====="
    lscpu || true

    echo
    echo "===== MEMORY ====="
    free -h || true
} > "$FREEZE/HARDWARE.txt" 2>&1


# ============================================================
# Completion marker helper
#
# Do not depend on exact filename.
# Count files whose complete content is RUN_COMPLETE.
# ============================================================

count_complete() {

    local mode="$1"
    local mode_dir="$TRAINDIR/$mode"

    if [[ ! -d "$mode_dir" ]]; then
        echo 0
        return
    fi

    grep -rlx \
        --include='*' \
        "RUN_COMPLETE" \
        "$mode_dir" \
        2>/dev/null |
        wc -l
}


expected_for_mode() {

    local mode="$1"

    case "$mode" in
        D0|D1|D2|D4|D5)
            echo 5
            ;;
        D3)
            echo 2
            ;;
        *)
            echo "ERROR: unknown mode $mode" >&2
            exit 1
            ;;
    esac
}


# ============================================================
# Execute modes sequentially.
#
# --resume is always supplied:
# - fresh run: starts normally
# - interrupted run: restores last_state.pt
#
# Trainer itself remains responsible for its frozen
# deterministic epoch sampling and checkpoint logic.
# ============================================================

MODES=(
    D0
    D1
    D2
    D3
    D4
    D5
)

for MODE in "${MODES[@]}"; do

    EXPECTED="$(
        expected_for_mode "$MODE"
    )"

    BEFORE="$(
        count_complete "$MODE"
    )"

    echo
    echo "===================================================================================================="
    echo "MODE=$MODE"
    echo "EXPECTED_NEW_RUNS=$EXPECTED"
    echo "COMPLETE_BEFORE=$BEFORE"
    echo "START=$(date -Is)"
    echo "===================================================================================================="

    if [[ "$BEFORE" -gt "$EXPECTED" ]]; then
        echo "ERROR: $MODE has more RUN_COMPLETE markers than expected"
        exit 1
    fi

    if [[ "$BEFORE" -eq "$EXPECTED" ]]; then
        echo "$MODE already complete; skipping."
        continue
    fi

    MODE_LOG="$LOGDIR/${MODE}.training.log"

    {
        echo "===================================================================================================="
        echo "HGV-V3 FORMAL E6"
        echo "MODE=$MODE"
        echo "EXPECTED_RUNS=$EXPECTED"
        echo "START=$(date -Is)"
        echo "TRAINER_SHA256=$TRAINER_SHA"
        echo "===================================================================================================="
    } | tee -a "$MODE_LOG"

    HGV_E6_MODE="$MODE" \
    python3 "$TRAINER" \
        --protocol G2 \
        --resume \
        2>&1 |
        tee -a "$MODE_LOG"

    PIPE_STATUS=("${PIPESTATUS[@]}")

    PY_STATUS="${PIPE_STATUS[0]}"
    TEE_STATUS="${PIPE_STATUS[1]}"

    if [[ "$PY_STATUS" -ne 0 ]]; then
        echo "ERROR: trainer failed for $MODE with exit=$PY_STATUS"
        exit "$PY_STATUS"
    fi

    if [[ "$TEE_STATUS" -ne 0 ]]; then
        echo "ERROR: tee failed for $MODE with exit=$TEE_STATUS"
        exit "$TEE_STATUS"
    fi

    AFTER="$(
        count_complete "$MODE"
    )"

    echo "COMPLETE_AFTER=$AFTER"
    echo "END=$(date -Is)"

    if [[ "$AFTER" -ne "$EXPECTED" ]]; then
        echo "ERROR: $MODE completion count mismatch"
        echo "expected=$EXPECTED"
        echo "observed=$AFTER"
        exit 1
    fi

    echo "MODE_${MODE}_TRAINING=PASS"

done


# ============================================================
# Final 27-new-run audit
# ============================================================

D0_N="$(count_complete D0)"
D1_N="$(count_complete D1)"
D2_N="$(count_complete D2)"
D3_N="$(count_complete D3)"
D4_N="$(count_complete D4)"
D5_N="$(count_complete D5)"

TOTAL_NEW="$(
    (
        D0_N +
        D1_N +
        D2_N +
        D3_N +
        D4_N +
        D5_N
    )
)"

{
    echo "D0=$D0_N"
    echo "D1=$D1_N"
    echo "D2=$D2_N"
    echo "D3=$D3_N"
    echo "D4=$D4_N"
    echo "D5=$D5_N"
    echo "TOTAL_NEW=$TOTAL_NEW"
} | tee "$FREEZE/TRAINING_COMPLETION_COUNTS.txt"

if [[ "$D0_N" -ne 5 ]] ||
   [[ "$D1_N" -ne 5 ]] ||
   [[ "$D2_N" -ne 5 ]] ||
   [[ "$D3_N" -ne 2 ]] ||
   [[ "$D4_N" -ne 5 ]] ||
   [[ "$D5_N" -ne 5 ]]; then

    echo "E6_NEW_TRAINING_MATRIX=FAIL"
    exit 1
fi

if [[ "$TOTAL_NEW" -ne 27 ]]; then
    echo "E6_NEW_TRAINING_TOTAL=FAIL"
    exit 1
fi

date -Is > "$FREEZE/END_TIME.txt"

{
    echo "E6_NEW_TRAINING_MATRIX=PASS"
    echo "E6_NEW_TRAINING_TOTAL=27"
    echo "FROZEN_D3_TO_REUSE=3"
    echo "FINAL_E6_CELLS_AFTER_REUSE=30"
    echo "TEST_OPENED=NO"
    echo "EXTERNAL_OPENED=NO"
    echo "E6_27RUN_TRAINING=PASS"
} | tee "$FREEZE/E6_27RUN_TRAINING_FREEZE.txt"

echo
echo "===================================================================================================="
echo "E6 27 NEW RUNS COMPLETE"
echo "NEW_RUNS=27"
echo "FROZEN_D3_REUSE=3"
echo "FINAL_E6_CELLS=30"
echo "TEST_OPENED=NO"
echo "EXTERNAL_OPENED=NO"
echo "E6_27RUN_TRAINING=PASS"
echo "===================================================================================================="
