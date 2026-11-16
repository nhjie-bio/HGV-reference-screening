#!/usr/bin/env bash
set -euo pipefail

BASE="${HGV_BASE:-HGV_V3_revision_20260915}"
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HGV_BASE="$BASE"
export HGV_RUN_TAG="${HGV_RUN_TAG:-20260924}"
export HGV_WORKERS="${HGV_WORKERS:-44}"
export E6_BOOTSTRAP_N="${E6_BOOTSTRAP_N:-1000}"
export E6_BOOTSTRAP_SEED="${E6_BOOTSTRAP_SEED:-20260924}"
export E11_SKANI_TOP="${E11_SKANI_TOP:-50}"
export E11_FASTANI_TOP="${E11_FASTANI_TOP:-50}"
export E12_REAL_REPEATS="${E12_REAL_REPEATS:-3}"
export E12_QUERY_N="${E12_QUERY_N:-50}"

# Keep 4 physical cores / 8 threads effectively free from BLAS oversubscription.
# Heavy pairwise tools are parallelized at the process level by HGV_WORKERS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260924}"

LOGDIR="$PIPE/master_logs_${HGV_RUN_TAG}"
mkdir -p "$LOGDIR"
MASTER_STATUS="$LOGDIR/MASTER_STATUS.tsv"

if [[ ! -s "$MASTER_STATUS" ]]; then
    printf 'stage\tstatus\tstart\tend\tlog\n' > "$MASTER_STATUS"
fi

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

run_stage() {
    local stage="$1"
    local script="$2"
    local log="$LOGDIR/${stage}.log"
    local start end
    start="$(stamp)"
    echo
    echo '===================================================================================================='
    echo "[$start] START $stage"
    echo "SCRIPT=$script"
    echo '===================================================================================================='
    set +e
    python "$PIPE/$script" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    end="$(stamp)"
    if [[ $rc -ne 0 ]]; then
        printf '%s\tFAIL\t%s\t%s\t%s\n' "$stage" "$start" "$end" "$log" >> "$MASTER_STATUS"
        echo "[$end] FAIL $stage rc=$rc"
        exit "$rc"
    fi
    printf '%s\tPASS_OR_SKIP\t%s\t%s\t%s\n' "$stage" "$start" "$end" "$log" >> "$MASTER_STATUS"
    echo "[$end] PASS/SKIP $stage"
}

{
    echo "HGV-V3 remaining experiment master pipeline v1"
    echo "BASE=$BASE"
    echo "PIPE=$PIPE"
    echo "RUN_TAG=$HGV_RUN_TAG"
    echo "HGV_WORKERS=$HGV_WORKERS"
    echo "E6_BOOTSTRAP_N=$E6_BOOTSTRAP_N"
    echo "E11_SKANI_TOP=$E11_SKANI_TOP"
    echo "E11_FASTANI_TOP=$E11_FASTANI_TOP"
    echo "E12_REAL_REPEATS=$E12_REAL_REPEATS"
    echo "E12_QUERY_N=$E12_QUERY_N"
    echo "START=$(stamp)"
} | tee "$LOGDIR/MASTER_CONFIG.txt"

# This order follows the frozen revision priority:
# E6 mechanism/statistics -> E11 -> E12 -> E7/E8 -> E9/E10 -> global freeze.
run_stage 00_preflight 00_preflight_remaining_v1.py
run_stage 10_e6_mechanism 10_summarize_e6_metrics_geometry_v1.py
run_stage 11_e6_statistics 11_e6_species_block_statistics_v1.py
run_stage 12_e6_finalize 12_finalize_e6_v1.py
run_stage 13a_e11_confirmation_preflight 13a_preflight_e11_confirmation_tools_v1.py
run_stage 13_e11_screening 13_run_e11_reference_screening_v1.py
run_stage 14_e12_scalability 14_run_e12_scalability_v1.py
run_stage 14a_e7_input_preflight 14a_preflight_e7_inputs_v1.py
run_stage 15_e7_sensitivity 15_run_e7_sensitivity_v1.py
run_stage 16_e8_fragmentation 16_run_e8_fragmentation_v1.py
run_stage 17_e9_explainability 17_run_e9_explainability_v1.py
run_stage 18_e10_biological_fidelity 18_run_e10_biological_fidelity_v1.py
run_stage 19_global_finalize 19_finalize_remaining_v1.py

END="$(stamp)"
echo "END=$END" | tee -a "$LOGDIR/MASTER_CONFIG.txt"
touch "$LOGDIR/MASTER_PIPELINE_COMPLETE"
echo
cat <<EOF
====================================================================================================
HGV-V3 REMAINING PIPELINE COMPLETE
====================================================================================================
Status table: $MASTER_STATUS
Completion marker: $LOGDIR/MASTER_PIPELINE_COMPLETE
EOF
