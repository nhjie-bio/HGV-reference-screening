#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SCRIPTS="$BASE/scripts_v3_20260917_e5_baselines"
E5="$BASE/09_baselines/E5_unified_baseline_v1_20260917"

FAST_RUNNER="$SCRIPTS/24d_run_e5_fastani_validation_bidirectional_v1.py"
SKANI_RUNNER="$SCRIPTS/24e_run_e5_skani_validation_v1.py"

FAST_LOG="$E5/05_fastani/validation_bidirectional_v1/logs"
SKANI_LOG="$E5/04_skani/validation_v1/logs"

mkdir -p "$FAST_LOG" "$SKANI_LOG"

echo "============================================================"
echo "FASTANI VALIDATION: REMAINING 8 SPLITS"
echo "============================================================"

for protocol in G2 G3 G4; do
    for seed in 20260915 20260916 20260917; do

        if [[ "$protocol" == "G2" && "$seed" == "20260915" ]]; then
            echo "[SKIP VERIFIED] FastANI $protocol seed$seed"
            continue
        fi

        echo
        echo "============================================================"
        echo "[FASTANI START] $protocol seed$seed"
        date
        echo "============================================================"

        python "$FAST_RUNNER" \
            --protocol "$protocol" \
            --seed "$seed" \
            --workers 40 \
            2>&1 | tee \
            "$FAST_LOG/${protocol}_seed${seed}.log"

        echo "============================================================"
        echo "[FASTANI DONE] $protocol seed$seed"
        date
        echo "============================================================"
    done
done


echo
echo "============================================================"
echo "SKANI VALIDATION: REMAINING 8 SPLITS"
echo "============================================================"

for protocol in G2 G3 G4; do
    for seed in 20260915 20260916 20260917; do

        if [[ "$protocol" == "G2" && "$seed" == "20260915" ]]; then
            echo "[SKIP VERIFIED] skani $protocol seed$seed"
            continue
        fi

        echo
        echo "============================================================"
        echo "[SKANI START] $protocol seed$seed"
        date
        echo "============================================================"

        python "$SKANI_RUNNER" \
            --protocol "$protocol" \
            --seed "$seed" \
            --workers 40 \
            2>&1 | tee \
            "$SKANI_LOG/${protocol}_seed${seed}.log"

        echo "============================================================"
        echo "[SKANI DONE] $protocol seed$seed"
        date
        echo "============================================================"
    done
done


echo
echo "============================================================"
echo "ALL REMAINING NATIVE VALIDATION RUNS COMPLETE"
echo "============================================================"
date
