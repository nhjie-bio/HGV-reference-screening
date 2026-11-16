#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SCRIPTS="$BASE/scripts_v3_20260917_e5_baselines"
E5="$BASE/09_baselines/E5_unified_baseline_v1_20260917"

RUNNER="$SCRIPTS/21b_run_e5_fastani_internal_bidirectional_v2.py"
LOGDIR="$E5/05_fastani/internal_bidirectional_v2/logs"

mkdir -p "$LOGDIR"

for protocol in G2 G3 G4; do
    for seed in 20260915 20260916 20260917; do

        # Already completed and inspected manually.
        if [[ "$protocol" == "G2" && "$seed" == "20260915" ]]; then
            echo "[SKIP VERIFIED] $protocol seed$seed"
            continue
        fi

        echo "============================================================"
        echo "[START] $protocol seed$seed"
        date
        echo "============================================================"

        python "$RUNNER" \
            --protocol "$protocol" \
            --seed "$seed" \
            --workers 40 \
            2>&1 | tee "$LOGDIR/${protocol}_seed${seed}.log"

        echo "============================================================"
        echo "[DONE] $protocol seed$seed"
        date
        echo "============================================================"
        echo
    done
done

echo "ALL REMAINING FASTANI INTERNAL RUNS COMPLETE"
