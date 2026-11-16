#!/usr/bin/env bash

set -euo pipefail

BASE="${HGV_BASE:-HGV_V3_revision_20260915}"

PIPE="$BASE/scripts_v3_20260920_e5e6_one_shot"

MASTER_OUT="$BASE/10_ablation/E5_E6_master_execution_v1_20260920"

mkdir -p "$MASTER_OUT"

LOG="$MASTER_OUT/E5_E6_MASTER.log"

export HGV_BASE="$BASE"

# 48 cores / 96 threads; leave 4 cores / 8 threads free.
export HGV_THREADS="${HGV_THREADS:-88}"

# FastANI launches one single-thread query process per worker.
export HGV_FASTANI_WORKERS="${HGV_FASTANI_WORKERS:-44}"

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32

export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec > >(tee -a "$LOG") 2>&1

echo "===================================================================================================="
echo "HGV-V3 E5 + E6 ONE-SHOT MASTER"
echo "===================================================================================================="
echo "DATE                : $(date -Is)"
echo "BASE                : $BASE"
echo "PIPE                : $PIPE"
echo "THREADS             : $HGV_THREADS"
echo "FASTANI_WORKERS     : $HGV_FASTANI_WORKERS"
echo "CONDA_DEFAULT_ENV   : ${CONDA_DEFAULT_ENV:-UNKNOWN}"
echo


echo "===================================================================================================="
echo "PRE-FLIGHT"
echo "===================================================================================================="

python3 - <<'PY'
import importlib
import shutil
import sys

mods = [
    "numpy",
    "pandas",
    "torch",
    "sklearn",
    "scipy",
]

for m in mods:
    importlib.import_module(m)
    print("PYTHON PASS", m)

for t in [
    "mash",
    "dashing2",
    "fastANI",
    "skani",
]:
    p = shutil.which(t)
    print("TOOL", t, p)

    if p is None:
        raise SystemExit(
            f"Required tool missing: {t}"
        )

import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY


echo
echo "===================================================================================================="
echo "PHASE 1 / E5 EXTERNAL 11-METHOD COMPLETION"
echo "===================================================================================================="

python3 -u \
"$PIPE/01_run_e5_external_full_v1.py"

E5DONE="$BASE/09_baselines/E5_external_unified_v1_20260920/02_native_external_full_v1/RUN_COMPLETE"

if [[ ! -s "$E5DONE" ]]
then
    echo "FATAL: E5 completion gate missing:"
    echo "$E5DONE"
    exit 20
fi

echo
echo "E5 GATE PASS:"
cat "$E5DONE"


echo
echo "===================================================================================================="
echo "PHASE 2 / E6 D0-D5 5-SEED ABLATION"
echo "===================================================================================================="

python3 -u \
"$PIPE/02_run_e6_d0_d5_ablation_v1.py"

E6DONE="$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/RUN_COMPLETE"

if [[ ! -s "$E6DONE" ]]
then
    echo "FATAL: E6 completion gate missing:"
    echo "$E6DONE"
    exit 30
fi

echo
echo "E6 GATE PASS:"
cat "$E6DONE"


echo
echo "===================================================================================================="
echo "FINAL MASTER GATE"
echo "===================================================================================================="

cat > "$MASTER_OUT/STATUS_COMPLETE.txt" <<EOF
HGV-V3 E5 + E6 COMPLETE
date=$(date -Is)

E5:
$E5DONE

E6:
$E6DONE
EOF

sha256sum \
"$BASE/09_baselines/E5_external_unified_v1_20260920/02_native_external_full_v1/source_data/E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv" \
"$BASE/09_baselines/E5_external_unified_v1_20260920/02_native_external_full_v1/source_data/E5_EXTERNAL_ALL11_SEED_LEVEL.tsv" \
"$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/source_data/E6_D0_D5_SEED_LEVEL_METRICS.tsv" \
"$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/source_data/E6_D0_D5_5SEED_SUMMARY.tsv" \
"$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/source_data/E6_D3_PAIRED_EFFECTS.tsv" \
> "$MASTER_OUT/FINAL_SOURCE_DATA_SHA256.txt"

echo
echo "===================================================================================================="
echo "ALL COMPLETE"
echo "===================================================================================================="
echo
echo "E5 summary:"
echo "$BASE/09_baselines/E5_external_unified_v1_20260920/02_native_external_full_v1/source_data/E5_EXTERNAL_ALL11_METHOD_SUMMARY.tsv"
echo
echo "E6 summary:"
echo "$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/source_data/E6_D0_D5_5SEED_SUMMARY.tsv"
echo
echo "E6 D3 paired effects:"
echo "$BASE/10_ablation/E6_D0_D5_G2_seed20260915_5seed_v1_20260920/source_data/E6_D3_PAIRED_EFFECTS.tsv"
echo
echo "Master status:"
echo "$MASTER_OUT/STATUS_COMPLETE.txt"
echo
echo "===================================================================================================="
