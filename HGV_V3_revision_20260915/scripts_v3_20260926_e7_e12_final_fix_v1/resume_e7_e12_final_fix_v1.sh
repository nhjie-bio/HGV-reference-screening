#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
V9="$BASE/scripts_v3_20260926_remaining_all_v9"
FIX="$BASE/scripts_v3_20260926_e7_e12_final_fix_v1"

MASTER="$FIX/run_all_e7_e12_final_fix_v1.sh"

OLD_E7="$BASE/11_sensitivity/E7_sensitivity_v1_20260925_v6"
NEW_E7="$BASE/11_sensitivity/E7_sensitivity_v2_gzipfix_20260925_v6"
E7_FINAL="$NEW_E7/final_validation_refreeze_v2_20260926"

OLD_E12="$BASE/16_scalability/E12_scalability_v1_20260925_v6"
NEW_E12="$BASE/16_scalability/E12_scalability_v2_gzipfix_20260926"

E5="$BASE/09_baselines/E5_unified_baseline_v1_20260917"
E5_AUDIT="$BASE/20_rebuttal/HGV_V3_E5_runtime_audit_v2_20260926"

LOGDIR="$FIX/logs"
mkdir -p "$LOGDIR"

export HGV_BASE="$BASE"
export HGV_RUN_TAG=20260925_v6

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32

echo "===================================================================================================="
echo "HGV-V3 E7/E12 FINAL FIX - RESUME AFTER E7 TRAINING"
echo "START=$(date --iso-8601=seconds)"
echo "===================================================================================================="

# ======================================================================================
# A. FIX ONLY THE PROVENANCE SOURCE ALIAS
#
# The failed wrapper expected canonical_kmer_one.cpp beside itself.
# The actual corrected source is canonical_kmer_gzip_batch_v1.cpp.
#
# We create a symbolic alias only for provenance hashing.
# No feature is regenerated here.
# ======================================================================================

echo
echo "===== A. PROVENANCE SOURCE FIX ====="

if [ ! -f "$FIX/canonical_kmer_gzip_batch_v1.cpp" ]; then
    echo "[FAIL] corrected gzip-aware extractor source missing"
    exit 2
fi

ln -sfn \
    canonical_kmer_gzip_batch_v1.cpp \
    "$FIX/canonical_kmer_one.cpp"

cat > "$FIX/EXTRACTOR_ALIAS_NOTE.txt" <<EOF
STATUS=PROVENANCE_ALIAS_ONLY
ALIAS=$FIX/canonical_kmer_one.cpp
ACTUAL_SOURCE=$FIX/canonical_kmer_gzip_batch_v1.cpp
PURPOSE=allow historical E7 wrapper provenance hashing to resolve the corrected gzip-aware extractor source
OLD_PLAIN_EXTRACTOR_USED_FOR_K5_K7=NO
EOF

echo "PROVENANCE_SOURCE_ALIAS=PASS"

# ======================================================================================
# B. VERIFY CORRECTED k5/k7 FEATURES
# ======================================================================================

echo
echo "===== B. VERIFY CORRECTED k5/k7 FEATURES ====="

python - <<'PY'
from pathlib import Path
import numpy as np

BASE = Path(
    "HGV_V3_revision_20260915"
)

E7 = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v2_gzipfix_20260925_v6"
)

for k, dim in [(5, 512), (7, 8192)]:

    p = (
        E7
        / f"features_k{k}"
        / f"canonical_k{k}_A2374.npy"
    )

    marker = (
        E7
        / f"features_k{k}"
        / "FEATURE_PASS.txt"
    )

    if not p.is_file():
        raise SystemExit(
            f"[FAIL] missing feature: {p}"
        )

    if not marker.is_file():
        raise SystemExit(
            f"[FAIL] missing marker: {marker}"
        )

    X = np.load(
        p,
        mmap_mode="r",
    )

    if X.shape != (2374, dim):
        raise SystemExit(
            f"[FAIL] k{k} shape={X.shape}"
        )

    zero_rows = 0
    bad_sum_rows = 0

    for st in range(
        0,
        2374,
        128,
    ):

        A = np.asarray(
            X[
                st:
                st + 128
            ],
            dtype=np.float64,
        )

        sums = A.sum(
            axis=1
        )

        nnz = np.count_nonzero(
            A,
            axis=1,
        )

        zero_rows += int(
            np.sum(
                nnz == 0
            )
        )

        bad_sum_rows += int(
            np.sum(
                ~np.isclose(
                    sums,
                    1.0,
                    atol=1e-5,
                )
            )
        )

    print(
        f"k{k}: shape={X.shape} "
        f"zero_rows={zero_rows} "
        f"bad_L1_rows={bad_sum_rows}"
    )

    if zero_rows != 0:
        raise SystemExit(
            f"[FAIL] k{k} has zero rows"
        )

    if bad_sum_rows != 0:
        raise SystemExit(
            f"[FAIL] k{k} has non-normalized rows"
        )

print(
    "K5_K7_FEATURE_GATE=PASS"
)
PY

# ======================================================================================
# C. VERIFY WHETHER ALL SIX NEW RUNS COMPLETED
# ======================================================================================

echo
echo "===== C. VERIFY SIX k5/k7 RUNS ====="

NEED_RESUME=0

for setting in k5 k7
do

    for seed in \
        20260915 \
        20260916 \
        20260917
    do

        RDIR="$NEW_E7/runs/$setting/model$seed"

        echo
        echo "CHECK $setting model$seed"

        for f in \
            "$RDIR/RUN_COMPLETE" \
            "$RDIR/best_model.pt" \
            "$RDIR/metrics.json" \
            "$RDIR/history.tsv"
        do

            if [ ! -f "$f" ]; then
                echo "[MISSING] $f"
                NEED_RESUME=1
            else
                echo "[OK] $f"
            fi

        done

    done

done

if [ "$NEED_RESUME" -eq 0 ]; then
    echo
    echo "E7_K5_K7_6RUNS=COMPLETE"
else
    echo
    echo "E7_K5_K7_6RUNS=INCOMPLETE"
    echo "Will resume only incomplete runs using existing corrected feature caches."
fi

# ======================================================================================
# D. VERIFY OLD NON-K CHECKPOINTS WERE NOT MODIFIED
#
# Compare copied v2 checkpoints directly against the original accepted E7-v1.
# ======================================================================================

echo
echo "===== D. NON-K CHECKPOINT INTEGRITY ====="

python - <<'PY'
from pathlib import Path
import hashlib

BASE = Path(
    "HGV_V3_revision_20260915"
)

OLD = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v1_20260925_v6"
)

NEW = (
    BASE
    / "11_sensitivity"
    / "E7_sensitivity_v2_gzipfix_20260925_v6"
)

settings = [
    "dim32",
    "dim128",
    "margin0p5",
    "margin1p5",
    "lambda0p1",
    "lambda0p5",
]

seeds = [
    20260915,
    20260916,
    20260917,
]

def sha(path):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for b in iter(
            lambda:
            f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                b
            )

    return h.hexdigest()

bad = []

for setting in settings:

    for seed in seeds:

        rel = (
            Path("runs")
            / setting
            / f"model{seed}"
            / "best_model.pt"
        )

        old = OLD / rel
        new = NEW / rel

        if not old.is_file():
            raise SystemExit(
                f"[FAIL] old checkpoint missing: {old}"
            )

        if not new.is_file():
            raise SystemExit(
                f"[FAIL] copied checkpoint missing: {new}"
            )

        a = sha(old)
        b = sha(new)

        status = (
            "PASS"
            if a == b
            else
            "FAIL"
        )

        print(
            setting,
            seed,
            status,
            a,
            b,
        )

        if a != b:
            bad.append(
                str(rel)
            )

if bad:

    print(
        "[FAIL] NON-K checkpoints changed:"
    )

    for x in bad:
        print(
            x
        )

    raise SystemExit(2)

print(
    "NON_K_CHECKPOINTS_IDENTICAL_TO_E7_V1=PASS"
)
PY

# ======================================================================================
# E. IF ANY OF THE SIX RUNS ARE INCOMPLETE, RESUME OLD TRAINER.
#
# If all six are already complete, DO NOT invoke training again.
# ======================================================================================

if [ "$NEED_RESUME" -ne 0 ]; then

    echo
    echo "===== E. RESUME INCOMPLETE k5/k7 RUNS ====="

    python \
        "$FIX/15_run_e7_sensitivity_gzipfix_v1.py" \
        2>&1 \
    | tee -a \
        "$LOGDIR/06_e7_k5_k7_training_resume.log"

else

    echo
    echo "===== E. TRAINING RESUME NOT NEEDED ====="

    echo "All six corrected k5/k7 runs already have:"
    echo "  RUN_COMPLETE"
    echo "  best_model.pt"
    echo "  metrics.json"
    echo "  history.tsv"

fi

# ======================================================================================
# F. FINAL SIX-RUN GATE
# ======================================================================================

echo
echo "===== F. FINAL SIX-RUN GATE ====="

for setting in k5 k7
do

    N=$(find \
        "$NEW_E7/runs/$setting" \
        -type f \
        -name RUN_COMPLETE \
        | wc -l)

    echo "$setting RUN_COMPLETE=$N"

    if [ "$N" -ne 3 ]; then
        echo "[FAIL] $setting expected 3 RUN_COMPLETE but got $N"
        exit 2
    fi

done

echo "E7_K5_K7_6RUNS_FINAL_GATE=PASS"

# ======================================================================================
# G. RECORD RECOVERY PROVENANCE
# ======================================================================================

cat > "$NEW_E7/E7_V2_RECOVERY_NOTE_20260926.txt" <<EOF
E7_V2_RECOVERY=PASS
DATE=$(date --iso-8601=seconds)

INITIAL_FAILURE_STAGE=post-training provenance finalization
INITIAL_FAILURE_REASON=historical E7 wrapper expected canonical_kmer_one.cpp in copied script directory

CORRECTED_FEATURE_EXTRACTOR=$FIX/canonical_kmer_gzip_batch_v1.cpp
CORRECTED_FEATURE_INPUT=real gzip FASTA
K5_K7_CORRECTED_FEATURE_GATE=PASS
K5_K7_NEW_MODEL_RUNS=6
NON_K_CHECKPOINTS_IDENTICAL_TO_E7_V1=YES

SCIENTIFIC_ACTION=continue from post-training finalization
OLD_E7_MODIFIED=NO
EOF

# ======================================================================================
# H. RESUME THE ORIGINAL MASTER SCRIPT AFTER THE FAILED TRAINER COMMAND
#
# We deliberately start at:
#
#   for setting in k5 k7
#
# This means:
# - no regeneration of features
# - no reinitialization of E7
# - no unconditional rerun of E7 training
#
# It continues:
# - run completeness checks
# - non-k checkpoint SHA gate
# - validation-only E7 refreeze
# - E12-v2 real gzip FASTA benchmark
# - E5 runtime audit
# - final freeze
# ======================================================================================

echo
echo "===== H. BUILD POST-TRAINING RESUME TAIL ====="

START_LINE=$(
    grep -n \
        '^for setting in k5 k7$' \
        "$MASTER" \
    | head -n 1 \
    | cut -d: -f1
)

if [ -z "$START_LINE" ]; then
    echo "[FAIL] Could not locate post-training resume point in master script."
    exit 2
fi

echo "MASTER_POST_TRAINING_START_LINE=$START_LINE"

TAIL="$FIX/.resume_post_training_tail_v1.sh"

{
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'

    cat <<EOF
BASE="$BASE"
V9="$V9"
FIX="$FIX"

OLD_E7="$OLD_E7"
NEW_E7="$NEW_E7"
E7_FINAL="$E7_FINAL"

OLD_E12="$OLD_E12"
NEW_E12="$NEW_E12"

E5="$E5"
E5_AUDIT="$E5_AUDIT"

LOGDIR="$LOGDIR"

export HGV_BASE="$BASE"
export HGV_RUN_TAG=20260925_v6

export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=32
EOF

    tail \
        -n +"$START_LINE" \
        "$MASTER"

} > "$TAIL"

chmod +x \
    "$TAIL"

echo "POST_TRAINING_RESUME_SCRIPT=PASS"

echo
echo "===== I. CONTINUE E7 FINALIZE -> E12-v2 -> E5 AUDIT -> FINAL GATE ====="

bash \
    "$TAIL"

echo
echo "===================================================================================================="
echo "RESUME_PIPELINE=PASS"
echo "END=$(date --iso-8601=seconds)"
echo "===================================================================================================="
