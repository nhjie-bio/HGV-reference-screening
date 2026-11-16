#!/usr/bin/env bash
set -u

BASE=HGV_V3_revision_20260915
E5="$BASE/09_baselines/E5_unified_baseline_v1_20260917"
OUT="$E5/06_dnaberts/00_feasibility_audit_v1_20260917"

mkdir -p "$OUT"

REPORT="$OUT/DNABERTS_ENVIRONMENT_AUDIT.txt"

{
echo "======================================================================"
echo "HGV-V3 E5 DNABERT-S FEASIBILITY / ENVIRONMENT AUDIT v1"
echo "======================================================================"
echo
echo "DATE"
date
echo

echo "--------------------------------------------------------------------"
echo "SYSTEM"
echo "--------------------------------------------------------------------"
uname -a || true
echo
echo "HOST:"
hostname || true
echo

echo "--------------------------------------------------------------------"
echo "CPU"
echo "--------------------------------------------------------------------"
lscpu | grep -E \
'Model name|Socket|Core|Thread|CPU\(s\)|NUMA' \
|| true
echo

echo "--------------------------------------------------------------------"
echo "MEMORY"
echo "--------------------------------------------------------------------"
free -h || true
echo

echo "--------------------------------------------------------------------"
echo "FILESYSTEM"
echo "--------------------------------------------------------------------"
df -h "$BASE" || true
echo

echo "--------------------------------------------------------------------"
echo "GPU / NVIDIA DRIVER"
echo "--------------------------------------------------------------------"

if command -v nvidia-smi >/dev/null 2>&1; then

    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.free,memory.used \
        --format=csv,noheader \
        || true

    echo
    nvidia-smi || true

else

    echo "nvidia-smi: NOT FOUND"

fi

echo

echo "--------------------------------------------------------------------"
echo "CUDA TOOLKIT"
echo "--------------------------------------------------------------------"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
else
    echo "nvcc: NOT FOUND"
fi

echo

echo "--------------------------------------------------------------------"
echo "CURRENT PYTHON"
echo "--------------------------------------------------------------------"

which python || true
python --version || true

python - <<'PY'
import sys

print("sys.executable =", sys.executable)
print("python =", sys.version)

mods = [
    "torch",
    "transformers",
    "einops",
    "peft",
    "accelerate",
    "sklearn",
    "numpy",
    "scipy",
    "triton",
]

for m in mods:
    try:
        x = __import__(m)

        version = getattr(
            x,
            "__version__",
            "UNKNOWN"
        )

        print(
            f"{m:15s} INSTALLED "
            f"version={version}"
        )

    except Exception as e:

        print(
            f"{m:15s} NOT_AVAILABLE "
            f"{type(e).__name__}: {e}"
        )

try:
    import torch

    print()
    print(
        "torch.cuda.is_available =",
        torch.cuda.is_available()
    )

    print(
        "torch.version.cuda =",
        torch.version.cuda
    )

    print(
        "torch.cuda.device_count =",
        torch.cuda.device_count()
    )

    if torch.cuda.is_available():

        for i in range(
            torch.cuda.device_count()
        ):

            prop = (
                torch.cuda.get_device_properties(i)
            )

            print(
                f"GPU {i}: "
                f"{prop.name}; "
                f"VRAM="
                f"{prop.total_memory / 1024**3:.2f} GiB"
            )

except Exception as e:

    print(
        "Torch CUDA audit failed:",
        repr(e)
    )
PY

echo

echo "--------------------------------------------------------------------"
echo "CONDA"
echo "--------------------------------------------------------------------"

which conda || true
conda --version || true
echo
conda env list || true

echo

echo "--------------------------------------------------------------------"
echo "EXISTING DNABERT / HUGGINGFACE CACHE"
echo "--------------------------------------------------------------------"

for d in \
    "$HOME/.cache/huggingface" \
    "$HOME/.cache/torch" \
    "$BASE/09_baselines/E5_unified_baseline_v1_20260917/06_dnaberts"
do
    echo
    echo "PATH: $d"

    if [[ -e "$d" ]]; then
        du -sh "$d" 2>/dev/null || true
        find "$d" -maxdepth 3 \
            -type d \
            \( \
              -iname '*dnabert*' \
              -o -iname '*zhihan*' \
            \) \
            -print 2>/dev/null \
            | head -n 50
    else
        echo "NOT PRESENT"
    fi
done

echo

echo "--------------------------------------------------------------------"
echo "NETWORK / MODEL ENDPOINT CHECK"
echo "--------------------------------------------------------------------"

if command -v curl >/dev/null 2>&1; then

    curl -L -I \
        --connect-timeout 15 \
        --max-time 30 \
        https://huggingface.co/zhihan1996/DNABERT-S \
        2>&1 \
        | head -n 20 \
        || true

else

    echo "curl: NOT FOUND"

fi

echo

echo "--------------------------------------------------------------------"
echo "PROJECT INPUT CHECK"
echo "--------------------------------------------------------------------"

MANIFEST="$BASE/01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv"

if [[ -f "$MANIFEST" ]]; then

    echo "master manifest: PRESENT"
    wc -l "$MANIFEST"
    sha256sum "$MANIFEST"

else

    echo "master manifest: MISSING"

fi

echo

echo "======================================================================"
echo "AUDIT COMPLETE"
echo "======================================================================"

} | tee "$REPORT"

sha256sum \
    "$REPORT" \
    > "$OUT/AUDIT_SHA256.txt"

echo
echo "Report:"
echo "$REPORT"

echo
echo "STATUS = AUDIT COMPLETE"
