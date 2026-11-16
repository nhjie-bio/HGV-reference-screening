#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SDIR="$BASE/scripts_v3_20260930_e12b_resource"
OUT="$BASE/16_scalability/E12B_synchronized_resource_v1_20260930"

PREF="$OUT/02_preflight"
PROT="$OUT/00_protocol"

mkdir -p "$PREF" "$PROT" "$OUT/logs"

CKPT="$BASE/09_external_validation/canonical_A_model_training_v1_20260917/G1/split20260915/model20260915/best_model.pt"

ENVFILE="$PREF/E12B_RESOLVED_ENV.sh"
FREEZE="$PROT/E12B_HGV_CHECKPOINT_FREEZE.txt"

echo "============================================================"
echo "HGV-V3 E12B CHECKPOINT FREEZE + ASSET DISCOVERY v1"
echo "============================================================"

# ------------------------------------------------------------
# 1. Freeze the exact canonical external HGV checkpoint
# ------------------------------------------------------------

if [[ ! -s "$CKPT" ]]; then
    echo "[FAIL] checkpoint missing:"
    echo "  $CKPT"
    exit 2
fi

echo "[PASS] canonical G1 HGV checkpoint exists"
echo "       $CKPT"

CKPT_SHA=$(sha256sum "$CKPT" | awk '{print $1}')

cat > "$FREEZE" <<EOF
HGV-V3 E12B HGV CHECKPOINT FREEZE
Date: 2026-09-30

Protocol:
  G1 external A-only model
  taxonomy split seed = 20260915
  model seed          = 20260915
  checkpoint policy   = validation-selected best_model.pt

Checkpoint:
$CKPT

SHA256:
$CKPT_SHA

Explicit exclusions:
  - do not use last_state.pt
  - do not use G2/G3/G4 checkpoint
  - do not use E6 D0-D5 checkpoint
  - do not use E12 historical G2 checkpoint
  - do not average three HGV models in the synchronized Table 7 row

Rationale:
The synchronized Table 7 HGV resource measurement and its
50-query B1 accuracy must correspond to the same A-only G1 model.
EOF

echo "[PASS] checkpoint freeze written"
echo "       $FREEZE"

# ------------------------------------------------------------
# 2. Patch resolved environment
# ------------------------------------------------------------

if [[ ! -f "$ENVFILE" ]]; then
    echo "[FAIL] missing resolved environment:"
    echo "  $ENVFILE"
    exit 2
fi

python3 - "$ENVFILE" "$CKPT" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
ckpt = sys.argv[2]

lines = p.read_text().splitlines()

target = "export E12B_HGV_G1_SEED15_CKPT="

new = []
done = False

for line in lines:
    if line.startswith(target):
        new.append(f"export E12B_HGV_G1_SEED15_CKPT='{ckpt}'")
        done = True
    else:
        new.append(line)

if not done:
    new.append(f"export E12B_HGV_G1_SEED15_CKPT='{ckpt}'")

p.write_text("\n".join(new) + "\n")
PY

echo "[PASS] E12B_RESOLVED_ENV.sh patched"

# ------------------------------------------------------------
# 3. Structural checkpoint inspection
# ------------------------------------------------------------

python3 - "$CKPT" "$PREF/HGV_CHECKPOINT_STRUCTURE.txt" <<'PY'
import sys
from pathlib import Path
import torch

ckpt = Path(sys.argv[1])
out = Path(sys.argv[2])

obj = torch.load(
    ckpt,
    map_location="cpu",
    weights_only=False,
)

lines = []
lines.append(f"path\t{ckpt}")
lines.append(f"type\t{type(obj).__name__}")

if isinstance(obj, dict):
    lines.append("top_level_keys\t" + ",".join(map(str, obj.keys())))

    for k in [
        "seed",
        "model_seed",
        "split_seed",
        "epoch",
        "best_epoch",
        "val_auroc",
        "val_auprc",
        "threshold",
        "config",
    ]:
        if k in obj:
            v = obj[k]
            if isinstance(v, (dict, list, tuple)):
                v = repr(v)[:2000]
            lines.append(f"{k}\t{v}")

    sd = None
    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "encoder_state_dict",
    ]:
        if key in obj and isinstance(obj[key], dict):
            sd = obj[key]
            lines.append(f"state_dict_source\t{key}")
            break

    if sd is None:
        tensor_like = {
            k: v for k, v in obj.items()
            if hasattr(v, "shape")
        }
        if tensor_like:
            sd = tensor_like
            lines.append("state_dict_source\ttop_level_tensor_dict")

    if sd is not None:
        lines.append(f"state_dict_keys\t{len(sd)}")
        for k, v in list(sd.items())[:40]:
            if hasattr(v, "shape"):
                lines.append(f"param\t{k}\t{tuple(v.shape)}")
else:
    lines.append("note\tcheckpoint object is not a dict")

out.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:30]))
PY

echo "[PASS] checkpoint structure inspected"

# ------------------------------------------------------------
# 4. Discover canonical k6 feature assets
# ------------------------------------------------------------

FEATURE_ROOT="$BASE/03_features/canonical_k6_full3154_v1"

echo
echo "===== CANONICAL K6 FEATURE ASSETS ====="

if [[ -d "$FEATURE_ROOT" ]]; then
    find "$FEATURE_ROOT" -maxdepth 3 -type f \
        \( -name '*.npy' -o \
           -name '*.npz' -o \
           -name '*.tsv' -o \
           -name '*.tsv.gz' -o \
           -name '*.json' -o \
           -name '*.txt' \) \
        -printf '%s\t%p\n' \
        | sort -nr \
        | tee "$PREF/CANONICAL_K6_ASSETS.tsv"
else
    echo "[FAIL] feature root missing: $FEATURE_ROOT"
    exit 2
fi

# ------------------------------------------------------------
# 5. Discover the single PCA artifact
# ------------------------------------------------------------

echo
echo "===== PCA ARTIFACT ====="

find \
    "$BASE/08_main_generalization" \
    "$BASE/09_external_validation" \
    "$BASE/09_baselines" \
    "$BASE/15_reference_screening" \
    -type f \
    \( -iname '*pca*.joblib' -o \
       -iname '*pca*.pkl' -o \
       -iname '*pca*.pickle' -o \
       -iname '*pca*.npz' \) \
    2>/dev/null \
    | sort \
    | tee "$PREF/PCA_ARTIFACT_EXACT.txt"

PCA_N=$(grep -c . "$PREF/PCA_ARTIFACT_EXACT.txt" || true)

if [[ "$PCA_N" -eq 1 ]]; then
    PCA=$(cat "$PREF/PCA_ARTIFACT_EXACT.txt")
    echo "[PASS] exactly one PCA artifact:"
    echo "       $PCA"

    if ! grep -q '^export E12B_PCA_ARTIFACT=' "$ENVFILE"; then
        echo "export E12B_PCA_ARTIFACT='$PCA'" >> "$ENVFILE"
    else
        sed -i \
          "s|^export E12B_PCA_ARTIFACT=.*|export E12B_PCA_ARTIFACT='$PCA'|" \
          "$ENVFILE"
    fi
else
    echo "[WARN] PCA artifact count = $PCA_N"
fi

# ------------------------------------------------------------
# 6. Inspect final E5 query table schema
# ------------------------------------------------------------

source "$ENVFILE"

echo
echo "===== E5 QUERY TABLE SCHEMA ====="

python3 - "$E12B_E5_QUERY_TABLE" "$PREF/E5_QUERY_TABLE_SCHEMA.txt" <<'PY'
import sys
import pandas as pd
from pathlib import Path

p = Path(sys.argv[1])
out = Path(sys.argv[2])

df = pd.read_csv(p, sep="\t", dtype=str)

lines = []
lines.append("shape\t" + str(df.shape))
lines.append("columns")
for i, c in enumerate(df.columns):
    lines.append(f"{i}\t{c}")

lines.append("")
lines.append("first_rows")
lines.append(df.head(8).to_csv(sep="\t", index=False))

out.write_text("\n".join(lines) + "\n")

print("shape =", df.shape)
print("columns:")
for c in df.columns:
    print("  ", c)
print()
print(df.head(5).to_string(index=False))
PY

# ------------------------------------------------------------
# 7. Find prior production command implementations
#    We reuse tested syntax rather than inventing new CLI calls.
# ------------------------------------------------------------

echo
echo "===== SEARCH EXISTING PRODUCTION SCRIPTS ====="

SEARCH_ROOTS=(
  "$BASE/scripts_v3_20260917_e5_baselines"
  "$BASE/scripts_v3_20260922_e5_external_dashing_fix"
  "$BASE/scripts_v3_20260922_e6_final"
  "$BASE/scripts_v3_20260916_hgv"
  "$BASE"
)

for term in \
    'mash sketch' \
    'mash dist' \
    'dashing2' \
    'skani' \
    'fastANI' \
    'DNABERT' \
    'canonical_k6_full3154' \
    'PCA64'
do
    safe=$(echo "$term" | tr ' /' '__')

    {
        echo "### SEARCH: $term"
        grep -RIn \
          --include='*.py' \
          --include='*.sh' \
          --include='*.txt' \
          --include='*.yaml' \
          --include='*.json' \
          -F "$term" \
          "${SEARCH_ROOTS[@]}" \
          2>/dev/null \
          | head -120 \
          || true
    } > "$PREF/SEARCH_${safe}.txt"

    echo
    echo "--- $term ---"
    head -25 "$PREF/SEARCH_${safe}.txt"
done

# ------------------------------------------------------------
# 8. Record installed tool help needed by the final runner
# ------------------------------------------------------------

echo
echo "===== TOOL HELP FREEZE ====="

{
    echo "### MASH"
    mash --version 2>&1 || true
    mash sketch -h 2>&1 || true
    mash dist -h 2>&1 || true
} > "$PREF/MASH_HELP.txt"

{
    echo "### DASHING2"
    dashing2 --version 2>&1 || true
    dashing2 sketch --help 2>&1 || true
    dashing2 cmp --help 2>&1 || true
} > "$PREF/DASHING2_HELP_FULL.txt"

{
    echo "### SKANI"
    skani --version 2>&1 || true
    skani sketch -h 2>&1 || true
    skani search -h 2>&1 || true
    skani dist -h 2>&1 || true
} > "$PREF/SKANI_HELP.txt"

{
    echo "### FASTANI"
    fastANI --version 2>&1 || true
    fastANI -h 2>&1 || true
} > "$PREF/FASTANI_HELP.txt"

echo "[PASS] native CLI help frozen"

# ------------------------------------------------------------
# 9. Check cgroup v2 / memory measurement support
# ------------------------------------------------------------

echo
echo "===== MEMORY MEASUREMENT SUPPORT ====="

{
    echo "cgroup fs:"
    stat -fc %T /sys/fs/cgroup 2>/dev/null || true

    echo
    echo "memory controllers:"
    cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null || true

    echo
    echo "systemd-run:"
    command -v systemd-run || true

    echo
    echo "time:"
    command -v /usr/bin/time || true

    echo
    echo "smem:"
    command -v smem || true

    echo
    echo "pss support:"
    test -r /proc/self/smaps_rollup && echo YES || echo NO
} | tee "$PREF/MEMORY_MEASUREMENT_SUPPORT.txt"

# ------------------------------------------------------------
# 10. Update SHA freeze
# ------------------------------------------------------------

{
    sha256sum "$CKPT"
    sha256sum "$FREEZE"
    sha256sum "$ENVFILE"
    sha256sum "$OUT/01_manifests/E12B_REFERENCES_2374.tsv"
    sha256sum "$OUT/01_manifests/E12B_QUERIES_50.tsv"
} > "$PROT/E12B_STAGE00B_SHA256.txt"

# ------------------------------------------------------------
# 11. Final Stage-00B gate
# ------------------------------------------------------------

echo
echo "============================================================"
echo "E12B STAGE 00B FINAL GATE"
echo "============================================================"

source "$ENVFILE"

FAIL=0

if [[ "$E12B_HGV_G1_SEED15_CKPT" != "$CKPT" ]]; then
    echo "[FAIL] resolved checkpoint mismatch"
    FAIL=1
else
    echo "[PASS] exact G1 seed20260915 best checkpoint frozen"
fi

if [[ ! -s "$E12B_REFS_TSV" ]]; then
    echo "[FAIL] reference manifest missing"
    FAIL=1
else
    echo "[PASS] reference manifest"
fi

if [[ ! -s "$E12B_QUERIES_TSV" ]]; then
    echo "[FAIL] query manifest missing"
    FAIL=1
else
    echo "[PASS] query manifest"
fi

for tool in mash dashing2 skani fastANI; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "[PASS] $tool available"
    else
        echo "[FAIL] $tool missing"
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo
    echo "E12B STAGE 00B = FAIL"
    exit 2
fi

echo
echo "E12B STAGE 00B = PASS"
echo
echo "Next input files:"
echo "  $ENVFILE"
echo "  $PREF/CANONICAL_K6_ASSETS.tsv"
echo "  $PREF/PCA_ARTIFACT_EXACT.txt"
echo "  $PREF/E5_QUERY_TABLE_SCHEMA.txt"
echo "  $PREF/MEMORY_MEASUREMENT_SUPPORT.txt"
echo
echo "No model was trained."
echo "No benchmark was run."
echo "No frozen historical result was modified."
