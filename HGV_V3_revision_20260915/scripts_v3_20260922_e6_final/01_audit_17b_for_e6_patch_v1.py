#!/usr/bin/env python3

from pathlib import Path
import hashlib
import re

BASE = Path("HGV_V3_revision_20260915")

SRC = (
    BASE
    / "scripts_v3_20260916_hgv"
    / "17b_train_main_hierarchical_v1.py"
)

OUT = (
    BASE
    / "scripts_v3_20260922_e6_final"
    / "01_AUDIT_17B_FOR_E6_PATCH.txt"
)

if not SRC.exists():
    raise FileNotFoundError(f"Missing source: {SRC}")

text = SRC.read_text()
lines = text.splitlines()

sha = hashlib.sha256(SRC.read_bytes()).hexdigest()

patterns = [
    r"ArgumentParser",
    r"add_argument",
    r"def main",
    r"if __name__",

    r"class ",
    r"BatchNorm",
    r"F\.normalize",
    r"normalize\(",

    r"deterministic",
    r"CUBLAS",
    r"manual_seed",
    r"default_rng",
    r"1000003",
    r"epoch_seed",

    r"50000",
    r"50_000",
    r"100000",
    r"100_000",
    r"replace=True",
    r"positive",
    r"negative",

    r"contrastive",
    r"margin",
    r"teacher",
    r"tau",
    r"lambda",
    r"MSE",
    r"mse_loss",
    r"0\.98",
    r"0\.25",

    r"torch\.cat",
    r"chunk",

    r"roc_auc",
    r"average_precision",
    r"AUROC",
    r"AUPRC",
    r"1e-8",
    r"patience",
    r"best_",
    r"checkpoint",
    r"torch\.save",
]

hits = []

for i, line in enumerate(lines):
    for pat in patterns:
        if re.search(pat, line, flags=re.IGNORECASE):
            hits.append(i)
            break

ranges = []
PAD = 12

for i in sorted(set(hits)):
    start = max(0, i - PAD)
    end = min(len(lines), i + PAD + 1)

    if not ranges or start > ranges[-1][1] + 1:
        ranges.append([start, end])
    else:
        ranges[-1][1] = max(ranges[-1][1], end)

with OUT.open("w") as f:

    def emit(s=""):
        print(s)
        f.write(s + "\n")

    emit("=" * 100)
    emit("HGV-V3 E6 SOURCE AUDIT")
    emit("=" * 100)
    emit(f"SOURCE={SRC}")
    emit(f"SHA256={sha}")
    emit(f"LINES={len(lines)}")
    emit(f"MATCHED_CONTEXT_BLOCKS={len(ranges)}")
    emit("")

    for n, (start, end) in enumerate(ranges, 1):
        emit("=" * 100)
        emit(f"BLOCK {n}: lines {start + 1}-{end}")
        emit("=" * 100)

        for j in range(start, end):
            emit(f"{j + 1:05d}: {lines[j]}")

        emit("")

    emit("=" * 100)
    emit("AUDIT_COMPLETE=PASS")
    emit("=" * 100)

print()
print("Audit written to:")
print(OUT)
