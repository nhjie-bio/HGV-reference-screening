#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import hashlib

BASE = Path("HGV_V3_revision_20260915")

SRCROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "00_native_validation/tasks"
)

OUTROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "00_native_validation/compat_tasks"
)

EXPECTED = {
    ("G2", "20260915"): 3874,
    ("G2", "20260916"): 3872,
    ("G2", "20260917"): 3874,
    ("G3", "20260915"): 3764,
    ("G3", "20260916"): 3764,
    ("G3", "20260917"): 3830,
    ("G4", "20260915"): 3916,
    ("G4", "20260916"): 3786,
    ("G4", "20260917"): 3832,
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

rows = []
total = 0

for protocol in ["G2", "G3", "G4"]:
    for seed in ["20260915", "20260916", "20260917"]:

        src = (
            SRCROOT /
            f"{protocol}_seed{seed}" /
            "native_validation_pairs.tsv"
        )

        df = pd.read_csv(src, sep="\t", dtype=str)

        expected = EXPECTED[(protocol, seed)]

        if len(df) != expected:
            raise RuntimeError(
                f"{protocol} {seed}: "
                f"expected {expected}, got {len(df)}"
            )

        required = [
            "pair_id",
            "accession_a",
            "accession_b",
            "fasta_a",
            "fasta_b",
            "label",
        ]

        for c in required:
            if c not in df.columns:
                raise RuntimeError(
                    f"{protocol} {seed}: missing {c}"
                )

        if "pair_index" in df.columns:
            raise RuntimeError(
                f"{protocol} {seed}: pair_index already exists"
            )

        # Compatibility identifier only.
        # Original pair_id remains unchanged.
        insert_at = df.columns.get_loc("pair_id") + 1
        df.insert(
            insert_at,
            "pair_index",
            df["pair_id"].astype(str)
        )

        if df["pair_index"].duplicated().any():
            raise RuntimeError(
                f"{protocol} {seed}: duplicate pair_index"
            )

        outdir = (
            OUTROOT /
            f"{protocol}_seed{seed}"
        )
        outdir.mkdir(parents=True, exist_ok=True)

        outfile = (
            outdir /
            "fastani_fixed_pairs.tsv"
        )

        df.to_csv(
            outfile,
            sep="\t",
            index=False
        )

        rows.append({
            "protocol": protocol,
            "split_seed": seed,
            "n_pairs": len(df),
            "source": str(src),
            "output": str(outfile),
            "sha256": sha256(outfile),
        })

        total += len(df)

        print(
            f"{protocol} seed{seed}: "
            f"{len(df)} PASS"
        )

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTROOT /
    "VALIDATION_COMPAT_TASK_SUMMARY.tsv",
    sep="\t",
    index=False
)

print()
print(summary[
    ["protocol", "split_seed", "n_pairs"]
].to_string(index=False))

print()
print("total =", total)

if total != 34512:
    raise RuntimeError(
        f"Expected total 34512, got {total}"
    )

print()
print("STATUS = PASS")
