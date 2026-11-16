#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import hashlib
import json
import sys

BASE = Path("HGV_V3_revision_20260915")

MANIFEST = (
    BASE /
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

PAIRROOT = (
    BASE /
    "05_pairs/pair_manifests_v1_20260916"
)

OUTROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "05_fastani/internal/tasks"
)

OUTROOT.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def norm_acc(x):
    x = str(x).strip()

    if x.startswith("RS_") or x.startswith("GB_"):
        x = x[3:]

    return x


print("===== LOAD FINAL MANIFEST =====")

m = pd.read_csv(MANIFEST, sep="\t", dtype=str)

required_manifest_cols = [
    "assembly_accession_final",
    "fasta_path",
    "gtdb_species_cluster",
    "gtdb_genus",
    "gtdb_family",
]

for c in required_manifest_cols:
    if c not in m.columns:
        raise RuntimeError(f"Missing manifest column: {c}")

m["norm_acc"] = m["assembly_accession_final"].map(norm_acc)

if m["norm_acc"].duplicated().any():
    dup = m.loc[m["norm_acc"].duplicated(), "norm_acc"].tolist()[:20]
    raise RuntimeError(f"Duplicate normalized accession(s): {dup}")

acc_to_path = dict(zip(m["norm_acc"], m["fasta_path"]))

print(f"manifest rows = {len(m)}")
print(f"unique accession = {len(acc_to_path)}")

missing_fasta = [
    (a, p)
    for a, p in acc_to_path.items()
    if not Path(p).is_file()
]

print(f"missing FASTA = {len(missing_fasta)}")

if missing_fasta:
    print(missing_fasta[:20])
    raise RuntimeError("Manifest contains missing FASTA path(s).")


PAIR_CANDIDATES = [
    ("assembly_accession_a", "assembly_accession_b"),
    ("accession_a", "accession_b"),
    ("genome_a", "genome_b"),
    ("query_accession", "reference_accession"),
    ("assembly_a", "assembly_b"),
    ("acc_a", "acc_b"),
    ("acc1", "acc2"),
    ("genome1", "genome2"),
]


def detect_pair_columns(df):
    # First prefer explicit conventional pairs.
    for a, b in PAIR_CANDIDATES:
        if a in df.columns and b in df.columns:
            return a, b, "explicit"

    # Otherwise infer columns by accession coverage.
    candidates = []

    manifest_ids = set(acc_to_path)

    for c in df.columns:
        vals = df[c].dropna().astype(str).map(norm_acc)

        if len(vals) == 0:
            continue

        sample = vals.iloc[: min(500, len(vals))]
        score = sample.isin(manifest_ids).mean()

        if score >= 0.95:
            candidates.append((c, score))

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not detect two accession columns.\n"
            f"Columns: {list(df.columns)}\n"
            f"Candidate matches: {candidates}"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[0][0], candidates[1][0], "inferred"


summary = []

for protocol in ["G2", "G3", "G4"]:
    for seed in ["20260915", "20260916", "20260917"]:

        src = (
            PAIRROOT /
            f"{protocol}_seed{seed}" /
            "test_pairs.tsv"
        )

        print()
        print("=" * 70)
        print(protocol, seed)
        print(src)

        if not src.is_file():
            raise FileNotFoundError(src)

        df = pd.read_csv(src, sep="\t", dtype=str)

        a_col, b_col, mode = detect_pair_columns(df)

        print(f"pair columns = {a_col}, {b_col} ({mode})")
        print(f"test pairs   = {len(df)}")

        a = df[a_col].map(norm_acc)
        b = df[b_col].map(norm_acc)

        missing_a = sorted(set(a) - set(acc_to_path))
        missing_b = sorted(set(b) - set(acc_to_path))

        if missing_a or missing_b:
            print("missing accession A:", missing_a[:20])
            print("missing accession B:", missing_b[:20])
            raise RuntimeError("Pair accession not found in frozen manifest.")

        outdir = OUTROOT / f"{protocol}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)

        norm = pd.DataFrame({
            "pair_index": range(len(df)),
            "protocol": protocol,
            "split_seed": seed,
            "accession_a": a,
            "accession_b": b,
            "fasta_a": [acc_to_path[x] for x in a],
            "fasta_b": [acc_to_path[x] for x in b],
        })

        # Preserve label if available.
        for label_col in [
            "label",
            "target",
            "same_species",
            "y",
            "is_same_species",
        ]:
            if label_col in df.columns:
                norm["label"] = df[label_col].values
                print(f"label column = {label_col}")
                break

        out_pairs = outdir / "fastani_fixed_pairs.tsv"
        norm.to_csv(out_pairs, sep="\t", index=False)

        # Unique FASTA inventory for this split.
        unique_paths = sorted(
            set(norm["fasta_a"]) |
            set(norm["fasta_b"])
        )

        with open(outdir / "unique_test_fasta.list", "w") as f:
            for p in unique_paths:
                f.write(p + "\n")

        # Pair uniqueness check.
        canonical_pair_keys = [
            tuple(sorted((x, y)))
            for x, y in zip(a, b)
        ]

        unique_pairs = len(set(canonical_pair_keys))
        duplicate_pairs = len(df) - unique_pairs

        # Self-pair check.
        self_pairs = sum(x == y for x, y in zip(a, b))

        report = {
            "protocol": protocol,
            "split_seed": seed,
            "source_pair_manifest": str(src),
            "source_pair_sha256": sha256_file(src),
            "pair_column_a": a_col,
            "pair_column_b": b_col,
            "pair_detection_mode": mode,
            "n_pairs": len(df),
            "n_unique_canonical_pairs": unique_pairs,
            "duplicate_pairs": duplicate_pairs,
            "self_pairs": self_pairs,
            "n_unique_test_genomes": len(unique_paths),
            "missing_manifest_accessions": 0,
            "missing_fasta": 0,
            "normalized_task_sha256": sha256_file(out_pairs),
        }

        with open(outdir / "TASK_AUDIT.json", "w") as f:
            json.dump(report, f, indent=2)

        if duplicate_pairs != 0:
            raise RuntimeError(
                f"{protocol} seed{seed}: duplicate pair(s) detected"
            )

        if self_pairs != 0:
            raise RuntimeError(
                f"{protocol} seed{seed}: self pair(s) detected"
            )

        summary.append(report)

        print(f"unique canonical pairs = {unique_pairs}")
        print(f"unique genomes         = {len(unique_paths)}")
        print("TASK AUDIT = PASS")


summary_df = pd.DataFrame(summary)

summary_path = OUTROOT / "FASTANI_INTERNAL_TASK_SUMMARY.tsv"
summary_df.to_csv(summary_path, sep="\t", index=False)

with open(OUTROOT / "FASTANI_INTERNAL_TASK_FREEZE_REPORT.txt", "w") as f:
    f.write("E5 FastANI internal fixed-pair task preparation\n")
    f.write("===============================================\n\n")
    f.write(f"Manifest: {MANIFEST}\n")
    f.write(f"Manifest SHA256: {sha256_file(MANIFEST)}\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\nSTATUS: PASS\n")

print()
print("=" * 70)
print("ALL FASTANI INTERNAL TASKS PREPARED")
print(summary_df[
    [
        "protocol",
        "split_seed",
        "n_pairs",
        "n_unique_test_genomes",
        "duplicate_pairs",
        "self_pairs",
    ]
].to_string(index=False))

print()
print("Output:")
print(summary_path)
print()
print("STATUS = PASS")
