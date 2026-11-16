#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import pandas as pd


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

MANIFEST = (
    BASE /
    "01_manifest/"
    "final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

PAIRROOT = (
    BASE /
    "05_pairs/"
    "pair_manifests_v1_20260916"
)

OUTROOT = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917/"
    "00_native_validation/tasks"
)

OUTROOT.mkdir(
    parents=True,
    exist_ok=True,
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


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def canon_pair(a, b):
    a = str(a)
    b = str(b)

    if a == b:
        raise RuntimeError(
            f"Self pair detected: {a}"
        )

    return (
        (a, b)
        if a < b
        else (b, a)
    )


print("=" * 86)
print(
    "E5 NATIVE BASELINE "
    "VALIDATION TASK PREPARATION v1"
)
print("=" * 86)

manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
    dtype=str,
)

required_manifest = [
    "assembly_accession_final",
    "fasta_path",
]

for c in required_manifest:
    if c not in manifest.columns:
        raise RuntimeError(
            f"Manifest missing {c}"
        )

if manifest[
    "assembly_accession_final"
].duplicated().any():
    raise RuntimeError(
        "Duplicate accession in "
        "frozen manifest"
    )

acc_to_path = dict(
    zip(
        manifest[
            "assembly_accession_final"
        ],
        manifest["fasta_path"],
    )
)

print(
    "manifest genomes =",
    len(acc_to_path)
)

summary = []

total_pairs = 0

for protocol in [
    "G2",
    "G3",
    "G4",
]:

    for seed in [
        "20260915",
        "20260916",
        "20260917",
    ]:

        src = (
            PAIRROOT /
            f"{protocol}_seed{seed}" /
            "val_pairs.tsv"
        )

        if not src.is_file():
            raise FileNotFoundError(
                src
            )

        df = pd.read_csv(
            src,
            sep="\t",
            dtype=str,
        )

        expected = EXPECTED[
            (protocol, seed)
        ]

        if len(df) != expected:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"expected {expected} "
                f"validation pairs, "
                f"found {len(df)}"
            )

        required_cols = [
            "pair_id",
            "accession_a",
            "accession_b",
            "label",
            "pair_type",
        ]

        for c in required_cols:
            if c not in df.columns:
                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    f"missing {c}"
                )

        unknown_a = (
            set(
                df["accession_a"]
            ) -
            set(acc_to_path)
        )

        unknown_b = (
            set(
                df["accession_b"]
            ) -
            set(acc_to_path)
        )

        if unknown_a or unknown_b:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "unknown accession(s)"
            )

        keys = [
            canon_pair(a, b)
            for a, b
            in zip(
                df["accession_a"],
                df["accession_b"],
            )
        ]

        duplicate_pairs = (
            len(keys) -
            len(set(keys))
        )

        self_pairs = sum(
            a == b
            for a, b
            in zip(
                df["accession_a"],
                df["accession_b"],
            )
        )

        if duplicate_pairs:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"{duplicate_pairs} "
                "duplicate pair(s)"
            )

        if self_pairs:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"{self_pairs} "
                "self pair(s)"
            )

        outdir = (
            OUTROOT /
            f"{protocol}_seed{seed}"
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        task = df.copy()

        task.insert(
            task.columns.get_loc(
                "accession_a"
            ) + 1,
            "fasta_a",
            [
                acc_to_path[x]
                for x in task[
                    "accession_a"
                ]
            ],
        )

        bpos = (
            task.columns.get_loc(
                "accession_b"
            ) + 1
        )

        task.insert(
            bpos,
            "fasta_b",
            [
                acc_to_path[x]
                for x in task[
                    "accession_b"
                ]
            ],
        )

        missing_fasta = []

        for p in (
            list(task["fasta_a"]) +
            list(task["fasta_b"])
        ):
            if not Path(p).is_file():
                missing_fasta.append(p)

        if missing_fasta:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"{len(missing_fasta)} "
                "missing FASTA(s)"
            )

        outfile = (
            outdir /
            "native_validation_pairs.tsv"
        )

        task.to_csv(
            outfile,
            sep="\t",
            index=False,
        )

        n_pos = int(
            (
                task["label"] == "1"
            ).sum()
        )

        n_neg = int(
            (
                task["label"] == "0"
            ).sum()
        )

        if n_pos + n_neg != len(task):
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "non-binary label detected"
            )

        report = {
            "protocol":
                protocol,

            "split_seed":
                seed,

            "source":
                str(src),

            "source_sha256":
                sha256_file(src),

            "output":
                str(outfile),

            "output_sha256":
                sha256_file(outfile),

            "n_pairs":
                len(task),

            "n_positive":
                n_pos,

            "n_negative":
                n_neg,

            "n_unique_genomes":
                len(
                    set(task["accession_a"]) |
                    set(task["accession_b"])
                ),

            "duplicate_pairs":
                0,

            "self_pairs":
                0,

            "missing_fasta":
                0,

            "status":
                "PASS",
        }

        with open(
            outdir /
            "VALIDATION_TASK_AUDIT.json",
            "w",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        summary.append(report)

        total_pairs += len(task)

        print(
            f"{protocol} seed{seed}: "
            f"pairs={len(task)} "
            f"pos={n_pos} "
            f"neg={n_neg} "
            f"genomes="
            f"{report['n_unique_genomes']} "
            "PASS"
        )


summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    OUTROOT /
    "NATIVE_VALIDATION_TASK_SUMMARY.tsv",
    sep="\t",
    index=False,
)

print()
print("=" * 86)
print("VALIDATION TASK SUMMARY")
print("=" * 86)

print(
    summary_df[
        [
            "protocol",
            "split_seed",
            "n_pairs",
            "n_positive",
            "n_negative",
            "n_unique_genomes",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "total validation pair "
    f"occurrences = {total_pairs}"
)

print()
print("STATUS = PASS")
