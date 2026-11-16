#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import pandas as pd

BASE = Path("HGV_V3_revision_20260915")

ROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "04_skani/internal_v1/results"
)

OUT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "04_skani/internal_v1/summary"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ("G2", "20260915"): 3872,
    ("G2", "20260916"): 3874,
    ("G2", "20260917"): 3874,
    ("G3", "20260915"): 3916,
    ("G3", "20260916"): 3852,
    ("G3", "20260917"): 3894,
    ("G4", "20260915"): 3832,
    ("G4", "20260916"): 3894,
    ("G4", "20260917"): 3874,
}


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)
            if not x:
                break
            h.update(x)

    return h.hexdigest()


coverage_rows = []
run_rows = []
sha_rows = []

print("=" * 88)
print("E5 SKANI INTERNAL v1 SUMMARY")
print("=" * 88)

for protocol in ["G2", "G3", "G4"]:
    for seed in ["20260915", "20260916", "20260917"]:

        d = ROOT / f"{protocol}_seed{seed}"

        result_file = (
            d /
            "skani_fixed_pair_results.tsv"
        )

        coverage_file = (
            d /
            "skani_coverage_summary.tsv"
        )

        audit_file = (
            d /
            "SKANI_RUN_AUDIT.json"
        )

        print()
        print(
            f"===== {protocol} seed{seed} ====="
        )

        for p in [
            result_file,
            coverage_file,
            audit_file,
        ]:
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing required file: {p}"
                )

        df = pd.read_csv(
            result_file,
            sep="\t",
        )

        cov = pd.read_csv(
            coverage_file,
            sep="\t",
        )

        with open(audit_file) as f:
            audit = json.load(f)

        expected = EXPECTED[
            (protocol, seed)
        ]

        if len(df) != expected:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"expected {expected}, "
                f"got {len(df)}"
            )

        if "skani_reported" not in df.columns:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "missing skani_reported"
            )

        if "skani_ani" not in df.columns:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "missing skani_ani"
            )

        n_reported = int(
            df["skani_reported"].sum()
        )

        n_missing = (
            expected - n_reported
        )

        n_ani = int(
            df["skani_ani"].notna().sum()
        )

        if n_ani != n_reported:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"ANI count {n_ani} != "
                f"reported count {n_reported}"
            )

        print(
            f"expected={expected} "
            f"reported={n_reported} "
            f"missing={n_missing} "
            f"coverage="
            f"{n_reported / expected:.6f}"
        )

        run_rows.append({
            "protocol":
                protocol,
            "split_seed":
                seed,
            "n_expected":
                expected,
            "n_reported":
                n_reported,
            "n_missing":
                n_missing,
            "coverage":
                n_reported / expected,
            "query_jobs":
                audit.get("query_jobs"),
            "wall_seconds":
                audit.get("wall_seconds"),
            "status":
                "PASS",
        })

        for r in cov.itertuples(
            index=False
        ):
            coverage_rows.append({
                "protocol":
                    protocol,
                "split_seed":
                    seed,
                "stratum":
                    r.stratum,
                "n_expected":
                    int(r.n_expected),
                "n_reported":
                    int(r.n_reported),
                "n_missing":
                    int(r.n_missing),
                "coverage":
                    float(r.coverage),
            })

        for p in [
            result_file,
            coverage_file,
            audit_file,
        ]:
            sha_rows.append({
                "protocol":
                    protocol,
                "split_seed":
                    seed,
                "file":
                    str(p),
                "sha256":
                    sha256_file(p),
            })


coverage_all = pd.DataFrame(
    coverage_rows
)

run_all = pd.DataFrame(
    run_rows
)

sha_all = pd.DataFrame(
    sha_rows
)


coverage_all.to_csv(
    OUT /
    "skani_internal_coverage_all_splits.tsv",
    sep="\t",
    index=False,
)

run_all.to_csv(
    OUT /
    "skani_internal_run_summary.tsv",
    sep="\t",
    index=False,
)

sha_all.to_csv(
    OUT /
    "skani_internal_result_sha256.tsv",
    sep="\t",
    index=False,
)


protocol_summary = (
    coverage_all
    .groupby(
        ["protocol", "stratum"],
        as_index=False,
    )
    .agg(
        n_splits=(
            "split_seed",
            "count",
        ),
        coverage_mean=(
            "coverage",
            "mean",
        ),
        coverage_sd=(
            "coverage",
            "std",
        ),
        n_expected_total=(
            "n_expected",
            "sum",
        ),
        n_reported_total=(
            "n_reported",
            "sum",
        ),
        n_missing_total=(
            "n_missing",
            "sum",
        ),
    )
)

protocol_summary.to_csv(
    OUT /
    "skani_internal_coverage_protocol_summary.tsv",
    sep="\t",
    index=False,
)


all_rows = coverage_all[
    coverage_all["stratum"] == "ALL"
]

positive_rows = coverage_all[
    coverage_all["stratum"]
    == "LABEL_1"
]

negative_rows = coverage_all[
    coverage_all["stratum"]
    == "LABEL_0"
]

total_expected = int(
    all_rows["n_expected"].sum()
)

total_reported = int(
    all_rows["n_reported"].sum()
)

total_missing = int(
    all_rows["n_missing"].sum()
)

if total_expected != 34882:
    raise RuntimeError(
        "Global pair count is not 34882"
    )

positive_complete = bool(
    (
        positive_rows["coverage"]
        == 1.0
    ).all()
)


print()
print("=" * 88)
print("ALL SPLIT COVERAGE")
print("=" * 88)

print(
    coverage_all.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("PROTOCOL-LEVEL SUMMARY")
print("=" * 88)

print(
    protocol_summary.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("GLOBAL AUDIT")
print("=" * 88)

print(
    f"total expected pairs  = "
    f"{total_expected}"
)

print(
    f"reported pairs        = "
    f"{total_reported}"
)

print(
    f"missing pairs         = "
    f"{total_missing}"
)

print(
    f"overall coverage      = "
    f"{total_reported / total_expected:.6f}"
)

print(
    f"all positive coverage=1.0 = "
    f"{positive_complete}"
)


report_file = (
    OUT /
    "SKANI_INTERNAL_V1_SUMMARY_REPORT.txt"
)

with open(report_file, "w") as f:

    f.write(
        "E5 skani Internal Benchmark v1\n"
    )

    f.write(
        "=" * 70 +
        "\n\n"
    )

    f.write(
        "skani version: 0.3.2\n"
    )

    f.write(
        "Protocols: G2/G3/G4\n"
    )

    f.write(
        "Split seeds: "
        "20260915/20260916/20260917\n"
    )

    f.write(
        "Missing native ANI output "
        "remains missing.\n"
    )

    f.write(
        "No missing ANI is imputed.\n\n"
    )

    f.write(
        f"Total expected pairs: "
        f"{total_expected}\n"
    )

    f.write(
        f"Reported pairs: "
        f"{total_reported}\n"
    )

    f.write(
        f"Missing pairs: "
        f"{total_missing}\n"
    )

    f.write(
        f"Overall coverage: "
        f"{total_reported / total_expected:.6f}\n"
    )

    f.write(
        f"All positive strata "
        f"coverage=1.0: "
        f"{positive_complete}\n\n"
    )

    f.write(
        "Protocol-level summary\n"
    )

    f.write(
        "----------------------\n"
    )

    f.write(
        protocol_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: PASS\n"
    )


print()
print("Output:")
print(OUT)

print()
print("Report:")
print(report_file)

print()
print("STATUS = PASS")
