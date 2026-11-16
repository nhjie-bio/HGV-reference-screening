#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import pandas as pd

BASE = Path("HGV_V3_revision_20260915")

ROOT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "05_fastani/internal_bidirectional_v2/results"
)

OUT = (
    BASE /
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "05_fastani/internal_bidirectional_v2/summary"
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
asymmetry_rows = []
run_rows = []
sha_rows = []

print("=" * 88)
print("E5 FASTANI INTERNAL BIDIRECTIONAL v2 SUMMARY")
print("=" * 88)

for protocol in ["G2", "G3", "G4"]:

    for seed in ["20260915", "20260916", "20260917"]:

        d = ROOT / f"{protocol}_seed{seed}"

        result_file = (
            d /
            "fastani_symmetric_fixed_pair_results.tsv"
        )

        coverage_file = (
            d /
            "fastani_bidirectional_coverage.tsv"
        )

        audit_file = (
            d /
            "FASTANI_BIDIRECTIONAL_AUDIT.json"
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
                    f"Required file missing: {p}"
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
                f"expected {expected} pairs, "
                f"got {len(df)}"
            )

        required_cols = [
            "fastani_pair_status",
            "fastani_symmetric_ani",
        ]

        for c in required_cols:
            if c not in df.columns:
                raise RuntimeError(
                    f"{protocol} seed{seed}: "
                    f"missing column {c}"
                )

        n_both = int(
            (
                df["fastani_pair_status"]
                == "BOTH"
            ).sum()
        )

        n_one = int(
            (
                df["fastani_pair_status"]
                == "ONE_DIRECTION_ONLY"
            ).sum()
        )

        n_none = int(
            (
                df["fastani_pair_status"]
                == "NONE"
            ).sum()
        )

        if (
            n_both +
            n_one +
            n_none
            != expected
        ):
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                "pair-status counts do not "
                "sum to expected count"
            )

        n_sym = int(
            df[
                "fastani_symmetric_ani"
            ].notna().sum()
        )

        if n_sym != n_both:
            raise RuntimeError(
                f"{protocol} seed{seed}: "
                f"symmetric ANI count {n_sym} "
                f"!= BOTH count {n_both}"
            )

        # Coverage table
        for r in cov.itertuples(index=False):

            coverage_rows.append({
                "protocol":
                    protocol,

                "split_seed":
                    seed,

                "stratum":
                    r.stratum,

                "n_expected":
                    int(r.n_expected),

                "n_both":
                    int(r.n_both),

                "n_one_direction":
                    int(r.n_one_direction),

                "n_none":
                    int(r.n_none),

                "coverage_both":
                    float(r.coverage_both),

                "coverage_any":
                    float(r.coverage_any),
            })

        asym = audit.get(
            "asymmetry",
            {}
        )

        asymmetry_rows.append({
            "protocol":
                protocol,

            "split_seed":
                seed,

            "n_bidirectional":
                asym.get(
                    "n_bidirectional"
                ),

            "mean_abs_delta":
                asym.get(
                    "mean_abs_delta"
                ),

            "median_abs_delta":
                asym.get(
                    "median_abs_delta"
                ),

            "max_abs_delta":
                asym.get(
                    "max_abs_delta"
                ),

            "wall_seconds":
                audit.get(
                    "wall_seconds"
                ),

            "query_jobs":
                audit.get(
                    "query_jobs"
                ),

            "workers":
                audit.get(
                    "workers"
                ),
        })

        run_rows.append({
            "protocol":
                protocol,

            "split_seed":
                seed,

            "n_expected":
                expected,

            "n_both":
                n_both,

            "n_one_direction":
                n_one,

            "n_none":
                n_none,

            "coverage_both":
                n_both / expected,

            "coverage_any":
                (n_both + n_one) / expected,

            "n_symmetric_ani":
                n_sym,

            "status":
                "PASS",
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

        print(
            f"expected={expected}  "
            f"BOTH={n_both}  "
            f"ONE={n_one}  "
            f"NONE={n_none}  "
            f"coverage_both="
            f"{n_both / expected:.6f}"
        )


coverage_all = pd.DataFrame(
    coverage_rows
)

asymmetry_all = pd.DataFrame(
    asymmetry_rows
)

run_all = pd.DataFrame(
    run_rows
)

sha_all = pd.DataFrame(
    sha_rows
)


# ------------------------------------------------------------
# Save all split-level outputs
# ------------------------------------------------------------

coverage_all.to_csv(
    OUT /
    "fastani_internal_coverage_all_splits.tsv",
    sep="\t",
    index=False,
)

asymmetry_all.to_csv(
    OUT /
    "fastani_internal_directional_asymmetry.tsv",
    sep="\t",
    index=False,
)

run_all.to_csv(
    OUT /
    "fastani_internal_run_summary.tsv",
    sep="\t",
    index=False,
)

sha_all.to_csv(
    OUT /
    "fastani_internal_result_sha256.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# Protocol-level mean ± SD over taxonomy split seeds
#
# IMPORTANT:
# split seed is the protocol variability unit.
# We do NOT pool all individual pairs as independent replicates.
# ------------------------------------------------------------

protocol_summary = (
    coverage_all
    .groupby(
        [
            "protocol",
            "stratum",
        ],
        as_index=False,
    )
    .agg(
        n_splits=(
            "split_seed",
            "count",
        ),

        coverage_both_mean=(
            "coverage_both",
            "mean",
        ),

        coverage_both_sd=(
            "coverage_both",
            "std",
        ),

        coverage_any_mean=(
            "coverage_any",
            "mean",
        ),

        coverage_any_sd=(
            "coverage_any",
            "std",
        ),

        n_expected_total=(
            "n_expected",
            "sum",
        ),

        n_both_total=(
            "n_both",
            "sum",
        ),

        n_one_direction_total=(
            "n_one_direction",
            "sum",
        ),

        n_none_total=(
            "n_none",
            "sum",
        ),
    )
)

protocol_summary.to_csv(
    OUT /
    "fastani_internal_coverage_protocol_summary.tsv",
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# Critical global audits
# ------------------------------------------------------------

all_rows = coverage_all[
    coverage_all["stratum"] == "ALL"
].copy()

positive_rows = coverage_all[
    coverage_all["stratum"] == "LABEL_1"
].copy()

negative_rows = coverage_all[
    coverage_all["stratum"] == "LABEL_0"
].copy()

total_expected = int(
    all_rows["n_expected"].sum()
)

total_both = int(
    all_rows["n_both"].sum()
)

total_one = int(
    all_rows[
        "n_one_direction"
    ].sum()
)

total_none = int(
    all_rows["n_none"].sum()
)

if total_expected != sum(EXPECTED.values()):
    raise RuntimeError(
        "Global expected-pair count mismatch"
    )

positive_complete = bool(
    (
        positive_rows[
            "coverage_both"
        ] == 1.0
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
print("PROTOCOL-LEVEL COVERAGE SUMMARY")
print("=" * 88)

print(
    protocol_summary.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("DIRECTIONAL ASYMMETRY")
print("=" * 88)

print(
    asymmetry_all.to_string(
        index=False
    )
)

print()
print("=" * 88)
print("GLOBAL AUDIT")
print("=" * 88)

print(
    f"total expected pairs       = "
    f"{total_expected}"
)

print(
    f"bidirectional reported     = "
    f"{total_both}"
)

print(
    f"one-direction-only pairs   = "
    f"{total_one}"
)

print(
    f"no-output pairs            = "
    f"{total_none}"
)

print(
    f"overall symmetric coverage = "
    f"{total_both / total_expected:.6f}"
)

print(
    f"all positive coverage=1.0  = "
    f"{positive_complete}"
)


# ------------------------------------------------------------
# Text report
# ------------------------------------------------------------

report_file = (
    OUT /
    "FASTANI_INTERNAL_V2_SUMMARY_REPORT.txt"
)

with open(report_file, "w") as f:

    f.write(
        "E5 FastANI Internal Bidirectional "
        "Benchmark v2\n"
    )

    f.write(
        "=" * 70 +
        "\n\n"
    )

    f.write(
        "Frozen protocols: "
        "G2, G3, G4\n"
    )

    f.write(
        "Split seeds: "
        "20260915, 20260916, "
        "20260917\n"
    )

    f.write(
        "Pair-level FastANI score: "
        "mean of A->B and B->A ANI "
        "when both directions are "
        "reported.\n"
    )

    f.write(
        "One-direction-only and no-output "
        "pairs remain missing for the "
        "symmetric ANI score.\n\n"
    )

    f.write(
        f"Total expected pairs: "
        f"{total_expected}\n"
    )

    f.write(
        f"Bidirectional reported: "
        f"{total_both}\n"
    )

    f.write(
        f"One-direction-only: "
        f"{total_one}\n"
    )

    f.write(
        f"No output: "
        f"{total_none}\n"
    )

    f.write(
        f"Overall symmetric coverage: "
        f"{total_both / total_expected:.6f}\n"
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
        "\n\nDirectional asymmetry\n"
    )

    f.write(
        "----------------------\n"
    )

    f.write(
        asymmetry_all.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: PASS\n"
    )


print()
print("Output directory:")
print(OUT)

print()
print("Report:")
print(report_file)

print()
print("STATUS = PASS")
