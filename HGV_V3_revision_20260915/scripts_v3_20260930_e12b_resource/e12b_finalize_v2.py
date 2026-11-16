#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import statistics

import numpy as np
import pandas as pd


BASE = Path("HGV_V3_revision_20260915")
OUT = (
    BASE
    / "16_scalability/"
      "E12B_synchronized_resource_v1_20260930"
)

RUN = OUT / "03_runs"
SUM = OUT / "04_summary/finalize_v2_20261001"
FREEZE = OUT / "05_final_freeze_v2_20261001"

SUM.mkdir(parents=True, exist_ok=True)
FREEZE.mkdir(parents=True, exist_ok=True)

ACC_PATH = (
    OUT
    / "04_summary/"
      "E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv"
)

if not ACC_PATH.exists():
    raise RuntimeError(
        f"Missing frozen Q50 accuracy table: {ACC_PATH}"
    )

ACC = pd.read_csv(
    ACC_PATH,
    sep="\t",
)

METHODS = [
    "HGV",
    "Raw6_BrayCurtis",
    "Raw6_Cosine",
    "Raw6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
    "DNABERT-S",
]

EXPECTED = set(METHODS)
OBSERVED = set(ACC["method"].astype(str))

if EXPECTED != OBSERVED:
    raise RuntimeError(
        "Accuracy method mismatch.\n"
        f"Expected={sorted(EXPECTED)}\n"
        f"Observed={sorted(OBSERVED)}"
    )

accmap = {
    str(row["method"]): row
    for _, row in ACC.iterrows()
}


def resource_path(method, rep, phase):
    return (
        RUN
        / method
        / f"rep{rep}"
        / phase
        / "resource"
        / "resource.json"
    )


def read_resource(method, rep, phase):
    p = resource_path(method, rep, phase)

    if not p.exists():
        return None

    x = json.loads(p.read_text())

    if int(x.get("exit_code", -999)) != 0:
        raise RuntimeError(
            f"Nonzero resource record: "
            f"{method} rep{rep} {phase}: {p}"
        )

    return x


# ------------------------------------------------------------------
# 1. Long-format audit of every retained synchronized measurement
# ------------------------------------------------------------------

long_rows = []

for method in METHODS:

    phases = (
        ["full"]
        if method == "FastANI"
        else ["build", "query", "full"]
    )

    for rep in (1, 2, 3):

        for phase in phases:

            x = read_resource(
                method,
                rep,
                phase,
            )

            if x is None:
                raise RuntimeError(
                    f"Missing resource measurement: "
                    f"{method} rep{rep} {phase}"
                )

            long_rows.append(
                {
                    "method": method,
                    "repeat": rep,
                    "phase": phase,
                    "wall_s": float(x["wall_s"]),
                    "peak_host_pss_mib":
                        float(x["peak_host_pss_mib"]),
                    "peak_gpu_mib":
                        float(x.get("peak_gpu_mib", 0.0)),
                    "sample_interval_s":
                        float(x.get("sample_interval_s", np.nan)),
                    "samples":
                        int(x.get("samples", 0)),
                    "resource_json":
                        str(resource_path(method, rep, phase)),
                }
            )

LONG = pd.DataFrame(long_rows)

LONG.to_csv(
    SUM / "E12B_RESOURCE_REPLICATES_LONG.tsv",
    sep="\t",
    index=False,
    float_format="%.6f",
)


def median_phase(method, phase, field):
    x = LONG[
        (LONG["method"] == method)
        &
        (LONG["phase"] == phase)
    ]

    if len(x) == 0:
        return np.nan

    vals = pd.to_numeric(
        x[field],
        errors="coerce",
    ).dropna().tolist()

    if not vals:
        return np.nan

    return float(statistics.median(vals))


def index_bytes(method):
    vals = []

    for rep in (1, 2, 3):

        root = RUN / method / f"rep{rep}"

        if method in {
            "HGV",
            "Raw6_BrayCurtis",
            "Raw6_Cosine",
            "Raw6_Euclidean",
            "PCA64_Euclidean",
            "PCA64_L2",
        }:
            p = (
                root
                / "build/output/"
                  "index_meta.json"
            )

            if p.exists():
                x = json.loads(p.read_text())
                vals.append(
                    int(x["payload_bytes"])
                )

        elif method in {
            "Mash",
            "Dashing2",
            "skani",
        }:
            p = (
                root
                / "build/"
                  "reference_index_bytes.txt"
            )

            if p.exists():
                vals.append(
                    int(p.read_text().strip())
                )

        elif method == "DNABERT-S":
            p = (
                root
                / "build/output/"
                  "index_meta.json"
            )

            if p.exists():
                x = json.loads(p.read_text())
                vals.append(
                    int(x["payload_bytes"])
                )

        elif method == "FastANI":
            # No persistent reference index in this E12B execution.
            pass

    if not vals:
        return np.nan

    return float(
        statistics.median(vals)
    )


rows = []

for method in METHODS:

    a = accmap[method]

    build_s = (
        np.nan
        if method == "FastANI"
        else median_phase(method, "build", "wall_s")
    )

    query_s = (
        np.nan
        if method == "FastANI"
        else median_phase(method, "query", "wall_s")
    )

    full_s = median_phase(
        method,
        "full",
        "wall_s",
    )

    host = median_phase(
        method,
        "full",
        "peak_host_pss_mib",
    )

    gpu = median_phase(
        method,
        "full",
        "peak_gpu_mib",
    )

    ib = index_bytes(method)

    host_status = "measured"

    if method == "DNABERT-S":
        host_status = (
            "PROVISIONAL: v2 runner retained full FASTA "
            "objects in process cache; rerun host-memory "
            "measurement with bounded cache before manuscript use"
        )

    rows.append(
        {
            "method": method,

            "Q50_R1_all_queries":
                float(
                    a["q50_recall_at_1_all_queries"]
                ),

            "query_hit_coverage":
                float(
                    a["query_hit_coverage"]
                ),

            "mean_pair_coverage":
                float(
                    a["mean_pair_coverage"]
                ),

            "top1_correct_given_hit":
                float(
                    a["top1_correct_given_hit"]
                )
                if pd.notna(
                    a["top1_correct_given_hit"]
                )
                else np.nan,

            "build_s_median":
                build_s,

            "warm_query_s_median":
                query_s,

            "complete_s_median":
                full_s,

            "peak_host_PSS_MiB_median":
                host,

            "host_memory_status":
                host_status,

            "peak_GPU_MiB_median":
                gpu,

            "reference_index_MiB":
                (
                    ib / (1024 ** 2)
                    if pd.notna(ib)
                    else np.nan
                ),
        }
    )

TABLE = pd.DataFrame(rows)

TABLE.to_csv(
    SUM / "TABLE7_SOURCE_DATA_PROVISIONAL.tsv",
    sep="\t",
    index=False,
    float_format="%.6f",
)

compact = TABLE[
    [
        "method",
        "Q50_R1_all_queries",
        "query_hit_coverage",
        "build_s_median",
        "warm_query_s_median",
        "complete_s_median",
        "peak_host_PSS_MiB_median",
        "peak_GPU_MiB_median",
        "reference_index_MiB",
        "host_memory_status",
    ]
].copy()

compact.to_csv(
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE_PROVISIONAL.tsv",
    sep="\t",
    index=False,
    float_format="%.4f",
)

# ------------------------------------------------------------------
# 2. Specific DNABERT-S audit
# ------------------------------------------------------------------

dna = TABLE[
    TABLE["method"] == "DNABERT-S"
].iloc[0]

dna_report = f"""E12B DNABERT-S CURRENT RESOURCE SUMMARY

build median s:
{dna['build_s_median']:.6f}

warm-query median s:
{dna['warm_query_s_median']:.6f}

complete median s:
{dna['complete_s_median']:.6f}

full-workflow host PSS median MiB:
{dna['peak_host_PSS_MiB_median']:.6f}

full-workflow GPU process memory median MiB:
{dna['peak_GPU_MiB_median']:.6f}

STATUS:
Runtime measurements are complete.
GPU measurement is complete.
Host-memory value is PROVISIONAL because e12b_dnaberts_v2.py
retains decoded FASTA records in fasta_cache across genomes.
Do not use the current ~10-GiB host PSS as a method-level
DNABERT-S memory claim in the manuscript.
"""

(SUM / "DNABERTS_RESOURCE_AUDIT.txt").write_text(
    dna_report
)

# ------------------------------------------------------------------
# 3. Overall audit
# ------------------------------------------------------------------

report = []

report.append(
    "HGV-V3 E12B synchronized resource benchmark provisional audit"
)

report.append("")
report.append("Methods: 11")
report.append("Repeats: 3")
report.append("References: 2374")
report.append("Queries: 50")
report.append("")

for method in METHODS:
    z = TABLE[
        TABLE["method"] == method
    ].iloc[0]

    report.append(
        f"{method}\t"
        f"build={z['build_s_median']}\t"
        f"query={z['warm_query_s_median']}\t"
        f"full={z['complete_s_median']}\t"
        f"host={z['peak_host_PSS_MiB_median']}\t"
        f"gpu={z['peak_GPU_MiB_median']}"
    )

report.append("")
report.append(
    "GATE: RESOURCE EXECUTION COMPLETE FOR ALL 11 METHODS."
)
report.append(
    "GATE: DNABERT-S HOST MEMORY NOT YET MANUSCRIPT-FROZEN."
)
report.append(
    "STATUS: PROVISIONAL PASS; one DNABERT-S bounded-cache memory rerun required."
)

audit_path = SUM / "E12B_FINAL_AUDIT_PROVISIONAL.txt"
audit_path.write_text(
    "\n".join(report) + "\n"
)

# ------------------------------------------------------------------
# 4. SHA freeze of provisional summary
# ------------------------------------------------------------------

freeze_files = [
    ACC_PATH,
    SUM / "E12B_RESOURCE_REPLICATES_LONG.tsv",
    SUM / "TABLE7_SOURCE_DATA_PROVISIONAL.tsv",
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE_PROVISIONAL.tsv",
    SUM / "DNABERTS_RESOURCE_AUDIT.txt",
    audit_path,
]

sha_lines = []

for p in freeze_files:
    h = hashlib.sha256(
        p.read_bytes()
    ).hexdigest()

    sha_lines.append(
        f"{h}  {p}"
    )

(FREEZE / "SHA256SUMS_PROVISIONAL.txt").write_text(
    "\n".join(sha_lines) + "\n"
)

(FREEZE / "E12B_STATUS_PROVISIONAL.txt").write_text(
    "E12B 11-METHOD EXECUTION: COMPLETE\n"
    "RESOURCE LOGS: COMPLETE\n"
    "DNABERT-S RUNTIME: COMPLETE\n"
    "DNABERT-S GPU MEMORY: COMPLETE\n"
    "DNABERT-S HOST MEMORY: PROVISIONAL\n"
    "FINAL MANUSCRIPT TABLE: NOT YET FROZEN\n"
)

print("=" * 120)
print("E12B PROVISIONAL TABLE 7")
print("=" * 120)

print(
    TABLE.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("RESOURCE EXECUTION: COMPLETE")
print(
    "DNABERT-S HOST MEMORY: PROVISIONAL "
    "(FASTA cache artifact)"
)
print("PROVISIONAL FINALIZE: PASS")
print()
print(
    "SOURCE =",
    SUM / "TABLE7_SOURCE_DATA_PROVISIONAL.tsv",
)
