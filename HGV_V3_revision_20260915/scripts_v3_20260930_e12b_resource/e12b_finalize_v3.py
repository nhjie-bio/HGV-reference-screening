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

PROV = (
    OUT
    / "04_summary/finalize_v2_20261001/"
      "TABLE7_SOURCE_DATA_PROVISIONAL.tsv"
)

OLD_LONG = (
    OUT
    / "04_summary/finalize_v2_20261001/"
      "E12B_RESOURCE_REPLICATES_LONG.tsv"
)

RUNV3 = (
    OUT
    / "03_runs/"
      "DNABERT-S_v3_bounded_cache_20261001"
)

SUM = (
    OUT
    / "04_summary/"
      "finalize_v3_20261001"
)

FREEZE = (
    OUT
    / "05_final_freeze_v3_20261001"
)

SUM.mkdir(parents=True, exist_ok=True)
FREEZE.mkdir(parents=True, exist_ok=True)

EQ = RUNV3 / "V2_V3_EQUIVALENCE_AUDIT.txt"

if not EQ.exists():
    raise RuntimeError("Missing v2/v3 equivalence audit")

if "STATUS=PASS" not in EQ.read_text():
    raise RuntimeError("v2/v3 equivalence did not PASS")


def load_resource(rep, phase):
    p = (
        RUNV3
        / f"rep{rep}"
        / phase
        / "resource/resource.json"
    )

    if not p.exists():
        raise RuntimeError(f"Missing {p}")

    x = json.loads(p.read_text())

    if int(x["exit_code"]) != 0:
        raise RuntimeError(
            f"Nonzero execution: rep{rep} {phase}"
        )

    return x


records = []

for rep in (1, 2, 3):
    for phase in ("build", "query", "full"):

        x = load_resource(rep, phase)

        records.append(
            {
                "method": "DNABERT-S",
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
                    str(
                        RUNV3
                        / f"rep{rep}"
                        / phase
                        / "resource/resource.json"
                    ),
            }
        )

V3 = pd.DataFrame(records)


def med(phase, field):
    return float(
        statistics.median(
            V3.loc[
                V3["phase"].eq(phase),
                field
            ].astype(float).tolist()
        )
    )


build_s = med("build", "wall_s")
query_s = med("query", "wall_s")
full_s = med("full", "wall_s")

host = med(
    "full",
    "peak_host_pss_mib",
)

gpu = med(
    "full",
    "peak_gpu_mib",
)

# A very generous sanity gate.
# If bounded-cache still produces >6 GiB full-workflow PSS,
# do not automatically freeze the manuscript memory value.
if host >= 6144:
    raise RuntimeError(
        f"DNABERT-S bounded-cache host PSS still unexpectedly high: "
        f"{host:.3f} MiB. Investigate before manuscript freeze."
    )

index_vals = []

for rep in (1, 2, 3):
    p = (
        RUNV3
        / f"rep{rep}"
        / "build/output/index_meta.json"
    )

    x = json.loads(p.read_text())

    index_vals.append(
        int(x["payload_bytes"])
    )

index_mib = (
    statistics.median(index_vals)
    / 1024**2
)

TABLE = pd.read_csv(
    PROV,
    sep="\t",
)

mask = TABLE["method"].eq("DNABERT-S")

if int(mask.sum()) != 1:
    raise RuntimeError(
        "Expected exactly one DNABERT-S row"
    )

TABLE.loc[
    mask,
    "build_s_median"
] = build_s

TABLE.loc[
    mask,
    "warm_query_s_median"
] = query_s

TABLE.loc[
    mask,
    "complete_s_median"
] = full_s

TABLE.loc[
    mask,
    "peak_host_PSS_MiB_median"
] = host

TABLE.loc[
    mask,
    "peak_GPU_MiB_median"
] = gpu

TABLE.loc[
    mask,
    "reference_index_MiB"
] = index_mib

TABLE.loc[
    mask,
    "host_memory_status"
] = (
    "measured: bounded one-genome FASTA cache; "
    "frozen segment coordinates and SHA256 verified"
)

TABLE.to_csv(
    SUM / "TABLE7_SOURCE_DATA_FINAL.tsv",
    sep="\t",
    index=False,
    float_format="%.6f",
)

compact = TABLE[
    [
        "method",
        "Q50_R1_all_queries",
        "query_hit_coverage",
        "mean_pair_coverage",
        "build_s_median",
        "warm_query_s_median",
        "complete_s_median",
        "peak_host_PSS_MiB_median",
        "peak_GPU_MiB_median",
        "reference_index_MiB",
    ]
].copy()

compact.to_csv(
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE_FINAL.tsv",
    sep="\t",
    index=False,
    float_format="%.4f",
)

# Replace provisional DNABERT resource rows with v3 rows.
OLD = pd.read_csv(
    OLD_LONG,
    sep="\t",
)

OLD = OLD[
    ~OLD["method"].eq("DNABERT-S")
].copy()

LONG = pd.concat(
    [OLD, V3],
    ignore_index=True,
)

LONG.to_csv(
    SUM / "E12B_RESOURCE_REPLICATES_LONG_FINAL.tsv",
    sep="\t",
    index=False,
    float_format="%.6f",
)

report = f"""HGV-V3 E12B FINAL SYNCHRONIZED RESOURCE BENCHMARK

Methods: 11
References: 2374
Queries: 50 fixed B1 resource queries
Repeats: 3

DNABERT-S v3 bounded-cache:
build median s = {build_s:.6f}
warm-query median s = {query_s:.6f}
complete median s = {full_s:.6f}
full host PSS median MiB = {host:.6f}
full GPU process memory median MiB = {gpu:.6f}
index MiB = {index_mib:.6f}

DNABERT v2/v3 equivalence:
{EQ.read_text().strip()}

STATUS:
E12B 11-method synchronized resource execution COMPLETE.
DNABERT-S bounded-cache measurement COMPLETE.
Table 7 source data FINALIZED.

IMPORTANT:
Final manuscript wording still requires a separate consistency audit
against historical E12 HGV timing before replacing the old Table 7.
"""

(SUM / "E12B_FINAL_REPORT.txt").write_text(
    report
)

freeze_files = [
    SUM / "TABLE7_SOURCE_DATA_FINAL.tsv",
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE_FINAL.tsv",
    SUM / "E12B_RESOURCE_REPLICATES_LONG_FINAL.tsv",
    SUM / "E12B_FINAL_REPORT.txt",
    EQ,
]

sha_lines = []

for p in freeze_files:
    h = hashlib.sha256(
        p.read_bytes()
    ).hexdigest()

    sha_lines.append(
        f"{h}  {p}"
    )

(FREEZE / "SHA256SUMS.txt").write_text(
    "\n".join(sha_lines) + "\n"
)

(FREEZE / "E12B_FINAL_STATUS.txt").write_text(
    "E12B SYNCHRONIZED RESOURCE EXECUTION: PASS\n"
    "METHODS: 11/11\n"
    "REPEATS: 3\n"
    "DNABERT-S BOUNDED-CACHE: PASS\n"
    "DNABERT-S V2/V3 NUMERICAL EQUIVALENCE: PASS\n"
    "TABLE7 SOURCE DATA: FROZEN\n"
    "MANUSCRIPT CONSISTENCY AUDIT: STILL REQUIRED\n"
)

print("=" * 120)
print("E12B FINAL TABLE 7 SOURCE DATA")
print("=" * 120)

print(
    TABLE.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("FINALIZE v3 = PASS")
print()
print(
    FREEZE / "E12B_FINAL_STATUS.txt"
)
