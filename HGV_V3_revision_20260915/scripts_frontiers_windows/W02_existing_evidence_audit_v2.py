#!/usr/bin/env python3

import csv
import os
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

W01 = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W01_v2"
)

OUT = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W02_v2"
)

E8 = ROOT / (
    "12_fragmentation/"
    "E8_fragmentation_v1_20260925_v6/"
    "E8_FRAGMENTATION_SUMMARY.tsv"
)

E12 = ROOT / (
    "16_scalability/"
    "E12B_synchronized_resource_v1_20260930"
)

TABLE7 = E12 / (
    "04_summary/finalize_v3_20261001/"
    "TABLE7_SOURCE_DATA_FINAL.tsv"
)

STATUS = E12 / (
    "05_final_freeze_v3_20261001/"
    "E12B_FINAL_STATUS.txt"
)

PAIRED = W01 / "W01_E15_PAIRED_HGV_PCA.tsv"
Q50 = W01 / "W01_E14_Q50_FROZEN_ACCURACY.tsv"
SENS = W01 / "W01_E14_500_STRATIFIED_RESAMPLINGS.tsv"

OUT.mkdir(parents=True, exist_ok=True)

report = []


def log(message=""):
    message = str(message)
    print(message)
    report.append(message)


def read_tsv(path):
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        columns = reader.fieldnames or []

    if any(None in row for row in rows):
        raise RuntimeError(
            "Malformed TSV field count: " + str(path)
        )

    return columns, rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def write_tsv(path, rows, columns):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)


def value(row, key):
    return str(row.get(key, "") or "")


log("=" * 78)
log("HGV V3 WINDOWS W02 - EXISTING EVIDENCE AUDIT")
log("=" * 78)

# ==========================================================
# 1. W01 HGV VS PCA
# ==========================================================

log("\n[1] HGV VS PCA-64 PAIRED RESULTS")

cols, paired = read_tsv(PAIRED)

log("Paired comparison rows: " + str(len(paired)))

for row in paired:

    if row["pca_baseline"] != "PCA64_L2":
        continue

    log(
        " | ".join([
            row["cohort"],
            "seed=" + row["hgv_seed"],
            "n=" + row["n_queries"],
            "HGV_R1=" + row["HGV_R1_pct"] + "%",
            "PCA_R1=" + row["PCA_R1_pct"] + "%",
            "Delta=" + row["HGV_minus_PCA_micro_pp"] + "pp",
            "HGV_only=" + row["HGV_only_correct"],
            "PCA_only=" + row["PCA_only_correct"],
            "McNemar_p=" + row["McNemar_exact_p"],
        ])
    )

log(
    "Note: B1 uses species Rank-1; "
    "B2_seen_genus uses genus Rank-1; "
    "B2_unseen_genus uses family Rank-1."
)

# ==========================================================
# 2. NEW STRATIFIED Q50 ACCURACY
# ==========================================================

log("\n[2] NEW STRATIFIED Q50 - FROZEN PREDICTIONS")

cols, q50 = read_tsv(Q50)

log("Prediction rows: " + str(len(q50)))

for row in q50:
    log(
        " | ".join([
            row["method"],
            "seed=" + row["model_seed"],
            "correct=" + row["top1_correct"] + "/50",
            "R1=" + row["top1_accuracy_pct"] + "%",
        ])
    )

log(
    "IMPORTANT: These values come from frozen predictions, "
    "not a new runtime benchmark."
)

# ==========================================================
# 3. Q50 SAMPLING SENSITIVITY
# ==========================================================

log("\n[3] 500 STRATIFIED RESAMPLINGS")

cols, sensitivity = read_tsv(SENS)

for row in sensitivity:

    if row["method"] not in (
        "HGV_Dashing2_tau0p98",
        "PCA64_L2",
        "PCA64_Euclidean"
    ):
        continue

    log(
        " | ".join([
            row["method"],
            "seed=" + row["model_seed"],
            "mean=" + row["mean_accuracy_pct"] + "%",
            "P2.5=" + row["p025_accuracy_pct"] + "%",
            "P97.5=" + row["p975_accuracy_pct"] + "%",
        ])
    )

log(
    "Resampling quantiles describe subset variation; "
    "they are not 500 independent experiments."
)

# ==========================================================
# 4. E12B ORIGINAL RESOURCE BENCHMARK
# ==========================================================

log("\n[4] ORIGINAL E12B RESOURCE RESULTS")

cols, table7 = read_tsv(TABLE7)

log("Original Table 7 rows: " + str(len(table7)))
log("Unique methods: " + str(
    len(set(r["method"] for r in table7))
))

required = {
    "method",
    "Q50_R1_all_queries",
    "build_s_median",
    "warm_query_s_median",
    "complete_s_median",
    "reference_index_MiB"
}

if not required.issubset(cols):
    raise RuntimeError(
        "Table 7 missing columns: "
        + str(sorted(required - set(cols)))
    )

missing_metric_rows = []

for row in table7:

    for field in (
        "Q50_R1_all_queries",
        "build_s_median",
        "warm_query_s_median",
        "complete_s_median",
        "reference_index_MiB"
    ):
        raw = str(row.get(field) or "").strip()

        if raw.lower() in {
            "", "na", "n/a", "nan", "null", "none", "-"
        }:
            missing_metric_rows.append({
                "method": row["method"],
                "metric": field,
                "original_value": raw,
                "status": "NOT_REPORTED"
            })
            continue

        try:
            number = float(raw)
        except ValueError:
            raise RuntimeError(
                "Invalid E12B numeric value: "
                + row["method"] + " / " + field
                + " = " + repr(raw)
            )

        if not math.isfinite(number):
            raise RuntimeError(
                "Nonfinite E12B value: "
                + row["method"] + " / " + field
            )

    log(
        " | ".join([
            row["method"],
            "R1=" + row["Q50_R1_all_queries"],
            "build_s=" + row["build_s_median"],
            "query_s=" + row["warm_query_s_median"],
            "complete_s=" + row["complete_s_median"],
            "index_MiB=" + row["reference_index_MiB"],
        ])
    )

write_tsv(
    OUT / "W02_E12B_MISSING_METRICS.tsv",
    missing_metric_rows,
    ["method", "metric", "original_value", "status"]
)

log(
    "E12B not-reported metric cells: "
    + str(len(missing_metric_rows))
)

if STATUS.is_file():
    log("\nOriginal E12B final status:")
    log(STATUS.read_text(encoding="utf-8").strip())

log(
    "IMPORTANT: Original E12B used the old Q50. "
    "Do not attribute these runtimes to the new Q50."
)

# ==========================================================
# 5. E8 FRAGMENTATION
# ==========================================================

log("\n[5] ORIGINAL E8 FRAGMENTATION")

e8_cols, e8_rows = read_tsv(E8)

log("E8 columns: " + ", ".join(e8_cols))
log("E8 rows: " + str(len(e8_rows)))

for row in e8_rows[:20]:
    log(
        " | ".join(
            f"{key}={value(row, key)}"
            for key in e8_cols
        )
    )

log(
    "IMPORTANT: E8 evaluates simulated fragmentation, "
    "not independently sourced real MAGs."
)

# ==========================================================
# 6. DISCOVER E5 MCC SOURCE FILES
# ==========================================================

log("\n[6] FIND G2/G3/G4 MCC SOURCE FILES")

search_roots = [
    ROOT / "09_baselines",
    ROOT / "08_main_generalization",
    ROOT / "17_statistics",
    ROOT / (
        "19_tables/"
        "final_manuscript_source_v1_20260926_182919/E5"
    ),
]

excluded_dirs = {
    "__pycache__",
    "per_genome",
    "scratch",
    "cache",
    "tmp",
    "temporary",
    "invalid_v1_provenance",
}

candidate_rows = []
visited = set()

for search_root in search_roots:

    if not search_root.is_dir():
        continue

    for directory, dirs, files in os.walk(
        search_root, followlinks=False
    ):

        dirs[:] = [
            d for d in dirs
            if d.lower() not in excluded_dirs
        ]

        for name in files:

            p = Path(directory) / name

            try:
                key = str(p.resolve()).lower()
            except OSError:
                continue

            if key in visited:
                continue

            visited.add(key)

            if p.suffix.lower() not in (
                ".tsv", ".csv"
            ):
                continue

            name_low = name.lower()

            relevant_name = any(
                x in name_low
                for x in (
                    "mcc", "metric", "summary",
                    "primary", "final", "freeze",
                    "merge", "comparison",
                    "protocol", "11method"
                )
            )

            if not relevant_name:
                continue

            try:
                size = p.stat().st_size
            except OSError:
                continue

            if size > 30 * 1024 * 1024:
                continue

            try:
                with p.open(
                    "r",
                    encoding="utf-8-sig",
                    errors="replace",
                    newline=""
                ) as f:
                    delimiter = (
                        "\t" if p.suffix.lower() == ".tsv"
                        else ","
                    )
                    reader = csv.reader(
                        f, delimiter=delimiter
                    )
                    header = next(reader, [])

                header_low = [
                    c.strip().lower() for c in header
                ]

            except Exception:
                continue

            has_mcc = any(
                "mcc" in c for c in header_low
            )

            has_metric = any(
                c in ("metric", "metric_name")
                for c in header_low
            )

            has_method = any(
                "method" in c for c in header_low
            )

            has_split = any(
                ("split" in c or "protocol" in c)
                for c in header_low
            )

            score = (
                6 * int(has_mcc)
                + 3 * int(has_metric)
                + 3 * int(has_method)
                + 2 * int(has_split)
                + 2 * int("final" in name_low)
                + 2 * int("freeze" in name_low)
                + 2 * int("11method" in name_low)
                + int("primary" in name_low)
            )

            if score < 3:
                continue

            candidate_rows.append({
                "score": score,
                "file": str(p.relative_to(ROOT)),
                "size_bytes": size,
                "has_mcc_column": has_mcc,
                "has_metric_column": has_metric,
                "has_method_column": has_method,
                "has_split_or_protocol_column": has_split,
                "columns": " | ".join(header[:30]),
            })

candidate_rows.sort(
    key=lambda r: (-r["score"], r["file"])
)

CANDIDATES = OUT / "W02_E5_MCC_CANDIDATES.tsv"

write_tsv(
    CANDIDATES,
    candidate_rows,
    [
        "score", "file", "size_bytes",
        "has_mcc_column", "has_metric_column",
        "has_method_column",
        "has_split_or_protocol_column",
        "columns"
    ]
)

log("Candidate files found: " + str(len(candidate_rows)))

for row in candidate_rows[:20]:

    log(
        f"score={row['score']} | "
        f"{row['file']}"
    )

log(
    "E5 source selection is not automatic: "
    "final frozen provenance must be confirmed."
)

# ==========================================================
# 7. PROVENANCE CHECK
# ==========================================================

log("\n[7] SOURCE CHECKSUMS")

for label, path in [
    ("W01_paired", PAIRED),
    ("W01_Q50", Q50),
    ("E8", E8),
    ("E12B_Table7", TABLE7),
]:

    log(label + " SHA256=" + sha256(path))

# ==========================================================
# 8. STATUS
# ==========================================================

log("\n" + "=" * 78)
log("W02 EXISTING EVIDENCE AUDIT: COMPLETE")
log("=" * 78)

log("HGV/PCA external paired data: AVAILABLE")
log("Species-stratified Q50 accuracy: AVAILABLE")
log("E8 fragmentation evidence: AVAILABLE")
log("Original E12B resource measurements: AVAILABLE")
log("G2/G3/G4 frozen MCC source: REQUIRES IDENTIFICATION")
log("New stratified Q50 runtime: NOT MEASURED")
log("Real MAG evaluation: NOT PERFORMED")
log("Public repository: NOT YET VERIFIED")

log("\nNo model training performed.")
log("No original experimental results modified.")

REPORT = OUT / "W02_EXISTING_EVIDENCE_REPORT.txt"

REPORT.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

log("\nReport saved: " + str(REPORT))
log("E5 candidates saved: " + str(CANDIDATES))
