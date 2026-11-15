#!/usr/bin/env python3
import csv
import json
import math
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

E5 = BASE / (
    "09_baselines/E5_unified_baseline_v1_20260917/"
    "07_unified_internal/"
    "e5_internal_11method_final_v1_20260918"
)

E11 = BASE / (
    "15_reference_screening/"
    "E11_reference_screening_v1_20260925_v6"
)

OUT = BASE / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W04"
)

FILES = {
    "E5_PRIMARY": E5 / "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv",
    "E5_PROTOCOL": E5 / "E5_INTERNAL_11METHOD_PROTOCOL_SUMMARY.tsv",
    "E5_SPLIT": E5 / "E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv",
    "E5_LONG": E5 / "E5_INTERNAL_11METHOD_LONG.tsv",
    "E11_SUMMARY": E11 / "E11_RETRIEVAL_SUMMARY.tsv",
    "E11_QUERY": E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv",
}

METHODS = (
    "HGV_Dashing2_tau0p98",
    "PCA64_Euclidean",
    "PCA64_L2"
)

PROTOCOLS = ("G2", "G3", "G4")


def require(condition, message):
    if not condition:
        raise RuntimeError("[STOP] " + message)


def read_tsv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        rows = list(reader)

    require(bool(columns), f"No header: {path}")
    require(
        not any(None in row for row in rows),
        f"Malformed TSV: {path}"
    )
    return columns, rows


def save_tsv(name, rows, fields):
    path = OUT / name

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def numeric(row, field):
    raw = str(row.get(field, "") or "").strip()
    require(raw != "", f"Missing {field}: {row}")
    value = float(raw)
    require(math.isfinite(value), f"Nonfinite {field}")
    return value


def key_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row.get(col, "") for col in keys)
        require(key not in result, f"Duplicate key: {key}")
        result[key] = row
    return result


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8)


print("=" * 76)
print("HGV WINDOWS W04 - FROZEN MCC AND E11 PROVENANCE")
print("=" * 76)

OUT.mkdir(parents=True, exist_ok=True)

for label, path in FILES.items():
    require(path.is_file(), f"Missing {label}: {path}")

# ------------------------------------------------------------
# 1. Read only official frozen E5 data.
# ------------------------------------------------------------

_, primary = read_tsv(FILES["E5_PRIMARY"])
_, protocol = read_tsv(FILES["E5_PROTOCOL"])
_, split = read_tsv(FILES["E5_SPLIT"])
_, long_data = read_tsv(FILES["E5_LONG"])

P = key_index(primary, ("method", "protocol"))
T = key_index(protocol, ("method", "protocol"))

L = key_index(
    [
        row for row in long_data
        if row["metric"].strip().lower() == "mcc"
    ],
    ("method", "protocol")
)

split_groups = defaultdict(list)

for row in split:
    split_groups[
        (row["method"], row["protocol"])
    ].append(row)

mcc_rows = []
comparison_rows = []

print()
print("[1] E5 FROZEN MCC CROSS-CHECK")

for group in PROTOCOLS:
    values = {}

    for method in METHODS:
        key = (method, group)

        require(key in P, f"Missing PRIMARY: {key}")
        require(key in T, f"Missing PROTOCOL: {key}")
        require(key in L, f"Missing LONG: {key}")
        require(key in split_groups, f"Missing SPLIT: {key}")

        rows = sorted(
            split_groups[key],
            key=lambda r: r["split_seed"]
        )

        seeds = [r["split_seed"] for r in rows]
        require(len(seeds) == 3, f"Expected 3 splits: {key}")
        require(len(set(seeds)) == 3, f"Duplicate split: {key}")

        split_mcc = [numeric(r, "MCC") for r in rows]

        mean_mcc = statistics.mean(split_mcc)
        sd_mcc = statistics.stdev(split_mcc)

        from_primary = numeric(P[key], "MCC_mean")
        from_protocol = numeric(T[key], "MCC_mean")
        from_long = numeric(L[key], "mean")

        require(
            close(mean_mcc, from_primary)
            and close(mean_mcc, from_protocol)
            and close(mean_mcc, from_long),
            f"MCC mismatch across frozen E5 sources: {key}"
        )

        require(
            close(sd_mcc, numeric(P[key], "MCC_sd_split_seed")),
            f"MCC SD mismatch: {key}"
        )

        values[method] = mean_mcc

        mcc_rows.append({
            "protocol": group,
            "method": method,
            "n_split_seeds": 3,
            "split_seeds": ",".join(seeds),
            "MCC_mean": f"{mean_mcc:.9f}",
            "MCC_sd_split_seed": f"{sd_mcc:.9f}",
            "split_MCC_values": ",".join(
                f"{v:.9f}" for v in split_mcc
            ),
            "metric_scope": P[key].get("metric_scope", ""),
            "coverage_type": P[key].get("coverage_type", ""),
            "source": "E5 official final frozen tables",
        })

    h = values[METHODS[0]]
    e = values[METHODS[1]]
    l = values[METHODS[2]]

    comparison_rows.append({
        "protocol": group,
        "HGV_MCC": f"{h:.9f}",
        "PCA64_Euclidean_MCC": f"{e:.9f}",
        "PCA64_L2_MCC": f"{l:.9f}",
        "HGV_minus_PCA_Euclidean": f"{h-e:.9f}",
        "HGV_minus_PCA_L2": f"{h-l:.9f}",
        "winner_HGV_vs_PCA_Euclidean":
            "HGV" if h > e else "PCA64_Euclidean",
    })

    print(
        f"{group}: HGV={h:.6f}, "
        f"PCA-Euclidean={e:.6f}, "
        f"PCA-L2={l:.6f}, "
        f"Delta(HGV-Euclidean)={h-e:+.6f}"
    )

save_tsv(
    "W04_E5_MCC_VERIFIED.tsv",
    mcc_rows,
    [
        "protocol", "method", "n_split_seeds",
        "split_seeds", "MCC_mean",
        "MCC_sd_split_seed", "split_MCC_values",
        "metric_scope", "coverage_type", "source"
    ]
)

save_tsv(
    "W04_E5_MCC_COMPARISON.tsv",
    comparison_rows,
    [
        "protocol", "HGV_MCC",
        "PCA64_Euclidean_MCC", "PCA64_L2_MCC",
        "HGV_minus_PCA_Euclidean",
        "HGV_minus_PCA_L2",
        "winner_HGV_vs_PCA_Euclidean"
    ]
)

print("[PASS] All 3 protocols cross-checked")

# ------------------------------------------------------------
# 2. Inspect successful E11 v6, not failed v7.
# ------------------------------------------------------------

print()
print("[2] E11 OFFICIAL RETRIEVAL AUDIT")

summary_cols, summary_rows = read_tsv(FILES["E11_SUMMARY"])
query_cols, query_rows = read_tsv(FILES["E11_QUERY"])

print("E11 summary rows:", len(summary_rows))
print("E11 summary columns:", " | ".join(summary_cols))
print("E11 query rows:", len(query_rows))
print("E11 query columns:", " | ".join(query_cols))

matching = []

for number, row in enumerate(summary_rows, 2):
    method_fields = [
        str(value or "")
        for key, value in row.items()
        if any(
            tag in key.lower()
            for tag in ("method", "model", "algorithm")
        )
    ]

    if not method_fields:
        method_fields = [str(v or "") for v in row.values()]

    method_text = " ".join(method_fields).lower()

    if "hgv" in method_text or "pca" in method_text:
        matching.append({
            "source_row": number,
            "source": "E11_RETRIEVAL_SUMMARY.tsv",
            "method_fields": " | ".join(method_fields),
            "full_row_json": json.dumps(
                row, ensure_ascii=False
            )
        })

save_tsv(
    "W04_E11_HGV_PCA_SOURCE_ROWS.tsv",
    matching,
    [
        "source_row", "source",
        "method_fields", "full_row_json"
    ]
)

print()
print("E11 HGV/PCA summary rows:", len(matching))

for row in matching[:30]:
    print(
        "ROW", row["source_row"],
        row["full_row_json"][:950]
    )

# ------------------------------------------------------------
# 3. Record all source hashes and exact E11 schemas.
# ------------------------------------------------------------

hash_rows = []

for label, path in FILES.items():
    hash_rows.append({
        "artifact": label,
        "sha256": sha256(path),
        "relative_path": str(path.relative_to(BASE))
    })

save_tsv(
    "W04_SOURCE_SHA256.tsv",
    hash_rows,
    ["artifact", "sha256", "relative_path"]
)

schema_report = [
    "E11 successful source: "
    + str(FILES["E11_SUMMARY"].relative_to(BASE)),
    "E11 summary rows: " + str(len(summary_rows)),
    "E11 query rows: " + str(len(query_rows)),
    "E11 summary columns:",
    " | ".join(summary_cols),
    "E11 query columns:",
    " | ".join(query_cols),
    "",
    "IMPORTANT:",
    "W01 external B2 metrics and E11 retrieval "
    "metrics must retain their original protocol definitions.",
    "The E11 FAILED_v7 directory was not used.",
]

(OUT / "W04_E11_PROTOCOL_INSPECTION.txt").write_text(
    "\n".join(schema_report) + "\n",
    encoding="utf-8"
)

status = [
    "W04 E5 MCC cross-check: PASS",
    "E5 G2/G3/G4 official values: VERIFIED",
    "E11 successful v6 source: LOCATED",
    "E11 summary and query schemas: EXTRACTED",
    "E11 protocol interpretation: PENDING",
    "Real MAG evaluation: NOT PERFORMED",
    "New Q50 runtime measurement: NOT PERFORMED",
    "Original data and models: UNCHANGED",
]

(OUT / "W04_STATUS.txt").write_text(
    "\n".join(status) + "\n",
    encoding="utf-8"
)

print()
print("=" * 76)
print("\n".join(status))
print("Output:", OUT)
PY
