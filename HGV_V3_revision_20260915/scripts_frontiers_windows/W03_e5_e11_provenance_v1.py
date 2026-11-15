#!/usr/bin/env python3

import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

OUT = BASE / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W03"
)

E5ROOT = BASE / (
    "09_baselines/"
    "E5_unified_baseline_v1_20260917/"
    "07_unified_internal/"
    "e5_internal_11method_final_v1_20260918"
)

E5FILES = {
    "primary": E5ROOT / "E5_INTERNAL_11METHOD_PRIMARY_TABLE.tsv",
    "protocol": E5ROOT / "E5_INTERNAL_11METHOD_PROTOCOL_SUMMARY.tsv",
    "split": E5ROOT / "E5_INTERNAL_11METHOD_SPLIT_LEVEL.tsv",
    "long": E5ROOT / "E5_INTERNAL_11METHOD_LONG.tsv",
}


def read_tsv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rd = csv.DictReader(f, delimiter="\t")
        columns = rd.fieldnames or []
        rows = list(rd)

    if not columns or any(None in r for r in rows):
        raise RuntimeError("Bad TSV format: " + str(path))

    return columns, rows


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


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(block)

    return h.hexdigest()


def pick(row, cols, terms):
    names = [
        c for c in cols
        if any(t in c.lower() for t in terms)
    ]

    return " | ".join(
        str(row.get(c, "") or "").strip()
        for c in names
        if str(row.get(c, "") or "").strip()
    )


def is_hgv_pca(text):
    t = (
        text.lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    return "hgv" in t or "pca64" in t


print("=" * 78)
print("HGV WINDOWS W03 - E5 MCC AND E11 RETRIEVAL PROVENANCE")
print("=" * 78)

OUT.mkdir(parents=True, exist_ok=True)

if not E5FILES["primary"].is_file():
    raise SystemExit(
        "[STOP] Missing official E5 primary table: "
        + str(E5FILES["primary"])
    )

schemas = []
matches = []
report = []

# ============================================================
# 1. Audit E5 frozen result tables
# ============================================================

for label, path in E5FILES.items():

    if not path.is_file():
        print("[MISSING]", label, path)

        schemas.append({
            "table": label,
            "status": "MISSING",
            "rows": "",
            "sha256": "",
            "columns": ""
        })
        continue

    cols, rows = read_tsv(path)
    digest = sha256(path)

    schemas.append({
        "table": label,
        "status": "FOUND",
        "rows": len(rows),
        "sha256": digest,
        "columns": " | ".join(cols)
    })

    print()
    print(
        f"[{label.upper()}] "
        f"rows={len(rows)} columns={len(cols)}"
    )
    print("SCHEMA:", " | ".join(cols[:50]))

    method_cols = [
        c for c in cols
        if (
            "method" in c.lower()
            or c.lower() in {
                "model", "algorithm", "baseline"
            }
        )
    ]

    protocol_cols = [
        c for c in cols
        if any(
            t in c.lower()
            for t in (
                "protocol", "holdout", "group", "split"
            )
        )
    ]

    metric_cols = [
        c for c in cols
        if c.lower() in {
            "metric", "metric_name", "measure"
        }
    ]

    mcc_cols = [
        c for c in cols
        if "mcc" in c.lower()
    ]

    label_matches = []

    for i, row in enumerate(rows, 2):

        method_text = pick(
            row,
            method_cols,
            ("method", "model", "algorithm", "baseline")
        )

        if not is_hgv_pca(method_text):
            continue

        metric_name = " | ".join(
            str(row.get(c, "") or "").strip()
            for c in metric_cols
        ).lower()

        if (
            metric_cols
            and metric_name
            and "mcc" not in metric_name
            and not mcc_cols
        ):
            continue

        mcc_values = {
            c: row.get(c, "")
            for c in mcc_cols
        }

        if "mcc" in metric_name:

            for c in cols:
                if c.lower() in (
                    "value",
                    "mean",
                    "std",
                    "sd",
                    "score",
                    "metric_value",
                    "mean_value"
                ):
                    mcc_values[c] = row.get(c, "")

            mcc_values["metric"] = metric_name

        match = {
            "table": label,
            "source_row": i,
            "method": method_text,
            "protocol": pick(
                row,
                protocol_cols,
                ("protocol", "holdout", "group", "split")
            ),
            "mcc_fields_json": json.dumps(
                mcc_values, ensure_ascii=False
            ),
            "original_row_json": json.dumps(
                row, ensure_ascii=False
            )
        }

        matches.append(match)
        label_matches.append(match)

    print(
        "HGV/PCA candidate MCC rows:",
        len(label_matches)
    )

    for m in label_matches[:16]:
        print(
            " ",
            m["method"][:36],
            "|",
            m["protocol"][:42],
            "|",
            m["mcc_fields_json"][:260]
        )


write_tsv(
    OUT / "W03_E5_SOURCE_SCHEMAS.tsv",
    schemas,
    [
        "table", "status", "rows",
        "sha256", "columns"
    ]
)

write_tsv(
    OUT / "W03_E5_HGV_PCA_RAW_MCC.tsv",
    matches,
    [
        "table",
        "source_row",
        "method",
        "protocol",
        "mcc_fields_json",
        "original_row_json"
    ]
)

# ============================================================
# 2. Locate E11 and external retrieval metric sources
# ============================================================

print()
print("[E11 / EXTERNAL RETRIEVAL SOURCE DISCOVERY]")

candidates = []

roots = [
    BASE / "15_reference_screening",
    BASE / "09_external_validation"
]

for root in roots:

    if not root.is_dir():
        continue

    for p in root.rglob("*.tsv"):

        name = p.name.lower()

        if not any(
            t in name
            for t in (
                "retriev",
                "rank",
                "screen",
                "metric",
                "summary",
                "external",
                "e11"
            )
        ):
            continue

        try:
            size = p.stat().st_size

            if size > 12 * 1024 * 1024:
                continue

            with p.open(
                "r",
                encoding="utf-8-sig",
                errors="replace"
            ) as f:
                header = f.readline().strip()[:650]

        except (OSError, UnicodeError):
            continue

        score = (
            5 * ("e11" in name)
            + 4 * ("retriev" in name)
            + 3 * ("rank" in name)
            + 2 * ("summary" in name)
            + int("family" in header.lower())
            + int("metric" in header.lower())
        )

        candidates.append({
            "score": score,
            "file": str(p.relative_to(BASE)),
            "bytes": size,
            "header_excerpt": header
        })

candidates.sort(
    key=lambda x: (-x["score"], x["file"])
)

write_tsv(
    OUT / "W03_E11_RETRIEVAL_CANDIDATES.tsv",
    candidates[:100],
    [
        "score",
        "file",
        "bytes",
        "header_excerpt"
    ]
)

print(
    "Candidate retrieval result files:",
    len(candidates)
)

for r in candidates[:18]:
    print(
        " ",
        r["score"],
        "|",
        r["file"]
    )

# ============================================================
# 3. Final status
# ============================================================

report.extend([
    "W03 E5/E11 source provenance inspection: COMPLETED",
    "Primary E5 table present: YES",
    "E5 HGV/PCA candidate rows: " + str(len(matches)),
    "E11/external retrieval candidate files: "
        + str(len(candidates)),
    "Scientific interpretation: PENDING per-metric protocol reconciliation",
    "W01 B2 family Rank-1 must not be substituted for "
        "the manuscript E11 family R@1 without checking retrieval protocol.",
    "No old model, TSV, or paper file changed."
])

(OUT / "W03_STATUS.txt").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print()
print("\n".join(report))
print("Output directory:", OUT)
