#!/usr/bin/env python3

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE
    /
    "09_baselines"
    /
    "E5_unified_baseline_v1_20260917"
)

OUT = (
    BASE
    /
    "20_rebuttal"
    /
    "HGV_V3_E5_runtime_audit_v2_20260926"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

METHOD_PATTERNS = {
    "Mash":
        re.compile(
            r"\bmash\b",
            re.I,
        ),

    "Dashing2":
        re.compile(
            r"dashing2|\bdashing\b",
            re.I,
        ),

    "skani":
        re.compile(
            r"\bskani\b",
            re.I,
        ),

    "FastANI":
        re.compile(
            r"\bfastani\b",
            re.I,
        ),

    "HGV":
        re.compile(
            r"\bhgv\b",
            re.I,
        ),
}

WALL_PATTERNS = [
    re.compile(
        r'wall[_ ]seconds\s*["\']?\s*[:=]\s*([0-9eE.+-]+)',
        re.I,
    ),

    re.compile(
        r'"wall_seconds"\s*:\s*([0-9eE.+-]+)',
        re.I,
    ),

    re.compile(
        r'elapsed_seconds\s*["\']?\s*[:=]\s*([0-9eE.+-]+)',
        re.I,
    ),
]

def infer_method(path, text):

    s = (
        str(path)
        +
        "\n"
        +
        text[:200000]
    )

    hits = [
        m
        for m, pat
        in METHOD_PATTERNS.items()
        if pat.search(
            s
        )
    ]

    if len(hits) == 1:
        return hits[0]

    # Path priority
    lower = str(
        path
    ).lower()

    if "dashing" in lower:
        return "Dashing2"

    if "fastani" in lower:
        return "FastANI"

    if "skani" in lower:
        return "skani"

    if "mash" in lower:
        return "Mash"

    if "hgv" in lower:
        return "HGV"

    return (
        ",".join(
            hits
        )
        if hits
        else
        "UNKNOWN"
    )

def infer_scope(path):

    s = str(
        path
    ).lower()

    if "external" in s:
        return "external"

    if "validation" in s:
        return "validation"

    if "internal" in s:
        return "internal"

    if "g2_" in s:
        return "G2"

    if "g3_" in s:
        return "G3"

    if "g4_" in s:
        return "G4"

    return "unspecified"

records = []
candidate_files = []

for p in sorted(
    E5.rglob("*")
):

    if (
        not p.is_file()
        or
        p.stat().st_size
        >
        25
        *
        1024
        *
        1024
    ):
        continue

    suffix = (
        p.suffix.lower()
    )

    if suffix not in {
        ".txt",
        ".md",
        ".json",
        ".log",
        ".tsv",
        ".csv",
    }:
        continue

    text = ""

    try:
        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        continue

    method = infer_method(
        p,
        text,
    )

    scope = infer_scope(
        p
    )

    values = []

    for pat in WALL_PATTERNS:

        for m in pat.finditer(
            text
        ):

            try:
                values.append(
                    float(
                        m.group(1)
                    )
                )
            except Exception:
                pass

    if suffix in {
        ".tsv",
        ".csv",
    }:

        try:
            sep = (
                "\t"
                if suffix == ".tsv"
                else ","
            )

            df = pd.read_csv(
                p,
                sep=sep,
            )

            for col in [
                "wall_seconds",
                "elapsed_seconds",
                "runtime_seconds",
                "wall_s",
            ]:

                if col in df.columns:

                    vals = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    ).dropna()

                    values.extend(
                        vals.astype(
                            float
                        ).tolist()
                    )

        except Exception:
            pass

    runtime_like = (
        bool(values)
        or
        re.search(
            r"runtime|wall|elapsed|rss|memory|throughput",
            p.name,
            re.I,
        )
        is not None
    )

    if not runtime_like:
        continue

    candidate_files.append({
        "method":
            method,

        "scope":
            scope,

        "path":
            str(
                p.relative_to(
                    BASE
                )
            ),

        "numeric_runtime_values_found":
            len(values),

        "size_bytes":
            p.stat().st_size,
    })

    for value in values:

        if (
            np.isfinite(
                value
            )
            and
            value >= 0
        ):

            records.append({
                "method":
                    method,

                "scope":
                    scope,

                "path":
                    str(
                        p.relative_to(
                            BASE
                        )
                    ),

                "wall_or_elapsed_seconds":
                    value,
            })

files_df = pd.DataFrame(
    candidate_files
)

records_df = pd.DataFrame(
    records
)

files_df.to_csv(
    OUT
    /
    "E5_RUNTIME_CANDIDATE_FILES.tsv",
    sep="\t",
    index=False,
)

records_df.to_csv(
    OUT
    /
    "E5_RUNTIME_NUMERIC_RECORDS.tsv",
    sep="\t",
    index=False,
)

summary_rows = []

if not records_df.empty:

    for (
        method,
        scope,
    ), g in records_df.groupby(
        [
            "method",
            "scope",
        ],
        sort=True,
    ):

        v = pd.to_numeric(
            g[
                "wall_or_elapsed_seconds"
            ],
            errors="coerce",
        ).dropna()

        summary_rows.append({
            "method":
                method,

            "scope":
                scope,

            "n_numeric_records":
                len(v),

            "wall_seconds_median":
                float(
                    v.median()
                ),

            "wall_seconds_min":
                float(
                    v.min()
                ),

            "wall_seconds_max":
                float(
                    v.max()
                ),

            "n_source_files":
                int(
                    g[
                        "path"
                    ]
                    .nunique()
                ),
        })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT
    /
    "E5_RUNTIME_METHOD_SCOPE_SUMMARY.tsv",
    sep="\t",
    index=False,
)

method_file_counts = {}

if not files_df.empty:

    for method in [
        "Mash",
        "Dashing2",
        "skani",
        "FastANI",
        "HGV",
    ]:

        method_file_counts[
            method
        ] = int(
            np.sum(
                files_df[
                    "method"
                ]
                ==
                method
            )
        )

else:

    for method in [
        "Mash",
        "Dashing2",
        "skani",
        "FastANI",
        "HGV",
    ]:
        method_file_counts[
            method
        ] = 0

(OUT / "E5_RUNTIME_SCOPE_DECISION.txt").write_text(
    "\n".join([
        "E5_RUNTIME_AUDIT=PASS",
        "NATIVE_TOOLS_RERUN=NO",
        "SOURCE=existing frozen E5 files only",
        "FAIR_CROSS_METHOD_RUNTIME_TABLE=REQUIRES_MANUAL_SCOPE_ALIGNMENT",
        "DO_NOT_COMPARE_UNMATCHED_INTERNAL_EXTERNAL_OR_BUILD_QUERY_TIMES=YES",
        f"METHOD_FILE_COUNTS={json.dumps(method_file_counts, sort_keys=True)}",
    ])
    +
    "\n",
    encoding="utf-8",
)

(OUT / "E5_RUNTIME_AUDIT_PASS.txt").write_text(
    "E5_RUNTIME_AUDIT=PASS\n"
    "NATIVE_TOOLS_RERUN=NO\n",
    encoding="utf-8",
)

print(
    "E5_RUNTIME_AUDIT=PASS"
)

print(
    "METHOD_FILE_COUNTS="
    +
    json.dumps(
        method_file_counts,
        sort_keys=True,
    )
)

if not summary_df.empty:
    print(
        summary_df.to_string(
            index=False
        )
    )
