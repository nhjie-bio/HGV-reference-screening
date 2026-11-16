from pathlib import Path
import pandas as pd
import json

ROOT = Path(
    "HGV_V3_revision_20260915/"
    "11_manuscript/HGV_V3_final_manuscript_figures_v3_20260926"
)

SRC = ROOT / "00_source_snapshot"

FILES = [
    "source_data/main_internal.tsv",
    "source_data/external_long.tsv",
    "source_data/external_raw.tsv",

    "source_data/e6_modes.tsv",
    "provenance/FIGURE4_E6_MODES_NORMALIZED.tsv",
    "source_data/e6_contrasts.tsv",

    "source_data/e7.tsv",
    "source_data/e8.tsv",

    "source_data/e9_ig.tsv",

    "source_data/e10_corr.tsv",
    "source_data/e10_tax.tsv",
    "source_data/e10_confound.tsv",
    "source_data/e10_tree_summary.tsv",
    "source_data/e10_tree_pairs.tsv",

    "source_data/e11.tsv",
    "source_data/e12.tsv",

    "provenance/FIG2_main_normalized.tsv",
    "provenance/CANONICAL_SOURCE_MANIFEST.tsv",
]

print("=" * 120)
print("HGV-V3 FINAL FIGURE SOURCE SCHEMA AUDIT v4")
print("=" * 120)

summary = []

for rel in FILES:
    p = SRC / rel

    print()
    print("#" * 120)
    print(f"FILE: {rel}")
    print("#" * 120)

    if not p.exists():
        print("[MISSING]")
        summary.append({
            "file": rel,
            "exists": False,
        })
        continue

    print(f"SIZE_BYTES: {p.stat().st_size}")

    try:
        df = pd.read_csv(p, sep="\t")
    except Exception as e:
        print(f"[TSV READ FAIL] {repr(e)}")

        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except Exception as e2:
            print(f"[AUTO READ FAIL] {repr(e2)}")
            summary.append({
                "file": rel,
                "exists": True,
                "readable": False,
            })
            continue

    print(f"SHAPE: {df.shape}")

    print()
    print("COLUMNS:")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:02d}. {repr(c)}")

    print()
    print("DTYPES:")
    for c, d in df.dtypes.items():
        print(f"  {c}: {d}")

    print()
    print("HEAD:")
    with pd.option_context(
        "display.max_columns", 100,
        "display.width", 240,
        "display.max_colwidth", 60,
    ):
        print(df.head(8).to_string(index=False))

    print()
    print("NON-NULL COUNTS:")
    for c in df.columns:
        print(f"  {c}: {int(df[c].notna().sum())}")

    # print compact unique values for small categorical columns
    print()
    print("SMALL UNIQUE VALUES:")
    for c in df.columns:
        try:
            n = df[c].nunique(dropna=True)
        except Exception:
            continue

        if 1 <= n <= 20:
            vals = df[c].dropna().astype(str).unique().tolist()
            vals = vals[:20]
            print(f"  {c} ({n}): {vals}")

    summary.append({
        "file": rel,
        "exists": True,
        "readable": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
    })

print()
print("=" * 120)
print("COMPACT JSON SUMMARY")
print("=" * 120)
print(json.dumps(summary, indent=2, ensure_ascii=False))

OUT = ROOT / "03_audit" / "CANONICAL_SOURCE_SCHEMA_AUDIT_v4.txt"

with OUT.open("w") as f:
    for x in summary:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print()
print(f"[PASS] Schema audit written to:")
print(OUT)
print("=" * 120)
