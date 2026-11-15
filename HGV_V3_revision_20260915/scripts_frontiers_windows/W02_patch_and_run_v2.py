import sys
import subprocess
from pathlib import Path

root = Path.cwd()
folder = root / "scripts_frontiers_windows"

src = folder / "W02_existing_evidence_audit_v1.py"
dst = folder / "W02_existing_evidence_audit_v2.py"

if not src.is_file():
    raise SystemExit("[STOP] W02-v1 not found")

s = src.read_text(encoding="utf-8-sig")

old = '''for row in table7:

    for field in (
        "Q50_R1_all_queries",
        "build_s_median",
        "warm_query_s_median",
        "complete_s_median",
        "reference_index_MiB"
    ):
        float(row[field])
'''

new = '''missing_metric_rows = []

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
'''

if s.count(old) != 1:
    raise SystemExit(
        "[STOP] Expected Table 7 validation block not found"
    )

s = s.replace(old, new, 1)

if "import math\n" not in s:
    if s.count("import hashlib\n") != 1:
        raise SystemExit("[STOP] Cannot add math import")
    s = s.replace(
        "import hashlib\n",
        "import hashlib\nimport math\n",
        1
    )

marker = "if STATUS.is_file():"

addition = '''write_tsv(
    OUT / "W02_E12B_MISSING_METRICS.tsv",
    missing_metric_rows,
    ["method", "metric", "original_value", "status"]
)

log(
    "E12B not-reported metric cells: "
    + str(len(missing_metric_rows))
)

'''

if s.count(marker) != 1:
    raise SystemExit("[STOP] E12B status marker not unique")

s = s.replace(marker, addition + marker, 1)

old_out = '"90_windows_minimal/W02"'
new_out = '"90_windows_minimal/W02_v2"'

if s.count(old_out) != 1:
    raise SystemExit("[STOP] W02 output directory not found")

s = s.replace(old_out, new_out, 1)

if dst.exists():
    if dst.read_text(encoding="utf-8-sig") != s:
        raise SystemExit("[STOP] Existing v2 differs")
else:
    dst.write_text(s, encoding="utf-8")

print("[PASS] W02-v2 prepared")
print("[PASS] Original W02-v1 preserved")
print("[PASS] Empty E12B metrics remain NOT_REPORTED")

result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(dst)],
    cwd=root
)

if result.returncode:
    raise SystemExit(result.returncode)

print()
print("===== EXECUTING W02-v2 =====", flush=True)

result = subprocess.run(
    [sys.executable, str(dst)],
    cwd=root
)

raise SystemExit(result.returncode)
