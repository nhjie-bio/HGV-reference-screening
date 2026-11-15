import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E11 = ROOT / "15_reference_screening/E11_reference_screening_v1_20260925_v6"
EXT = ROOT / "09_external_validation/external_one_shot_evaluation_v1_20260917/external_query_level_results.tsv"
OUT = ROOT / "21_frontiers_revision_20261112/90_windows_minimal/W05"

LEVELS = {
    "B1": ("species", "gtdb_species_cluster", 391),
    "B2_seen_genus": ("genus", "gtdb_genus", 196),
    "B2_unseen_genus": ("family", "gtdb_family", 193),
}
SEEDS = ("20260915", "20260916", "20260917")

def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if any(None in r for r in rows):
        raise RuntimeError("Malformed TSV: " + str(path))
    return rows

def bit(value):
    v = float(value)
    if not math.isfinite(v) or v not in (0.0, 1.0):
        raise RuntimeError("Invalid binary value: " + str(value))
    return int(v)

def external_r1(row, level):
    raw = (row.get(level + "_first_rank") or "").strip()

    if raw and raw.lower() not in ("na", "nan", "none"):
        return int(float(raw) == 1.0)

    truth = (row.get("query_" + level) or "").strip()
    top = (row.get("top1_" + level) or "").strip()
    flag = (row.get("any_reference_above_threshold") or "").lower().strip()

    if truth and top and truth != top:
        return 0

    if truth and not top and flag in ("false", "0", "no"):
        return 0

    raise RuntimeError(
        "Unresolved missing rank: " + row["query_accession"]
    )

def save(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=fields, delimiter="\t",
            lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("W05 - E11 AND EXTERNAL PROTOCOL RECONCILIATION")
print("=" * 72)

print("\n[1] E11 PROVENANCE")

for name in (
    "E11_PROVENANCE.json",
    "E11_REFERENCE_SCREENING_REPORT.txt",
):
    p = E11 / name
    if not p.is_file():
        raise RuntimeError("Missing provenance: " + str(p))
    print("\nFILE:", name)
    print(p.read_text(encoding="utf-8-sig"))

summary = {
    r["external_type"]: r
    for r in load(E11 / "E11_RETRIEVAL_SUMMARY.tsv")
}

queries = {
    (r["external_type"], r["query_accession"]): r
    for r in load(E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv")
}

external = {}

for r in load(EXT):
    if (
        r["method_type"] != "HGV"
        or r["method"] != "HGV_Dashing2_tau0p98"
    ):
        continue

    key = (
        r["cohort"],
        r["model_seed"],
        r["query_accession"]
    )

    if key in external:
        raise RuntimeError("Duplicate external key: " + str(key))

    external[key] = r

if len(queries) != 780 or len(external) != 2340:
    raise RuntimeError("Unexpected E11 or HGV external row count")

results = []
differences = []

print("\n[2] PER-QUERY COMPARISON")

for cohort, (level, target_field, expected_n) in LEVELS.items():

    qset = {
        accession: r
        for (group, accession), r in queries.items()
        if group == cohort
    }

    if len(qset) != expected_n:
        raise RuntimeError("E11 cohort count mismatch: " + cohort)

    e11_correct = sum(
        bit(r["Recall@1"]) for r in qset.values()
    )

    reported = float(summary[cohort]["Recall@1"])
    if not math.isclose(
        e11_correct / expected_n, reported,
        rel_tol=0, abs_tol=1e-10
    ):
        raise RuntimeError("E11 summary/query mismatch: " + cohort)

    for seed in SEEDS:

        both_correct = 0
        both_wrong = 0
        e11_only = 0
        external_only = 0
        taxon_matches = 0
        thresholds = set()

        for acc, q in qset.items():

            key = (cohort, seed, acc)

            if key not in external:
                raise RuntimeError("Missing external query: " + str(key))

            r = external[key]

            a = bit(q["Recall@1"])
            b = external_r1(r, level)

            thresholds.add(r.get("threshold", ""))

            if q["target_taxon"] == r["query_" + level]:
                taxon_matches += 1

            if a and b:
                both_correct += 1
            elif not a and not b:
                both_wrong += 1
            elif a:
                e11_only += 1
            else:
                external_only += 1

            if a != b:
                differences.append({
                    "cohort": cohort,
                    "seed": seed,
                    "query_accession": acc,
                    "E11_R1": a,
                    "External_R1": b,
                    "E11_first_relevant_rank":
                        q["first_relevant_rank"],
                    "External_first_rank":
                        r[level + "_first_rank"],
                    "External_threshold":
                        r.get("threshold", ""),
                    "E11_target_taxon":
                        q["target_taxon"],
                    "External_query_taxon":
                        r["query_" + level],
                })

        row = {
            "cohort": cohort,
            "seed": seed,
            "n": expected_n,
            "E11_R1": round(reported, 8),
            "External_R1": round(
                (both_correct + external_only) / expected_n, 8
            ),
            "both_correct": both_correct,
            "E11_only_correct": e11_only,
            "External_only_correct": external_only,
            "both_wrong": both_wrong,
            "matching_target_taxon": taxon_matches,
            "thresholds": ";".join(sorted(thresholds)),
        }

        results.append(row)

        print(
            cohort, seed,
            "E11=", row["E11_R1"],
            "External=", row["External_R1"],
            "E11_only=", e11_only,
            "External_only=", external_only,
            "taxon_match=", taxon_matches,
            "/", expected_n
        )

save(
    OUT / "W05_E11_EXTERNAL_COMPARISON.tsv",
    results,
    list(results[0])
)

save(
    OUT / "W05_E11_EXTERNAL_DISAGREEMENTS.tsv",
    differences,
    [
        "cohort", "seed", "query_accession",
        "E11_R1", "External_R1",
        "E11_first_relevant_rank",
        "External_first_rank",
        "External_threshold",
        "E11_target_taxon",
        "External_query_taxon"
    ]
)

print("\n[3] STATUS")
print("E11 summary and per-query values: PASS")
print("Paired comparison rows:", len(results))
print("Different outcomes:", len(differences))
print("No models loaded or modified")
print("No experiments rerun")
print("Output:", OUT)
