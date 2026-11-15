#!/usr/bin/env python3
import csv
import hashlib
import io
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MASTER = ROOT / (
    "01_manifest/final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

SOURCE = ROOT / (
    "09_external_validation/"
    "external_one_shot_evaluation_v1_20260917/"
    "external_query_level_results.tsv"
)

Q50 = ROOT / (
    "21_frontiers_revision_20261112/"
    "02_E14_Q50_stratified/01_manifest/"
    "E14_Q50_STRATIFIED_MANIFEST.tsv"
)

OUT = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal/W01_v2"
)

HGV = "HGV_Dashing2_tau0p98"

EXPECTED = (
    "690b34e738664f6c22be96abd5db136514400607e00"
    "d7f10a18d37be898b58ef"
)

COHORTS = [
    "B1",
    "B2_seen_genus",
    "B2_unseen_genus"
]

LEVEL = {
    "B1": "species",
    "B2_seen_genus": "genus",
    "B2_unseen_genus": "family"
}

PCA = ["PCA64_L2", "PCA64_Euclidean"]

SEEDS = [
    "20260915",
    "20260916",
    "20260917"
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_tsv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fields):
    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)



def rank1(row, level="species"):

    value = str(
        row.get(level + "_first_rank") or ""
    ).strip()

    missing_tokens = {
        "", "na", "n/a", "nan", "none", "null", "-"
    }

    if value.lower() in missing_tokens:

        truth = str(
            row.get("query_" + level) or ""
        ).strip()

        top = str(
            row.get("top1_" + level) or ""
        ).strip()

        flag = str(
            row.get("any_reference_above_threshold") or ""
        ).strip().lower()

        count = str(
            row.get("references_above_threshold") or ""
        ).strip()

        try:
            zero_count = bool(count) and float(count) == 0.0
        except ValueError:
            zero_count = False

        if truth and truth.lower() not in missing_tokens:

            if top and top.lower() not in missing_tokens:

                if top != truth:
                    return 0

            elif (
                flag in {"false", "0", "no", "n"}
                or zero_count
            ):
                return 0

        raise RuntimeError(
            "Ambiguous or contradictory missing rank: "
            + str(row.get("query_accession"))
            + " level=" + level
        )

    try:
        x = float(value)
    except ValueError:
        raise RuntimeError(
            "Invalid rank: " + repr(value)
        )

    require(math.isfinite(x), "Nonfinite rank")

    return int(x == 1.0)

def pct(value):
    return round(value * 100, 4)


def quantile(values, p):
    values = sorted(values)
    z = (len(values) - 1) * p
    a = int(z)
    b = math.ceil(z)

    if a == b:
        return values[a]

    return (
        values[a] * (b - z)
        + values[b] * (z - a)
    )


def cluster_bootstrap(groups, seed, repeat=1000):
    values = [
        (sum(v), len(v))
        for _, v in sorted(groups.items())
    ]

    rng = random.Random(seed)
    boot = []

    for _ in range(repeat):
        sample = [
            values[rng.randrange(len(values))]
            for _ in values
        ]

        numerator = sum(a for a, n in sample)
        denominator = sum(n for a, n in sample)

        boot.append(numerator / denominator)

    return (
        pct(quantile(boot, 0.025)),
        pct(quantile(boot, 0.975))
    )


def mcnemar_exact(b, c):
    n = b + c

    if n == 0:
        return 1.0

    p = 2 * sum(
        math.comb(n, k)
        for k in range(min(b, c) + 1)
    ) / (2 ** n)

    return min(1.0, p)


def reconstruct_q50(groups):
    rng = random.Random(20261112)

    result = []
    used = set()

    for species in sorted(groups):
        accession = rng.choice(sorted(groups[species]))

        result.append(
            (accession, species, 1, 20261112)
        )
        used.add(accession)

    eligible = [
        sp for sp in sorted(groups)
        if len(groups[sp]) >= 2
    ]

    need = 50 - len(result)

    require(
        need >= 0 and len(eligible) >= need,
        "Cannot sample 50 queries"
    )

    extra_species = rng.sample(eligible, need)

    for species in sorted(extra_species):
        remaining = [
            a for a in sorted(groups[species])
            if a not in used
        ]

        accession = rng.choice(remaining)

        result.append(
            (accession, species, 2, 20261112)
        )
        used.add(accession)

    return sorted(
        result,
        key=lambda r: (r[1], r[0])
    )


print("=" * 72)
print("HGV V3 WINDOWS MINIMAL REVISION - W01")
print("FROZEN E14 + E15 STATISTICAL ANALYSIS")
print("=" * 72)

require(MASTER.is_file(), f"Missing: {MASTER}")
require(SOURCE.is_file(), f"Missing: {SOURCE}")

OUT.mkdir(parents=True, exist_ok=True)

# ==================================================
# 1. MASTER AND EXTERNAL DATA CHECK
# ==================================================

master_rows = read_tsv(MASTER)

require(
    len(master_rows) == 3154,
    "Master count is not 3154"
)

master = {
    r["assembly_accession_final"].strip(): r
    for r in master_rows
}

require(
    len(master) == 3154,
    "Duplicate master accession"
)

counts = Counter(
    r["final_cohort"].strip()
    for r in master_rows
)

expected_counts = {
    "A": 2374,
    "B1": 391,
    "B2_seen_genus": 196,
    "B2_unseen_genus": 193
}

require(
    counts == expected_counts,
    "Unexpected cohort counts: " + str(counts)
)

print("\n[1] MASTER:", dict(counts))

external = read_tsv(SOURCE)

require(
    len(external) == 6240,
    "External record count is not 6240"
)

print("External prediction rows:", len(external))

pred = defaultdict(dict)

for r in external:

    acc = r["query_accession"].strip()
    method = r["method"].strip()
    cohort = r["cohort"].strip()

    seed = (
        r["model_seed"].strip()
        if method == HGV
        else "fixed"
    )

    require(acc in master, "Unknown query: " + acc)

    require(
        master[acc]["final_cohort"].strip() == cohort,
        "Cohort mismatch: " + acc
    )

    require(
        master[acc]["gtdb_species_cluster"].strip()
        == r["query_species"].strip(),
        "Taxonomy mismatch: " + acc
    )

    key = (cohort, method, seed)

    require(
        acc not in pred[key],
        "Duplicate prediction: " + str(key)
    )

    pred[key][acc] = r

b1_set = {
    acc for acc, r in master.items()
    if r["final_cohort"] == "B1"
}

for method in [HGV] + PCA:
    seeds = SEEDS if method == HGV else ["fixed"]

    for seed in seeds:
        require(
            set(pred[("B1", method, seed)]) == b1_set,
            "Incomplete predictions: "
            + method + "/" + seed
        )

print("[PASS] 391 B1 queries and frozen predictions")

# ==================================================
# 2. E15: HGV vs PCA PAIRED COMPARISON
# ==================================================

print("\n[2] E15 HGV vs PCA-64")

pairs = []
failures = []
failure_groups = defaultdict(lambda: [0, 0])

for cohort in COHORTS:

    level = LEVEL[cohort]

    for seed in SEEDS:

        h = pred[(cohort, HGV, seed)]

        for pca in PCA:

            p = pred[(cohort, pca, "fixed")]

            require(
                set(h) == set(p) and len(h) > 0,
                f"Unpaired cohort: {cohort} {seed} {pca}"
            )

            n = len(h)

            hc = 0
            pc = 0

            h_only = 0
            p_only = 0
            both = 0
            neither = 0

            differences = defaultdict(list)
            h_by_species = defaultdict(list)
            p_by_species = defaultdict(list)

            for acc in sorted(h):

                a = rank1(h[acc], level)
                z = rank1(p[acc], level)

                hc += a
                pc += z

                h_only += int(a == 1 and z == 0)
                p_only += int(a == 0 and z == 1)
                both += int(a == 1 and z == 1)
                neither += int(a == 0 and z == 0)

                species_cluster = h[acc]["query_species"]

                differences[species_cluster].append(a - z)
                h_by_species[species_cluster].append(a)
                p_by_species[species_cluster].append(z)

                if cohort == "B1" and pca == "PCA64_L2":

                    row = master[acc]

                    group_key = (
                        seed,
                        species_cluster,
                        row["gtdb_species"],
                        row["gtdb_genus"]
                    )

                    failure_groups[group_key][0] += 1
                    failure_groups[group_key][1] += int(a == 0)

                    if a == 0:

                        failures.append({
                            "model_seed": seed,
                            "query_accession": acc,
                            "query_species": row["gtdb_species"],
                            "query_genus": row["gtdb_genus"],
                            "query_family": row["gtdb_family"],
                            "query_cluster": species_cluster,
                            "HGV_top1_reference":
                                h[acc].get("top1_reference", ""),
                            "HGV_top1_species":
                                h[acc].get("top1_species", ""),
                            "PCA64_L2_correct": z,
                            "PCA64_L2_top1_species":
                                p[acc].get("top1_species", "")
                        })

            low, high = cluster_bootstrap(
                differences,
                20261113 + int(seed[-2:])
                + len(cohort) + len(pca)
            )

            pairs.append({
                "cohort": cohort,
                "rank1_level": level,
                "hgv_seed": seed,
                "pca_baseline": pca,
                "n_queries": n,
                "n_species_clusters": len(differences),
                "HGV_R1_pct": pct(hc / n),
                "PCA_R1_pct": pct(pc / n),
                "HGV_minus_PCA_micro_pp":
                    pct((hc - pc) / n),
                "HGV_macro_by_species_pct": pct(
                    sum(
                        sum(v) / len(v)
                        for v in h_by_species.values()
                    ) / len(h_by_species)
                ),
                "PCA_macro_by_species_pct": pct(
                    sum(
                        sum(v) / len(v)
                        for v in p_by_species.values()
                    ) / len(p_by_species)
                ),
                "paired_cluster_CI_low_pp": low,
                "paired_cluster_CI_high_pp": high,
                "HGV_only_correct": h_only,
                "PCA_only_correct": p_only,
                "both_correct": both,
                "both_wrong": neither,
                "McNemar_exact_p": round(
                    mcnemar_exact(h_only, p_only), 8
                )
            })

write_tsv(
    OUT / "W01_E15_PAIRED_HGV_PCA.tsv",
    pairs,
    list(pairs[0])
)

failure_columns = [
    "model_seed",
    "query_accession",
    "query_species",
    "query_genus",
    "query_family",
    "query_cluster",
    "HGV_top1_reference",
    "HGV_top1_species",
    "PCA64_L2_correct",
    "PCA64_L2_top1_species"
]

write_tsv(
    OUT / "W01_E15_B1_RANK1_FAILURES.tsv",
    failures,
    failure_columns
)

failure_summary = []

for key, values in sorted(failure_groups.items()):

    seed, cluster, species, genus = key
    n, errors = values

    failure_summary.append({
        "model_seed": seed,
        "species_cluster": cluster,
        "gtdb_species": species,
        "gtdb_genus": genus,
        "n_queries": n,
        "hgv_failures": errors,
        "failure_pct": pct(errors / n)
    })

write_tsv(
    OUT / "W01_E15_FAILURES_BY_SPECIES.tsv",
    failure_summary,
    list(failure_summary[0])
)

print("[PASS] HGV/PCA paired analysis")
print("[PASS] Rank-1 failure analysis")
print("Paired comparison rows:", len(pairs))
print("HGV failure rows:", len(failures))

# ==================================================
# 3. E14: RECONSTRUCT FROZEN Q50
# ==================================================

print("\n[3] E14 FROZEN Q50")

groups = defaultdict(list)

for acc, r in pred[("B1", HGV, SEEDS[0])].items():
    groups[r["query_species"]].append(acc)

require(
    len(groups) == 40,
    "B1 species count is not 40"
)

selected_rows = reconstruct_q50(groups)

buffer = io.StringIO(newline="")

writer = csv.writer(
    buffer,
    delimiter="\t",
    lineterminator="\r\n"
)

writer.writerow([
    "query_accession",
    "query_species",
    "selection_round",
    "sampling_seed"
])

writer.writerows(selected_rows)

content = buffer.getvalue().encode("utf-8")

sha = hashlib.sha256(content).hexdigest()

matched = sha == EXPECTED

if Q50.is_file():
    existing_sha = hashlib.sha256(
        Q50.read_bytes()
    ).hexdigest()

    matched = matched and existing_sha == EXPECTED

print("Reconstructed SHA256:", sha)
print("Matches frozen Ubuntu Q50:", matched)

if matched:

    Q50.parent.mkdir(parents=True, exist_ok=True)

    if not Q50.exists():
        Q50.write_bytes(content)

    selected = {r[0] for r in selected_rows}

    metrics = []

    for (cohort, method, seed), table in sorted(pred.items()):

        if cohort != "B1":
            continue

        if not selected.issubset(table):
            continue

        correct = sum(
            rank1(table[acc])
            for acc in selected
        )

        metrics.append({
            "method": method,
            "model_seed": seed,
            "n_query": 50,
            "top1_correct": correct,
            "top1_accuracy_pct": pct(correct / 50),
            "provenance":
                "recomputed from existing frozen predictions"
        })

    write_tsv(
        OUT / "W01_E14_Q50_FROZEN_ACCURACY.tsv",
        metrics,
        list(metrics[0])
    )

    # 500 species-stratified resamples;
    # prediction analysis only, NOT runtime testing.

    rng = random.Random(20261113)

    draw_scores = defaultdict(list)

    eligible = [
        sp for sp in sorted(groups)
        if len(groups[sp]) >= 2
    ]

    methods = [
        (method, seed, table)
        for (cohort, method, seed), table in sorted(pred.items())
        if cohort == "B1" and set(table) == b1_set
    ]

    for repetition in range(500):

        picks = {
            sp: [rng.choice(sorted(groups[sp]))]
            for sp in sorted(groups)
        }

        for sp in rng.sample(eligible, 50 - len(picks)):

            remaining = [
                acc for acc in sorted(groups[sp])
                if acc not in picks[sp]
            ]

            picks[sp].append(rng.choice(remaining))

        accessions = [
            acc
            for members in picks.values()
            for acc in members
        ]

        for method, seed, table in methods:

            accuracy = sum(
                rank1(table[acc])
                for acc in accessions
            ) / 50

            draw_scores[(method, seed)].append(accuracy)

    resampling = []

    for (method, seed), scores in sorted(draw_scores.items()):

        resampling.append({
            "method": method,
            "model_seed": seed,
            "draws": len(scores),
            "mean_accuracy_pct": pct(
                sum(scores) / len(scores)
            ),
            "p025_accuracy_pct": pct(
                quantile(scores, 0.025)
            ),
            "p975_accuracy_pct": pct(
                quantile(scores, 0.975)
            )
        })

    write_tsv(
        OUT / "W01_E14_500_STRATIFIED_RESAMPLINGS.tsv",
        resampling,
        list(resampling[0])
    )

    print("[PASS] Frozen Q50 reproduced")
    print("[PASS] Existing Q50 accuracy calculated")
    print("[PASS] 500 stratified resamplings calculated")

else:

    diagnostic = OUT / "W01_Q50_DIAGNOSTIC_NOT_FROZEN.tsv"

    diagnostic.write_bytes(content)

    print("[STOP] Frozen Q50 checksum mismatch")
    print("E14 results were not generated.")
    print("E15 results remain available.")

# ==================================================
# 4. CHECK REMAINING FROZEN ASSETS
# ==================================================

print("\n[4] REMAINING ASSET CHECK")

assets = [
    (
        "E5 internal results",
        ROOT / "09_baselines/E5_unified_baseline_v1_20260917"
    ),
    (
        "E8 fragmentation",
        ROOT / (
            "12_fragmentation/E8_fragmentation_v1_20260925_v6/"
            "E8_FRAGMENTATION_SUMMARY.tsv"
        )
    ),
    (
        "E12B final Table 7",
        ROOT / (
            "16_scalability/E12B_synchronized_resource_v1_20260930/"
            "04_summary/finalize_v3_20261001/"
            "TABLE7_SOURCE_DATA_FINAL.tsv"
        )
    ),
    (
        "HGV canonical checkpoint",
        ROOT / (
            "09_external_validation/"
            "canonical_A_model_training_v1_20260917/"
            "G1/split20260915/model20260915/best_model.pt"
        )
    ),
    (
        "PCA frozen",
        ROOT / (
            "09_external_validation/"
            "canonical_validation_threshold_freeze_v1_20260917/"
            "PCA64_G1_TRAIN_FROZEN.npz"
        )
    ),
    (
        "GTDB metadata",
        ROOT / "00_metadata/bac120_metadata_r232.tsv.gz"
    )
]

asset_rows = []

for label, path in assets:

    exists = path.exists()

    asset_rows.append({
        "asset": label,
        "exists": exists,
        "path": str(path)
    })

    print(
        "[FOUND]" if exists else "[MISSING]",
        label
    )

write_tsv(
    OUT / "W01_NEXT_ASSET_CHECK.tsv",
    asset_rows,
    ["asset", "exists", "path"]
)

# ==================================================
# 5. FINAL STATUS
# ==================================================

status = (
    "E15 paired HGV/PCA and Rank-1 analysis: COMPLETE\n"
    "E14 frozen Q50 prediction accuracy: "
    + ("COMPLETE" if matched else "BLOCKED_SHA") + "\n"
    "E14 new 11-method runtime measurement: NOT DONE\n"
    "E13 real MAG evaluation: NOT DONE\n"
    "G2/G3/G4 internal MCC: NOT YET AUDITED\n"
    "Public repository: NOT YET PUBLISHED\n"
    "Original experiments and models unchanged.\n"
)

(OUT / "W01_STATUS.txt").write_text(
    status,
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("WINDOWS W01 FINISHED")
print("=" * 72)
print(status)
print("Output:", OUT)
