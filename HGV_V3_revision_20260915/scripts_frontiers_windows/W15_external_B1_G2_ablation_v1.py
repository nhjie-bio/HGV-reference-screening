#!/usr/bin/env python3

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

WORK = ROOT / (
    "21_frontiers_revision_20261112/90_windows_minimal"
)

OUT = WORK / "W15_EXTERNAL_B1_G2_ABLATION_v1_20261114"

W13 = ROOT / (
    "scripts_frontiers_windows/"
    "W13_HGV_two_questions_oneshot_v1.py"
)

W14 = WORK / "W14_E6_MANIFEST_RECOVERED_v1_20261114"

E11 = ROOT / (
    "15_reference_screening/"
    "E11_reference_screening_v1_20260925_v6"
)

E6 = ROOT / (
    "10_ablation/"
    "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

MANIFEST = E6 / (
    "04_canonical_30cell_manifest_v1_20260923/"
    "E6_30CELL_CANONICAL_MANIFEST.tsv"
)

SEEDS = (
    20260915,
    20260916,
    20260917,
    20260918,
    20260919
)

MODES = ("D0", "D1", "D3")

KS = (1, 5, 10, 20, 50, 100)

TARGETS = (0.95, 0.99, 1.0)

BOOTSTRAP = 2000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def write_tsv(filename, data):
    require(
        len(data) > 0,
        "Empty output: " + filename
    )

    path = OUT / filename

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(data[0]),
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(data)


def import_w13():
    require(
        W13.is_file(),
        "Original W13 script missing"
    )

    spec = importlib.util.spec_from_file_location(
        "w13_for_w15",
        W13
    )

    require(
        spec is not None and spec.loader is not None,
        "Cannot import W13"
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def cluster_bootstrap(
    difference,
    species,
    seed=20261114
):
    groups = defaultdict(list)

    for i, label in enumerate(species):
        groups[str(label)].append(i)

    require(
        len(groups) >= 2,
        "Insufficient species clusters"
    )

    indices = list(groups.values())

    values = np.asarray(
        difference,
        dtype=np.float64
    )

    sums = np.array([
        values[idx].sum()
        for idx in indices
    ])

    sizes = np.array([
        len(idx)
        for idx in indices
    ])

    rng = np.random.default_rng(seed)

    samples = np.zeros(
        BOOTSTRAP,
        dtype=np.float64
    )

    for i in range(BOOTSTRAP):

        counts = np.bincount(
            rng.integers(
                len(sizes),
                size=len(sizes)
            ),
            minlength=len(sizes)
        )

        samples[i] = float(
            (counts @ sums) /
            (counts @ sizes)
        )

    ci = np.quantile(
        samples,
        [0.025, 0.975]
    )

    return ci, len(indices)


def recall_metrics(
    method,
    ranks,
    n_reference,
    n_query
):
    ranks = np.asarray(
        ranks,
        dtype=np.int32
    )

    require(
        ranks.shape == (n_query,),
        "Rank shape mismatch: " + method
    )

    require(
        (ranks >= 1).all()
        and (ranks <= n_reference).all(),
        "Invalid rank: " + method
    )

    result = {
        "method": method,
        "queries": n_query,
        "reference_n": n_reference,
        "embedding_dim": 64,
        "bytes_per_genome": 256
    }

    for k in KS:
        result[f"R{k}"] = round(
            float((ranks <= k).mean()),
            8
        )

        result[f"hit_{k}"] = int(
            (ranks <= k).sum()
        )

    result["MRR"] = round(
        float(np.mean(1.0 / ranks)),
        8
    )

    return result


def candidate_budget(
    method,
    ranks,
    n_query,
    n_reference
):
    sorted_ranks = np.sort(
        np.asarray(ranks, dtype=int)
    )

    result = []

    for target in TARGETS:

        index = math.ceil(
            n_query * target
        ) - 1

        k = int(sorted_ranks[index])

        result.append({
            "method": method,
            "target": target,
            "queries": n_query,
            "K": k,
            "reference_n": n_reference,
            "candidate_percent": round(
                100 * k / n_reference,
                5
            ),
            "excluded_percent": round(
                100 * (1 - k / n_reference),
                5
            )
        })

    return result


def main():

    print("=" * 78)
    print(
        "W15 | EXTERNAL E11-B1 | "
        "G2 D0/D1/D3 FIVE-SEED RETRIEVAL"
    )
    print("=" * 78, flush=True)

    require(
        not OUT.exists(),
        "Output already exists: " + str(OUT)
    )

    status_file = W14 / "W14_STATUS.txt"

    require(
        status_file.is_file(),
        "W14 status file missing"
    )

    status = status_file.read_text(
        encoding="utf-8-sig"
    )

    require(
        "W14_E6_D0_D1_D3=PASS" in status
        and "CHECKPOINT_SHA256=15/15" in status,
        "W14 checkpoint audit did not PASS"
    )

    mod = import_w13()

    mm, fi, X = mod.load_common()

    split_path, split_column, roles = (
        mod.resolve_g2_split(mm)
    )

    references = sorted([
        accession
        for accession, role in roles.items()
        if role == "train"
    ])

    g2_test = sorted([
        accession
        for accession, role in roles.items()
        if role == "test"
    ])

    require(
        len(references) == 1662
        and len(g2_test) == 356,
        "G2 split count mismatch"
    )

    require(
        set(references).isdisjoint(g2_test),
        "G2 train/test overlap"
    )

    ref_species = np.array([
        mm[a]["gtdb_species_cluster"]
        for a in references
    ], dtype=object)

    ref_species_set = set(ref_species)

    test_species_set = {
        mm[a]["gtdb_species_cluster"]
        for a in g2_test
    }

    require(
        ref_species_set.isdisjoint(
            test_species_set
        ),
        "Species leakage between G2 train/test"
    )

    # --------------------------------------------------
    # E11 B1 query selection
    # --------------------------------------------------

    e11_order = mod.order(
        E11 / "E11_QUERY_ORDER.tsv",
        780
    )

    metadata = mod.rows(
        E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv"
    )

    require(
        len(metadata) == 780,
        "E11 metadata count mismatch"
    )

    official = {
        r["query_accession"]: r
        for r in metadata
    }

    require(
        len(official) == 780
        and set(official) == set(e11_order),
        "E11 query accession mismatch"
    )

    b1 = [
        a for a in e11_order
        if official[a]["external_type"] == "B1"
    ]

    require(
        len(b1) == 391,
        "Expected 391 E11-B1 queries"
    )

    filter_records = []
    selected = []

    for accession in b1:

        require(
            accession in mm
            and accession in fi,
            "Missing B1 frozen feature"
        )

        require(
            accession not in roles
            and accession not in references,
            "B1 overlaps original G2 A cohort"
        )

        row = official[accession]

        require(
            row["target_level"]
            == "gtdb_species_cluster",
            "Unexpected B1 taxonomy target"
        )

        truth = row["target_taxon"].strip()

        require(
            truth
            and truth
            == mm[accession][
                "gtdb_species_cluster"
            ].strip(),
            "E11/master species label mismatch"
        )

        present = truth in ref_species_set

        filter_records.append({
            "accession": accession,
            "E11_group": "B1",
            "target_species": truth,
            "G2_reference_species_present":
                int(present),
            "included_W15": int(present),
            "reason": (
                "included"
                if present
                else "absent_in_G2_training_reference"
            )
        })

        if present:
            selected.append(accession)

    n = len(selected)

    require(
        n >= 20,
        "Too few matched-species queries: "
        + str(n)
    )

    species_labels = np.array([
        mm[a]["gtdb_species_cluster"]
        for a in selected
    ], dtype=object)

    require(
        len(set(species_labels)) >= 2,
        "Insufficient species clusters"
    )

    print(
        "[FILTER] B1 total=391;",
        "eligible=", n,
        "excluded=", 391 - n,
        flush=True
    )

    print(
        "[REFERENCE] G2 train=1662;",
        "no HGV retraining",
        flush=True
    )

    # --------------------------------------------------
    # Verify 15 frozen checkpoints
    # --------------------------------------------------

    manifest = {
        r["cell_id"]: r
        for r in mod.rows(MANIFEST)
    }

    checkpoint_rows = mod.rows(
        W14 / "W14_FROZEN_CHECKPOINTS_SHA256.tsv"
    )

    require(
        len(checkpoint_rows) == 15,
        "Expected 15 checkpoint audit rows"
    )

    checkpoints = {}
    verified = []

    for rec in checkpoint_rows:

        mode = rec["mode"]
        seed = int(rec["model_seed"])

        require(
            mode in MODES and seed in SEEDS,
            "Unexpected model in W14"
        )

        cell = f"{mode}_seed{seed}"

        require(
            cell in manifest
            and rec["cell_id"] == cell,
            "Manifest checkpoint mismatch"
        )

        canonical = manifest[cell]

        require(
            canonical["protocol"] == "G2"
            and canonical["split_seed"]
            == "20260915"
            and canonical["mode"] == mode
            and int(canonical["model_seed"])
            == seed,
            "Incorrect frozen model metadata"
        )

        path = Path(rec["checkpoint"])

        expected = canonical[
            "checkpoint_sha256"
        ].lower()

        require(
            path.is_file()
            and rec["expected_sha256"].lower()
            == expected
            and sha256(path) == expected,
            "Checkpoint SHA256 mismatch: " + cell
        )

        require(
            (mode, seed) not in checkpoints,
            "Duplicate checkpoint: " + cell
        )

        checkpoints[(mode, seed)] = path

        verified.append({
            "cell_id": cell,
            "source_block":
                canonical["source_block"],
            "path": str(path),
            "sha256": expected,
            "status": "PASS"
        })

    require(
        len(checkpoints) == 15,
        "Incomplete checkpoint set"
    )

    print(
        "[CHECKPOINTS] 15/15 SHA256 PASS",
        flush=True
    )

    OUT.mkdir(
        parents=True,
        exist_ok=False
    )

    write_tsv(
        "W15_B1_QUERY_SELECTION.tsv",
        filter_records
    )

    write_tsv(
        "W15_CHECKPOINTS_VERIFIED.tsv",
        verified
    )

    # --------------------------------------------------
    # Load original canonical k6 features
    # --------------------------------------------------

    Xr = np.asarray(
        X[[fi[a] for a in references]],
        dtype=np.float32
    )

    Xq = np.asarray(
        X[[fi[a] for a in selected]],
        dtype=np.float32
    )

    # --------------------------------------------------
    # HGV frozen inference
    # --------------------------------------------------

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from sklearn.decomposition import PCA

    torch.set_num_threads(
        min(8, os.cpu_count() or 1)
    )

    class HGV(nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(2080, 256)
            self.bn1 = nn.BatchNorm1d(256)

            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)

            self.fc3 = nn.Linear(128, 64)

        def forward(self, x):

            x = F.relu(
                self.bn1(self.fc1(x))
            )

            x = F.relu(
                self.bn2(self.fc2(x))
            )

            return F.normalize(
                self.fc3(x),
                p=2,
                dim=1
            )

    @torch.no_grad()
    def embed(model, features):

        model.eval()

        chunks = []

        for i in range(
            0, len(features), 256
        ):

            batch = torch.from_numpy(
                np.array(
                    features[i:i + 256],
                    dtype=np.float32,
                    copy=True
                )
            )

            chunks.append(
                model(batch).cpu().numpy()
            )

        z = np.concatenate(
            chunks
        ).astype(np.float32)

        require(
            z.shape == (len(features), 64)
            and np.isfinite(z).all(),
            "Invalid embeddings"
        )

        return z

    ranks = {}

    per_query = []
    curves = []
    k_required = []

    for mode in MODES:

        for seed in SEEDS:

            path = checkpoints[(mode, seed)]

            state = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            require(
                isinstance(state, dict)
                and "model_state_dict" in state,
                "Invalid checkpoint state"
            )

            model = HGV().eval()

            model.load_state_dict(
                state["model_state_dict"],
                strict=True
            )

            zr = embed(model, Xr)
            zq = embed(model, Xq)

            method = f"{mode}_seed{seed}"

            D = mod.edist(zq, zr)

            r, eligible = mod.getranks(
                D,
                species_labels,
                ref_species
            )

            require(
                bool(eligible.all()),
                "Unexpected unmatched query"
            )

            ranks[method] = r

            curves.append(
                recall_metrics(
                    method,
                    r,
                    len(references),
                    n
                )
            )

            k_required.extend(
                candidate_budget(
                    method,
                    r,
                    n,
                    len(references)
                )
            )

            for accession, species, rank in zip(
                selected,
                species_labels,
                r
            ):

                per_query.append({
                    "method": method,
                    "accession": accession,
                    "target_species": species,
                    "first_species_rank": int(rank),
                    "R1": int(rank <= 1),
                    "R5": int(rank <= 5),
                    "R20": int(rank <= 20)
                })

            print(
                "[HGV]", method,
                "R1=",
                round(float((r <= 1).mean()), 5),
                "R20=",
                round(float((r <= 20).mean()), 5),
                flush=True
            )

            del model, state

    # --------------------------------------------------
    # PCA64 trained on G2 references ONLY
    # --------------------------------------------------

    fit = PCA(
        n_components=64,
        svd_solver="randomized",
        random_state=20260915,
        iterated_power=5
    )

    fit.fit(Xr)

    Rp = fit.transform(
        Xr
    ).astype(np.float32)

    Qp = fit.transform(
        Xq
    ).astype(np.float32)

    pca_methods = (
        (
            "G2train_PCA64_Euclidean",
            mod.edist(Qp, Rp)
        ),
        (
            "G2train_PCA64_L2",
            mod.edist(
                mod.unit(Qp),
                mod.unit(Rp)
            )
        )
    )

    for method, D in pca_methods:

        r, eligible = mod.getranks(
            D,
            species_labels,
            ref_species
        )

        require(
            bool(eligible.all()),
            "PCA eligibility mismatch"
        )

        ranks[method] = r

        curves.append(
            recall_metrics(
                method,
                r,
                len(references),
                n
            )
        )

        k_required.extend(
            candidate_budget(
                method,
                r,
                n,
                len(references)
            )
        )

        for accession, species, rank in zip(
            selected,
            species_labels,
            r
        ):

            per_query.append({
                "method": method,
                "accession": accession,
                "target_species": species,
                "first_species_rank": int(rank),
                "R1": int(rank <= 1),
                "R5": int(rank <= 5),
                "R20": int(rank <= 20)
            })

        print(
            "[PCA]", method,
            "R1=",
            round(float((r <= 1).mean()), 5),
            "R20=",
            round(float((r <= 20).mean()), 5),
            flush=True
        )

    # --------------------------------------------------
    # Five-seed summary
    # --------------------------------------------------

    summary = []

    for mode in MODES:

        all_ranks = np.stack([
            ranks[f"{mode}_seed{s}"]
            for s in SEEDS
        ])

        k99 = [
            next(
                x["K"]
                for x in k_required
                if x["method"]
                == f"{mode}_seed{s}"
                and x["target"] == 0.99
            )
            for s in SEEDS
        ]

        for k in (1, 5, 20):

            values = (
                all_ranks <= k
            ).mean(axis=1)

            summary.append({
                "method": mode,
                "K": k,
                "queries": n,
                "five_seed_mean": round(
                    float(values.mean()),
                    8
                ),
                "five_seed_SD": round(
                    float(values.std(ddof=1)),
                    8
                ),
                "K99_median": int(
                    np.median(k99)
                ),
                "K99_min": min(k99),
                "K99_max": max(k99)
            })

    # --------------------------------------------------
    # Paired comparisons
    # --------------------------------------------------

    comparisons = (
        ("D3", "D0"),
        ("D3", "D1"),
        ("D0", "G2train_PCA64_L2"),
        ("D3", "G2train_PCA64_L2")
    )

    by_seed = []
    clustered = []

    for lhs, rhs in comparisons:

        A = np.stack([
            ranks[f"{lhs}_seed{s}"]
            for s in SEEDS
        ])

        if rhs in MODES:

            B = np.stack([
                ranks[f"{rhs}_seed{s}"]
                for s in SEEDS
            ])

        else:

            B = np.stack([
                ranks[rhs]
                for _ in SEEDS
            ])

        for k in (1, 5, 20):

            seed_deltas = []

            for i, seed in enumerate(SEEDS):

                a = A[i] <= k
                b = B[i] <= k

                a_only = int(
                    np.count_nonzero(
                        a & ~b
                    )
                )

                b_only = int(
                    np.count_nonzero(
                        ~a & b
                    )
                )

                delta = float(
                    a.mean() - b.mean()
                )

                seed_deltas.append(delta)

                by_seed.append({
                    "method_A": lhs,
                    "method_B": rhs,
                    "seed": seed,
                    "K": k,
                    "queries": n,
                    "A_correct": int(a.sum()),
                    "B_correct": int(b.sum()),
                    "A_only": a_only,
                    "B_only": b_only,
                    "delta": round(delta, 8),
                    "mcnemar_p_unadjusted": round(
                        mod.mcnemar(
                            a_only,
                            b_only
                        ),
                        8
                    )
                })

            # Average the same query over five seeds.
            # Bootstrap independent species clusters,
            # not the five training seeds.

            per_query_delta = (
                (A <= k).mean(axis=0)
                -
                (B <= k).mean(axis=0)
            )

            ci, n_clusters = cluster_bootstrap(
                per_query_delta,
                species_labels,
                seed=20261114 + k
            )

            clustered.append({
                "method_A": lhs,
                "method_B": rhs,
                "K": k,
                "queries": n,
                "species_clusters": n_clusters,
                "five_seed_mean_delta": round(
                    float(
                        np.mean(seed_deltas)
                    ),
                    8
                ),
                "bootstrap_ci_low": round(
                    float(ci[0]),
                    8
                ),
                "bootstrap_ci_high": round(
                    float(ci[1]),
                    8
                ),
                "seed_deltas": ";".join(
                    f"{x:+.6f}"
                    for x in seed_deltas
                ),
                "note":
                    "POST_HOC_species_cluster_bootstrap"
            })

    # --------------------------------------------------
    # Save all numerical outputs
    # --------------------------------------------------

    write_tsv(
        "W15_RECALL_BY_MODEL.tsv",
        curves
    )

    write_tsv(
        "W15_REQUIRED_CANDIDATES.tsv",
        k_required
    )

    write_tsv(
        "W15_FIVE_SEED_SUMMARY.tsv",
        summary
    )

    write_tsv(
        "W15_PER_QUERY_RANKS.tsv",
        per_query
    )

    write_tsv(
        "W15_PAIRED_BY_SEED.tsv",
        by_seed
    )

    write_tsv(
        "W15_CLUSTER_BOOTSTRAP.tsv",
        clustered
    )

    # --------------------------------------------------
    # Decision report
    # --------------------------------------------------

    report = [
        "W15 EXTERNAL E11-B1 / G2 FROZEN REFERENCE",
        f"B1 total=391",
        f"Eligible={n}",
        f"Excluded={391-n}",
        f"G2 references={len(references)}",
        f"Species clusters={len(set(species_labels))}",
        "",
        "All 15 frozen checkpoints verified.",
        "No HGV retraining.",
        "PCA fitted on G2 training references only.",
        "Single HGV and PCA64: 256 bytes/genome.",
        "Post-hoc exploratory analysis.",
        "",
        "FIVE-SEED RESULTS:"
    ]

    for row in summary:

        if row["K"] in (1, 20):

            report.append(
                f"{row['method']} "
                f"R@{row['K']}="
                f"{row['five_seed_mean']:.5f} "
                f"SD={row['five_seed_SD']:.5f}; "
                f"K99 median={row['K99_median']} "
                f"range={row['K99_min']}-"
                f"{row['K99_max']}"
            )

    report.append("")
    report.append("D3 VS D0/D1:")

    for row in clustered:

        if (
            row["method_A"] == "D3"
            and row["method_B"] in ("D0", "D1")
            and row["K"] in (1, 20)
        ):

            report.append(
                f"D3-{row['method_B']} "
                f"R@{row['K']} "
                f"delta="
                f"{row['five_seed_mean_delta']:+.5f} "
                f"CI=["
                f"{row['bootstrap_ci_low']:+.5f},"
                f"{row['bootstrap_ci_high']:+.5f}] "
                f"seed_deltas={row['seed_deltas']}"
            )

    report.append("")
    report.append("PCA64 COMPARISON:")

    for row in curves:

        if row["method"].startswith(
            "G2train_PCA64"
        ):

            report.append(
                f"{row['method']} "
                f"R@1={row['R1']:.5f} "
                f"R@20={row['R20']:.5f}"
            )

    for row in k_required:

        if (
            row["target"] == 0.99
            and row["method"].startswith(
                "G2train_PCA64"
            )
        ):

            report.append(
                f"{row['method']} "
                f"K99={row['K']}"
            )

    report.extend([
        "",
        "INTERPRETATION:",
        "D3 benefit requires a reproducible paired gain",
        "over D0 on this external matched-species subset.",
        "Five model seeds are not independent taxonomy splits.",
        "Bootstrap confidence intervals are exploratory.",
        "Do not pool G2-trained results with G1-trained E11.",
        "Smaller K does not imply faster end-to-end runtime.",
        "Near-duplicate screening is outside W15 scope."
    ])

    (
        OUT / "W15_DECISION_REPORT.txt"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    provenance = {
        "status": "PASS",
        "analysis": "POST_HOC_EXPLORATORY",
        "E11_B1_total": 391,
        "eligible_queries": n,
        "excluded_queries": 391 - n,
        "G2_training_references": len(references),
        "species_clusters": len(set(species_labels)),
        "G2_split_file": str(split_path),
        "G2_split_column": split_column,
        "G2_split_sha256": sha256(split_path),
        "HGV_modes": list(MODES),
        "HGV_seeds": list(SEEDS),
        "checkpoint_count": 15,
        "HGV_retraining": False,
        "PCA_fit": "G2_TRAIN_ONLY",
        "PCA_components": 64,
        "PCA_solver": "randomized",
        "PCA_seed": 20260915,
        "PCA_iterated_power": 5,
        "bootstrap_replicates": BOOTSTRAP,
        "bootstrap_unit": "species_cluster",
        "source_hashes": {
            "W13_script": sha256(W13),
            "E6_manifest": sha256(MANIFEST),
            "W14_checkpoint_audit": sha256(
                W14 /
                "W14_FROZEN_CHECKPOINTS_SHA256.tsv"
            ),
            "E11_results": sha256(
                E11 /
                "E11_QUERY_RETRIEVAL_RESULTS.tsv"
            ),
            "feature_matrix": sha256(
                mod.FEAT /
                "canonical_k6_features.float32.npy"
            ),
            "W15_script": sha256(
                Path(__file__)
            )
        },
        "previous_outputs_modified": False
    }

    (
        OUT / "W15_PROVENANCE.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    (
        OUT / "W15_STATUS.txt"
    ).write_text(
        "W15_STATUS=PASS\n"
        f"B1_ELIGIBLE={n}\n"
        f"B1_EXCLUDED={391-n}\n"
        "CHECKPOINT_SHA256=15/15\n"
        "HGV_RETRAINING=NO\n"
        "PCA_FIT=G2_TRAIN_ONLY\n",
        encoding="utf-8"
    )

    print("=" * 78)

    print(
        "W15_STATUS=PASS",
        "eligible=", n,
        "reference=1662",
        flush=True
    )

    for row in clustered:

        if (
            row["method_A"] == "D3"
            and row["method_B"] in ("D0", "D1")
            and row["K"] in (1, 20)
        ):

            print(
                "D3 -", row["method_B"],
                "K", row["K"],
                "delta",
                row["five_seed_mean_delta"],
                "CI",
                row["bootstrap_ci_low"],
                row["bootstrap_ci_high"],
                flush=True
            )

    print("OUTPUT:", OUT, flush=True)


if __name__ == "__main__":

    try:
        main()

    except Exception:

        if OUT.exists():

            (
                OUT / "W15_STATUS.txt"
            ).write_text(
                "W15_STATUS=INCOMPLETE\n",
                encoding="utf-8"
            )

            (
                OUT / "W15_ERROR.txt"
            ).write_text(
                traceback.format_exc(),
                encoding="utf-8"
            )

        raise
