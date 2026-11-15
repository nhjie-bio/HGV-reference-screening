#!/usr/bin/env python3
"""HGV-V3 E13 real-MAG frozen inference, with E11 replay and checkpoint gates.

Requires numpy and torch.
Reads frozen artifacts and writes only a new W10 output directory.
"""

import csv
import hashlib
import json
import os
import re

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent

WORK = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal"
)

FEAT = ROOT / (
    "03_features/"
    "canonical_k6_full3154_v1"
)

MASTER = ROOT / (
    "01_manifest/"
    "final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

E11 = ROOT / (
    "15_reference_screening/"
    "E11_reference_screening_v1_20260925_v6"
)

TRAIN = ROOT / (
    "09_external_validation/"
    "canonical_A_model_training_v1_20260917/"
    "G1/split20260915"
)

PCA = ROOT / (
    "09_external_validation/"
    "canonical_validation_threshold_freeze_v1_20260917/"
    "PCA64_G1_TRAIN_FROZEN.npz"
)

W09 = WORK / "W09_E13_MAG_FEATURES"

OUT = WORK / "W10_E13_MAG_FROZEN_RETRIEVAL"

SEEDS = (
    20260915,
    20260916,
    20260917
)

BINS = (
    "50-70",
    "70-80",
    "80-90",
    "90-100"
)

LEVELS = (
    "species",
    "genus",
    "family"
)


def check(ok, why):

    if not ok:
        raise RuntimeError("[STOP] " + why)


def hash_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def load_tsv(path):

    check(
        path.is_file(),
        "Missing " + str(path)
    )

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        fields = list(reader.fieldnames or [])
        rows = list(reader)

    check(
        bool(fields)
        and not any(None in r for r in rows),
        "Bad TSV " + str(path)
    )

    return fields, rows


def save_tsv(path, rows, columns):

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(rows)


def order_accessions(path, expected):

    fields, rows = load_tsv(path)

    check(
        len(rows) == expected,
        "Wrong order length " + str(path)
    )

    options = (
        "reference_accession",
        "query_accession",
        "assembly_accession",
        "assembly_accession_final",
        "accession"
    )

    column = next(
        (x for x in options if x in fields),
        None
    )

    if column is None:

        possible = [
            c for c in fields
            if all(
                re.fullmatch(
                    r"GC[AF]_\d+\.\d+",
                    (r.get(c) or "").strip()
                )
                for r in rows[:min(10, len(rows))]
            )
        ]

        check(
            len(possible) == 1,
            "Cannot infer accession column: "
            + str(fields)
        )

        column = possible[0]

    accessions = [
        (r[column] or "").strip()
        for r in rows
    ]

    check(
        len(set(accessions)) == expected,
        "Repeated or empty accession"
    )

    return accessions


# ============================================================
# 1. EXACT ORIGINAL HGV ARCHITECTURE
# ============================================================

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
            self.bn1(
                self.fc1(x)
            )
        )

        x = F.relu(
            self.bn2(
                self.fc2(x)
            )
        )

        x = self.fc3(x)

        return F.normalize(
            x,
            p=2,
            dim=1
        )


@torch.no_grad()
def embed(model, matrix):

    model.eval()

    parts = []

    for i in range(0, len(matrix), 256):

        batch = torch.from_numpy(
            np.array(
                matrix[i:i + 256],
                dtype=np.float32,
                copy=True
            )
        )

        result = model(batch)

        parts.append(
            result.cpu().numpy().astype(
                np.float32,
                copy=False
            )
        )

    return np.concatenate(
        parts,
        axis=0
    )


def euclidean(query, reference):

    query = np.asarray(
        query,
        dtype=np.float64
    )

    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    d2 = (
        np.sum(query * query, axis=1)[:, None]
        +
        np.sum(reference * reference, axis=1)[None, :]
        -
        2.0 * (query @ reference.T)
    )

    return np.sqrt(
        np.maximum(d2, 0.0)
    )


def rank_of(order, reference_labels, target):

    matches = np.flatnonzero(
        reference_labels[order] == target
    )

    if len(matches) == 0:
        return None

    return int(matches[0] + 1)


def main():

    print("=" * 72)
    print("W10 - E13 REAL MAG HGV/PCA FROZEN INFERENCE")
    print("=" * 72)

    check(
        not OUT.exists(),
        "Output directory already exists: " + str(OUT)
    )

    required = [
        ROOT / (
            "scripts_frontiers_windows/"
            "W07_k6_windows_parity_v1.py"
        ),
        MASTER,
        PCA,
        FEAT / "canonical_k6_features.float32.npy",
        FEAT / "canonical_k6_rows.tsv",
        W09 / "E13_MAG_CANONICAL_K6_80x2080.float32.npy",
        W09 / "E13_MAG_FEATURE_ROWS.tsv",
        E11 / "E11_REFERENCE_EMBEDDINGS_3SEED.npy",
        E11 / "E11_QUERY_EMBEDDINGS_3SEED.npy",
        E11 / "E11_PROVENANCE.json",
        E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv",
        E11 / "E11_REFERENCE_ORDER.tsv",
        E11 / "E11_QUERY_ORDER.tsv"
    ]

    for path in required:

        check(
            path.is_file(),
            "Missing " + str(path)
        )

    # ========================================================
    # 2. VERIFY ORIGINAL FROZEN INPUTS
    # ========================================================

    w07_status = (
        WORK / "W07_feature_parity/W07_STATUS.txt"
    ).read_text(encoding="utf-8")

    check(
        "W07_FEATURE_PARITY=PASS" in w07_status,
        "W07 feature parity not passed"
    )

    w09_status = (
        W09 / "W09_STATUS.txt"
    ).read_text(encoding="utf-8")

    check(
        "W09_MAG_FEATURE_EXTRACTION=PASS" in w09_status,
        "W09 extraction not passed"
    )

    provenance = json.loads(
        (
            E11 / "E11_PROVENANCE.json"
        ).read_text(encoding="utf-8")
    )

    original_features = (
        FEAT / "canonical_k6_features.float32.npy"
    )

    check(
        hash_file(original_features)
        == provenance["feature_npy_sha256"],
        "Original feature matrix SHA256 mismatch"
    )

    check(
        hash_file(MASTER)
        == provenance["final_manifest_sha256"],
        "Original manifest SHA256 mismatch"
    )

    mag_config = json.loads(
        (
            W09 / "E13_MAG_FEATURE_CONFIG.json"
        ).read_text(encoding="utf-8")
    )

    mag_path = (
        W09 / "E13_MAG_CANONICAL_K6_80x2080.float32.npy"
    )

    check(
        hash_file(mag_path)
        == mag_config["feature_matrix_sha256"],
        "MAG feature matrix SHA256 mismatch"
    )

    _, feature_rows = load_tsv(
        FEAT / "canonical_k6_rows.tsv"
    )

    _, manifest = load_tsv(MASTER)

    _, mags = load_tsv(
        W09 / "E13_MAG_FEATURE_ROWS.tsv"
    )

    check(
        len(feature_rows) == 3154
        and len(manifest) == 3154
        and len(mags) == 80,
        "Unexpected manifest or MAG count"
    )

    feature_index = {
        r["assembly_accession"]: i
        for i, r in enumerate(feature_rows)
    }

    master_index = {
        r["assembly_accession_final"]: r
        for r in manifest
    }

    check(
        len(feature_index) == 3154
        and len(master_index) == 3154,
        "Duplicate original accession"
    )

    refs = order_accessions(
        E11 / "E11_REFERENCE_ORDER.tsv",
        2374
    )

    old_queries = order_accessions(
        E11 / "E11_QUERY_ORDER.tsv",
        780
    )

    check(
        not (set(refs) & set(old_queries)),
        "E11 reference/query accession overlap"
    )

    check(
        all(
            master_index[a]["final_cohort"] == "A"
            for a in refs
        ),
        "E11 reference order contains non-A genome"
    )

    check(
        set(refs)
        ==
        {
            accession
            for accession, r in master_index.items()
            if r["final_cohort"] == "A"
        },
        "E11 reference A set differs"
    )

    check(
        all(
            a in feature_index
            for a in refs + old_queries
        ),
        "Missing original feature accession"
    )

    Aidx = [
        feature_index[a]
        for a in refs
    ]

    Eidx = [
        feature_index[a]
        for a in old_queries
    ]

    X = np.load(
        original_features,
        mmap_mode="r",
        allow_pickle=False
    )

    XM = np.load(
        mag_path,
        allow_pickle=False
    )

    refz = np.load(
        E11 / "E11_REFERENCE_EMBEDDINGS_3SEED.npy",
        mmap_mode="r",
        allow_pickle=False
    )

    queryz = np.load(
        E11 / "E11_QUERY_EMBEDDINGS_3SEED.npy",
        mmap_mode="r",
        allow_pickle=False
    )

    check(
        X.shape == (3154, 2080)
        and XM.shape == (80, 2080),
        "Feature matrix shape mismatch"
    )

    check(
        refz.shape == (3, 2374, 64)
        and queryz.shape == (3, 780, 64),
        "E11 embedding shape mismatch"
    )

    for i, r in enumerate(mags):

        check(
            int(r["row_index"]) == i,
            "MAG row index mismatch"
        )

    print(
        "[PASS] Frozen manifests, hashes, "
        "and embedding orders",
        flush=True
    )

    # ========================================================
    # 3. HGV CHECKPOINT LOAD + EMBEDDING PARITY
    # ========================================================

    torch.set_num_threads(
        min(8, max(1, os.cpu_count() or 1))
    )

    magz = np.empty(
        (3, 80, 64),
        dtype=np.float32
    )

    parity_rows = []

    for si, seed in enumerate(SEEDS):

        checkpoint = (
            TRAIN / f"model{seed}/best_model.pt"
        )

        check(
            checkpoint.is_file(),
            "Missing checkpoint " + str(checkpoint)
        )

        digest = hash_file(checkpoint)

        expected_digest = provenance[
            "canonical_A_checkpoints"
        ][str(seed)]

        check(
            digest == expected_digest,
            "Checkpoint SHA256 mismatch: " + str(seed)
        )

        try:

            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True
            )

        except Exception as exc:

            raise RuntimeError(
                f"Safe checkpoint load failed: "
                f"{seed}: {exc}"
            ) from exc

        check(
            isinstance(state, dict)
            and "model_state_dict" in state,
            "Checkpoint missing model_state_dict"
        )

        model = HGV().eval()

        model.load_state_dict(
            state["model_state_dict"],
            strict=True
        )

        original_ref = embed(
            model,
            X[Aidx[:8]]
        )

        original_query = embed(
            model,
            X[Eidx[:8]]
        )

        error_ref = float(
            np.max(
                np.abs(
                    original_ref.astype(np.float64)
                    -
                    refz[si, :8].astype(np.float64)
                )
            )
        )

        error_query = float(
            np.max(
                np.abs(
                    original_query.astype(np.float64)
                    -
                    queryz[si, :8].astype(np.float64)
                )
            )
        )

        check(
            max(error_ref, error_query) <= 2e-5,
            (
                f"Frozen embedding mismatch "
                f"seed={seed}: "
                f"reference={error_ref}, "
                f"query={error_query}"
            )
        )

        magz[si] = embed(
            model,
            XM
        )

        norms = np.linalg.norm(
            magz[si],
            axis=1
        )

        check(
            np.isfinite(magz[si]).all()
            and np.max(np.abs(norms - 1)) < 1e-4,
            "Invalid HGV MAG embeddings"
        )

        parity_rows.append({
            "seed": seed,
            "checkpoint_sha256": digest,
            "reference_max_abs_error": error_ref,
            "external_max_abs_error": error_query
        })

        print(
            f"[PASS] HGV seed={seed}: "
            f"archived embedding error "
            f"A={error_ref:.3g}, E={error_query:.3g}; "
            f"80 MAG embedded",
            flush=True
        )

        del model
        del state

    # ========================================================
    # 4. REPLAY E11 RETRIEVAL
    # ========================================================

    old_seed_distances = {
        f"HGV_seed{SEEDS[si]}": euclidean(
            queryz[si],
            refz[si]
        )
        for si in range(3)
    }

    D_old = np.mean(
        np.stack(
            list(old_seed_distances.values())
        ),
        axis=0
    )

    _, official_rows = load_tsv(
        E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv"
    )

    official_by_acc = {
        r["query_accession"]: r
        for r in official_rows
    }

    check(
        len(official_by_acc) == 780,
        "E11 query keys not unique"
    )

    label_columns = (
        "gtdb_species_cluster",
        "gtdb_genus",
        "gtdb_family"
    )

    labels_old = {
        level: np.asarray(
            [
                master_index[a][level]
                for a in refs
            ],
            dtype=object
        )
        for level in label_columns
    }

    replay_errors = []

    for i, accession in enumerate(old_queries):

        check(
            accession in official_by_acc,
            "E11 query absent: " + accession
        )

        official = official_by_acc[accession]

        target_level = official["target_level"]

        check(
            target_level in labels_old,
            "Unknown E11 target level"
        )

        target_taxon = official["target_taxon"]

        order = np.argsort(
            D_old[i],
            kind="mergesort"
        )

        rank = rank_of(
            order,
            labels_old[target_level],
            target_taxon
        )

        for k in (1, 5, 20):

            observed = int(
                rank is not None and rank <= k
            )

            expected = int(
                float(official[f"Recall@{k}"])
            )

            if observed != expected:

                replay_errors.append({
                    "accession": accession,
                    "k": k,
                    "observed": observed,
                    "expected": expected
                })

    check(
        not replay_errors,
        "E11 ranking replay mismatch: "
        + str(replay_errors[:6])
    )

    print(
        "[PASS] E11 exact-ranking replay: "
        "780/780, Recall@1/5/20",
        flush=True
    )

    # ========================================================
    # 5. FROZEN PCA64
    # ========================================================

    with np.load(
        PCA,
        allow_pickle=False
    ) as p:

        pca_mean = p["mean"].astype(
            np.float32
        )

        components = p["components"].astype(
            np.float32
        )

    check(
        pca_mean.shape == (2080,)
        and components.shape == (64, 2080),
        "Frozen PCA shape mismatch"
    )

    X_A = np.asarray(
        X[Aidx],
        dtype=np.float32
    )

    X_E = np.asarray(
        X[Eidx],
        dtype=np.float32
    )

    P_A = (
        (X_A - pca_mean)
        @ components.T
    ).astype(np.float32)

    P_E = (
        (X_E - pca_mean)
        @ components.T
    ).astype(np.float32)

    P_M = (
        (XM - pca_mean)
        @ components.T
    ).astype(np.float32)

    def unit(matrix):

        return matrix / np.clip(
            np.linalg.norm(
                matrix,
                axis=1,
                keepdims=True
            ),
            1e-12,
            None
        )

    print(
        "[PASS] Frozen A-only PCA transform; "
        "no MAG fitting",
        flush=True
    )

    # ========================================================
    # 6. SAME-PROTOCOL E11 HGV VS PCA
    # ========================================================

    old_methods = dict(
        old_seed_distances
    )

    old_methods["HGV_3seed_ensemble"] = D_old

    old_methods["PCA64_Euclidean"] = euclidean(
        P_E,
        P_A
    )

    old_methods["PCA64_L2"] = euclidean(
        unit(P_E),
        unit(P_A)
    )

    e11_matched = []

    cohorts = (
        "B1",
        "B2_seen_genus",
        "B2_unseen_genus"
    )

    for method, distance_matrix in old_methods.items():

        for cohort in cohorts:

            cohort_indices = [
                i
                for i, accession in enumerate(old_queries)
                if official_by_acc[accession][
                    "external_type"
                ] == cohort
            ]

            hits = {
                1: 0,
                5: 0,
                20: 0
            }

            eligible = 0

            for i in cohort_indices:

                query = official_by_acc[
                    old_queries[i]
                ]

                target_level = query["target_level"]
                target_taxon = query["target_taxon"]

                order = np.argsort(
                    distance_matrix[i],
                    kind="mergesort"
                )

                rank = rank_of(
                    order,
                    labels_old[target_level],
                    target_taxon
                )

                if rank is not None:

                    eligible += 1

                    for k in hits:

                        hits[k] += int(
                            rank <= k
                        )

            row = {
                "method": method,
                "cohort": cohort,
                "n_queries": len(cohort_indices),
                "n_reference_target_present": eligible
            }

            for k in (1, 5, 20):

                row[f"hits_at_{k}"] = hits[k]

                row[f"Recall@{k}"] = (
                    f"{hits[k] / len(cohort_indices):.8f}"
                )

            e11_matched.append(row)

    print(
        "[PASS] Same-protocol E11 "
        "HGV-vs-PCA comparison computed",
        flush=True
    )

    # ========================================================
    # 7. MAG TARGET ELIGIBILITY
    # ========================================================

    mag_tax = {
        level: [
            r["gtdb_" + level]
            for r in mags
        ]
        for level in LEVELS
    }

    reference_tax = {
        level: np.asarray(
            [
                master_index[a]["gtdb_" + level]
                for a in refs
            ],
            dtype=object
        )
        for level in LEVELS
    }

    reference_tax_sets = {
        level: set(reference_tax[level].tolist())
        for level in LEVELS
    }

    for i, r in enumerate(mags):

        target_level = r[
            "reference_target_level"
        ]

        check(
            target_level in LEVELS
            and r["quality_bin"] in BINS,
            "Invalid MAG target or bin"
        )

        matches = {
            level: (
                mag_tax[level][i]
                in reference_tax_sets[level]
            )
            for level in LEVELS
        }

        check(
            matches[target_level],
            "MAG target not present in A: "
            + r["accession"]
        )

        if target_level == "genus":

            check(
                not matches["species"],
                "Genus target has seen species: "
                + r["accession"]
            )

        if target_level == "family":

            check(
                not matches["species"]
                and not matches["genus"],
                "Family target has seen species/genus: "
                + r["accession"]
            )

    original_numeric = {
        accession.split("_")[1]
        for accession in master_index
    }

    duplicate_numeric = [
        r["accession"]
        for r in mags
        if r["accession"].split("_")[1]
        in original_numeric
    ]

    check(
        not duplicate_numeric,
        "MAG matches original numeric assembly ID: "
        + str(duplicate_numeric)
    )

    print(
        "[PASS] MAG reference eligibility "
        "(80/80) and accession overlap screen",
        flush=True
    )

    # ========================================================
    # 8. REAL MAG RETRIEVAL
    # ========================================================

    distances = {}

    for i, seed in enumerate(SEEDS):

        distances[f"HGV_seed{seed}"] = euclidean(
            magz[i],
            refz[i]
        )

    distances["HGV_3seed_ensemble"] = np.mean(
        np.stack(
            [
                distances[f"HGV_seed{seed}"]
                for seed in SEEDS
            ]
        ),
        axis=0
    )

    distances["PCA64_Euclidean"] = euclidean(
        P_M,
        P_A
    )

    distances["PCA64_L2"] = euclidean(
        unit(P_M),
        unit(P_A)
    )

    query_rows = []

    for method, distance_matrix in distances.items():

        for i, mag in enumerate(mags):

            target_level = mag[
                "reference_target_level"
            ]

            target_taxon = mag[
                "gtdb_" + target_level
            ]

            order = np.argsort(
                distance_matrix[i],
                kind="mergesort"
            )

            rank = rank_of(
                order,
                reference_tax[target_level],
                target_taxon
            )

            check(
                rank is not None,
                "No eligible MAG reference target: "
                + mag["accession"]
            )

            top_index = int(
                order[0]
            )

            query_rows.append({
                "method": method,
                "accession": mag["accession"],
                "quality_bin": mag["quality_bin"],
                "target_level": target_level,
                "target_taxon": target_taxon,
                "completeness": mag["completeness"],
                "contamination": mag["contamination"],
                "first_relevant_rank": rank,
                "R1": int(rank <= 1),
                "R5": int(rank <= 5),
                "R20": int(rank <= 20),
                "top1_reference": refs[top_index],
                "top1_reference_target_taxon":
                    reference_tax[target_level][top_index],
                "top1_distance": (
                    f"{float(distance_matrix[i, top_index]):.10g}"
                )
            })

    # ========================================================
    # 9. E13 STRATIFIED SUMMARY
    # ========================================================

    summary = []

    for method in distances:

        for quality_bin in (
            *BINS,
            "ALL_BINS"
        ):

            for level in (
                *LEVELS,
                "ALL_LEVELS"
            ):

                selected = [
                    r for r in query_rows

                    if r["method"] == method

                    and (
                        quality_bin == "ALL_BINS"
                        or r["quality_bin"] == quality_bin
                    )

                    and (
                        level == "ALL_LEVELS"
                        or r["target_level"] == level
                    )
                ]

                n = len(selected)

                if n == 0:
                    continue

                row = {
                    "method": method,
                    "quality_bin": quality_bin,
                    "target_level": level,
                    "n": n
                }

                for k in (1, 5, 20):

                    hits = sum(
                        r[f"R{k}"]
                        for r in selected
                    )

                    row[f"hits_at_{k}"] = hits

                    row[f"Recall@{k}"] = (
                        f"{hits / n:.8f}"
                    )

                summary.append(row)

    # ========================================================
    # 10. SAVE RESULTS
    # ========================================================

    OUT.mkdir(
        parents=True,
        exist_ok=False
    )

    np.save(
        OUT / "W10_MAG_HGV_3SEED_80x64.npy",
        magz
    )

    np.save(
        OUT / "W10_MAG_PCA64_80x64.npy",
        P_M
    )

    save_tsv(
        OUT / "W10_EMBEDDING_PARITY.tsv",
        parity_rows,
        list(parity_rows[0])
    )

    save_tsv(
        OUT / "W10_E11_MATCHED_HGV_PCA.tsv",
        e11_matched,
        list(e11_matched[0])
    )

    save_tsv(
        OUT / "W10_E13_QUERY_RETRIEVAL.tsv",
        query_rows,
        list(query_rows[0])
    )

    save_tsv(
        OUT / "W10_E13_RETRIEVAL_SUMMARY.tsv",
        summary,
        list(summary[0])
    )

    final_provenance = {
        "status": "PASS",
        "seed_list": list(SEEDS),
        "mag_count": 80,
        "HGV_checkpoint_SHA256": {
            str(r["seed"]): r["checkpoint_sha256"]
            for r in parity_rows
        },
        "PCA_A_only_SHA256": hash_file(PCA),
        "mag_input_SHA256": hash_file(mag_path),
        "E11_rank_replay": "780/780 PASS",
        "checkpoint_retrained": False,
        "PCA_refitted": False,
        "MAG_near_duplicate_ANI_screen": "PENDING",
        "MAG_taxonomy_label_audit": "PENDING",
        "metric_denominator": (
            "Observed GTDB-target-eligible MAGs "
            "in each prespecified stratum"
        )
    }

    (
        OUT / "W10_PROVENANCE.json"
    ).write_text(
        json.dumps(
            final_provenance,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    (
        OUT / "W10_STATUS.txt"
    ).write_text(
        "W10_FROZEN_INFERENCE=PASS\n"
        "E11_780_QUERY_REPLAY=PASS\n"
        "MAG_COUNT=80\n"
        "MAG_NEAR_DUPLICATE_QC=PENDING\n",
        encoding="utf-8"
    )

    print("=" * 72)
    print("W10_FROZEN_INFERENCE=PASS")
    print("No model training performed")

    print()
    print("E11 same-protocol Recall@1:")

    for row in e11_matched:

        if row["method"] in (
            "HGV_3seed_ensemble",
            "PCA64_Euclidean",
            "PCA64_L2"
        ):

            print(
                row["cohort"],
                row["method"],
                "R1=" + row["Recall@1"],
                "n=" + str(row["n_queries"])
            )

    print()
    print("Real MAG retrieval:")

    for row in summary:

        if (
            row["quality_bin"] == "ALL_BINS"
            and row["target_level"] == "ALL_LEVELS"
        ):

            print(
                row["method"],
                "R1=" + row["Recall@1"],
                "R5=" + row["Recall@5"],
                "R20=" + row["Recall@20"],
                "n=" + str(row["n"])
            )

    print()
    print("Output:", OUT)


if __name__ == "__main__":
    main()
