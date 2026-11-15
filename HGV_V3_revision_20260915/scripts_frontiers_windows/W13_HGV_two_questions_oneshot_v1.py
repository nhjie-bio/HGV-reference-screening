#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "21_frontiers_revision_20261112/90_windows_minimal"

OUT = WORK / "W13_TWO_QUESTIONS_20261114_v1"

E11 = ROOT / "15_reference_screening/E11_reference_screening_v1_20260925_v6"
W10 = WORK / "W10_E13_MAG_FROZEN_RETRIEVAL"
E6 = ROOT / "10_ablation/E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
SPLITS = ROOT / "04_splits/g1_g4_v1_20260916"

MASTER = ROOT / "01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv"
FEAT = ROOT / "03_features/canonical_k6_full3154_v1"
PCA = ROOT / "09_external_validation/canonical_validation_threshold_freeze_v1_20260917/PCA64_G1_TRAIN_FROZEN.npz"

SEEDS = (20260915, 20260916, 20260917, 20260918, 20260919)
E11_SEEDS = SEEDS[:3]

KS = (1, 5, 10, 20, 50, 100)
TARGETS = (.95, .99, 1.0)
RNG_SEED = 20261114
BOOT = 1000


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    need(path.is_file(), "Missing file: " + str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        data = list(r)
    need(
        bool(r.fieldnames) and not any(None in x for x in data),
        "Invalid TSV: " + str(path)
    )
    return data


def write(name, data):
    need(bool(data), "Empty table: " + name)
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(data[0]), delimiter="\t",
            lineterminator="\n", extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(data)


def order(path, n):
    r = rows(path)
    need(len(r) == n, "Unexpected order length: " + str(path))

    options = (
        "reference_accession", "query_accession",
        "assembly_accession_final", "assembly_accession", "accession"
    )
    col = next((c for c in options if c in r[0]), None)

    if col is None:
        matches = [
            c for c in r[0]
            if all(
                re.fullmatch(
                    r"GC[AF]_\d+\.\d+",
                    str(x.get(c, "")).strip()
                )
                for x in r[:8]
            )
        ]
        need(len(matches) == 1, "Cannot identify accession column")
        col = matches[0]

    a = [x[col].strip() for x in r]
    need(len(set(a)) == n, "Repeated accession")
    return a


def edist(q, r):
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    d2 = (
        (q * q).sum(1)[:, None]
        + (r * r).sum(1)[None, :]
        - 2 * (q @ r.T)
    )
    return np.sqrt(np.maximum(0, d2))


def unit(a):
    return a / np.maximum(
        np.linalg.norm(a, axis=1, keepdims=True), 1e-12
    )


def getranks(D, qtax, rtax):
    need(D.shape == (len(qtax), len(rtax)), "Ranking dimension mismatch")
    rtax = np.asarray(rtax, dtype=object)

    result = np.full(len(qtax), len(rtax) + 1, dtype=np.int32)
    eligible = np.zeros(len(qtax), dtype=bool)

    for i, label in enumerate(qtax):
        if not label or label in ("NA", "None", "nan"):
            continue

        matches = rtax == label
        if not matches.any():
            continue

        eligible[i] = True
        ids = np.argsort(D[i], kind="mergesort")
        result[i] = int(np.flatnonzero(matches[ids])[0] + 1)

    return result, eligible


def curve(method, cohort, ranks, eligible, width_bytes, refn):
    n = len(ranks)
    e = int(eligible.sum())
    need(n > 0 and e > 0, "No eligible queries for " + cohort)

    rr = ranks[eligible]
    module = "E11" if refn == 2374 else "E6"

    row = dict(
        module=module, method=method, cohort=cohort,
        queries=n, eligible=e, reference_n=refn,
        vector_bytes=width_bytes
    )

    for k in KS:
        h = int((rr <= k).sum())
        row[f"hit_{k}"] = h
        row[f"R{k}_eligible"] = round(h / e, 8)
        row[f"R{k}_all"] = round(h / n, 8)

    row["MRR_eligible"] = round(float(np.mean(1.0 / rr)), 8)

    targets = []
    for t in TARGETS:
        s = np.sort(rr)
        pos = max(0, math.ceil(t * e) - 1)
        K = int(s[pos])

        targets.append(dict(
            module=module, method=method, cohort=cohort,
            target=t, K=K, eligible=e, reference_n=refn,
            keep_percent=round(100 * K / refn, 5),
            prune_percent=round(100 * (1 - K / refn), 5),
            vector_bytes=width_bytes
        ))

    return row, targets


def mcnemar(a, b):
    n = a + b
    if n == 0:
        return 1.0
    m = min(a, b)
    return min(
        1.0,
        2 * sum(math.comb(n, i) for i in range(m + 1)) / (2 ** n)
    )


def block_ci(a, b, clusters):
    groups = defaultdict(list)

    for i, g in enumerate(clusters):
        groups[str(g)].append(i)

    groups = list(groups.values())
    need(len(groups) >= 2, "Insufficient taxonomy blocks")

    values = np.asarray(a, float) - np.asarray(b, float)
    totals = np.asarray([values[g].sum() for g in groups])
    ns = np.asarray([len(g) for g in groups])

    rng = np.random.default_rng(RNG_SEED)
    results = np.empty(BOOT)

    for i in range(BOOT):
        c = np.bincount(
            rng.integers(len(groups), size=len(groups)),
            minlength=len(groups)
        )
        results[i] = float(c @ totals / (c @ ns))

    ci = np.quantile(results, [.025, .975])
    return (float(ci[0]), float(ci[1])), len(groups)


def paired(a_name, b_name, cohort, ra, rb, eligible, clusters, refn):
    output = []

    for k in (1, 5, 20):
        a = ra[eligible] <= k
        b = rb[eligible] <= k
        cl = np.asarray(clusters, dtype=object)[eligible]

        ci, ng = block_ci(a, b, cl)

        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())

        output.append(dict(
            module="E11" if refn == 2374 else "E6",
            cohort=cohort, method_A=a_name, method_B=b_name,
            K=k, n=int(eligible.sum()),
            A_correct=int(a.sum()), B_correct=int(b.sum()),
            A_only=only_a, B_only=only_b,
            delta=round(float(a.mean() - b.mean()), 8),
            ci_low=round(ci[0], 8),
            ci_high=round(ci[1], 8),
            taxonomic_blocks=ng,
            mcnemar_p_unadjusted=round(mcnemar(only_a, only_b), 8)
        ))

    return output


def load_common():
    master = rows(MASTER)
    feature_rows = rows(FEAT / "canonical_k6_rows.tsv")

    need(
        len(master) == 3154 and len(feature_rows) == 3154,
        "Master/feature row mismatch"
    )

    mm = {r["assembly_accession_final"]: r for r in master}
    fi = {r["assembly_accession"]: i for i, r in enumerate(feature_rows)}

    need(len(mm) == len(fi) == 3154, "Duplicate original accessions")

    X = np.load(
        FEAT / "canonical_k6_features.float32.npy",
        mmap_mode="r", allow_pickle=False
    )
    need(X.shape == (3154, 2080), "Original feature shape mismatch")

    return mm, fi, X


# ==========================================================
# MODULE A: E11 - SAME 64D / 256 BYTES
# ==========================================================

def module_e11(mm, fi, X):
    print("[E11] Equal-budget HGV vs PCA64", flush=True)

    status = (W10 / "W10_STATUS.txt").read_text(encoding="utf-8")
    need("W10_FROZEN_INFERENCE=PASS" in status, "W10 not passed")

    ref = order(E11 / "E11_REFERENCE_ORDER.tsv", 2374)
    qry = order(E11 / "E11_QUERY_ORDER.tsv", 780)

    official = {
        x["query_accession"]: x
        for x in rows(E11 / "E11_QUERY_RETRIEVAL_RESULTS.tsv")
    }

    need(len(official) == 780 and set(qry) == set(official), "E11 mismatch")
    need(all(a in fi and a in mm for a in ref + qry), "E11 missing accession")

    with np.load(PCA, allow_pickle=False) as p:
        mu = np.asarray(p["mean"], np.float32)
        comp = np.asarray(p["components"], np.float32)

    need(
        mu.shape == (2080,) and comp.shape == (64, 2080),
        "Frozen PCA dimension mismatch"
    )

    Xr = np.asarray(X[[fi[a] for a in ref]], np.float32)
    Xq = np.asarray(X[[fi[a] for a in qry]], np.float32)

    Pr = ((Xr - mu) @ comp.T).astype(np.float32)
    Pq = ((Xq - mu) @ comp.T).astype(np.float32)

    rz = np.load(
        E11 / "E11_REFERENCE_EMBEDDINGS_3SEED.npy",
        mmap_mode="r", allow_pickle=False
    )
    qz = np.load(
        E11 / "E11_QUERY_EMBEDDINGS_3SEED.npy",
        mmap_mode="r", allow_pickle=False
    )

    need(
        rz.shape == (3, 2374, 64) and qz.shape == (3, 780, 64),
        "Frozen HGV shape mismatch"
    )

    D = {
        f"HGV_{s}": edist(qz[i], rz[i])
        for i, s in enumerate(E11_SEEDS)
    }

    D["HGV_ensemble_768B"] = np.mean(
        np.stack([D[f"HGV_{s}"] for s in E11_SEEDS]),
        axis=0
    )

    D["PCA64_Euclidean"] = edist(Pq, Pr)
    D["PCA64_L2"] = edist(unit(Pq), unit(Pr))

    tax_fields = (
        "gtdb_species_cluster", "gtdb_genus", "gtdb_family"
    )

    reference_tax = {
        t: np.asarray([mm[a][t] for a in ref], dtype=object)
        for t in tax_fields
    }

    groups = defaultdict(list)
    for i, a in enumerate(qry):
        groups[official[a]["external_type"]].append(i)

    need(
        {k: len(groups[k]) for k in (
            "B1", "B2_seen_genus", "B2_unseen_genus"
        )} == {
            "B1": 391, "B2_seen_genus": 196, "B2_unseen_genus": 193
        },
        "E11 cohort sizes changed"
    )

    ranked = {}
    curves = []
    required = []
    detail = []

    for method, distance in D.items():
        idx_by_tax = defaultdict(list)

        for i, a in enumerate(qry):
            idx_by_tax[official[a]["target_level"]].append(i)

        full = np.full(780, 2375, dtype=np.int32)
        eligible = np.zeros(780, dtype=bool)

        for tax, indices in idx_by_tax.items():
            need(tax in reference_tax, "Unknown taxonomy: " + tax)

            target = [
                official[qry[i]]["target_taxon"]
                for i in indices
            ]

            r, e = getranks(
                distance[indices],
                target,
                reference_tax[tax]
            )

            full[indices] = r
            eligible[indices] = e

        ranked[method] = (full, eligible)

        if method == "HGV_ensemble_768B":
            for i, a in enumerate(qry):
                for k in (1, 5, 20):
                    expected = int(float(official[a][f"Recall@{k}"]))
                    observed = int(full[i] <= k)
                    need(
                        expected == observed,
                        "E11 frozen replay mismatch: " + a
                    )

        for cohort, idx in groups.items():
            ix = np.asarray(idx, dtype=int)

            z, v = curve(
                method, cohort, full[ix], eligible[ix],
                768 if method == "HGV_ensemble_768B" else 256,
                2374
            )

            curves.append(z)
            required.extend(v)

            for i in ix:
                rec = dict(
                    method=method,
                    query=qry[i],
                    cohort=cohort,
                    target_taxon=official[qry[i]]["target_taxon"],
                    eligible=int(eligible[i]),
                    rank=int(full[i]) if eligible[i] else "",
                    width_bytes=768 if method == "HGV_ensemble_768B" else 256
                )
                for k in KS:
                    rec[f"hit_{k}"] = int(full[i] <= k)
                detail.append(rec)

    pairs = []

    for cohort, ix in groups.items():
        ix = np.asarray(ix, int)

        clusters = [
            official[qry[i]]["target_taxon"]
            for i in ix
        ]

        hgv_methods = [
            f"HGV_{s}" for s in E11_SEEDS
        ] + ["HGV_ensemble_768B"]

        for hm in hgv_methods:
            for pm in ("PCA64_L2", "PCA64_Euclidean"):
                hr, he = ranked[hm]
                pr, pe = ranked[pm]

                need(
                    np.array_equal(he, pe),
                    "Eligibility differs between methods"
                )

                pairs.extend(
                    paired(
                        hm, pm, cohort,
                        hr[ix], pr[ix], he[ix], clusters, 2374
                    )
                )

    b1 = next(
        r for r in curves
        if r["method"] == "HGV_ensemble_768B"
        and r["cohort"] == "B1"
    )

    need(b1["hit_1"] == 385, "Original E11 B1 Rank-1 mismatch")

    write("E11_RECALL_CURVES.tsv", curves)
    write("E11_REQUIRED_K.tsv", required)
    write("E11_PAIRED_DIFFERENCES.tsv", pairs)
    write("E11_PER_QUERY.tsv", detail)

    print("[E11 PASS] Recall, K95/K99/K100 and paired comparisons", flush=True)

    for r in required:
        if r["cohort"] == "B1" and r["target"] in (.99, 1.0):
            print(" ", r["method"], r["target"], "K=", r["K"])

    return dict(
        status="PASS",
        queries=780,
        references=2374,
        frozen_PCA_sha256=digest(PCA)
    )


# ==========================================================
# MODULE B: E6 - D0 / D1 / D3 SAME G2
# ==========================================================

def normrole(value):
    x = str(value or "").strip().lower().replace("-", "_")
    if x in ("train", "training", "tr"):
        return "train"
    if x in ("validation", "validate", "val", "dev", "valid"):
        return "val"
    if x in ("test", "testing", "te", "holdout", "heldout"):
        return "test"
    return None


def resolve_g2_split(mm):
    need(SPLITS.is_dir(), "Original G2 split directory missing")

    files = sorted(SPLITS.rglob("*.tsv"))

    named = [
        p for p in files
        if (
            "g2" in p.as_posix().lower()
            or "species" in p.name.lower()
        )
        and (
            "20260915" in p.as_posix()
            or "seed15" in p.name.lower()
        )
    ]

    candidates = []
    expected = {"train": 1662, "val": 356, "test": 356}

    for path in named or files:
        try:
            rr = rows(path)
        except Exception:
            continue

        if not rr:
            continue

        fields = rr[0].keys()

        acccol = next(
            (
                c for c in (
                    "assembly_accession_final",
                    "assembly_accession",
                    "accession",
                    "genome_accession"
                )
                if c in fields
            ),
            None
        )

        if not acccol:
            continue

        rolecols = [
            f for f in fields
            if f.lower() in (
                "role", "set", "subset", "split_role",
                "split", "partition", "dataset_split", "split_label"
            )
            or ("g2" in f.lower() and "20260915" in f)
        ]

        for col in rolecols:
            mapping = {}

            for r in rr:
                acc = str(r.get(acccol, "")).strip()

                if acc not in mm:
                    continue
                if mm[acc].get("final_cohort", "").strip() != "A":
                    continue

                if "protocol" in fields and "G2" not in r["protocol"].upper():
                    continue

                if (
                    "taxonomy_split_seed" in fields
                    and r["taxonomy_split_seed"].strip() != "20260915"
                ):
                    continue

                if (
                    "split_seed" in fields
                    and r["split_seed"].strip() != "20260915"
                ):
                    continue

                role = normrole(r.get(col))

                if role:
                    if acc in mapping and mapping[acc] != role:
                        raise RuntimeError("Conflicting split roles")
                    mapping[acc] = role

            if Counter(mapping.values()) == expected and len(mapping) == 2374:
                candidates.append((path, col, mapping))

    need(
        candidates,
        "Cannot identify original G2 1662/356/356 split assignments"
    )

    unique = {tuple(sorted(x[2].items())) for x in candidates}
    need(len(unique) == 1, "Ambiguous G2 split assignments")

    return candidates[0]


def to_local(path_string):
    s = str(path_string).strip().strip('"').replace("\\", "/")

    marker = "HGV_V3_revision_20260915/"
    pos = s.find(marker)

    if pos >= 0:
        return ROOT / s[pos + len(marker):]

    return Path(s)


def get_checkpoint(mode, seed, manifest):
    direct = E6 / (
        f"03_training/{mode}/G2/split20260915/"
        f"model{seed}/best_model.pt"
    )

    candidates = []

    if direct.is_file():
        candidates.append(direct)

    manifest_rows = rows(manifest)

    matches = [
        r for r in manifest_rows
        if mode in [str(v).strip() for v in r.values()]
        and str(seed) in [str(v).strip() for v in r.values()]
    ]

    record = matches[0] if len(matches) == 1 else None

    if record:
        for key, value in record.items():
            value = str(value).strip()

            if value.endswith("best_model.pt"):
                p = to_local(value)
                if p.is_file():
                    candidates.append(p)

    # Original first three D3 seeds may be stored in
    # canonical main G2 training, rather than E6/03_training.
    if not candidates and mode == "D3":
        for base in (
            ROOT / "07_hgv",
            ROOT / "08_evaluation",
            ROOT / "10_ablation"
        ):
            if not base.is_dir():
                continue

            for p in base.rglob("best_model.pt"):
                text = p.as_posix().replace("\\", "/")

                if (
                    p.parent.name == f"model{seed}"
                    and "/G2/" in text
                    and "/split20260915/" in text
                    and not any(
                        f"/{x}/" in text
                        for x in ("D0", "D1", "D2", "D4", "D5")
                    )
                ):
                    candidates.append(p)

    candidates = list({
        str(p.resolve()): p for p in candidates
    }.values())

    if record:
        hashes = [
            str(v).strip().lower()
            for key, v in record.items()
            if (
                ("sha" in key.lower() or "hash" in key.lower())
                and ("checkpoint" in key.lower() or "model" in key.lower())
                and re.fullmatch(r"[0-9a-fA-F]{64}", str(v).strip())
            )
        ]

        if hashes:
            candidates = [
                p for p in candidates
                if digest(p).lower() in hashes
            ]

    need(
        len(candidates) == 1,
        f"Cannot uniquely identify {mode} seed={seed} checkpoint; "
        f"candidate_count={len(candidates)}"
    )

    return candidates[0]


def module_e6(mm, fi, X):
    print("[E6] G2 D0/D1/D3 five-seed retrieval", flush=True)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(min(8, os.cpu_count() or 1))

    class HGV(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2080, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            return F.normalize(self.fc3(x), p=2, dim=1)

    @torch.no_grad()
    def embed(model, matrix):
        model.eval()
        output = []

        for i in range(0, len(matrix), 256):
            x = torch.from_numpy(
                np.array(
                    matrix[i:i + 256],
                    dtype=np.float32,
                    copy=True
                )
            )
            output.append(model(x).cpu().numpy())

        return np.concatenate(output, axis=0).astype(np.float32)

    split_path, role_column, roles = resolve_g2_split(mm)

    ref = sorted(a for a, v in roles.items() if v == "train")
    qry = sorted(a for a, v in roles.items() if v == "test")

    need(
        len(ref) == 1662 and len(qry) == 356,
        "G2 train/test counts mismatch"
    )

    need(all(a in fi for a in ref + qry), "G2 feature missing")

    train_species = {
        mm[a]["gtdb_species_cluster"] for a in ref
    }
    test_species = {
        mm[a]["gtdb_species_cluster"] for a in qry
    }

    need(
        not train_species.intersection(test_species),
        "Species leakage in G2"
    )

    Xr = np.asarray(X[[fi[a] for a in ref]], np.float32)
    Xq = np.asarray(X[[fi[a] for a in qry]], np.float32)

    ref_tax = {
        tax: np.asarray(
            [mm[a]["gtdb_" + tax] for a in ref],
            dtype=object
        )
        for tax in ("genus", "family")
    }

    q_tax = {
        tax: [mm[a]["gtdb_" + tax] for a in qry]
        for tax in ("genus", "family")
    }

    eligible = {
        tax: np.asarray([
            x in set(ref_tax[tax])
            for x in q_tax[tax]
        ], dtype=bool)
        for tax in ("genus", "family")
    }

    need(
        all(int(e.sum()) >= 2 for e in eligible.values()),
        "Insufficient eligible G2 queries"
    )

    manifest = E6 / (
        "04_canonical_30cell_manifest_v1_20260923/"
        "E6_30CELL_CANONICAL_MANIFEST.tsv"
    )
    need(manifest.is_file(), "E6 canonical 30-cell manifest missing")

    ranked = {}
    curves = []
    required = []
    detail = []
    checkpoints = []

    for mode in ("D0", "D1", "D3"):
        for seed in SEEDS:
            ckpt = get_checkpoint(mode, seed, manifest)

            state = torch.load(
                ckpt, map_location="cpu", weights_only=True
            )

            need(
                isinstance(state, dict)
                and "model_state_dict" in state,
                "Invalid E6 checkpoint"
            )

            model = HGV().eval()
            model.load_state_dict(
                state["model_state_dict"], strict=True
            )

            zr = embed(model, Xr)
            zq = embed(model, Xq)

            need(
                zr.shape == (1662, 64)
                and zq.shape == (356, 64)
                and np.isfinite(zr).all()
                and np.isfinite(zq).all(),
                "E6 embedding error"
            )

            D = edist(zq, zr)

            checkpoints.append(dict(
                mode=mode, seed=seed,
                path=str(ckpt), sha256=digest(ckpt)
            ))

            for tax in ("genus", "family"):
                r, e = getranks(D, q_tax[tax], ref_tax[tax])

                need(
                    np.array_equal(e, eligible[tax]),
                    "E6 eligibility mismatch"
                )

                ranked[(mode, seed, tax)] = r

                z, v = curve(
                    f"{mode}_seed{seed}",
                    f"G2_{tax}",
                    r, e, 256, len(ref)
                )

                curves.append(z)
                required.extend(v)

                for i, acc in enumerate(qry):
                    detail.append(dict(
                        method=mode, seed=seed,
                        tax_level=tax, accession=acc,
                        eligible=int(e[i]),
                        rank=int(r[i]) if e[i] else "",
                        hit_1=int(r[i] <= 1),
                        hit_5=int(r[i] <= 5),
                        hit_20=int(r[i] <= 20)
                    ))

            print("[E6 PASS]", mode, seed, flush=True)
            del model, state

    comparisons = []
    averages = []

    for tax in ("genus", "family"):
        e = eligible[tax]

        for mode in ("D0", "D1", "D3"):
            seedr = np.stack([
                ranked[(mode, seed, tax)]
                for seed in SEEDS
            ])

            for k in (1, 5, 20):
                hits = (seedr[:, e] <= k).astype(float)

                averages.append(dict(
                    method=mode,
                    task=f"G2_{tax}",
                    K=k,
                    eligible=int(e.sum()),
                    mean_recall_5seeds=round(float(hits.mean()), 8),
                    seed_sd=round(float(hits.mean(1).std(ddof=1)), 8)
                ))

        for baseline in ("D0", "D1"):
            da = np.stack([
                ranked[("D3", s, tax)] for s in SEEDS
            ])
            db = np.stack([
                ranked[(baseline, s, tax)] for s in SEEDS
            ])

            for k in (1, 5, 20):
                a = (da[:, e] <= k).mean(0)
                b = (db[:, e] <= k).mean(0)

                ci, groups = block_ci(
                    a, b, np.asarray(q_tax[tax], dtype=object)[e]
                )

                seed_deltas = [
                    float(
                        (da[i, e] <= k).mean()
                        - (db[i, e] <= k).mean()
                    )
                    for i in range(5)
                ]

                comparisons.append(dict(
                    tax_level=tax,
                    contrast=f"D3_minus_{baseline}",
                    K=k,
                    eligible=int(e.sum()),
                    delta_mean_5seeds=round(
                        float(np.mean(seed_deltas)), 8
                    ),
                    cluster_boot_lo=round(ci[0], 8),
                    cluster_boot_hi=round(ci[1], 8),
                    n_taxon_blocks=groups,
                    seed_deltas=";".join(
                        f"{x:.6f}" for x in seed_deltas
                    ),
                    note="post_hoc_bootstrap_CI_not_confirmatory"
                ))

    # PCA fitted exclusively on the SAME G2 training references.
    # This is a new exploratory baseline, not the frozen G1 PCA.
    try:
        from sklearn.decomposition import PCA as SKPCA

        fit = SKPCA(
            n_components=64,
            svd_solver="randomized",
            random_state=20260915,
            iterated_power=5
        )

        fit.fit(Xr)

        Rp = fit.transform(Xr).astype(np.float32)
        Qp = fit.transform(Xq).astype(np.float32)

        pca_methods = (
            ("G2train_PCA64_Euclidean", edist(Qp, Rp)),
            ("G2train_PCA64_L2", edist(unit(Qp), unit(Rp)))
        )

        for method, D in pca_methods:
            for tax in ("genus", "family"):
                r, e = getranks(D, q_tax[tax], ref_tax[tax])
                z, v = curve(
                    method, f"G2_{tax}", r, e, 256, len(ref)
                )
                curves.append(z)
                required.extend(v)

        pca_status = "G2_train_only_fit_randomized_PCA"

    except ImportError:
        pca_status = "SKIPPED_sklearn_unavailable"

    write("E6_5SEED_RECALL.tsv", curves)
    write("E6_REQUIRED_K.tsv", required)
    write("E6_PAIRED_D3_CONTRASTS.tsv", comparisons)
    write("E6_MEAN_RECALL_5SEEDS.tsv", averages)
    write("E6_PER_QUERY.tsv", detail)
    write("E6_CHECKPOINTS_SHA256.tsv", checkpoints)

    print("[E6 PASS] 15 frozen HGV checkpoints evaluated", flush=True)

    for r in comparisons:
        if r["tax_level"] == "genus" and r["K"] in (1, 20):
            print(
                r["contrast"], "K", r["K"],
                "delta", r["delta_mean_5seeds"],
                "CI", r["cluster_boot_lo"], r["cluster_boot_hi"]
            )

    return dict(
        status="PASS",
        queries=356,
        references=1662,
        split_file=str(split_path),
        split_role_column=role_column,
        checkpoints=len(checkpoints),
        PCA_G2=pca_status,
        no_HGV_retraining=True
    )


# ==========================================================
# MODULE C: E13 REAL MAG SECONDARY ANALYSIS
# ==========================================================

def module_e13():
    print("[E13] 80 real MAGs, equal-budget comparison", flush=True)

    data = rows(W10 / "W10_E13_QUERY_RETRIEVAL.tsv")

    methods = [
        f"HGV_seed{s}" for s in E11_SEEDS
    ] + [
        "HGV_3seed_ensemble",
        "PCA64_Euclidean",
        "PCA64_L2"
    ]

    by = defaultdict(dict)

    for r in data:
        if r["method"] in methods:
            acc = r["accession"]
            method = r["method"]

            need(method not in by[acc], "Duplicate MAG result")
            by[acc][method] = r

    need(
        len(by) == 80
        and all(set(methods) == set(r) for r in by.values()),
        "MAG query/method completeness failure"
    )

    summary = []
    comparisons = []
    detail = []

    for level in ("species", "genus", "family"):
        candidates = [
            acc for acc, x in by.items()
            if x[methods[0]]["target_level"] == level
        ]

        expected = {"species": 40, "genus": 24, "family": 16}
        need(
            len(candidates) == expected[level],
            "MAG target count changed"
        )

        clusters = [
            by[a][methods[0]]["target_taxon"]
            for a in candidates
        ]

        for method in methods:
            ranks = np.array([
                int(by[a][method]["first_relevant_rank"])
                for a in candidates
            ])

            for k in (1, 5, 20):
                summary.append(dict(
                    method=method,
                    task=level,
                    K=k,
                    n=len(candidates),
                    hit=int((ranks <= k).sum()),
                    recall=round(float((ranks <= k).mean()), 8),
                    vector_bytes=768 if "ensemble" in method else 256,
                    nonredundancy_qc="PENDING"
                ))

            for acc, rank in zip(candidates, ranks):
                detail.append(dict(
                    method=method, accession=acc,
                    level=level, rank=int(rank)
                ))

        hgv_methods = [
            f"HGV_seed{s}" for s in E11_SEEDS
        ] + ["HGV_3seed_ensemble"]

        for hm in hgv_methods:
            for pm in ("PCA64_L2", "PCA64_Euclidean"):
                for k in (1, 5, 20):
                    a = np.array([
                        int(by[q][hm]["first_relevant_rank"]) <= k
                        for q in candidates
                    ], dtype=int)

                    b = np.array([
                        int(by[q][pm]["first_relevant_rank"]) <= k
                        for q in candidates
                    ], dtype=int)

                    only_a = int(((a == 1) & (b == 0)).sum())
                    only_b = int(((a == 0) & (b == 1)).sum())

                    rec = dict(
                        group=level,
                        model_A=hm,
                        model_B=pm,
                        K=k,
                        n=len(a),
                        A_only=only_a,
                        B_only=only_b,
                        delta=round(float(a.mean() - b.mean()), 8),
                        mcnemar_p_unadjusted=round(
                            mcnemar(only_a, only_b), 8
                        ),
                        nonredundancy_qc="PENDING"
                    )

                    try:
                        ci, groups = block_ci(a, b, clusters)
                        rec.update(
                            ci_low=round(ci[0], 8),
                            ci_high=round(ci[1], 8),
                            clusters=groups
                        )
                    except RuntimeError:
                        rec.update(
                            ci_low="", ci_high="",
                            clusters=len(set(clusters))
                        )

                    comparisons.append(rec)

    write("E13_MAG_RECALL.tsv", summary)
    write("E13_MAG_PAIRED.tsv", comparisons)
    write("E13_MAG_PER_QUERY.tsv", detail)

    print("[E13 PASS] Preliminary results; nonredundancy QC pending")

    return dict(
        status="PASS_PRELIMINARY",
        mag_queries=80,
        nonredundancy_qc="PENDING"
    )


# ==========================================================
# MAIN: RUN ALL MODULES; NEVER FALSE-PASS
# ==========================================================

def main():
    print("=" * 76)
    print("W13 | HGV TWO SCIENTIFIC QUESTIONS")
    print("=" * 76)

    need(not OUT.exists(), "Output exists; refusing overwrite: " + str(OUT))

    OUT.mkdir(parents=True, exist_ok=False)

    try:
        mm, fi, X = load_common()
    except Exception as exc:
        (OUT / "W13_STATUS.txt").write_text(
            "STATUS=FAILED\nCOMMON_INPUT=" + str(exc) + "\n",
            encoding="utf-8"
        )
        raise

    results = {}

    modules = (
        ("E11_EQUAL_BUDGET", lambda: module_e11(mm, fi, X)),
        ("E6_SELECTIVE_DISTILLATION", lambda: module_e6(mm, fi, X)),
        ("E13_REAL_MAG", module_e13)
    )

    for name, fn in modules:
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = dict(
                status="INCOMPLETE",
                error=str(exc),
                traceback=traceback.format_exc()
            )
            print("[STOP]", name, exc, flush=True)

    allpass = all(
        r["status"].startswith("PASS")
        for r in results.values()
    )

    provenance = dict(
        study="post_hoc_frozen_exploratory",
        no_HGV_training=True,
        no_change_prior_freezes=True,
        E11_PCA_fit="original_frozen_G1",
        E6_PCA_fit="G2_train_only_if_available",
        bootstrap_reps=BOOT,
        bootstrap_rng=RNG_SEED,
        modules=results,
        scope_note=(
            "E11 and E6 use different training and reference splits; "
            "their results must not be pooled."
        ),
        model_count_note=(
            "Three-seed HGV ensemble is 768 B; "
            "single-model HGV and PCA64 are 256 B."
        ),
        MAG_note="E13 requires nonredundancy and taxonomy QC.",
        script_sha256=digest(Path(__file__))
    )

    (OUT / "W13_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    status = (
        "PASS_ALL_MODULES"
        if allpass
        else "PARTIAL_NEEDS_INPUT_REVIEW"
    )

    lines = ["W13_STATUS=" + status] + [
        f'{k}={v["status"]}' for k, v in results.items()
    ]

    (OUT / "W13_STATUS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    report = [
        "HGV V3 W13 | SCIENTIFIC DECISION REPORT",
        "Post-hoc frozen exploratory analyses",
        "E11 and E6 have different training/reference splits.",
        "Single HGV and PCA64: 256 B; HGV ensemble: 768 B.",
        ""
    ]

    if results["E11_EQUAL_BUDGET"]["status"] == "PASS":
        rr = rows(OUT / "E11_REQUIRED_K.tsv")
        report.append("QUESTION B: E11 B1 minimum candidate counts")

        for method in [
            *[f"HGV_{s}" for s in E11_SEEDS],
            "PCA64_Euclidean",
            "PCA64_L2",
            "HGV_ensemble_768B"
        ]:
            records = [
                r for r in rr
                if r["method"] == method and r["cohort"] == "B1"
            ]
            report.append(
                method + " | " + ", ".join(
                    f"K{float(r['target'])*100:g}={r['K']}"
                    for r in records
                )
            )

        report.append(
            "Lower K at equal recall means fewer candidates, "
            "not necessarily lower end-to-end runtime."
        )
        report.append("")

    if results["E6_SELECTIVE_DISTILLATION"]["status"] == "PASS":
        rr = rows(OUT / "E6_PAIRED_D3_CONTRASTS.tsv")
        report.append("QUESTION A: E6 G2 D3 vs D0/D1")

        for r in rr:
            if r["tax_level"] == "genus" and int(r["K"]) in (1, 20):
                report.append(
                    f'{r["contrast"]} K{r["K"]}: '
                    f'delta={r["delta_mean_5seeds"]}, '
                    f'CI=[{r["cluster_boot_lo"]},'
                    f'{r["cluster_boot_hi"]}]'
                )

        report.append(
            "Five model seeds are not five independent taxonomy splits."
        )
        report.append(
            "These exploratory confidence intervals do not establish "
            "a prespecified confirmatory effect."
        )
        report.append("")

    if results["E13_REAL_MAG"]["status"].startswith("PASS"):
        report.append(
            "E13: 80 MAG results remain preliminary until "
            "near-duplicate and taxonomy QC."
        )

    for k, v in results.items():
        if not v["status"].startswith("PASS"):
            report.append(
                "INCOMPLETE " + k + ": " + v.get("error", "unknown")
            )

    report.append(
        "Do not infer universal superiority, ANI replacement, "
        "or end-to-end runtime advantage."
    )

    (OUT / "W13_DECISION_REPORT.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("\n".join(lines))
    print("OUTPUT:", OUT)

    if not allpass:
        print("See W13_PROVENANCE.json for exact incomplete-module reasons.")
        sys.exit(2)


if __name__ == "__main__":
    main()
