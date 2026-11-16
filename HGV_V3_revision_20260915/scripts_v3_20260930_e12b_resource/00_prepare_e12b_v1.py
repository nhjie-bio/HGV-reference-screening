#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import sys
import json
import gzip
import shutil
import hashlib
import platform
import subprocess
from pathlib import Path

import pandas as pd

BASE = Path("HGV_V3_revision_20260915")
OUT = BASE / "16_scalability/E12B_synchronized_resource_v1_20260930"

P_PROTOCOL = OUT / "00_protocol"
P_MANIFEST = OUT / "01_manifests"
P_PREFLIGHT = OUT / "02_preflight"

for p in (P_PROTOCOL, P_MANIFEST, P_PREFLIGHT):
    p.mkdir(parents=True, exist_ok=True)

REPORT = []
FAIL = []
WARN = []

def log(x=""):
    print(x)
    REPORT.append(str(x))

def fail(x):
    FAIL.append(str(x))
    log("[FAIL] " + str(x))

def warn(x):
    WARN.append(str(x))
    log("[WARN] " + str(x))

def ok(x):
    log("[PASS] " + str(x))

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def norm_acc(x):
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.search(r"(GC[AF]_\d+\.\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(GC[AF]_\d+)", s)
    if m:
        return m.group(1)
    return s.strip()

def find_col(df, aliases, required=True):
    lc = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lc:
            return lc[a.lower()]
    for c in df.columns:
        cc = c.lower()
        for a in aliases:
            if a.lower() in cc:
                return c
    if required:
        raise RuntimeError(
            "Cannot resolve column from aliases: "
            + ", ".join(aliases)
            + "\nColumns = "
            + ", ".join(df.columns)
        )
    return None

def read_table(path):
    path = Path(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", dtype=str)
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except Exception:
        return pd.read_csv(path, dtype=str)

def shell(cmd):
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 999, repr(e)

log("=" * 88)
log("HGV-V3 E12B SYNCHRONIZED RESOURCE BENCHMARK - PREPARATION v1")
log("=" * 88)

# ---------------------------------------------------------------------
# 1. Frozen master manifest
# ---------------------------------------------------------------------

MASTER = (
    BASE
    / "01_manifest/final_3154_freeze_v1_20260916/"
      "final_master_manifest_3154.tsv"
)

if not MASTER.exists():
    fail(f"Master manifest missing: {MASTER}")
    sys.exit(2)

ok(f"master manifest found: {MASTER}")
log(f"master sha256 = {sha256(MASTER)}")

df = read_table(MASTER)

acc_col = find_col(
    df,
    ["assembly_accession", "accession", "assembly"]
)
cohort_col = find_col(
    df,
    ["source_cohort", "cohort", "dataset", "set"]
)
fasta_col = find_col(
    df,
    ["fasta_path", "genome_path", "assembly_path", "path"]
)

species_col = find_col(
    df,
    [
        "gtdb_species_cluster",
        "species_cluster",
        "gtdb_species",
        "species"
    ]
)

df["_acc"] = df[acc_col].map(norm_acc)
df["_cohort"] = df[cohort_col].fillna("").astype(str).str.strip()
df["_fasta"] = df[fasta_col].astype(str)
df["_species"] = df[species_col].astype(str)

def is_A(x):
    s = str(x).strip().lower().replace("-", "_")
    return (
        s == "a"
        or s == "development"
        or s.startswith("a_")
        or "development" in s
    )

def is_B1(x):
    s = str(x).strip().lower().replace("-", "_")
    return (
        s == "b1"
        or s.startswith("b1_")
        or "matched" in s
    )

A = df[df["_cohort"].map(is_A)].copy()
B1 = df[df["_cohort"].map(is_B1)].copy()

if len(A) != 2374:
    fail(
        f"A reference count != 2374. observed={len(A)}. "
        f"cohort column={cohort_col}"
    )
else:
    ok("A references = 2374")

if len(B1) != 391:
    fail(
        f"B1 count != 391. observed={len(B1)}. "
        f"cohort column={cohort_col}"
    )
else:
    ok("B1 external genomes = 391")

A = A.sort_values("_acc").reset_index(drop=True)
B1 = B1.sort_values("_acc").reset_index(drop=True)

# Frozen E12B resource subset:
# accession-sorted first 50 B1 genomes.
Q50 = B1.iloc[:50].copy()

refs_tsv = P_MANIFEST / "E12B_REFERENCES_2374.tsv"
queries_tsv = P_MANIFEST / "E12B_QUERIES_50.tsv"

A[["_acc", "_species", "_fasta"]].rename(
    columns={
        "_acc": "assembly_accession",
        "_species": "gtdb_species",
        "_fasta": "fasta_path",
    }
).to_csv(refs_tsv, sep="\t", index=False)

Q50[["_acc", "_species", "_fasta"]].rename(
    columns={
        "_acc": "assembly_accession",
        "_species": "gtdb_species",
        "_fasta": "fasta_path",
    }
).to_csv(queries_tsv, sep="\t", index=False)

refs_paths = P_MANIFEST / "E12B_REFERENCES_2374.paths"
queries_paths = P_MANIFEST / "E12B_QUERIES_50.paths"

refs_paths.write_text(
    "\n".join(A["_fasta"].tolist()) + "\n"
)
queries_paths.write_text(
    "\n".join(Q50["_fasta"].tolist()) + "\n"
)

ok(f"reference manifest written: {refs_tsv}")
ok(f"query manifest written: {queries_tsv}")

# FASTA existence
missing_refs = [x for x in A["_fasta"] if not Path(x).is_file()]
missing_q = [x for x in Q50["_fasta"] if not Path(x).is_file()]

if missing_refs:
    fail(f"missing reference FASTA = {len(missing_refs)}")
    (P_PREFLIGHT / "MISSING_REFERENCE_FASTA.txt").write_text(
        "\n".join(missing_refs) + "\n"
    )
else:
    ok("2374/2374 reference FASTA present")

if missing_q:
    fail(f"missing query FASTA = {len(missing_q)}")
    (P_PREFLIGHT / "MISSING_QUERY_FASTA.txt").write_text(
        "\n".join(missing_q) + "\n"
    )
else:
    ok("50/50 query FASTA present")

# Query taxonomic composition
q_species_counts = (
    Q50.groupby("_species")
       .size()
       .sort_values(ascending=False)
       .rename("n_queries")
       .reset_index()
       .rename(columns={"_species": "gtdb_species"})
)

q_species_counts.to_csv(
    P_MANIFEST / "E12B_QUERY_SPECIES_COUNTS.tsv",
    sep="\t",
    index=False,
)

log(
    "E12B Q50 distinct species = "
    + str(Q50["_species"].nunique())
)
log(
    "E12B Q50 max queries in one species = "
    + str(int(q_species_counts["n_queries"].max()))
)

# ---------------------------------------------------------------------
# 2. Compare against historical E12 query files where possible
# ---------------------------------------------------------------------

scal_root = BASE / "16_scalability"
query_candidates = []

if scal_root.exists():
    for p in scal_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "quer" not in name:
            continue
        if p.stat().st_size > 20_000_000:
            continue
        if p.suffix.lower() not in {
            ".txt", ".tsv", ".csv", ".list", ".manifest"
        }:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        ids = set(re.findall(r"GC[AF]_\d+\.\d+", txt))
        if len(ids) == 50:
            query_candidates.append((p, ids))

q50set = set(Q50["_acc"])

matched_historical = []
mismatched_historical = []

for p, ids in query_candidates:
    if ids == q50set:
        matched_historical.append(str(p))
    else:
        mismatched_historical.append(str(p))

if matched_historical:
    ok(
        "historical E12 50-query set matches accession-sorted "
        "B1 first 50"
    )
    (P_PREFLIGHT / "MATCHED_HISTORICAL_E12_QUERY_FILES.txt").write_text(
        "\n".join(matched_historical) + "\n"
    )
else:
    warn(
        "No historical 50-query manifest was automatically matched. "
        "Q50 is frozen from the documented accession-sorted policy."
    )

if mismatched_historical:
    warn(
        f"Found {len(mismatched_historical)} other 50-accession "
        "query files with different sets."
    )
    (
        P_PREFLIGHT
        / "OTHER_50QUERY_FILES_DIFFERENT_SET.txt"
    ).write_text(
        "\n".join(mismatched_historical) + "\n"
    )

# ---------------------------------------------------------------------
# 3. Final E5 external query-level table
# ---------------------------------------------------------------------

expected_e5 = (
    BASE
    / "09_baselines/E5_external_unified_v1_20260920/"
      "06_external_final_freeze_v2_20260922/"
      "E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv"
)

if expected_e5.exists():
    e5_query_table = expected_e5
else:
    candidates = list(
        BASE.rglob("E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv")
    )
    e5_query_table = candidates[0] if len(candidates) == 1 else None

if e5_query_table is None:
    fail("Cannot uniquely resolve E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv")
else:
    ok(f"E5 final query table: {e5_query_table}")
    log(f"E5 final query table sha256 = {sha256(e5_query_table)}")

    e5 = read_table(e5_query_table)

    # Detect query accession column by overlap with the frozen Q50 set.
    best_col = None
    best_n = -1

    for c in e5.columns:
        try:
            vals = set(e5[c].map(norm_acc))
            n = len(vals & q50set)
            if n > best_n:
                best_n = n
                best_col = c
        except Exception:
            pass

    if best_col is None or best_n == 0:
        fail(
            "Could not identify query accession column in final E5 table."
        )
    else:
        ok(
            f"E5 query column resolved: {best_col}; "
            f"Q50 overlap={best_n}/50"
        )

        e5["_query_acc_norm"] = e5[best_col].map(norm_acc)

        e5q = e5[e5["_query_acc_norm"].isin(q50set)].copy()
        e5q.to_csv(
            P_MANIFEST / "E12B_E5_FROZEN_Q50_ROWS.tsv",
            sep="\t",
            index=False,
        )

        log(f"E5 Q50 rows retained = {len(e5q)}")

        if set(e5q["_query_acc_norm"]) != q50set:
            missing = sorted(q50set - set(e5q["_query_acc_norm"]))
            fail(
                "Not all 50 queries appear in final E5 query table. "
                f"Missing={len(missing)}"
            )
            (
                P_PREFLIGHT
                / "E5_Q50_MISSING_ACCESSIONS.txt"
            ).write_text("\n".join(missing) + "\n")
        else:
            ok("all 50 E12B queries found in final E5 query table")

# ---------------------------------------------------------------------
# 4. G1 train split
# ---------------------------------------------------------------------

split_root = BASE / "04_splits/g1_g4_v1_20260916"
split_candidates = []

if split_root.exists():
    for p in split_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".tsv", ".csv", ".txt"}:
            continue
        if "g1" not in p.name.lower():
            continue

        try:
            x = read_table(p)
        except Exception:
            continue

        try:
            ac = find_col(
                x,
                ["assembly_accession", "accession", "assembly"],
                required=False,
            )
            rc = find_col(
                x,
                ["role", "split_role", "partition"],
                required=False,
            )
            if ac is None or rc is None:
                continue

            train_ids = set(
                x.loc[
                    x[rc].astype(str).str.lower().eq("train"),
                    ac
                ].map(norm_acc)
            )

            if len(train_ids) == 1662:
                split_candidates.append((p, train_ids, x, ac, rc))
        except Exception:
            pass

if not split_candidates:
    fail("Could not resolve a 1662-genome G1 training split.")
else:
    unique_sets = {}
    for p, ids, x, ac, rc in split_candidates:
        key = hashlib.sha256(
            "\n".join(sorted(ids)).encode()
        ).hexdigest()
        unique_sets.setdefault(key, []).append(
            (p, ids, x, ac, rc)
        )

    if len(unique_sets) != 1:
        fail(
            f"Multiple non-identical G1 train sets found: "
            f"{len(unique_sets)}"
        )
        lines = []
        for key, vals in unique_sets.items():
            for z in vals:
                lines.append(f"{key}\t{z[0]}")
        (P_PREFLIGHT / "G1_SPLIT_CANDIDATES.tsv").write_text(
            "\n".join(lines) + "\n"
        )
    else:
        vals = list(unique_sets.values())[0]
        p, ids, x, ac, rc = vals[0]
        ok(f"G1 train split resolved: {p}")
        log(f"G1 train genomes = {len(ids)}")

        m = df[df["_acc"].isin(ids)].copy()

        if len(m) != 1662:
            fail(
                f"G1 train join to master returned {len(m)}, "
                "expected 1662."
            )
        else:
            m[["_acc", "_species", "_fasta"]].rename(
                columns={
                    "_acc": "assembly_accession",
                    "_species": "gtdb_species",
                    "_fasta": "fasta_path",
                }
            ).sort_values("assembly_accession").to_csv(
                P_MANIFEST / "E12B_G1_TRAIN_1662.tsv",
                sep="\t",
                index=False,
            )
            ok("G1 train 1662 manifest frozen")

# ---------------------------------------------------------------------
# 5. HGV checkpoint discovery
# ---------------------------------------------------------------------

ckpt_roots = [
    BASE / "07_hgv",
    BASE / "08_main_generalization",
    BASE / "09_external_validation",
    BASE / "09_baselines",
    BASE / "15_reference_screening",
    BASE / "16_scalability",
]

ckpts = []

for root in ckpt_roots:
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue

        s = str(p).lower()
        score = 0

        if "20260915" in s:
            score += 6
        if "seed20260915" in s:
            score += 4
        if "g1" in s:
            score += 8
        if "canonical" in s:
            score += 5
        if "external" in s:
            score += 5
        if "a_only" in s or "a-only" in s:
            score += 4
        if "final" in s:
            score += 2
        if "freeze" in s:
            score += 2

        if re.search(r"g[234]", s):
            score -= 15
        if "ablation" in s:
            score -= 15
        if "sensitivity" in s:
            score -= 15
        if "fragment" in s:
            score -= 15

        ckpts.append((score, p))

ckpts.sort(key=lambda z: (-z[0], str(z[1])))

ckpt_report = P_PREFLIGHT / "HGV_CHECKPOINT_CANDIDATES.tsv"

with ckpt_report.open("w") as f:
    f.write("score\tpath\n")
    for score, p in ckpts[:100]:
        f.write(f"{score}\t{p}\n")

hgv_ckpt = None

if not ckpts:
    fail("No HGV checkpoint candidates found.")
else:
    top_score = ckpts[0][0]
    top = [p for score, p in ckpts if score == top_score]

    log(f"HGV checkpoint top score = {top_score}")
    log(f"HGV checkpoint top-count = {len(top)}")

    # Only auto-freeze when there is one clearly dominant candidate.
    second_score = ckpts[1][0] if len(ckpts) > 1 else -999

    if len(top) == 1 and top_score >= 12 and top_score > second_score:
        hgv_ckpt = top[0]
        ok(f"HGV G1 candidate auto-resolved: {hgv_ckpt}")
    else:
        warn(
            "HGV checkpoint not auto-frozen because candidate "
            "selection is ambiguous. See HGV_CHECKPOINT_CANDIDATES.tsv."
        )

# ---------------------------------------------------------------------
# 6. PCA frozen artifact discovery
# ---------------------------------------------------------------------

pca_candidates = []

for root in [
    BASE / "08_main_generalization",
    BASE / "09_external_validation",
    BASE / "09_baselines",
    BASE / "15_reference_screening",
]:
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if "pca" not in n:
            continue
        if p.suffix.lower() in {
            ".joblib", ".pkl", ".pickle", ".npz"
        }:
            pca_candidates.append(p)

(P_PREFLIGHT / "PCA_ARTIFACT_CANDIDATES.txt").write_text(
    "\n".join(map(str, sorted(pca_candidates))) + "\n"
)

if pca_candidates:
    ok(f"PCA artifact candidates found = {len(pca_candidates)}")
else:
    warn(
        "No serialized PCA artifact found automatically. "
        "E12B can reconstruct the G1 PCA transform from the "
        "frozen 1662-genome G1 training set if necessary."
    )

# ---------------------------------------------------------------------
# 7. DNABERT-S frozen assets
# ---------------------------------------------------------------------

DNAROOT = (
    BASE
    / "09_baselines/E5_unified_baseline_v1_20260917/"
      "06_dnaberts"
)

strict_candidates = (
    list(DNAROOT.rglob("DNABERTS_STRICT32_SEGMENT_MANIFEST.tsv.gz"))
    if DNAROOT.exists()
    else []
)

l2_candidates = (
    list(DNAROOT.rglob("DNABERTS_GENOME_EMBEDDINGS_L2_FLOAT32.npy"))
    if DNAROOT.exists()
    else []
)

raw_candidates = (
    list(DNAROOT.rglob("DNABERTS_GENOME_EMBEDDINGS_RAW_FLOAT32.npy"))
    if DNAROOT.exists()
    else []
)

snapshot_candidates = []

hf_root = DNAROOT / "hf_cache/hub"
if hf_root.exists():
    snapshot_candidates = [
        p for p in hf_root.rglob("*")
        if p.is_dir()
        and p.parent.name == "snapshots"
        and (p / "config.json").exists()
    ]

def one_or_warn(name, xs):
    if len(xs) == 1:
        ok(f"{name}: {xs[0]}")
        return xs[0]
    if len(xs) == 0:
        fail(f"{name}: not found")
        return None
    warn(f"{name}: multiple candidates ({len(xs)})")
    return None

strict_manifest = one_or_warn(
    "DNABERT strict32 manifest",
    strict_candidates,
)
dnabert_l2 = one_or_warn(
    "DNABERT L2 embedding matrix",
    l2_candidates,
)
dnabert_raw = one_or_warn(
    "DNABERT raw embedding matrix",
    raw_candidates,
)
dnabert_snapshot = one_or_warn(
    "DNABERT local HF snapshot",
    snapshot_candidates,
)

# Conda env check
rc, out = shell("conda env list")
if "dnabert_s_e5" in out:
    ok("conda env dnabert_s_e5 found")
else:
    warn("conda env dnabert_s_e5 not visible in conda env list")

# ---------------------------------------------------------------------
# 8. Native tools
# ---------------------------------------------------------------------

tool_specs = {
    "mash": ["mash --version", "mash -h"],
    "dashing2": ["dashing2 --version", "dashing2 --help"],
    "skani": ["skani --version", "skani -h"],
    "fastANI": ["fastANI --version", "fastANI -h"],
}

tool_rows = []

for tool, cmds in tool_specs.items():
    exe = shutil.which(tool)

    if exe is None and tool == "fastANI":
        exe = shutil.which("fastani")

    if exe is None:
        fail(f"native tool missing from PATH: {tool}")
        tool_rows.append((tool, "", "MISSING"))
        continue

    version_text = ""
    for cmd in cmds:
        rc, text = shell(cmd)
        if text:
            version_text = text.splitlines()[0][:500]
            break

    ok(f"{tool}: {exe} :: {version_text}")
    tool_rows.append((tool, exe, version_text))

pd.DataFrame(
    tool_rows,
    columns=["tool", "executable", "version_or_help"],
).to_csv(
    P_PREFLIGHT / "NATIVE_TOOL_VERSIONS.tsv",
    sep="\t",
    index=False,
)

# Save Dashing2 help because canonical/cache flags must match installed 2.1.20.
if shutil.which("dashing2"):
    rc, d2help = shell("dashing2 sketch --help")
    (P_PREFLIGHT / "DASHING2_SKETCH_HELP.txt").write_text(
        d2help + "\n"
    )

    canon_lines = [
        x for x in d2help.splitlines()
        if "canon" in x.lower()
    ]
    (P_PREFLIGHT / "DASHING2_CANONICAL_HELP_LINES.txt").write_text(
        "\n".join(canon_lines) + "\n"
    )

# ---------------------------------------------------------------------
# 9. Existing E12 directories
# ---------------------------------------------------------------------

e12_dirs = []

if scal_root.exists():
    for p in scal_root.iterdir():
        if p.is_dir() and "e12" in p.name.lower():
            e12_dirs.append(p)

(P_PREFLIGHT / "EXISTING_E12_DIRECTORIES.txt").write_text(
    "\n".join(map(str, sorted(e12_dirs))) + "\n"
)

if e12_dirs:
    ok(f"existing E12 directories recorded = {len(e12_dirs)}")
else:
    warn("No existing E12 directory found under 16_scalability")

# ---------------------------------------------------------------------
# 10. Hardware freeze
# ---------------------------------------------------------------------

hardware = []

hardware.append(f"platform\t{platform.platform()}")
hardware.append(f"python\t{sys.version.replace(os.linesep, ' ')}")

for cmd_name, cmd in [
    ("lscpu", "lscpu"),
    ("memory", "free -h"),
    ("gpu", "nvidia-smi --query-gpu=name,memory.total,driver_version "
            "--format=csv,noheader"),
    ("filesystem", f"df -h {BASE}"),
    ("uname", "uname -a"),
]:
    rc, text = shell(cmd)
    hardware.append(f"\n## {cmd_name}\n{text}")

(P_PROTOCOL / "HARDWARE_FREEZE.txt").write_text(
    "\n".join(hardware) + "\n"
)

# ---------------------------------------------------------------------
# 11. Protocol freeze
# ---------------------------------------------------------------------

protocol = f"""HGV-V3 E12B synchronized resource benchmark protocol
Date: 2026-09-30

PURPOSE
-------
Reviewer #1 requested a comparison including accuracy, runtime and memory.
E12B adds synchronized resource measurements without retraining HGV or
reopening G2/G3/G4, E6, E7, E8, E9, E10, E11 or the existing E12 Table 6.

FROZEN WORKLOAD
---------------
References:
  2,374 Development-A genomes

Queries:
  fixed resource subset of 50 B1 genomes
  selection rule = accession-sorted first 50 B1 accessions
  this subset is NOT a replacement for the full 391-query Table 3 result

REPEATS
-------
3 independent execution repeats per timed phase

CPU BUDGET
----------
Maximum CPU worker/thread budget = 48
Native tools may use fewer threads if that is their native behavior.
Do not describe all programs as using exactly 48 threads unless verified.

HGV MODEL
---------
E12B must use one canonical A-only / G1 HGV checkpoint,
model seed 20260915.

The historical E12 G2 checkpoint used for Table 6 is NOT to be paired
with B1 accuracy in the synchronized Table 7.

DEPLOYMENT PHASES
-----------------
1. Reference build/preprocessing
2. Warm query:
   precomputed reference representation exists;
   query FASTA preprocessing remains included
3. Complete first-use workflow
4. Reference index / representation size
5. Peak host memory
6. GPU memory where applicable

ACCURACY
--------
Main synchronized endpoint:
  B1 same-species Recall@1 on the frozen 50-query resource subset.

This accuracy is supplementary to, not a replacement for,
the full 391-query B1 result.

FASTANI / SKANI
---------------
No-hit comparisons remain missing.
Report separately:
  pair coverage
  query hit coverage
  top-1 correctness among hit queries
  all-50 top-1 success with no-hit query treated as retrieval failure

MEMORY
------
Host memory and GPU memory are separate quantities.
Do not compare PyTorch GPU allocated/reserved directly with host RSS.

Preferred synchronized host-memory measure:
  process-group PSS or cgroup-v2 group peak if available.

PROHIBITED
----------
- no runtime extrapolation
- no simulated values in the manuscript resource table
- no test/external retuning
- no replacement of original E12/Table 6
- no overwriting any frozen result
"""

(P_PROTOCOL / "E12B_PROTOCOL_FREEZE.txt").write_text(protocol)

# ---------------------------------------------------------------------
# 12. Resolved environment file
# ---------------------------------------------------------------------

env_lines = []

env_lines.append(f"export E12B_BASE='{BASE}'")
env_lines.append(f"export E12B_OUT='{OUT}'")
env_lines.append(f"export E12B_REFS_TSV='{refs_tsv}'")
env_lines.append(f"export E12B_QUERIES_TSV='{queries_tsv}'")
env_lines.append(f"export E12B_REFS_PATHS='{refs_paths}'")
env_lines.append(f"export E12B_QUERIES_PATHS='{queries_paths}'")

if e5_query_table:
    env_lines.append(
        f"export E12B_E5_QUERY_TABLE='{e5_query_table}'"
    )

if hgv_ckpt:
    env_lines.append(
        f"export E12B_HGV_G1_SEED15_CKPT='{hgv_ckpt}'"
    )
else:
    env_lines.append(
        "export E12B_HGV_G1_SEED15_CKPT=''"
    )

if strict_manifest:
    env_lines.append(
        f"export E12B_DNABERT_STRICT32='{strict_manifest}'"
    )
if dnabert_l2:
    env_lines.append(
        f"export E12B_DNABERT_L2='{dnabert_l2}'"
    )
if dnabert_raw:
    env_lines.append(
        f"export E12B_DNABERT_RAW='{dnabert_raw}'"
    )
if dnabert_snapshot:
    env_lines.append(
        f"export E12B_DNABERT_SNAPSHOT='{dnabert_snapshot}'"
    )

(P_PREFLIGHT / "E12B_RESOLVED_ENV.sh").write_text(
    "\n".join(env_lines) + "\n"
)

# ---------------------------------------------------------------------
# 13. SHA freeze
# ---------------------------------------------------------------------

freeze_files = [
    MASTER,
    refs_tsv,
    queries_tsv,
    refs_paths,
    queries_paths,
    P_PROTOCOL / "E12B_PROTOCOL_FREEZE.txt",
]

if e5_query_table:
    freeze_files.append(e5_query_table)

if hgv_ckpt:
    freeze_files.append(hgv_ckpt)

if strict_manifest:
    freeze_files.append(strict_manifest)

sha_lines = []

for p in freeze_files:
    if p and Path(p).is_file():
        sha_lines.append(f"{sha256(Path(p))}  {p}")

(P_PROTOCOL / "SHA256_FREEZE.txt").write_text(
    "\n".join(sha_lines) + "\n"
)

# ---------------------------------------------------------------------
# 14. Final gate
# ---------------------------------------------------------------------

log("")
log("=" * 88)
log("FINAL PREFLIGHT GATE")
log("=" * 88)

if hgv_ckpt is None:
    fail(
        "Canonical G1 seed20260915 HGV checkpoint still needs exact "
        "resolution before benchmark execution."
    )

if FAIL:
    log(f"E12B PREPARATION STATUS = NEEDS_RESOLUTION ({len(FAIL)} issues)")
    for x in FAIL:
        log(" - " + x)
else:
    ok("all mandatory E12B preparation gates passed")
    log("E12B PREPARATION STATUS = PASS")

if WARN:
    log("")
    log("WARNINGS")
    for x in WARN:
        log(" - " + x)

report_path = P_PREFLIGHT / "E12B_PREFLIGHT_REPORT.txt"
report_path.write_text("\n".join(REPORT) + "\n")

summary = {
    "status": "PASS" if not FAIL else "NEEDS_RESOLUTION",
    "n_fail": len(FAIL),
    "n_warn": len(WARN),
    "reference_genomes": int(len(A)),
    "b1_genomes": int(len(B1)),
    "resource_queries": int(len(Q50)),
    "resource_query_species": int(Q50["_species"].nunique()),
    "hgv_checkpoint": str(hgv_ckpt) if hgv_ckpt else None,
    "master_manifest": str(MASTER),
    "e5_query_table": str(e5_query_table) if e5_query_table else None,
}

(P_PREFLIGHT / "E12B_PREFLIGHT_SUMMARY.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

print("")
print("REPORT =", report_path)
print("ENV    =", P_PREFLIGHT / "E12B_RESOLVED_ENV.sh")
print("SHA    =", P_PROTOCOL / "SHA256_FREEZE.txt")
