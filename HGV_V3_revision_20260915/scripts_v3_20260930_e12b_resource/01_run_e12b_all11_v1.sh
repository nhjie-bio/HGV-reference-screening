#!/usr/bin/env bash
set -euo pipefail

BASE=HGV_V3_revision_20260915
SDIR="$BASE/scripts_v3_20260930_e12b_resource"
OUT="$BASE/16_scalability/E12B_synchronized_resource_v1_20260930"

ENVFILE="$OUT/02_preflight/E12B_RESOLVED_ENV.sh"

THREADS=48
REPEATS=3

RUNROOT="$OUT/03_runs"
SUMMARY="$OUT/04_summary"
FREEZE="$OUT/05_final_freeze"
LOGS="$OUT/logs"

mkdir -p "$RUNROOT" "$SUMMARY" "$FREEZE" "$LOGS"

if [[ ! -s "$ENVFILE" ]]; then
    echo "[FAIL] missing E12B resolved environment:"
    echo "  $ENVFILE"
    exit 2
fi

source "$ENVFILE"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "======================================================================"
echo "HGV-V3 E12B SYNCHRONIZED 11-METHOD RESOURCE BENCHMARK v1"
echo "======================================================================"
echo "BASE     = $BASE"
echo "OUT      = $OUT"
echo "THREADS  = $THREADS"
echo "REPEATS  = $REPEATS"
echo
echo "This run does NOT train HGV."
echo "This run does NOT modify E5/E6/E7/E8/E9/E10/E11/E12."
echo "======================================================================"

# ======================================================================
# 1. HARD GATES
# ======================================================================

for f in \
    "$E12B_REFS_TSV" \
    "$E12B_QUERIES_TSV" \
    "$E12B_REFS_PATHS" \
    "$E12B_QUERIES_PATHS" \
    "$E12B_E5_QUERY_TABLE" \
    "$E12B_HGV_G1_SEED15_CKPT" \
    "$E12B_PCA_ARTIFACT" \
    "$E12B_DNABERT_STRICT32" \
    "$E12B_DNABERT_SNAPSHOT"
do
    if [[ ! -s "$f" && ! -d "$f" ]]; then
        echo "[FAIL] required input missing:"
        echo "  $f"
        exit 2
    fi
done

for tool in mash dashing2 skani fastANI; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "[FAIL] $tool not found in PATH"
        exit 2
    }
done

REF_N=$(($(wc -l < "$E12B_REFS_TSV") - 1))
QUERY_N=$(($(wc -l < "$E12B_QUERIES_TSV") - 1))

[[ "$REF_N" -eq 2374 ]] || {
    echo "[FAIL] refs=$REF_N expected 2374"
    exit 2
}

[[ "$QUERY_N" -eq 50 ]] || {
    echo "[FAIL] queries=$QUERY_N expected 50"
    exit 2
}

echo "[PASS] frozen workload = 2374 references + 50 queries"

# ======================================================================
# 2. PROCESS-GROUP RESOURCE MONITOR
#    Peak host memory = summed PSS within process group.
#    GPU memory is kept separate.
# ======================================================================

cat > "$SDIR/e12b_monitor.py" <<'PYMON'
#!/usr/bin/env python3

import os
import sys
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path


def pids_in_pgrp(pgid):
    out = []
    proc = Path("/proc")

    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        try:
            if os.getpgid(pid) == pgid:
                out.append(pid)
        except Exception:
            pass

    return out


def pss_kib(pid):
    p = Path(f"/proc/{pid}/smaps_rollup")
    try:
        for line in p.read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except Exception:
        pass

    p = Path(f"/proc/{pid}/status")
    try:
        for line in p.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass

    return 0


def gpu_mib_for_pids(pids):
    pset = set(pids)
    try:
        x = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )

        total = 0.0

        for line in x.stdout.splitlines():
            z = [a.strip() for a in line.split(",")]
            if len(z) < 2:
                continue

            try:
                pid = int(z[0])
                mem = float(z[1])
            except Exception:
                continue

            if pid in pset:
                total += mem

        return total
    except Exception:
        return 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample-sec", type=float, default=0.20)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
args = ap.parse_args()

cmd = list(args.cmd)
if cmd and cmd[0] == "--":
    cmd = cmd[1:]

if not cmd:
    raise SystemExit("No command supplied.")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

stdout_path = out / "stdout.log"
stderr_path = out / "stderr.log"
json_path = out / "resource.json"

start = time.perf_counter()

with stdout_path.open("w") as fo, stderr_path.open("w") as fe:
    proc = subprocess.Popen(
        cmd,
        stdout=fo,
        stderr=fe,
        start_new_session=True,
    )

    pgid = os.getpgid(proc.pid)

    peak_pss = 0
    peak_gpu = 0.0
    samples = 0

    while True:
        pids = pids_in_pgrp(pgid)

        pss = sum(pss_kib(pid) for pid in pids)
        gpu = gpu_mib_for_pids(pids)

        peak_pss = max(peak_pss, pss)
        peak_gpu = max(peak_gpu, gpu)
        samples += 1

        rc = proc.poll()

        if rc is not None:
            # one last sample
            pids = pids_in_pgrp(pgid)
            pss = sum(pss_kib(pid) for pid in pids)
            gpu = gpu_mib_for_pids(pids)

            peak_pss = max(peak_pss, pss)
            peak_gpu = max(peak_gpu, gpu)
            break

        time.sleep(args.sample_sec)

end = time.perf_counter()

data = {
    "name": args.name,
    "command": cmd,
    "exit_code": int(proc.returncode),
    "wall_s": float(end - start),
    "peak_host_pss_kib": int(peak_pss),
    "peak_host_pss_mib": float(peak_pss / 1024.0),
    "peak_gpu_mib": float(peak_gpu),
    "samples": int(samples),
    "sample_interval_s": float(args.sample_sec),
    "memory_definition":
        "sum of /proc/<pid>/smaps_rollup Pss within spawned process group",
}

json_path.write_text(json.dumps(data, indent=2) + "\n")

print(
    f"{args.name}\t"
    f"rc={proc.returncode}\t"
    f"wall={data['wall_s']:.6f}s\t"
    f"host_pss={data['peak_host_pss_mib']:.3f}MiB\t"
    f"gpu={data['peak_gpu_mib']:.3f}MiB"
)

if proc.returncode != 0:
    print("STDERR:")
    try:
        print(stderr_path.read_text()[-10000:])
    except Exception:
        pass

    raise SystemExit(proc.returncode)
PYMON

chmod +x "$SDIR/e12b_monitor.py"

# ======================================================================
# 3. RAW / PCA / HGV RUNNER
# ======================================================================

cat > "$SDIR/e12b_vector_methods.py" <<'PYVEC'
#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import gzip
import json
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd


K = 6
SPACE = 4 ** K
MASK = SPACE - 1

RC = np.zeros(SPACE, dtype=np.int16)

for code in range(SPACE):
    x = code
    r = 0

    for _ in range(K):
        b = x & 3
        r = (r << 2) | (3 - b)
        x >>= 2

    RC[code] = r

CAN_CODES = sorted(
    {min(i, int(RC[i])) for i in range(SPACE)}
)

assert len(CAN_CODES) == 2080

CODE2IDX = np.full(SPACE, -1, dtype=np.int16)

for idx, c in enumerate(CAN_CODES):
    CODE2IDX[c] = idx


BASEMAP = np.full(256, -1, dtype=np.int8)

for ch, v in [
    (ord("A"), 0),
    (ord("C"), 1),
    (ord("G"), 2),
    (ord("T"), 3),
    (ord("a"), 0),
    (ord("c"), 1),
    (ord("g"), 2),
    (ord("t"), 3),
]:
    BASEMAP[ch] = v


def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def one_feature(path):
    counts = np.zeros(2080, dtype=np.float64)
    valid_windows = 0

    code = 0
    valid_len = 0

    with open_text(path) as f:
        for line in f:
            if not line:
                continue

            if line.startswith(">"):
                code = 0
                valid_len = 0
                continue

            s = line.strip().encode("ascii", errors="ignore")

            for ch in s:
                b = int(BASEMAP[ch])

                if b < 0:
                    code = 0
                    valid_len = 0
                    continue

                code = ((code << 2) | b) & MASK
                valid_len += 1

                if valid_len >= K:
                    c = min(code, int(RC[code]))
                    idx = int(CODE2IDX[c])
                    counts[idx] += 1.0
                    valid_windows += 1

    if valid_windows == 0:
        raise RuntimeError(f"No valid 6-mer windows: {path}")

    counts /= float(valid_windows)

    return counts.astype(np.float32)


def load_manifest(path):
    df = pd.read_csv(path, sep="\t", dtype=str)

    required = {
        "assembly_accession",
        "gtdb_species",
        "fasta_path",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path}: missing columns {sorted(missing)}"
        )

    return df


def extract_manifest(df, workers):
    paths = df["fasta_path"].tolist()

    ctx = mp.get_context("fork")

    with ctx.Pool(
        processes=workers,
        maxtasksperchild=64,
    ) as pool:
        rows = pool.map(
            one_feature,
            paths,
            chunksize=2,
        )

    X = np.stack(rows).astype(np.float32)

    if X.shape != (len(df), 2080):
        raise RuntimeError(f"bad feature shape {X.shape}")

    return X


def load_pca(path):
    z = np.load(path, allow_pickle=True)

    comp = None
    mean = None

    for k in z.files:
        a = np.asarray(z[k])

        if a.ndim == 2 and 64 in a.shape and 2080 in a.shape:
            if a.shape == (64, 2080):
                comp = a.astype(np.float32)
            elif a.shape == (2080, 64):
                comp = a.T.astype(np.float32)

        if a.ndim == 1 and a.shape[0] == 2080:
            lk = k.lower()
            if "mean" in lk:
                mean = a.astype(np.float32)

    if comp is None:
        raise RuntimeError(
            "Could not resolve PCA components. "
            f"Keys={[(k, np.asarray(z[k]).shape) for k in z.files]}"
        )

    if mean is None:
        # fallback: only if exactly one 2080-vector exists
        candidates = [
            np.asarray(z[k]).astype(np.float32)
            for k in z.files
            if np.asarray(z[k]).ndim == 1
            and np.asarray(z[k]).shape[0] == 2080
        ]

        if len(candidates) != 1:
            raise RuntimeError("Could not uniquely resolve PCA mean.")

        mean = candidates[0]

    return mean, comp


def pca_transform(X, pca_path, l2=False):
    mean, comp = load_pca(pca_path)

    Z = (X.astype(np.float32) - mean[None, :]) @ comp.T

    Z = Z.astype(np.float32)

    if l2:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        Z = Z / n

    return Z.astype(np.float32)


def hgv_transform(X, ckpt_path, threads):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(int(threads))

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
            x = self.fc3(x)
            x = F.normalize(x, p=2, dim=1, eps=1e-12)
            return x

    obj = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, dict):
        raise RuntimeError("Unexpected HGV checkpoint type.")

    sd = obj.get("model_state_dict", obj)

    model = HGV()
    model.load_state_dict(sd, strict=True)
    model.eval()

    out = []

    with torch.inference_mode():
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(
                np.asarray(X[i:i+1024], dtype=np.float32)
            )

            z = model(xb)
            out.append(z.cpu().numpy())

    Z = np.concatenate(out, axis=0).astype(np.float32)

    norms = np.linalg.norm(Z, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(
            f"HGV output norm failure: "
            f"{norms.min()} {norms.max()}"
        )

    return Z


def transform(method, X, args):
    if method.startswith("Raw6_"):
        return X.astype(np.float32)

    if method == "PCA64_Euclidean":
        return pca_transform(
            X,
            args.pca,
            l2=False,
        )

    if method == "PCA64_L2":
        return pca_transform(
            X,
            args.pca,
            l2=True,
        )

    if method == "HGV":
        return hgv_transform(
            X,
            args.ckpt,
            args.workers,
        )

    raise RuntimeError(method)


def euclidean_top1(Q, R):
    q2 = np.sum(Q * Q, axis=1, keepdims=True)
    r2 = np.sum(R * R, axis=1)[None, :]

    d2 = q2 + r2 - 2.0 * (Q @ R.T)
    d2 = np.maximum(d2, 0.0)

    idx = np.argmin(d2, axis=1)
    score = np.sqrt(d2[np.arange(len(Q)), idx])

    return idx, score


def cosine_top1(Q, R):
    qn = Q / np.maximum(
        np.linalg.norm(Q, axis=1, keepdims=True),
        1e-12,
    )

    rn = R / np.maximum(
        np.linalg.norm(R, axis=1, keepdims=True),
        1e-12,
    )

    sim = qn @ rn.T

    idx = np.argmax(sim, axis=1)
    dist = 1.0 - sim[np.arange(len(Q)), idx]

    return idx, dist


def bray_top1(Q, R):
    idxs = []
    scores = []

    for q in Q:
        num = np.abs(R - q[None, :]).sum(axis=1)
        den = np.abs(R + q[None, :]).sum(axis=1)

        d = np.divide(
            num,
            den,
            out=np.ones_like(num),
            where=den > 0,
        )

        j = int(np.argmin(d))

        idxs.append(j)
        scores.append(float(d[j]))

    return np.asarray(idxs), np.asarray(scores)


def top1(method, Q, R):
    if method == "Raw6_BrayCurtis":
        return bray_top1(Q, R)

    if method == "Raw6_Cosine":
        return cosine_top1(Q, R)

    return euclidean_top1(Q, R)


def write_predictions(
    method,
    qdf,
    rdf,
    idx,
    score,
    out,
):
    rows = []

    for i, j in enumerate(idx):
        qs = qdf.iloc[i]
        rs = rdf.iloc[int(j)]

        rows.append(
            {
                "method": method,
                "query_accession": qs["assembly_accession"],
                "query_species": qs["gtdb_species"],
                "top1_reference": rs["assembly_accession"],
                "top1_species": rs["gtdb_species"],
                "top1_score_or_distance": float(score[i]),
                "correct_species": int(
                    qs["gtdb_species"] == rs["gtdb_species"]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        out,
        sep="\t",
        index=False,
    )


ap = argparse.ArgumentParser()

ap.add_argument("--method", required=True)
ap.add_argument(
    "--phase",
    required=True,
    choices=["build", "query", "full"],
)

ap.add_argument("--refs", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--ckpt")
ap.add_argument("--pca")
ap.add_argument("--index-in")

args = ap.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

rdf = load_manifest(args.refs)
qdf = load_manifest(args.queries)

if len(rdf) != 2374:
    raise RuntimeError(f"refs={len(rdf)}")

if len(qdf) != 50:
    raise RuntimeError(f"queries={len(qdf)}")

if args.phase == "build":
    Xr = extract_manifest(rdf, args.workers)
    R = transform(args.method, Xr, args)

    path = outdir / "reference_index.float32.npy"
    np.save(path, R.astype(np.float32))

    meta = {
        "method": args.method,
        "phase": args.phase,
        "shape": list(R.shape),
        "dtype": str(R.dtype),
        "payload_bytes": int(R.nbytes),
        "file_bytes": int(path.stat().st_size),
    }

    (outdir / "index_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print(json.dumps(meta, indent=2))

elif args.phase == "query":
    if not args.index_in:
        raise RuntimeError("--index-in required")

    R = np.load(args.index_in).astype(np.float32)

    Xq = extract_manifest(qdf, args.workers)
    Q = transform(args.method, Xq, args)

    idx, score = top1(args.method, Q, R)

    write_predictions(
        args.method,
        qdf,
        rdf,
        idx,
        score,
        outdir / "top1_predictions.tsv",
    )

    print(
        "query_r1=",
        float(
            pd.read_csv(
                outdir / "top1_predictions.tsv",
                sep="\t",
            )["correct_species"].mean()
        ),
    )

elif args.phase == "full":
    Xr = extract_manifest(rdf, args.workers)
    Xq = extract_manifest(qdf, args.workers)

    R = transform(args.method, Xr, args)
    Q = transform(args.method, Xq, args)

    idx, score = top1(args.method, Q, R)

    write_predictions(
        args.method,
        qdf,
        rdf,
        idx,
        score,
        outdir / "top1_predictions.tsv",
    )

    meta = {
        "method": args.method,
        "phase": args.phase,
        "reference_shape": list(R.shape),
        "reference_payload_bytes": int(R.nbytes),
    }

    (outdir / "full_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
PYVEC

chmod +x "$SDIR/e12b_vector_methods.py"

# ======================================================================
# 4. FROZEN Q50 ACCURACY EXTRACTION
#    Accuracy is not re-tuned.
#    HGV row uses model_seed 20260915 only.
# ======================================================================

cat > "$SDIR/e12b_extract_q50_accuracy.py" <<'PYACC'
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("HGV_V3_revision_20260915")
OUT = (
    BASE
    / "16_scalability/"
      "E12B_synchronized_resource_v1_20260930"
)

Q = (
    OUT
    / "01_manifests/"
      "E12B_QUERIES_50.tsv"
)

E5 = (
    BASE
    / "09_baselines/"
      "E5_external_unified_v1_20260920/"
      "06_external_final_freeze_v2_20260922/"
      "E5_EXTERNAL_ALL11_QUERY_LEVEL_FINAL.tsv"
)

q = pd.read_csv(Q, sep="\t", dtype=str)
df = pd.read_csv(E5, sep="\t", dtype=str)

qset = set(q["assembly_accession"])

x = df[
    df["query_accession"].isin(qset)
].copy()

if "cohort" in x.columns:
    x = x[x["cohort"].astype(str).eq("B1")].copy()

# HGV must correspond to the exact frozen G1 seed20260915 checkpoint.
is_hgv = (
    x["method_type"].astype(str).str.upper().eq("HGV")
    if "method_type" in x.columns
    else x["method"].astype(str).str.upper().str.contains("HGV")
)

keep = (
    (~is_hgv)
    |
    (
        is_hgv
        &
        x["model_seed"].astype(str).eq("20260915")
    )
)

x = x[keep].copy()

for c in [
    "eligible",
    "query_covered",
    "pair_coverage_query",
    "first_relevant_rank",
]:
    x[c] = pd.to_numeric(
        x[c],
        errors="coerce",
    )

def canonical(name):
    s = str(name).lower()

    if "hgv" in s:
        return "HGV"

    if "raw" in s and "bray" in s:
        return "Raw6_BrayCurtis"

    if "raw" in s and "cos" in s:
        return "Raw6_Cosine"

    if "raw" in s and "euclid" in s:
        return "Raw6_Euclidean"

    if "pca" in s and "l2" in s:
        return "PCA64_L2"

    if "pca" in s and "euclid" in s:
        return "PCA64_Euclidean"

    if "dashing" in s:
        return "Dashing2"

    if "fastani" in s.replace("_", ""):
        return "FastANI"

    if "skani" in s:
        return "skani"

    if "dnabert" in s:
        return "DNABERT-S"

    if "mash" in s:
        return "Mash"

    return str(name)

x["canonical_method"] = x["method"].map(canonical)

expected = {
    "HGV",
    "Raw6_BrayCurtis",
    "Raw6_Cosine",
    "Raw6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash",
    "Dashing2",
    "FastANI",
    "skani",
    "DNABERT-S",
}

observed = set(x["canonical_method"])

missing = expected - observed

if missing:
    print(
        x[
            [
                "method_type",
                "method",
                "model_seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["method_type", "method", "model_seed"]
        )
        .to_string(index=False)
    )

    raise RuntimeError(
        f"Missing methods after canonicalization: {sorted(missing)}"
    )

rows = []

for method in sorted(expected):
    z = x[
        x["canonical_method"].eq(method)
    ].copy()

    # exactly one record per query for synchronized Q50
    nq = z["query_accession"].nunique()

    if nq != 50:
        raise RuntimeError(
            f"{method}: query count={nq}, expected 50; rows={len(z)}"
        )

    # If duplicates remain per query, fail rather than silently average.
    dup = z["query_accession"].duplicated().sum()

    if dup:
        raise RuntimeError(
            f"{method}: duplicated query rows={dup}"
        )

    covered = z["query_covered"].fillna(0).eq(1)

    success = z["first_relevant_rank"].eq(1)

    rows.append(
        {
            "method": method,
            "n_queries": 50,
            "q50_recall_at_1_all_queries":
                float(success.mean()),
            "query_hit_coverage":
                float(covered.mean()),
            "mean_pair_coverage":
                float(
                    z["pair_coverage_query"].mean()
                ),
            "top1_correct_given_hit":
                float(
                    success[covered].mean()
                )
                if covered.any()
                else np.nan,
            "n_hit_queries":
                int(covered.sum()),
            "n_top1_success":
                int(success.sum()),
        }
    )

out = pd.DataFrame(rows)

path = (
    OUT
    / "04_summary/"
      "E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv"
)

path.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(
    path,
    sep="\t",
    index=False,
)

print(out.to_string(index=False))
print()
print("WROTE", path)
PYACC

chmod +x "$SDIR/e12b_extract_q50_accuracy.py"

echo
echo "======================================================================"
echo "STEP 1 - FREEZE 50-QUERY ACCURACY FROM EXISTING E5"
echo "======================================================================"

python "$SDIR/e12b_extract_q50_accuracy.py" \
    2>&1 | tee "$LOGS/01_accuracy_q50.log"

# ======================================================================
# 5. RESOURCE MONITOR HELPER
# ======================================================================

run_mon () {
    local name="$1"
    local od="$2"
    shift 2

    mkdir -p "$od"

    python "$SDIR/e12b_monitor.py" \
        --name "$name" \
        --out "$od" \
        -- "$@"
}

# ======================================================================
# 6. VECTOR METHODS: HGV + RAW3 + PCA2
# ======================================================================

VECTOR_METHODS=(
  HGV
  Raw6_BrayCurtis
  Raw6_Cosine
  Raw6_Euclidean
  PCA64_Euclidean
  PCA64_L2
)

echo
echo "======================================================================"
echo "STEP 2 - HGV / RAW / PCA"
echo "======================================================================"

for rep in $(seq 1 "$REPEATS"); do
    echo
    echo "----- VECTOR REPEAT $rep/$REPEATS -----"

    for method in "${VECTOR_METHODS[@]}"; do
        echo
        echo "### $method / rep$rep"

        MROOT="$RUNROOT/$method/rep${rep}"

        mkdir -p \
            "$MROOT/build" \
            "$MROOT/query" \
            "$MROOT/full"

        EXTRA=()

        if [[ "$method" == "HGV" ]]; then
            EXTRA+=(
              --ckpt "$E12B_HGV_G1_SEED15_CKPT"
            )
        fi

        if [[ "$method" == PCA64_* ]]; then
            EXTRA+=(
              --pca "$E12B_PCA_ARTIFACT"
            )
        fi

        run_mon \
          "${method}.build.rep${rep}" \
          "$MROOT/build/resource" \
          python "$SDIR/e12b_vector_methods.py" \
            --method "$method" \
            --phase build \
            --refs "$E12B_REFS_TSV" \
            --queries "$E12B_QUERIES_TSV" \
            --outdir "$MROOT/build/output" \
            --workers "$THREADS" \
            "${EXTRA[@]}"

        INDEX="$MROOT/build/output/reference_index.float32.npy"

        [[ -s "$INDEX" ]] || {
            echo "[FAIL] missing vector index: $INDEX"
            exit 2
        }

        run_mon \
          "${method}.query.rep${rep}" \
          "$MROOT/query/resource" \
          python "$SDIR/e12b_vector_methods.py" \
            --method "$method" \
            --phase query \
            --refs "$E12B_REFS_TSV" \
            --queries "$E12B_QUERIES_TSV" \
            --outdir "$MROOT/query/output" \
            --workers "$THREADS" \
            --index-in "$INDEX" \
            "${EXTRA[@]}"

        run_mon \
          "${method}.full.rep${rep}" \
          "$MROOT/full/resource" \
          python "$SDIR/e12b_vector_methods.py" \
            --method "$method" \
            --phase full \
            --refs "$E12B_REFS_TSV" \
            --queries "$E12B_QUERIES_TSV" \
            --outdir "$MROOT/full/output" \
            --workers "$THREADS" \
            "${EXTRA[@]}"
    done
done

# ======================================================================
# 7. MASH
#    Same production parameters already used in V3:
#    k=21, sketch size=1000.
# ======================================================================

echo
echo "======================================================================"
echo "STEP 3 - MASH"
echo "======================================================================"

for rep in $(seq 1 "$REPEATS"); do
    MROOT="$RUNROOT/Mash/rep${rep}"

    mkdir -p \
      "$MROOT/build" \
      "$MROOT/query" \
      "$MROOT/full"

    REFP="$MROOT/build/ref_k21_s1000"

    run_mon \
      "Mash.build.rep${rep}" \
      "$MROOT/build/resource" \
      mash sketch \
        -p "$THREADS" \
        -k 21 \
        -s 1000 \
        -l "$E12B_REFS_PATHS" \
        -o "$REFP"

    [[ -s "${REFP}.msh" ]] || {
        echo "[FAIL] Mash reference sketch missing"
        exit 2
    }

    stat -c '%s' "${REFP}.msh" \
      > "$MROOT/build/reference_index_bytes.txt"

    run_mon \
      "Mash.query.rep${rep}" \
      "$MROOT/query/resource" \
      bash -lc "
        set -euo pipefail
        QP='$MROOT/query/query_k21_s1000'
        rm -f \"\${QP}.msh\" '$MROOT/query/mash_dist.tsv'
        mash sketch \
          -p '$THREADS' \
          -k 21 \
          -s 1000 \
          -l '$E12B_QUERIES_PATHS' \
          -o \"\$QP\"
        mash dist \
          -p '$THREADS' \
          '${REFP}.msh' \
          \"\${QP}.msh\" \
          > '$MROOT/query/mash_dist.tsv'
      "

    run_mon \
      "Mash.full.rep${rep}" \
      "$MROOT/full/resource" \
      bash -lc "
        set -euo pipefail
        RP='$MROOT/full/ref'
        QP='$MROOT/full/query'
        rm -f \"\${RP}.msh\" \"\${QP}.msh\" '$MROOT/full/mash_dist.tsv'

        mash sketch \
          -p '$THREADS' \
          -k 21 \
          -s 1000 \
          -l '$E12B_REFS_PATHS' \
          -o \"\$RP\"

        mash sketch \
          -p '$THREADS' \
          -k 21 \
          -s 1000 \
          -l '$E12B_QUERIES_PATHS' \
          -o \"\$QP\"

        mash dist \
          -p '$THREADS' \
          \"\${RP}.msh\" \
          \"\${QP}.msh\" \
          > '$MROOT/full/mash_dist.tsv'
      "
done

# ======================================================================
# 8. DASHING2
#    IMPORTANT:
#    Installed v2.1.20 local help showed canonicalization enabled by
#    default; therefore DO NOT pass --no-canon.
# ======================================================================

echo
echo "======================================================================"
echo "STEP 4 - DASHING2"
echo "======================================================================"

D2HELP="$OUT/02_preflight/DASHING2_HELP_FULL.txt"

if grep -q -- '--cache-sketches' "$D2HELP"; then
    D2_CACHE_FLAG="--cache-sketches"
elif grep -q -- '--cache' "$D2HELP"; then
    D2_CACHE_FLAG="--cache"
else
    echo "[FAIL] cannot resolve Dashing2 cache flag"
    exit 2
fi

if ! grep -q -- '--outprefix' "$D2HELP"; then
    echo "[FAIL] installed Dashing2 has no --outprefix in frozen help"
    exit 2
fi

echo "Dashing2 cache flag = $D2_CACHE_FLAG"

for rep in $(seq 1 "$REPEATS"); do
    MROOT="$RUNROOT/Dashing2/rep${rep}"

    mkdir -p \
      "$MROOT/build/cache" \
      "$MROOT/query" \
      "$MROOT/full/cache"

    run_mon \
      "Dashing2.build.rep${rep}" \
      "$MROOT/build/resource" \
      dashing2 sketch \
        -p "$THREADS" \
        -k 31 \
        -S 1024 \
        "$D2_CACHE_FLAG" \
        --outprefix "$MROOT/build/cache/" \
        -F "$E12B_REFS_PATHS"

    du -sb "$MROOT/build/cache" \
      | awk '{print $1}' \
      > "$MROOT/build/reference_index_bytes.txt"

    run_mon \
      "Dashing2.query.rep${rep}" \
      "$MROOT/query/resource" \
      dashing2 sketch \
        -p "$THREADS" \
        -k 31 \
        -S 1024 \
        "$D2_CACHE_FLAG" \
        --outprefix "$MROOT/build/cache/" \
        --cmpout "$MROOT/query/dashing2_rect.tsv" \
        -Q "$E12B_QUERIES_PATHS" \
        -F "$E12B_REFS_PATHS"

    [[ -s "$MROOT/query/dashing2_rect.tsv" ]] || {
        echo "[FAIL] Dashing2 rectangular output empty"
        exit 2
    }

    run_mon \
      "Dashing2.full.rep${rep}" \
      "$MROOT/full/resource" \
      dashing2 sketch \
        -p "$THREADS" \
        -k 31 \
        -S 1024 \
        "$D2_CACHE_FLAG" \
        --outprefix "$MROOT/full/cache/" \
        --cmpout "$MROOT/full/dashing2_rect.tsv" \
        -Q "$E12B_QUERIES_PATHS" \
        -F "$E12B_REFS_PATHS"

    [[ -s "$MROOT/full/dashing2_rect.tsv" ]] || {
        echo "[FAIL] Dashing2 full output empty"
        exit 2
    }
done

# ======================================================================
# 9. SKANI
#    Build reusable reference database, then query it.
# ======================================================================

echo
echo "======================================================================"
echo "STEP 5 - SKANI"
echo "======================================================================"

for rep in $(seq 1 "$REPEATS"); do
    MROOT="$RUNROOT/skani/rep${rep}"

    mkdir -p \
      "$MROOT/build" \
      "$MROOT/query" \
      "$MROOT/full"

    run_mon \
      "skani.build.rep${rep}" \
      "$MROOT/build/resource" \
      bash -lc "
        set -euo pipefail
        rm -rf '$MROOT/build/db'
        mapfile -t REFS < '$E12B_REFS_PATHS'
        skani sketch \
          -t '$THREADS' \
          -o '$MROOT/build/db' \
          \"\${REFS[@]}\"
      "

    [[ -d "$MROOT/build/db" ]] || {
        echo "[FAIL] skani database missing"
        exit 2
    }

    du -sb "$MROOT/build/db" \
      | awk '{print $1}' \
      > "$MROOT/build/reference_index_bytes.txt"

    run_mon \
      "skani.query.rep${rep}" \
      "$MROOT/query/resource" \
      skani search \
        -t "$THREADS" \
        -d "$MROOT/build/db" \
        --ql "$E12B_QUERIES_PATHS" \
        -o "$MROOT/query/skani.tsv"

    run_mon \
      "skani.full.rep${rep}" \
      "$MROOT/full/resource" \
      bash -lc "
        set -euo pipefail
        rm -rf '$MROOT/full/db'
        mapfile -t REFS < '$E12B_REFS_PATHS'

        skani sketch \
          -t '$THREADS' \
          -o '$MROOT/full/db' \
          \"\${REFS[@]}\"

        skani search \
          -t '$THREADS' \
          -d '$MROOT/full/db' \
          --ql '$E12B_QUERIES_PATHS' \
          -o '$MROOT/full/skani.tsv'
      "
done

# ======================================================================
# 10. FASTANI
#     No persistent reference index in this benchmark.
#     Its synchronized value is complete many-to-many wall time.
# ======================================================================

echo
echo "======================================================================"
echo "STEP 6 - FASTANI"
echo "======================================================================"

for rep in $(seq 1 "$REPEATS"); do
    MROOT="$RUNROOT/FastANI/rep${rep}"

    mkdir -p "$MROOT/full"

    run_mon \
      "FastANI.full.rep${rep}" \
      "$MROOT/full/resource" \
      fastANI \
        --ql "$E12B_QUERIES_PATHS" \
        --rl "$E12B_REFS_PATHS" \
        --threads "$THREADS" \
        --output "$MROOT/full/fastani.tsv"

    # Empty individual pairs are allowed; the file itself should exist.
    [[ -e "$MROOT/full/fastani.tsv" ]] || {
        echo "[FAIL] FastANI output file missing"
        exit 2
    }
done

# ======================================================================
# 11. DNABERT-S
#     Frozen protocol:
#       32 x 1-kb strict-ACGT segments per genome
#       token-level masked mean
#       mean of 32 segment vectors
#       final genome-vector L2 normalization
#
#     This runner first requires the strict32 manifest to contain either
#     sequence strings or resolvable segment coordinates.
# ======================================================================

cat > "$SDIR/e12b_dnaberts.py" <<'PYDNA'
#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import sys
import json
import gzip
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_col(df, names, required=False):
    low = {c.lower(): c for c in df.columns}

    for n in names:
        if n.lower() in low:
            return low[n.lower()]

    for c in df.columns:
        lc = c.lower()
        for n in names:
            if n.lower() in lc:
                return c

    if required:
        raise RuntimeError(
            f"Cannot resolve {names}; columns={list(df.columns)}"
        )

    return None


def norm_acc(x):
    s = str(x)

    m = re.search(r"(GC[AF]_\d+\.\d+)", s)

    if m:
        return m.group(1)

    return s.strip()


def read_fasta(path):
    op = gzip.open if str(path).endswith(".gz") else open

    records = {}

    name = None
    parts = []

    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(parts).upper()

                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())

    if name is not None:
        records[name] = "".join(parts).upper()

    return records


ap = argparse.ArgumentParser()

ap.add_argument(
    "--mode",
    required=True,
    choices=["validate", "build", "query", "full"],
)

ap.add_argument("--strict32", required=True)
ap.add_argument("--subset", required=True)
ap.add_argument("--refs")
ap.add_argument("--queries")
ap.add_argument("--snapshot", required=True)
ap.add_argument("--index-in")
ap.add_argument("--outdir", required=True)
ap.add_argument("--batch", type=int, default=32)

args = ap.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

seg = pd.read_csv(
    args.strict32,
    sep="\t",
    dtype=str,
)

subset = pd.read_csv(
    args.subset,
    sep="\t",
    dtype=str,
)

acc_col = find_col(
    seg,
    [
        "assembly_accession",
        "genome_accession",
        "accession",
    ],
    required=True,
)

seg["_acc"] = seg[acc_col].map(norm_acc)

wanted = subset["assembly_accession"].map(norm_acc).tolist()
wanted_set = set(wanted)

seg = seg[
    seg["_acc"].isin(wanted_set)
].copy()

counts = seg.groupby("_acc").size()

bad = {
    k: int(v)
    for k, v in counts.items()
    if int(v) != 32
}

missing = sorted(
    wanted_set - set(counts.index)
)

if missing or bad:
    raise RuntimeError(
        f"strict32 subset invalid; "
        f"missing={len(missing)} bad_count={list(bad.items())[:10]}"
    )

seq_col = find_col(
    seg,
    [
        "segment_sequence",
        "sequence",
        "dna_sequence",
        "segment_seq",
    ],
    required=False,
)

fasta_col = find_col(
    seg,
    [
        "fasta_path",
        "genome_path",
        "assembly_path",
    ],
    required=False,
)

contig_col = find_col(
    seg,
    [
        "contig",
        "contig_id",
        "seqid",
        "record_id",
        "sequence_id",
    ],
    required=False,
)

start_col = find_col(
    seg,
    [
        "start0",
        "segment_start",
        "window_start",
        "start",
    ],
    required=False,
)

end_col = find_col(
    seg,
    [
        "end0",
        "segment_end",
        "window_end",
        "end",
    ],
    required=False,
)

subset_path = dict(
    zip(
        subset["assembly_accession"].map(norm_acc),
        subset["fasta_path"],
    )
)

fasta_cache = {}


def get_sequence(row):
    if seq_col is not None:
        s = str(row[seq_col]).upper()

        if len(s) == 1000 and set(s) <= set("ACGT"):
            return s

    # Coordinate fallback.
    if start_col is None:
        raise RuntimeError(
            "Strict32 manifest has no embedded segment sequence "
            "and no resolvable start coordinate.\n"
            f"Columns={list(seg.columns)}"
        )

    acc = row["_acc"]

    path = (
        str(row[fasta_col])
        if fasta_col is not None
        and pd.notna(row[fasta_col])
        and str(row[fasta_col]).strip()
        else subset_path[acc]
    )

    if path not in fasta_cache:
        fasta_cache[path] = read_fasta(path)

    recs = fasta_cache[path]

    if contig_col is None:
        raise RuntimeError(
            "Coordinate-based strict32 manifest requires a contig/record "
            "column; none was detected."
        )

    contig = str(row[contig_col])

    if contig not in recs:
        # tolerate description suffix differences
        cand = [
            k for k in recs
            if k == contig
            or k.startswith(contig)
            or contig.startswith(k)
        ]

        if len(cand) != 1:
            raise RuntimeError(
                f"Cannot resolve contig {contig} in {path}"
            )

        contig = cand[0]

    st = int(float(row[start_col]))

    if end_col is not None:
        en = int(float(row[end_col]))
    else:
        en = st + 1000

    s = recs[contig][st:en].upper()

    if len(s) != 1000 or set(s) - set("ACGT"):
        raise RuntimeError(
            f"Recovered segment is not strict 1000-bp ACGT: "
            f"{acc} {contig} {st}:{en}"
        )

    return s


if args.mode == "validate":
    print("strict32 rows =", len(seg))
    print("subset genomes =", len(wanted))
    print("columns =", list(seg.columns))
    print("sequence column =", seq_col)
    print("fasta column =", fasta_col)
    print("contig column =", contig_col)
    print("start column =", start_col)
    print("end column =", end_col)

    for _, row in seg.head(4).iterrows():
        s = get_sequence(row)
        print(row["_acc"], len(s), s[:20])

    print("VALIDATION PASS")
    raise SystemExit(0)


import torch
from transformers import AutoTokenizer, AutoModel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type != "cuda":
    raise RuntimeError("DNABERT-S E12B requires CUDA.")

tokenizer = AutoTokenizer.from_pretrained(
    args.snapshot,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    args.snapshot,
    trust_remote_code=True,
    local_files_only=True,
)

model = model.to(device)
model.eval()

torch.cuda.reset_peak_memory_stats()


def encode_genomes(accessions):
    vectors = []

    with torch.inference_mode():
        for ii, acc in enumerate(accessions, start=1):
            z = seg[
                seg["_acc"].eq(acc)
            ].copy()

            if len(z) != 32:
                raise RuntimeError(
                    f"{acc}: strict32 count={len(z)}"
                )

            seqs = [
                get_sequence(row)
                for _, row in z.iterrows()
            ]

            # exactly one genome = 32 segments per production batch
            tok = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            tok = {
                k: v.to(device)
                for k, v in tok.items()
            }

            hidden = model(**tok)[0]

            if "attention_mask" in tok:
                mask = tok["attention_mask"].unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1)
                segvec = (
                    (hidden * mask).sum(dim=1)
                    / denom
                )
            else:
                segvec = hidden.mean(dim=1)

            gv = segvec.mean(dim=0)

            gv = gv / torch.linalg.vector_norm(gv).clamp_min(1e-12)

            vectors.append(
                gv.detach().float().cpu().numpy()
            )

            if ii % 100 == 0 or ii == len(accessions):
                print(
                    f"[{ii}/{len(accessions)}] encoded",
                    flush=True,
                )

    return np.stack(vectors).astype(np.float32)


def scan(Q, R):
    sim = Q @ R.T
    idx = np.argmax(sim, axis=1)
    return idx, sim[np.arange(len(Q)), idx]


if args.mode == "build":
    R = encode_genomes(wanted)

    p = outdir / "reference_dnaberts_l2.float32.npy"
    np.save(p, R)

    meta = {
        "shape": list(R.shape),
        "payload_bytes": int(R.nbytes),
        "file_bytes": int(p.stat().st_size),
        "gpu_allocated_peak_mib":
            float(torch.cuda.max_memory_allocated() / 1024**2),
        "gpu_reserved_peak_mib":
            float(torch.cuda.max_memory_reserved() / 1024**2),
    }

    (outdir / "index_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print(json.dumps(meta, indent=2))

elif args.mode == "query":
    if args.index_in is None:
        raise RuntimeError("--index-in required")

    R = np.load(args.index_in).astype(np.float32)
    Q = encode_genomes(wanted)

    idx, score = scan(Q, R)

    pd.DataFrame(
        {
            "query_accession": wanted,
            "top1_index": idx,
            "top1_similarity": score,
        }
    ).to_csv(
        outdir / "top1_predictions.tsv",
        sep="\t",
        index=False,
    )

    meta = {
        "gpu_allocated_peak_mib":
            float(torch.cuda.max_memory_allocated() / 1024**2),
        "gpu_reserved_peak_mib":
            float(torch.cuda.max_memory_reserved() / 1024**2),
    }

    (outdir / "gpu_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

elif args.mode == "full":
    if not args.refs or not args.queries:
        raise RuntimeError("--refs and --queries required")

    rdf = pd.read_csv(
        args.refs,
        sep="\t",
        dtype=str,
    )

    qdf = pd.read_csv(
        args.queries,
        sep="\t",
        dtype=str,
    )

    racc = rdf["assembly_accession"].map(norm_acc).tolist()
    qacc = qdf["assembly_accession"].map(norm_acc).tolist()

    # seg already contains union subset passed via --subset
    R = encode_genomes(racc)
    Q = encode_genomes(qacc)

    idx, score = scan(Q, R)

    pd.DataFrame(
        {
            "query_accession": qacc,
            "top1_index": idx,
            "top1_similarity": score,
        }
    ).to_csv(
        outdir / "top1_predictions.tsv",
        sep="\t",
        index=False,
    )

    meta = {
        "reference_payload_bytes": int(R.nbytes),
        "gpu_allocated_peak_mib":
            float(torch.cuda.max_memory_allocated() / 1024**2),
        "gpu_reserved_peak_mib":
            float(torch.cuda.max_memory_reserved() / 1024**2),
    }

    (outdir / "full_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
PYDNA

chmod +x "$SDIR/e12b_dnaberts.py"

echo
echo "======================================================================"
echo "STEP 7 - DNABERT-S VALIDATION"
echo "======================================================================"

# Build a combined 2424-genome subset only for the FULL stage.
DNACOMB="$OUT/01_manifests/E12B_DNABERT_COMBINED_2424.tsv"

python3 - "$E12B_REFS_TSV" "$E12B_QUERIES_TSV" "$DNACOMB" <<'PY'
import sys
import pandas as pd

r = pd.read_csv(sys.argv[1], sep="\t", dtype=str)
q = pd.read_csv(sys.argv[2], sep="\t", dtype=str)

x = pd.concat([r, q], ignore_index=True)

assert len(x) == 2424
assert x["assembly_accession"].nunique() == 2424

x.to_csv(
    sys.argv[3],
    sep="\t",
    index=False,
)
PY

conda run -n dnabert_s_e5 \
  python "$SDIR/e12b_dnaberts.py" \
    --mode validate \
    --strict32 "$E12B_DNABERT_STRICT32" \
    --subset "$E12B_QUERIES_TSV" \
    --snapshot "$E12B_DNABERT_SNAPSHOT" \
    --outdir "$RUNROOT/DNABERT-S/validate" \
    2>&1 | tee "$LOGS/07_dnabert_validation.log"

echo "[PASS] DNABERT-S strict32 schema is executable"

echo
echo "======================================================================"
echo "STEP 8 - DNABERT-S RESOURCE BENCHMARK"
echo "======================================================================"

for rep in $(seq 1 "$REPEATS"); do
    MROOT="$RUNROOT/DNABERT-S/rep${rep}"

    mkdir -p \
      "$MROOT/build" \
      "$MROOT/query" \
      "$MROOT/full"

    run_mon \
      "DNABERT-S.build.rep${rep}" \
      "$MROOT/build/resource" \
      conda run -n dnabert_s_e5 \
        python "$SDIR/e12b_dnaberts.py" \
          --mode build \
          --strict32 "$E12B_DNABERT_STRICT32" \
          --subset "$E12B_REFS_TSV" \
          --snapshot "$E12B_DNABERT_SNAPSHOT" \
          --outdir "$MROOT/build/output"

    DNAINDEX="$MROOT/build/output/reference_dnaberts_l2.float32.npy"

    [[ -s "$DNAINDEX" ]] || {
        echo "[FAIL] DNABERT-S reference embedding file missing"
        exit 2
    }

    run_mon \
      "DNABERT-S.query.rep${rep}" \
      "$MROOT/query/resource" \
      conda run -n dnabert_s_e5 \
        python "$SDIR/e12b_dnaberts.py" \
          --mode query \
          --strict32 "$E12B_DNABERT_STRICT32" \
          --subset "$E12B_QUERIES_TSV" \
          --snapshot "$E12B_DNABERT_SNAPSHOT" \
          --index-in "$DNAINDEX" \
          --outdir "$MROOT/query/output"

    run_mon \
      "DNABERT-S.full.rep${rep}" \
      "$MROOT/full/resource" \
      conda run -n dnabert_s_e5 \
        python "$SDIR/e12b_dnaberts.py" \
          --mode full \
          --strict32 "$E12B_DNABERT_STRICT32" \
          --subset "$DNACOMB" \
          --refs "$E12B_REFS_TSV" \
          --queries "$E12B_QUERIES_TSV" \
          --snapshot "$E12B_DNABERT_SNAPSHOT" \
          --outdir "$MROOT/full/output"
done

# ======================================================================
# 12. FINAL SUMMARY / TABLE 7 SOURCE DATA
# ======================================================================

cat > "$SDIR/e12b_finalize.py" <<'PYFIN'
#!/usr/bin/env python3

from pathlib import Path
import json
import statistics
import numpy as np
import pandas as pd


BASE = Path("HGV_V3_revision_20260915")
OUT = (
    BASE
    / "16_scalability/"
      "E12B_synchronized_resource_v1_20260930"
)

RUN = OUT / "03_runs"
SUM = OUT / "04_summary"
FREEZE = OUT / "05_final_freeze"

SUM.mkdir(parents=True, exist_ok=True)
FREEZE.mkdir(parents=True, exist_ok=True)

ACC = pd.read_csv(
    SUM / "E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv",
    sep="\t",
)

acc = {
    row["method"]: row
    for _, row in ACC.iterrows()
}

methods = [
    "HGV",
    "Raw6_BrayCurtis",
    "Raw6_Cosine",
    "Raw6_Euclidean",
    "PCA64_Euclidean",
    "PCA64_L2",
    "Mash",
    "Dashing2",
    "skani",
    "FastANI",
    "DNABERT-S",
]


def read_resource(method, rep, phase):
    p = (
        RUN
        / method
        / f"rep{rep}"
        / phase
        / "resource"
        / "resource.json"
    )

    if not p.exists():
        return None

    return json.loads(p.read_text())


def median_field(method, phase, key):
    vals = []

    for rep in (1, 2, 3):
        x = read_resource(method, rep, phase)

        if x is not None and x.get("exit_code") == 0:
            v = x.get(key)

            if v is not None:
                vals.append(float(v))

    if not vals:
        return np.nan

    return float(statistics.median(vals))


def index_bytes(method):
    vals = []

    for rep in (1, 2, 3):
        root = RUN / method / f"rep{rep}"

        if method in {
            "HGV",
            "Raw6_BrayCurtis",
            "Raw6_Cosine",
            "Raw6_Euclidean",
            "PCA64_Euclidean",
            "PCA64_L2",
        }:
            p = root / "build/output/index_meta.json"

            if p.exists():
                vals.append(
                    int(
                        json.loads(
                            p.read_text()
                        )["payload_bytes"]
                    )
                )

        elif method in {"Mash", "Dashing2", "skani"}:
            p = root / "build/reference_index_bytes.txt"

            if p.exists():
                vals.append(int(p.read_text().strip()))

        elif method == "DNABERT-S":
            p = root / "build/output/index_meta.json"

            if p.exists():
                vals.append(
                    int(
                        json.loads(
                            p.read_text()
                        )["payload_bytes"]
                    )
                )

    if not vals:
        return np.nan

    return float(statistics.median(vals))


rows = []

for method in methods:
    a = acc[method]

    build_s = median_field(
        method,
        "build",
        "wall_s",
    )

    query_s = median_field(
        method,
        "query",
        "wall_s",
    )

    full_s = median_field(
        method,
        "full",
        "wall_s",
    )

    host = median_field(
        method,
        "full",
        "peak_host_pss_mib",
    )

    gpu = median_field(
        method,
        "full",
        "peak_gpu_mib",
    )

    ib = index_bytes(method)

    rows.append(
        {
            "method": method,
            "Q50_R1_all_queries":
                float(a["q50_recall_at_1_all_queries"]),
            "query_hit_coverage":
                float(a["query_hit_coverage"]),
            "mean_pair_coverage":
                float(a["mean_pair_coverage"]),
            "top1_correct_given_hit":
                float(a["top1_correct_given_hit"])
                if pd.notna(a["top1_correct_given_hit"])
                else np.nan,
            "build_s_median":
                build_s,
            "warm_query_s_median":
                query_s,
            "complete_s_median":
                full_s,
            "peak_host_PSS_MiB_median":
                host,
            "peak_GPU_MiB_median":
                gpu,
            "reference_index_MiB":
                ib / 1024**2
                if pd.notna(ib)
                else np.nan,
        }
    )

tab = pd.DataFrame(rows)

path = SUM / "TABLE7_SOURCE_DATA.tsv"

tab.to_csv(
    path,
    sep="\t",
    index=False,
    float_format="%.6f",
)

# Compact manuscript-oriented table
compact = tab[
    [
        "method",
        "Q50_R1_all_queries",
        "query_hit_coverage",
        "mean_pair_coverage",
        "build_s_median",
        "warm_query_s_median",
        "complete_s_median",
        "peak_host_PSS_MiB_median",
        "peak_GPU_MiB_median",
        "reference_index_MiB",
    ]
].copy()

compact.to_csv(
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE.tsv",
    sep="\t",
    index=False,
    float_format="%.4f",
)

print()
print("=" * 120)
print("E12B TABLE 7 SOURCE DATA")
print("=" * 120)
print(
    tab.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# Full audit
for method in methods:
    for phase in ["build", "query", "full"]:
        vals = []

        for rep in (1, 2, 3):
            x = read_resource(method, rep, phase)

            if x is not None:
                vals.append(x)

        if method == "FastANI" and phase in {"build", "query"}:
            continue

        if not vals:
            raise RuntimeError(
                f"Missing resource records: {method} {phase}"
            )

        if any(int(x["exit_code"]) != 0 for x in vals):
            raise RuntimeError(
                f"Nonzero execution: {method} {phase}"
            )

print()
print("[PASS] all synchronized resource records complete")

# Freeze hashes
import hashlib

freeze_files = [
    SUM / "TABLE7_SOURCE_DATA.tsv",
    SUM / "TABLE7_MANUSCRIPT_CANDIDATE.tsv",
    SUM / "E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv",
    OUT / "00_protocol/E12B_PROTOCOL_FREEZE.txt",
    OUT / "00_protocol/E12B_HGV_CHECKPOINT_FREEZE.txt",
]

lines = []

for p in freeze_files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(f"{h}  {p}")

(FREEZE / "SHA256SUMS.txt").write_text(
    "\n".join(lines) + "\n"
)

(FREEZE / "E12B_FINAL_STATUS.txt").write_text(
    "E12B SYNCHRONIZED RESOURCE BENCHMARK: PASS\n"
    "References: 2374\n"
    "Queries: 50 B1 resource-subset genomes\n"
    "Repeats: 3\n"
    "HGV: canonical G1 split20260915 model20260915 best_model.pt\n"
    "No HGV retraining performed.\n"
)

print()
print("FINAL STATUS = PASS")
print("TABLE7 SOURCE =", path)
PYFIN

chmod +x "$SDIR/e12b_finalize.py"

echo
echo "======================================================================"
echo "STEP 9 - FINALIZE"
echo "======================================================================"

python "$SDIR/e12b_finalize.py" \
    2>&1 | tee "$LOGS/09_finalize_e12b.log"

echo
echo "======================================================================"
echo "E12B COMPLETE"
echo "======================================================================"
echo
echo "Main outputs:"
echo "  $SUMMARY/TABLE7_SOURCE_DATA.tsv"
echo "  $SUMMARY/TABLE7_MANUSCRIPT_CANDIDATE.tsv"
echo "  $SUMMARY/E12B_Q50_ACCURACY_FROM_FROZEN_E5.tsv"
echo "  $FREEZE/E12B_FINAL_STATUS.txt"
echo "  $FREEZE/SHA256SUMS.txt"
echo
cat "$FREEZE/E12B_FINAL_STATUS.txt"
