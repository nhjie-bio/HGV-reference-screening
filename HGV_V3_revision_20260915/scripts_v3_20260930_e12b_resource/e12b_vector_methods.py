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
