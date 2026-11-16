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
