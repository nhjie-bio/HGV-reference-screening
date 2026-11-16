#!/usr/bin/env python3

from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModel


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

MASTER = (
    BASE /
    "01_manifest/"
    "final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

SEGMENT_MANIFEST = (
    E5 /
    "06_dnaberts/"
    "03_strict32_selection_audit_v1_20260918/"
    "DNABERTS_STRICT32_SEGMENT_MANIFEST.tsv.gz"
)

PROTOCOL_FREEZE = (
    E5 /
    "06_dnaberts/"
    "06_production_protocol_freeze_v1_20260918/"
    "DNABERTS_PRODUCTION_PROTOCOL_FREEZE.txt"
)

OUT = (
    E5 /
    "06_dnaberts/"
    "07_production_embeddings_v1_20260918"
)

PER_GENOME = (
    OUT /
    "per_genome"
)

MODEL_ID = "zhihan1996/DNABERT-S"

REVISION = (
    "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
)

LOCAL_MODEL_DIR = (
    E5
    / "06_dnaberts"
    / "hf_cache"
    / "hub"
    / "models--zhihan1996--DNABERT-S"
    / "snapshots"
    / REVISION
)

BATCH_SIZE = 32
N_GENOMES = 3154
SEGMENTS_PER_GENOME = 32
N_SEGMENTS = 100928
WINDOW_BP = 1000
EMBED_DIM = 768


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def sha256_sequence(seq):

    return hashlib.sha256(
        seq.encode("ascii")
    ).hexdigest()


def fasta_records(path):

    path = Path(path)

    if str(path).endswith(".gz"):

        handle = gzip.open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    else:

        handle = open(
            path,
            "rt",
            encoding="utf-8",
            errors="replace",
        )

    with handle as f:

        name = None
        chunks = []

        for raw in f:

            line = raw.strip()

            if not line:
                continue

            if line.startswith(">"):

                if name is not None:

                    yield (
                        name,
                        "".join(chunks)
                    )

                name = (
                    line[1:]
                    .split()[0]
                )

                chunks = []

            else:

                chunks.append(
                    line
                )

        if name is not None:

            yield (
                name,
                "".join(chunks)
            )


def atomic_save_npy(
    path,
    array,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "wb",
    ) as f:

        np.save(
            f,
            array,
        )

    os.replace(
        tmp,
        path,
    )


def atomic_write_json(
    path,
    obj,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    with open(
        tmp,
        "w",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
        )

    os.replace(
        tmp,
        path,
    )


print("=" * 100)
print(
    "DNABERT-S PRODUCTION GENOME EMBEDDINGS v1"
)
print("=" * 100)

print(
    "model       =",
    MODEL_ID
)

print(
    "revision    =",
    REVISION
)

print(
    "local model =",
    LOCAL_MODEL_DIR
)

print(
    "batch size  =",
    BATCH_SIZE
)

print(
    "segments/genome =",
    SEGMENTS_PER_GENOME
)

print(
    "genomes     =",
    N_GENOMES
)


# ================================================================
# Preconditions
# ================================================================

for required in [
    MASTER,
    SEGMENT_MANIFEST,
    PROTOCOL_FREEZE,
    LOCAL_MODEL_DIR,
]:

    if not required.exists():

        raise FileNotFoundError(
            required
        )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PER_GENOME.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_STATUS = (
    OUT /
    "STATUS_COMPLETE.txt"
)


if FINAL_STATUS.is_file():

    print()
    print(
        "Production output is already COMPLETE."
    )

    print(
        FINAL_STATUS.read_text()
    )

    sys.exit(0)


# ================================================================
# Load frozen manifests
# ================================================================

master = pd.read_csv(
    MASTER,
    sep="\t",
    dtype=str,
)


segments = pd.read_csv(
    SEGMENT_MANIFEST,
    sep="\t",
    compression="gzip",
    dtype={
        "assembly_accession":
            str,
    },
)


if len(master) != N_GENOMES:

    raise RuntimeError(
        f"Master genomes={len(master)}, "
        f"expected={N_GENOMES}"
    )


if len(segments) != N_SEGMENTS:

    raise RuntimeError(
        f"Segments={len(segments)}, "
        f"expected={N_SEGMENTS}"
    )


if (
    "assembly_accession_final"
    not in master.columns
):

    raise RuntimeError(
        "Master missing "
        "assembly_accession_final"
    )


required_segment_cols = [
    "assembly_accession",
    "fasta_path",
    "segment_index",
    "contig",
    "start_0based",
    "end_exclusive",
    "window_bp",
    "non_acgt_count",
    "sequence_sha256",
    "token_count",
    "unk_token_count",
]


for c in required_segment_cols:

    if c not in segments.columns:

        raise RuntimeError(
            f"Segment manifest missing {c}"
        )


master_accessions = (
    master[
        "assembly_accession_final"
    ]
    .astype(str)
    .tolist()
)


if len(
    set(
        master_accessions
    )
) != N_GENOMES:

    raise RuntimeError(
        "Duplicate master accessions"
    )


segment_accessions = set(
    segments[
        "assembly_accession"
    ].astype(str)
)


if segment_accessions != set(
    master_accessions
):

    raise RuntimeError(
        "Master/segment accession "
        "sets differ"
    )


counts = (
    segments.groupby(
        "assembly_accession"
    ).size()
)


if not (
    counts == SEGMENTS_PER_GENOME
).all():

    raise RuntimeError(
        "Not every genome has "
        "exactly 32 segments"
    )


if not (
    pd.to_numeric(
        segments[
            "non_acgt_count"
        ]
    )
    == 0
).all():

    raise RuntimeError(
        "Non-ACGT segment found"
    )


if not (
    pd.to_numeric(
        segments[
            "unk_token_count"
        ]
    )
    == 0
).all():

    raise RuntimeError(
        "UNK segment found"
    )


if int(
    pd.to_numeric(
        segments[
            "token_count"
        ]
    ).max()
) > 512:

    raise RuntimeError(
        "Frozen segment exceeds "
        "512 tokens"
    )


print()
print(
    "Master SHA256  =",
    sha256_file(
        MASTER
    )
)

print(
    "Segment SHA256 =",
    sha256_file(
        SEGMENT_MANIFEST
    )
)

print(
    "Protocol SHA256=",
    sha256_file(
        PROTOCOL_FREEZE
    )
)


# ================================================================
# Frozen-run configuration
# ================================================================

script_path = Path(
    __file__
).resolve()


config = {
    "model_id":
        MODEL_ID,

    "revision":
        REVISION,

    "local_model_dir":
        str(
            LOCAL_MODEL_DIR
        ),

    "master_sha256":
        sha256_file(
            MASTER
        ),

    "segment_manifest_sha256":
        sha256_file(
            SEGMENT_MANIFEST
        ),

    "protocol_freeze_sha256":
        sha256_file(
            PROTOCOL_FREEZE
        ),

    "script_sha256":
        sha256_file(
            script_path
        ),

    "batch_size":
        BATCH_SIZE,

    "segments_per_genome":
        SEGMENTS_PER_GENOME,

    "window_bp":
        WINDOW_BP,

    "embedding_dim":
        EMBED_DIM,

    "dtype":
        "float32",

    "tf32":
        False,

    "segment_pooling":
        "attention-mask-weighted arithmetic mean",

    "genome_pooling":
        "arithmetic mean of 32 segment embeddings",
}


CONFIG_FILE = (
    OUT /
    "RUN_CONFIG.json"
)


if CONFIG_FILE.is_file():

    with open(
        CONFIG_FILE
    ) as f:

        previous = json.load(
            f
        )

    if previous != config:

        raise RuntimeError(
            "Existing production directory "
            "has a different RUN_CONFIG. "
            "Do not mix runs."
        )

    print()
    print(
        "Existing RUN_CONFIG verified; "
        "resume is permitted."
    )

else:

    atomic_write_json(
        CONFIG_FILE,
        config,
    )


# ================================================================
# GPU/model setup
# ================================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable"
    )


device = torch.device(
    "cuda:0"
)


torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(
    torch.backends,
    "cudnn"
):

    torch.backends.cudnn.allow_tf32 = False


torch.set_float32_matmul_precision(
    "highest"
)


print()
print(
    "GPU =",
    torch.cuda.get_device_name(0)
)

print(
    "torch =",
    torch.__version__
)

print(
    "transformers =",
    transformers.__version__
)


print()
print(
    "===== LOAD FROZEN MODEL ====="
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        str(
            LOCAL_MODEL_DIR
        ),
        trust_remote_code=True,
        local_files_only=True,
    )
)


model = (
    AutoModel.from_pretrained(
        str(
            LOCAL_MODEL_DIR
        ),
        trust_remote_code=True,
        local_files_only=True,
    )
)


model = model.to(
    device
)

model.eval()


if int(
    model.config.hidden_size
) != EMBED_DIM:

    raise RuntimeError(
        "Unexpected hidden dimension"
    )


if next(
    model.parameters()
).dtype != torch.float32:

    raise RuntimeError(
        "Model is not FP32"
    )


print(
    "model class =",
    model.__class__.__name__
)

print(
    "hidden size =",
    model.config.hidden_size
)

print(
    "pad id      =",
    tokenizer.pad_token_id
)

print(
    "unk id      =",
    tokenizer.unk_token_id
)


# ================================================================
# Segment grouping
# ================================================================

group_indices = (
    segments.groupby(
        "assembly_accession",
        sort=False,
    ).indices
)


# ================================================================
# Inference helper
# ================================================================

def infer_genome(
    seqs,
):

    if len(
        seqs
    ) != SEGMENTS_PER_GENOME:

        raise RuntimeError(
            "Expected exactly 32 sequences"
        )


    enc = tokenizer(
        seqs,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


    max_tokens = int(
        enc[
            "input_ids"
        ].shape[1]
    )


    if max_tokens > 512:

        raise RuntimeError(
            f"Tokenized batch length "
            f"{max_tokens} >512"
        )


    if (
        tokenizer.unk_token_id
        is not None
    ):

        unk_total = int(
            (
                enc[
                    "input_ids"
                ]
                ==
                tokenizer.unk_token_id
            ).sum()
        )

    else:

        unk_total = 0


    if unk_total != 0:

        raise RuntimeError(
            f"Unexpected UNK tokens: "
            f"{unk_total}"
        )


    inputs = {
        k:
            v.to(
                device
            )

        for k, v
        in enc.items()
    }


    torch.cuda.synchronize(
        device
    )

    t0 = time.perf_counter()


    with torch.inference_mode():

        hidden = model(
            **inputs
        )[0]


        mask = (
            inputs[
                "attention_mask"
            ]
            .unsqueeze(-1)
            .to(
                hidden.dtype
            )
        )


        segment_embeddings = (
            (
                hidden *
                mask
            ).sum(
                dim=1
            )
            /
            mask.sum(
                dim=1
            ).clamp_min(
                1.0
            )
        )


        genome_embedding = (
            segment_embeddings.mean(
                dim=0
            )
        )


    torch.cuda.synchronize(
        device
    )


    inference_sec = (
        time.perf_counter()
        -
        t0
    )


    if segment_embeddings.shape != (
        SEGMENTS_PER_GENOME,
        EMBED_DIM,
    ):

        raise RuntimeError(
            "Unexpected segment "
            "embedding shape"
        )


    if genome_embedding.shape != (
        EMBED_DIM,
    ):

        raise RuntimeError(
            "Unexpected genome "
            "embedding shape"
        )


    if not torch.isfinite(
        genome_embedding
    ).all():

        raise RuntimeError(
            "Non-finite genome embedding"
        )


    segment_norms = (
        torch.linalg.vector_norm(
            segment_embeddings,
            dim=1,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    genome = (
        genome_embedding
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


    return {
        "genome_embedding":
            genome,

        "inference_sec":
            float(
                inference_sec
            ),

        "max_tokens":
            max_tokens,

        "segment_norm_min":
            float(
                segment_norms.min()
            ),

        "segment_norm_mean":
            float(
                segment_norms.mean()
            ),

        "segment_norm_max":
            float(
                segment_norms.max()
            ),
    }


# ================================================================
# Deterministic production loop
# ================================================================

print()
print(
    "===== PRODUCTION INFERENCE ====="
)


session_start = (
    time.perf_counter()
)

session_computed = 0
session_skipped = 0


torch.cuda.reset_peak_memory_stats(
    device
)


for genome_index, accession in enumerate(
    master_accessions
):

    genome_dir = (
        PER_GENOME /
        f"{genome_index:04d}"
    )

    genome_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    emb_file = (
        genome_dir /
        "embedding_raw.npy"
    )

    audit_file = (
        genome_dir /
        "audit.json"
    )

    done_file = (
        genome_dir /
        "DONE"
    )


    # ------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------

    if done_file.is_file():

        if (
            not emb_file.is_file()
            or
            not audit_file.is_file()
        ):

            raise RuntimeError(
                f"{accession}: DONE exists "
                "but checkpoint is incomplete"
            )


        e = np.load(
            emb_file
        )


        if (
            e.shape != (
                EMBED_DIM,
            )
            or
            e.dtype != np.float32
            or
            not np.isfinite(
                e
            ).all()
        ):

            raise RuntimeError(
                f"{accession}: invalid "
                "existing embedding"
            )


        session_skipped += 1

        continue


    # Recover a checkpoint that completed before
    # the final DONE marker was written.

    if (
        emb_file.is_file()
        and
        audit_file.is_file()
    ):

        e = np.load(
            emb_file
        )

        with open(
            audit_file
        ) as f:

            a = json.load(
                f
            )


        if (
            e.shape
            ==
            (
                EMBED_DIM,
            )
            and
            e.dtype
            ==
            np.float32
            and
            np.isfinite(
                e
            ).all()
            and
            a.get(
                "status"
            )
            ==
            "PASS"
            and
            a.get(
                "assembly_accession"
            )
            ==
            accession
        ):

            done_file.write_text(
                "PASS\n"
            )

            session_skipped += 1

            continue


        raise RuntimeError(
            f"{accession}: partial "
            "checkpoint cannot be "
            "safely resumed"
        )


    if (
        emb_file.exists()
        !=
        audit_file.exists()
    ):

        raise RuntimeError(
            f"{accession}: inconsistent "
            "partial checkpoint"
        )


    # ------------------------------------------------------------
    # Frozen 32 segment rows
    # ------------------------------------------------------------

    idx = group_indices.get(
        accession
    )


    if idx is None:

        raise RuntimeError(
            f"No segments for "
            f"{accession}"
        )


    g = (
        segments.iloc[
            idx
        ]
        .sort_values(
            "segment_index"
        )
        .copy()
    )


    if len(g) != (
        SEGMENTS_PER_GENOME
    ):

        raise RuntimeError(
            f"{accession}: "
            f"{len(g)} segments"
        )


    expected_indices = list(
        range(
            1,
            SEGMENTS_PER_GENOME + 1,
        )
    )


    observed_indices = (
        pd.to_numeric(
            g[
                "segment_index"
            ]
        )
        .astype(int)
        .tolist()
    )


    if (
        observed_indices
        !=
        expected_indices
    ):

        raise RuntimeError(
            f"{accession}: invalid "
            "segment_index ordering"
        )


    fasta_paths = (
        g[
            "fasta_path"
        ]
        .astype(str)
        .unique()
    )


    if len(
        fasta_paths
    ) != 1:

        raise RuntimeError(
            f"{accession}: multiple "
            "FASTA paths"
        )


    fasta_path = Path(
        fasta_paths[0]
    )


    if not fasta_path.is_file():

        raise FileNotFoundError(
            fasta_path
        )


    # ------------------------------------------------------------
    # Read only required contigs.
    # ------------------------------------------------------------

    extract_start = (
        time.perf_counter()
    )


    wanted_contigs = set(
        g[
            "contig"
        ].astype(str)
    )


    found = {}


    for contig, seq in (
        fasta_records(
            fasta_path
        )
    ):

        if contig in wanted_contigs:

            found[
                contig
            ] = seq.upper()


    missing = (
        wanted_contigs -
        set(
            found.keys()
        )
    )


    if missing:

        raise RuntimeError(
            f"{accession}: missing "
            f"contigs {sorted(missing)[:5]}"
        )


    seqs = []


    for r in g.itertuples(
        index=False
    ):

        seq = found[
            str(
                r.contig
            )
        ][
            int(
                r.start_0based
            ):
            int(
                r.end_exclusive
            )
        ]


        if len(
            seq
        ) != WINDOW_BP:

            raise RuntimeError(
                f"{accession}: "
                "segment length !=1000"
            )


        if any(
            base not in "ACGT"
            for base in seq
        ):

            raise RuntimeError(
                f"{accession}: "
                "non-ACGT sequence"
            )


        observed_sha = (
            sha256_sequence(
                seq
            )
        )


        if observed_sha != str(
            r.sequence_sha256
        ):

            raise RuntimeError(
                f"{accession}: "
                "sequence SHA mismatch "
                f"segment {r.segment_index}"
            )


        seqs.append(
            seq
        )


    extract_sec = (
        time.perf_counter()
        -
        extract_start
    )


    # ------------------------------------------------------------
    # GPU inference
    # ------------------------------------------------------------

    result = infer_genome(
        seqs
    )


    embedding = result[
        "genome_embedding"
    ]


    embedding_norm = float(
        np.linalg.norm(
            embedding
        )
    )


    if (
        not np.isfinite(
            embedding_norm
        )
        or
        embedding_norm <= 0
    ):

        raise RuntimeError(
            f"{accession}: invalid "
            "genome embedding norm"
        )


    audit = {
        "status":
            "PASS",

        "genome_index":
            genome_index,

        "assembly_accession":
            accession,

        "fasta_path":
            str(
                fasta_path
            ),

        "n_segments":
            SEGMENTS_PER_GENOME,

        "window_bp":
            WINDOW_BP,

        "sequence_sha_verification":
            "PASS",

        "unk_total":
            0,

        "max_tokens":
            result[
                "max_tokens"
            ],

        "segment_norm_min":
            result[
                "segment_norm_min"
            ],

        "segment_norm_mean":
            result[
                "segment_norm_mean"
            ],

        "segment_norm_max":
            result[
                "segment_norm_max"
            ],

        "genome_embedding_norm":
            embedding_norm,

        "extract_sec":
            float(
                extract_sec
            ),

        "inference_sec":
            result[
                "inference_sec"
            ],

        "total_compute_sec":
            float(
                extract_sec
                +
                result[
                    "inference_sec"
                ]
            ),
    }


    # Embedding first, audit second,
    # DONE marker last.

    atomic_save_npy(
        emb_file,
        embedding,
    )


    atomic_write_json(
        audit_file,
        audit,
    )


    done_file.write_text(
        "PASS\n"
    )


    session_computed += 1


    completed = (
        session_computed
        +
        session_skipped
    )


    if (
        completed % 50 == 0
        or completed == N_GENOMES
    ):

        elapsed = (
            time.perf_counter()
            -
            session_start
        )


        if session_computed > 0:

            genomes_per_sec = (
                session_computed /
                elapsed
            )

            remaining = (
                N_GENOMES -
                completed
            )

            eta_sec = (
                remaining /
                genomes_per_sec
                if genomes_per_sec > 0
                else float("nan")
            )

        else:

            genomes_per_sec = 0.0
            eta_sec = 0.0


        print(
            f"[progress] "
            f"{completed}/{N_GENOMES}; "
            f"computed={session_computed}; "
            f"resumed={session_skipped}; "
            f"session={elapsed/60:.2f} min; "
            f"ETA={eta_sec/60:.2f} min"
        )


# ================================================================
# Aggregate final 3154 x 768 matrices
# ================================================================

print()
print(
    "===== FINAL AGGREGATION ====="
)


raw = np.empty(
    (
        N_GENOMES,
        EMBED_DIM,
    ),
    dtype=np.float32,
)


audit_rows = []


for genome_index, accession in enumerate(
    master_accessions
):

    genome_dir = (
        PER_GENOME /
        f"{genome_index:04d}"
    )

    emb_file = (
        genome_dir /
        "embedding_raw.npy"
    )

    audit_file = (
        genome_dir /
        "audit.json"
    )

    done_file = (
        genome_dir /
        "DONE"
    )


    if (
        not emb_file.is_file()
        or
        not audit_file.is_file()
        or
        not done_file.is_file()
    ):

        raise RuntimeError(
            f"{accession}: final "
            "checkpoint incomplete"
        )


    e = np.load(
        emb_file
    )


    if (
        e.shape != (
            EMBED_DIM,
        )
        or
        e.dtype != np.float32
        or
        not np.isfinite(
            e
        ).all()
    ):

        raise RuntimeError(
            f"{accession}: invalid "
            "final embedding"
        )


    raw[
        genome_index
    ] = e


    with open(
        audit_file
    ) as f:

        a = json.load(
            f
        )


    if (
        a.get(
            "assembly_accession"
        )
        != accession
        or
        a.get(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            f"{accession}: audit mismatch"
        )


    audit_rows.append(
        a
    )


norms = np.linalg.norm(
    raw,
    axis=1,
)


if (
    not np.isfinite(
        norms
    ).all()
    or
    np.any(
        norms <= 0
    )
):

    raise RuntimeError(
        "Invalid raw genome norms"
    )


l2 = (
    raw /
    norms[
        :,
        None
    ]
).astype(
    np.float32
)


l2_norms = np.linalg.norm(
    l2,
    axis=1,
)


if not np.allclose(
    l2_norms,
    1.0,
    rtol=1e-5,
    atol=1e-6,
):

    raise RuntimeError(
        "L2 normalization failed"
    )


RAW_FILE = (
    OUT /
    "DNABERTS_GENOME_EMBEDDINGS_RAW_FLOAT32.npy"
)

L2_FILE = (
    OUT /
    "DNABERTS_GENOME_EMBEDDINGS_L2_FLOAT32.npy"
)

AUDIT_FILE = (
    OUT /
    "DNABERTS_GENOME_EMBEDDING_AUDIT.tsv"
)

MANIFEST_FILE = (
    OUT /
    "DNABERTS_GENOME_EMBEDDING_MANIFEST.tsv"
)


atomic_save_npy(
    RAW_FILE,
    raw,
)

atomic_save_npy(
    L2_FILE,
    l2,
)


audit_df = pd.DataFrame(
    audit_rows
).sort_values(
    "genome_index"
)


audit_df.to_csv(
    AUDIT_FILE,
    sep="\t",
    index=False,
)


out_manifest = pd.DataFrame({
    "embedding_index":
        np.arange(
            N_GENOMES,
            dtype=int,
        ),

    "assembly_accession":
        master_accessions,

    "raw_l2_norm":
        norms,

    "normalized_l2_norm":
        l2_norms,
})


out_manifest.to_csv(
    MANIFEST_FILE,
    sep="\t",
    index=False,
)


# ================================================================
# Final audit/report
# ================================================================

sum_inference_sec = float(
    pd.to_numeric(
        audit_df[
            "inference_sec"
        ]
    ).sum()
)

sum_compute_sec = float(
    pd.to_numeric(
        audit_df[
            "total_compute_sec"
        ]
    ).sum()
)


inference_throughput = (
    N_SEGMENTS /
    sum_inference_sec
)

pipeline_throughput = (
    N_SEGMENTS /
    sum_compute_sec
)


peak_alloc_mb = (
    torch.cuda.max_memory_allocated(
        device
    )
    /
    1024**2
)

peak_reserved_mb = (
    torch.cuda.max_memory_reserved(
        device
    )
    /
    1024**2
)


REPORT = (
    OUT /
    "DNABERTS_PRODUCTION_EMBEDDING_REPORT.txt"
)


with open(
    REPORT,
    "w",
) as f:

    f.write(
        "DNABERT-S Production Genome Embeddings v1\n"
    )

    f.write(
        "=" * 84 +
        "\n\n"
    )

    f.write(
        f"Model: {MODEL_ID}\n"
    )

    f.write(
        f"Revision: {REVISION}\n"
    )

    f.write(
        f"Local snapshot: "
        f"{LOCAL_MODEL_DIR}\n\n"
    )

    f.write(
        f"Genomes: {N_GENOMES}\n"
    )

    f.write(
        f"Segments: {N_SEGMENTS}\n"
    )

    f.write(
        f"Segments/genome: "
        f"{SEGMENTS_PER_GENOME}\n"
    )

    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Genome embedding dimension: "
        f"{EMBED_DIM}\n"
    )

    f.write(
        "Dtype: float32\n"
    )

    f.write(
        "TF32: disabled\n\n"
    )

    f.write(
        "Segment pooling: "
        "attention-mask-weighted mean "
        "over non-padding token hidden states\n"
    )

    f.write(
        "Genome pooling: arithmetic mean "
        "over 32 segment embeddings\n\n"
    )

    f.write(
        "All sequence SHA checks: PASS\n"
    )

    f.write(
        "All genome embeddings finite: PASS\n"
    )

    f.write(
        "All normalized embedding norms "
        "approximately 1: PASS\n\n"
    )

    f.write(
        f"Raw embedding norm min: "
        f"{norms.min():.10f}\n"
    )

    f.write(
        f"Raw embedding norm median: "
        f"{np.median(norms):.10f}\n"
    )

    f.write(
        f"Raw embedding norm max: "
        f"{norms.max():.10f}\n\n"
    )

    f.write(
        f"Summed model inference sec: "
        f"{sum_inference_sec:.3f}\n"
    )

    f.write(
        f"Summed extraction+inference sec: "
        f"{sum_compute_sec:.3f}\n"
    )

    f.write(
        f"Model inference throughput "
        f"(segments/sec): "
        f"{inference_throughput:.6f}\n"
    )

    f.write(
        f"Pipeline throughput "
        f"(segments/sec): "
        f"{pipeline_throughput:.6f}\n"
    )

    f.write(
        f"Current-session peak allocated MB: "
        f"{peak_alloc_mb:.3f}\n"
    )

    f.write(
        f"Current-session peak reserved MB: "
        f"{peak_reserved_mb:.3f}\n\n"
    )

    f.write(
        "STATUS: COMPLETE\n"
    )


# ================================================================
# Hash final outputs
# ================================================================

HASH_FILE = (
    OUT /
    "FINAL_SHA256.txt"
)


with open(
    HASH_FILE,
    "w",
) as f:

    for p in [
        MASTER,
        SEGMENT_MANIFEST,
        PROTOCOL_FREEZE,
        script_path,
        CONFIG_FILE,
        RAW_FILE,
        L2_FILE,
        AUDIT_FILE,
        MANIFEST_FILE,
        REPORT,
    ]:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


FINAL_STATUS.write_text(
    "STATUS = COMPLETE\n"
)


print()
print("=" * 100)
print(
    "FINAL DNABERT-S PRODUCTION EMBEDDING SUMMARY"
)
print("=" * 100)

print(
    "genomes =",
    N_GENOMES
)

print(
    "segments =",
    N_SEGMENTS
)

print(
    "raw shape =",
    raw.shape
)

print(
    "L2 shape =",
    l2.shape
)

print(
    "raw dtype =",
    raw.dtype
)

print(
    "finite =",
    bool(
        np.isfinite(
            raw
        ).all()
    )
)

print(
    "L2 norm min/median/max =",
    float(
        l2_norms.min()
    ),
    float(
        np.median(
            l2_norms
        )
    ),
    float(
        l2_norms.max()
    )
)

print(
    "model inference throughput =",
    f"{inference_throughput:.3f}",
    "segments/sec"
)

print(
    "pipeline throughput =",
    f"{pipeline_throughput:.3f}",
    "segments/sec"
)

print(
    "peak reserved MB =",
    f"{peak_reserved_mb:.1f}"
)

print()
print(
    "RAW:",
    RAW_FILE
)

print(
    "L2:",
    L2_FILE
)

print(
    "MANIFEST:",
    MANIFEST_FILE
)

print()
print(
    "STATUS = COMPLETE"
)
