#!/usr/bin/env python3

from pathlib import Path
import gzip
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel


BASE = Path(
    "HGV_V3_revision_20260915"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

STRICT = (
    E5 /
    "06_dnaberts/"
    "03_strict32_selection_audit_v1_20260918"
)

SEGMENT_MANIFEST = (
    STRICT /
    "DNABERTS_STRICT32_SEGMENT_MANIFEST.tsv.gz"
)

OUT = (
    E5 /
    "06_dnaberts/"
    "05_batch_benchmark_v4_20260918"
)

if OUT.exists():
    raise RuntimeError(
        f"Output already exists:\n{OUT}\n"
        "Use a new version; do not overwrite."
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
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

if not LOCAL_MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"Frozen local DNABERT-S snapshot not found:\n"
        f"{LOCAL_MODEL_DIR}"
    )

REQUIRED_LOCAL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
]

for required_name in REQUIRED_LOCAL_FILES:
    required_path = (
        LOCAL_MODEL_DIR /
        required_name
    )

    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required DNABERT-S snapshot file missing:\n"
            f"{required_path}"
        )

BATCH_SIZES = [
    1,
    4,
    8,
    16,
    32,
]

N_REPRESENTATIVE = 128
N_STRESS_POOL = 64
N_REPEATS = 2

MAX_RESERVED_FRACTION = 0.85


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


print("=" * 96)
print(
    "DNABERT-S RTX3060 BATCH BENCHMARK v4"
)
print("=" * 96)

print(
    "model    =",
    MODEL_ID
)

print(
    "revision =",
    REVISION
)

print(
    "local snapshot =",
    LOCAL_MODEL_DIR
)

print(
    "torch    =",
    torch.__version__
)

print(
    "HF       =",
    transformers.__version__
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable"
    )


device = torch.device(
    "cuda:0"
)

gpu_prop = (
    torch.cuda.get_device_properties(
        device
    )
)

total_vram_bytes = int(
    gpu_prop.total_memory
)

print(
    "GPU      =",
    gpu_prop.name
)

print(
    "VRAM GiB =",
    total_vram_bytes /
    1024**3
)


# Keep DNABERT-S inference in FP32.
torch.backends.cuda.matmul.allow_tf32 = False

if hasattr(
    torch.backends,
    "cudnn"
):
    torch.backends.cudnn.allow_tf32 = False

torch.set_float32_matmul_precision(
    "highest"
)


# ================================================================
# Load frozen segment manifest.
# ================================================================

segments = pd.read_csv(
    SEGMENT_MANIFEST,
    sep="\t",
    compression="gzip",
    dtype={
        "assembly_accession": str,
    },
)


if len(segments) != 100928:

    raise RuntimeError(
        f"Expected 100928 segments, "
        f"found {len(segments)}"
    )


required = [
    "assembly_accession",
    "fasta_path",
    "segment_index",
    "contig",
    "start_0based",
    "end_exclusive",
    "sequence_sha256",
    "token_count",
    "unk_token_count",
    "non_acgt_count",
]


for c in required:

    if c not in segments.columns:

        raise RuntimeError(
            f"Segment manifest missing {c}"
        )


if not (
    segments[
        "non_acgt_count"
    ] == 0
).all():

    raise RuntimeError(
        "Non-ACGT segment found"
    )


if not (
    segments[
        "unk_token_count"
    ] == 0
).all():

    raise RuntimeError(
        "UNK segment found"
    )


if (
    segments[
        "token_count"
    ].max()
    > 512
):

    raise RuntimeError(
        "Token count exceeds 512"
    )


segments = (
    segments.reset_index(
        drop=True
    )
)

segments["row_id"] = (
    np.arange(
        len(segments),
        dtype=np.int64,
    )
)


print()
print(
    "segment manifest rows =",
    len(segments)
)

print(
    "token range =",
    int(
        segments[
            "token_count"
        ].min()
    ),
    "-",
    int(
        segments[
            "token_count"
        ].max()
    ),
)


# ================================================================
# Deterministic benchmark sample.
# ================================================================

rep_ids = np.linspace(
    0,
    len(segments) - 1,
    N_REPRESENTATIVE,
    dtype=np.int64,
)


if len(
    np.unique(
        rep_ids
    )
) != N_REPRESENTATIVE:

    raise RuntimeError(
        "Representative IDs not unique"
    )


rep_df = (
    segments.iloc[
        rep_ids
    ]
    .copy()
)


stress_df = (
    segments.sort_values(
        [
            "token_count",
            "assembly_accession",
            "segment_index",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )
    .head(
        N_STRESS_POOL
    )
    .copy()
)


selected = (
    pd.concat(
        [
            rep_df,
            stress_df,
        ],
        ignore_index=True,
    )
    .drop_duplicates(
        subset=[
            "row_id"
        ]
    )
)


print(
    "representative segments =",
    len(rep_df)
)

print(
    "stress pool segments    =",
    len(stress_df)
)

print(
    "unique sequences to load=",
    len(selected)
)


# ================================================================
# Extract exact sequences from FASTAs.
# ================================================================

sequence_by_rowid = {}


print()
print(
    "===== EXTRACT BENCHMARK SEQUENCES ====="
)


for fasta_path, group in (
    selected.groupby(
        "fasta_path",
        sort=False,
    )
):

    fasta_path = Path(
        fasta_path
    )


    if not fasta_path.is_file():

        raise FileNotFoundError(
            fasta_path
        )


    wanted_contigs = set(
        group[
            "contig"
        ].astype(str)
    )


    found = {}


    for name, seq in (
        fasta_records(
            fasta_path
        )
    ):

        if name in wanted_contigs:

            found[name] = (
                seq.upper()
            )


    missing = (
        wanted_contigs -
        set(
            found.keys()
        )
    )


    if missing:

        raise RuntimeError(
            f"Missing contigs in {fasta_path}: "
            f"{sorted(missing)[:5]}"
        )


    for r in group.itertuples(
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


        if len(seq) != 1000:

            raise RuntimeError(
                "Extracted sequence "
                "length !=1000"
            )


        if any(
            base not in "ACGT"
            for base in seq
        ):

            raise RuntimeError(
                "Non-ACGT sequence "
                "in benchmark sample"
            )


        if (
            sha256_sequence(
                seq
            )
            !=
            str(
                r.sequence_sha256
            )
        ):

            raise RuntimeError(
                "Sequence SHA mismatch: "
                f"{r.assembly_accession} "
                f"segment {r.segment_index}"
            )


        sequence_by_rowid[
            int(
                r.row_id
            )
        ] = seq


if len(
    sequence_by_rowid
) != len(
    selected
):

    raise RuntimeError(
        "Sequence extraction "
        "count mismatch"
    )


rep_sequences = [
    sequence_by_rowid[
        int(x)
    ]
    for x in rep_df[
        "row_id"
    ]
]


stress_sequences = [
    sequence_by_rowid[
        int(x)
    ]
    for x in stress_df[
        "row_id"
    ]
]


print(
    "sequence SHA validation = PASS"
)


# ================================================================
# Model/tokenizer.
# ================================================================

print()
print(
    "===== LOAD FROZEN DNABERT-S ====="
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        str(LOCAL_MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )
)


model = (
    AutoModel.from_pretrained(
        str(LOCAL_MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )
)


model = model.to(
    device
)

model.eval()


parameter_dtype = next(
    model.parameters()
).dtype


print(
    "tokenizer =",
    tokenizer.__class__.__name__
)

print(
    "pad token =",
    tokenizer.pad_token,
    tokenizer.pad_token_id
)

print(
    "model     =",
    model.__class__.__name__
)

print(
    "dtype     =",
    parameter_dtype
)

print(
    "hidden    =",
    model.config.hidden_size
)


if parameter_dtype != torch.float32:

    raise RuntimeError(
        f"Expected FP32 model, "
        f"found {parameter_dtype}"
    )


# Reproduce recorded tokenizer lengths
# for all benchmark sequences.

for row in (
    selected.itertuples(
        index=False
    )
):

    seq = sequence_by_rowid[
        int(
            row.row_id
        )
    ]

    ids = tokenizer(
        seq,
        add_special_tokens=True,
        truncation=False,
    )[
        "input_ids"
    ]


    if (
        len(ids)
        !=
        int(
            row.token_count
        )
    ):

        raise RuntimeError(
            "Tokenizer length mismatch "
            f"for row {row.row_id}: "
            f"{len(ids)} vs "
            f"{row.token_count}"
        )


print(
    "tokenizer length "
    "reproduction = PASS"
)


# ================================================================
# Forward helper.
#
# Pool only non-padding token positions.
# This preserves the single-sequence arithmetic mean
# while making padded batches mathematically consistent.
# ================================================================

def forward_embeddings(
    seq_batch,
):

    enc = tokenizer(
        seq_batch,
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
            f"Batch token length "
            f"{max_tokens} >512"
        )


    inputs = {
        k: v.to(
            device,
            non_blocking=False,
        )
        for k, v in (
            enc.items()
        )
    }


    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        hidden = outputs[0]

        mask = (
            inputs[
                "attention_mask"
            ]
            .unsqueeze(-1)
            .to(
                hidden.dtype
            )
        )

        embedding = (
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


    if embedding.shape != (
        len(
            seq_batch
        ),
        768,
    ):

        raise RuntimeError(
            "Unexpected embedding shape"
        )


    if not torch.isfinite(
        embedding
    ).all():

        raise RuntimeError(
            "Non-finite embedding"
        )


    return (
        embedding,
        max_tokens,
    )


# ================================================================
# Batching equivalence audit.
# ================================================================

print()
print(
    "===== BATCHING EQUIVALENCE AUDIT ====="
)


equiv_sequences = (
    rep_sequences[:8]
)


individual = []


for seq in equiv_sequences:

    emb, _ = (
        forward_embeddings(
            [seq]
        )
    )

    individual.append(
        emb.detach().cpu()
    )


individual = torch.cat(
    individual,
    dim=0,
)


batch_emb, batch_tokens = (
    forward_embeddings(
        equiv_sequences
    )
)


batch_cpu = (
    batch_emb
    .detach()
    .cpu()
)


cosines = (
    F.cosine_similarity(
        individual,
        batch_cpu,
        dim=1,
    )
)


max_abs_diff = float(
    torch.max(
        torch.abs(
            individual -
            batch_cpu
        )
    )
)


min_cosine = float(
    cosines.min()
)


mean_cosine = float(
    cosines.mean()
)


print(
    "sequences       =",
    len(
        equiv_sequences
    )
)

print(
    "padded tokens   =",
    batch_tokens
)

print(
    "min cosine      =",
    f"{min_cosine:.10f}"
)

print(
    "mean cosine     =",
    f"{mean_cosine:.10f}"
)

print(
    "max abs diff    =",
    f"{max_abs_diff:.10e}"
)


BATCH_EQUIV_PASS = (
    min_cosine >=
    0.9999
)


print(
    "batch equivalence =",
    "PASS"
    if BATCH_EQUIV_PASS
    else "FAIL"
)


if not BATCH_EQUIV_PASS:

    raise RuntimeError(
        "Batch padding changes "
        "embeddings beyond tolerance"
    )


del batch_emb
del batch_cpu
del individual

gc.collect()
torch.cuda.empty_cache()


# ================================================================
# Benchmark helper.
# ================================================================

def run_sequence_set(
    seqs,
    batch_size,
):

    n_done = 0
    max_batch_tokens = 0


    t0 = time.perf_counter()


    for i in range(
        0,
        len(seqs),
        batch_size,
    ):

        batch = seqs[
            i:
            i + batch_size
        ]


        emb, n_tokens = (
            forward_embeddings(
                batch
            )
        )


        max_batch_tokens = max(
            max_batch_tokens,
            n_tokens,
        )


        n_done += len(
            batch
        )


        # Force a tiny materialization so the
        # computation cannot be optimized away.
        _ = float(
            emb[
                0,
                0
            ].item()
        )


        del emb


    torch.cuda.synchronize(
        device
    )


    elapsed = (
        time.perf_counter()
        -
        t0
    )


    return (
        elapsed,
        n_done,
        max_batch_tokens,
    )


# ================================================================
# Benchmark batch sizes.
# ================================================================

print()
print(
    "===== REPRESENTATIVE + STRESS BENCHMARK ====="
)


benchmark_rows = []


for bs in BATCH_SIZES:

    print()
    print(
        f"--- batch_size={bs} ---"
    )


    status = "PASS"
    error = ""

    elapsed_values = []
    throughput_values = []

    representative_peak_alloc = np.nan
    representative_peak_reserved = np.nan
    representative_max_tokens = np.nan

    stress_peak_alloc = np.nan
    stress_peak_reserved = np.nan
    stress_max_tokens = np.nan


    try:

        # ----------------------------
        # Warm-up.
        # ----------------------------

        warm_n = min(
            bs,
            len(
                rep_sequences
            )
        )

        warm_emb, _ = (
            forward_embeddings(
                rep_sequences[
                    :warm_n
                ]
            )
        )

        torch.cuda.synchronize(
            device
        )

        del warm_emb

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(
            device
        )


        # ----------------------------
        # Representative benchmark.
        # ----------------------------

        peak_alloc_runs = []
        peak_reserved_runs = []
        max_token_runs = []


        for repeat in range(
            N_REPEATS
        ):

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(
                device
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )


            elapsed, n_done, max_tokens = (
                run_sequence_set(
                    rep_sequences,
                    bs,
                )
            )


            peak_alloc = (
                torch.cuda.max_memory_allocated(
                    device
                )
            )

            peak_reserved = (
                torch.cuda.max_memory_reserved(
                    device
                )
            )


            elapsed_values.append(
                elapsed
            )

            throughput_values.append(
                n_done /
                elapsed
            )

            peak_alloc_runs.append(
                peak_alloc
            )

            peak_reserved_runs.append(
                peak_reserved
            )

            max_token_runs.append(
                max_tokens
            )


        representative_peak_alloc = (
            max(
                peak_alloc_runs
            )
        )

        representative_peak_reserved = (
            max(
                peak_reserved_runs
            )
        )

        representative_max_tokens = (
            max(
                max_token_runs
            )
        )


        # ----------------------------
        # Worst-token stress batch.
        # ----------------------------

        stress_batch = (
            stress_sequences[
                :bs
            ]
        )


        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(
            device
        )

        torch.cuda.reset_peak_memory_stats(
            device
        )


        stress_emb, stress_max_tokens = (
            forward_embeddings(
                stress_batch
            )
        )


        torch.cuda.synchronize(
            device
        )


        stress_peak_alloc = (
            torch.cuda.max_memory_allocated(
                device
            )
        )

        stress_peak_reserved = (
            torch.cuda.max_memory_reserved(
                device
            )
        )


        del stress_emb


    except (
        torch.cuda.OutOfMemoryError,
        RuntimeError,
    ) as e:

        msg = str(
            e
        )

        if (
            isinstance(
                e,
                torch.cuda.OutOfMemoryError
            )
            or
            "out of memory"
            in msg.lower()
        ):

            status = "OOM"

        else:

            status = "ERROR"

        error = msg[
            :1000
        ]

        gc.collect()
        torch.cuda.empty_cache()


    elapsed_median = (
        float(
            np.median(
                elapsed_values
            )
        )
        if elapsed_values
        else np.nan
    )


    throughput_median = (
        float(
            np.median(
                throughput_values
            )
        )
        if throughput_values
        else np.nan
    )


    reserved_fraction = (
        float(
            stress_peak_reserved /
            total_vram_bytes
        )
        if np.isfinite(
            stress_peak_reserved
        )
        else np.nan
    )


    safety_pass = bool(
        status == "PASS"
        and
        np.isfinite(
            reserved_fraction
        )
        and
        reserved_fraction
        <=
        MAX_RESERVED_FRACTION
    )


    row = {
        "batch_size":
            bs,

        "status":
            status,

        "n_representative":
            len(
                rep_sequences
            ),

        "n_repeats":
            N_REPEATS,

        "elapsed_median_sec":
            elapsed_median,

        "throughput_median_segments_per_sec":
            throughput_median,

        "representative_max_padded_tokens":
            representative_max_tokens,

        "representative_peak_allocated_mb":
            (
                representative_peak_alloc /
                1024**2
                if np.isfinite(
                    representative_peak_alloc
                )
                else np.nan
            ),

        "representative_peak_reserved_mb":
            (
                representative_peak_reserved /
                1024**2
                if np.isfinite(
                    representative_peak_reserved
                )
                else np.nan
            ),

        "stress_batch_size":
            min(
                bs,
                len(
                    stress_sequences
                )
            ),

        "stress_max_padded_tokens":
            stress_max_tokens,

        "stress_peak_allocated_mb":
            (
                stress_peak_alloc /
                1024**2
                if np.isfinite(
                    stress_peak_alloc
                )
                else np.nan
            ),

        "stress_peak_reserved_mb":
            (
                stress_peak_reserved /
                1024**2
                if np.isfinite(
                    stress_peak_reserved
                )
                else np.nan
            ),

        "stress_reserved_fraction_total_vram":
            reserved_fraction,

        "safety_limit_fraction":
            MAX_RESERVED_FRACTION,

        "safe_with_headroom":
            safety_pass,

        "error":
            error,
    }


    benchmark_rows.append(
        row
    )


    print(
        "status       =",
        status
    )

    print(
        "median seq/s =",
        (
            f"{throughput_median:.3f}"
            if np.isfinite(
                throughput_median
            )
            else "NA"
        )
    )

    print(
        "stress peak reserved MB =",
        (
            f"{row['stress_peak_reserved_mb']:.1f}"
            if np.isfinite(
                row[
                    "stress_peak_reserved_mb"
                ]
            )
            else "NA"
        )
    )

    print(
        "VRAM fraction =",
        (
            f"{reserved_fraction:.4f}"
            if np.isfinite(
                reserved_fraction
            )
            else "NA"
        )
    )

    print(
        "safe headroom =",
        safety_pass
    )


benchmark = pd.DataFrame(
    benchmark_rows
)


# ================================================================
# Recommend fastest safe configuration.
# ================================================================

safe = benchmark[
    (
        benchmark[
            "safe_with_headroom"
        ]
        ==
        True
    )
    &
    (
        benchmark[
            "status"
        ]
        ==
        "PASS"
    )
].copy()


if len(safe) == 0:

    raise RuntimeError(
        "No safe batch size found"
    )


safe = safe.sort_values(
    [
        "throughput_median_segments_per_sec",
        "batch_size",
    ],
    ascending=[
        False,
        True,
    ],
)


recommended = safe.iloc[
    0
]


recommended_bs = int(
    recommended[
        "batch_size"
    ]
)


print()
print("=" * 96)
print(
    "FINAL BATCH BENCHMARK"
)
print("=" * 96)

print(
    benchmark.to_string(
        index=False
    )
)

print()
print(
    "Batch-equivalence min cosine =",
    f"{min_cosine:.10f}"
)

print(
    "Batch-equivalence max abs diff =",
    f"{max_abs_diff:.10e}"
)

print()
print(
    "RECOMMENDED BATCH SIZE =",
    recommended_bs
)


# ================================================================
# Save results.
# ================================================================

benchmark_file = (
    OUT /
    "DNABERTS_BATCH_BENCHMARK.tsv"
)


benchmark.to_csv(
    benchmark_file,
    sep="\t",
    index=False,
)


metadata = {
    "model_id":
        MODEL_ID,

    "revision":
        REVISION,

    "local_model_snapshot":
        str(LOCAL_MODEL_DIR),

    "torch":
        torch.__version__,

    "transformers":
        transformers.__version__,

    "gpu":
        gpu_prop.name,

    "total_vram_bytes":
        total_vram_bytes,

    "dtype":
        str(
            parameter_dtype
        ),

    "tf32":
        False,

    "n_representative":
        N_REPRESENTATIVE,

    "n_stress_pool":
        N_STRESS_POOL,

    "n_repeats":
        N_REPEATS,

    "batch_sizes":
        BATCH_SIZES,

    "max_reserved_fraction":
        MAX_RESERVED_FRACTION,

    "batch_equivalence_min_cosine":
        min_cosine,

    "batch_equivalence_mean_cosine":
        mean_cosine,

    "batch_equivalence_max_abs_diff":
        max_abs_diff,

    "batch_equivalence_pass":
        BATCH_EQUIV_PASS,

    "recommended_batch_size":
        recommended_bs,
}


metadata_file = (
    OUT /
    "DNABERTS_BATCH_BENCHMARK_METADATA.json"
)


with open(
    metadata_file,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


report = (
    OUT /
    "DNABERTS_BATCH_BENCHMARK_REPORT.txt"
)


with open(
    report,
    "w",
) as f:

    f.write(
        "DNABERT-S Batch Benchmark v4\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )

    f.write(
        f"Model: {MODEL_ID}\n"
    )

    f.write(
        f"Revision: {REVISION}\n"
    )

    f.write(
        f"Local snapshot: {LOCAL_MODEL_DIR}\n"
    )

    f.write(
        f"GPU: {gpu_prop.name}\n"
    )

    f.write(
        f"Total VRAM GiB: "
        f"{total_vram_bytes/1024**3:.4f}\n"
    )

    f.write(
        f"Dtype: {parameter_dtype}\n"
    )

    f.write(
        "TF32: disabled\n\n"
    )

    f.write(
        "Pooling implementation:\n"
        "mean over token hidden states at "
        "attention_mask==1 only; padding positions "
        "are excluded.\n\n"
    )

    f.write(
        "Batch equivalence:\n"
        f"min cosine = {min_cosine:.10f}\n"
        f"mean cosine = {mean_cosine:.10f}\n"
        f"max abs diff = {max_abs_diff:.10e}\n"
        f"PASS = {BATCH_EQUIV_PASS}\n\n"
    )

    f.write(
        benchmark.to_string(
            index=False
        )
    )

    f.write(
        "\n\nRecommended batch size: "
        f"{recommended_bs}\n"
    )

    f.write(
        "\nSTATUS: PASS\n"
    )


script_path = Path(
    __file__
).resolve()


with open(
    OUT /
    "PROVENANCE_SHA256.txt",
    "w",
) as f:

    for p in [
        SEGMENT_MANIFEST,
        script_path,
        benchmark_file,
        metadata_file,
        report,
    ]:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


print()
print(
    "Report:"
)
print(report)

print()
print(
    "STATUS = PASS"
)
