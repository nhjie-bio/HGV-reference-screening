#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import torch
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

OUT = (
    E5 /
    "06_dnaberts/"
    "01_model_smoke_v1_20260918"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_ID = "zhihan1996/DNABERT-S"

REVISION = (
    "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:

            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


print("=" * 88)
print("DNABERT-S MODEL / TOKENIZER SMOKE TEST v1")
print("=" * 88)

print("model    =", MODEL_ID)
print("revision =", REVISION)
print("python   =", sys.version.split()[0])
print("torch    =", torch.__version__)
print("HF       =", transformers.__version__)

print()

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable"
    )

device = torch.device("cuda:0")

print(
    "GPU      =",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM GiB =",
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)


# ------------------------------------------------------------
# Load tokenizer/model from a FIXED revision.
# ------------------------------------------------------------

print()
print("===== LOAD TOKENIZER =====")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
)

print(
    "tokenizer class =",
    tokenizer.__class__.__name__
)

print(
    "model_max_length =",
    tokenizer.model_max_length
)


print()
print("===== LOAD MODEL =====")

model = AutoModel.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
)

model = model.to(device)
model.eval()

print(
    "model class =",
    model.__class__.__name__
)

print(
    "hidden_size =",
    model.config.hidden_size
)

print(
    "layers      =",
    model.config.num_hidden_layers
)

print(
    "heads       =",
    model.config.num_attention_heads
)

if int(model.config.hidden_size) != 768:
    raise RuntimeError(
        "Unexpected hidden size"
    )


# ------------------------------------------------------------
# Four deterministic tokenizer/inference tests.
# ------------------------------------------------------------

seq_acgt = (
    "ACGT" * 250
)

seq_with_n = (
    seq_acgt[:495]
    + "NNNNNNNNNN"
    + seq_acgt[505:]
)

seq_lower = (
    seq_acgt.lower()
)

seq_all_n = (
    "N" * 1000
)


cases = {
    "pure_acgt":
        seq_acgt,

    "internal_10N":
        seq_with_n,

    "lowercase_acgt":
        seq_lower,

    "all_N":
        seq_all_n,
}


rows = []
embeddings = {}


print()
print("===== TOKENIZER + GPU INFERENCE =====")


for name, seq in cases.items():

    encoded = tokenizer(
        seq,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = (
        encoded["input_ids"]
    )

    attention_mask = (
        encoded.get(
            "attention_mask"
        )
    )

    token_count = int(
        input_ids.shape[1]
    )

    unk_id = (
        tokenizer.unk_token_id
    )

    if unk_id is None:

        n_unk = 0

    else:

        n_unk = int(
            (
                input_ids ==
                int(unk_id)
            ).sum()
        )


    print()
    print(
        f"[{name}]"
    )

    print(
        "bp length   =",
        len(seq)
    )

    print(
        "token count =",
        token_count
    )

    print(
        "UNK count   =",
        n_unk
    )


    # Do not silently truncate during smoke test.
    if token_count > 512:

        print(
            "FORWARD = SKIPPED "
            "(token_count > 512)"
        )

        rows.append({
            "case":
                name,

            "bp_length":
                len(seq),

            "token_count":
                token_count,

            "unk_count":
                n_unk,

            "forward":
                "SKIPPED_GT512",

            "hidden_shape":
                "",

            "embedding_dim":
                np.nan,

            "embedding_norm":
                np.nan,

            "finite":
                False,
        })

        continue


    gpu_inputs = {
        k: v.to(device)
        for k, v
        in encoded.items()
    }


    torch.cuda.reset_peak_memory_stats(
        device
    )


    with torch.inference_mode():

        outputs = model(
            **gpu_inputs
        )

        hidden = outputs[0]

        # Official DNABERT-S quick-start pooling:
        # arithmetic mean across token hidden states.
        embedding = torch.mean(
            hidden[0],
            dim=0,
        )


    emb_np = (
        embedding
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    finite = bool(
        np.isfinite(
            emb_np
        ).all()
    )


    peak_mb = (
        torch.cuda.max_memory_allocated(
            device
        ) /
        1024**2
    )


    print(
        "hidden shape =",
        tuple(
            hidden.shape
        )
    )

    print(
        "embedding    =",
        tuple(
            embedding.shape
        )
    )

    print(
        "finite       =",
        finite
    )

    print(
        "norm         =",
        float(
            np.linalg.norm(
                emb_np
            )
        )
    )

    print(
        "peak CUDA MB =",
        f"{peak_mb:.2f}"
    )


    if hidden.shape[-1] != 768:

        raise RuntimeError(
            f"{name}: "
            "hidden dim != 768"
        )

    if embedding.shape[0] != 768:

        raise RuntimeError(
            f"{name}: "
            "embedding dim != 768"
        )

    if not finite:

        raise RuntimeError(
            f"{name}: "
            "non-finite embedding"
        )


    embeddings[name] = emb_np


    rows.append({
        "case":
            name,

        "bp_length":
            len(seq),

        "token_count":
            token_count,

        "unk_count":
            n_unk,

        "forward":
            "PASS",

        "hidden_shape":
            str(
                tuple(
                    hidden.shape
                )
            ),

        "embedding_dim":
            int(
                embedding.shape[0]
            ),

        "embedding_norm":
            float(
                np.linalg.norm(
                    emb_np
                )
            ),

        "finite":
            finite,

        "peak_cuda_mb":
            float(
                peak_mb
            ),
    })


# ------------------------------------------------------------
# Lowercase-vs-uppercase diagnostic.
# ------------------------------------------------------------

if (
    "pure_acgt" in embeddings
    and
    "lowercase_acgt" in embeddings
):

    a = embeddings[
        "pure_acgt"
    ]

    b = embeddings[
        "lowercase_acgt"
    ]

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    cosine = float(
        np.dot(a, b) /
        denom
    )

    print()
    print(
        "uppercase/lowercase "
        "embedding cosine =",
        cosine
    )

else:

    cosine = None


# ------------------------------------------------------------
# Save audit.
# ------------------------------------------------------------

import pandas as pd

df = pd.DataFrame(
    rows
)

result_file = (
    OUT /
    "DNABERTS_SMOKE_RESULTS.tsv"
)

df.to_csv(
    result_file,
    sep="\t",
    index=False,
)


meta = {
    "model_id":
        MODEL_ID,

    "revision":
        REVISION,

    "python":
        sys.version,

    "torch":
        torch.__version__,

    "transformers":
        transformers.__version__,

    "cuda_available":
        torch.cuda.is_available(),

    "torch_cuda":
        torch.version.cuda,

    "gpu":
        torch.cuda.get_device_name(0),

    "hidden_size":
        int(
            model.config.hidden_size
        ),

    "uppercase_lowercase_cosine":
        cosine,
}


with open(
    OUT /
    "DNABERTS_SMOKE_METADATA.json",
    "w",
) as f:

    json.dump(
        meta,
        f,
        indent=2,
    )


script = Path(
    __file__
).resolve()

with open(
    OUT /
    "SCRIPT_SHA256.txt",
    "w",
) as f:

    f.write(
        f"{sha256_file(script)}  "
        f"{script}\n"
    )


print()
print("=" * 88)
print("FINAL SMOKE SUMMARY")
print("=" * 88)

print(
    df.to_string(
        index=False
    )
)

print()
print(
    "Result:",
    result_file
)

print()
print("STATUS = PASS")
