#!/usr/bin/env python3

from pathlib import Path
from bisect import bisect_right
import gzip
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


BASE = Path(
    "HGV_V3_revision_20260915/"
    "HGV_V3_revision_20260915"
)

MANIFEST = (
    BASE /
    "01_manifest/"
    "final_3154_freeze_v1_20260916/"
    "final_master_manifest_3154.tsv"
)

E5 = (
    BASE /
    "09_baselines/"
    "E5_unified_baseline_v1_20260917"
)

OUT = (
    E5 /
    "06_dnaberts/"
    "03_strict32_selection_audit_v1_20260918"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_ID = "zhihan1996/DNABERT-S"

REVISION = (
    "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
)

WINDOW_BP = 1000
N_SEGMENTS = 32

VALID_BASES = set("ACGT")


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


def strict_valid_intervals(
    seq,
    window_bp,
):

    """
    Return inclusive start-coordinate intervals [a,b]
    for windows of length window_bp containing only A/C/G/T.

    This avoids building an O(genome_length) boolean start array.

    A bad base at position p invalidates starts:
        p-window_bp+1 <= start <= p
    clipped to legal start coordinates.
    """

    L = len(seq)

    if L < window_bp:
        return []


    max_start = (
        L -
        window_bp
    )


    bad_positions = [
        i
        for i, base in
        enumerate(seq)
        if base not in VALID_BASES
    ]


    if not bad_positions:

        return [
            (
                0,
                max_start,
            )
        ]


    invalid = []


    for p in bad_positions:

        a = max(
            0,
            p - window_bp + 1,
        )

        b = min(
            p,
            max_start,
        )

        if a <= b:

            invalid.append(
                (
                    a,
                    b,
                )
            )


    if not invalid:

        return [
            (
                0,
                max_start,
            )
        ]


    # Merge invalid intervals.

    merged = []

    cur_a, cur_b = (
        invalid[0]
    )

    for a, b in (
        invalid[1:]
    ):

        if a <= (
            cur_b + 1
        ):

            cur_b = max(
                cur_b,
                b,
            )

        else:

            merged.append(
                (
                    cur_a,
                    cur_b,
                )
            )

            cur_a, cur_b = (
                a,
                b,
            )


    merged.append(
        (
            cur_a,
            cur_b,
        )
    )


    # Complement inside [0, max_start].

    valid = []

    cursor = 0


    for a, b in merged:

        if cursor < a:

            valid.append(
                (
                    cursor,
                    a - 1,
                )
            )

        cursor = max(
            cursor,
            b + 1,
        )


    if cursor <= max_start:

        valid.append(
            (
                cursor,
                max_start,
            )
        )


    return valid


def midpoint_targets(
    total,
    n,
):

    if total < n:

        raise ValueError(
            f"Need >= {n} valid starts; "
            f"found {total}"
        )


    return [
        (
            (2 * i + 1)
            * total
        )
        //
        (2 * n)

        for i in range(n)
    ]


print("=" * 96)
print(
    "DNABERT-S STRICT-ACGT "
    "32x1kb SELECTION AUDIT v1"
)
print("=" * 96)

print(
    "window bp        =",
    WINDOW_BP
)

print(
    "segments/genome  =",
    N_SEGMENTS
)

print(
    "ambiguity policy = STRICT ACGT ONLY"
)

print(
    "model            =",
    MODEL_ID
)

print(
    "revision         =",
    REVISION
)


# ================================================================
# Manifest
# ================================================================

manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
    dtype=str,
)


for c in [
    "assembly_accession_final",
    "fasta_path",
]:

    if c not in manifest.columns:

        raise RuntimeError(
            f"Manifest missing {c}"
        )


if len(manifest) != 3154:

    raise RuntimeError(
        f"Expected 3154 genomes; "
        f"found {len(manifest)}"
    )


if manifest[
    "assembly_accession_final"
].duplicated().any():

    raise RuntimeError(
        "Duplicate accession "
        "in frozen manifest"
    )


print()
print(
    "manifest genomes =",
    len(manifest)
)

print(
    "manifest SHA256  =",
    sha256_file(
        MANIFEST
    )
)


# ================================================================
# Tokenizer
# ================================================================

print()
print(
    "===== LOAD FROZEN TOKENIZER ====="
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
)


UNK_ID = (
    tokenizer.unk_token_id
)


print(
    "tokenizer class =",
    tokenizer.__class__.__name__
)

print(
    "UNK token id    =",
    UNK_ID
)


# ================================================================
# Process
# ================================================================

segment_rows = []
genome_rows = []

t0 = time.time()


print()
print(
    "===== SELECT STRICT 32 WINDOWS / GENOME ====="
)


for genome_idx, row in enumerate(
    manifest.itertuples(
        index=False
    ),
    start=1,
):

    accession = str(
        row.assembly_accession_final
    )

    fasta_path = Path(
        str(
            row.fasta_path
        )
    )


    if not fasta_path.is_file():

        raise FileNotFoundError(
            fasta_path
        )


    contigs = []

    assembly_bp = 0
    total_contigs = 0

    interval_records = []


    for contig_idx, (
        contig_name,
        raw_seq,
    ) in enumerate(
        fasta_records(
            fasta_path
        ),
        start=1,
    ):

        total_contigs += 1

        seq = raw_seq.upper()

        L = len(seq)

        assembly_bp += L


        contigs.append({
            "index":
                contig_idx,

            "name":
                contig_name,

            "seq":
                seq,

            "length":
                L,
        })


        if L < WINDOW_BP:
            continue


        valid_intervals = (
            strict_valid_intervals(
                seq,
                WINDOW_BP,
            )
        )


        for a, b in (
            valid_intervals
        ):

            count = (
                b -
                a +
                1
            )


            if count <= 0:
                continue


            interval_records.append({
                "contig_index":
                    contig_idx,

                "contig_name":
                    contig_name,

                "seq":
                    seq,

                "contig_length":
                    L,

                "start_first":
                    a,

                "start_last":
                    b,

                "n_valid_starts":
                    count,
            })


    total_valid_starts = int(
        sum(
            x[
                "n_valid_starts"
            ]
            for x in interval_records
        )
    )


    if (
        total_valid_starts <
        N_SEGMENTS
    ):

        genome_rows.append({
            "assembly_accession":
                accession,

            "assembly_bp":
                assembly_bp,

            "n_contigs":
                total_contigs,

            "n_valid_intervals":
                len(
                    interval_records
                ),

            "strict_valid_start_count":
                total_valid_starts,

            "n_selected":
                0,

            "n_unique_selected":
                0,

            "status":
                "INSUFFICIENT_STRICT_STARTS",
        })

        continue


    cumulative = []

    running = 0


    for x in interval_records:

        running += int(
            x[
                "n_valid_starts"
            ]
        )

        cumulative.append(
            running
        )


    targets = midpoint_targets(
        total_valid_starts,
        N_SEGMENTS,
    )


    selected = []
    selected_keys = []


    for segment_index, target in enumerate(
        targets,
        start=1,
    ):

        interval_idx = (
            bisect_right(
                cumulative,
                target,
            )
        )


        if interval_idx >= len(
            interval_records
        ):

            raise RuntimeError(
                f"{accession}: "
                "target interval overflow"
            )


        previous_end = (
            0
            if interval_idx == 0
            else cumulative[
                interval_idx - 1
            ]
        )


        offset = int(
            target -
            previous_end
        )


        rec = interval_records[
            interval_idx
        ]


        start = (
            int(
                rec[
                    "start_first"
                ]
            )
            +
            offset
        )

        end = (
            start +
            WINDOW_BP
        )


        seq = rec["seq"][
            start:end
        ]


        if len(seq) != WINDOW_BP:

            raise RuntimeError(
                f"{accession}: "
                "window length != 1000"
            )


        if any(
            base not in VALID_BASES
            for base in seq
        ):

            raise RuntimeError(
                f"{accession}: "
                "non-ACGT found in "
                "strict selected window"
            )


        key = (
            rec[
                "contig_index"
            ],
            start,
            end,
        )


        selected_keys.append(
            key
        )


        selected.append({
            "assembly_accession":
                accession,

            "fasta_path":
                str(
                    fasta_path
                ),

            "segment_index":
                segment_index,

            "global_strict_rank":
                int(
                    target
                ),

            "strict_valid_start_count":
                total_valid_starts,

            "contig_index":
                int(
                    rec[
                        "contig_index"
                    ]
                ),

            "contig":
                rec[
                    "contig_name"
                ],

            "contig_length":
                int(
                    rec[
                        "contig_length"
                    ]
                ),

            "start_0based":
                start,

            "end_exclusive":
                end,

            "window_bp":
                WINDOW_BP,

            "non_acgt_count":
                0,

            "sequence_sha256":
                sha256_sequence(
                    seq
                ),

            "_seq":
                seq,
        })


    unique_selected = len(
        set(
            selected_keys
        )
    )


    if unique_selected != N_SEGMENTS:

        raise RuntimeError(
            f"{accession}: "
            f"selected only "
            f"{unique_selected} "
            "unique windows"
        )


    # ------------------------------------------------------------
    # Tokenizer verification.
    # ------------------------------------------------------------

    seqs = [
        x["_seq"]
        for x in selected
    ]


    encoded = tokenizer(
        seqs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )


    ids_batch = (
        encoded[
            "input_ids"
        ]
    )


    if len(
        ids_batch
    ) != N_SEGMENTS:

        raise RuntimeError(
            f"{accession}: "
            "tokenizer batch mismatch"
        )


    token_counts = []
    unk_counts = []


    for x, ids in zip(
        selected,
        ids_batch,
    ):

        token_count = len(
            ids
        )


        if UNK_ID is None:

            unk_count = 0

        else:

            unk_count = int(
                sum(
                    token_id == UNK_ID
                    for token_id in ids
                )
            )


        if unk_count != 0:

            raise RuntimeError(
                f"{accession}: "
                "strict ACGT window "
                "generated UNK token"
            )


        if token_count > 512:

            raise RuntimeError(
                f"{accession}: "
                "strict ACGT window "
                f"has {token_count} tokens"
            )


        x[
            "token_count"
        ] = (
            token_count
        )

        x[
            "unk_token_count"
        ] = (
            unk_count
        )


        token_counts.append(
            token_count
        )

        unk_counts.append(
            unk_count
        )


        del x["_seq"]

        segment_rows.append(
            x
        )


    genome_rows.append({
        "assembly_accession":
            accession,

        "assembly_bp":
            assembly_bp,

        "n_contigs":
            total_contigs,

        "n_valid_intervals":
            len(
                interval_records
            ),

        "strict_valid_start_count":
            total_valid_starts,

        "n_selected":
            N_SEGMENTS,

        "n_unique_selected":
            unique_selected,

        "token_min":
            int(
                min(
                    token_counts
                )
            ),

        "token_median":
            float(
                np.median(
                    token_counts
                )
            ),

        "token_max":
            int(
                max(
                    token_counts
                )
            ),

        "unk_total":
            int(
                sum(
                    unk_counts
                )
            ),

        "status":
            "PASS_STRICT32",
    })


    if (
        genome_idx % 100 == 0
        or genome_idx == len(
            manifest
        )
    ):

        elapsed = (
            time.time() -
            t0
        )

        print(
            f"[progress] "
            f"{genome_idx}/"
            f"{len(manifest)} "
            f"genomes; "
            f"elapsed="
            f"{elapsed:.1f}s"
        )


# ================================================================
# Audit
# ================================================================

segments = pd.DataFrame(
    segment_rows
)

genomes = pd.DataFrame(
    genome_rows
)


EXPECTED_SEGMENTS = (
    3154 *
    N_SEGMENTS
)


n_pass_genomes = int(
    (
        genomes[
            "status"
        ]
        ==
        "PASS_STRICT32"
    ).sum()
)


print()
print(
    "===== STRICT32 AUDIT ====="
)

print(
    "genomes          =",
    len(genomes)
)

print(
    "PASS_STRICT32    =",
    n_pass_genomes
)

print(
    "segments         =",
    len(segments),
    "/",
    EXPECTED_SEGMENTS
)


if len(
    genomes
) != 3154:

    raise RuntimeError(
        "Genome audit row count "
        "is not 3154"
    )


if n_pass_genomes != 3154:

    bad = genomes[
        genomes[
            "status"
        ]
        !=
        "PASS_STRICT32"
    ]

    bad.to_csv(
        OUT /
        "FAILED_STRICT32_GENOMES.tsv",
        sep="\t",
        index=False,
    )

    raise RuntimeError(
        f"Strict32 feasibility failed "
        f"for "
        f"{3154 - n_pass_genomes} "
        "genomes"
    )


if len(
    segments
) != EXPECTED_SEGMENTS:

    raise RuntimeError(
        "Segment count mismatch"
    )


if not (
    segments[
        "non_acgt_count"
    ] == 0
).all():

    raise RuntimeError(
        "Non-ACGT sequence leaked "
        "into strict segments"
    )


if not (
    segments[
        "unk_token_count"
    ] == 0
).all():

    raise RuntimeError(
        "UNK token found"
    )


if not (
    segments[
        "token_count"
    ] <= 512
).all():

    raise RuntimeError(
        "Token count >512"
    )


dup = (
    segments.groupby(
        "assembly_accession"
    )[
        [
            "contig_index",
            "start_0based",
            "end_exclusive",
        ]
    ]
    .apply(
        lambda x:
        x.duplicated().sum()
    )
)


if int(
    dup.sum()
) != 0:

    raise RuntimeError(
        "Duplicate windows detected"
    )


# ================================================================
# Summary
# ================================================================

token_summary = {
    "n_genomes":
        int(
            len(
                genomes
            )
        ),

    "n_segments":
        int(
            len(
                segments
            )
        ),

    "segments_per_genome":
        N_SEGMENTS,

    "window_bp":
        WINDOW_BP,

    "strict_non_acgt_required":
        0,

    "unk_token_total":
        int(
            segments[
                "unk_token_count"
            ].sum()
        ),

    "token_count_min":
        int(
            segments[
                "token_count"
            ].min()
        ),

    "token_count_median":
        float(
            segments[
                "token_count"
            ].median()
        ),

    "token_count_mean":
        float(
            segments[
                "token_count"
            ].mean()
        ),

    "token_count_p95":
        float(
            segments[
                "token_count"
            ].quantile(
                0.95
            )
        ),

    "token_count_p99":
        float(
            segments[
                "token_count"
            ].quantile(
                0.99
            )
        ),

    "token_count_max":
        int(
            segments[
                "token_count"
            ].max()
        ),

    "strict_valid_starts_min":
        int(
            genomes[
                "strict_valid_start_count"
            ].min()
        ),

    "strict_valid_starts_median":
        float(
            genomes[
                "strict_valid_start_count"
            ].median()
        ),

    "strict_valid_starts_max":
        int(
            genomes[
                "strict_valid_start_count"
            ].max()
        ),
}


# ================================================================
# Output
# ================================================================

segment_file = (
    OUT /
    "DNABERTS_STRICT32_SEGMENT_MANIFEST.tsv.gz"
)

genome_file = (
    OUT /
    "DNABERTS_STRICT32_GENOME_SUMMARY.tsv"
)

json_file = (
    OUT /
    "DNABERTS_STRICT32_AUDIT.json"
)


segments.to_csv(
    segment_file,
    sep="\t",
    index=False,
    compression="gzip",
)

genomes.to_csv(
    genome_file,
    sep="\t",
    index=False,
)


with open(
    json_file,
    "w",
) as f:

    json.dump(
        token_summary,
        f,
        indent=2,
    )


report = (
    OUT /
    "DNABERTS_STRICT32_SELECTION_REPORT.txt"
)


with open(
    report,
    "w",
) as f:

    f.write(
        "DNABERT-S Strict-ACGT "
        "32x1kb Selection Audit v1\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )

    f.write(
        f"Manifest: {MANIFEST}\n"
    )

    f.write(
        f"Manifest SHA256: "
        f"{sha256_file(MANIFEST)}\n\n"
    )

    f.write(
        f"Model: {MODEL_ID}\n"
    )

    f.write(
        f"Revision: {REVISION}\n\n"
    )

    f.write(
        "Candidate final preprocessing:\n"
    )

    f.write(
        "  uppercase input\n"
    )

    f.write(
        "  segment length = 1000 bp\n"
    )

    f.write(
        "  segments/genome = 32\n"
    )

    f.write(
        "  segments never cross contig boundaries\n"
    )

    f.write(
        "  only A/C/G/T accepted\n"
    )

    f.write(
        "  no N or other IUPAC ambiguity\n"
    )

    f.write(
        "  all strict valid window starts form "
        "the genome sampling space\n"
    )

    f.write(
        "  32 deterministic midpoint quantiles "
        "are selected from that space\n\n"
    )

    for k, v in (
        token_summary.items()
    ):

        f.write(
            f"{k}: {v}\n"
        )

    f.write(
        "\nSTATUS: AUDIT PASS\n"
    )


script_path = Path(
    __file__
).resolve()


input_script_sha = (
    OUT /
    "INPUT_AND_SCRIPT_SHA256.txt"
)


with open(
    input_script_sha,
    "w",
) as f:

    f.write(
        f"{sha256_file(MANIFEST)}  "
        f"{MANIFEST}\n"
    )

    f.write(
        f"{sha256_file(script_path)}  "
        f"{script_path}\n"
    )


result_files = [
    segment_file,
    genome_file,
    json_file,
    report,
    input_script_sha,
]


with open(
    OUT /
    "RESULT_SHA256.txt",
    "w",
) as f:

    for p in result_files:

        f.write(
            f"{sha256_file(p)}  "
            f"{p}\n"
        )


# ================================================================
# Final
# ================================================================

print()
print("=" * 96)
print(
    "FINAL STRICT32 SELECTION AUDIT"
)
print("=" * 96)

for k, v in (
    token_summary.items()
):

    print(
        f"{k}: {v}"
    )

print()

print(
    "Segment manifest:"
)

print(
    segment_file
)

print()

print(
    "STATUS = AUDIT PASS"
)
