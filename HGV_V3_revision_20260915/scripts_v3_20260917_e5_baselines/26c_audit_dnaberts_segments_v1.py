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
    "02_segment_feasibility_audit_v1_20260918"
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

# Audit only.
# These are candidate ambiguity limits, not yet frozen.
BAD_BP_THRESHOLDS = [
    0,
    1,
    5,
    10,
    20,
    50,
    100,
]


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


def midpoint_targets(
    total_positions,
    n,
):

    # Exact deterministic midpoint quantiles:
    #
    # floor((i + 0.5) * total / n)
    #
    # written with integer arithmetic.

    return [
        (
            (2 * i + 1)
            * total_positions
        )
        //
        (2 * n)

        for i in range(n)
    ]


print("=" * 96)
print(
    "DNABERT-S 32x1kb "
    "SEGMENT FEASIBILITY AUDIT v1"
)
print("=" * 96)

print(
    "window bp    =",
    WINDOW_BP
)

print(
    "segments/genome =",
    N_SEGMENTS
)

print(
    "model        =",
    MODEL_ID
)

print(
    "revision     =",
    REVISION
)

print()


# ================================================================
# Frozen manifest
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
        f"Expected 3154 genomes, "
        f"found {len(manifest)}"
    )


if manifest[
    "assembly_accession_final"
].duplicated().any():

    raise RuntimeError(
        "Duplicate accession "
        "in frozen manifest"
    )


print(
    "manifest genomes =",
    len(manifest)
)

print(
    "manifest SHA256 =",
    sha256_file(
        MANIFEST
    )
)


# ================================================================
# Frozen tokenizer, cache only.
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


print(
    "tokenizer class =",
    tokenizer.__class__.__name__
)

print(
    "UNK token id    =",
    tokenizer.unk_token_id
)


UNK_ID = tokenizer.unk_token_id


# ================================================================
# Process genomes
# ================================================================

window_rows = []
genome_rows = []

t0 = time.time()


print()
print(
    "===== SCAN 3154 GENOMES ====="
)


for genome_index, r in enumerate(
    manifest.itertuples(
        index=False
    ),
    start=1,
):

    accession = str(
        r.assembly_accession_final
    )

    fasta_path = Path(
        str(
            r.fasta_path
        )
    )


    if not fasta_path.is_file():

        raise FileNotFoundError(
            fasta_path
        )


    all_contig_count = 0
    assembly_bp = 0

    eligible = []


    for contig_name, seq_raw in (
        fasta_records(
            fasta_path
        )
    ):

        all_contig_count += 1

        seq = seq_raw.upper()

        L = len(seq)

        assembly_bp += L


        if L >= WINDOW_BP:

            n_starts = (
                L -
                WINDOW_BP +
                1
            )

            eligible.append({
                "name":
                    contig_name,

                "seq":
                    seq,

                "length":
                    L,

                "n_starts":
                    n_starts,
            })


    eligible_contigs = len(
        eligible
    )

    total_starts = int(
        sum(
            x["n_starts"]
            for x in eligible
        )
    )


    if total_starts <= 0:

        genome_rows.append({
            "assembly_accession":
                accession,

            "assembly_bp":
                assembly_bp,

            "n_contigs":
                all_contig_count,

            "n_eligible_contigs":
                eligible_contigs,

            "eligible_window_starts":
                total_starts,

            "n_requested":
                N_SEGMENTS,

            "n_selected":
                0,

            "n_unique_windows":
                0,

            "status":
                "NO_1KB_WINDOW",
        })

        continue


    cumulative_ends = []

    running = 0

    for x in eligible:

        running += int(
            x["n_starts"]
        )

        cumulative_ends.append(
            running
        )


    targets = midpoint_targets(
        total_starts,
        N_SEGMENTS,
    )


    selected = []

    selected_keys = []


    for segment_index, global_start in (
        enumerate(
            targets,
            start=1,
        )
    ):

        contig_index = bisect_right(
            cumulative_ends,
            global_start,
        )

        if contig_index >= len(
            eligible
        ):

            raise RuntimeError(
                f"{accession}: "
                "target mapping "
                "overflow"
            )


        previous_end = (
            0
            if contig_index == 0
            else cumulative_ends[
                contig_index - 1
            ]
        )


        local_start = int(
            global_start -
            previous_end
        )


        contig = eligible[
            contig_index
        ]


        local_end = (
            local_start +
            WINDOW_BP
        )


        seq = contig["seq"][
            local_start:
            local_end
        ]


        if len(seq) != WINDOW_BP:

            raise RuntimeError(
                f"{accession}: "
                "non-1000-bp "
                "selected window"
            )


        key = (
            contig["name"],
            local_start,
            local_end,
        )

        selected_keys.append(
            key
        )


        # Uppercase sequence is already guaranteed.
        a_count = seq.count("A")
        c_count = seq.count("C")
        g_count = seq.count("G")
        t_count = seq.count("T")

        acgt_count = (
            a_count +
            c_count +
            g_count +
            t_count
        )

        non_acgt_count = (
            WINDOW_BP -
            acgt_count
        )

        n_count = seq.count("N")

        other_ambiguity_count = (
            non_acgt_count -
            n_count
        )


        selected.append({
            "assembly_accession":
                accession,

            "segment_index":
                segment_index,

            "contig":
                contig["name"],

            "contig_length":
                int(
                    contig["length"]
                ),

            "global_eligible_start":
                int(
                    global_start
                ),

            "local_start_0based":
                local_start,

            "local_end_exclusive":
                local_end,

            "window_bp":
                WINDOW_BP,

            "A_count":
                a_count,

            "C_count":
                c_count,

            "G_count":
                g_count,

            "T_count":
                t_count,

            "N_count":
                n_count,

            "other_ambiguity_count":
                other_ambiguity_count,

            "non_acgt_count":
                non_acgt_count,

            "non_acgt_fraction":
                non_acgt_count /
                WINDOW_BP,

            "_seq":
                seq,
        })


    # ------------------------------------------------------------
    # Tokenize all 32 windows together on CPU.
    # No truncation is permitted in this audit.
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


    input_ids_batch = (
        encoded[
            "input_ids"
        ]
    )


    if len(
        input_ids_batch
    ) != len(
        selected
    ):

        raise RuntimeError(
            f"{accession}: "
            "tokenizer batch "
            "size mismatch"
        )


    for x, ids in zip(
        selected,
        input_ids_batch,
    ):

        token_count = len(
            ids
        )

        if UNK_ID is None:

            unk_count = 0

        else:

            unk_count = sum(
                1
                for token_id in ids
                if token_id == UNK_ID
            )


        x["token_count"] = (
            token_count
        )

        x["unk_token_count"] = (
            int(
                unk_count
            )
        )

        x[
            "token_count_le_512"
        ] = (
            token_count <= 512
        )

        del x["_seq"]

        window_rows.append(
            x
        )


    unique_windows = len(
        set(
            selected_keys
        )
    )


    token_counts = [
        x["token_count"]
        for x in selected
    ]

    bad_counts = [
        x["non_acgt_count"]
        for x in selected
    ]


    genome_rows.append({
        "assembly_accession":
            accession,

        "assembly_bp":
            assembly_bp,

        "n_contigs":
            all_contig_count,

        "n_eligible_contigs":
            eligible_contigs,

        "eligible_window_starts":
            total_starts,

        "n_requested":
            N_SEGMENTS,

        "n_selected":
            len(
                selected
            ),

        "n_unique_windows":
            unique_windows,

        "token_count_min":
            min(
                token_counts
            ),

        "token_count_median":
            float(
                np.median(
                    token_counts
                )
            ),

        "token_count_max":
            max(
                token_counts
            ),

        "n_token_gt512":
            sum(
                x > 512
                for x in token_counts
            ),

        "non_acgt_count_min":
            min(
                bad_counts
            ),

        "non_acgt_count_median":
            float(
                np.median(
                    bad_counts
                )
            ),

        "non_acgt_count_max":
            max(
                bad_counts
            ),

        "status":
            (
                "PASS_32_UNIQUE"
                if (
                    len(
                        selected
                    ) ==
                    N_SEGMENTS
                    and
                    unique_windows ==
                    N_SEGMENTS
                )
                else
                "NONUNIQUE_OR_INCOMPLETE"
            ),
    })


    if (
        genome_index % 100 == 0
        or genome_index == len(
            manifest
        )
    ):

        elapsed = (
            time.time() -
            t0
        )

        print(
            f"[progress] "
            f"{genome_index}/"
            f"{len(manifest)} "
            f"genomes; "
            f"elapsed="
            f"{elapsed:.1f}s"
        )


# ================================================================
# DataFrames
# ================================================================

windows = pd.DataFrame(
    window_rows
)

genomes = pd.DataFrame(
    genome_rows
)


expected_windows = (
    3154 *
    N_SEGMENTS
)


print()
print(
    "===== BASIC AUDIT ====="
)

print(
    "genomes audited       =",
    len(genomes)
)

print(
    "windows expected      =",
    expected_windows
)

print(
    "windows selected      =",
    len(windows)
)

print(
    "genomes PASS_32_UNIQUE=",
    int(
        (
            genomes["status"]
            ==
            "PASS_32_UNIQUE"
        ).sum()
    )
)


# ================================================================
# Threshold feasibility
# ================================================================

threshold_rows = []


print()
print(
    "===== AMBIGUITY THRESHOLD FEASIBILITY ====="
)


for max_bad_bp in (
    BAD_BP_THRESHOLDS
):

    pass_mask = (
        (
            windows[
                "non_acgt_count"
            ]
            <= max_bad_bp
        )
        &
        (
            windows[
                "token_count"
            ]
            <= 512
        )
    )


    temp = pd.DataFrame({
        "assembly_accession":
            windows[
                "assembly_accession"
            ],

        "pass":
            pass_mask.astype(
                int
            ),
    })


    per_genome = (
        temp.groupby(
            "assembly_accession"
        )["pass"]
        .sum()
    )


    n_windows_pass = int(
        pass_mask.sum()
    )

    n_all32 = int(
        (
            per_genome ==
            32
        ).sum()
    )

    n_ge31 = int(
        (
            per_genome >=
            31
        ).sum()
    )

    n_ge30 = int(
        (
            per_genome >=
            30
        ).sum()
    )


    row = {
        "max_non_acgt_bp":
            max_bad_bp,

        "max_non_acgt_fraction":
            max_bad_bp /
            WINDOW_BP,

        "n_windows_total":
            len(windows),

        "n_windows_pass":
            n_windows_pass,

        "window_pass_fraction":
            n_windows_pass /
            len(windows),

        "n_genomes_all32_pass":
            n_all32,

        "genome_all32_fraction":
            n_all32 /
            len(genomes),

        "n_genomes_ge31_pass":
            n_ge31,

        "n_genomes_ge30_pass":
            n_ge30,

        "min_pass_windows_per_genome":
            int(
                per_genome.min()
            ),

        "median_pass_windows_per_genome":
            float(
                per_genome.median()
            ),

        "mean_pass_windows_per_genome":
            float(
                per_genome.mean()
            ),
    }


    threshold_rows.append(
        row
    )


    print(
        f"bad_bp<={max_bad_bp:3d} "
        f"({max_bad_bp/WINDOW_BP:.3%}): "
        f"windows="
        f"{n_windows_pass}/"
        f"{len(windows)} "
        f"({n_windows_pass/len(windows):.6f}); "
        f"genomes32="
        f"{n_all32}/"
        f"{len(genomes)}"
    )


threshold_df = pd.DataFrame(
    threshold_rows
)


# ================================================================
# Tokenizer-wide audit
# ================================================================

token_summary = {
    "n_windows":
        int(
            len(windows)
        ),

    "token_count_min":
        int(
            windows[
                "token_count"
            ].min()
        ),

    "token_count_median":
        float(
            windows[
                "token_count"
            ].median()
        ),

    "token_count_mean":
        float(
            windows[
                "token_count"
            ].mean()
        ),

    "token_count_p95":
        float(
            windows[
                "token_count"
            ].quantile(
                0.95
            )
        ),

    "token_count_p99":
        float(
            windows[
                "token_count"
            ].quantile(
                0.99
            )
        ),

    "token_count_max":
        int(
            windows[
                "token_count"
            ].max()
        ),

    "n_token_gt512":
        int(
            (
                windows[
                    "token_count"
                ]
                > 512
            ).sum()
        ),

    "n_windows_with_unk":
        int(
            (
                windows[
                    "unk_token_count"
                ]
                > 0
            ).sum()
        ),

    "unk_token_count_total":
        int(
            windows[
                "unk_token_count"
            ].sum()
        ),

    "non_acgt_count_max":
        int(
            windows[
                "non_acgt_count"
            ].max()
        ),

    "non_acgt_fraction_max":
        float(
            windows[
                "non_acgt_fraction"
            ].max()
        ),
}


# ================================================================
# Write outputs
# ================================================================

window_file = (
    OUT /
    "DNABERTS_WINDOW_AUDIT.tsv.gz"
)

genome_file = (
    OUT /
    "DNABERTS_GENOME_AUDIT.tsv"
)

threshold_file = (
    OUT /
    "DNABERTS_THRESHOLD_FEASIBILITY.tsv"
)

token_json = (
    OUT /
    "DNABERTS_TOKENIZER_AUDIT.json"
)


windows.to_csv(
    window_file,
    sep="\t",
    index=False,
    compression="gzip",
)

genomes.to_csv(
    genome_file,
    sep="\t",
    index=False,
)

threshold_df.to_csv(
    threshold_file,
    sep="\t",
    index=False,
)


with open(
    token_json,
    "w",
) as f:

    json.dump(
        token_summary,
        f,
        indent=2,
    )


# ================================================================
# Report
# ================================================================

report = (
    OUT /
    "DNABERTS_SEGMENT_FEASIBILITY_REPORT.txt"
)


with open(
    report,
    "w",
) as f:

    f.write(
        "DNABERT-S Segment Feasibility Audit v1\n"
    )

    f.write(
        "=" * 80 +
        "\n\n"
    )

    f.write(
        f"Frozen manifest: {MANIFEST}\n"
    )

    f.write(
        "Manifest SHA256: "
        f"{sha256_file(MANIFEST)}\n\n"
    )

    f.write(
        f"Model/tokenizer: "
        f"{MODEL_ID}\n"
    )

    f.write(
        f"Revision: "
        f"{REVISION}\n\n"
    )

    f.write(
        "Candidate sampling rule:\n"
    )

    f.write(
        "  - uppercase FASTA sequence\n"
    )

    f.write(
        "  - 1000-bp windows only\n"
    )

    f.write(
        "  - windows never cross contig boundaries\n"
    )

    f.write(
        "  - contigs shorter than 1000 bp "
        "cannot provide windows\n"
    )

    f.write(
        "  - all legal window-start coordinates "
        "are pooled conceptually across contigs\n"
    )

    f.write(
        "  - 32 deterministic midpoint quantiles "
        "are selected in that eligible coordinate\n\n"
    )

    f.write(
        f"Genomes audited: "
        f"{len(genomes)}\n"
    )

    f.write(
        f"Expected windows: "
        f"{expected_windows}\n"
    )

    f.write(
        f"Selected windows: "
        f"{len(windows)}\n"
    )

    f.write(
        "Genomes with 32 unique windows: "
        f"{int((genomes['status']=='PASS_32_UNIQUE').sum())}\n\n"
    )

    f.write(
        "Tokenizer audit:\n"
    )

    for k, v in (
        token_summary.items()
    ):

        f.write(
            f"  {k}: {v}\n"
        )

    f.write(
        "\nAmbiguity threshold feasibility:\n"
    )

    f.write(
        threshold_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSTATUS: AUDIT COMPLETE\n"
    )


# ================================================================
# Provenance
# ================================================================

script_path = Path(
    __file__
).resolve()


with open(
    OUT /
    "INPUT_AND_SCRIPT_SHA256.txt",
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
    genome_file,
    threshold_file,
    token_json,
    report,
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
# Final console output
# ================================================================

print()
print("=" * 96)
print(
    "FINAL DNABERT-S SEGMENT AUDIT"
)
print("=" * 96)

print(
    "Genomes:",
    len(genomes)
)

print(
    "Windows:",
    len(windows),
    "/",
    expected_windows
)

print(
    "32 unique windows:",
    int(
        (
            genomes["status"]
            ==
            "PASS_32_UNIQUE"
        ).sum()
    ),
    "/",
    len(genomes)
)

print()

print(
    "TOKENIZER:"
)

for k, v in (
    token_summary.items()
):

    print(
        f"{k}: {v}"
    )

print()

print(
    "THRESHOLD FEASIBILITY:"
)

print(
    threshold_df.to_string(
        index=False
    )
)

print()
print(
    "Report:"
)
print(report)

print()
print(
    "STATUS = AUDIT COMPLETE"
)
