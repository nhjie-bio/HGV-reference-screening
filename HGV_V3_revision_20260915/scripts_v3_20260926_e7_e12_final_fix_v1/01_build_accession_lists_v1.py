#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(
    "HGV_V3_revision_20260915"
)

FIX = BASE / "scripts_v3_20260926_e7_e12_final_fix_v1"

sys.path.insert(
    0,
    str(FIX)
)

os.environ["HGV_BASE"] = str(BASE)
os.environ["HGV_RUN_TAG"] = "20260925_v6"

import common as C

manifest = C.load_manifest()
mc = C.manifest_columns(manifest)
fmap = C.get_fasta_map(manifest)

meta = manifest.set_index(
    mc["accession"]
)

groups = {
    "A": [],
    "B1": [],
    "B2_seen_genus": [],
    "B2_unseen_genus": [],
}

for accession, row in meta.iterrows():

    acc = str(accession)

    cohort = C.infer_cohort(
        row,
        mc,
        accession=acc,
    )

    if cohort in groups:
        groups[cohort].append(acc)

for k in groups:
    groups[k] = sorted(groups[k])

assert len(groups["A"]) == 2374, len(groups["A"])
assert len(groups["B1"]) == 391, len(groups["B1"])

out = FIX / "lists"
out.mkdir(
    parents=True,
    exist_ok=True,
)

def write_list(path, accessions):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for acc in accessions:

            p = Path(
                fmap[acc]
            )

            if not p.is_file():
                raise SystemExit(
                    f"Missing FASTA: {acc} {p}"
                )

            f.write(
                f"{acc}\t{p}\n"
            )

write_list(
    out / "A2374_fasta.tsv",
    groups["A"],
)

write_list(
    out / "B1_Q50_fasta.tsv",
    groups["B1"][:50],
)

write_list(
    out / "ALL3154_fasta.tsv",
    sorted(
        sum(
            groups.values(),
            [],
        )
    ),
)

pd.DataFrame({
    "cohort": [
        "A",
        "B1",
        "B2_seen_genus",
        "B2_unseen_genus",
    ],
    "n": [
        len(groups["A"]),
        len(groups["B1"]),
        len(groups["B2_seen_genus"]),
        len(groups["B2_unseen_genus"]),
    ]
}).to_csv(
    out / "COHORT_COUNTS.tsv",
    sep="\t",
    index=False,
)

print(
    "ACCESSION_LISTS=PASS"
)
