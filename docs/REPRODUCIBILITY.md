# Reproducibility notes

## Scope

This release provides source code, frozen checkpoints, feature representations, manifests, result tables, and selected analysis scripts. It is designed for inspection and targeted replay of the reported analyses.

## Not a one-command reconstruction

The archive does not include raw genome FASTA downloads, every historical intermediate, all third-party pretrained weights, external bioinformatics tool binaries, or prebuilt environments. Some historical workflows were executed on Linux and refer to tools that are not available on a standard Windows installation.

## Current revision scripts

- W01: frozen prediction audit and species-stratified Q50 accuracy analysis.
- W08-W10: real MAG selection/download, feature extraction, and frozen inference.
- W13: equal-budget candidate retrieval and exploratory MAG summaries.
- W14: manifest-locked D0/D1/D3 checkpoint recovery and G2 retrieval.
- W15: external matched-species retrieval using G2-trained D0/D1/D3 models.

W11 and W12 were not part of the supplied final supplementary code archive and are not represented as completed analyses.

## Environment

The original Linux manuscript environment is recorded in the source tables and Code archive README. The W13-W15 Windows inference scripts require Python with NumPy, PyTorch, and scikit-learn. The exact environment should be reconstructed from the included execution records before attempting a full replay.

## Path portability

Machine-specific absolute roots were replaced with the portable prefix `HGV_V3_revision_20260915/`. Extract both supplementary archives into a folder with that name beneath the repository root. Historical scripts may still need environment-specific adaptation; do not overwrite the only copy of frozen results.

## Integrity

Use the archive-level `FILE_INVENTORY.tsv` and `supplementary/SHA256SUMS.txt` to verify files before analysis.
