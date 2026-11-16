# HGV: Compact Whole-Genome Embeddings for Bacterial Reference Screening

This repository accompanies the HGV V3 manuscript revision. It provides the revision analysis scripts and the complete supplementary data/code archives.

## What is included

- `supplementary/HGV_Supplementary_Data_1_PUBLIC.zip`: frozen accession manifests, source tables, figure data, external evaluation results, E12B resource records, and W13-W15 revision results.
- `supplementary/HGV_Supplementary_Code_1_PUBLIC.zip`: original implementation and analysis scripts, canonical feature matrix, frozen model checkpoints, PCA transform, saved embeddings, and revision scripts.
- `HGV_V3_revision_20260915/scripts_frontiers_windows/`: directly browsable Windows revision scripts W01-W10 and W13-W15.

## Download and extract the complete replay package

1. Download both ZIP archives from `supplementary/`.
2. Create a folder named `HGV_V3_revision_20260915` in the repository root.
3. Extract **both** archives into that same `HGV_V3_revision_20260915` folder, preserving their internal relative paths. Merge folders when Windows asks.
4. Read the `README.txt` inside each archive before running any script.
5. Run W13-W15 from the extracted project layout. These analyses reuse frozen inputs/checkpoints and do not retrain HGV.

The archives normalize historical machine-specific absolute paths to the portable project-root prefix `HGV_V3_revision_20260915/`. The W13-W15 scripts derive their project root from their own location. Some older historical scripts may require environment-specific tools or targeted path adaptation.

## Data availability

The 3,154-genome accession/version manifest and GTDB taxonomy records are supplied in Supplementary Data 1. Raw genome FASTA files are not redistributed; they should be retrieved from the accession versions in the manifest and the original source databases.

## Reproducibility scope

The supplementary code archive is an inspection and targeted-replay package. It contains frozen models, feature matrices, transforms, manifests, and selected scripts, but it does **not** claim that every historical experiment can be reconstructed in one command on a clean machine. Some external tools, raw FASTA downloads, and third-party pretrained DNABERT-S weights are not included. See the archive README files and `docs/REPRODUCIBILITY.md`.

## License

No project-wide reuse license is assigned in this release. Third-party data and software remain subject to their respective terms. The authors should add an explicit license only after confirming ownership and co-author approval.

## Release integrity

`supplementary/SHA256SUMS.txt` records SHA256 checksums for the two public supplementary archives. Each archive also contains a regenerated `FILE_INVENTORY.tsv` for every included payload file.

## Contact

For questions about the HGV implementation and supplementary materials, contact the corresponding author listed in the associated manuscript.

## Browsable implementation and frozen assets (incremental update 001)

The original code and data archives are now also exposed under [HGV_V3_revision_20260915](HGV_V3_revision_20260915/). The existing ZIP downloads remain unchanged.

- Start with the [experiment-to-code/data map](docs/HGV_REPRODUCTION_MAP.md).
- Browse the [model checkpoint inventory](docs/MODEL_CHECKPOINTS.tsv) and [code index](docs/CODE_INDEX.tsv).
- Frozen pair manifests, Dashing2-augmented training/validation pairs and protocol/configuration freezes have also been added from the original project; these inputs were missing from the replay ZIP.
- Check the [incremental update source inventory](docs/releases/update_001_source_inventory.tsv) for exact paths, provenance and SHA256 values.
- Future updates use [the Windows incremental updater](tools/hgv_incremental/README.md); only added or changed paths are committed.

Code visibility and checkpoint availability do not imply that every historical run has been repeated on a clean environment. The map identifies external dependencies, retained historical scripts and remaining gaps.
