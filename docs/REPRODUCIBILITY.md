# Reproduction notes

## Repository layout and access

Clone the repository or extract GitHub's repository ZIP. The `HGV_V3_revision_20260915/` tree already contains the published implementation, selected trained checkpoints, canonical feature matrix, PCA transform, split/pair manifests, teacher-augmented pairs, source tables and revision analyses. Start with [HGV_REPRODUCTION_MAP.md](HGV_REPRODUCTION_MAP.md) for the experiment-specific entry points and [MODEL_CHECKPOINTS.tsv](MODEL_CHECKPOINTS.tsv) for checkpoint identity.

The reference snapshot for the manuscript is commit `adb9202b5d4a12fd6460f71fe862ccf3650d911c`. It includes the expanded archives and 84 added original-project input/provenance files. Documentation can evolve without changing the cited snapshot.

## Supplied and external inputs

The three canonical external HGV models and fifteen D0/D1/D3 identities used in the additional five-seed comparison are represented by 18 physical checkpoint files. The first three D3 identities reuse the species-holdout main-training checkpoint paths; resolve them through the W14/W15 manifests. The frozen PCA transform and feature/embedding assets are included. The repository does not contain every internal/sensitivity/fragmentation checkpoint.

Retrieve exact raw FASTA accession versions from NCBI and the required complete GTDB R232 metadata/tree files from GTDB. External tool binaries, environments and third-party DNABERT-S weights must be obtained separately. These external requirements are described in the experiment map and original scripts.

## Environment and paths

Run the normalized original public scripts from the repository root so `HGV_V3_revision_20260915/` resolves correctly. W13-W15 derive their project root from their script location. Some older scripts and newly added original provenance/configuration records retain machine-specific paths; inspect and remap them in a separate working copy. Training or rerunning a script can write into its original output directory, so preserve the frozen public assets.

Use each experiment's recorded environment and native tool versions. Windows inference scripts use Python, NumPy, PyTorch and scikit-learn; shell/native baseline workflows require a compatible Linux/WSL setup. The repository updater is a file-management tool and does not install or run the scientific software. A single clean-environment reconstruction of every historical experiment has not been validated.

## Current additional analyses

- W01 v2: frozen-prediction accuracy resampling and rank-one failures.
- W08-W10: selected real MAG manifests/download, features and frozen inference.
- W13: equal-vector-budget candidate counts and exploratory MAG summaries.
- W14: checkpoint recovery and identity provenance.
- W15: final external five-seed D0/D1/D3 analysis, superseding W13's unfinished ablation branch.

The MAG sequence-level near-duplicate and independent taxonomy-label audits remain incomplete. The new species-stratified 50-query analysis has accuracy results but no new synchronized timings. W11/W12 are not represented as completed analyses. These statuses are retained in the manuscript and source records.

## Archives and checksums

The supplementary ZIP files are unchanged historical snapshots. The expanded repository additionally contains training inputs absent from those ZIP files; a current clone does not need the archives extracted over it. Their README and FILE_INVENTORY basenames collide, so separate copies are retained under [archive_provenance](archive_provenance/).

Use [the exposed-file source inventory](releases/update_001_source_inventory.tsv), [the checkpoint inventory](MODEL_CHECKPOINTS.tsv) and Git commit identifiers to check published assets. Use `supplementary/SHA256SUMS.txt` for the historical ZIP downloads. Earlier internal checksum files describe their historical execution states, including states before public path normalization.
