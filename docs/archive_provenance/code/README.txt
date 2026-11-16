HGV Supplementary Code 1 — frozen implementation and replay assets

Extract both archives into the same directory named `HGV_V3_revision_20260915` under the repository root, preserving relative paths.
This is an inspection/replay bundle for the manuscript revision, not a claim that every original experiment
has been rerun on a clean installation. Original filenames and scripts are retained for traceability.

Included:
- Original project Python/shell/C++ scripts and configurations; historic versions are retained where needed
  to identify the exact analysis. Use the active entry points below rather than running every script.
- Three canonical external HGV checkpoints (seeds 20260915, 20260916, 20260917), with configuration records.
- Fifteen SHA256-verified D0/D1/D3 checkpoints used in the external five-seed ablation. For the first three
  selective seeds, the canonical main-training checkpoints are the source; the verified manifest resolves this.
- Canonical 3,154 x 2,080 feature matrix, row/vocabulary maps, frozen PCA transform, MAG features, and saved
  reference/query embeddings for the three-model screening analysis.

Main active revision entry points in scripts_frontiers_windows:
W01_existing_prediction_audit_v2.py: stratified accuracy and failure audit from frozen predictions.
W09_e13_mag_features_v1.py: features for selected downloaded MAG assemblies.
W10_e13_frozen_inference_v1.py: frozen MAG inference with original-embedding parity and retrieval replay gates.
W13_HGV_two_questions_oneshot_v1.py: equal-budget candidate summaries and exploratory MAG summaries;
  its incomplete ablation branch is superseded by W15.
W14_e6_manifest_fix_only_v1.py: recovered checkpoint source manifest.
W15_external_B1_G2_ablation_v1.py: current five-seed external distillation analysis (Table S3).

Existing original workflows:
scripts_v3_20260916_hgv: original encoder/training implementation.
scripts_v3_20260930_e12b_resource: synchronized resource protocol and finalization.
The scripts record the original project-relative and server paths; inspect their arguments/configuration
before running. Some original records contain absolute Linux/Windows paths and require remapping to the
extracted root. Do not execute scripts over the only copy of frozen outputs.

Environment:
The original manuscript environment records Python 3.9.23, NumPy 2.0.2, scikit-learn 1.6.1,
PyTorch 2.8.0+cu128 and an NVIDIA RTX 3060; the additional Windows scripts require NumPy, PyTorch and
scikit-learn for inference/PCA. Shell/native-tool workflows require the original external tools and a
compatible Linux environment. These requirements have not been reinstalled or performance-tested here.

Not included:
Raw genome FASTA downloads, third-party pretrained DNABERT-S weights, tool binaries/virtual environments,
every historical intermediate, and every checkpoint from the original internal/fragmentation sweeps.
Original source tables are in Supplementary Data 1. Genomes can be obtained using the accession versions
in its manifest. The code is supplied for inspection and targeted replay, not as a fully validated one-command
reconstruction of the entire study. No new training or timing values were generated during document revision.

Integrity:
FILE_INVENTORY.tsv is regenerated for this public normalized archive and gives each supplied payload file's SHA256 and size. Historical internal SHA256 files are retained as execution provenance; use the top-level inventory to verify this public package. Checkpoint hashes match W10/W15 records.
Third-party material retains its original terms. Public repository: https://github.com/nhjie-bio/HGV-reference-screening. A Zenodo DOI has not yet been assigned. No project-wide reuse licence is assigned in this release.

Public-release path normalization:
- Historical absolute Linux/Windows project paths have been normalized to the portable prefix `HGV_V3_revision_20260915/`.
- W13-W15 use paths relative to their script/project layout. Historical scripts are retained for provenance and may require environment-specific tool installation or targeted path adaptation.
- This archive is an inspection and targeted-replay package, not a claim of one-command reconstruction of every experiment.
