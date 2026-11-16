HGV Supplementary Data 1 — source records for the November 2026 revision

This archive preserves original project-relative paths. The manuscript uses descriptive cohort names;
the original filenames retain execution labels to preserve provenance.

1. 01_manifest/final_3154_freeze_v1_20260916: all 3,154 assembly accession versions, cohort membership,
   taxonomy and frozen original-benchmark QC records.
2. 18_figures/HGV_V3_manuscript_figure_package_v1_20260926/source_data: original figure and analysis
   source tables, including internal, external, ablation, sensitivity, interpretation and screening results.
3. 16_scalability/.../finalize_v3_20261001 and 05_final_freeze_v3_20261001: final synchronized resource
   summaries and provenance. The manuscript deliberately omits host-memory measurements with a mixed
   PSS/RSS basis; their presence in an archived source file does not make them comparable.
4. 21_frontiers_revision_20261112/02_E14_Q50_stratified/01_manifest: new species-stratified 50-query manifest.
5. 21_frontiers_revision_20261112/90_windows_minimal:
   W01_v2 — frozen accuracy resampling and rank-one failures (Tables S4–S5).
   W08_E13_MAG_FASTA — 80 MAG accession/metadata records, without redistribution of downloaded FASTA.
   W09 and W10 — MAG feature/retrieval provenance and completeness-stratified outcomes (Table S2).
   W13 — equal-budget candidate counts and per-query ranks (Table S1), plus MAG endpoints.
   W14 — checkpoint recovery provenance, retained to explain checkpoint identity.
   W15 — latest five-seed external ablation, superseding the incomplete W13 ablation branch (Table S3).

Status and interpretation:
- W13 E11 candidate-budget results are complete; its incomplete E6 branch is not the final ablation source.
- W15 is the current external distillation source (254 eligible queries, 1,662 references, 26 species).
- MAG near-duplicate ANI screening and independent taxonomy-label audit were not completed.
- Stratified 50-query files report accuracy only, not new synchronized runtime measurements.
- Recall must be separated into species, genus and family endpoints; no pooled 80-MAG accuracy is claimed.
- Bootstrap/permutation details and denominators are specified in the manuscript and Supplementary Information.

FILE_INVENTORY.tsv is regenerated for this public normalized archive and records every supplied payload file, size and SHA256. Historical internal SHA256 freeze files are retained as execution provenance; use the top-level FILE_INVENTORY.tsv to verify the exact public archive. Public repository: https://github.com/nhjie-bio/HGV-reference-screening. A Zenodo DOI has not yet been assigned.

Public-release path normalization:
- Historical absolute project paths have been replaced with the portable project-root prefix `HGV_V3_revision_20260915/`.
- Extract this archive into a directory named `HGV_V3_revision_20260915` under the repository root, preserving all relative paths.
- Original accession identifiers, numerical results, model hashes, and result tables are retained.
- Raw genome FASTA files are not redistributed; retrieve them from the accession versions in the manifest.
