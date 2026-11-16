# HGV: experiment, implementation and asset map

All paths in the table are relative to [`HGV_V3_revision_20260915/`](../HGV_V3_revision_20260915/). Original names preserve execution provenance; the manuscript uses descriptive cohort/endpoint names. The exposed archive files are byte-identical to the already published PUBLIC ZIP files. Added original training inputs are identified separately in [the source inventory](releases/update_001_source_inventory.tsv).

| Task / manuscript evidence | Implementation or entry point | Inputs / frozen evidence |
|---|---|---|
| GTDB release and cohort design | `scripts/00_fetch_gtdb_r232_metadata.sh`, `01_inspect_gtdb_metadata.py`, `02_build_candidate_manifest.py`, `03_audit_independence_and_design.py`, `04_select_cohorts.py` | `00_metadata/` release records; `01_manifest/final_3154_freeze_v1_20260916/` accession versions, taxonomy and cohort assignments |
| Canonical reverse-complement 6-mer map | `scripts_v3_20260916_features/10_canonical_k6_extract_v1.cpp`, `11_build_canonical_k6_cache_v1.py` | `03_features/canonical_k6_full3154_v1/`: 3,154 x 2,080 float32 matrix and row/vocabulary maps |
| Random / species / genus / family splits | `scripts_v3_20260916_splits/` | `04_splits/g1_g4_v1_20260916/` |
| Pair sampling and teacher targets | `scripts_v3_20260916_pairs/`, `scripts_v3_20260916_teacher/` | Added `05_pairs/pair_manifests_v1_20260916/`; `06_teacher/teacher_geometry_audit_v1_20260916/teacher_augmented/`; G1 targets in `06_teacher/g1_teacher_augmentation_v2_20260917/` |
| Original HGV training | `scripts_v3_20260916_hgv/17b_train_main_hierarchical_v1.py` and `18b_train_canonical_A_g1_v1.py` | Added `07_hgv/final_hgv_config_freeze_v1_20260916/` and main/external protocol freeze files; checkpoints/configs in `08_main_generalization/` and `09_external_validation/` |
| Validation thresholds and blind evaluation | `scripts_v3_20260916_hgv/17c1_freeze_validation_thresholds_v1.py`, `17c2_blind_test_evaluation_v1.py`, `18c_freeze_canonical_validation_thresholds_v1.py`, `18d_external_one_shot_evaluation_v1.py` | Frozen splits, pairs, features; `09_external_validation/canonical_validation_threshold_freeze_v1_20260917/PCA64_G1_TRAIN_FROZEN.npz` |
| Internal 11-method comparison (Table 2 / Fig. 2) | `scripts_v3_20260917_e5_baselines/` including FastANI, skani, Mash/Dashing2 joins and DNABERT-S strict32 sampling/inference | Final manuscript tables under `18_figures/HGV_V3_manuscript_figure_package_v1_20260926/source_data/`; source filenames retain E5 labels |
| Cross-method external comparison (Table 3 / Fig. 3) | `scripts_v3_20260920_e5_external/`, `scripts_v3_20260922_e5_external_dashing_fix/` | Final external 11-method tables under the same figure source-data directory; use final repaired Dashing2 records |
| D0-D5 training and evaluation (Table 4 / Fig. 4) | `scripts_v3_20260922_e6_final/03_train_e6_d0_d5_from_17b_v1.py`, `scripts_v3_20260923_e6_evaluation/`; summaries `scripts_v3_20260926_remaining_all_v9/10_*`, `11_*`, `12_*` | Frozen ablation results in figure source data; selected checkpoints in `10_ablation/` and main-training directory |
| Sensitivity / fragmentation (Table 5 / Fig. 5) | Final k5/k7 and validation fixes in `scripts_v3_20260926_e7_e12_final_fix_v1/`; fragmentation `scripts_v3_20260926_remaining_all_v9/16_run_e8_fragmentation_v1.py` | Final e7/e8 tables in figure source data; full sensitivity/fragmentation checkpoint sweeps are not supplied |
| Attribution / biological interpretation (Fig. 6) | `scripts_v3_20260926_remaining_all_v9/17_run_e9_explainability_v1.py`, `18_run_e10_biological_fidelity_v1.py`; tree closure `scripts_v3_20260926_tree_final_closure_v3/` | e9/e10/confound/tree results in figure source data |
| Three-model screening (Section 2.7 / Fig. 7) | `scripts_v3_20260926_remaining_all_v9/13_run_e11_reference_screening_v1.py` | `15_reference_screening/` frozen embeddings, ranks and provenance |
| Single-checkpoint scalability and synchronized resources (Tables 6–7 / Fig. 7) | `scripts_v3_20260926_e7_e12_final_fix_v1/08_run_e12_gzip_real_fasta_v2.py`; `scripts_v3_20260930_e12b_resource/` including bounded-cache DNABERT-S v3 and `e12b_finalize_v3.py` | `16_scalability/E12B_synchronized_resource_v1_20260930/04_summary/finalize_v3_20261001/` and `05_final_freeze_v3_20261001/` |
| Revision Tables S1-S5 | `scripts_frontiers_windows/W01_existing_prediction_audit_v2.py`, W08-W10, W13-W15 | `21_frontiers_revision_20261112/`; W15 supersedes the unfinished W13 external-ablation branch; W13 candidate-budget analysis remains current |

## Checkpoints and what they support

[MODEL_CHECKPOINTS.tsv](MODEL_CHECKPOINTS.tsv) lists 18 physical `.pt` files with exact SHA256 values: three canonical external models and fifteen D0/D1/D3 model identities for the five-seed external ablation. The first three D3 identities use the species-holdout main-training checkpoint paths; they are not duplicate files under D3. Keep those paths and use the W14/W15 manifests to resolve identities. No weights were retrained, changed, merged, or inferred for this update. Not every checkpoint from all original internal, sensitivity or fragmentation runs is included.

The frozen PCA transform is supplied alongside the HGV checkpoints. The 64-D single-model storage comparison is distinct from the three-model ensemble's storage. Public availability does not change reported performance or outstanding analysis status.

## Version selection

Historical script versions are retained because later scripts sometimes import or patch earlier ones. They are not all active entry points. Prefer the final sources identified above over a global 'largest version number' rule: the final gzip-aware sensitivity fix supersedes the earlier sensitivity runner; W15 supersedes only W13's unfinished external-ablation branch. Archived source files can also describe failed or superseded runs; their presence is provenance, not evidence of completion.

## Before replay or training

Run original public scripts from the repository root, where the normalized relative `HGV_V3_revision_20260915/` prefix resolves correctly. W13-W15 derive their paths from the script location. Older scripts and the newly supplied original provenance records may contain absolute machine paths. Inspect and remap those paths in a separate working copy before replay; archive normalization is not a guarantee that every output/configuration path is portable. Training scripts can write to the original output layout, so keep the public frozen tree intact and use a separate output/project copy.

The added pair and teacher tables close specific input gaps in the two original HGV training entry points. This update verifies packaging and checksums, not clean-environment execution of the complete training/baseline pipeline. Native sketch/ANI and shell workflows need a compatible Linux/WSL environment and installed tool versions; the Windows updater only manages repository files. Installing it does not make those bioinformatics workflows native Windows applications.

Original environment records are retained in the archives and source tables. They describe historical runs, not a newly tested unified environment lockfile. Use the relevant experiment's record when rebuilding its environment. Python/NumPy/PyTorch/scikit-learn requirements differ from native tools and the DNABERT-S environment.

## Downloaded data and remaining gaps

- Raw FASTA for the 3,154 benchmark assemblies and selected MAGs is not redistributed. Use exact accession versions and cohort records from the manifests to retrieve it from NCBI. Do not substitute newer assembly versions silently.
- Full GTDB R232 metadata (`bac120_metadata_r232.tsv.gz`, approximately 276 MiB locally), taxonomy and trees remain external downloads. The original GTDB download script records the release URL; `00_metadata/` contains release descriptions and checksum records. The selected cohort taxonomy is already in the supplied manifest.
- Third-party DNABERT-S pretrained weights, external executable binaries, environments, large native sketch caches and every historical intermediate are not included. Baseline construction scripts are supplied; required third-party materials must be obtained separately under their own terms.
- The MAG near-duplicate ANI screen and independent taxonomy-label audit remain incomplete. The stratified 50-query analysis reports accuracy, not a newly timed synchronized resource run. Public packaging does not mark these analyses as completed.
- Resource source records include historical fields omitted from the manuscript because their measurement bases differed. Use the final manuscript's comparable columns rather than treating every source-table memory field as a directly comparable metric.

The repository has no project-wide reuse license assigned; this update does not assign one. Original public ZIP files and their checksum file remain unchanged. New source inventories describe the exposed bytes and added inputs; historical internal checksums remain historical provenance, including where earlier public-path normalization changed textual files.
