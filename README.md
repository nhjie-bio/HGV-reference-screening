# HGV reference screening

HGV maps canonical whole-assembly 6-mer profiles to normalized 64-dimensional vectors for bacterial reference candidate screening. This repository accompanies the manuscript **HGV: compact whole-genome embeddings for bacterial reference screening across taxonomic holdouts**.

## Start here

1. Clone this repository, or use GitHub's **Code → Download ZIP** and extract it once.
2. Open the [experiment-to-code/data map](docs/HGV_REPRODUCTION_MAP.md) to locate the training, evaluation and analysis entry points.
3. Read the [reproduction notes](docs/REPRODUCIBILITY.md) before running an experiment. The project tree is already expanded; a clone does not require extracting the two supplementary archives over it.

```bash
git clone https://github.com/nhjie-bio/HGV-reference-screening.git
cd HGV-reference-screening
```

The browsable project is [HGV_V3_revision_20260915/](HGV_V3_revision_20260915/). Original filenames preserve experiment provenance. Use the final entry points in the map; historical scripts are also retained where later code depends on them.

## Available materials

| Material | Location |
|---|---|
| Exact assembly accessions, taxonomy and cohort assignments | [01_manifest](HGV_V3_revision_20260915/01_manifest/) |
| Canonical features and split records | [03_features](HGV_V3_revision_20260915/03_features/), [04_splits](HGV_V3_revision_20260915/04_splits/) |
| Pair manifests, teacher-augmented pairs and training configuration | [05_pairs](HGV_V3_revision_20260915/05_pairs/), [06_teacher](HGV_V3_revision_20260915/06_teacher/), [07_hgv](HGV_V3_revision_20260915/07_hgv/) |
| HGV training and evaluation implementation | [training scripts](HGV_V3_revision_20260915/scripts_v3_20260916_hgv/), [full code index](docs/CODE_INDEX.tsv) |
| Baseline implementations | [internal comparisons](HGV_V3_revision_20260915/scripts_v3_20260917_e5_baselines/); external and finalization entry points in the experiment map |
| Frozen HGV checkpoints and PCA transform | [18-checkpoint inventory](docs/MODEL_CHECKPOINTS.tsv), [external evaluation assets](HGV_V3_revision_20260915/09_external_validation/) |
| Screening and synchronized resource records | [15_reference_screening](HGV_V3_revision_20260915/15_reference_screening/), [16_scalability](HGV_V3_revision_20260915/16_scalability/) |
| Figure source data and additional revision analyses | [18_figures](HGV_V3_revision_20260915/18_figures/), [21_frontiers_revision_20261112](HGV_V3_revision_20260915/21_frontiers_revision_20261112/) |

## Scope and external inputs

The supplied checkpoints, feature matrices, saved embeddings and source tables support inspection and targeted replay. The added pair/teacher tables and configuration freezes supply inputs for the original HGV training entry points. Full reconstruction of every historical run has not been tested on a clean installation.

Raw benchmark/MAG FASTA files, complete GTDB release downloads, third-party DNABERT-S pretrained weights, external executables and every historical checkpoint/intermediate are not redistributed. Retrieve exact assembly versions from NCBI using the supplied manifests, and obtain the documented GTDB release and third-party tools separately. Native sketch/ANI workflows require the relevant Linux/WSL environment; the Windows update tool manages repository files.

The matched-species screening results, distant-taxon retrieval results and their denominators are distinct. The MAG pilot's sequence-level near-duplicate and independent label audits remain incomplete. Species-stratified resampling supplies accuracy results, not new resource timings. See the [experiment map](docs/HGV_REPRODUCTION_MAP.md) for these boundaries.

## Version and integrity

The manuscript's public code/data snapshot is [commit adb9202b5d4a12fd6460f71fe862ccf3650d911c](https://github.com/nhjie-bio/HGV-reference-screening/tree/adb9202b5d4a12fd6460f71fe862ccf3650d911c). Later documentation updates do not change that snapshot. [The source inventory](docs/releases/update_001_source_inventory.tsv) records exposed file provenance and SHA256 values; [the model inventory](docs/MODEL_CHECKPOINTS.tsv) records checkpoint hashes.

The two [supplementary ZIP files](supplementary/) remain historical downloads with their original checksums. They contain the earlier replay package and do not include all subsequently added training inputs. Use the expanded repository for the current material collection. Their separate README/inventory files are preserved under [archive provenance](docs/archive_provenance/).

## Maintaining the repository

For prepared incremental packages, the [one-click Windows launcher](tools/hgv_incremental/README.md) checks files, applies changes, commits only listed paths and verifies the push. Future updates contain only new or changed files. User-specific local repository paths are stored in each local package, not in this repository.

## License and contact

No project-wide reuse license has been assigned; see [LICENSE_STATUS.md](LICENSE_STATUS.md). Third-party materials retain their original terms. For scientific or implementation questions, contact the corresponding author listed in the manuscript.
