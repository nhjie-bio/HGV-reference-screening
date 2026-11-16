#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGV-V3 E6-E12 scientific result audit (READ-ONLY)

This script does NOT retrain models, retune thresholds, modify frozen outputs,
or call external tools. It reads the accepted final E6-E12 output directories
and creates a new compact audit package under:

  <BASE>/20_rebuttal/HGV_V3_scientific_results_audit_v1_<timestamp>/

Outputs:
  FINAL_SCIENTIFIC_RESULTS.md
  SOURCE_DATA_INDEX.tsv
  CLAIM_AND_SCOPE_AUDIT.tsv
  MANUSCRIPT_RESULT_MAP.tsv
  E6_PRIMARY_CONTRASTS.tsv
  E7_SENSITIVITY_WITH_DELTAS.tsv
  E8_FRAGMENTATION_WITH_DELTAS.tsv
  E9_UNDEFINED_CORRELATIONS.tsv
  E9_TOP20_IG.tsv
  E9_TOP20_PERMUTATION.tsv
  E10_TREE_STATUS.txt
  E12_SCOPE_AUDIT.tsv
  AUDIT_PASS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE = Path("HGV_V3_revision_20260915")
FINAL_TAG = "20260925_v6"

E6ROOT_REL = Path(
    "10_ablation/E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

DIRS = {
    "E6_MECHANISM": E6ROOT_REL / f"07_mechanism_geometry_v1_{FINAL_TAG}",
    "E6_STATISTICS": Path("17_statistics") / f"E6_species_block_bootstrap_v1_{FINAL_TAG}",
    "E6_FINAL": E6ROOT_REL / f"08_final_freeze_v1_{FINAL_TAG}",
    "E7": Path("11_sensitivity") / f"E7_sensitivity_v1_{FINAL_TAG}",
    "E8": Path("12_fragmentation") / f"E8_fragmentation_v1_{FINAL_TAG}",
    "E9": Path("13_explainability") / f"E9_explainability_v1_{FINAL_TAG}",
    "E10": Path("14_biological_validation") / f"E10_biological_fidelity_v1_{FINAL_TAG}",
    "E11": Path("15_reference_screening") / f"E11_reference_screening_v1_{FINAL_TAG}",
    "E12": Path("16_scalability") / f"E12_scalability_v1_{FINAL_TAG}",
    "GLOBAL": Path("20_rebuttal") / f"HGV_V3_remaining_completion_freeze_v1_{FINAL_TAG}",
}

PASS_MARKERS = {
    "E6_MECHANISM": "E6_MECHANISM_GEOMETRY_PASS.txt",
    "E6_STATISTICS": "E6_SPECIES_BLOCK_STATS_PASS.txt",
    "E6_FINAL": "E6_FINAL_FREEZE_PASS.txt",
    "E7": "E7_SENSITIVITY_PASS.txt",
    "E8": "E8_FRAGMENTATION_PASS.txt",
    "E9": "E9_EXPLAINABILITY_PASS.txt",
    "E10": "E10_BIOLOGICAL_FIDELITY_PASS.txt",
    "E11": "E11_REFERENCE_SCREENING_PASS.txt",
    "E12": "E12_SCALABILITY_PASS.txt",
    "GLOBAL": "HGV_V3_REMAINING_COMPLETION_PASS.txt",
}

EXPECTED_TABLES = {
    "E6_MECHANISM": [
        "E6_MODE_METRIC_SUMMARY.tsv",
        "E6_TRUE_TEACHER_FIDELITY_SUMMARY.tsv",
        "E6_D3_PAIRED_GEOMETRY_CONTRASTS.tsv",
        "E6_D3_VS_D4_SHUFFLED_CONTROL.tsv",
        "E6_EMBEDDING_NORM_AUDIT_COPY.tsv",
    ],
    "E6_STATISTICS": [
        "E6_SEED_LEVEL_TEST_METRICS.tsv",
        "E6_D3_PAIRED_CONTRASTS_HOLM.tsv",
        "E6_SECONDARY_PAIRED_SEED_DELTAS.tsv",
    ],
    "E6_FINAL": [
        "E6_CLAIM_DECISION_TABLE.tsv",
    ],
    "E7": [
        "E7_RUN_METRICS.tsv",
        "E7_SENSITIVITY_SUMMARY.tsv",
    ],
    "E8": [
        "E8_REALIZED_SEQUENCE_AMOUNT.tsv",
        "E8_RUN_METRICS.tsv",
        "E8_FRAGMENTATION_SUMMARY.tsv",
    ],
    "E9": [
        "E9_CONFOUND_CORRELATIONS.tsv",
        "E9_KMER_INTEGRATED_GRADIENT_RANKING.tsv",
        "E9_TOP50_KMER_PERMUTATION_IMPORTANCE.tsv",
    ],
    "E10": [
        "E10_CORRELATION_SUMMARY.tsv",
        "E10_ANI_RANGE_FIDELITY.tsv",
        "E10_TAXONOMIC_DISTANCE_CONSISTENCY.tsv",
        "E10_TREE_COMPONENT_STATUS.txt",
    ],
    "E11": [
        "E11_RETRIEVAL_SUMMARY.tsv",
        "E11_CONFIRMATION_COVERAGE_BY_COHORT.tsv",
        "E11_CONFIRMATION_COVERAGE_OVERALL.tsv",
    ],
    "E12": [
        "E12_REAL_FASTA_SUMMARY.tsv",
        "E12_VECTOR_MICROBENCHMARK.tsv",
    ],
}

MANUSCRIPT_MAP = [
    ("E6", "Distillation ablation", "Main/Supplement", "D0-D5 mode metrics + D3 paired AUROC/AUPRC contrasts + geometry", "R1-5; R3-4; R3-5"),
    ("E7", "Sensitivity analysis", "Supplement/Main concise", "OFAT k, dimension, margin, lambda stability", "R1-5"),
    ("E8", "Fragmentation robustness", "Main or Supplement figure", "Full vs 100kb/50kb/20kb/5kb/5x5kb degradation", "R1-5"),
    ("E9", "Explainability", "Supplement", "IG + top-50 permutation + GC/length/contigs/N50/completeness confound audit", "R1-4"),
    ("E10", "Biological fidelity", "Main/Supplement", "HGV distance vs ANI/AF + ANI range + taxonomic strata; no phylogeny claim", "R1-3; R3-5"),
    ("E11", "Reference screening", "Main figure/table", "External retrieval, Recall@K/MRR, candidate reduction, ANI confirmation coverage", "R1-3"),
    ("E12", "Scalability", "Main/Supplement", "Real FASTA HGV pipeline + vector microbenchmark; baseline timing scope must be audited separately", "R1-7"),
]


def die(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(base: Path, p: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"Missing expected table: {path}")
    return pd.read_csv(path, sep="\t")


def format_value(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(float(x)):
            return "NA"
        ax = abs(float(x))
        if ax != 0 and ax < 1e-4:
            return f"{float(x):.3e}"
        return f"{float(x):.6f}"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    return str(x)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    cols = [str(c) for c in x.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in x.iterrows():
        vals = [format_value(row[c]).replace("|", "\\|").replace("\n", " ") for c in x.columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = ap.parse_args()

    base = args.base.resolve()
    if not base.is_dir():
        die(f"BASE not found: {base}")

    now = datetime.now().astimezone()
    out = base / "20_rebuttal" / f"HGV_V3_scientific_results_audit_v1_{now.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)

    resolved: Dict[str, Path] = {k: base / v for k, v in DIRS.items()}

    # ------------------------------------------------------------------
    # Gate 1: accepted final directories + PASS markers.
    # ------------------------------------------------------------------
    source_index_rows: List[dict] = []
    for block, d in resolved.items():
        if not d.is_dir():
            die(f"Accepted final directory missing for {block}: {d}")
        marker = d / PASS_MARKERS[block]
        if not marker.is_file():
            die(f"PASS marker missing for {block}: {marker}")
        source_index_rows.append({
            "block": block,
            "kind": "PASS_MARKER",
            "path": rel(base, marker),
            "rows": "",
            "columns": "",
            "size_bytes": marker.stat().st_size,
            "sha256": sha256_file(marker),
            "mtime": datetime.fromtimestamp(marker.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        })

    # ------------------------------------------------------------------
    # Gate 2: expected compact result tables.
    # ------------------------------------------------------------------
    table_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for block, names in EXPECTED_TABLES.items():
        d = resolved[block]
        for name in names:
            p = d / name
            if not p.exists():
                # Embedding norm copy is useful but can be absent if source wasn't brought forward.
                if block == "E6_MECHANISM" and name == "E6_EMBEDDING_NORM_AUDIT_COPY.tsv":
                    continue
                die(f"Missing expected result artifact: {p}")
            if p.suffix.lower() == ".tsv":
                df = read_tsv(p)
                table_cache[(block, name)] = df
                nrow, ncol = df.shape
            else:
                nrow, ncol = "", ""
            source_index_rows.append({
                "block": block,
                "kind": "RESULT_TABLE" if p.suffix.lower() == ".tsv" else "REPORT",
                "path": rel(base, p),
                "rows": nrow,
                "columns": ncol,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            })

    source_index = pd.DataFrame(source_index_rows)
    source_index.to_csv(out / "SOURCE_DATA_INDEX.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------
    # E6
    # ------------------------------------------------------------------
    e6_mode = table_cache[("E6_MECHANISM", "E6_MODE_METRIC_SUMMARY.tsv")].copy()
    e6_teacher = table_cache[("E6_MECHANISM", "E6_TRUE_TEACHER_FIDELITY_SUMMARY.tsv")].copy()
    e6_primary = table_cache[("E6_STATISTICS", "E6_D3_PAIRED_CONTRASTS_HOLM.tsv")].copy()
    e6_claim = table_cache[("E6_FINAL", "E6_CLAIM_DECISION_TABLE.tsv")].copy()
    write_tsv(out / "E6_PRIMARY_CONTRASTS.tsv", e6_primary)

    strict_all = False
    if "supports_strict_improvement_claim" in e6_claim.columns and len(e6_claim):
        vals = e6_claim["supports_strict_improvement_claim"].astype(str).str.lower()
        strict_all = bool(vals.isin(["true", "1", "yes"]).all())

    # ------------------------------------------------------------------
    # E7: calculate deltas relative to frozen canonical baseline.
    # ------------------------------------------------------------------
    e7 = table_cache[("E7", "E7_SENSITIVITY_SUMMARY.tsv")].copy()
    baseline_name = "canonical_k6_dim64_margin1_lambda0p25"
    if "setting" not in e7.columns or baseline_name not in set(e7["setting"].astype(str)):
        die("E7 canonical frozen baseline row not found")
    b = e7[e7["setting"].astype(str) == baseline_name].iloc[0]

    e7_delta = e7.copy()
    for metric in ["test_AUROC", "test_AUPRC", "test_MCC", "test_F1", "test_BalancedAccuracy"]:
        c = metric + "_mean"
        if c in e7_delta.columns:
            e7_delta[f"delta_{metric}_vs_canonical"] = pd.to_numeric(e7_delta[c], errors="coerce") - float(b[c])
    write_tsv(out / "E7_SENSITIVITY_WITH_DELTAS.tsv", e7_delta)

    # ------------------------------------------------------------------
    # E8: calculate deltas relative to full assembly.
    # ------------------------------------------------------------------
    e8 = table_cache[("E8", "E8_FRAGMENTATION_SUMMARY.tsv")].copy()
    if "condition" not in e8.columns or "full" not in set(e8["condition"].astype(str)):
        die("E8 full-assembly baseline row not found")
    eb = e8[e8["condition"].astype(str) == "full"].iloc[0]
    e8_delta = e8.copy()
    for metric in ["AUROC", "AUPRC", "MCC", "F1", "BalancedAccuracy"]:
        c = metric + "_mean"
        if c in e8_delta.columns:
            e8_delta[f"delta_{metric}_vs_full"] = pd.to_numeric(e8_delta[c], errors="coerce") - float(eb[c])
    write_tsv(out / "E8_FRAGMENTATION_WITH_DELTAS.tsv", e8_delta)

    # ------------------------------------------------------------------
    # E9: undefined correlation rows + top features.
    # ------------------------------------------------------------------
    e9_corr = table_cache[("E9", "E9_CONFOUND_CORRELATIONS.tsv")].copy()
    e9_ig = table_cache[("E9", "E9_KMER_INTEGRATED_GRADIENT_RANKING.tsv")].copy()
    e9_perm = table_cache[("E9", "E9_TOP50_KMER_PERMUTATION_IMPORTANCE.tsv")].copy()

    if "spearman_r" in e9_corr.columns:
        undef = e9_corr[pd.to_numeric(e9_corr["spearman_r"], errors="coerce").isna()].copy()
    else:
        undef = pd.DataFrame()
    write_tsv(out / "E9_UNDEFINED_CORRELATIONS.tsv", undef)

    if "rank" in e9_ig.columns:
        e9_ig_top20 = e9_ig.sort_values("rank").head(20).copy()
    else:
        e9_ig_top20 = e9_ig.head(20).copy()
    write_tsv(out / "E9_TOP20_IG.tsv", e9_ig_top20)

    if "AUROC_drop" in e9_perm.columns:
        e9_perm_top20 = e9_perm.sort_values("AUROC_drop", ascending=False).head(20).copy()
    else:
        e9_perm_top20 = e9_perm.head(20).copy()
    write_tsv(out / "E9_TOP20_PERMUTATION.tsv", e9_perm_top20)

    # ------------------------------------------------------------------
    # E10.
    # ------------------------------------------------------------------
    e10_corr = table_cache[("E10", "E10_CORRELATION_SUMMARY.tsv")].copy()
    e10_range = table_cache[("E10", "E10_ANI_RANGE_FIDELITY.tsv")].copy()
    e10_tax = table_cache[("E10", "E10_TAXONOMIC_DISTANCE_CONSISTENCY.tsv")].copy()
    tree_status_path = resolved["E10"] / "E10_TREE_COMPONENT_STATUS.txt"
    tree_status = tree_status_path.read_text(encoding="utf-8", errors="replace").strip()
    (out / "E10_TREE_STATUS.txt").write_text(tree_status + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # E11.
    # ------------------------------------------------------------------
    e11_ret = table_cache[("E11", "E11_RETRIEVAL_SUMMARY.tsv")].copy()
    e11_cov_cohort = table_cache[("E11", "E11_CONFIRMATION_COVERAGE_BY_COHORT.tsv")].copy()
    e11_cov_overall = table_cache[("E11", "E11_CONFIRMATION_COVERAGE_OVERALL.tsv")].copy()

    # ------------------------------------------------------------------
    # E12.
    # ------------------------------------------------------------------
    e12_real = table_cache[("E12", "E12_REAL_FASTA_SUMMARY.tsv")].copy()
    e12_vec = table_cache[("E12", "E12_VECTOR_MICROBENCHMARK.tsv")].copy()

    e12_scope = pd.DataFrame([
        {
            "component": "HGV real FASTA canonical-k6 feature extraction",
            "present_in_E12_v1": True,
            "evidence": "E12_REAL_FASTA_END_TO_END.tsv / E12_REAL_FASTA_SUMMARY.tsv",
            "interpretation": "real frozen FASTA, not cached-vector-only timing",
        },
        {
            "component": "HGV model inference",
            "present_in_E12_v1": True,
            "evidence": "reference_model_inference_wall_s / query_model_inference_wall_s",
            "interpretation": "reported separately from feature extraction",
        },
        {
            "component": "HGV exact 64-D scan",
            "present_in_E12_v1": True,
            "evidence": "exact_scan_cold_s / exact_scan_warm_median_s",
            "interpretation": "exact O(Nd), d=64",
        },
        {
            "component": "vector index size / bytes per genome / process RSS",
            "present_in_E12_v1": True,
            "evidence": "vector_index_bytes / bytes_per_genome / process_maxrss_kb",
            "interpretation": "HGV benchmark process",
        },
        {
            "component": "10^4-10^6 vector-search microbenchmark",
            "present_in_E12_v1": True,
            "evidence": "E12_VECTOR_MICROBENCHMARK.tsv",
            "interpretation": "synthetic tiling of real HGV vectors; NOT million-real-genome end-to-end",
        },
        {
            "component": "Mash/Dashing2/skani/FastANI matched end-to-end timing in E12 v1",
            "present_in_E12_v1": False,
            "evidence": "not produced by 14_run_e12_scalability_v1.py",
            "interpretation": "must be sourced from an already-frozen E5/runtime block if available; do not claim E12 v1 alone provides this comparison",
        },
    ])
    write_tsv(out / "E12_SCOPE_AUDIT.tsv", e12_scope)

    # ------------------------------------------------------------------
    # Claim and scope audit.
    # ------------------------------------------------------------------
    claim_rows = [
        {
            "topic": "E6 universal strict improvement",
            "status": "NOT_SUPPORTED" if not strict_all else "SUPPORTED",
            "evidence": "E6_CLAIM_DECISION_TABLE.tsv",
            "allowed_wording": "Report comparator-specific deltas/CIs/Holm p-values; use geometry/mechanism wording where strict improvement is unsupported.",
        },
        {
            "topic": "E7 sensitivity",
            "status": "EXECUTED",
            "evidence": "E7_SENSITIVITY_SUMMARY.tsv",
            "allowed_wording": "OFAT stability only; do not describe as full hyperparameter-grid search.",
        },
        {
            "topic": "E8 fragmentation",
            "status": "EXECUTED",
            "evidence": "E8_FRAGMENTATION_SUMMARY.tsv",
            "allowed_wording": "Frozen full-assembly models tested on reduced sequence amount; no fragmentation-specific retraining or threshold retuning.",
        },
        {
            "topic": "E9 attribution causality",
            "status": "DESCRIPTIVE_ONLY",
            "evidence": "E9 IG/permutation/confound tables",
            "allowed_wording": "Model sensitivity/association, not biological causality. Attribution uses frozen G2 D3 seed20260915.",
        },
        {
            "topic": "E9 undefined correlations",
            "status": f"{len(undef)}_UNDEFINED_ROWS",
            "evidence": "E9_UNDEFINED_CORRELATIONS.tsv",
            "allowed_wording": "Undefined when an input is constant; do not impute 0.",
        },
        {
            "topic": "E10 phylogeny/tree reconstruction",
            "status": "NO_TREE_CLAIM",
            "evidence": "E10_TREE_COMPONENT_STATUS.txt",
            "allowed_wording": "ANI/AF and taxonomic consistency only unless a separate frozen tree analysis is later justified.",
        },
        {
            "topic": "E11 ANI tool no-hit semantics",
            "status": "REPORT_COVERAGE",
            "evidence": "E11_CONFIRMATION_COVERAGE_OVERALL.tsv",
            "allowed_wording": "No-hit is structural missing under tool defaults; do not impute ANI=0.",
        },
        {
            "topic": "E12 million-genome evidence",
            "status": "VECTOR_MICROBENCHMARK_ONLY",
            "evidence": "E12_VECTOR_MICROBENCHMARK.tsv",
            "allowed_wording": "10^4-10^6 vector-search microbenchmark; not million-real-genome end-to-end.",
        },
        {
            "topic": "E12 fair native-baseline runtime comparison",
            "status": "NOT_IN_E12_V1",
            "evidence": "E12_SCOPE_AUDIT.tsv",
            "allowed_wording": "Do not claim E12 v1 alone benchmarks Mash/Dashing2/skani/FastANI end-to-end. First audit existing E5 runtime/memory outputs.",
        },
    ]
    claim_df = pd.DataFrame(claim_rows)
    write_tsv(out / "CLAIM_AND_SCOPE_AUDIT.tsv", claim_df)

    manuscript_df = pd.DataFrame(
        MANUSCRIPT_MAP,
        columns=["experiment", "section_role", "recommended_location", "source_evidence", "reviewer_comments"],
    )
    write_tsv(out / "MANUSCRIPT_RESULT_MAP.tsv", manuscript_df)

    # ------------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------------
    report = []
    report.append("# HGV-V3 E6-E12 Scientific Result Audit")
    report.append("")
    report.append(f"- Accepted final tag: `{FINAL_TAG}`")
    report.append(f"- Generated: `{now.isoformat(timespec='seconds')}`")
    report.append("- Existing scientific outputs were read only.")
    report.append("- This report distinguishes **pipeline completion** from **claim/evidence completeness**.")
    report.append("")

    report.append("## E6 — D0-D5 ablation and formal statistics")
    report.append("")
    report.append("### Mode-level test metrics")
    report.append(df_to_markdown(e6_mode, 20))
    report.append("")
    report.append("### D3 paired primary contrasts with species-block bootstrap + Holm")
    report.append(df_to_markdown(e6_primary, 20))
    report.append("")
    report.append("### Final claim decision")
    report.append(df_to_markdown(e6_claim, 20))
    report.append("")
    report.append("### True-teacher fidelity summary")
    report.append(df_to_markdown(e6_teacher, 50))
    report.append("")

    report.append("## E7 — OFAT sensitivity")
    report.append("")
    report.append(df_to_markdown(e7_delta, 30))
    report.append("")
    report.append("Canonical baseline is the frozen E6 D3 k=6 / d=64 / margin=1 / lambda=0.25 result; it was not retrained in E7.")
    report.append("")

    report.append("## E8 — Fragmentation robustness")
    report.append("")
    report.append(df_to_markdown(e8_delta, 20))
    report.append("")
    report.append("Frozen full-assembly models are reused; no fragmentation-specific retraining or threshold retuning.")
    report.append("")

    report.append("## E9 — Explainability and confound audit")
    report.append("")
    report.append(f"- Undefined Spearman rows due to constant/non-estimable input: **{len(undef)}**.")
    report.append("- Undefined correlations are retained as NA and must not be replaced by 0.")
    report.append("- Attribution target is the frozen G2 D3 seed20260915 model; this is not a multi-seed attribution analysis.")
    report.append("")
    report.append("### Confound correlations")
    report.append(df_to_markdown(e9_corr, 80))
    report.append("")
    report.append("### Top-20 integrated-gradient canonical 6-mers")
    report.append(df_to_markdown(e9_ig_top20, 20))
    report.append("")
    report.append("### Top-20 permutation effects ranked by AUROC drop")
    report.append(df_to_markdown(e9_perm_top20, 20))
    report.append("")

    report.append("## E10 — Biological fidelity")
    report.append("")
    report.append("### HGV distance vs ANI / AF correlations")
    report.append(df_to_markdown(e10_corr, 80))
    report.append("")
    report.append("### ANI-range fidelity")
    report.append(df_to_markdown(e10_range, 30))
    report.append("")
    report.append("### Taxonomic-distance consistency")
    report.append(df_to_markdown(e10_tax, 20))
    report.append("")
    report.append(f"Tree component status: `{tree_status}`")
    report.append("")

    report.append("## E11 — Reference screening")
    report.append("")
    report.append("### Retrieval summary")
    report.append(df_to_markdown(e11_ret, 20))
    report.append("")
    report.append("### ANI confirmation coverage overall")
    report.append(df_to_markdown(e11_cov_overall, 20))
    report.append("")
    report.append("### ANI confirmation coverage by external cohort")
    report.append(df_to_markdown(e11_cov_cohort, 30))
    report.append("")

    report.append("## E12 — Scalability")
    report.append("")
    report.append("### Real-FASTA HGV benchmark")
    report.append(df_to_markdown(e12_real, 20))
    report.append("")
    report.append("### Vector-search microbenchmark")
    report.append(df_to_markdown(e12_vec, 30))
    report.append("")
    report.append("### Scope audit")
    report.append(df_to_markdown(e12_scope, 20))
    report.append("")
    report.append("**Important:** E12 v1 benchmarks the HGV pipeline and HGV vector search. It does not itself provide matched end-to-end Mash/Dashing2/skani/FastANI timings. Before deciding that reviewer R1-7 is fully closed, audit the already-frozen E5/runtime outputs for those method costs.")
    report.append("")

    report.append("## Claim/scope boundaries")
    report.append("")
    report.append(df_to_markdown(claim_df, 30))
    report.append("")
    report.append("## Manuscript / reviewer mapping")
    report.append("")
    report.append(df_to_markdown(manuscript_df, 30))
    report.append("")
    report.append("HGV_V3_SCIENTIFIC_RESULTS_AUDIT=PASS")
    report.append("NO_FROZEN_INPUT_MODIFIED=YES")

    (out / "FINAL_SCIENTIFIC_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "AUDIT_PASS.txt").write_text(
        "HGV_V3_SCIENTIFIC_RESULTS_AUDIT=PASS\n"
        "NO_FROZEN_INPUT_MODIFIED=YES\n"
        f"FINAL_TAG={FINAL_TAG}\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("HGV-V3 E6-E12 SCIENTIFIC RESULT AUDIT")
    print("=" * 100)
    print(f"OUT={out}")
    print(f"E6_strict_improvement_all_comparators={strict_all}")
    print(f"E9_undefined_correlation_rows={len(undef)}")
    print(f"E10_tree_status={tree_status}")
    print("E12_native_baseline_runtime_in_E12_v1=NO")
    print("HGV_V3_SCIENTIFIC_RESULTS_AUDIT=PASS")
    print("NO_FROZEN_INPUT_MODIFIED=YES")
    print("=" * 100)


if __name__ == "__main__":
    main()
