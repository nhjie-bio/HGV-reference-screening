#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import json
import sys


# ============================================================
# HGV-V3 E6
# Build canonical 30-cell manifest
#
# Canonical design:
#
# D0: 20260915-20260919 -> new formal E6
# D1: 20260915-20260919 -> new formal E6
# D2: 20260915-20260919 -> new formal E6
# D3: 20260915-20260917 -> frozen main HGV
#     20260918-20260919 -> new formal E6
# D4: 20260915-20260919 -> new formal E6
# D5: 20260915-20260919 -> new formal E6
#
# Total = 30 cells
#
# IMPORTANT:
# - READ ONLY existing training artifacts
# - DO NOT retrain
# - DO NOT modify frozen main models
# - DO NOT modify E6 training outputs
# ============================================================


BASE = Path(
    "HGV_V3_revision_20260915"
)

E6ROOT = (
    BASE
    / "10_ablation"
    / "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

E6_TRAIN = E6ROOT / "03_training"

TRAIN_FREEZE = (
    E6ROOT
    / "01_execution_freeze"
    / "E6_27RUN_TRAINING_FREEZE_v2.txt"
)

OUTDIR = (
    E6ROOT
    / "04_canonical_30cell_manifest_v1_20260923"
)

MANIFEST = OUTDIR / "E6_30CELL_CANONICAL_MANIFEST.tsv"
AUDIT_JSON = OUTDIR / "E6_30CELL_MANIFEST_AUDIT.json"
REPORT = OUTDIR / "E6_30CELL_MANIFEST_REPORT.txt"
SHA_FILE = OUTDIR / "E6_30CELL_SHA256.txt"
FREEZE = OUTDIR / "E6_30CELL_MANIFEST_FREEZE_v1.txt"

SPLIT_SEED = 20260915

MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
]

MODES = {
    "D0": "contrastive_only",
    "D1": "all_pair_distillation",
    "D2": "negative_only_distillation",
    "D3": "informative_negative_distillation",
    "D4": "shuffled_teacher_control",
    "D5": "informative_negative_without_L2",
}

REQUIRED_FILES = [
    "best_model.pt",
    "config.json",
    "history.tsv",
    "RUN_COMPLETE",
    "run_summary.json",
    "val_predictions.tsv",
]


def die(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        die(f"Cannot parse JSON: {path}: {e}")


def first_line(path):
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
    return line.rstrip("\n\r")


def find_main_d3_root():
    """
    Prefer the frozen source-of-truth path defined in the project record.

    If that exact path is unavailable, search only under
    08_main_generalization for a unique candidate containing:

        G2/split20260915/model20260915/best_model.pt

    We never search historical E6 v1/v2 directories.
    """

    primary = (
        BASE
        / "08_main_generalization"
        / "hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916"
    )

    expected = (
        primary
        / "G2"
        / f"split{SPLIT_SEED}"
        / "model20260915"
        / "best_model.pt"
    )

    if expected.is_file():
        return primary, "primary_exact_path"

    search_root = BASE / "08_main_generalization"

    if not search_root.is_dir():
        die(
            "Frozen main-generalization directory does not exist: "
            f"{search_root}"
        )

    candidates = set()

    for model_file in search_root.rglob("best_model.pt"):

        run_dir = model_file.parent

        if run_dir.name != "model20260915":
            continue

        if run_dir.parent.name != f"split{SPLIT_SEED}":
            continue

        if run_dir.parent.parent.name != "G2":
            continue

        candidate_root = run_dir.parent.parent.parent

        name_lower = candidate_root.name.lower()

        if "tau0p98" not in name_lower and "0p98" not in name_lower:
            continue

        candidates.add(candidate_root.resolve())

    candidates = sorted(candidates)

    if len(candidates) != 1:
        print(
            "\n[DIAGNOSTIC] Candidate frozen-main D3 roots:",
            file=sys.stderr,
        )

        if candidates:
            for p in candidates:
                print(f"  {p}", file=sys.stderr)
        else:
            print("  NONE", file=sys.stderr)

        die(
            "Could not uniquely resolve frozen-main D3 root. "
            "Do not guess the source path."
        )

    return candidates[0], "unique_discovery"


def validate_training_freeze():
    if not TRAIN_FREEZE.is_file():
        die(f"Missing E6 training freeze: {TRAIN_FREEZE}")

    txt = TRAIN_FREEZE.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if not txt.strip():
        die(f"E6 training freeze is empty: {TRAIN_FREEZE}")

    return sha256_file(TRAIN_FREEZE), txt


def resolve_run_dir(mode, model_seed, main_root):
    """
    D3 seeds 15/16/17 are reused from frozen main.
    D3 seeds 18/19 are new formal E6.
    All other modes use formal E6 runs.
    """

    if mode == "D3" and model_seed in (
        20260915,
        20260916,
        20260917,
    ):
        run_dir = (
            main_root
            / "G2"
            / f"split{SPLIT_SEED}"
            / f"model{model_seed}"
        )

        return run_dir, "frozen_main_D3"

    run_dir = (
        E6_TRAIN
        / mode
        / "G2"
        / f"split{SPLIT_SEED}"
        / f"model{model_seed}"
    )

    return run_dir, "formal_E6_20260922"


def validate_run(mode, model_seed, run_dir):
    if not run_dir.is_dir():
        die(
            f"Missing run directory for {mode} "
            f"seed={model_seed}: {run_dir}"
        )

    missing = []

    for name in REQUIRED_FILES:
        p = run_dir / name

        if not p.is_file():
            missing.append(name)

    if missing:
        die(
            f"{mode} seed={model_seed} missing required files: "
            f"{missing}\nrun_dir={run_dir}"
        )

    marker = run_dir / "RUN_COMPLETE"

    if marker.stat().st_size <= 0:
        die(
            f"Empty RUN_COMPLETE marker: "
            f"{mode} seed={model_seed}: {marker}"
        )

    # JSON integrity gate.
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "run_summary.json")

    # Validation output must have a non-empty header.
    val_header = first_line(run_dir / "val_predictions.tsv")

    if not val_header.strip():
        die(
            f"Empty validation prediction header: "
            f"{mode} seed={model_seed}"
        )

    return config, summary, val_header


def main():
    print("=" * 100)
    print("HGV-V3 E6 CANONICAL 30-CELL MANIFEST BUILDER v1")
    print("=" * 100)

    if not E6ROOT.is_dir():
        die(f"Formal E6 root does not exist: {E6ROOT}")

    freeze_sha, freeze_text = validate_training_freeze()

    print(f"[PASS] Formal E6 root: {E6ROOT}")
    print(f"[PASS] Training freeze: {TRAIN_FREEZE}")
    print(f"[INFO] Training freeze SHA256: {freeze_sha}")

    main_root, main_resolution = find_main_d3_root()

    print(f"[PASS] Frozen main D3 root: {main_root}")
    print(f"[INFO] Main D3 resolution: {main_resolution}")

    # Never overwrite a previous canonical freeze.
    if OUTDIR.exists() and any(OUTDIR.iterdir()):
        die(
            "Canonical 30-cell output directory already contains files:\n"
            f"{OUTDIR}\n"
            "Do not overwrite a frozen manifest. "
            "Create a new version if regeneration is necessary."
        )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []
    val_headers = {}

    for mode in ["D0", "D1", "D2", "D3", "D4", "D5"]:

        for model_seed in MODEL_SEEDS:

            run_dir, source_block = resolve_run_dir(
                mode,
                model_seed,
                main_root,
            )

            config, summary, val_header = validate_run(
                mode,
                model_seed,
                run_dir,
            )

            cell_id = f"{mode}_seed{model_seed}"

            val_headers[cell_id] = val_header

            best_model = run_dir / "best_model.pt"
            config_json = run_dir / "config.json"
            history_tsv = run_dir / "history.tsv"
            run_complete = run_dir / "RUN_COMPLETE"
            run_summary = run_dir / "run_summary.json"
            val_predictions = run_dir / "val_predictions.tsv"

            row = {
                "cell_id": cell_id,
                "mode": mode,
                "mode_description": MODES[mode],
                "protocol": "G2",
                "split_seed": SPLIT_SEED,
                "model_seed": model_seed,
                "source_block": source_block,
                "run_dir": str(run_dir.resolve()),
                "best_model_pt": str(best_model.resolve()),
                "config_json": str(config_json.resolve()),
                "history_tsv": str(history_tsv.resolve()),
                "run_complete_marker": str(run_complete.resolve()),
                "run_summary_json": str(run_summary.resolve()),
                "val_predictions_tsv": str(
                    val_predictions.resolve()
                ),
                "checkpoint_sha256": sha256_file(best_model),
                "config_sha256": sha256_file(config_json),
                "run_summary_sha256": sha256_file(run_summary),
                "val_predictions_sha256": sha256_file(
                    val_predictions
                ),
            }

            rows.append(row)

            print(
                f"[PASS] {cell_id:<18} "
                f"source={source_block:<18} "
                f"{run_dir}"
            )

    # ========================================================
    # Structural audits
    # ========================================================

    if len(rows) != 30:
        die(f"Expected 30 cells, found {len(rows)}")

    keys = [
        (r["mode"], int(r["model_seed"]))
        for r in rows
    ]

    if len(set(keys)) != 30:
        die("Duplicate mode/model_seed cells detected.")

    mode_counts = {
        mode: sum(r["mode"] == mode for r in rows)
        for mode in MODES
    }

    for mode, n in mode_counts.items():
        if n != 5:
            die(f"{mode}: expected 5 cells, found {n}")

    d3_rows = [
        r for r in rows
        if r["mode"] == "D3"
    ]

    d3_main = [
        r for r in d3_rows
        if r["source_block"] == "frozen_main_D3"
    ]

    d3_new = [
        r for r in d3_rows
        if r["source_block"] == "formal_E6_20260922"
    ]

    if len(d3_main) != 3:
        die(
            "D3 canonical composition invalid: "
            f"expected 3 frozen-main cells, found {len(d3_main)}"
        )

    if len(d3_new) != 2:
        die(
            "D3 canonical composition invalid: "
            f"expected 2 new E6 cells, found {len(d3_new)}"
        )

    expected_main_seeds = {
        20260915,
        20260916,
        20260917,
    }

    observed_main_seeds = {
        int(r["model_seed"])
        for r in d3_main
    }

    if observed_main_seeds != expected_main_seeds:
        die(
            "Frozen-main D3 seed composition mismatch: "
            f"{sorted(observed_main_seeds)}"
        )

    expected_new_d3 = {
        20260918,
        20260919,
    }

    observed_new_d3 = {
        int(r["model_seed"])
        for r in d3_new
    }

    if observed_new_d3 != expected_new_d3:
        die(
            "New-E6 D3 seed composition mismatch: "
            f"{sorted(observed_new_d3)}"
        )

    # Header differences are recorded rather than silently hidden.
    unique_headers = sorted(set(val_headers.values()))

    # ========================================================
    # Write canonical manifest
    # ========================================================

    fieldnames = [
        "cell_id",
        "mode",
        "mode_description",
        "protocol",
        "split_seed",
        "model_seed",
        "source_block",
        "run_dir",
        "best_model_pt",
        "config_json",
        "history_tsv",
        "run_complete_marker",
        "run_summary_json",
        "val_predictions_tsv",
        "checkpoint_sha256",
        "config_sha256",
        "run_summary_sha256",
        "val_predictions_sha256",
    ]

    with MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(rows)

    manifest_sha = sha256_file(MANIFEST)

    # ========================================================
    # Audit JSON
    # ========================================================

    audit = {
        "status": "PASS",
        "purpose": (
            "Canonical 30-cell E6 D0-D5 evaluation manifest"
        ),
        "protocol": "G2",
        "split_seed": SPLIT_SEED,
        "expected_cells": 30,
        "observed_cells": len(rows),
        "model_seeds": MODEL_SEEDS,
        "mode_counts": mode_counts,
        "d3_composition": {
            "frozen_main_count": len(d3_main),
            "frozen_main_seeds": sorted(
                int(r["model_seed"])
                for r in d3_main
            ),
            "new_e6_count": len(d3_new),
            "new_e6_seeds": sorted(
                int(r["model_seed"])
                for r in d3_new
            ),
        },
        "formal_e6_root": str(E6ROOT.resolve()),
        "frozen_main_d3_root": str(main_root.resolve()),
        "main_root_resolution": main_resolution,
        "training_freeze": str(TRAIN_FREEZE.resolve()),
        "training_freeze_sha256": freeze_sha,
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": manifest_sha,
        "validation_header_variants": len(unique_headers),
        "validation_headers": unique_headers,
        "historical_e6_v1_v2_used": False,
        "test_data_read": False,
        "threshold_selection_performed": False,
        "notes": [
            (
                "D3 model seeds 20260915/16/17 are reused "
                "from the frozen main HGV training."
            ),
            (
                "D3 model seeds 20260918/19 are reused from "
                "the formal E6 20260922 training."
            ),
            (
                "D0/D1/D2/D4/D5 use the five formal E6 "
                "model seeds 20260915-20260919."
            ),
            (
                "This step does not read test predictions "
                "or select thresholds."
            ),
        ],
    }

    with AUDIT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    # ========================================================
    # Human-readable report
    # ========================================================

    report_lines = [
        "=" * 80,
        "HGV-V3 E6 30-CELL CANONICAL MANIFEST REPORT",
        "=" * 80,
        "",
        "STATUS=PASS",
        f"protocol=G2",
        f"split_seed={SPLIT_SEED}",
        f"cells={len(rows)}",
        "",
        "Mode counts:",
    ]

    for mode in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        report_lines.append(
            f"  {mode}={mode_counts[mode]}"
        )

    report_lines += [
        "",
        "Canonical D3 composition:",
        (
            "  frozen_main="
            + ",".join(
                str(x)
                for x in sorted(observed_main_seeds)
            )
        ),
        (
            "  new_formal_E6="
            + ",".join(
                str(x)
                for x in sorted(observed_new_d3)
            )
        ),
        "",
        f"Frozen main D3 root:",
        f"  {main_root}",
        "",
        f"Formal E6 root:",
        f"  {E6ROOT}",
        "",
        f"Validation header variants={len(unique_headers)}",
        "",
        f"Manifest:",
        f"  {MANIFEST}",
        f"Manifest SHA256:",
        f"  {manifest_sha}",
        "",
        "Historical E6 v1/v2 used: NO",
        "Test data read: NO",
        "Threshold selection performed: NO",
        "",
        "E6_30CELL_MANIFEST=PASS",
        "",
    ]

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ========================================================
    # SHA256 provenance
    # ========================================================

    script_path = Path(__file__).resolve()

    sha_lines = [
        f"{sha256_file(MANIFEST)}  {MANIFEST}",
        f"{sha256_file(AUDIT_JSON)}  {AUDIT_JSON}",
        f"{sha256_file(REPORT)}  {REPORT}",
        f"{sha256_file(script_path)}  {script_path}",
        f"{freeze_sha}  {TRAIN_FREEZE}",
    ]

    SHA_FILE.write_text(
        "\n".join(sha_lines) + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Final freeze marker
    # ========================================================

    freeze_lines = [
        "HGV-V3 E6 CANONICAL 30-CELL MANIFEST FREEZE",
        "VERSION=v1",
        "DATE=2026-09-23",
        "STATUS=PASS",
        f"PROTOCOL=G2",
        f"SPLIT_SEED={SPLIT_SEED}",
        "CELLS=30",
        "D0=5",
        "D1=5",
        "D2=5",
        "D3=5",
        "D4=5",
        "D5=5",
        "D3_FROZEN_MAIN=3",
        "D3_NEW_E6=2",
        "HISTORICAL_E6_V1_V2_USED=NO",
        "TEST_DATA_READ=NO",
        "THRESHOLD_SELECTION_PERFORMED=NO",
        f"MANIFEST_SHA256={manifest_sha}",
        "E6_30CELL_MANIFEST=PASS",
    ]

    FREEZE.write_text(
        "\n".join(freeze_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)
    print(f"cells          = {len(rows)}")

    for mode in ["D0", "D1", "D2", "D3", "D4", "D5"]:
        print(
            f"{mode:<5}           = "
            f"{mode_counts[mode]}"
        )

    print(
        "D3 frozen main = "
        f"{sorted(observed_main_seeds)}"
    )
    print(
        "D3 new E6      = "
        f"{sorted(observed_new_d3)}"
    )
    print(
        "val headers    = "
        f"{len(unique_headers)} variant(s)"
    )
    print(f"manifest       = {MANIFEST}")
    print(f"manifest sha   = {manifest_sha}")
    print()
    print("E6_30CELL_MANIFEST=PASS")


if __name__ == "__main__":
    main()
