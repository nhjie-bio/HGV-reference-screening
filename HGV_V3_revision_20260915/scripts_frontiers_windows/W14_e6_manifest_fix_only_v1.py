#!/usr/bin/env python3
"""Recover canonical E6 checkpoints and run only D0/D1/D3 retrieval."""

import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

W13_SCRIPT = ROOT / (
    "scripts_frontiers_windows/"
    "W13_HGV_two_questions_oneshot_v1.py"
)

WORK = ROOT / (
    "21_frontiers_revision_20261112/"
    "90_windows_minimal"
)

W13_OUT = WORK / "W13_TWO_QUESTIONS_20261114_v1"
OUT = WORK / "W14_E6_MANIFEST_RECOVERED_v1_20261114"

E6 = ROOT / (
    "10_ablation/"
    "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

MANIFEST = E6 / (
    "04_canonical_30cell_manifest_v1_20260923/"
    "E6_30CELL_CANONICAL_MANIFEST.tsv"
)

SEEDS = (
    20260915, 20260916, 20260917,
    20260918, 20260919
)

MODES = ("D0", "D1", "D3")


def require(condition, message):
    if not condition:
        raise RuntimeError("[STOP] " + message)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_w13():
    require(
        W13_SCRIPT.is_file(),
        "Missing original W13 script"
    )

    spec = importlib.util.spec_from_file_location(
        "w13_original_frozen", W13_SCRIPT
    )

    require(
        spec is not None and spec.loader is not None,
        "Cannot import original W13"
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Redirect all new results to W14 only.
    module.OUT = OUT

    return module


def locate_checkpoint(record):
    source = str(record["best_model_pt"]).strip()

    require(
        source.endswith("/best_model.pt")
        or source.endswith("\\best_model.pt"),
        "Invalid checkpoint path: " + record["cell_id"]
    )

    normalized = source.replace("\\", "/")
    marker = "HGV_V3_revision_20260915/"

    require(
        marker in normalized,
        "Checkpoint outside frozen project"
    )

    relative = normalized.split(marker, 1)[1]
    original_path = ROOT / relative

    expected = str(
        record["checkpoint_sha256"]
    ).strip().lower()

    require(
        len(expected) == 64
        and all(c in "0123456789abcdef" for c in expected),
        "Invalid checkpoint SHA256"
    )

    # Alternative is accepted ONLY if its bytes have the
    # exact SHA256 recorded in the canonical manifest.
    alternative = ROOT / (
        "10_ablation/"
        "E6_D0_D5_G2_seed20260915_5seed_v2_20260921/"
        f"runs/{record['mode']}_modelseed"
        f"{record['model_seed']}/best_model.pt"
    )

    candidates = [original_path]

    if not original_path.is_file():
        candidates.append(alternative)

    valid = []

    for path in candidates:
        if not path.is_file():
            continue

        observed = sha256(path)

        if observed == expected:
            valid.append((path, observed))

    require(
        len(valid) == 1,
        (
            f"{record['cell_id']} checkpoint missing "
            f"or SHA256 mismatch. "
            f"Expected={expected}; "
            f"original={original_path}; "
            f"alternative={alternative}"
        )
    )

    return valid[0]


def main():
    print("=" * 74)
    print("W14 - E6 D0/D1/D3 MANIFEST-LOCKED RETRIEVAL")
    print("=" * 74)

    require(
        not OUT.exists(),
        "Output already exists: " + str(OUT)
    )

    require(
        W13_OUT.is_dir(),
        "Previous W13 output missing"
    )

    previous_status = (
        W13_OUT / "W13_STATUS.txt"
    ).read_text(encoding="utf-8")

    require(
        "E11_EQUAL_BUDGET=PASS" in previous_status,
        "Previous E11 not passed"
    )

    require(
        "E13_REAL_MAG=PASS_PRELIMINARY" in previous_status,
        "Previous E13 not passed"
    )

    require(
        MANIFEST.is_file(),
        "Canonical E6 manifest missing"
    )

    mod = load_w13()

    records = mod.rows(MANIFEST)

    by_cell = {}

    for record in records:
        cell = record.get("cell_id", "")

        require(
            cell not in by_cell,
            "Duplicate manifest cell: " + cell
        )

        by_cell[cell] = record

    checkpoints = {}
    qc = []

    # =====================================================
    # STEP 1: VERIFY ALL 15 ORIGINAL CHECKPOINTS
    # =====================================================

    for mode in MODES:
        for seed in SEEDS:

            cell = f"{mode}_seed{seed}"

            require(
                cell in by_cell,
                "Missing canonical cell: " + cell
            )

            record = by_cell[cell]

            require(
                record["mode"] == mode
                and str(record["model_seed"]) == str(seed)
                and record["protocol"] == "G2"
                and str(record["split_seed"]) == "20260915",
                "Invalid manifest metadata: " + cell
            )

            path, observed = locate_checkpoint(record)

            checkpoints[(mode, seed)] = path

            qc.append({
                "cell_id": cell,
                "mode": mode,
                "model_seed": seed,
                "source_block": record.get(
                    "source_block", ""
                ),
                "checkpoint": str(path),
                "expected_sha256":
                    record["checkpoint_sha256"],
                "actual_sha256": observed,
                "status": "PASS_SHA256"
            })

            print(
                f"[PASS] {cell} | "
                f"{record.get('source_block', '')} | "
                "SHA256",
                flush=True
            )

    require(
        len(checkpoints) == 15,
        "Expected exactly 15 original checkpoints"
    )

    OUT.mkdir(
        parents=True,
        exist_ok=False
    )

    mod.write(
        "W14_FROZEN_CHECKPOINTS_SHA256.tsv",
        qc
    )

    # =====================================================
    # STEP 2: OVERRIDE ONLY CHECKPOINT RESOLUTION
    # =====================================================

    def verified_checkpoint(mode, seed, unused_manifest):
        key = (mode, int(seed))

        require(
            key in checkpoints,
            "Noncanonical checkpoint requested: " + str(key)
        )

        path = checkpoints[key]

        record = by_cell[
            f"{mode}_seed{seed}"
        ]

        require(
            sha256(path)
            == record["checkpoint_sha256"].lower(),
            "Checkpoint changed: " + str(path)
        )

        return path

    mod.get_checkpoint = verified_checkpoint

    # =====================================================
    # STEP 3: REUSE ORIGINAL W13 E6 EVALUATION
    # =====================================================

    try:
        mm, fi, X = mod.load_common()

        result = mod.module_e6(
            mm, fi, X
        )

        require(
            result.get("status") == "PASS"
            and result.get("checkpoints") == 15,
            "E6 did not finish all 15 runs"
        )

        provenance = {
            "status": "PASS",
            "experiment":
                "post_hoc frozen E6 G2 retrieval",
            "script_sha256": sha256(Path(__file__)),
            "original_W13_sha256": sha256(W13_SCRIPT),
            "E6_manifest_sha256": sha256(MANIFEST),
            "checkpoints": qc,
            "E6_results": result,
            "no_model_training": True,
            "no_earlier_output_overwrite": True,
            "E11_E13_rerun": False,
            "inference_note": (
                "Cluster bootstrap is exploratory. "
                "Five seeds are not independent "
                "taxonomy splits."
            )
        }

        (
            OUT / "W14_PROVENANCE.json"
        ).write_text(
            json.dumps(
                provenance,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        (
            OUT / "W14_STATUS.txt"
        ).write_text(
            "W14_E6_D0_D1_D3=PASS\n"
            "CHECKPOINT_SHA256=15/15\n"
            "E6_RETRIEVAL=PASS\n"
            "E11_E13_RERUN=NO\n",
            encoding="utf-8"
        )

        print()
        print("=" * 74)
        print("W14_E6_D0_D1_D3=PASS")
        print("All 15 canonical checkpoints verified")

        print()
        print("===== G2 GENUS: D3 - D0 / D1 =====")

        comparisons = mod.rows(
            OUT / "E6_PAIRED_D3_CONTRASTS.tsv"
        )

        for row in comparisons:
            if (
                row["tax_level"] == "genus"
                and int(row["K"]) in (1, 20)
            ):
                print(
                    row["contrast"],
                    "K=", row["K"],
                    "delta=", row["delta_mean_5seeds"],
                    "CI=[",
                    row["cluster_boot_lo"],
                    row["cluster_boot_hi"],
                    "]"
                )

        print()
        print("===== G2 GENUS: FIVE-SEED RECALL =====")

        means = mod.rows(
            OUT / "E6_MEAN_RECALL_5SEEDS.tsv"
        )

        for row in means:
            if (
                row["task"] == "G2_genus"
                and int(row["K"]) in (1, 20)
            ):
                print(
                    row["method"],
                    "K=", row["K"],
                    "mean=", row["mean_recall_5seeds"],
                    "SD=", row["seed_sd"]
                )

        print()
        print("Output:", OUT)

    except Exception as exc:

        (
            OUT / "W14_STATUS.txt"
        ).write_text(
            "W14_E6_D0_D1_D3=INCOMPLETE\n"
            "ERROR=" + str(exc) + "\n",
            encoding="utf-8"
        )

        (
            OUT / "W14_ERROR.txt"
        ).write_text(
            traceback.format_exc(),
            encoding="utf-8"
        )

        raise


if __name__ == "__main__":
    main()
