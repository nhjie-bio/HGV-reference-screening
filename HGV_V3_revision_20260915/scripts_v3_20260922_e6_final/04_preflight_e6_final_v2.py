#!/usr/bin/env python3

from pathlib import Path
import csv
import hashlib
import importlib.util
import inspect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


BASE = Path(
    "HGV_V3_revision_20260915"
)

DIR = (
    BASE
    / "scripts_v3_20260922_e6_final"
)

PARENT = (
    DIR
    / "17b_train_main_hierarchical_v1.ORIGINAL.py"
)

TRAINER = (
    DIR
    / "03_train_e6_d0_d5_from_17b_v1.py"
)

PATCH_REPORT = (
    DIR
    / "02_E6_PATCH_REPORT.txt"
)

E6_ROOT = (
    BASE
    / "10_ablation/"
      "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

OUTDIR = (
    E6_ROOT
    / "00_input_freeze"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = (
    OUTDIR
    / "E6_PREFLIGHT_REPORT_v1.txt"
)

JSON_OUT = (
    OUTDIR
    / "E6_PREFLIGHT_AUDIT_v1.json"
)

EXPECTED_PARENT_SHA = (
    "97185a30f8651867e0399a5082a9f45d14232fe4df9d449c152d3ee1191ad356"
)

MODES = (
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
)

ALL_SEEDS = [
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
]

D3_NEW_SEEDS = [
    20260918,
    20260919,
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(
                1024 * 1024
            )

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def require(
    condition,
    message,
):
    if not condition:
        raise RuntimeError(
            message
        )


def load_module(
    path,
    name,
    mode=None,
):

    if mode is not None:
        os.environ[
            "HGV_E6_MODE"
        ] = mode

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    require(
        spec is not None,
        f"Cannot create module spec: {path}",
    )

    require(
        spec.loader is not None,
        f"Missing module loader: {path}",
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def read_teacher_file(path):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        fields = (
            reader.fieldnames
            or []
        )

        rows = list(
            reader
        )

    label_candidates = [
        "label",
        "y",
        "same_species",
        "target",
    ]

    label_col = None

    for c in label_candidates:
        if c in fields:
            label_col = c
            break

    require(
        label_col is not None,
        (
            "Cannot identify label column "
            f"in {path}"
        ),
    )

    preferred_teacher = [
        "dashing2_set_distance",
        "dashing2_distance",
        "dashing_distance",
        "d2_distance",
        "teacher_distance_dashing2",
    ]

    teacher_col = None

    for c in preferred_teacher:
        if c in fields:
            teacher_col = c
            break

    if teacher_col is None:

        hits = [
            c
            for c in fields
            if (
                "dashing" in c.lower()
                and
                "distance" in c.lower()
            )
        ]

        require(
            len(hits) == 1,
            (
                "Cannot uniquely identify "
                f"Dashing2 distance in {path}; "
                f"hits={hits}"
            ),
        )

        teacher_col = hits[0]

    labels = np.asarray(
        [
            int(
                float(
                    r[label_col]
                )
            )
            for r in rows
        ],
        dtype=np.int64,
    )

    teacher = np.asarray(
        [
            float(
                r[teacher_col]
            )
            for r in rows
        ],
        dtype=np.float32,
    )

    require(
        labels.size > 0,
        f"Empty teacher file: {path}",
    )

    require(
        np.all(
            np.isin(
                labels,
                [0, 1],
            )
        ),
        f"Invalid labels: {path}",
    )

    require(
        np.all(
            np.isfinite(
                teacher
            )
        ),
        f"Nonfinite teacher values: {path}",
    )

    require(
        np.all(
            (teacher >= 0)
            &
            (teacher <= 1)
        ),
        f"Teacher outside [0,1]: {path}",
    )

    return {
        "labels": labels,
        "teacher": teacher,
        "label_column": label_col,
        "teacher_column": teacher_col,
        "rows": len(rows),
    }


def array_equal_multiset(
    a,
    b,
):
    return np.array_equal(
        np.sort(
            np.asarray(a)
        ),
        np.sort(
            np.asarray(b)
        ),
    )


# ============================================================
# Gate 1: files and provenance
# ============================================================

require(
    PARENT.is_file(),
    f"Missing parent trainer: {PARENT}",
)

require(
    TRAINER.is_file(),
    f"Missing E6 trainer: {TRAINER}",
)

require(
    PATCH_REPORT.is_file(),
    f"Missing patch report: {PATCH_REPORT}",
)

parent_sha = sha256(
    PARENT
)

trainer_sha = sha256(
    TRAINER
)

require(
    parent_sha
    == EXPECTED_PARENT_SHA,
    (
        "PARENT SHA256 FAIL\n"
        f"expected={EXPECTED_PARENT_SHA}\n"
        f"observed={parent_sha}"
    ),
)

patch_text = (
    PATCH_REPORT
    .read_text(
        encoding="utf-8"
    )
)

require(
    "PATCH_COMPLETE=PASS"
    in patch_text,
    "Patch report does not contain PATCH_COMPLETE=PASS",
)

require(
    "STATIC_PARITY_GATE=PASS"
    in patch_text,
    "Patch report does not contain STATIC_PARITY_GATE=PASS",
)


# ============================================================
# Gate 2: import parent and six E6 modes
# ============================================================

parent = load_module(
    PARENT,
    "hgv_parent_17b",
)

modules = {}

for mode in MODES:

    modules[mode] = load_module(
        TRAINER,
        f"hgv_e6_{mode.lower()}",
        mode=mode,
    )

    m = modules[mode]

    require(
        m.ABLATION_MODE == mode,
        (
            f"{mode}: ABLATION_MODE "
            f"mismatch: {m.ABLATION_MODE}"
        ),
    )

    require(
        list(
            m.PROTOCOLS
        )
        == ["G2"],
        f"{mode}: protocol lock failed",
    )

    require(
        list(
            m.SPLIT_SEEDS
        )
        == [20260915],
        f"{mode}: split seed lock failed",
    )

    expected_seeds = (
        D3_NEW_SEEDS
        if mode == "D3"
        else ALL_SEEDS
    )

    require(
        list(
            m.MODEL_SEEDS
        )
        == expected_seeds,
        (
            f"{mode}: model seed matrix mismatch\n"
            f"expected={expected_seeds}\n"
            f"observed={list(m.MODEL_SEEDS)}"
        ),
    )


# ============================================================
# Gate 3: frozen constants
# ============================================================

constant_names = [
    "INPUT_DIM",
    "H1",
    "H2",
    "EMBED_DIM",
    "MARGIN",
    "TAU",
    "LAMBDA_DIST",
    "LR",
    "WEIGHT_DECAY",
    "PAIR_BATCH",
    "EMBED_BATCH",
    "POS_PER_EPOCH",
    "NEG_PER_EPOCH",
    "MAX_EPOCHS",
    "PATIENCE",
    "GRAD_CLIP",
    "EPS",
]

constant_audit = {}

for name in constant_names:

    parent_value = getattr(
        parent,
        name,
    )

    constant_audit[
        name
    ] = parent_value

    for mode in MODES:

        observed = getattr(
            modules[mode],
            name,
        )

        require(
            observed
            == parent_value,
            (
                f"{mode}: frozen constant "
                f"{name} changed: "
                f"{observed} != {parent_value}"
            ),
        )


# ============================================================
# Gate 4:
# Functions that MUST be source-identical to 17b
# ============================================================

exact_functions = [
    "deterministic",
    "sample_epoch",
    "better",
]

function_hashes = {}

for fn_name in exact_functions:

    parent_src = inspect.getsource(
        getattr(
            parent,
            fn_name,
        )
    )

    e6_src = inspect.getsource(
        getattr(
            modules["D3"],
            fn_name,
        )
    )

    require(
        parent_src == e6_src,
        (
            f"D3 source parity failed "
            f"for {fn_name}"
        ),
    )

    function_hashes[
        fn_name
    ] = hashlib.sha256(
        parent_src.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Gate 5:
# Parent HGV forward vs E6 D3 runtime parity
# and D5 no-L2 single-factor behavior
# ============================================================

torch.manual_seed(
    20260915
)

parent_model = (
    parent.HGV()
    .cpu()
    .eval()
)

torch.manual_seed(
    20260915
)

d3_model = (
    modules["D3"]
    .HGV()
    .cpu()
    .eval()
)

torch.manual_seed(
    20260915
)

d5_model = (
    modules["D5"]
    .HGV()
    .cpu()
    .eval()
)

parent_state = (
    parent_model
    .state_dict()
)

d3_state = (
    d3_model
    .state_dict()
)

d5_state = (
    d5_model
    .state_dict()
)

require(
    list(
        parent_state.keys()
    )
    == list(
        d3_state.keys()
    )
    == list(
        d5_state.keys()
    ),
    "Architecture/state_dict key mismatch",
)

for k in parent_state:

    require(
        torch.equal(
            parent_state[k],
            d3_state[k],
        ),
        f"D3 initial state mismatch: {k}",
    )

    require(
        torch.equal(
            parent_state[k],
            d5_state[k],
        ),
        f"D5 initial state mismatch: {k}",
    )

g = torch.Generator(
    device="cpu"
)

g.manual_seed(
    20260915
)

Xsynthetic = torch.randn(
    16,
    parent.INPUT_DIM,
    generator=g,
)

with torch.no_grad():

    z_parent = parent_model(
        Xsynthetic
    )

    z_d3 = d3_model(
        Xsynthetic
    )

    z_d5 = d5_model(
        Xsynthetic
    )

max_parent_d3 = float(
    torch.max(
        torch.abs(
            z_parent
            - z_d3
        )
    )
)

require(
    max_parent_d3 <= 1e-7,
    (
        "D3 HGV forward differs from parent; "
        f"max_abs={max_parent_d3}"
    ),
)

d3_norm = (
    torch.linalg
    .vector_norm(
        z_d3,
        ord=2,
        dim=1,
    )
)

require(
    torch.allclose(
        d3_norm,
        torch.ones_like(
            d3_norm
        ),
        atol=1e-6,
        rtol=1e-6,
    ),
    "D3 output is not L2 normalized",
)

d5_re_normalized = F.normalize(
    z_d5,
    p=2,
    dim=1,
)

require(
    torch.allclose(
        d5_re_normalized,
        z_d3,
        atol=1e-6,
        rtol=1e-6,
    ),
    (
        "D5 is not equivalent to "
        "D3 pre-normalization geometry"
    ),
)


# ============================================================
# Gate 6:
# Resolve the EXACT frozen canonical G2 seed20260915
# teacher split.
#
# Do NOT search by substring "20260915", because the project
# root itself contains HGV_V3_revision_20260915.
# ============================================================

teacher_root = Path(
    parent.TEACHER_ROOT
)

require(
    teacher_root.is_dir(),
    (
        "Missing teacher root: "
        f"{teacher_root}"
    ),
)

split_id = "G2_seed20260915"

split_dir = (
    teacher_root
    / split_id
)

require(
    split_dir.is_dir(),
    (
        "Missing exact canonical teacher split: "
        f"{split_dir}"
    ),
)

train_path = (
    split_dir
    / "train_teacher.tsv"
)

val_path = (
    split_dir
    / "val_teacher.tsv"
)

require(
    train_path.is_file(),
    (
        "Missing canonical train teacher file: "
        f"{train_path}"
    ),
)

require(
    val_path.is_file(),
    (
        "Missing canonical validation teacher file: "
        f"{val_path}"
    ),
)

# Explicitly prohibit accidental resolution to another split.
require(
    split_dir.name == "G2_seed20260915",
    (
        "Canonical split identity mismatch: "
        f"{split_dir.name}"
    ),
)

train_raw = read_teacher_file(
    train_path
)

val_raw = read_teacher_file(
    val_path
)


# ============================================================
# Gate 7:
# D0-D5 pair-level mask and target semantics
# ============================================================

role_audits = {}

for role, raw in [
    (
        "train",
        train_raw,
    ),
    (
        "validation",
        val_raw,
    ),
]:

    labels = (
        raw[
            "labels"
        ]
    )

    teacher = (
        raw[
            "teacher"
        ]
    )

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    informative_negative = (
        negative
        &
        (
            teacher
            < parent.TAU
        )
    )

    selective = (
        positive
        |
        informative_negative
    )

    expected_masks = {
        "D0":
            np.zeros(
                labels.shape,
                dtype=np.bool_,
            ),

        "D1":
            np.ones(
                labels.shape,
                dtype=np.bool_,
            ),

        "D2":
            negative,

        "D3":
            selective,

        "D4":
            selective,

        "D5":
            selective,
    }

    mode_outputs = {}

    for mode in MODES:

        pairs = {
            "labels":
                labels.copy(),

            "teacher":
                teacher.copy(),
        }

        audit = (
            modules[mode]
            .prepare_ablation_pairs(
                pairs,
                split_id,
                role,
            )
        )

        mask = np.asarray(
            pairs[
                "distill_mask"
            ],
            dtype=np.bool_,
        )

        target = np.asarray(
            pairs[
                "teacher_target"
            ],
            dtype=np.float32,
        )

        require(
            np.array_equal(
                mask,
                expected_masks[
                    mode
                ],
            ),
            (
                f"{role} {mode}: "
                "distillation mask mismatch"
            ),
        )

        if mode != "D4":

            require(
                np.array_equal(
                    target,
                    teacher,
                ),
                (
                    f"{role} {mode}: "
                    "teacher target unexpectedly changed"
                ),
            )

        mode_outputs[
            mode
        ] = {
            "mask":
                mask,

            "target":
                target,

            "audit":
                audit,
        }

    # D3/D4/D5 masks must be byte-identical.
    require(
        np.array_equal(
            mode_outputs[
                "D3"
            ][
                "mask"
            ],
            mode_outputs[
                "D4"
            ][
                "mask"
            ],
        ),
        f"{role}: D4 mask != D3 mask",
    )

    require(
        np.array_equal(
            mode_outputs[
                "D3"
            ][
                "mask"
            ],
            mode_outputs[
                "D5"
            ][
                "mask"
            ],
        ),
        f"{role}: D5 mask != D3 mask",
    )

    # D4 preserves conditional teacher distributions.
    d4_target = (
        mode_outputs[
            "D4"
        ][
            "target"
        ]
    )

    require(
        array_equal_multiset(
            teacher[
                positive
            ],
            d4_target[
                positive
            ],
        ),
        (
            f"{role}: D4 positive "
            "target distribution changed"
        ),
    )

    require(
        array_equal_multiset(
            teacher[
                informative_negative
            ],
            d4_target[
                informative_negative
            ],
        ),
        (
            f"{role}: D4 informative-negative "
            "target distribution changed"
        ),
    )

    easy_negative = (
        negative
        &
        ~informative_negative
    )

    require(
        np.array_equal(
            teacher[
                easy_negative
            ],
            d4_target[
                easy_negative
            ],
        ),
        (
            f"{role}: D4 changed "
            "non-selected easy negatives"
        ),
    )

    selected = selective

    selected_unique = np.unique(
        teacher[
            selected
        ]
    ).size

    changed_selected = int(
        np.sum(
            teacher[
                selected
            ]
            !=
            d4_target[
                selected
            ]
        )
    )

    if (
        int(
            selected.sum()
        ) > 1
        and
        selected_unique > 1
    ):
        require(
            changed_selected > 0,
            (
                f"{role}: D4 shuffle did not "
                "change any selected pair-target mapping"
            ),
        )

    # D4 must be deterministic across repeated preparation.
    pairs_repeat = {
        "labels":
            labels.copy(),

        "teacher":
            teacher.copy(),
    }

    modules[
        "D4"
    ].prepare_ablation_pairs(
        pairs_repeat,
        split_id,
        role,
    )

    require(
        np.array_equal(
            pairs_repeat[
                "teacher_target"
            ],
            d4_target,
        ),
        (
            f"{role}: D4 shuffle "
            "is not deterministic"
        ),
    )

    role_audits[
        role
    ] = {
        "rows":
            int(
                labels.size
            ),

        "positive":
            int(
                positive.sum()
            ),

        "negative":
            int(
                negative.sum()
            ),

        "informative_negative":
            int(
                informative_negative.sum()
            ),

        "D0_distilled":
            int(
                mode_outputs[
                    "D0"
                ][
                    "mask"
                ].sum()
            ),

        "D1_distilled":
            int(
                mode_outputs[
                    "D1"
                ][
                    "mask"
                ].sum()
            ),

        "D2_distilled":
            int(
                mode_outputs[
                    "D2"
                ][
                    "mask"
                ].sum()
            ),

        "D3_distilled":
            int(
                mode_outputs[
                    "D3"
                ][
                    "mask"
                ].sum()
            ),

        "D4_distilled":
            int(
                mode_outputs[
                    "D4"
                ][
                    "mask"
                ].sum()
            ),

        "D5_distilled":
            int(
                mode_outputs[
                    "D5"
                ][
                    "mask"
                ].sum()
            ),

        "D4_changed_selected":
            changed_selected,
    }


# ============================================================
# Gate 8:
# D3 evaluator numerical parity with parent 17b
# on a deterministic synthetic pair set.
# ============================================================

rng = np.random.default_rng(
    20260915
)

n_genomes = 24
n_pairs = 64

a_idx = rng.integers(
    0,
    n_genomes,
    size=n_pairs,
    dtype=np.int64,
)

b_idx = rng.integers(
    0,
    n_genomes,
    size=n_pairs,
    dtype=np.int64,
)

# Avoid self-pairs.
same = (
    a_idx == b_idx
)

b_idx[
    same
] = (
    b_idx[
        same
    ]
    + 1
) % n_genomes

labels = np.asarray(
    [
        i % 2
        for i in range(
            n_pairs
        )
    ],
    dtype=np.int64,
)

teacher = rng.uniform(
    0.0,
    1.0,
    size=n_pairs,
).astype(
    np.float32
)

g2 = torch.Generator(
    device="cpu"
)

g2.manual_seed(
    20260916
)

X_eval = torch.randn(
    n_genomes,
    parent.INPUT_DIM,
    generator=g2,
)

pairs_parent = {
    "a_idx":
        a_idx,

    "b_idx":
        b_idx,

    "labels":
        labels,

    "teacher":
        teacher,
}

pairs_d3 = {
    "a_idx":
        a_idx.copy(),

    "b_idx":
        b_idx.copy(),

    "labels":
        labels.copy(),

    "teacher":
        teacher.copy(),
}

modules[
    "D3"
].prepare_ablation_pairs(
    pairs_d3,
    "synthetic_G2_seed20260915",
    "validation",
)

parent_model.eval()

r_parent = parent.evaluate(
    parent_model,
    X_eval,
    pairs_parent,
)

r_d3 = (
    modules[
        "D3"
    ]
    .evaluate(
        parent_model,
        X_eval,
        pairs_d3,
    )
)

scalar_keys = [
    "AUROC",
    "AUPRC",
    "score",
    "contrastive",
    "distill",
]

for k in scalar_keys:

    delta = abs(
        float(
            r_parent[k]
        )
        -
        float(
            r_d3[k]
        )
    )

    require(
        delta <= 1e-7,
        (
            f"D3 evaluator parity failed "
            f"for {k}; delta={delta}"
        ),
    )

for k in [
    "similarity",
    "distance",
]:

    require(
        np.allclose(
            r_parent[k],
            r_d3[k],
            atol=1e-7,
            rtol=1e-7,
        ),
        (
            f"D3 evaluator array parity "
            f"failed for {k}"
        ),
    )


# ============================================================
# Gate 9:
# Formal E6 run accounting
# ============================================================

new_run_count = sum(
    len(
        modules[
            mode
        ].MODEL_SEEDS
    )
    for mode in MODES
)

require(
    new_run_count == 27,
    (
        "New run count must be 27; "
        f"observed={new_run_count}"
    ),
)

reused_d3_count = 3

final_cell_count = (
    new_run_count
    + reused_d3_count
)

require(
    final_cell_count == 30,
    (
        "Final E6 cell count must be 30; "
        f"observed={final_cell_count}"
    ),
)


# ============================================================
# Final report
# ============================================================

audit = {
    "status":
        "PASS",

    "parent_trainer":
        str(
            PARENT
        ),

    "parent_sha256":
        parent_sha,

    "e6_trainer":
        str(
            TRAINER
        ),

    "e6_trainer_sha256":
        trainer_sha,

    "protocol":
        "G2",

    "split_seed":
        20260915,

    "teacher_split_dir":
        str(
            split_dir
        ),

    "train_teacher_sha256":
        sha256(
            train_path
        ),

    "validation_teacher_sha256":
        sha256(
            val_path
        ),

    "frozen_constants":
        constant_audit,

    "exact_function_hashes":
        function_hashes,

    "parent_vs_D3_forward_max_abs":
        max_parent_d3,

    "role_audits":
        role_audits,

    "new_runs":
        new_run_count,

    "reused_frozen_D3_runs":
        reused_d3_count,

    "final_E6_cells":
        final_cell_count,

    "new_run_matrix": {
        "D0":
            ALL_SEEDS,

        "D1":
            ALL_SEEDS,

        "D2":
            ALL_SEEDS,

        "D3":
            D3_NEW_SEEDS,

        "D4":
            ALL_SEEDS,

        "D5":
            ALL_SEEDS,
    },
}

JSON_OUT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

lines = [
    "=" * 100,
    "HGV-V3 FORMAL E6 PREFLIGHT v1",
    "=" * 100,
    f"PARENT_SHA256={parent_sha}",
    f"E6_TRAINER_SHA256={trainer_sha}",
    f"TEACHER_SPLIT={split_dir}",
    "",
    "GATE_PARENT_PROVENANCE=PASS",
    "GATE_PATCH_REPORT=PASS",
    "GATE_PROTOCOL_LOCK=PASS",
    "GATE_SEED_MATRIX=PASS",
    "GATE_FROZEN_CONSTANTS=PASS",
    "GATE_DETERMINISTIC_FUNCTION_PARITY=PASS",
    "GATE_D3_FORWARD_PARITY=PASS",
    "GATE_D5_NO_L2_SINGLE_FACTOR=PASS",
    "GATE_D0_D5_MASK_SEMANTICS=PASS",
    "GATE_D4_DISTRIBUTION_PRESERVATION=PASS",
    "GATE_D4_DETERMINISM=PASS",
    "GATE_D3_EVALUATOR_PARITY=PASS",
    "",
]

for role in [
    "train",
    "validation",
]:

    x = role_audits[
        role
    ]

    lines.extend(
        [
            f"[{role}]",
            f"rows={x['rows']}",
            f"positive={x['positive']}",
            f"negative={x['negative']}",
            (
                "informative_negative="
                f"{x['informative_negative']}"
            ),
            f"D0_distilled={x['D0_distilled']}",
            f"D1_distilled={x['D1_distilled']}",
            f"D2_distilled={x['D2_distilled']}",
            f"D3_distilled={x['D3_distilled']}",
            f"D4_distilled={x['D4_distilled']}",
            f"D5_distilled={x['D5_distilled']}",
            (
                "D4_changed_selected="
                f"{x['D4_changed_selected']}"
            ),
            "",
        ]
    )

lines.extend(
    [
        "RUN_ACCOUNTING:",
        "D0=5 new",
        "D1=5 new",
        "D2=5 new",
        "D3=2 new + 3 reused frozen",
        "D4=5 new",
        "D5=5 new",
        f"NEW_RUNS={new_run_count}",
        f"REUSED_D3={reused_d3_count}",
        f"FINAL_E6_CELLS={final_cell_count}",
        "",
        "TEST_OPENED=NO",
        "EXTERNAL_OPENED=NO",
        "",
        "E6_PREFLIGHT=PASS",
        "=" * 100,
    ]
)

REPORT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)

print(
    "\n".join(
        lines
    )
)
