#!/usr/bin/env python3

from pathlib import Path
import difflib
import hashlib

BASE = Path(
    "HGV_V3_revision_20260915"
)

DIR = (
    BASE
    / "scripts_v3_20260922_e6_final"
)

SRC = (
    DIR
    / "17b_train_main_hierarchical_v1.ORIGINAL.py"
)

DST = (
    DIR
    / "03_train_e6_d0_d5_from_17b_v1.py"
)

REPORT = (
    DIR
    / "02_E6_PATCH_REPORT.txt"
)

DIFF = (
    DIR
    / "02_E6_ALLOWED_DIFF.patch"
)


EXPECTED_PARENT_SHA256 = (
    "97185a30f8651867e0399a5082a9f45d14232fe4df9d449c152d3ee1191ad356"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def must_replace(text, old, new, label):
    n = text.count(old)

    if n != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 match, found {n}"
        )

    return text.replace(
        old,
        new,
        1,
    )


if not SRC.is_file():
    raise FileNotFoundError(
        f"Missing frozen parent trainer: {SRC}"
    )


parent_sha = sha256_file(SRC)

if parent_sha != EXPECTED_PARENT_SHA256:
    raise RuntimeError(
        "PARENT_SHA256_MISMATCH\n"
        f"expected={EXPECTED_PARENT_SHA256}\n"
        f"observed={parent_sha}"
    )


original = SRC.read_text(
    encoding="utf-8"
)

s = original


# ============================================================
# 1. Ensure os import exists
# ============================================================

if "import os\n" not in s:

    anchor = "import platform\n"

    if s.count(anchor) != 1:
        raise RuntimeError(
            "Cannot safely insert import os"
        )

    s = s.replace(
        anchor,
        "import os\n" + anchor,
        1,
    )


# ============================================================
# 2. Replace only execution matrix/output globals.
#
# Original trainer:
#   G2/G3/G4
#   split seeds 15/16/17
#   model seeds 15/16/17
#
# E6:
#   G2 only
#   split seed 20260915 only
#
# D0/D1/D2/D4/D5:
#   5 model seeds
#
# D3:
#   only new seeds 20260918/20260919
#   frozen seed15/16/17 are reused later.
# ============================================================

out_start = s.index(
    "OUT = (",
    s.index("EVAL_FREEZE"),
)

input_dim_pos = s.index(
    "INPUT_DIM = 2080"
)

new_execution_globals = r'''
E6_ROOT = (
    BASE
    / "10_ablation/"
      "E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922"
)

VALID_ABLATION_MODES = (
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
)

ABLATION_MODE = (
    os.environ
    .get(
        "HGV_E6_MODE",
        "",
    )
    .strip()
    .upper()
)

if ABLATION_MODE not in VALID_ABLATION_MODES:
    raise RuntimeError(
        "HGV_E6_MODE must be one of "
        "D0,D1,D2,D3,D4,D5. "
        f"Observed: {ABLATION_MODE!r}"
    )

OUT = (
    E6_ROOT
    / "03_training"
    / ABLATION_MODE
)

PROTOCOLS = [
    "G2",
]

SPLIT_SEEDS = [
    20260915,
]

ALL_E6_MODEL_SEEDS = [
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
]

if ABLATION_MODE == "D3":
    MODEL_SEEDS = [
        20260918,
        20260919,
    ]
else:
    MODEL_SEEDS = list(
        ALL_E6_MODEL_SEEDS
    )

D4_SHUFFLE_SEED = 20260915

PARENT_17B_SHA256 = (
    "97185a30f8651867e0399a5082a9f45d14232fe4df9d449c152d3ee1191ad356"
)


'''

s = (
    s[:out_start]
    + new_execution_globals
    + s[input_dim_pos:]
)


# ============================================================
# 3. Insert E6 mechanistic helper.
#
# IMPORTANT:
#
# D0:
#   no distillation
#
# D1:
#   all-pair distillation
#
# D2:
#   all negative pairs
#
# D3:
#   EXACT final 17b selective rule:
#
#   positive
#   OR
#   negative & teacher_distance < 0.98
#
# D4:
#   EXACT D3 mask
#   but shuffled teacher target.
#
#   Shuffle is label-stratified:
#     positives within positives
#     informative negatives within informative negatives
#
# D5:
#   EXACT D3 distillation rule
#   but no output L2 normalization.
#
# The fixed D4 permutation depends only on:
#   D4_SHUFFLE_SEED
#   split_id
#   role
#
# It does NOT depend on model seed.
# ============================================================

helper_anchor = "\n\nclass HGV(nn.Module):\n"

if s.count(helper_anchor) != 1:
    raise RuntimeError(
        "Cannot safely locate HGV class anchor"
    )


helper_code = r'''

def array_sha256(x):

    x = np.ascontiguousarray(x)

    return hashlib.sha256(
        x.tobytes()
    ).hexdigest()


def d4_role_seed(
    split_id,
    role,
):

    payload = (
        f"{D4_SHUFFLE_SEED}|"
        f"{split_id}|"
        f"{role}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**32)
    )


def prepare_ablation_pairs(
    pairs,
    split_id,
    role,
):

    labels = np.asarray(
        pairs["labels"],
        dtype=np.int64,
    )

    teacher = np.asarray(
        pairs["teacher"],
        dtype=np.float32,
    ).copy()

    if labels.shape != teacher.shape:
        raise RuntimeError(
            "Label/teacher shape mismatch"
        )

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    informative_negative = (
        negative
        & (
            teacher
            < TAU
        )
    )

    selective_mask = (
        positive
        | informative_negative
    )

    if ABLATION_MODE == "D0":

        mask = np.zeros(
            labels.shape,
            dtype=np.bool_,
        )

    elif ABLATION_MODE == "D1":

        mask = np.ones(
            labels.shape,
            dtype=np.bool_,
        )

    elif ABLATION_MODE == "D2":

        mask = negative.copy()

    elif ABLATION_MODE in (
        "D3",
        "D4",
        "D5",
    ):

        mask = (
            selective_mask.copy()
        )

    else:

        raise RuntimeError(
            f"Unknown mode: {ABLATION_MODE}"
        )

    target = teacher.copy()

    shuffle_seed = None
    changed_selected = 0

    positive_distribution_preserved = True
    informative_negative_distribution_preserved = True

    if ABLATION_MODE == "D4":

        shuffle_seed = (
            d4_role_seed(
                split_id,
                role,
            )
        )

        rng = np.random.default_rng(
            shuffle_seed
        )

        positive_ids = np.flatnonzero(
            positive
        )

        informative_negative_ids = (
            np.flatnonzero(
                informative_negative
            )
        )

        for ids in (
            positive_ids,
            informative_negative_ids,
        ):

            if ids.size > 1:

                source_ids = ids.copy()

                rng.shuffle(
                    source_ids
                )

                target[ids] = (
                    teacher[
                        source_ids
                    ]
                )

        if positive_ids.size:

            positive_distribution_preserved = bool(
                np.array_equal(
                    np.sort(
                        teacher[
                            positive_ids
                        ]
                    ),
                    np.sort(
                        target[
                            positive_ids
                        ]
                    ),
                )
            )

        if informative_negative_ids.size:

            informative_negative_distribution_preserved = bool(
                np.array_equal(
                    np.sort(
                        teacher[
                            informative_negative_ids
                        ]
                    ),
                    np.sort(
                        target[
                            informative_negative_ids
                        ]
                    ),
                )
            )

        changed_selected = int(
            np.sum(
                target[mask]
                != teacher[mask]
            )
        )

        if not (
            positive_distribution_preserved
            and informative_negative_distribution_preserved
        ):

            raise RuntimeError(
                "D4 target distribution preservation failed"
            )

    pairs[
        "distill_mask"
    ] = mask.astype(
        np.bool_,
        copy=False,
    )

    pairs[
        "teacher_target"
    ] = target.astype(
        np.float32,
        copy=False,
    )

    audit = {
        "mode":
            ABLATION_MODE,

        "split_id":
            str(
                split_id
            ),

        "role":
            str(
                role
            ),

        "n_pairs":
            int(
                labels.size
            ),

        "n_positive":
            int(
                positive.sum()
            ),

        "n_negative":
            int(
                negative.sum()
            ),

        "n_informative_negative":
            int(
                informative_negative.sum()
            ),

        "n_distilled":
            int(
                mask.sum()
            ),

        "distilled_fraction":
            float(
                mask.mean()
            ),

        "teacher_original_sha256":
            array_sha256(
                teacher
            ),

        "distill_mask_sha256":
            array_sha256(
                mask
            ),

        "teacher_target_sha256":
            array_sha256(
                target
            ),

        "d4_shuffle_seed":
            (
                None
                if shuffle_seed is None
                else int(
                    shuffle_seed
                )
            ),

        "d4_changed_selected_targets":
            int(
                changed_selected
            ),

        "d4_positive_distribution_preserved":
            bool(
                positive_distribution_preserved
            ),

        "d4_informative_negative_distribution_preserved":
            bool(
                informative_negative_distribution_preserved
            ),
    }

    return audit


'''

s = s.replace(
    helper_anchor,
    helper_code
    + helper_anchor,
    1,
)


# ============================================================
# 4. D5 only:
#    remove output L2 normalization.
#
# All other modes use EXACT original forward normalization.
# ============================================================

old_forward_return = '''        return F.normalize(
            x,
            p=2,
            dim=1,
        )
'''

new_forward_return = '''        if ABLATION_MODE == "D5":
            return x

        return F.normalize(
            x,
            p=2,
            dim=1,
        )
'''

s = must_replace(
    s,
    old_forward_return,
    new_forward_return,
    "HGV forward normalization",
)


# ============================================================
# 5. Prepare ablation masks/targets immediately after the
#    exact same train/val teacher files are loaded.
# ============================================================

old_pair_load = '''    val = load_pairs(
        val_path,
        acc_to_idx,
    )

    config = {
'''

new_pair_load = '''    val = load_pairs(
        val_path,
        acc_to_idx,
    )

    train_ablation_audit = (
        prepare_ablation_pairs(
            train,
            split_id,
            "train",
        )
    )

    val_ablation_audit = (
        prepare_ablation_pairs(
            val,
            split_id,
            "validation",
        )
    )

    config = {
'''

s = must_replace(
    s,
    old_pair_load,
    new_pair_load,
    "pair preparation insertion",
)


# ============================================================
# 6. Validation evaluator:
#    use frozen per-pair mask and target.
#
# Distance/score/contrastive computation remain unchanged.
# ============================================================

old_eval_teacher = '''    teacher = torch.as_tensor(
        pairs["teacher"],
        dtype=torch.float32,
        device=X.device,
    )
'''

new_eval_teacher = '''    teacher_target = torch.as_tensor(
        pairs["teacher_target"],
        dtype=torch.float32,
        device=X.device,
    )

    distill_mask = torch.as_tensor(
        pairs["distill_mask"],
        dtype=torch.bool,
        device=X.device,
    )
'''

s = must_replace(
    s,
    old_eval_teacher,
    new_eval_teacher,
    "validation teacher tensor",
)


old_eval_mask = '''    mask = (
        (y > 0.5)
        |
        (
            (y < 0.5)
            & (teacher < TAU)
        )
    )
'''

new_eval_mask = '''    mask = distill_mask
'''

s = must_replace(
    s,
    old_eval_mask,
    new_eval_mask,
    "validation distillation mask",
)


# ============================================================
# 7. Training batches:
#    keep exact same sampled batch IDs / RNG.
#
#    Only teacher target and mask vary by E6 mode.
# ============================================================

old_train_teacher = '''            teacher = (
                torch.as_tensor(
                    train[
                        "teacher"
                    ][batch],
                    dtype=torch.float32,
                    device=X.device,
                )
            )
'''

new_train_teacher = '''            teacher_target = (
                torch.as_tensor(
                    train[
                        "teacher_target"
                    ][batch],
                    dtype=torch.float32,
                    device=X.device,
                )
            )

            distill_mask = (
                torch.as_tensor(
                    train[
                        "distill_mask"
                    ][batch],
                    dtype=torch.bool,
                    device=X.device,
                )
            )
'''

s = must_replace(
    s,
    old_train_teacher,
    new_train_teacher,
    "training teacher tensor",
)


old_train_mask = '''            mask = (
                (y > 0.5)
                |
                (
                    (y < 0.5)
                    & (
                        teacher
                        < TAU
                    )
                )
            )
'''

new_train_mask = '''            mask = distill_mask
'''

s = must_replace(
    s,
    old_train_mask,
    new_train_mask,
    "training distillation mask",
)


# Both validation and train MSE targets.
teacher_target_replacements = (
    s.count(
        "teacher[mask],"
    )
)

if teacher_target_replacements != 2:
    raise RuntimeError(
        "Expected exactly two teacher[mask] MSE targets; "
        f"found {teacher_target_replacements}"
    )

s = s.replace(
    "teacher[mask],",
    "teacher_target[mask],",
)


# ============================================================
# 8. Add provenance/config fields.
#
# IMPORTANT:
# Use single-line anchors from the real frozen 17b.
# Do not depend on whitespace-sensitive multi-line formatting.
# ============================================================

config_anchor = '        "model_seed": model_seed,\n'

if s.count(config_anchor) != 1:
    raise RuntimeError(
        "config model_seed anchor: expected exactly 1 match, "
        f"found {s.count(config_anchor)}"
    )

config_replacement = (
    '        "model_seed": model_seed,\n'
    '\n'
    '        "e6_ablation_mode":\n'
    '            ABLATION_MODE,\n'
    '\n'
    '        "e6_design":\n'
    '            "canonical_G2_split20260915_5seed",\n'
    '\n'
    '        "parent_trainer_sha256":\n'
    '            PARENT_17B_SHA256,\n'
    '\n'
    '        "train_ablation_audit":\n'
    '            train_ablation_audit,\n'
    '\n'
    '        "validation_ablation_audit":\n'
    '            val_ablation_audit,\n'
)

s = s.replace(
    config_anchor,
    config_replacement,
    1,
)


old_output_norm = (
    '        "output_normalization":\n'
    '            "L2",\n'
)

new_output_norm = (
    '        "output_normalization":\n'
    '            (\n'
    '                "none"\n'
    '                if ABLATION_MODE == "D5"\n'
    '                else "L2"\n'
    '            ),\n'
)

if s.count(old_output_norm) != 1:
    raise RuntimeError(
        "config output normalization: expected exactly 1 match, "
        f"found {s.count(old_output_norm)}"
    )

s = s.replace(
    old_output_norm,
    new_output_norm,
    1,
)


# ============================================================
# 9. Hard-stop accidental G3/G4 invocation.
# ============================================================

old_main = '''def main():

    a = args()
'''

new_main = '''def main():

    a = args()

    if a.protocol not in (
        "all",
        "G2",
    ):
        raise RuntimeError(
            "Formal E6 is locked to G2 only."
        )

    print(
        f"E6_ABLATION_MODE={ABLATION_MODE}"
    )

    print(
        "E6_PROTOCOL=G2"
    )

    print(
        "E6_SPLIT_SEED=20260915"
    )
'''

s = must_replace(
    s,
    old_main,
    new_main,
    "main G2 lock",
)


# ============================================================
# 10. Static parity gates:
#     these exact final-17b mechanisms MUST remain present.
# ============================================================

required_snippets = {
    "epoch_rng":
'''    epoch_seed = (
        model_seed
        + epoch * 1000003
    ) % (2**32)
''',

    "positive_sampling":
'''    p = rng.choice(
        pos,
        POS_PER_EPOCH,
        replace=True,
    )
''',

    "negative_sampling":
'''    n = rng.choice(
        neg,
        NEG_PER_EPOCH,
        replace=True,
    )
''',

    "bn_safe_forward":
'''            xpair = torch.cat(
                [
                    X[a],
                    X[b],
                ],
                dim=0,
            )

            zpair = model(
                xpair
            )

            za, zb = torch.chunk(
                zpair,
                2,
                dim=0,
            )
''',

    "contrastive_loss":
'''            loss_con = (
                y * d.square()
                +
                (1.0 - y)
                * F.relu(
                    MARGIN - d
                ).square()
            ).mean()
''',

    "total_loss":
'''            loss = (
                loss_con
                +
                LAMBDA_DIST
                * loss_dist
            )
''',

    "gradient_clip":
'''            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
''',

    "checkpoint_tiebreak":
'''        and new_con
        < best_con - EPS
''',
}


for name, snippet in required_snippets.items():

    if snippet not in s:
        raise RuntimeError(
            f"STATIC_PARITY_FAIL: missing {name}"
        )


# ============================================================
# 11. Verify constants stayed frozen.
# ============================================================

frozen_constant_snippets = [
    "INPUT_DIM = 2080",
    "H1 = 256",
    "H2 = 128",
    "EMBED_DIM = 64",
    "MARGIN = 1.0",
    "TAU = 0.98",
    "LAMBDA_DIST = 0.25",
    "LR = 5e-4",
    "WEIGHT_DECAY = 1e-4",
    "PAIR_BATCH = 512",
    "POS_PER_EPOCH = 50000",
    "NEG_PER_EPOCH = 50000",
    "MAX_EPOCHS = 50",
    "PATIENCE = 8",
    "GRAD_CLIP = 1.0",
    "EPS = 1e-8",
]


for snippet in frozen_constant_snippets:

    if snippet not in s:
        raise RuntimeError(
            "STATIC_PARITY_FAIL: "
            f"missing frozen constant {snippet}"
        )


# ============================================================
# 12. Compile before writing.
# ============================================================

compile(
    s,
    str(DST),
    "exec",
)


DST.write_text(
    s,
    encoding="utf-8",
)


dst_sha = sha256_file(
    DST
)


diff_text = "".join(
    difflib.unified_diff(
        original.splitlines(
            keepends=True
        ),
        s.splitlines(
            keepends=True
        ),
        fromfile=str(SRC),
        tofile=str(DST),
    )
)

DIFF.write_text(
    diff_text,
    encoding="utf-8",
)


report_lines = [
    "=" * 100,
    "HGV-V3 E6 TRAINER PATCH REPORT",
    "=" * 100,
    f"PARENT={SRC}",
    f"PARENT_SHA256={parent_sha}",
    f"OUTPUT={DST}",
    f"OUTPUT_SHA256={dst_sha}",
    "",
    "FORMAL_E6_SCOPE:",
    "  protocol=G2",
    "  split_seed=20260915",
    "",
    "MODES:",
    "  D0=no distillation",
    "  D1=all-pair distillation",
    "  D2=all-negative distillation",
    "  D3=final selective distillation",
    "  D4=D3 mask + label-stratified shuffled teacher target",
    "  D5=D3 without output L2 normalization",
    "",
    "MODEL_SEEDS:",
    "  D0/D1/D2/D4/D5=20260915..20260919",
    "  D3 new only=20260918,20260919",
    "  D3 seed15/16/17=reuse frozen main later",
    "",
    "UNCHANGED_CORE_GATES:",
    "  epoch RNG formula=PASS",
    "  50k positive sampling=PASS",
    "  50k negative sampling=PASS",
    "  BN-safe concatenated forward=PASS",
    "  contrastive loss=PASS",
    "  lambda=0.25=PASS",
    "  tau=0.98=PASS",
    "  checkpoint ranking/tiebreak=PASS",
    "  gradient clipping=PASS",
    "",
    "TEST_REFERENCED_DURING_TRAINING=NO",
    "",
    "PYTHON_COMPILE=PASS",
    "STATIC_PARITY_GATE=PASS",
    "PATCH_COMPLETE=PASS",
    "=" * 100,
]


REPORT.write_text(
    "\n".join(
        report_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    "\n".join(
        report_lines
    )
)
