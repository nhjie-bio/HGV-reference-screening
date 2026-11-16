#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# =============================================================================
# HGV-V3 E5 external baseline completeness audit
#
# READ-ONLY with respect to all existing project results.
# The only files written are NEW audit reports under:
#
#   09_baselines/E5_external_completeness_audit_v1_20260919/run_*/
#
# This script DOES NOT:
#   - train anything
#   - run Mash/Dashing2/FastANI/skani/DNABERT-S
#   - modify HGV checkpoints
#   - modify thresholds
#   - modify pair manifests
#   - modify B1/B2 results
#   - overwrite old scripts/results
# =============================================================================


BASE = Path(
    os.environ.get(
        "HGV_BASE",
        "HGV_V3_revision_20260915"
    )
).resolve()


# -----------------------------------------------------------------------------
# Locations that may contain actual completed result files
# -----------------------------------------------------------------------------

RESULT_ROOTS = [
    BASE / "09_external_validation",
    BASE / "09_baselines",
]

# Search scripts separately.
# Script existence is only "capability evidence" and NEVER counts as completion.
SCRIPT_ROOTS = sorted(
    p for p in BASE.glob("scripts*")
    if p.is_dir()
)


# -----------------------------------------------------------------------------
# Methods
#
# Core external methods are already expected:
#   HGV / RawK6 / PCA64
#
# Main audit target:
#   Mash / Dashing2 / FastANI / skani / DNABERT-S
# -----------------------------------------------------------------------------

METHODS = {
    "HGV": [
        r"(?<![a-z0-9])hgv(?![a-z0-9])",
    ],

    "RawK6": [
        r"raw[_-]?k6",
        r"raw6mer",
        r"raw[_-]?6mer",
    ],

    "PCA64": [
        r"pca[_-]?64",
        r"pca64",
    ],

    "Mash": [
        r"(?<![a-z0-9])mash(?![a-z0-9])",
    ],

    "Dashing2": [
        r"dashing2",
        r"dashing[_-]?2",
        r"dashing",
    ],

    "FastANI": [
        r"fastani",
        r"fast[_-]?ani",
    ],

    "skani": [
        r"(?<![a-z0-9])skani(?![a-z0-9])",
    ],

    "DNABERT-S": [
        r"dnabert[_-]?s",
        r"dnaberts",
        r"dna[_-]?bert",
    ],
}


# -----------------------------------------------------------------------------
# External endpoints
# -----------------------------------------------------------------------------

ENDPOINTS = {
    "B1": [
        r"(?<![a-z0-9])b1(?![a-z0-9])",
        r"matched[_ -]?species",
        r"(?<![a-z0-9])cd1(?![a-z0-9])",
    ],

    "B2_seen": [
        r"b2[_ -]?seen",
        r"(?<!un)seen[_ -]?genus",
        r"b2.*(?<!un)seen.*genus",
        r"(?<![a-z0-9])cd2a(?![a-z0-9])",
    ],

    "B2_unseen": [
        r"b2[_ -]?unseen",
        r"unseen[_ -]?genus",
        r"b2.*unseen.*genus",
        r"(?<![a-z0-9])cd2b(?![a-z0-9])",
    ],
}


# -----------------------------------------------------------------------------
# Retrieval-result indicators
# -----------------------------------------------------------------------------

RETRIEVAL_PATTERNS = [
    r"retrieval",
    r"recall@",
    r"\br@\d+",
    r"\bmrr\b",
    r"top[_ -]?k",
    r"nearest",
    r"rank",
    r"query[_ -]?level",
    r"query[_ -]?id",
    r"reference[_ -]?(database|id|accession)",
    r"candidate[_ -]?reduction",
]


RESULT_HINT_PATTERNS = [
    r"metric",
    r"result",
    r"report",
    r"evaluation",
    r"eval",
    r"retrieval",
    r"summary",
    r"query",
    r"rank",
    r"recall",
    r"mrr",
    r"prediction",
]


INTERNAL_HINT_PATTERNS = [
    r"internal",
    r"blind[_ -]?test",
    r"g2",
    r"g3",
    r"g4",
    r"unified",
    r"threshold[_ -]?freeze",
    r"native",
]


SOURCE_CODE_EXT = {
    ".py",
    ".sh",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
}


TEXT_EXT = {
    ".txt",
    ".tsv",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".yaml",
    ".yml",
    ".out",
    ".sha256",
    ".ini",
    ".cfg",
}


COMPRESSED_TEXT_SUFFIXES = {
    ".gz",
}


# Only first 8 MB of a text-like file is inspected.
# This prevents accidentally reading very large matrices/results into RAM.
MAX_TEXT_BYTES = 8 * 1024 * 1024


CORE_METHODS = [
    "HGV",
    "RawK6",
    "PCA64",
]


E5_GAP_METHODS = [
    "Mash",
    "Dashing2",
    "FastANI",
    "skani",
    "DNABERT-S",
]


# =============================================================================
# Helpers
# =============================================================================

def rx_any(patterns, text):
    return any(
        re.search(p, text, flags=re.I | re.S)
        for p in patterns
    )


def normalize_blob(path: Path, text: str) -> str:
    """
    IMPORTANT:
    Use path relative to project root.

    Do NOT use absolute path for method detection, because project root itself
    contains 'HGV_V3', which would otherwise falsely mark every file as HGV.
    """
    try:
        rel = path.relative_to(BASE)
    except Exception:
        rel = path

    return str(rel).lower() + "\n" + text.lower()


def safe_read_text(path: Path) -> str:
    """
    Read text-like files only.
    Large/binary files are NOT parsed.
    """

    try:
        # gzip text files
        if path.suffix.lower() in COMPRESSED_TEXT_SUFFIXES:
            with gzip.open(
                path,
                "rt",
                encoding="utf-8",
                errors="replace",
            ) as fh:
                return fh.read(MAX_TEXT_BYTES)

        # normal text or source-code files
        if (
            path.suffix.lower() in (TEXT_EXT | SOURCE_CODE_EXT)
            or path.name.lower().endswith(
                (
                    ".tsv.gz",
                    ".csv.gz",
                    ".txt.gz",
                    ".json.gz",
                )
            )
        ):
            with path.open(
                "rt",
                encoding="utf-8",
                errors="replace",
            ) as fh:
                return fh.read(MAX_TEXT_BYTES)

    except Exception:
        return ""

    return ""


def file_is_result_like(path: Path, blob: str) -> bool:
    """
    Source code can never count as result evidence.
    """

    if path.suffix.lower() in SOURCE_CODE_EXT:
        return False

    name = path.name.lower()

    if rx_any(RESULT_HINT_PATTERNS, name):
        return True

    if (
        path.suffix.lower() in TEXT_EXT
        or path.suffix.lower() in COMPRESSED_TEXT_SUFFIXES
    ):
        return rx_any(RETRIEVAL_PATTERNS, blob)

    return False


def external_context(path: Path, blob: str) -> bool:
    """
    Formal external context.

    Everything under 09_external_validation is potentially external.

    For 09_baselines, the DIRECTORY/PATH itself must indicate external/B1/B2.
    Merely mentioning 'external' inside an internal report is NOT enough.
    """

    p = str(path).lower()

    if "09_external_validation" in p:
        return True

    if "09_baselines" in p:
        if rx_any(
            [
                r"external",
                r"(?<![a-z0-9])b1(?![a-z0-9])",
                r"(?<![a-z0-9])b2(?![a-z0-9])",
                r"b2[_-]?(seen|unseen)",
            ],
            p,
        ):
            return True

    return False


def method_hits(blob: str):
    out = []

    for method, pats in METHODS.items():
        if rx_any(pats, blob):
            out.append(method)

    return out


def endpoint_hits(blob: str):
    out = []

    for endpoint, pats in ENDPOINTS.items():
        if rx_any(pats, blob):
            out.append(endpoint)

    return out


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def ensure_base():
    if not BASE.exists():
        print(
            f"ERROR: project root does not exist: {BASE}",
            file=sys.stderr,
        )
        sys.exit(2)

    for root in RESULT_ROOTS:
        if not root.exists():
            print(
                f"WARNING: expected result root missing: {root}",
                file=sys.stderr,
            )


def make_outdir():
    """
    Every execution gets a NEW run directory.
    Nothing is overwritten.
    """

    parent = (
        BASE
        / "09_baselines"
        / "E5_external_completeness_audit_v1_20260919"
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    out = parent / stamp

    suffix = 0

    while out.exists():
        suffix += 1
        out = parent / f"{stamp}_{suffix:02d}"

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    return out


def iter_files(roots):
    """
    Iterate files without changing anything.

    The audit output directory itself is explicitly excluded so that
    rerunning this audit cannot accidentally treat a previous audit report
    as experimental evidence.
    """

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            # Never audit our own audit report.
            if (
                "e5_external_completeness_audit"
                in str(p).lower()
            ):
                continue

            try:
                rp = p.resolve()
            except Exception:
                rp = p

            if rp in seen:
                continue

            seen.add(rp)

            yield p


# =============================================================================
# Main audit
# =============================================================================

def main():

    ensure_base()

    outdir = make_outdir()

    evidence_rows = []

    endpoint_evidence = defaultdict(
        lambda: defaultdict(list)
    )

    internal_evidence = defaultdict(list)

    script_evidence = defaultdict(list)

    tree_rows = []


    # =========================================================================
    # 1. Scan result directories
    # =========================================================================

    for path in iter_files(RESULT_ROOTS):

        try:
            st = path.stat()
        except OSError:
            continue

        text = safe_read_text(path)

        blob = normalize_blob(
            path,
            text,
        )

        methods = method_hits(blob)

        endpoints = endpoint_hits(blob)

        has_retrieval = rx_any(
            RETRIEVAL_PATTERNS,
            blob,
        )

        is_result = file_is_result_like(
            path,
            blob,
        )

        is_external = external_context(
            path,
            blob,
        )


        # ---------------------------------------------------------------------
        # File snapshot for manual follow-up
        # ---------------------------------------------------------------------

        if methods or endpoints or has_retrieval:

            tree_rows.append(
                {
                    "path": str(path),
                    "size_bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(
                        st.st_mtime
                    ).isoformat(
                        timespec="seconds"
                    ),
                    "methods": ",".join(methods),
                    "endpoints": ",".join(endpoints),
                    "retrieval_marker": int(
                        has_retrieval
                    ),
                    "external_context": int(
                        is_external
                    ),
                    "result_like": int(
                        is_result
                    ),
                }
            )


        # ---------------------------------------------------------------------
        # FORMAL external retrieval evidence
        #
        # To count:
        #   external context
        #   + result-like file
        #   + retrieval marker
        #   + identifiable method
        #   + identifiable endpoint
        #
        # Script files cannot enter this block.
        # ---------------------------------------------------------------------

        if (
            is_external
            and is_result
            and has_retrieval
            and methods
            and endpoints
        ):

            for method in methods:

                for endpoint in endpoints:

                    endpoint_evidence[
                        method
                    ][
                        endpoint
                    ].append(
                        str(path)
                    )

                    evidence_rows.append(
                        {
                            "method": method,
                            "endpoint": endpoint,
                            "evidence_type":
                                "FORMAL_EXTERNAL_RETRIEVAL_RESULT",
                            "path": str(path),
                            "size_bytes": st.st_size,
                            "mtime":
                                datetime.fromtimestamp(
                                    st.st_mtime
                                ).isoformat(
                                    timespec="seconds"
                                ),
                        }
                    )


        # ---------------------------------------------------------------------
        # Internal baseline evidence
        #
        # Useful for distinguishing:
        #
        #   "method was run internally"
        #
        # versus
        #
        #   "method not found anywhere"
        #
        # Internal evidence NEVER counts as external completion.
        # ---------------------------------------------------------------------

        if (
            "09_baselines"
            in str(path).lower()
            and methods
            and not is_external
        ):

            if rx_any(
                INTERNAL_HINT_PATTERNS,
                blob,
            ):

                for method in methods:
                    internal_evidence[
                        method
                    ].append(
                        str(path)
                    )


    # =========================================================================
    # 2. Scan scripts
    #
    # An existing external-retrieval script means only:
    #     "code appears to exist"
    #
    # It DOES NOT mean:
    #     "the experiment was executed"
    # =========================================================================

    for path in iter_files(SCRIPT_ROOTS):

        if path.suffix.lower() not in SOURCE_CODE_EXT:
            continue

        text = safe_read_text(path)

        if not text:
            continue

        blob = normalize_blob(
            path,
            text,
        )

        methods = method_hits(blob)

        if (
            methods
            and rx_any(
                [
                    r"external",
                    r"(?<![a-z0-9])b1(?![a-z0-9])",
                    r"(?<![a-z0-9])b2(?![a-z0-9])",
                ],
                blob,
            )
            and rx_any(
                RETRIEVAL_PATTERNS,
                blob,
            )
        ):

            for method in methods:

                script_evidence[
                    method
                ].append(
                    str(path)
                )


    # =========================================================================
    # 3. De-duplicate evidence paths
    # =========================================================================

    for store in (
        internal_evidence,
        script_evidence,
    ):

        for method in list(store):
            store[method] = sorted(
                set(
                    store[method]
                )
            )


    for method in endpoint_evidence:

        for ep in endpoint_evidence[method]:

            endpoint_evidence[
                method
            ][
                ep
            ] = sorted(
                set(
                    endpoint_evidence[
                        method
                    ][
                        ep
                    ]
                )
            )


    # =========================================================================
    # 4. Build method-level completion table
    # =========================================================================

    summary_rows = []

    all_methods = (
        CORE_METHODS
        + E5_GAP_METHODS
    )


    for method in all_methods:

        ep_flags = {
            ep:
                bool(
                    endpoint_evidence[
                        method
                    ][
                        ep
                    ]
                )
            for ep in ENDPOINTS
        }

        n_ep = sum(
            ep_flags.values()
        )


        if n_ep == len(ENDPOINTS):

            status = (
                "COMPLETE_EXTERNAL_RETRIEVAL"
            )

        elif n_ep > 0:

            status = (
                "PARTIAL_EXTERNAL_RETRIEVAL"
            )

        elif internal_evidence.get(method):

            status = (
                "INTERNAL_ONLY_NO_EXTERNAL_RETRIEVAL"
            )

        elif script_evidence.get(method):

            status = (
                "SCRIPT_EXISTS_NO_EXTERNAL_RESULT"
            )

        else:

            status = (
                "NO_RESULT_EVIDENCE"
            )


        all_external_files = []

        for ep in ENDPOINTS:
            all_external_files.extend(
                endpoint_evidence[
                    method
                ][
                    ep
                ]
            )


        summary_rows.append(
            {
                "method": method,

                "B1":
                    "YES"
                    if ep_flags["B1"]
                    else "NO",

                "B2_seen":
                    "YES"
                    if ep_flags["B2_seen"]
                    else "NO",

                "B2_unseen":
                    "YES"
                    if ep_flags["B2_unseen"]
                    else "NO",

                "status": status,

                "n_external_evidence_files":
                    len(
                        set(
                            all_external_files
                        )
                    ),

                "n_internal_evidence_files":
                    len(
                        internal_evidence.get(
                            method,
                            [],
                        )
                    ),

                "n_external_retrieval_scripts":
                    len(
                        script_evidence.get(
                            method,
                            [],
                        )
                    ),
            }
        )


    # =========================================================================
    # 5. E5 external gate
    #
    # The actual unresolved E5 methods are:
    #
    #   Mash
    #   Dashing2
    #   FastANI
    #   skani
    #   DNABERT-S
    #
    # Core HGV/Raw/PCA are included in the report as sanity controls.
    # =========================================================================

    gap_methods = []


    for method in E5_GAP_METHODS:

        row = next(
            r
            for r in summary_rows
            if r["method"] == method
        )

        missing = [
            ep
            for ep in ENDPOINTS
            if row[ep] == "NO"
        ]


        if missing:

            gap_methods.append(
                (
                    method,
                    missing,
                    row["status"],
                )
            )


    if not gap_methods:

        gate = (
            "PASS_E5_EXTERNAL_COMPLETE"
        )

    else:

        gate = (
            "FAIL_E5_EXTERNAL_GAPS_REMAIN"
        )


    # =========================================================================
    # 6. Write summary table
    # =========================================================================

    summary_tsv = (
        outdir
        / "E5_EXTERNAL_COMPLETENESS.tsv"
    )


    with summary_tsv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        w = csv.DictWriter(
            fh,
            fieldnames=list(
                summary_rows[0].keys()
            ),
            delimiter="\t",
        )

        w.writeheader()

        w.writerows(
            summary_rows
        )


    # =========================================================================
    # 7. Formal evidence file table
    # =========================================================================

    evidence_tsv = (
        outdir
        / "E5_EXTERNAL_EVIDENCE_FILES.tsv"
    )


    with evidence_tsv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        fields = [
            "method",
            "endpoint",
            "evidence_type",
            "path",
            "size_bytes",
            "mtime",
        ]

        w = csv.DictWriter(
            fh,
            fieldnames=fields,
            delimiter="\t",
        )

        w.writeheader()

        w.writerows(
            sorted(
                evidence_rows,
                key=lambda r: (
                    r["method"],
                    r["endpoint"],
                    r["path"],
                ),
            )
        )


    # =========================================================================
    # 8. Relevant file snapshot
    #
    # This is useful if automatic classification gives an ambiguous result.
    # =========================================================================

    tree_tsv = (
        outdir
        / "RELEVANT_FILE_SNAPSHOT.tsv"
    )


    with tree_tsv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        fields = [
            "path",
            "size_bytes",
            "mtime",
            "methods",
            "endpoints",
            "retrieval_marker",
            "external_context",
            "result_like",
        ]

        w = csv.DictWriter(
            fh,
            fieldnames=fields,
            delimiter="\t",
        )

        w.writeheader()

        w.writerows(
            sorted(
                tree_rows,
                key=lambda r:
                    r["path"],
            )
        )


    # =========================================================================
    # 9. Gap-only output
    # =========================================================================

    gap_txt = (
        outdir
        / "E5_EXTERNAL_GAPS.txt"
    )


    with gap_txt.open(
        "w",
        encoding="utf-8",
    ) as fh:

        fh.write(
            f"E5_EXTERNAL_GATE\t{gate}\n"
        )


        if not gap_methods:

            fh.write(
                "No gap among "
                "Mash/Dashing2/FastANI/skani/DNABERT-S "
                "for B1/B2_seen/B2_unseen retrieval evidence.\n"
            )

        else:

            for (
                method,
                missing,
                status,
            ) in gap_methods:

                fh.write(
                    f"{method}\t"
                    f"missing={','.join(missing)}\t"
                    f"status={status}\n"
                )


    # =========================================================================
    # 10. Human-readable report
    # =========================================================================

    report = (
        outdir
        / "E5_EXTERNAL_AUDIT_REPORT.txt"
    )


    with report.open(
        "w",
        encoding="utf-8",
    ) as fh:

        fh.write(
            "=" * 96
            + "\n"
        )

        fh.write(
            "HGV-V3 E5 EXTERNAL BASELINE COMPLETENESS AUDIT v1\n"
        )

        fh.write(
            "=" * 96
            + "\n"
        )

        fh.write(
            f"Project root : {BASE}\n"
        )

        fh.write(
            "Audit time   : "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
        )

        fh.write(
            "Mode         : READ-ONLY scan of existing project results; "
            "writes only this new audit directory\n"
        )

        fh.write(
            f"Gate         : {gate}\n\n"
        )


        fh.write(
            "Definition of completion:\n"
        )

        fh.write(
            "  Formal result evidence must be found for the method in each "
            "B1/B2_seen/B2_unseen external retrieval task.\n"
        )

        fh.write(
            "  Internal G2/G3/G4 evidence and an existing script do NOT "
            "count as external completion.\n\n"
        )


        header = (
            f"{'Method':<12} "
            f"{'B1':<4} "
            f"{'B2seen':<7} "
            f"{'B2unseen':<9} "
            f"{'Status'}\n"
        )

        fh.write(
            header
        )

        fh.write(
            "-" * 96
            + "\n"
        )


        for r in summary_rows:

            fh.write(
                f"{r['method']:<12} "
                f"{r['B1']:<4} "
                f"{r['B2_seen']:<7} "
                f"{r['B2_unseen']:<9} "
                f"{r['status']}\n"
            )


        fh.write(
            "\nGAPS THAT BLOCK E5 EXTERNAL FREEZE\n"
        )

        fh.write(
            "-" * 96
            + "\n"
        )


        if gap_methods:

            for (
                method,
                missing,
                status,
            ) in gap_methods:

                fh.write(
                    f"{method}: missing "
                    f"{', '.join(missing)} "
                    f"({status})\n"
                )

        else:

            fh.write(
                "NONE\n"
            )


        # ---------------------------------------------------------------------
        # Known external core source-of-truth
        # ---------------------------------------------------------------------

        fh.write(
            "\nCORE EXTERNAL SOURCE-OF-TRUTH CHECK\n"
        )

        fh.write(
            "-" * 96
            + "\n"
        )


        expected_core = [

            BASE
            / "09_external_validation"
            / "external_one_shot_evaluation_v1_20260917"
            / "EXTERNAL_EVALUATION_REPORT.txt",

            BASE
            / "09_external_validation"
            / "external_one_shot_evaluation_v1_20260917"
            / "external_metrics_seed_level.tsv",

            BASE
            / "09_external_validation"
            / "external_one_shot_evaluation_v1_20260917"
            / "external_metrics_summary_long.tsv",
        ]


        for p in expected_core:

            fh.write(
                f"{'FOUND' if p.exists() else 'MISSING'}\t"
                f"{p}\n"
            )


        # ---------------------------------------------------------------------
        # Evidence paths
        # ---------------------------------------------------------------------

        fh.write(
            "\nTOP FORMAL EXTERNAL EVIDENCE FILES BY METHOD/ENDPOINT\n"
        )

        fh.write(
            "-" * 96
            + "\n"
        )


        for method in all_methods:

            fh.write(
                f"\n[{method}]\n"
            )

            for ep in ENDPOINTS:

                files = endpoint_evidence[
                    method
                ][
                    ep
                ]

                fh.write(
                    f"  {ep}: "
                    f"{len(files)} file(s)\n"
                )

                for p in files[:8]:

                    fh.write(
                        f"    - {p}\n"
                    )

                if len(files) > 8:

                    fh.write(
                        f"    ... "
                        f"+{len(files) - 8} more\n"
                    )


        # ---------------------------------------------------------------------
        # Existing scripts - not evidence of execution
        # ---------------------------------------------------------------------

        fh.write(
            "\nEXTERNAL-RETRIEVAL SCRIPT EVIDENCE "
            "(CAPABILITY ONLY; NOT COMPLETION)\n"
        )

        fh.write(
            "-" * 96
            + "\n"
        )


        for method in all_methods:

            files = script_evidence.get(
                method,
                [],
            )

            fh.write(
                f"{method}: "
                f"{len(files)} script(s)\n"
            )

            for p in files[:6]:

                fh.write(
                    f"  - {p}\n"
                )


        fh.write(
            "\nINTERPRETATION RULE\n"
        )

        fh.write(
            "-" * 96
            + "\n"
        )

        fh.write(
            "If Gate=PASS_E5_EXTERNAL_COMPLETE, "
            "E5 external can be reviewed for freeze.\n"
        )

        fh.write(
            "If Gate=FAIL_E5_EXTERNAL_GAPS_REMAIN, "
            "only the listed missing method/endpoints should be run next.\n"
        )

        fh.write(
            "Do not alter frozen HGV teacher/tau/lambda/"
            "checkpoints/thresholds based on this audit.\n"
        )


    # =========================================================================
    # 11. Hash only newly generated audit outputs
    # =========================================================================

    sha_file = (
        outdir
        / "AUDIT_OUTPUT_SHA256.txt"
    )


    outputs = [
        summary_tsv,
        evidence_tsv,
        tree_tsv,
        gap_txt,
        report,
    ]


    with sha_file.open(
        "w",
        encoding="utf-8",
    ) as fh:

        for p in outputs:

            fh.write(
                f"{sha256(p)}  "
                f"{p.name}\n"
            )


    # =========================================================================
    # 12. Terminal summary
    # =========================================================================

    print(
        "=" * 96
    )

    print(
        "HGV-V3 E5 EXTERNAL BASELINE COMPLETENESS AUDIT"
    )

    print(
        "=" * 96
    )

    print(
        f"BASE    : {BASE}"
    )

    print(
        f"OUTDIR  : {outdir}"
    )

    print(
        f"GATE    : {gate}"
    )

    print()


    for r in summary_rows:

        print(
            f"{r['method']:<12} "
            f"B1={r['B1']:<3} "
            f"B2_seen={r['B2_seen']:<3} "
            f"B2_unseen={r['B2_unseen']:<3} "
            f"{r['status']}"
        )


    print()


    if gap_methods:

        print(
            "GAPS:"
        )

        for (
            method,
            missing,
            status,
        ) in gap_methods:

            print(
                f"  - {method}: "
                f"{', '.join(missing)} "
                f"[{status}]"
            )

    else:

        print(
            "GAPS: NONE"
        )


    print()

    print(
        "Primary outputs:"
    )

    print(
        f"  {report}"
    )

    print(
        f"  {summary_tsv}"
    )

    print(
        f"  {gap_txt}"
    )


if __name__ == "__main__":
    main()
