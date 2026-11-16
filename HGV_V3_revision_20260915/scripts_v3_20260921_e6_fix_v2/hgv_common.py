#!/usr/bin/env python3

from pathlib import Path
import hashlib
import os
import re
import subprocess
import numpy as np


ACC_RE = re.compile(r"(GC[AF]_\d+\.\d+)")


def extract_acc(x):
    if x is None:
        return None
    m = ACC_RE.search(str(x))
    return m.group(1) if m else None


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_list(path):
    return [
        x.strip()
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def run_logged(cmd, stdout_path, stderr_path, check=True, cwd=None):
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        p = subprocess.run(
            [str(x) for x in cmd],
            stdout=out,
            stderr=err,
            cwd=cwd,
        )

    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed with return code "
            f"{p.returncode}:\n{' '.join(map(str, cmd))}\n"
            f"stderr: {stderr_path}"
        )

    return p.returncode


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_dashing_rectangular(
    path,
    query_accs,
    ref_accs,
):
    """
    Supports several Dashing2 human-readable rectangular layouts:
      1) query ref similarity
      2) ref query similarity
      3) query-label + Nref numeric columns
      4) pure Q x R numeric matrix
      5) header with reference labels + query rows
    """
    path = Path(path)

    qmap = {x: i for i, x in enumerate(query_accs)}
    rmap = {x: i for i, x in enumerate(ref_accs)}

    nq = len(query_accs)
    nr = len(ref_accs)

    M = np.full((nq, nr), np.nan, dtype=np.float32)

    lines = [
        x.rstrip("\n")
        for x in path.open(errors="replace")
        if x.strip()
    ]

    if not lines:
        raise RuntimeError(f"Empty Dashing2 output: {path}")

    filled = 0

    # ------------------------------------------------------------
    # Long format: q/ref/score or ref/q/score
    # ------------------------------------------------------------
    for line in lines:
        toks = line.split()

        acc_tokens = [
            (i, extract_acc(t))
            for i, t in enumerate(toks)
        ]

        qhits = [
            (i, a)
            for i, a in acc_tokens
            if a in qmap
        ]

        rhits = [
            (i, a)
            for i, a in acc_tokens
            if a in rmap
        ]

        if not qhits or not rhits:
            continue

        nums = [
            _to_float(t)
            for t in toks
        ]
        nums = [
            x for x in nums
            if x is not None
        ]

        if not nums:
            continue

        q = qhits[0][1]
        r = rhits[0][1]

        # Rectangular output is expected to contain one similarity.
        score = nums[-1]

        M[qmap[q], rmap[r]] = score
        filled += 1

    if filled >= int(0.8 * nq * nr):
        return M

    # ------------------------------------------------------------
    # Header reference labels + query rows.
    # ------------------------------------------------------------
    first = lines[0].split()

    header_refs = [
        extract_acc(x)
        for x in first
    ]

    header_ref_hits = [
        x for x in header_refs
        if x in rmap
    ]

    if len(header_ref_hits) >= int(0.8 * nr):
        cols = [
            rmap[x]
            for x in header_refs
            if x in rmap
        ]

        for line in lines[1:]:
            toks = line.split()

            q = extract_acc(toks[0])

            if q not in qmap:
                continue

            vals = [
                _to_float(x)
                for x in toks[1:]
            ]

            vals = [
                x for x in vals
                if x is not None
            ]

            if len(vals) != len(cols):
                continue

            M[qmap[q], cols] = np.asarray(
                vals,
                dtype=np.float32,
            )

        if np.isfinite(M).sum() >= int(0.8 * nq * nr):
            return M

    # ------------------------------------------------------------
    # One query label + R numeric columns.
    # ------------------------------------------------------------
    rows_seen = 0

    for line in lines:
        toks = line.split()

        if not toks:
            continue

        q = extract_acc(toks[0])

        if q not in qmap:
            continue

        vals = [
            _to_float(x)
            for x in toks[1:]
        ]

        vals = [
            x for x in vals
            if x is not None
        ]

        if len(vals) == nr:
            M[qmap[q], :] = np.asarray(
                vals,
                dtype=np.float32,
            )
            rows_seen += 1

    if rows_seen == nq:
        return M

    # ------------------------------------------------------------
    # Pure numeric matrix.
    # ------------------------------------------------------------
    allnums = []

    for line in lines:
        for t in line.split():
            x = _to_float(t)
            if x is not None:
                allnums.append(x)

    if len(allnums) == nq * nr:
        return np.asarray(
            allnums,
            dtype=np.float32,
        ).reshape(nq, nr)

    raise RuntimeError(
        "Unable to parse Dashing2 rectangular output.\n"
        f"path={path}\n"
        f"expected shape=({nq},{nr})\n"
        f"finite entries found={np.isfinite(M).sum()}"
    )


def parse_dashing_allpairs(path, ordered_accs):
    """
    Parse the frozen Dashing2 SetSketch symmetric matrix used by HGV-V3.

    Exact frozen format:
      line 1: comment
      line 2: options
      line 3: "#Sources" + N FASTA paths
      next N lines:
          row FASTA path + N matrix cells

    Dashing2 stores:
      diagonal/lower triangle = "-"
      upper triangle          = native similarity in [0,1]

    The returned matrix is reordered to ordered_accs and is fully symmetric.
    Diagonal similarity is set to 1.0.
    """
    path = Path(path)

    n = len(ordered_accs)

    if len(set(ordered_accs)) != n:
        raise RuntimeError(
            "ordered_accs contains duplicate accessions"
        )

    desired_index = {
        acc: i
        for i, acc in enumerate(ordered_accs)
    }

    M = np.full(
        (n, n),
        np.nan,
        dtype=np.float32,
    )

    np.fill_diagonal(
        M,
        1.0,
    )

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as f:

        line1 = f.readline().rstrip("\r\n")
        line2 = f.readline().rstrip("\r\n")
        line3 = f.readline().rstrip("\r\n")

        if not line1.startswith(
            "#Dashing2 Symmetric pairwise Output"
        ):
            raise RuntimeError(
                "Unexpected Dashing2 line 1:\n"
                f"{line1[:200]}"
            )

        if not line2.startswith(
            "#Dashing2Options:"
        ):
            raise RuntimeError(
                "Unexpected Dashing2 options line:\n"
                f"{line2[:200]}"
            )

        source_fields = line3.split("\t")

        if (
            not source_fields
            or source_fields[0] != "#Sources"
        ):
            raise RuntimeError(
                "Dashing2 #Sources header not found"
            )

        sources = source_fields[1:]

        if len(sources) != n:
            raise RuntimeError(
                f"Dashing2 source count={len(sources)}, "
                f"expected={n}"
            )

        source_accs = [
            extract_acc(x)
            for x in sources
        ]

        if any(x is None for x in source_accs):
            bad = [
                sources[i]
                for i, x in enumerate(source_accs)
                if x is None
            ]

            raise RuntimeError(
                "Could not extract accession from "
                f"Dashing2 source path(s): {bad[:5]}"
            )

        if len(set(source_accs)) != n:
            raise RuntimeError(
                "Duplicate accession in Dashing2 #Sources"
            )

        source_set = set(source_accs)
        desired_set = set(ordered_accs)

        if source_set != desired_set:
            missing = sorted(
                desired_set - source_set
            )

            extra = sorted(
                source_set - desired_set
            )

            raise RuntimeError(
                "Dashing2 source/accession set mismatch.\n"
                f"missing={missing[:10]}\n"
                f"extra={extra[:10]}"
            )

        upper_count = 0

        for row_i in range(n):

            line = f.readline()

            if not line:
                raise RuntimeError(
                    "Dashing2 matrix ended early at "
                    f"row {row_i}/{n}"
                )

            parts = (
                line.rstrip("\r\n")
                .split("\t")
            )

            if len(parts) != n + 1:
                raise RuntimeError(
                    f"Dashing2 row {row_i}: "
                    f"fields={len(parts)}, "
                    f"expected={n + 1}"
                )

            row_path = parts[0]

            if row_path != sources[row_i]:
                raise RuntimeError(
                    "Dashing2 row/source order mismatch "
                    f"at row {row_i}"
                )

            row_acc = source_accs[row_i]

            # Matrix columns 0..row_i include lower triangle + diagonal.
            lower_diag = parts[
                1:row_i + 2
            ]

            bad_lower = [
                (j, token)
                for j, token in enumerate(lower_diag)
                if token != "-"
            ]

            if bad_lower:
                raise RuntimeError(
                    f"Dashing2 row {row_i}: "
                    "lower/diagonal region contains "
                    f"non '-' token(s): {bad_lower[:5]}"
                )

            out_i = desired_index[
                row_acc
            ]

            # Upper triangle: columns row_i+1 .. N-1.
            for col_j in range(
                row_i + 1,
                n,
            ):

                token = parts[
                    col_j + 1
                ]

                if token == "-":
                    raise RuntimeError(
                        "Unexpected '-' in Dashing2 "
                        "upper triangle: "
                        f"row={row_i}, col={col_j}"
                    )

                try:
                    similarity = float(
                        token
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Invalid Dashing2 similarity: "
                        f"row={row_i}, col={col_j}, "
                        f"token={token!r}"
                    ) from e

                if not (
                    0.0
                    <= similarity
                    <= 1.0
                ):
                    raise RuntimeError(
                        "Dashing2 similarity outside "
                        f"[0,1]: {similarity}"
                    )

                col_acc = source_accs[
                    col_j
                ]

                out_j = desired_index[
                    col_acc
                ]

                M[
                    out_i,
                    out_j,
                ] = similarity

                M[
                    out_j,
                    out_i,
                ] = similarity

                upper_count += 1

        extra_nonempty = 0

        for line in f:
            if line.strip():
                extra_nonempty += 1

        if extra_nonempty:
            raise RuntimeError(
                "Unexpected extra Dashing2 "
                f"matrix rows: {extra_nonempty}"
            )

    expected_upper = (
        n
        * (n - 1)
        // 2
    )

    if upper_count != expected_upper:
        raise RuntimeError(
            "Dashing2 upper-triangle count mismatch: "
            f"{upper_count} != {expected_upper}"
        )

    finite = np.isfinite(M)

    if finite.sum() != n * n:
        raise RuntimeError(
            "Dashing2 reconstructed matrix incomplete: "
            f"finite={finite.sum()}, expected={n*n}"
        )

    if not np.allclose(
        M,
        M.T,
        rtol=0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Dashing2 reconstructed matrix "
            "is not symmetric"
        )

    if not np.allclose(
        np.diag(M),
        1.0,
        rtol=0,
        atol=0,
    ):
        raise RuntimeError(
            "Dashing2 diagonal is not exactly 1"
        )

    mn = float(
        np.min(M)
    )

    mx = float(
        np.max(M)
    )

    print(
        "Dashing2 frozen SetSketch parse PASS"
    )
    print(
        "  sources          =",
        n,
    )
    print(
        "  upper pairs      =",
        upper_count,
    )
    print(
        "  expected pairs   =",
        expected_upper,
    )
    print(
        "  matrix shape     =",
        M.shape,
    )
    print(
        "  finite cells     =",
        int(
            finite.sum()
        ),
    )
    print(
        "  similarity min   =",
        mn,
    )
    print(
        "  similarity max   =",
        mx,
    )

    return M

