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
    Parses common PHYLIP/full/lower-triangle Dashing2 all-pair output.
    Returns symmetric similarity matrix.
    """
    path = Path(path)

    n = len(ordered_accs)
    amap = {x: i for i, x in enumerate(ordered_accs)}

    lines = [
        x.rstrip("\n")
        for x in path.open(errors="replace")
        if x.strip()
    ]

    if not lines:
        raise RuntimeError(f"Empty Dashing2 all-pair file: {path}")

    # PHYLIP often begins with N.
    start = 0

    first = lines[0].split()

    if (
        len(first) == 1
        and first[0].isdigit()
    ):
        declared = int(first[0])
        if declared != n:
            raise RuntimeError(
                f"Dashing2 matrix declares N={declared}, expected {n}"
            )
        start = 1

    data_lines = lines[start:]

    M = np.full(
        (n, n),
        np.nan,
        dtype=np.float32,
    )

    np.fill_diagonal(M, 1.0)

    recognized_rows = 0

    for sequential_i, line in enumerate(data_lines):
        toks = line.split()

        if not toks:
            continue

        a = extract_acc(toks[0])

        if a in amap:
            i = amap[a]
            vals0 = toks[1:]
        else:
            i = sequential_i
            vals0 = toks

        vals = [
            _to_float(x)
            for x in vals0
        ]

        if any(x is None for x in vals):
            continue

        vals = [
            float(x)
            for x in vals
        ]

        if len(vals) == n:
            M[i, :] = np.asarray(
                vals,
                dtype=np.float32,
            )
            recognized_rows += 1

        elif len(vals) == i + 1:
            M[i, :i + 1] = vals
            M[:i + 1, i] = vals
            recognized_rows += 1

        elif len(vals) == i:
            if i > 0:
                M[i, :i] = vals
                M[:i, i] = vals
            M[i, i] = 1.0
            recognized_rows += 1

    if np.isfinite(M).sum() == n * n:
        return M

    # Long pair format fallback.
    M2 = np.full(
        (n, n),
        np.nan,
        dtype=np.float32,
    )
    np.fill_diagonal(M2, 1.0)

    count = 0

    for line in lines:
        toks = line.split()

        accs = [
            extract_acc(x)
            for x in toks
        ]

        hits = [
            x for x in accs
            if x in amap
        ]

        if len(hits) < 2:
            continue

        nums = [
            _to_float(x)
            for x in toks
        ]
        nums = [
            x for x in nums
            if x is not None
        ]

        if not nums:
            continue

        a, b = hits[0], hits[1]
        s = float(nums[-1])

        ia = amap[a]
        ib = amap[b]

        M2[ia, ib] = s
        M2[ib, ia] = s

        count += 1

    expected_edges = n * (n - 1) // 2

    if count >= int(0.95 * expected_edges):
        return M2

    raise RuntimeError(
        "Unable to reconstruct Dashing2 all-pair matrix.\n"
        f"path={path}\n"
        f"recognized_rows={recognized_rows}\n"
        f"long_edges={count}\n"
        f"expected_edges={expected_edges}"
    )
