#!/usr/bin/env python3
import os, tempfile
from pathlib import Path
import pandas as pd

# Avoid dependence on the real server root during the unit test.
os.environ.setdefault('HGV_BASE', '/tmp/hgv_v5_unit_test_nonexistent')
from common import load_dashing2_source_accessions, load_dashing_teacher_for_pairs

sources = [
    'GCF_000000001.1',
    'GCF_000000002.1',
    'GCF_000000003.1',
    'GCF_000000004.1',
]
paths = [f'/x/{a}/{a}_genomic.fna.gz' for a in sources]
# Full-width symmetric output with upper-triangle similarities only.
rows = [
    [paths[0], '-', '0.90', '0.20', '0.00'],
    [paths[1], '-', '-',    '0.75', '0.10'],
    [paths[2], '-', '-',    '-',    '0.95'],
    [paths[3], '-', '-',    '-',    '-'],
]
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 'dashing.tsv'
    with p.open('w') as f:
        f.write('#Dashing2 Symmetric pairwise Output\n')
        f.write('#Dashing2Options: Dashing2Options;k:31;sketchsize:1024;sketchtype:onepermsetsketch;canon\n')
        f.write('#Sources\t' + '\t'.join(paths) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')
    got_sources = load_dashing2_source_accessions(p)
    assert got_sources == sources, (got_sources, sources)
    q = pd.DataFrame({
        'accession_a': [sources[0], sources[2], sources[3], sources[0]],
        'accession_b': [sources[1], sources[1], sources[2], sources[3]],
    })
    d = load_dashing_teacher_for_pairs(q, p)
    expected = [0.10, 0.25, 0.05, 1.00]
    assert len(d) == len(expected)
    for x, y in zip(d.tolist(), expected):
        assert abs(x-y) < 1e-6, (d, expected)
print('DASHING_MATRIX_UNIT_TEST=PASS')
