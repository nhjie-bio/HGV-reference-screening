#!/usr/bin/env python3
from pathlib import Path
import tempfile
import common

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    d=root/'G2_seed20260915'
    d.mkdir(parents=True)
    expected={
        'train':'train_unique_pair_pool.tsv',
        'val':'val_pairs.tsv',
        'test':'test_pairs.tsv',
    }
    for name in expected.values():
        (d/name).write_text('accession_a\taccession_b\tlabel\n', encoding='utf-8')
    old=common.PAIR_ROOT
    common.PAIR_ROOT=root
    try:
        for role,name in expected.items():
            got=common.find_pair_file('G2',20260915,role)
            assert got == d/name, (role,got,d/name)
    finally:
        common.PAIR_ROOT=old
print('PAIR_LAYOUT_UNIT_TEST=PASS')
