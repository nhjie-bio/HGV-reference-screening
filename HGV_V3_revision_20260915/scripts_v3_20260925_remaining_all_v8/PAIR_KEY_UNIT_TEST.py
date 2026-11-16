#!/usr/bin/env python3
from common import normalize_accession_id, canonical_pair_key

cases = [
    ('GCF_000001.1', 'GCA_000002.2'),
    ('RS_GCF_000001.1', 'GB_GCA_000002.2'),
    ('/tmp/GCF_000001.1_genomic.fna.gz', '/x/GCA_000002.2/file.fna'),
]
keys = [canonical_pair_key(a,b) for a,b in cases]
assert keys[0] == keys[1] == keys[2], keys
assert normalize_accession_id('RS_GCF_000001.1') == 'GCF_000001.1'
print('PAIR_KEY_UNIT_TEST=PASS')
