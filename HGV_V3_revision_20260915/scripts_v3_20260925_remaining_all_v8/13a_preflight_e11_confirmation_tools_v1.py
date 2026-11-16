#!/usr/bin/env python3
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from common import *

OUT = BASE / f'logs/E11_confirmation_preflight_{DATE_TAG}'
PASS = 'E11_CONFIRMATION_TOOLS_PREFLIGHT_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
probe=None
for _,r in manifest.iterrows():
    if infer_cohort(r,mc) == 'A':
        probe=str(r[mc['accession']]); break
if probe is None:
    die('Could not resolve an A-cohort probe genome')
f=fmap[probe]
if not f.exists():
    die(f'Probe FASTA missing: {f}')

sk_ani,sk_af1,sk_af2,sk_status=run_skani_pair(f,f)
if sk_status != 'OK' or not np.isfinite(sk_ani) or sk_ani < 99.0:
    die(f'skani self-comparison gate failed: accession={probe} status={sk_status} ANI={sk_ani}')

with tempfile.TemporaryDirectory(prefix='e11_preflight_fastani_') as td:
    fa_ani,fa_af1,fa_af2,fa_status=run_fastani_pair(f,f,Path(td)/'self.tsv')
if fa_status != 'OK' or not np.isfinite(fa_ani) or fa_ani < 99.0:
    die(f'FastANI self-comparison gate failed: accession={probe} status={fa_status} ANI={fa_ani}')

atomic_text(OUT/'E11_CONFIRMATION_TOOLS_PREFLIGHT.txt',
    '\n'.join([
        'HGV-V3 E11 CONFIRMATION TOOLS PREFLIGHT',
        f'probe_accession={probe}',
        f'probe_fasta={f}',
        f'skani_status={sk_status}',
        f'skani_ANI={sk_ani}',
        f'skani_AF_ref={sk_af1}',
        f'skani_AF_query={sk_af2}',
        f'fastANI_status={fa_status}',
        f'fastANI_ANI={fa_ani}',
        f'fastANI_AF={fa_af1}',
    ])+'\n')
atomic_text(OUT/PASS,'STATUS=PASS\nSKANI_SELF=PASS\nFASTANI_SELF=PASS\n')
print(f'[PASS] skani self comparison: accession={probe} ANI={sk_ani:.6f}')
print(f'[PASS] FastANI self comparison: accession={probe} ANI={fa_ani:.6f}')
print('E11_CONFIRMATION_TOOLS_PREFLIGHT=PASS')
print(OUT)
