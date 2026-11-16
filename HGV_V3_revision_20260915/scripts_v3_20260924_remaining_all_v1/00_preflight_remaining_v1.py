#!/usr/bin/env python3
from pathlib import Path
import json, os, shutil
import numpy as np
import pandas as pd

from common import *

OUT = BASE / f'logs/remaining_all_preflight_v1_{DATE_TAG}'
PASS = 'REMAINING_ALL_PREFLIGHT_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

checks=[]
def ck(name, ok, detail=''):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise RuntimeError(f'{name}: {detail}')

try:
    ck('BASE exists', BASE.exists(), BASE)
    m=load_manifest(); mc=manifest_columns(m)
    ck('final manifest SHA', sha256(FINAL_MANIFEST)==FINAL_MANIFEST_SHA256, sha256(FINAL_MANIFEST))
    cohort_counts={}
    for _,r in m.iterrows():
        c=infer_cohort(r,mc); cohort_counts[c]=cohort_counts.get(c,0)+1
    ck('development A count', cohort_counts.get('A',0)==2374, cohort_counts)
    ck('external total count', len(m)-cohort_counts.get('A',0)==780, cohort_counts)

    X, idx, rows, ac=feature_accession_index()
    ck('feature shape', tuple(X.shape)==(3154,2080), X.shape)
    ck('feature accession coverage', len(idx)==3154, len(idx))

    p=load_pairs('G2',20260915,'test')
    ck('canonical G2 test pair count', len(p)==3872, len(p))

    ck('E6 test freeze exists', E6_TEST_FREEZE.exists(), E6_TEST_FREEZE)
    ck('E6 predictions exist', E6_PRED_LONG.exists(), E6_PRED_LONG)
    ck('E6 thresholds exist', E6_THRESHOLDS.exists(), E6_THRESHOLDS)
    cells=list((E6_TEST_ROOT/'cell_predictions').glob('D*_seed*.tsv')) if (E6_TEST_ROOT/'cell_predictions').exists() else []
    ck('E6 30 cell predictions', len(cells)==30, len(cells))

    ck('Dashing2 production teacher exists', TEACHER_DASHING.exists(), TEACHER_DASHING)
    for seed in MODEL_SEEDS_3:
        ck(f'main D3 checkpoint seed{seed}', main_d3_checkpoint(seed).exists(), main_d3_checkpoint(seed))
    ca=canonical_a_checkpoints()
    ck('canonical A 3 checkpoints', len(ca)==3 and all(p.exists() for p in ca.values()), ca)

    ck('skani available', shutil.which('skani') is not None, shutil.which('skani'))
    ck('fastANI available', (shutil.which('fastANI') or shutil.which('fastani')) is not None, shutil.which('fastANI') or shutil.which('fastani'))
    ck('g++ available', shutil.which('g++') is not None, shutil.which('g++'))

    # FASTA existence audit. Do not rehash/reselect; just path presence.
    fmap=get_fasta_map(m)
    missing=[str(p) for p in fmap.values() if not p.exists()]
    ck('3154 FASTA paths present', not missing, f'missing={len(missing)}')

    atomic_json(OUT/'preflight.json', {'checks':checks,'cohort_counts':cohort_counts,'environment':environment_snapshot()})
    atomic_text(OUT/PASS, 'STATUS=PASS\nNO_FROZEN_INPUT_MODIFIED=YES\n')
    write_sha256_manifest(OUT)
except Exception as e:
    atomic_json(OUT/'preflight.json', {'checks':checks,'error':repr(e),'environment':environment_snapshot()})
    raise
