#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from common import *

OUT = BASE / f'logs/E7_input_preflight_{DATE_TAG}'
PASS = OUT / 'E7_INPUT_PREFLIGHT_PASS.txt'
if PASS.exists():
    print(f'[SKIP] Existing PASS: {PASS}')
    raise SystemExit(0)
OUT.mkdir(parents=True, exist_ok=True)

checks=[]
def ck(name, ok, detail=''):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise RuntimeError(f'{name}: {detail}')

try:
    expected_paths={
        'train': PAIR_ROOT/'G2_seed20260915'/'train_unique_pair_pool.tsv',
        'val': PAIR_ROOT/'G2_seed20260915'/'val_pairs.tsv',
        'test': PAIR_ROOT/'G2_seed20260915'/'test_pairs.tsv',
    }
    expected_n={'train':18176,'val':3874,'test':3872}
    pairs={}
    for role in ['train','val','test']:
        resolved=find_pair_file('G2',20260915,role)
        ck(f'G2 seed15 {role} pair path', resolved == expected_paths[role], resolved)
        df=load_pairs('G2',20260915,role)
        pairs[role]=df
        ck(f'G2 seed15 {role} pair count', len(df)==expected_n[role], len(df))
        labs=sorted(set(df.label.astype(int).tolist()))
        ck(f'G2 seed15 {role} has both labels', labs==[0,1], labs)

    # All E7 pair endpoints must belong to the development-A feature universe.
    m=load_manifest(); mc=manifest_columns(m)
    mi=m.set_index(mc['accession'])
    A={a for a,r in mi.iterrows() if infer_cohort(r,mc,accession=a)=='A'}
    ck('development A accession count', len(A)==2374, len(A))
    for role,df in pairs.items():
        endpoints=set(df.accession_a)|set(df.accession_b)
        missing=sorted(endpoints-A)
        ck(f'{role} endpoints within A', not missing, f'endpoints={len(endpoints)} missing={missing[:5]}')

    # E7 needs the production teacher for train+validation; test teacher is not used for tuning.
    for role in ['train','val']:
        td=load_dashing_teacher_for_pairs(pairs[role][['accession_a','accession_b']], TEACHER_DASHING)
        ck(f'Dashing2 {role} coverage', len(td)==len(pairs[role]) and np.isfinite(td).all(), f'{len(td)}/{len(pairs[role])}')
        ck(f'Dashing2 {role} distance range', bool((td>=-1e-7).all() and (td<=1.0+1e-7).all()), f'min={float(td.min()):.8g} max={float(td.max()):.8g}')

    # Downstream E8 depends on the frozen main threshold file; gate it before long E7 training starts.
    ck('main frozen threshold file exists', MAIN_THRESHOLD_FILE.exists(), MAIN_THRESHOLD_FILE)
    th={seed:find_threshold(MAIN_THRESHOLD_FILE,seed,'G2',20260915) for seed in MODEL_SEEDS_3}
    ck('main G2 seed15 thresholds resolvable', len(th)==3 and all(np.isfinite(list(th.values()))), th)

    test_acc=set(pairs['test'].accession_a)|set(pairs['test'].accession_b)
    ck('G2 seed15 test genome count for E8/E9', len(test_acc)==356, len(test_acc))

    atomic_json(OUT/'E7_INPUT_PREFLIGHT.json', {'checks':checks,'resolved_pair_paths':{k:str(find_pair_file('G2',20260915,k)) for k in ['train','val','test']}})
    atomic_text(PASS, 'STATUS=PASS\nPAIR_LAYOUT=FROZEN_V3\nTRAIN_UNIQUE_POOL=YES\nNO_FROZEN_INPUT_MODIFIED=YES\n')
    write_sha256_manifest(OUT)
    print('E7_INPUT_PREFLIGHT=PASS')
    print(OUT)
except Exception as e:
    atomic_json(OUT/'E7_INPUT_PREFLIGHT.json', {'checks':checks,'error':repr(e)})
    raise
