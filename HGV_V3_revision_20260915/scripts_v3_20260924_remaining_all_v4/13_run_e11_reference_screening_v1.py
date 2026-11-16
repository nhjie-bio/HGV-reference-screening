#!/usr/bin/env python3
from __future__ import annotations
import os, math, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'15_reference_screening/E11_reference_screening_v1_{DATE_TAG}'
PASS = 'E11_REFERENCE_SCREENING_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

manifest=load_manifest(); mc=manifest_columns(manifest)
X, xidx, rows, acc_col=feature_accession_index()
meta=manifest.set_index(mc['accession'])

# Resolve A vs external and B2 subtype from taxonomy relationship when manifest labels are incomplete.
cohort={}
for acc,r in meta.iterrows():
    cohort[acc]=infer_cohort(r,mc)
A=[a for a,c in cohort.items() if c=='A']
EXT=[a for a,c in cohort.items() if c!='A']
if len(A)!=2374 or len(EXT)!=780:
    die(f'Unexpected cohort counts A={len(A)} external={len(EXT)}')

A_species=set(meta.loc[A,mc['species']].astype(str))
A_genus=set(meta.loc[A,mc['genus']].astype(str))
# Infer external classes exactly from frozen design if explicit labels missing.
for acc in EXT:
    if cohort[acc]=='EXTERNAL_UNKNOWN':
        sp=str(meta.loc[acc,mc['species']]); ge=str(meta.loc[acc,mc['genus']])
        if sp in A_species:
            cohort[acc]='B1'
        elif ge in A_genus:
            cohort[acc]='B2_seen_genus'
        else:
            cohort[acc]='B2_unseen_genus'
counts=pd.Series([cohort[a] for a in EXT]).value_counts().to_dict()
expected={'B1':391,'B2_seen_genus':196,'B2_unseen_genus':193}
if counts != expected:
    die(f'External subtype inference/count mismatch. expected={expected} got={counts}')

A_idx=np.array([xidx[a] for a in A],dtype=int)
E_idx=np.array([xidx[a] for a in EXT],dtype=int)
XA=np.asarray(X[A_idx],dtype=np.float32)
XE=np.asarray(X[E_idx],dtype=np.float32)

# Frozen canonical A-only 3-seed model embeddings.
seed_ref=[]; seed_query=[]
for seed,ckpt in canonical_a_checkpoints().items():
    model,device,_=load_model_checkpoint(ckpt,2080,64,True)
    seed_ref.append(embed_matrix(model,XA,device))
    seed_query.append(embed_matrix(model,XE,device))
seed_ref=np.stack(seed_ref,axis=0)   # 3 x R x 64
seed_query=np.stack(seed_query,axis=0)
np.save(OUT/'E11_REFERENCE_EMBEDDINGS_3SEED.npy',seed_ref)
np.save(OUT/'E11_QUERY_EMBEDDINGS_3SEED.npy',seed_query)
pd.DataFrame({'reference_accession':A}).to_csv(OUT/'E11_REFERENCE_ORDER.tsv',sep='\t',index=False)
pd.DataFrame({'query_accession':EXT,'external_type':[cohort[a] for a in EXT]}).to_csv(OUT/'E11_QUERY_ORDER.tsv',sep='\t',index=False)

# Exact retrieval. Ensemble distance is mean Euclidean distance across the three frozen models.
top_rows=[]; query_rows=[]
R=len(A)
for qi,qacc in enumerate(EXT):
    dist_seed=[]
    for si in range(3):
        d=np.linalg.norm(seed_ref[si]-seed_query[si,qi][None,:],axis=1)
        dist_seed.append(d)
    dist_seed=np.stack(dist_seed,axis=0)
    dmean=dist_seed.mean(axis=0)
    order=np.argsort(dmean,kind='mergesort')[:50]
    qtype=cohort[qacc]
    if qtype=='B1': taxcol=mc['species']
    elif qtype=='B2_seen_genus': taxcol=mc['genus']
    else: taxcol=mc['family']
    qtax=str(meta.loc[qacc,taxcol])
    relevant=np.array([str(meta.loc[a,taxcol])==qtax for a in A],dtype=bool)
    rel_order=np.flatnonzero(relevant[np.argsort(dmean,kind='mergesort')])
    first_rank=int(rel_order[0]+1) if len(rel_order) else 10**9
    row={'query_accession':qacc,'external_type':qtype,'target_level':taxcol,'target_taxon':qtax,
         'n_reference':R,'first_relevant_rank':first_rank,'MRR':0.0 if first_rank>=10**9 else 1.0/first_rank}
    for k in KS:
        row[f'Recall@{k}']=int(first_rank<=k)
        row[f'candidate_reduction@{k}']=1.0-k/R
        row[f'FN@{k}']=int(first_rank>k)
    query_rows.append(row)
    for rank,ri in enumerate(order,1):
        racc=A[int(ri)]
        top_rows.append({
            'query_accession':qacc,'external_type':qtype,'rank':rank,'reference_accession':racc,
            'ensemble_distance':float(dmean[ri]),
            'seed20260915_distance':float(dist_seed[0,ri]),
            'seed20260916_distance':float(dist_seed[1,ri]),
            'seed20260917_distance':float(dist_seed[2,ri]),
            'query_species':str(meta.loc[qacc,mc['species']]),'query_genus':str(meta.loc[qacc,mc['genus']]),'query_family':str(meta.loc[qacc,mc['family']]),
            'reference_species':str(meta.loc[racc,mc['species']]),'reference_genus':str(meta.loc[racc,mc['genus']]),'reference_family':str(meta.loc[racc,mc['family']]),
            'relevant':int(str(meta.loc[racc,taxcol])==qtax),
        })
qdf=pd.DataFrame(query_rows); tdf=pd.DataFrame(top_rows)
qdf.to_csv(OUT/'E11_QUERY_RETRIEVAL_RESULTS.tsv',sep='\t',index=False)
tdf.to_csv(OUT/'E11_TOP50_CANDIDATES.tsv',sep='\t',index=False)

summary=[]
for qtype,g in qdf.groupby('external_type',sort=True):
    row={'external_type':qtype,'n_queries':len(g),'MRR':g.MRR.mean()}
    for k in KS:
        row[f'Recall@{k}']=g[f'Recall@{k}'].mean()
        row[f'FN_rate@{k}']=g[f'FN@{k}'].mean()
        row[f'candidate_reduction@{k}']=g[f'candidate_reduction@{k}'].mean()
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'E11_RETRIEVAL_SUMMARY.tsv',sep='\t',index=False)

# Confirmatory ANI. Primary confirmation = skani on ensemble Top-50.
# FastANI also confirms ensemble Top-50 by default; lower E11_FASTANI_TOP only for an explicitly documented reduced run.
fmap=get_fasta_map(manifest)
workers=int(os.environ.get('HGV_WORKERS','44'))
skani_top=int(os.environ.get('E11_SKANI_TOP','50'))
fastani_top=int(os.environ.get('E11_FASTANI_TOP','50'))

def skani_job(rec):
    q,r,rank=rec
    ani,af1,af2,status=run_skani_pair(fmap[q],fmap[r])
    return q,r,rank,ani,af1,af2,status

def fastani_job(rec):
    q,r,rank,tmpbase=rec
    tmp=Path(tmpbase)/f'{os.getpid()}_{sha_text(q+r)[:12]}.txt'
    ani,af1,af2,status=run_fastani_pair(fmap[q],fmap[r],tmp)
    return q,r,rank,ani,af1,af2,status

skani_out=OUT/'E11_SKANI_CONFIRM_TOP50.tsv'
existing=set()
skrows=[]
if skani_out.exists():
    old=pd.read_csv(skani_out,sep='\t',dtype=str,keep_default_na=False)
    skrows=old.to_dict('records'); existing={(r['query_accession'],r['reference_accession']) for r in skrows}
tasks=[(r.query_accession,r.reference_accession,int(r.rank)) for r in tdf.itertuples() if int(r.rank)<=skani_top and (r.query_accession,r.reference_accession) not in existing]
if tasks:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(skani_job,x) for x in tasks]
        for i,f in enumerate(as_completed(futs),1):
            q,r,rank,ani,af1,af2,status=f.result()
            skrows.append({'query_accession':q,'reference_accession':r,'rank':rank,'ANI':ani,'AF_ref':af1,'AF_query':af2,'status':status})
            if i%500==0:
                pd.DataFrame(skrows).to_csv(skani_out,sep='\t',index=False)
pd.DataFrame(skrows).to_csv(skani_out,sep='\t',index=False)

fast_out=OUT/'E11_FASTANI_CONFIRM_TOP50.tsv'
farows=[]; existing=set()
if fast_out.exists():
    old=pd.read_csv(fast_out,sep='\t',dtype=str,keep_default_na=False)
    farows=old.to_dict('records'); existing={(r['query_accession'],r['reference_accession']) for r in farows}
fast_tasks=[(r.query_accession,r.reference_accession,int(r.rank)) for r in tdf.itertuples() if int(r.rank)<=fastani_top and (r.query_accession,r.reference_accession) not in existing]
if fast_tasks:
    with tempfile.TemporaryDirectory(prefix='e11_fastani_') as td:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(fastani_job,(*x,td)) for x in fast_tasks]
            for i,f in enumerate(as_completed(futs),1):
                q,r,rank,ani,af1,af2,status=f.result()
                farows.append({'query_accession':q,'reference_accession':r,'rank':rank,'ANI':ani,'AF':af1,'status':status})
                if i%250==0:
                    pd.DataFrame(farows).to_csv(fast_out,sep='\t',index=False)
pd.DataFrame(farows).to_csv(fast_out,sep='\t',index=False)

# Confirmation coverage and best ANI per query.
sk=pd.DataFrame(skrows)
if len(sk):
    sk['ANI']=pd.to_numeric(sk.ANI,errors='coerce')
    conf=sk.groupby('query_accession').agg(skani_pairs=('reference_accession','size'),skani_reported_ani=('ANI','count'),skani_max_ani=('ANI','max')).reset_index()
else:
    conf=pd.DataFrame(columns=['query_accession','skani_pairs','skani_reported_ani','skani_max_ani'])
qdf.merge(conf,on='query_accession',how='left').to_csv(OUT/'E11_QUERY_SCREENING_WITH_CONFIRMATION.tsv',sep='\t',index=False)

sk_fail=(pd.Series([r['status'] for r in skrows])!='OK').mean() if skrows else 1.0
fa_fail=(pd.Series([r['status'] for r in farows])!='OK').mean() if farows else 1.0
if sk_fail>0.10:
    die(f'skani confirmation failure fraction too high: {sk_fail:.3f}')
if fa_fail>0.20:
    die(f'FastANI confirmation failure fraction too high: {fa_fail:.3f}')

report=[
    'HGV-V3 E11 REFERENCE SCREENING v1',
    f'Reference A={len(A)}; external queries={len(EXT)} (B1=391, B2-seen=196, B2-unseen=193).',
    'Ranking = exact Euclidean scan in frozen 64-D canonical A-only embeddings; ensemble distance is mean over 3 frozen model seeds.',
    'K = 1,5,10,20,50.',
    'Relevance: B1 species; B2 seen-genus genus; B2 unseen-genus family.',
    f'Confirmatory skani: Top-{skani_top}; FastANI confirmation: Top-{fastani_top}.',
    'External data are evaluation-only; no model/threshold/tau/lambda selection occurs here.',
]
atomic_text(OUT/'E11_REFERENCE_SCREENING_REPORT.txt','\n'.join(report)+'\n')
atomic_json(OUT/'E11_PROVENANCE.json',{
    'final_manifest_sha256':sha256(FINAL_MANIFEST),'feature_npy_sha256':sha256(find_feature_npy()),
    'canonical_A_checkpoints':{str(k):sha256(v) for k,v in canonical_a_checkpoints().items()},
    'skani_failure_fraction':float(sk_fail),'fastani_failure_fraction':float(fa_fail),
    'workers':workers,'skani_top':skani_top,'fastani_top':fastani_top,
})
atomic_text(OUT/PASS,'STATUS=PASS\nEXTERNAL_RETUNING=NO\nREFERENCE_N=2374\nQUERY_N=780\n')
write_sha256_manifest(OUT)
print('E11_REFERENCE_SCREENING=PASS')
print(OUT)
