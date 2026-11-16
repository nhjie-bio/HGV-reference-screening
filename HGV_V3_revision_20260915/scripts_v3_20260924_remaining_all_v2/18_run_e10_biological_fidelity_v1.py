#!/usr/bin/env python3
from __future__ import annotations
import os, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'14_biological_validation/E10_biological_fidelity_v1_{DATE_TAG}'
PASS = 'E10_BIOLOGICAL_FIDELITY_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
test=load_pairs('G2',20260915,'test')

# Mean frozen D3 test geometry across five E6 seeds.
e6=pd.read_csv(E6_PRED_LONG,sep='\t',dtype=str,keep_default_na=False)
mode_c=resolve_col(e6.columns,['mode','ablation_mode']); seed_c=resolve_col(e6.columns,['model_seed','seed'])
a_c=resolve_col(e6.columns,['accession_a']); b_c=resolve_col(e6.columns,['accession_b']); d_c=resolve_col(e6.columns,['distance']); s_c=resolve_col(e6.columns,['similarity_score','score'])
g=e6[e6[mode_c]=='D3'].copy()
g['key']=[f'{min(a,b)}|{max(a,b)}' for a,b in zip(g[a_c],g[b_c])]
g[d_c]=pd.to_numeric(g[d_c]); g[s_c]=pd.to_numeric(g[s_c])
agg=g.groupby('key').agg(hgv_distance_mean=(d_c,'mean'),hgv_distance_sd=(d_c,'std'),hgv_similarity_mean=(s_c,'mean')).reset_index()
base=test.copy(); base['key']=[f'{min(a,b)}|{max(a,b)}' for a,b in zip(base.accession_a,base.accession_b)]
base=base.merge(agg,on='key',how='left')
if base.hgv_distance_mean.isna().any(): die('Failed to join E6 D3 mean geometry to all G2 test pairs')

# Resume-safe skani pairwise confirmation on canonical G2 test pairs.
ANI_FILE=OUT/'E10_SKANI_G2_TEST.tsv'
rows=[]; done=set()
if ANI_FILE.exists():
    old=pd.read_csv(ANI_FILE,sep='\t',dtype=str,keep_default_na=False); rows=old.to_dict('records'); done={(r['accession_a'],r['accession_b']) for r in rows}
workers=int(os.environ.get('HGV_WORKERS','44'))

def job(rec):
    a,b=rec
    ani,af1,af2,status=run_skani_pair(fmap[a],fmap[b])
    return a,b,ani,af1,af2,status

tasks=[(a,b) for a,b in zip(base.accession_a,base.accession_b) if (a,b) not in done]
if tasks:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(job,t) for t in tasks]
        for i,f in enumerate(as_completed(futs),1):
            a,b,ani,af1,af2,status=f.result()
            rows.append({'accession_a':a,'accession_b':b,'ANI':ani,'AF_ref':af1,'AF_query':af2,'status':status})
            if i%250==0: pd.DataFrame(rows).to_csv(ANI_FILE,sep='\t',index=False)
pd.DataFrame(rows).to_csv(ANI_FILE,sep='\t',index=False)
ani=pd.DataFrame(rows)
ani['ANI']=pd.to_numeric(ani.ANI,errors='coerce'); ani['AF_ref']=pd.to_numeric(ani.AF_ref,errors='coerce'); ani['AF_query']=pd.to_numeric(ani.AF_query,errors='coerce')
ani['AF_min']=ani[['AF_ref','AF_query']].min(axis=1)
full=base.merge(ani[['accession_a','accession_b','ANI','AF_ref','AF_query','AF_min','status']],on=['accession_a','accession_b'],how='left')
full['ani_distance']=1-full.ANI/100.0
full.to_csv(OUT/'E10_HGV_SKANI_PAIR_SOURCE.tsv',sep='\t',index=False)

# Correlations overall, positive-only, negative-only, and pair-type strata.
rows_corr=[]
def add_corr(name,df):
    for target in ['ANI','AF_min','ani_distance']:
        sub=df[['hgv_distance_mean',target]].dropna()
        if len(sub)<3: continue
        sr,sp=spearman(sub.hgv_distance_mean,sub[target]); pr,pp=pearson(sub.hgv_distance_mean,sub[target])
        rows_corr.append({'subset':name,'target':target,'n':len(sub),'spearman_r':sr,'spearman_p':sp,'pearson_r':pr,'pearson_p':pp})
add_corr('all_reported',full)
add_corr('positive',full[full.label==1])
add_corr('negative',full[full.label==0])
for pt,sg in full.groupby('pair_type',sort=True): add_corr(f'pair_type:{pt}',sg)
pd.DataFrame(rows_corr).to_csv(OUT/'E10_CORRELATION_SUMMARY.tsv',sep='\t',index=False)

# ANI-range fidelity to directly address weak/global-correlation concern.
bins=[-np.inf,90,95,97,99,np.inf]; labels=['<90','90-95','95-97','97-99','>=99']
full['ANI_range']=pd.cut(full.ANI,bins=bins,labels=labels,right=False)
range_rows=[]
for rng,sg in full.groupby('ANI_range',observed=True):
    sub=sg[['hgv_distance_mean','ANI','AF_min']].dropna(subset=['hgv_distance_mean','ANI'])
    if not len(sub): continue
    sr,sp=spearman(sub.hgv_distance_mean,sub.ANI)
    range_rows.append({'ANI_range':str(rng),'n':len(sub),'hgv_distance_mean':sub.hgv_distance_mean.mean(),'hgv_distance_sd':sub.hgv_distance_mean.std(ddof=1),
                       'ANI_mean':sub.ANI.mean(),'AF_min_mean':sub.AF_min.mean(),'spearman_hgv_vs_ANI':sr,'p_value':sp})
pd.DataFrame(range_rows).to_csv(OUT/'E10_ANI_RANGE_FIDELITY.tsv',sep='\t',index=False)

# Taxonomic-distance consistency from frozen metadata.
meta=manifest.set_index(mc['accession'])
def tax_stratum(a,b):
    if str(meta.loc[a,mc['species']])==str(meta.loc[b,mc['species']]): return 'same_species'
    if str(meta.loc[a,mc['genus']])==str(meta.loc[b,mc['genus']]): return 'same_genus_diff_species'
    if str(meta.loc[a,mc['family']])==str(meta.loc[b,mc['family']]): return 'same_family_diff_genus'
    return 'different_family'
full['taxonomic_stratum']=[tax_stratum(a,b) for a,b in zip(full.accession_a,full.accession_b)]
taxsum=full.groupby('taxonomic_stratum').agg(n=('key','size'),hgv_distance_mean=('hgv_distance_mean','mean'),hgv_distance_sd=('hgv_distance_mean','std'),ANI_mean=('ANI','mean'),AF_min_mean=('AF_min','mean')).reset_index()
taxsum.to_csv(OUT/'E10_TAXONOMIC_DISTANCE_CONSISTENCY.tsv',sep='\t',index=False)

# Formal tree component is conditional by design. The cleaned V3 project normally removed the unused bac120_r232.tree.
trees=list((BASE/'00_metadata').glob('*bac120*.tree'))
if trees:
    tree_status='AVAILABLE_BUT_NOT_AUTO_USED: tree leaf mapping requires an explicit frozen accession-to-leaf audit before patristic statistics.'
else:
    tree_status='SKIPPED_NO_FROZEN_TREE: no formal GTDB tree retained in the cleaned V3 source-of-truth; no phylogeny reconstruction claim is made.'
atomic_text(OUT/'E10_TREE_COMPONENT_STATUS.txt',tree_status+'\n')

pos=full[full.label==1]
pos_cov=float(pos.ANI.notna().mean()); overall_cov=float(full.ANI.notna().mean())
if pos_cov < 0.90:
    die(f'skani positive-pair ANI coverage too low: {pos_cov:.3f}')

atomic_text(OUT/'E10_BIOLOGICAL_FIDELITY_REPORT.txt','\n'.join([
    'HGV-V3 E10 BIOLOGICAL FIDELITY v1',
    f'Canonical G2 test pairs={len(full)}; D3 geometry averaged across five frozen E6 model seeds.',
    f'skani reported ANI coverage overall={overall_cov:.4f}; positive={pos_cov:.4f}.',
    'Analyses: HGV distance vs ANI/AF; positive/negative and difficulty strata; ANI-range fidelity; taxonomy-distance consistency.',
    tree_status,
    'Interpretation boundary: these analyses can support taxonomically/genomically consistent geometry, not a claim that HGV reconstructs phylogeny or replaces ANI.',
])+'\n')
atomic_json(OUT/'E10_PROVENANCE.json',{'E6_predictions_sha256':sha256(E6_PRED_LONG),'test_pairs_sha256':sha256(find_pair_file('G2',20260915,'test')),
                                       'skani_version':tool_version('skani'),'workers':workers,'overall_ANI_coverage':overall_cov,'positive_ANI_coverage':pos_cov})
atomic_text(OUT/PASS,'STATUS=PASS\nANI_AF_FIDELITY=YES\nTREE_CLAIM=NO\n')
write_sha256_manifest(OUT)
print('E10_BIOLOGICAL_FIDELITY=PASS')
print(OUT)
