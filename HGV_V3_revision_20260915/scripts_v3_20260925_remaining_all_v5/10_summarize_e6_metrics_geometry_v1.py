#!/usr/bin/env python3
import numpy as np
import pandas as pd
from common import *

OUT = E6ROOT / f'07_mechanism_geometry_v1_{DATE_TAG}'
PASS = 'E6_MECHANISM_GEOMETRY_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

pred = pd.read_csv(E6_PRED_LONG, sep='\t', dtype=str, keep_default_na=False)
cols=list(pred.columns)
mode_c=resolve_col(cols,['mode','ablation_mode'])
seed_c=resolve_col(cols,['model_seed','seed'])
a_c=resolve_col(cols,['accession_a'])
b_c=resolve_col(cols,['accession_b'])
y_c=resolve_col(cols,['label'])
pt_c=resolve_col(cols,['pair_type'])
d_c=resolve_col(cols,['distance'])
s_c=resolve_col(cols,['similarity_score','score'])
t_c=resolve_col(cols,['frozen_threshold','threshold'])

pred2=pd.DataFrame({
    'mode':pred[mode_c], 'model_seed':pd.to_numeric(pred[seed_c]).astype(int),
    'accession_a':pred[a_c].map(normalize_accession_id), 'accession_b':pred[b_c].map(normalize_accession_id),
    'label':pd.to_numeric(pred[y_c]).astype(int),'pair_type':pred[pt_c],
    'distance':pd.to_numeric(pred[d_c]).astype(float),
    'similarity_score':pd.to_numeric(pred[s_c]).astype(float),
    'threshold':pd.to_numeric(pred[t_c]).astype(float),
})

# 30-cell integrity: exact formal design, not only total cell count.
cells=pred2[['mode','model_seed']].drop_duplicates().sort_values(['mode','model_seed'])
expected_cells={(f'D{i}', int(seed)) for i in range(6) for seed in MODEL_SEEDS_5}
got_cells={(str(m), int(s)) for m,s in cells[['mode','model_seed']].itertuples(index=False, name=None)}
if got_cells != expected_cells:
    missing=sorted(expected_cells-got_cells)
    extra=sorted(got_cells-expected_cells)
    die(f'E6 30-cell design mismatch: got={len(got_cells)} missing={missing} extra={extra}')

metric_rows=[]
strata_rows=[]
for (mode,seed),g in pred2.groupby(['mode','model_seed'],sort=True):
    th=float(g['threshold'].iloc[0])
    m=binary_metrics(g.label.to_numpy(),g.similarity_score.to_numpy(),th)
    metric_rows.append({'mode':mode,'model_seed':seed,'n_pairs':len(g),'threshold':th,**m})
    for ptype,sg in g.groupby('pair_type',sort=True):
        strata_rows.append({
            'mode':mode,'model_seed':seed,'pair_type':ptype,'n':len(sg),
            'distance_mean':sg.distance.mean(),'distance_sd':sg.distance.std(ddof=1),
            'distance_median':sg.distance.median(),'score_mean':sg.similarity_score.mean(),
            'score_sd':sg.similarity_score.std(ddof=1),'score_median':sg.similarity_score.median(),
        })
metrics=pd.DataFrame(metric_rows)
strata=pd.DataFrame(strata_rows)

# True frozen Dashing2 teacher geometry on the authoritative canonical G2 test manifest.
# Do NOT derive the master pair universe from any model cell. The frozen pair
# manifest is the source of truth for all 30 cells.
# D4 training used shuffled targets, but evaluation here ALWAYS uses the true
# production teacher so D3-vs-D4 is a valid mechanism control.
master_pairs=load_pairs('G2', 20260915, 'test')[
    ['accession_a','accession_b','label','pair_type']
].copy().reset_index(drop=True)
master_pairs['accession_a']=master_pairs['accession_a'].map(normalize_accession_id)
master_pairs['accession_b']=master_pairs['accession_b'].map(normalize_accession_id)
if len(master_pairs) != 3872:
    die(f'Canonical G2 test manifest must contain 3872 pairs, got {len(master_pairs)}')
teacher_dist=load_dashing_teacher_for_pairs(master_pairs[['accession_a','accession_b']])
if len(teacher_dist)!=len(master_pairs) or not np.isfinite(teacher_dist).all():
    die('Dashing2 true-teacher join failed for one or more canonical G2 test pairs')
master_pairs['teacher_distance']=teacher_dist
master_pairs['teacher_similarity']=1.0-teacher_dist
master_pairs['informative_negative']=(master_pairs.label.astype(int)==0)&(master_pairs.teacher_distance<0.98)
master_pairs.to_csv(OUT/'E6_TRUE_TEACHER_TEST_GEOMETRY.tsv',sep='\t',index=False)

def _pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return np.nan
    return float(np.corrcoef(x,y)[0,1])

def _spearman(x,y):
    rx=pd.Series(np.asarray(x,float)).rank(method='average').to_numpy(float)
    ry=pd.Series(np.asarray(y,float)).rank(method='average').to_numpy(float)
    return _pearson(rx,ry)

teacher_rows=[]
mm=master_pairs.copy()
mm['key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(mm.accession_a,mm.accession_b)]
if mm['key'].duplicated().any():
    die(f'Duplicate normalized pair keys in D3 master: {mm.loc[mm.key.duplicated(keep=False), "key"].head(10).tolist()}')
master_key_set=set(mm['key'])
for (mode,seed),g in pred2.groupby(['mode','model_seed'],sort=True):
    gg=g.copy()
    gg['key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(gg.accession_a,gg.accession_b)]
    if gg['key'].duplicated().any():
        die(f'Duplicate normalized pair keys for {mode} seed={seed}: {gg.loc[gg.key.duplicated(keep=False), "key"].head(10).tolist()}')
    cell_key_set=set(gg['key'])
    if cell_key_set != master_key_set:
        missing=sorted(master_key_set-cell_key_set)[:10]
        extra=sorted(cell_key_set-master_key_set)[:10]
        die(f'Normalized pair-set mismatch for {mode} seed={seed}: missing={len(master_key_set-cell_key_set)} extra={len(cell_key_set-master_key_set)} missing_examples={missing} extra_examples={extra}')
    gg=gg.merge(mm[['key','teacher_distance','informative_negative']],on='key',how='left',validate='one_to_one')
    if gg.teacher_distance.isna().any(): die(f'Teacher join missing after normalized pair audit for {mode} seed={seed}')
    masks={
        'all':np.ones(len(gg),dtype=bool),
        'positive':gg.label.to_numpy(int)==1,
        'negative':gg.label.to_numpy(int)==0,
        'informative_negative':gg.informative_negative.to_numpy(bool),
    }
    for ptype in sorted(gg.pair_type.unique()):
        masks[f'pair_type:{ptype}']=gg.pair_type.to_numpy(str)==ptype
    for stratum,mask in masks.items():
        if int(mask.sum())<3: continue
        sd=gg.loc[mask,'distance'].to_numpy(float)
        # For normalized modes, training distillation targets student d/2.
        # D5 intentionally has no unit-sphere constraint, but the historical E6
        # protocol still evaluates similarity as 1-d/2; use the same scale here.
        student_dist=sd/2.0
        td=gg.loc[mask,'teacher_distance'].to_numpy(float)
        teacher_rows.append({
            'mode':mode,'model_seed':seed,'stratum':stratum,'n':int(mask.sum()),
            'teacher_distance_mean':float(np.mean(td)),
            'student_distance_norm_mean':float(np.mean(student_dist)),
            'pearson_student_teacher_distance':_pearson(student_dist,td),
            'spearman_student_teacher_distance':_spearman(student_dist,td),
            'mse_student_teacher_distance':float(np.mean((student_dist-td)**2)),
            'mae_student_teacher_distance':float(np.mean(np.abs(student_dist-td))),
        })
teacher_df=pd.DataFrame(teacher_rows)
teacher_df.to_csv(OUT/'E6_TRUE_TEACHER_FIDELITY_BY_MODE_SEED.tsv',sep='\t',index=False)
teacher_summary=teacher_df.groupby(['mode','stratum'],as_index=False).agg(
    n_seeds=('model_seed','nunique'),
    spearman_mean=('spearman_student_teacher_distance','mean'),
    spearman_sd=('spearman_student_teacher_distance','std'),
    pearson_mean=('pearson_student_teacher_distance','mean'),
    pearson_sd=('pearson_student_teacher_distance','std'),
    mse_mean=('mse_student_teacher_distance','mean'),
    mse_sd=('mse_student_teacher_distance','std'),
)
teacher_summary.to_csv(OUT/'E6_TRUE_TEACHER_FIDELITY_SUMMARY.tsv',sep='\t',index=False)

metrics.to_csv(OUT/'E6_RECOMPUTED_RUN_METRICS.tsv',sep='\t',index=False)
strata.to_csv(OUT/'E6_PAIRTYPE_GEOMETRY.tsv',sep='\t',index=False)

summary=[]
for mode,g in metrics.groupby('mode',sort=True):
    row={'mode':mode,'n_seeds':len(g)}
    for col in ['AUROC','AUPRC','MCC','F1','BalancedAccuracy','threshold']:
        row[f'{col}_mean']=g[col].mean(); row[f'{col}_sd']=g[col].std(ddof=1)
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'E6_MODE_METRIC_SUMMARY.tsv',sep='\t',index=False)

# Pairwise cell-aligned score/distance comparison to D3.
paired=[]
control=[]
for seed in MODEL_SEEDS_5:
    base=pred2[(pred2.mode=='D3')&(pred2.model_seed==seed)].copy()
    base['key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(base.accession_a,base.accession_b)]
    base=base.set_index('key')
    for mode in ['D0','D1','D2','D4','D5']:
        g=pred2[(pred2.mode==mode)&(pred2.model_seed==seed)].copy()
        g['key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(g.accession_a,g.accession_b)]
        g=g.set_index('key')
        if set(g.index)!=set(base.index):
            die(f'Pair set mismatch D3 vs {mode} seed={seed}')
        g=g.loc[base.index]
        delta_s=base.similarity_score.to_numpy()-g.similarity_score.to_numpy()
        delta_d=base.distance.to_numpy()-g.distance.to_numpy()
        paired.append({
            'comparison':f'D3-{mode}','model_seed':seed,'n_pairs':len(base),
            'delta_score_mean':float(np.mean(delta_s)),'delta_score_sd':float(np.std(delta_s,ddof=1)),
            'delta_score_abs_mean':float(np.mean(np.abs(delta_s))),
            'delta_distance_mean':float(np.mean(delta_d)),'delta_distance_abs_mean':float(np.mean(np.abs(delta_d))),
            'score_pearson':float(np.corrcoef(base.similarity_score.to_numpy(),g.similarity_score.to_numpy())[0,1]),
        })
        if mode=='D4':
            for ptype,ix in base.groupby('pair_type').groups.items():
                ix=list(ix)
                bsub=base.loc[ix].similarity_score.to_numpy(); gsub=g.loc[ix].similarity_score.to_numpy()
                control.append({'model_seed':seed,'pair_type':ptype,'n':len(ix),
                                'D3_D4_score_pearson':float(np.corrcoef(bsub,gsub)[0,1]) if len(ix)>1 else np.nan,
                                'D3_minus_D4_score_mean':float(np.mean(bsub-gsub)),
                                'D3_minus_D4_score_abs_mean':float(np.mean(np.abs(bsub-gsub)))})
pd.DataFrame(paired).to_csv(OUT/'E6_D3_PAIRED_GEOMETRY_CONTRASTS.tsv',sep='\t',index=False)
pd.DataFrame(control).to_csv(OUT/'E6_D3_VS_D4_SHUFFLED_CONTROL.tsv',sep='\t',index=False)

# Bring forward norm audit as immutable evidence reference without modifying source.
norm_src=E6_TEST_ROOT/'E6_G2_EMBEDDING_NORM_AUDIT.tsv'
if norm_src.exists():
    ndf=pd.read_csv(norm_src,sep='\t',dtype=str,keep_default_na=False)
    ndf.to_csv(OUT/'E6_EMBEDDING_NORM_AUDIT_COPY.tsv',sep='\t',index=False)

report=[]
report.append('HGV-V3 E6 MECHANISM / GEOMETRY SUMMARY v1')
report.append(f'Input predictions: {E6_PRED_LONG}')
report.append(f'Input SHA256: {sha256(E6_PRED_LONG)}')
report.append('Cells: 30 = 6 modes x 5 model seeds')
report.append('This stage performs no training, threshold selection, or test-driven tuning.')
report.append('D4 is evaluated as the shuffled-teacher control against the TRUE frozen Dashing2 teacher; D5 retains its no-L2 semantics from frozen E6.')
atomic_text(OUT/'E6_MECHANISM_GEOMETRY_REPORT.txt','\n'.join(report)+'\n')
atomic_text(OUT/PASS,'STATUS=PASS\nTRAINING_PERFORMED=NO\nTHRESHOLD_RESELECTED=NO\nTEST_TUNING=NO\n')
write_sha256_manifest(OUT)
print('E6_MECHANISM_GEOMETRY=PASS')
print(OUT)
