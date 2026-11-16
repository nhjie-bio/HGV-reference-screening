#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *

OUT = BASE / f'17_statistics/E6_species_block_bootstrap_v1_{DATE_TAG}'
PASS = 'E6_SPECIES_BLOCK_STATS_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

B = int(os.environ.get('E6_BOOTSTRAP_N','1000'))
RNG_SEED = int(os.environ.get('E6_BOOTSTRAP_SEED','20260924'))

pred = pd.read_csv(E6_PRED_LONG, sep='\t', dtype=str, keep_default_na=False)
cols=list(pred.columns)
mode_c=resolve_col(cols,['mode','ablation_mode'])
seed_c=resolve_col(cols,['model_seed','seed'])
a_c=resolve_col(cols,['accession_a'])
b_c=resolve_col(cols,['accession_b'])
y_c=resolve_col(cols,['label'])
s_c=resolve_col(cols,['similarity_score','score'])
t_c=resolve_col(cols,['frozen_threshold','threshold'])
pt_c=resolve_col(cols,['pair_type'], required=False)

df=pd.DataFrame({
    'mode':pred[mode_c], 'model_seed':pd.to_numeric(pred[seed_c]).astype(int),
    'accession_a':pred[a_c].map(normalize_accession_id), 'accession_b':pred[b_c].map(normalize_accession_id),
    'label':pd.to_numeric(pred[y_c]).astype(int), 'score':pd.to_numeric(pred[s_c]).astype(float),
    'threshold':pd.to_numeric(pred[t_c]).astype(float),
    'pair_type':pred[pt_c] if pt_c else '',
})

m=load_manifest(); mc=manifest_columns(m)
tax=m.set_index(mc['accession'])
species_map=tax[mc['species']].astype(str).to_dict()
df['species_a']=df.accession_a.map(species_map)
df['species_b']=df.accession_b.map(species_map)
if df[['species_a','species_b']].isna().any().any():
    die('Missing species annotation for one or more E6 test-pair accessions')

# Exact formal 30-cell design gate.
expected_cells={(f'D{i}', int(seed)) for i in range(6) for seed in MODEL_SEEDS_5}
got_cells={(str(m), int(s)) for m,s in df[['mode','model_seed']].drop_duplicates().itertuples(index=False, name=None)}
if got_cells != expected_cells:
    missing=sorted(expected_cells-got_cells)
    extra=sorted(got_cells-expected_cells)
    die(f'E6 30-cell design mismatch before statistics: got={len(got_cells)} missing={missing} extra={extra}')

# Canonical pair order and integrity across cells, anchored to the frozen G2
# test manifest rather than to any particular D3 model cell.
df['pair_key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(df.accession_a,df.accession_b)]
master=load_pairs('G2',20260915,'test').copy()
master['accession_a']=master.accession_a.map(normalize_accession_id)
master['accession_b']=master.accession_b.map(normalize_accession_id)
master['pair_key']=['|'.join(canonical_pair_key(a,b)) for a,b in zip(master.accession_a,master.accession_b)]
if len(master)!=3872 or master.pair_key.duplicated().any():
    die(f'Canonical G2 test manifest integrity failure: rows={len(master)} duplicate_keys={int(master.pair_key.duplicated().sum())}')
master_keys=tuple(sorted(master.pair_key))
for (mode,seed),g in df.groupby(['mode','model_seed']):
    keys=tuple(sorted(g.pair_key))
    if keys != master_keys:
        gs=set(keys); ms=set(master_keys)
        die(f'Pair set mismatch at {mode} seed={seed}: missing={len(ms-gs)} extra={len(gs-ms)} missing_examples={sorted(ms-gs)[:10]} extra_examples={sorted(gs-ms)[:10]}')

master=master.sort_values('pair_key').reset_index(drop=True)
# Species labels are taken from the frozen final manifest.
master['species_a']=master.accession_a.map(species_map)
master['species_b']=master.accession_b.map(species_map)
if master[['species_a','species_b']].isna().any().any():
    die('Missing species annotation for canonical G2 test manifest')
y=master.label.to_numpy(int)
sa=master.species_a.to_numpy(str); sb=master.species_b.to_numpy(str)
all_species=np.array(sorted(set(sa)|set(sb)),dtype=object)
score_by={}
th_by={}
for mode in [f'D{i}' for i in range(6)]:
    for seed in MODEL_SEEDS_5:
        g=df[(df['mode']==mode)&(df['model_seed']==seed)].sort_values('pair_key').reset_index(drop=True)
        if len(g)!=len(master) or not np.array_equal(g['pair_key'].to_numpy(),master['pair_key'].to_numpy()):
            die(f'Alignment failure {mode} seed={seed}: rows={len(g)} expected={len(master)}')
        if not np.array_equal(g['label'].to_numpy(int), y):
            bad=np.where(g['label'].to_numpy(int) != y)[0][:10].tolist()
            die(f'Label alignment failure {mode} seed={seed}: first_bad_indices={bad}')
        if g['threshold'].nunique(dropna=False) != 1:
            die(f'Frozen threshold is not constant within cell {mode} seed={seed}')
        score_by[(mode,seed)]=g['score'].to_numpy(float)
        th_by[(mode,seed)]=float(g['threshold'].iloc[0])

# Final aligned-array gate before any inferential calculation.
if len(score_by) != 30 or len(th_by) != 30:
    die(f'Aligned E6 cell cache incomplete: score_cells={len(score_by)} threshold_cells={len(th_by)}')
for key, arr in score_by.items():
    if len(arr) != 3872 or not np.isfinite(arr).all():
        die(f'Invalid aligned score array for {key}: n={len(arr)} finite={bool(np.isfinite(arr).all())}')

# Observed seed-level metrics.
seed_rows=[]
for mode in [f'D{i}' for i in range(6)]:
    for seed in MODEL_SEEDS_5:
        met=binary_metrics(y,score_by[(mode,seed)],th_by[(mode,seed)])
        seed_rows.append({'mode':mode,'model_seed':seed,**met})
seed_df=pd.DataFrame(seed_rows)
seed_df.to_csv(OUT/'E6_SEED_LEVEL_TEST_METRICS.tsv',sep='\t',index=False)

# Two-way species cluster bootstrap. Sample species clusters with replacement;
# a same-species pair gets cluster multiplicity c_i, a cross-species pair gets c_i*c_j.
# This preserves paired comparison because every mode/seed uses the exact same weights per replicate.
rng=np.random.default_rng(RNG_SEED)
comparators=['D0','D1','D2','D4','D5']
metrics=['AUROC','AUPRC']
boot_rows=[]
valid=0
attempt=0
max_attempt=B*10
while valid < B and attempt < max_attempt:
    attempt += 1
    draw=rng.choice(all_species,size=len(all_species),replace=True)
    uniq,cnt=np.unique(draw,return_counts=True)
    c=dict(zip(uniq,cnt.astype(float)))
    w=np.array([c.get(a,0.0) if a==b else c.get(a,0.0)*c.get(b,0.0) for a,b in zip(sa,sb)],dtype=float)
    mask=w>0
    if not mask.any() or len(np.unique(y[mask]))<2:
        continue
    # Need both weighted classes.
    if w[(y==1)].sum()==0 or w[(y==0)].sum()==0:
        continue
    rep_values={}
    ok=True
    for mode in ['D3']+comparators:
        for seed in MODEL_SEEDS_5:
            sc=score_by[(mode,seed)]
            try:
                auroc=float(roc_auc_score(y,sc,sample_weight=w))
                auprc=float(average_precision_score(y,sc,sample_weight=w))
            except Exception:
                ok=False; break
            rep_values[(mode,seed,'AUROC')]=auroc
            rep_values[(mode,seed,'AUPRC')]=auprc
        if not ok: break
    if not ok: continue
    valid += 1
    for comp in comparators:
        for metric in metrics:
            ds=[rep_values[('D3',s,metric)]-rep_values[(comp,s,metric)] for s in MODEL_SEEDS_5]
            boot_rows.append({'replicate':valid,'comparison':f'D3-{comp}','metric':metric,'delta_mean_5seed':float(np.mean(ds))})
if valid != B:
    die(f'Only {valid}/{B} valid bootstrap replicates after {attempt} attempts')
boot=pd.DataFrame(boot_rows)
boot.to_csv(OUT/'E6_SPECIES_BLOCK_BOOTSTRAP_LONG.tsv',sep='\t',index=False)

# Observed contrasts + percentile CI + bootstrap two-sided p.
obs=[]
for comp in comparators:
    for metric in metrics:
        d3=seed_df[seed_df['mode']=='D3'].sort_values('model_seed')[metric].to_numpy(float)
        xx=seed_df[seed_df['mode']==comp].sort_values('model_seed')[metric].to_numpy(float)
        observed=float(np.mean(d3-xx))
        vals=boot[(boot.comparison==f'D3-{comp}')&(boot.metric==metric)].delta_mean_5seed.to_numpy(float)
        lo,hi=np.quantile(vals,[0.025,0.975])
        # add-one bootstrap sign p-value
        nle=int((vals<=0).sum()); nge=int((vals>=0).sum())
        p=min(1.0, 2.0*min((nle+1)/(len(vals)+1),(nge+1)/(len(vals)+1)))
        obs.append({'comparison':f'D3-{comp}','metric':metric,'observed_delta_mean_5seed':observed,
                    'bootstrap_mean':float(np.mean(vals)),'ci95_low':float(lo),'ci95_high':float(hi),'p_raw_two_sided':p,
                    'bootstrap_n':len(vals),'bootstrap_unit':'two-way species-cluster multiplicity'})
res=pd.DataFrame(obs)
res['p_holm']=holm_adjust(res.p_raw_two_sided.to_numpy(float))
res['direction_supported']=(res.observed_delta_mean_5seed>0)&(res.ci95_low>0)
res['holm_significant']=res.p_holm<0.05
res.to_csv(OUT/'E6_D3_PAIRED_CONTRASTS_HOLM.tsv',sep='\t',index=False)

# Secondary metrics paired seed differences without inferential overclaim.
sec=[]
for comp in comparators:
    for metric in ['MCC','F1','BalancedAccuracy']:
        d3=seed_df[seed_df['mode']=='D3'].sort_values('model_seed')[metric].to_numpy(float)
        xx=seed_df[seed_df['mode']==comp].sort_values('model_seed')[metric].to_numpy(float)
        delta=d3-xx
        sec.append({'comparison':f'D3-{comp}','metric':metric,'delta_mean':delta.mean(),'delta_sd':delta.std(ddof=1),
                    'n_model_seeds':len(delta),'note':'descriptive paired seed variability; primary inference is AUROC/AUPRC species-block bootstrap'})
pd.DataFrame(sec).to_csv(OUT/'E6_SECONDARY_PAIRED_SEED_DELTAS.tsv',sep='\t',index=False)

report=[
    'HGV-V3 E6 FORMAL STATISTICS v1',
    f'Bootstrap replicates={B}',
    f'Bootstrap RNG seed={RNG_SEED}',
    'Primary comparisons: D3 vs D0/D1/D2/D4/D5.',
    'Primary metrics: paired delta AUROC and paired delta AUPRC.',
    'Bootstrap: species-cluster resampling with common replicate weights across all methods and seeds.',
    'Multiplicity correction: Holm across all 10 primary hypotheses.',
    'Five model seeds quantify initialization variability and are not treated as biological replicates.',
    'No threshold/model/hyperparameter was changed after test evaluation.',
]
atomic_text(OUT/'E6_STATISTICS_REPORT.txt','\n'.join(report)+'\n')
atomic_text(OUT/PASS,'STATUS=PASS\nBOOTSTRAP_N=%d\nHOLM=YES\nTEST_TUNING=NO\n'%B)
write_sha256_manifest(OUT)
print('E6_SPECIES_BLOCK_STATS=PASS')
print(OUT)
