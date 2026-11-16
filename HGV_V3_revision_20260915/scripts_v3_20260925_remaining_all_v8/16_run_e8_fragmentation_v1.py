#!/usr/bin/env python3
from __future__ import annotations
import hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'12_fragmentation/E8_fragmentation_v1_{DATE_TAG}'
PASS = 'E8_FRAGMENTATION_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
Xfull,xidx,_,_=feature_accession_index()
test=load_pairs('G2',20260915,'test')
test_acc=sorted(set(test.accession_a)|set(test.accession_b))
if len(test_acc)!=356:
    die(f'Expected 356 G2 seed15 test genomes, got {len(test_acc)}')
idx={a:i for i,a in enumerate(test_acc)}
code_to_idx,kmer_names=build_canonical_vocab(6)

# Deterministic sequence amount generator; never crosses contig boundaries for k-mer counting.
def rng_for(acc,cond):
    h=hashlib.sha256(f'{acc}|{cond}|20260924'.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],'little'))

def sample_contigs(acc, cond):
    contigs=load_fasta_contigs(fmap[acc])
    if cond=='full': return contigs, sum(map(len,contigs)), len(contigs)
    rng=rng_for(acc,cond)
    if cond=='5x5kb':
        fr=[]
        for _ in range(5):
            candidates=[c for c in contigs if len(c)>=5000]
            if candidates:
                c=candidates[int(rng.integers(len(candidates)))]; st=int(rng.integers(0,len(c)-5000+1)); fr.append(c[st:st+5000])
            else:
                # deterministic fallback: accumulate up to 5kb from a shuffled contig order, preserving boundaries.
                order=rng.permutation(len(contigs)); rem=5000
                for j in order:
                    if rem<=0: break
                    c=contigs[int(j)]
                    if not c: continue
                    take=min(rem,len(c)); st=int(rng.integers(0,len(c)-take+1)); fr.append(c[st:st+take]); rem-=take
        return fr,sum(map(len,fr)),len(fr)
    target={'100kb':100000,'50kb':50000,'20kb':20000,'5kb':5000}[cond]
    order=rng.permutation(len(contigs)); fr=[]; rem=target
    for j in order:
        if rem<=0: break
        c=contigs[int(j)]
        if not c: continue
        take=min(rem,len(c)); st=int(rng.integers(0,len(c)-take+1)); fr.append(c[st:st+take]); rem-=take
    # if assembly shorter than target, use all available sequence once; no duplicated sequence is invented.
    return fr,sum(map(len,fr)),len(fr)

conditions=['full','100kb','50kb','20kb','5kb','5x5kb']
features={}
amount_rows=[]
for cond in conditions:
    mat=np.empty((len(test_acc),2080),dtype=np.float32)
    for i,acc in enumerate(test_acc):
        if cond=='full':
            mat[i]=np.asarray(Xfull[xidx[acc]],dtype=np.float32)
            stats=fasta_stats(fmap[acc]); bp=int(stats['genome_length']); nf=int(stats['contig_count'])
        else:
            fr,bp,nf=sample_contigs(acc,cond)
            mat[i]=count_canonical_kmers_from_contigs(fr,6,code_to_idx)
        amount_rows.append({'condition':cond,'accession':acc,'realized_bp':bp,'n_fragments':nf,'feature_sum':float(mat[i].sum())})
    features[cond]=mat
    np.save(OUT/f'E8_features_{cond}.npy',mat)
pd.DataFrame(amount_rows).to_csv(OUT/'E8_REALIZED_SEQUENCE_AMOUNT.tsv',sep='\t',index=False)
pd.DataFrame({'accession':test_acc}).to_csv(OUT/'E8_TEST_GENOME_ORDER.tsv',sep='\t',index=False)

# Frozen D3 full-trained main models and validation-derived thresholds.
pair_ai=np.array([idx[a] for a in test.accession_a],dtype=int)
pair_bi=np.array([idx[b] for b in test.accession_b],dtype=int)
y=test.label.to_numpy(int)
rows=[]; pred_rows=[]
for seed in MODEL_SEEDS_3:
    ckpt=main_d3_checkpoint(seed)
    model,device,_=load_model_checkpoint(ckpt,2080,64,True)
    th=find_threshold(MAIN_THRESHOLD_FILE,seed,'G2',20260915)
    for cond in conditions:
        Z=embed_matrix(model,features[cond],device,batch_size=1024)
        d=np.linalg.norm(Z[pair_ai]-Z[pair_bi],axis=1); score=1-d/2.0
        met=binary_metrics(y,score,th)
        rows.append({'model_seed':seed,'condition':cond,'threshold_from_full_validation':th,'n_pairs':len(y),**met})
        for i in range(len(y)):
            pred_rows.append({'model_seed':seed,'condition':cond,'accession_a':test.accession_a.iloc[i],'accession_b':test.accession_b.iloc[i],
                              'label':int(y[i]),'pair_type':test.pair_type.iloc[i],'distance':float(d[i]),'similarity_score':float(score[i]),'frozen_threshold':th})
metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'E8_RUN_METRICS.tsv',sep='\t',index=False)
pd.DataFrame(pred_rows).to_csv(OUT/'E8_PAIR_PREDICTIONS_LONG.tsv',sep='\t',index=False)

summ=[]
for cond,g in metrics.groupby('condition',sort=False):
    row={'condition':cond,'n_seeds':len(g)}
    for c in ['AUROC','AUPRC','MCC','F1','BalancedAccuracy']:
        row[c+'_mean']=g[c].mean(); row[c+'_sd']=g[c].std(ddof=1)
    summ.append(row)
pd.DataFrame(summ).to_csv(OUT/'E8_FRAGMENTATION_SUMMARY.tsv',sep='\t',index=False)

atomic_text(OUT/'E8_FRAGMENTATION_REPORT.txt','\n'.join([
    'HGV-V3 E8 FRAGMENTATION ROBUSTNESS v1',
    'No model retraining is performed. The frozen full-assembly G2 D3 models are evaluated on reduced sequence amount.',
    'Conditions: full, 100kb, 50kb, 20kb, 5kb, 5x5kb.',
    'Fragment selection is deterministic by accession+condition; k-mers never cross contig or sampled-fragment boundaries.',
    'The original full-assembly validation-derived threshold is retained for every fragmentation condition; no fragment-specific threshold is selected.',
])+'\n')
atomic_json(OUT/'E8_PROVENANCE.json',{
    'test_pairs_sha256':sha256(find_pair_file('G2',20260915,'test')),
    'full_feature_sha256':sha256(find_feature_npy()),
    'checkpoints':{str(s):sha256(main_d3_checkpoint(s)) for s in MODEL_SEEDS_3},
    'threshold_file_sha256':sha256(MAIN_THRESHOLD_FILE),
})
atomic_text(OUT/PASS,'STATUS=PASS\nMODEL_RETRAINING=NO\nFRAGMENT_THRESHOLD_RETUNING=NO\n')
write_sha256_manifest(OUT)
print('E8_FRAGMENTATION=PASS')
print(OUT)
