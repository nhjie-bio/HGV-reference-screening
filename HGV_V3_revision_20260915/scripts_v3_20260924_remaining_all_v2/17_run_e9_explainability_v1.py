#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'13_explainability/E9_explainability_v1_{DATE_TAG}'
PASS = 'E9_EXPLAINABILITY_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
meta=manifest.set_index(mc['accession'])
X,xidx,_,_=feature_accession_index()
test=load_pairs('G2',20260915,'test')
test_acc=sorted(set(test.accession_a)|set(test.accession_b))
idx={a:i for i,a in enumerate(test_acc)}
Xt=np.asarray(X[[xidx[a] for a in test_acc]],dtype=np.float32)

# Frozen D3 seed15 model for attribution; all 3 seeds remain represented in main/E8 evidence.
seed=20260915
ckpt=main_d3_checkpoint(seed)
model,device,_=load_model_checkpoint(ckpt,2080,64,True)
Z=embed_matrix(model,Xt,device,batch_size=1024)
pai=np.array([idx[a] for a in test.accession_a],int)
pbi=np.array([idx[b] for b in test.accession_b],int)
dist=np.linalg.norm(Z[pai]-Z[pbi],axis=1)
score=1-dist/2

# Confound values. Prefer frozen manifest fields; compute FASTA-derived values for missing core assembly fields.
stats={}
for acc in test_acc:
    r=meta.loc[acc]
    vals={}
    for key in ['gc','length','contigs','n50','completeness']:
        c=mc.get(key)
        try:
            vals[key]=float(r[c]) if c and str(r[c]).strip()!='' else np.nan
        except Exception:
            vals[key]=np.nan
    if not np.isfinite(vals['gc']) or not np.isfinite(vals['length']) or not np.isfinite(vals['contigs']) or not np.isfinite(vals['n50']):
        fs=fasta_stats(fmap[acc])
        if not np.isfinite(vals['gc']): vals['gc']=fs['gc_fraction']
        elif vals['gc']>1.0: vals['gc']/=100.0
        if not np.isfinite(vals['length']): vals['length']=fs['genome_length']
        if not np.isfinite(vals['contigs']): vals['contigs']=fs['contig_count']
        if not np.isfinite(vals['n50']): vals['n50']=fs['n50']
    elif vals['gc']>1.0:
        vals['gc']/=100.0
    stats[acc]=vals

conf_rows=[]
for i,(a,b) in enumerate(zip(test.accession_a,test.accession_b)):
    row={'accession_a':a,'accession_b':b,'label':int(test.label.iloc[i]),'pair_type':test.pair_type.iloc[i],
         'hgv_distance':float(dist[i]),'hgv_similarity_score':float(score[i])}
    for key in ['gc','length','contigs','n50','completeness']:
        va=stats[a][key]; vb=stats[b][key]
        row[f'abs_delta_{key}']=abs(va-vb) if np.isfinite(va) and np.isfinite(vb) else np.nan
    conf_rows.append(row)
conf=pd.DataFrame(conf_rows)
conf.to_csv(OUT/'E9_PAIR_CONFOUND_SOURCE.tsv',sep='\t',index=False)

corr=[]
for subset_name,g in [('all',conf),('positive',conf[conf.label==1]),('negative',conf[conf.label==0])]:
    for col in [c for c in conf.columns if c.startswith('abs_delta_')]:
        rr,pp=spearman(g[col],g.hgv_distance)
        corr.append({'subset':subset_name,'confound':col,'n':int(g[[col,'hgv_distance']].dropna().shape[0]),'spearman_r':rr,'p_value':pp})
for pt,g in conf.groupby('pair_type',sort=True):
    for col in [c for c in conf.columns if c.startswith('abs_delta_')]:
        rr,pp=spearman(g[col],g.hgv_distance)
        corr.append({'subset':f'pair_type:{pt}','confound':col,'n':int(g[[col,'hgv_distance']].dropna().shape[0]),'spearman_r':rr,'p_value':pp})
pd.DataFrame(corr).to_csv(OUT/'E9_CONFOUND_CORRELATIONS.tsv',sep='\t',index=False)

# Integrated-gradient style pair-distance attribution on a deterministic balanced subset.
torch,_,_=import_torch()
rng=np.random.default_rng(20260924)
pos=np.where(test.label.to_numpy(int)==1)[0]; neg=np.where(test.label.to_numpy(int)==0)[0]
sel=np.concatenate([rng.choice(pos,min(128,len(pos)),replace=False),rng.choice(neg,min(128,len(neg)),replace=False)])
steps=int(os.environ.get('E9_IG_STEPS','16'))
attr=np.zeros(2080,dtype=np.float64)
model.eval()
for batch_start in range(0,len(sel),16):
    ix=sel[batch_start:batch_start+16]
    xa=torch.as_tensor(Xt[pai[ix]],device=device,dtype=torch.float32)
    xb=torch.as_tensor(Xt[pbi[ix]],device=device,dtype=torch.float32)
    ga=torch.zeros_like(xa); gb=torch.zeros_like(xb)
    for step in range(1,steps+1):
        alpha=step/steps
        aa=(xa*alpha).detach().requires_grad_(True)
        bb=(xb*alpha).detach().requires_grad_(True)
        zz=model(torch.cat([aa,bb],dim=0)); n=len(ix); za=zz[:n]; zb=zz[n:]
        d=torch.linalg.norm(za-zb,dim=1).sum()
        g1,g2=torch.autograd.grad(d,[aa,bb],retain_graph=False,create_graph=False)
        ga+=g1.detach(); gb+=g2.detach()
    iga=(xa*(ga/steps)).detach().cpu().numpy(); igb=(xb*(gb/steps)).detach().cpu().numpy()
    attr += np.abs(iga).sum(axis=0)+np.abs(igb).sum(axis=0)
attr/=max(1,2*len(sel))
_,kmer_names=build_canonical_vocab(6)
order=np.argsort(-attr)
igdf=pd.DataFrame({'rank':np.arange(1,2081),'canonical_kmer':[kmer_names[i] for i in order],'feature_index':order,'mean_abs_pair_IG':attr[order]})
igdf.to_csv(OUT/'E9_KMER_INTEGRATED_GRADIENT_RANKING.tsv',sep='\t',index=False)

# Permutation importance for the top-50 IG features: permute one feature across test genomes, re-embed, quantify AUROC/AUPRC change.
from sklearn.metrics import roc_auc_score, average_precision_score
base_auc=float(roc_auc_score(test.label.to_numpy(int),score)); base_ap=float(average_precision_score(test.label.to_numpy(int),score))
perm=[]
prng=np.random.default_rng(20260924)
for rank0,fi in enumerate(order[:50],1):
    Xp=Xt.copy(); perm_ix=prng.permutation(len(Xp)); Xp[:,fi]=Xp[perm_ix,fi]
    Zp=embed_matrix(model,Xp,device,batch_size=1024)
    dp=np.linalg.norm(Zp[pai]-Zp[pbi],axis=1); sp=1-dp/2
    auc=float(roc_auc_score(test.label.to_numpy(int),sp)); ap=float(average_precision_score(test.label.to_numpy(int),sp))
    perm.append({'ig_rank':rank0,'feature_index':int(fi),'canonical_kmer':kmer_names[int(fi)],
                 'baseline_AUROC':base_auc,'permuted_AUROC':auc,'AUROC_drop':base_auc-auc,
                 'baseline_AUPRC':base_ap,'permuted_AUPRC':ap,'AUPRC_drop':base_ap-ap})
pd.DataFrame(perm).to_csv(OUT/'E9_TOP50_KMER_PERMUTATION_IMPORTANCE.tsv',sep='\t',index=False)

atomic_text(OUT/'E9_EXPLAINABILITY_REPORT.txt','\n'.join([
    'HGV-V3 E9 EXPLAINABILITY v1',
    'Attribution target: frozen G2 D3 seed20260915 pair distance.',
    f'Integrated-gradient subset: {len(sel)} canonical G2 test pairs, balanced positive/negative where available; zero baseline; {steps} steps.',
    'Permutation audit: top-50 IG canonical 6-mer features permuted across the 356 frozen G2 test genomes, then AUROC/AUPRC recomputed.',
    'Confound audit: absolute pairwise differences in GC, genome length, contig count, N50, and completeness (when available), evaluated overall and by pair class.',
    'These analyses describe model sensitivity/association; they do not assign biological causality to individual k-mers.',
])+'\n')
atomic_json(OUT/'E9_PROVENANCE.json',{'checkpoint_sha256':sha256(ckpt),'feature_sha256':sha256(find_feature_npy()),
                                      'test_pair_sha256':sha256(find_pair_file('G2',20260915,'test')),'IG_steps':steps,'IG_pair_n':len(sel)})
atomic_text(OUT/PASS,'STATUS=PASS\nATTRIBUTION=IG_PLUS_PERMUTATION\nCONFOUND_AUDIT=YES\n')
write_sha256_manifest(OUT)
print('E9_EXPLAINABILITY=PASS')
print(OUT)
