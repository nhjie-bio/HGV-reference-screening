#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'11_sensitivity/E7_sensitivity_v1_{DATE_TAG}'
PASS = OUT / 'E7_SENSITIVITY_PASS.txt'
if PASS.exists():
    print(f'[SKIP] {PASS}'); raise SystemExit(0)
OUT.mkdir(parents=True,exist_ok=True)

workers=int(os.environ.get('HGV_WORKERS','44'))
manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
meta=manifest.set_index(mc['accession'])
A=sorted([a for a,r in meta.iterrows() if infer_cohort(r,mc,accession=a)=='A'])
if len(A)!=2374: die(f'A count !=2374: {len(A)}')

# Compile generic canonical extractor.
src=Path(__file__).resolve().parent/'canonical_kmer_one.cpp'
exe=OUT/'canonical_kmer_one'
if not exe.exists():
    cp=run_cmd(['g++','-O3','-std=c++17','-o',str(exe),str(src)],timeout=120)
    if cp.returncode!=0: die(cp.stderr)


def kdim(k:int)->int:
    # reverse-complement canonical equivalence classes
    # even k has 4^(k/2) self reverse-complement palindromes.
    return (4**k + (4**(k//2) if k%2==0 else 0))//2


def one_extract(args):
    acc,k,path,outbin,exe_s=args
    if outbin.exists() and outbin.stat().st_size==kdim(k)*4:
        return acc,'SKIP'
    tmp=outbin.with_suffix('.tmp')
    cp=subprocess.run([exe_s,str(path),str(k),str(tmp)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if cp.returncode!=0:
        return acc,'FAIL:'+cp.stderr[:300]
    if tmp.stat().st_size!=kdim(k)*4:
        return acc,f'FAIL:dim bytes {tmp.stat().st_size}'
    tmp.replace(outbin)
    return acc,'OK'


def ensure_features(k:int):
    d=OUT/f'features_k{k}'; d.mkdir(parents=True,exist_ok=True)
    npy=d/f'canonical_k{k}_A2374.npy'; order=d/'accession_order.tsv'; marker=d/'FEATURE_PASS.txt'
    if marker.exists() and npy.exists() and order.exists():
        X=np.load(npy,mmap_mode='r')
        if X.shape!=(2374,kdim(k)): die(f'cached k{k} shape mismatch {X.shape}')
        return X,{a:i for i,a in enumerate(pd.read_csv(order,sep='\t').accession.astype(str))}
    raw=d/'raw'; raw.mkdir(exist_ok=True)
    tasks=[(a,k,fmap[a],raw/f'{a}.bin',str(exe)) for a in A]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(one_extract,t) for t in tasks]
        bad=[]
        for i,f in enumerate(as_completed(futs),1):
            a,status=f.result()
            if status.startswith('FAIL'): bad.append((a,status))
            if i%250==0: print(f'[k{k}] extracted/checked {i}/{len(tasks)}')
    if bad: die(f'k{k} extraction failures: {bad[:10]}')
    arr=np.empty((len(A),kdim(k)),dtype=np.float32)
    for i,a in enumerate(A):
        v=np.fromfile(raw/f'{a}.bin',dtype=np.float32)
        if v.size!=kdim(k): die(f'k{k} bad vector {a}: {v.size}')
        arr[i]=v
    np.save(npy,arr)
    pd.DataFrame({'accession':A}).to_csv(order,sep='\t',index=False)
    atomic_text(marker,f'STATUS=PASS\nk={k}\ndim={kdim(k)}\nrows=2374\n')
    return np.load(npy,mmap_mode='r'),{a:i for i,a in enumerate(A)}

# Frozen k6 cache mapped to A only.
X6_all,x6idx,_,_=feature_accession_index()
X6=np.asarray(X6_all[[x6idx[a] for a in A]],dtype=np.float32)
idxA={a:i for i,a in enumerate(A)}

train=load_pairs('G2',20260915,'train')
val=load_pairs('G2',20260915,'val')
test=load_pairs('G2',20260915,'test')
# true production teacher distances, same frozen Dashing2 SetSketch cache.
train_td=load_dashing_teacher_for_pairs(train)
val_td=load_dashing_teacher_for_pairs(val)


def pair_indices(df,idx):
    try:
        aa=np.array([idx[a] for a in df.accession_a],dtype=np.int64)
        bb=np.array([idx[b] for b in df.accession_b],dtype=np.int64)
    except KeyError as e: die(f'Pair accession missing from feature index: {e}')
    return aa,bb,df.label.to_numpy(np.int64)


def train_one(setting_name, X, idx, embed_dim, margin, lam, seed, outdir):
    torch,_,F=import_torch()
    if (outdir/'RUN_COMPLETE').exists():
        return pd.read_json(outdir/'metrics.json',typ='series').to_dict()
    if outdir.exists() and any(outdir.iterdir()):
        die(f'Partial E7 run requires manual inspection (not overwritten): {outdir}')
    outdir.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=make_hgv(X.shape[1],embed_dim,True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    taa,tbb,ty=pair_indices(train,idx); vaa,vbb,vy=pair_indices(val,idx); qaa,qbb,qy=pair_indices(test,idx)
    pos=np.where(ty==1)[0]; neg=np.where(ty==0)[0]
    Xnp=np.asarray(X,dtype=np.float32)

    def forward_pairs(aidx,bidx,training=False):
        za=[]; zb=[]
        model.eval()
        with torch.no_grad():
            for st in range(0,len(aidx),2048):
                ai=aidx[st:st+2048]; bi=bidx[st:st+2048]
                xx=np.concatenate([Xnp[ai],Xnp[bi]],axis=0)
                z=model(torch.as_tensor(xx,device=device))
                n=len(ai); za.append(z[:n].cpu().numpy()); zb.append(z[n:].cpu().numpy())
        return np.concatenate(za),np.concatenate(zb)

    def val_eval():
        za,zb=forward_pairs(vaa,vbb)
        d=np.linalg.norm(za-zb,axis=1); score=1-d/2.0
        from sklearn.metrics import roc_auc_score,average_precision_score
        au=float(roc_auc_score(vy,score)); ap=float(average_precision_score(vy,score))
        con=np.mean(vy*(d**2)+(1-vy)*(np.maximum(margin-d,0.0)**2))
        return au,ap,float(con),score

    best_score=-1; best_con=float('inf'); best_epoch=-1; bad_epochs=0
    history=[]
    for epoch in range(1,51):
        model.train(); rng=np.random.default_rng(seed + epoch*1000003)
        sel=np.concatenate([rng.choice(pos,50000,replace=True),rng.choice(neg,50000,replace=True)])
        rng.shuffle(sel)
        epoch_loss=0.0; nb=0
        for st in range(0,len(sel),512):
            ix=sel[st:st+512]; ai=taa[ix]; bi=tbb[ix]; yy=ty[ix]; td=train_td[ix]
            xx=np.concatenate([Xnp[ai],Xnp[bi]],axis=0)
            z=model(torch.as_tensor(xx,device=device)); n=len(ix); za=z[:n]; zb=z[n:]
            dist=torch.linalg.norm(za-zb,dim=1)
            yyt=torch.as_tensor(yy,dtype=torch.float32,device=device)
            con=(yyt*dist.pow(2)+(1-yyt)*torch.relu(margin-dist).pow(2)).mean()
            mask_np=(yy==1)|((yy==0)&(td<0.98))
            if mask_np.any():
                mask=torch.as_tensor(mask_np,device=device)
                target=torch.as_tensor(td,dtype=torch.float32,device=device)
                student=dist/2.0
                dl=(student[mask]-target[mask]).pow(2).mean()
            else:
                dl=dist.sum()*0.0
            loss=con+lam*dl
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            epoch_loss+=float(loss.detach().cpu()); nb+=1
        au,ap,vcon,_=val_eval(); sel_score=(au+ap)/2.0
        history.append({'epoch':epoch,'train_loss':epoch_loss/max(nb,1),'val_AUROC':au,'val_AUPRC':ap,'val_contrastive_loss':vcon,'selection_score':sel_score})
        improved = (sel_score > best_score + 1e-8) or (abs(sel_score-best_score)<=1e-8 and vcon < best_con)
        if improved:
            best_score=sel_score; best_con=vcon; best_epoch=epoch; bad_epochs=0
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'setting':setting_name,'seed':seed,
                        'input_dim':X.shape[1],'embed_dim':embed_dim,'margin':margin,'lambda_dist':lam},outdir/'best_model.pt')
        else:
            bad_epochs+=1
        if bad_epochs>=8: break
    pd.DataFrame(history).to_csv(outdir/'history.tsv',sep='\t',index=False)
    # reload best and freeze threshold on validation only
    ck=torch.load(outdir/'best_model.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model_state_dict']); model.eval()
    vza,vzb=forward_pairs(vaa,vbb); vd=np.linalg.norm(vza-vzb,axis=1); vs=1-vd/2
    th,vmet=freeze_threshold(vy,vs)
    tza,tzb=forward_pairs(qaa,qbb); td=np.linalg.norm(tza-tzb,axis=1); ts=1-td/2
    tmet=binary_metrics(qy,ts,th)
    metrics={'setting':setting_name,'model_seed':seed,'input_dim':int(X.shape[1]),'embed_dim':embed_dim,'margin':margin,'lambda_dist':lam,
             'best_epoch':best_epoch,'val_threshold':th,'val_AUROC':vmet['AUROC'],'val_AUPRC':vmet['AUPRC'],**{f'test_{k}':v for k,v in tmet.items()}}
    atomic_json(outdir/'metrics.json',metrics)
    pd.DataFrame({'accession_a':test.accession_a,'accession_b':test.accession_b,'label':qy,'pair_type':test.pair_type,
                  'distance':td,'similarity_score':ts,'threshold':th}).to_csv(outdir/'test_predictions.tsv',sep='\t',index=False)
    atomic_text(outdir/'RUN_COMPLETE','STATUS=PASS\n')
    return metrics

# Non-baseline OFAT settings. Canonical D3 k6/d64/m1/lambda.25 is reused from frozen E6/main; never retrained here.
settings=[
    ('k5','k',5,64,1.0,0.25),('k7','k',7,64,1.0,0.25),
    ('dim32','dim',6,32,1.0,0.25),('dim128','dim',6,128,1.0,0.25),
    ('margin0p5','margin',6,64,0.5,0.25),('margin1p5','margin',6,64,1.5,0.25),
    ('lambda0p1','lambda',6,64,1.0,0.10),('lambda0p5','lambda',6,64,1.0,0.50),
]
feature_cache={6:(X6,idxA)}
results=[]
for name,factor,k,dim,margin,lam in settings:
    if k not in feature_cache: feature_cache[k]=ensure_features(k)
    Xk,idx=feature_cache[k]
    for seed in MODEL_SEEDS_3:
        run=OUT/'runs'/name/f'model{seed}'
        print(f'[E7] {name} seed={seed}')
        results.append(train_one(name,Xk,idx,dim,margin,lam,seed,run))

# Frozen canonical baseline from E6 D3 seed15-17 test predictions.
e6=pd.read_csv(E6_PRED_LONG,sep='\t',dtype=str,keep_default_na=False)
mode_c=resolve_col(e6.columns,['mode','ablation_mode']); seed_c=resolve_col(e6.columns,['model_seed','seed'])
yc=resolve_col(e6.columns,['label']); sc=resolve_col(e6.columns,['similarity_score','score']); tc=resolve_col(e6.columns,['threshold','frozen_threshold'])
base_rows=[]
for seed in MODEL_SEEDS_3:
    g=e6[(e6[mode_c]=='D3')&(pd.to_numeric(e6[seed_c])==seed)]
    met=binary_metrics(pd.to_numeric(g[yc]).to_numpy(int),pd.to_numeric(g[sc]).to_numpy(float),float(pd.to_numeric(g[tc]).iloc[0]))
    base_rows.append({'setting':'canonical_k6_dim64_margin1_lambda0p25','model_seed':seed,'input_dim':2080,'embed_dim':64,'margin':1.0,'lambda_dist':0.25,
                      **{f'test_{k}':v for k,v in met.items()},'source':'frozen E6 D3; not retrained'})
allres=pd.DataFrame(base_rows+results)
allres.to_csv(OUT/'E7_RUN_METRICS.tsv',sep='\t',index=False)

# Factor-oriented table includes baseline repeated conceptually for each factor only at summarization.
summ=[]
for setting,g in allres.groupby('setting',sort=False):
    row={'setting':setting,'n_seeds':len(g)}
    for c in ['test_AUROC','test_AUPRC','test_MCC','test_F1','test_BalancedAccuracy']:
        row[c+'_mean']=pd.to_numeric(g[c]).mean(); row[c+'_sd']=pd.to_numeric(g[c]).std(ddof=1)
    first=g.iloc[0]
    for c in ['input_dim','embed_dim','margin','lambda_dist']: row[c]=first.get(c,'')
    summ.append(row)
pd.DataFrame(summ).to_csv(OUT/'E7_SENSITIVITY_SUMMARY.tsv',sep='\t',index=False)

atomic_text(OUT/'E7_SENSITIVITY_REPORT.txt','\n'.join([
    'HGV-V3 E7 OFAT SENSITIVITY v1',
    'Frozen canonical baseline: k=6, dim=64, margin=1.0, lambda=0.25, D3; reused from frozen E6 D3 seeds 20260915/16/17 and not retrained.',
    'New settings: k=5/7; dim=32/128; margin=0.5/1.5; lambda=0.1/0.5; 3 model seeds each.',
    'All new settings use canonical G2 split_seed=20260915, true frozen Dashing2 SetSketch teacher, tau=0.98, same optimizer/batch/epoch sampling/early stopping policy.',
    'Threshold for each new run is selected on validation only and frozen before its test evaluation.',
])+'\n')
atomic_json(OUT/'E7_PROVENANCE.json',{'teacher_sha256':sha256(TEACHER_DASHING),'manifest_sha256':sha256(FINAL_MANIFEST),'workers':workers,
                                      'canonical_baseline_source':str(E6_PRED_LONG),'extractor_source_sha256':sha256(src)})
atomic_text(PASS,'STATUS=PASS\nOFAT=YES\nNEW_RUNS=24\nCANONICAL_BASELINE_RETRAINED=NO\nTEST_TUNING=NO\n')
write_sha256_manifest(OUT)
print('E7_SENSITIVITY=PASS')
print(OUT)
