#!/usr/bin/env python3
from __future__ import annotations
import json, os, resource, shutil, subprocess, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from common import *

OUT = BASE / f'16_scalability/E12_scalability_v1_{DATE_TAG}'
PASS = 'E12_SCALABILITY_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

E11 = BASE / f'15_reference_screening/E11_reference_screening_v1_{DATE_TAG}'
if not (E11/'E11_REFERENCE_SCREENING_PASS.txt').exists():
    die(f'E11 PASS required before E12: {E11}')

# Compile exact canonical extractor used only for benchmark/re-extraction; frozen feature cache is never modified.
src=Path(__file__).resolve().parent/'canonical_kmer_one.cpp'
bin_path=OUT/'canonical_kmer_one'
cp=run_cmd(['g++','-O3','-std=c++17','-o',str(bin_path),str(src)],timeout=120)
if cp.returncode!=0:
    die(f'Failed to compile benchmark extractor: {cp.stderr}')

manifest=load_manifest(); mc=manifest_columns(manifest); fmap=get_fasta_map(manifest)
meta=manifest.set_index(mc['accession'])
A=[a for a,r in meta.iterrows() if infer_cohort(r,mc)=='A']
ALL=sorted(meta.index.astype(str).tolist())
EXT=[a for a,r in meta.iterrows() if infer_cohort(r,mc)!='A']
# fixed query set: first 50 B1 accessions in lexical order (deterministic, external-only)
A_species=set(meta.loc[A,mc['species']].astype(str))
B1=sorted([a for a in EXT if str(meta.loc[a,mc['species']]) in A_species])
Q=B1[:int(os.environ.get('E12_QUERY_N','50'))]
if len(Q)<10: die('Insufficient external B1 queries for E12')

ckpt=canonical_a_checkpoints()[20260915]
model,device,_=load_model_checkpoint(ckpt,2080,64,True)
workers=int(os.environ.get('HGV_WORKERS','44'))
repeats=int(os.environ.get('E12_REAL_REPEATS','3'))
Ns=[100,500,1000,2000,3000]
Ns=sorted(set([n for n in Ns if n<=len(ALL)]))

# Deterministic nested real-genome database order from all 3,154 frozen genomes.
# This is a runtime/scalability benchmark, not an accuracy/relevance evaluation.
DB_order=ALL

def one_extract(args):
    acc, fasta, outbin, exe = args
    t0=time.perf_counter()
    cp=subprocess.run([exe,str(fasta),'6',str(outbin)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    dt=time.perf_counter()-t0
    if cp.returncode!=0:
        return acc,dt,False,cp.stderr[:500]
    return acc,dt,True,cp.stdout.strip()[:200]


def extract_batch(accessions, workdir: Path):
    tasks=[]
    for acc in accessions:
        tasks.append((acc,fmap[acc],workdir/f'{acc}.bin',str(bin_path)))
    t0=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(one_extract,x) for x in tasks]
        for f in as_completed(futs):
            rows.append(f.result())
    wall=time.perf_counter()-t0
    bad=[r for r in rows if not r[2]]
    if bad: die(f'Feature extraction failed for {len(bad)} genomes, examples={bad[:3]}')
    mat=[]
    for acc in accessions:
        a=np.fromfile(workdir/f'{acc}.bin',dtype=np.float32)
        if a.size!=2080: die(f'Unexpected k6 dimension for {acc}: {a.size}')
        mat.append(a)
    return np.stack(mat),wall,rows

bench=[]
# Query feature extraction once per repeat, plus each N build re-extraction. This is real FASTA I/O, no cache shortcut.
for rep in range(1,repeats+1):
    with tempfile.TemporaryDirectory(prefix=f'e12_rep{rep}_') as td:
        td=Path(td)
        # Need directories to exist inside helper call.
        (td/'queries').mkdir(parents=True,exist_ok=True)
        qmat,qwall,qrows=extract_batch(Q,td/'queries')
        tq0=time.perf_counter(); qz=embed_matrix(model,qmat,device,batch_size=4096); qinfer=time.perf_counter()-tq0
        for n in Ns:
            rdir=td/f'refs_{n}'; rdir.mkdir(parents=True,exist_ok=True)
            refs=DB_order[:n]
            rmat,build_wall,rrows=extract_batch(refs,rdir)
            t0=time.perf_counter(); rz=embed_matrix(model,rmat,device,batch_size=4096); infer_wall=time.perf_counter()-t0
            # index = contiguous float32 vector database
            idxfile=td/f'hgv_vectors_{n}.npy'; np.save(idxfile,rz)
            # exact query scan; 5 timing repeats, first cold-ish, remaining warm
            qtimes=[]
            for qr in range(5):
                t=time.perf_counter()
                # squared Euclidean exact scan; retrieve top50
                for qvec in qz:
                    d=np.sum((rz-qvec[None,:])**2,axis=1)
                    _=np.argpartition(d,min(49,len(d)-1))[:min(50,len(d))]
                qtimes.append(time.perf_counter()-t)
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            bench.append({
                'repeat':rep,'N_reference':n,'Q':len(Q),
                'real_fasta_reference_feature_wall_s':build_wall,
                'real_fasta_query_feature_wall_s':qwall,
                'reference_model_inference_wall_s':infer_wall,
                'query_model_inference_wall_s':qinfer,
                'exact_scan_cold_s':qtimes[0],
                'exact_scan_warm_median_s':float(np.median(qtimes[1:])),
                'vector_index_bytes':idxfile.stat().st_size,
                'bytes_per_genome':idxfile.stat().st_size/n,
                'process_maxrss_kb':rss,
                'hardware_threads_requested':workers,
            })
bench_df=pd.DataFrame(bench)
bench_df.to_csv(OUT/'E12_REAL_FASTA_END_TO_END.tsv',sep='\t',index=False)

# Summaries cold + warm median across three independently re-extracted real-FASTA repeats.
sumrows=[]
for n,g in bench_df.groupby('N_reference'):
    row={'N_reference':n,'repeats':len(g),'Q':int(g.Q.iloc[0])}
    for c in ['real_fasta_reference_feature_wall_s','real_fasta_query_feature_wall_s','reference_model_inference_wall_s','query_model_inference_wall_s','exact_scan_cold_s','exact_scan_warm_median_s','vector_index_bytes','bytes_per_genome','process_maxrss_kb']:
        row[c+'_median']=float(g[c].median()); row[c+'_min']=float(g[c].min()); row[c+'_max']=float(g[c].max())
    sumrows.append(row)
pd.DataFrame(sumrows).to_csv(OUT/'E12_REAL_FASTA_SUMMARY.tsv',sep='\t',index=False)

# Vector-search microbenchmark, explicitly synthetic expansion from real frozen 64-D reference vectors.
ref3=np.load(E11/'E11_REFERENCE_EMBEDDINGS_3SEED.npy')
basevec=ref3.mean(axis=0).astype(np.float32)
qvec=np.load(E11/'E11_QUERY_EMBEDDINGS_3SEED.npy').mean(axis=0).astype(np.float32)[:10]
vecrows=[]
for n in [10_000,100_000,1_000_000]:
    reps=int(np.ceil(n/len(basevec)))
    db=np.tile(basevec,(reps,1))[:n].copy()
    times=[]
    for j in range(5):
        t=time.perf_counter()
        for q in qvec:
            d=np.sum((db-q[None,:])**2,axis=1)
            _=np.argpartition(d,49)[:50]
        times.append(time.perf_counter()-t)
    vecrows.append({'N_vectors':n,'Q':len(qvec),'dimension':64,'database_bytes':db.nbytes,
                    'exact_scan_first_s':times[0],'exact_scan_warm_median_s':float(np.median(times[1:])),
                    'data_origin':'synthetic tiling of real frozen HGV vectors; not real million-genome end-to-end'})
    del db
# Optional FAISS, if installed.
try:
    import faiss
    for n in [10_000,100_000,1_000_000]:
        reps=int(np.ceil(n/len(basevec))); db=np.tile(basevec,(reps,1))[:n].copy()
        t=time.perf_counter(); index=faiss.IndexFlatL2(64); index.add(db); build=time.perf_counter()-t
        times=[]
        for j in range(5):
            t=time.perf_counter(); _=index.search(qvec,50); times.append(time.perf_counter()-t)
        vecrows.append({'N_vectors':n,'Q':len(qvec),'dimension':64,'database_bytes':db.nbytes,
                        'exact_scan_first_s':times[0],'exact_scan_warm_median_s':float(np.median(times[1:])),
                        'index_build_s':build,'data_origin':'FAISS IndexFlatL2 on synthetic tiling of real frozen HGV vectors'})
        del db,index
except Exception:
    pass
pd.DataFrame(vecrows).to_csv(OUT/'E12_VECTOR_MICROBENCHMARK.tsv',sep='\t',index=False)

atomic_text(OUT/'E12_SCALABILITY_REPORT.txt','\n'.join([
    'HGV-V3 E12 SCALABILITY v1',
    'SC1 uses real frozen FASTA paths and re-extracts canonical k=6 features; it does not time only cached vectors.',
    f'Real frozen-genome database N={Ns}; fixed external B1 query Q={len(Q)}; repeats={repeats}.',
    'Build, model inference, exact query scan, index bytes, bytes/genome, and process RSS are reported separately.',
    'The 10^4/10^5/10^6 vector benchmark is explicitly a vector-search microbenchmark created by tiling real frozen 64-D vectors; it is not million-real-genome end-to-end evidence.',
    'No database-size-independent complexity claim is made; exact vector scan remains O(Nd), d=64.',
])+'\n')
atomic_json(OUT/'E12_PROVENANCE.json',{
    'manifest_sha256':sha256(FINAL_MANIFEST),'canonical_A_checkpoint_sha256':sha256(ckpt),
    'extractor_source_sha256':sha256(src),'extractor_binary_sha256':sha256(bin_path),
    'workers':workers,'real_repeats':repeats,'N':Ns,'Q':Q,
})
atomic_text(OUT/PASS,'STATUS=PASS\nREAL_FASTA_BENCHMARK=YES\nVECTOR_MICROBENCHMARK_SEPARATE=YES\n')
write_sha256_manifest(OUT)
print('E12_SCALABILITY=PASS')
print(OUT)
