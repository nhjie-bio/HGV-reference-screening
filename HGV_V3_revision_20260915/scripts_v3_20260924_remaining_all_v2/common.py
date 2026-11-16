#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE = Path(os.environ.get('HGV_BASE', 'HGV_V3_revision_20260915')).resolve()
DATE_TAG = os.environ.get('HGV_RUN_TAG', '20260924')

FINAL_MANIFEST = BASE / '01_manifest/final_3154_freeze_v1_20260916/final_master_manifest_3154.tsv'
FINAL_MANIFEST_SHA256 = 'b7c34ba059e221822eca574509f5e8b3a751862f2b1e1a9a8f03620cda40cae3'
FEATURE_DIR = BASE / '03_features/canonical_k6_full3154_v1'
PAIR_ROOT = BASE / '05_pairs/pair_manifests_v1_20260916'
TEACHER_DASHING = BASE / '06_teacher/teacher_cache_v1_20260916/dashing2_set/dashing2_set_similarity.tsv'
MAIN_D3_ROOT = BASE / '08_main_generalization/hgv_d3_dashing2_tau0p98_bn_safe_v1_20260916'
MAIN_THRESHOLD_FILE = BASE / '08_main_generalization/validation_threshold_freeze_v1_20260917/FROZEN_VALIDATION_THRESHOLDS.tsv'
CANONICAL_A_ROOT = BASE / '09_external_validation/canonical_A_model_training_v1_20260917'
EXTERNAL_EVAL_ROOT = BASE / '09_external_validation/external_one_shot_evaluation_v1_20260917'
E6ROOT = BASE / '10_ablation/E6_D0_D5_G2_seed20260915_5seed_final_v1_20260922'
E6_TEST_ROOT = E6ROOT / '06_canonical_g2_test_v1_20260924'
E6_PRED_LONG = E6_TEST_ROOT / 'E6_G2_TEST_PAIR_PREDICTIONS_LONG.tsv'
E6_RUN_METRICS = E6_TEST_ROOT / 'E6_G2_TEST_RUN_METRICS.tsv'
E6_TEST_FREEZE = E6_TEST_ROOT / 'E6_G2_TEST_FREEZE_v1.txt'
E6_THRESHOLD_ROOT = E6ROOT / '05_validation_threshold_freeze_v1_20260924'
E6_THRESHOLDS = E6_THRESHOLD_ROOT / 'E6_FROZEN_VALIDATION_THRESHOLDS.tsv'
E6_MANIFEST_ROOT = E6ROOT / '04_canonical_30cell_manifest_v1_20260923'
E6_MANIFEST = E6_MANIFEST_ROOT / 'E6_30CELL_CANONICAL_MANIFEST.tsv'

SCRIPTS_SOURCE_E6 = BASE / 'scripts_v3_20260922_e6_final/03_train_e6_d0_d5_from_17b_v1.py'

MODEL_SEEDS_3 = [20260915, 20260916, 20260917]
MODEL_SEEDS_5 = [20260915, 20260916, 20260917, 20260918, 20260919]
KS = [1, 5, 10, 20, 50]


def die(msg: str, code: int = 2) -> None:
    print(f'[FAIL] {msg}', file=sys.stderr)
    raise SystemExit(code)


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + '\n')


def ensure_new_output(out: Path, pass_marker: str) -> bool:
    """Return True if stage should run, False if already PASS. Never overwrite partial outputs."""
    marker = out / pass_marker
    if marker.exists():
        print(f'[SKIP] Existing PASS: {marker}')
        return False
    if out.exists() and any(out.iterdir()):
        die(f'Output directory is non-empty but no PASS marker exists: {out}\nInspect/remove only the incomplete new-stage directory; frozen inputs are never touched.')
    out.mkdir(parents=True, exist_ok=True)
    return True


def write_sha256_manifest(root: Path, filename: str = 'SHA256SUMS.txt') -> Path:
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != filename:
            rows.append(f'{sha256(p)}  {p.relative_to(root)}')
    out = root / filename
    atomic_text(out, '\n'.join(rows) + ('\n' if rows else ''))
    return out


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def resolve_col(cols: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # relaxed normalized match
    norm = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in cols}
    for cand in candidates:
        k = re.sub(r'[^a-z0-9]+', '', cand.lower())
        if k in norm:
            return norm[k]
    if required:
        die(f'Cannot resolve column among candidates={candidates}; available={list(cols)}')
    return None


def load_manifest() -> pd.DataFrame:
    if not FINAL_MANIFEST.exists():
        die(f'Missing final manifest: {FINAL_MANIFEST}')
    got = sha256(FINAL_MANIFEST)
    if got != FINAL_MANIFEST_SHA256:
        die(f'Final manifest SHA mismatch. expected={FINAL_MANIFEST_SHA256} got={got}')
    df = pd.read_csv(FINAL_MANIFEST, sep='\t', dtype=str, keep_default_na=False)
    if len(df) != 3154:
        die(f'Final manifest row count must be 3154, got {len(df)}')
    return df


def manifest_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        'accession': resolve_col(cols, ['assembly_accession','accession','genome_accession','accession_version']),
        'cohort': resolve_col(cols, ['cohort','dataset','set','cohort_name','analysis_cohort'], required=False),
        'subtype': resolve_col(cols, ['subtype','cohort_subtype','external_subtype','b2_subtype'], required=False),
        'species': resolve_col(cols, ['species_cluster','gtdb_species_cluster','species','gtdb_species']),
        'genus': resolve_col(cols, ['genus','gtdb_genus']),
        'family': resolve_col(cols, ['family','gtdb_family']),
        'fasta': resolve_col(cols, ['fasta_path','genome_fasta','fasta','path','genomic_fna_path']),
        'biosample': resolve_col(cols, ['biosample','biosample_accession'], required=False),
        'gc': resolve_col(cols, ['gc_percent','gc','gc_content','gc_fraction'], required=False),
        'length': resolve_col(cols, ['genome_length','assembly_length','total_length','size_bp','genome_size'], required=False),
        'contigs': resolve_col(cols, ['contig_count','contigs','num_contigs'], required=False),
        'n50': resolve_col(cols, ['contig_n50','n50','n50_bp'], required=False),
        'completeness': resolve_col(cols, ['completeness','checkm_completeness','gtdb_completeness'], required=False),
    }


def infer_cohort(row: pd.Series, mcols: Dict[str, Optional[str]]) -> str:
    acc = str(row[mcols['accession']])
    text_parts = []
    for k in ['cohort','subtype']:
        c = mcols.get(k)
        if c:
            text_parts.append(str(row[c]))
    text = ' '.join(text_parts).lower().replace('_','-')
    if re.search(r'\bb1\b|matched[- ]species', text):
        return 'B1'
    if re.search(r'\bb2\b', text) and ('unseen-genus' in text or 'unseen genus' in text):
        return 'B2_unseen_genus'
    if re.search(r'\bb2\b', text) and ('seen-genus' in text or 'seen genus' in text):
        return 'B2_seen_genus'
    if 'b2-unseen' in text or 'b2_unseen' in text:
        return 'B2_unseen_genus'
    if 'b2-seen' in text or 'b2_seen' in text:
        return 'B2_seen_genus'
    if re.search(r'\ba\b|development|refseq', text):
        return 'A'
    # Frozen design fallback: development GCF, external GCA.
    if acc.startswith('GCF_'):
        return 'A'
    if acc.startswith('GCA_'):
        # Need subtype if unavailable. Use explicit taxonomy relationship later; temporary B2_unknown.
        return 'EXTERNAL_UNKNOWN'
    return 'UNKNOWN'


def normalize_fasta_path(p: str) -> Path:
    q = Path(p)
    if not q.is_absolute():
        q = BASE / q
    return q


def open_fasta(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
    return open(path, 'rt', encoding='utf-8', errors='ignore')


def fasta_stats(path: Path) -> Dict[str, float]:
    lengths: List[int] = []
    gc = 0
    acgt = 0
    cur = 0
    with open_fasta(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    lengths.append(cur)
                cur = 0
                continue
            s = line.strip().upper()
            cur += len(s)
            gc += s.count('G') + s.count('C')
            acgt += sum(s.count(b) for b in 'ACGT')
        if cur:
            lengths.append(cur)
    total = sum(lengths)
    n50 = 0
    if lengths:
        cum = 0
        for x in sorted(lengths, reverse=True):
            cum += x
            if cum >= total / 2:
                n50 = x
                break
    return {
        'genome_length': float(total),
        'contig_count': float(len(lengths)),
        'n50': float(n50),
        'gc_fraction': float(gc / acgt) if acgt else float('nan'),
    }


def _find_feature_npy() -> Path:
    if not FEATURE_DIR.exists():
        die(f'Missing feature dir: {FEATURE_DIR}')
    candidates = sorted(FEATURE_DIR.rglob('*.npy'))
    good = []
    for p in candidates:
        try:
            a = np.load(p, mmap_mode='r')
            if a.ndim == 2 and a.shape == (3154, 2080):
                good.append(p)
        except Exception:
            pass
    if len(good) != 1:
        die(f'Expected exactly one (3154,2080) npy in {FEATURE_DIR}, found {good}')
    return good[0]


def _find_feature_rows() -> Path:
    # Prefer filenames suggesting row/index/accession order.
    cand = []
    for p in FEATURE_DIR.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.tsv','.csv','.txt'}:
            n = p.name.lower()
            score = sum(x in n for x in ['row','index','accession','manifest'])
            cand.append((score, p))
    cand.sort(key=lambda x: (-x[0], str(x[1])))
    for _, p in cand:
        try:
            sep = '\t' if p.suffix.lower() != '.csv' else ','
            df = pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)
            if len(df) == 3154:
                try:
                    resolve_col(df.columns, ['assembly_accession','accession','genome_accession','accession_version'])
                    return p
                except SystemExit:
                    pass
        except Exception:
            pass
    die(f'Could not find 3154-row feature row-index file under {FEATURE_DIR}')


def find_feature_npy() -> Path:
    return _find_feature_npy()


def find_feature_rows() -> Path:
    return _find_feature_rows()


def load_feature_cache() -> Tuple[np.ndarray, pd.DataFrame, str]:
    npy = _find_feature_npy()
    rows = _find_feature_rows()
    X = np.load(npy, mmap_mode='r')
    rdf = pd.read_csv(rows, sep='\t' if rows.suffix.lower() != '.csv' else ',', dtype=str, keep_default_na=False)
    acc_col = resolve_col(rdf.columns, ['assembly_accession','accession','genome_accession','accession_version'])
    if len(set(rdf[acc_col])) != 3154:
        die(f'Feature row index accessions are not unique: {rows}')
    return X, rdf, acc_col


def feature_accession_index() -> Tuple[np.ndarray, Dict[str,int], pd.DataFrame, str]:
    X, rows, acc_col = load_feature_cache()
    index = {a:i for i,a in enumerate(rows[acc_col].astype(str))}
    return X, index, rows, acc_col


def find_pair_file(protocol: str, split_seed: int, role: str) -> Path:
    d = PAIR_ROOT / f'{protocol}_seed{split_seed}'
    p = d / f'{role}_pairs.tsv'
    if not p.exists():
        # relaxed fallback
        matches = list(PAIR_ROOT.rglob(f'{role}_pairs.tsv'))
        matches = [x for x in matches if protocol in str(x.parent) and str(split_seed) in str(x.parent)]
        if len(matches) == 1:
            return matches[0]
        die(f'Missing pair file for {protocol} seed={split_seed} role={role}: {p}')
    return p


def load_pairs(protocol: str='G2', split_seed: int=20260915, role: str='test') -> pd.DataFrame:
    p = find_pair_file(protocol, split_seed, role)
    df = pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False)
    ac = resolve_col(df.columns, ['accession_a','genome_a','a_accession'])
    bc = resolve_col(df.columns, ['accession_b','genome_b','b_accession'])
    lc = resolve_col(df.columns, ['label','y','same_species'])
    pc = resolve_col(df.columns, ['pair_type','difficulty','negative_stratum'], required=False)
    out = pd.DataFrame({
        'accession_a': df[ac].astype(str),
        'accession_b': df[bc].astype(str),
        'label': pd.to_numeric(df[lc], errors='raise').astype(int),
        'pair_type': df[pc].astype(str) if pc else np.where(pd.to_numeric(df[lc]).astype(int)==1,'positive','negative'),
    })
    return out


def canonical_pair_key(a: str, b: str) -> Tuple[str,str]:
    return (a,b) if a <= b else (b,a)


def build_canonical_vocab(k: int) -> Tuple[np.ndarray, List[str]]:
    n = 4 ** k
    rc = np.zeros(n, dtype=np.int64)
    def revcomp_code(x: int) -> int:
        y = 0
        for _ in range(k):
            b = x & 3
            y = (y << 2) | (3 - b)
            x >>= 2
        return y
    reps = []
    for x in range(n):
        r = revcomp_code(x)
        rc[x] = r
        if x <= r:
            reps.append(x)
    rep_to_idx = {r:i for i,r in enumerate(reps)}
    code_to_idx = np.empty(n, dtype=np.int32)
    for x in range(n):
        code_to_idx[x] = rep_to_idx[min(x, int(rc[x]))]
    alphabet = 'ACGT'
    names = []
    for r in reps:
        x = r
        s = ['A'] * k
        for i in range(k-1, -1, -1):
            s[i] = alphabet[x & 3]
            x >>= 2
        names.append(''.join(s))
    return code_to_idx, names


def count_canonical_kmers_from_contigs(contigs: Iterable[str], k: int, code_to_idx: np.ndarray) -> np.ndarray:
    out = np.zeros(int(code_to_idx.max()) + 1, dtype=np.float64)
    mask = (1 << (2*k)) - 1
    m = {'A':0,'C':1,'G':2,'T':3}
    valid_total = 0
    for seq in contigs:
        code = 0
        run = 0
        for ch in seq.upper():
            b = m.get(ch)
            if b is None:
                code = 0; run = 0; continue
            code = ((code << 2) | b) & mask
            run += 1
            if run >= k:
                out[code_to_idx[code]] += 1.0
                valid_total += 1
    if valid_total:
        out /= valid_total
    return out.astype(np.float32)


def load_fasta_contigs(path: Path) -> List[str]:
    contigs=[]; cur=[]
    with open_fasta(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur:
                    contigs.append(''.join(cur).upper()); cur=[]
            else:
                cur.append(line.strip())
        if cur:
            contigs.append(''.join(cur).upper())
    return contigs


def extract_accession(text: str) -> Optional[str]:
    m = re.search(r'(GC[AF]_\d+(?:\.\d+)?)', text)
    return m.group(1) if m else None


def load_dashing_teacher_for_pairs(pairs: pd.DataFrame, path: Path = TEACHER_DASHING) -> np.ndarray:
    """Load true Dashing2 similarity for requested pairs only. Supports edge-list and common triangular-matrix layouts."""
    if not path.exists():
        die(f'Missing Dashing2 teacher: {path}')
    required = {canonical_pair_key(a,b) for a,b in zip(pairs.accession_a, pairs.accession_b)}
    found: Dict[Tuple[str,str], float] = {}

    # First try pandas edge-list header.
    try:
        head = pd.read_csv(path, sep='\t', nrows=5, dtype=str, keep_default_na=False, comment=None)
        cols = list(head.columns)
        ac = resolve_col(cols, ['accession_a','source_a','genome_a','a'], required=False)
        bc = resolve_col(cols, ['accession_b','source_b','genome_b','b'], required=False)
        sc = resolve_col(cols, ['similarity','set_similarity','score','jaccard'], required=False)
        if ac and bc and sc:
            for chunk in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=200000):
                for a,b,s in zip(chunk[ac], chunk[bc], chunk[sc]):
                    aa = extract_accession(str(a)) or str(a)
                    bb = extract_accession(str(b)) or str(b)
                    key = canonical_pair_key(aa,bb)
                    if key in required:
                        try: found[key] = float(s)
                        except Exception: pass
            if len(found) == len(required):
                return np.array([1.0 - found[canonical_pair_key(a,b)] for a,b in zip(pairs.accession_a,pairs.accession_b)], dtype=np.float32)
    except Exception:
        pass

    # Generic text parse.
    with open(path, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = [x.rstrip('\n') for x in f if x.strip()]
    # Generic 3-column edge list without reliable header.
    for line in lines:
        if line.startswith('#'):
            continue
        parts = re.split(r'\t+|\s+', line.strip())
        if len(parts) >= 3:
            a = extract_accession(parts[0]); b = extract_accession(parts[1])
            if a and b:
                try: s = float(parts[2])
                except Exception: continue
                key = canonical_pair_key(a,b)
                if key in required: found[key] = s
    if len(found) == len(required):
        return np.array([1.0 - found[canonical_pair_key(a,b)] for a,b in zip(pairs.accession_a,pairs.accession_b)], dtype=np.float32)

    # Matrix parser. Identify source names from a #Sources header or a block of path-like lines.
    source_names: List[str] = []
    matrix_start = None
    for i,line in enumerate(lines[:5000]):
        if line.startswith('#Sources'):
            parts = line.split('\t')
            accs = [extract_accession(x) for x in parts[1:]]
            accs = [x for x in accs if x]
            if len(accs) > 10:
                source_names = accs
                matrix_start = i+1
                break
    if not source_names:
        # collect consecutive accession-bearing nonnumeric lines after #Sources
        start = None
        for i,line in enumerate(lines):
            if line.startswith('#Sources'):
                start = i+1; break
        if start is not None:
            j=start
            while j < len(lines):
                a = extract_accession(lines[j])
                if not a:
                    break
                # A matrix row can also start with accession and numeric values. Stop when numeric tail appears.
                parts = re.split(r'\t+|\s+', lines[j].strip())
                numeric_tail = 0
                for z in parts[1:]:
                    try: float(z); numeric_tail += 1
                    except Exception: pass
                if numeric_tail:
                    break
                source_names.append(a); j += 1
            if len(source_names) > 10:
                matrix_start = j
    if not source_names:
        # Infer from row labels if each row starts with an accession.
        row_accs=[]
        for line in lines:
            if line.startswith('#'): continue
            parts = re.split(r'\t+|\s+', line.strip())
            a = extract_accession(parts[0]) if parts else None
            if a:
                nums=[]
                for z in parts[1:]:
                    try: nums.append(float(z))
                    except Exception: pass
                if nums:
                    row_accs.append(a)
        if len(row_accs) > 10:
            source_names = row_accs
            matrix_start = 0

    if source_names:
        n = len(source_names)
        idx = {a:i for i,a in enumerate(source_names)}
        req_idx = {}
        for a,b in required:
            if a in idx and b in idx:
                i,j = idx[a], idx[b]
                if i > j: i,j = j,i
                req_idx[(i,j)] = (a,b)
        # Gather matrix rows with numeric values, optional row label.
        rows=[]
        for line in lines[matrix_start or 0:]:
            if line.startswith('#'): continue
            parts = re.split(r'\t+|\s+', line.strip())
            vals=[]
            for z in parts:
                try: vals.append(float(z))
                except Exception: pass
            if vals:
                rows.append(vals)
        if len(rows) >= n:
            rows = rows[:n]
            for i, vals in enumerate(rows):
                # supported layouts: lower triangle i+1; upper triangle n-i; full n.
                if len(vals) == n:
                    for (ii,jj),(a,b) in req_idx.items():
                        if ii == i:
                            found[(a,b)] = vals[jj]
                elif len(vals) in {i+1, i}:
                    # lower triangular incl/excl diagonal
                    offset = 0 if len(vals)==i+1 else 1
                    for (ii,jj),(a,b) in req_idx.items():
                        if jj == i and ii <= i:
                            pos = ii if offset==0 else ii
                            if pos < len(vals): found[(a,b)] = vals[pos]
                elif len(vals) in {n-i, n-i-1}:
                    diag = (len(vals) == n-i)
                    for (ii,jj),(a,b) in req_idx.items():
                        if ii == i and jj >= i:
                            pos = jj-i if diag else jj-i-1
                            if 0 <= pos < len(vals): found[(a,b)] = vals[pos]
        if len(found) == len(required):
            return np.array([1.0 - found[canonical_pair_key(a,b)] for a,b in zip(pairs.accession_a,pairs.accession_b)], dtype=np.float32)

    missing = list(required - set(found))[:10]
    die(f'Could not parse complete Dashing2 teacher for requested pairs. found={len(found)}/{len(required)} missing_examples={missing} path={path}')


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as e:
        die(f'PyTorch unavailable: {e}')


def make_hgv(input_dim: int=2080, embed_dim: int=64, normalize_output: bool=True):
    torch, nn, F = import_torch()
    class HGV(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, embed_dim)
        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            if normalize_output:
                x = F.normalize(x, p=2, dim=1)
            return x
    return HGV()


def checkpoint_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ['model_state_dict','state_dict','model']:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # maybe direct state dict
        if any(str(k).endswith('weight') for k in ckpt):
            return ckpt
    die('Cannot find model state_dict in checkpoint')


def load_model_checkpoint(path: Path, input_dim: int=2080, embed_dim: int=64, normalize_output: bool=True, device: Optional[str]=None):
    torch, _, _ = import_torch()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = checkpoint_state(ckpt)
    # Strip common prefixes.
    nsd = {}
    for k,v in sd.items():
        nk = str(k)
        for pref in ['module.','model.']:
            if nk.startswith(pref): nk = nk[len(pref):]
        nsd[nk] = v
    model = make_hgv(input_dim, embed_dim, normalize_output).to(device)
    model.load_state_dict(nsd, strict=True)
    model.eval()
    return model, device, ckpt


def embed_matrix(model, X: np.ndarray, device: str, batch_size: int=4096) -> np.ndarray:
    torch, _, _ = import_torch()
    outs=[]
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(np.asarray(X[i:i+batch_size], dtype=np.float32), device=device)
            z = model(xb).detach().cpu().numpy().astype(np.float32)
            outs.append(z)
    return np.concatenate(outs, axis=0)


def find_best_model(root: Path, model_seed: int, protocol: Optional[str]=None, split_seed: Optional[int]=None) -> Path:
    """Resolve a checkpoint using exact path components, never date substrings.

    The project/root names themselves contain values such as 20260915/20260916,
    so substring matching can falsely select every checkpoint. Formal checkpoint
    identity is encoded by directory components such as G2/split20260915/model20260915.
    """
    pats = list(root.rglob('best_model.pt'))
    target=[]
    model_part = f'model{int(model_seed)}'
    split_part = f'split{int(split_seed)}' if split_seed is not None else None
    for p in pats:
        parts = set(p.parts)
        if model_part not in parts:
            continue
        if protocol is not None and str(protocol) not in parts:
            continue
        if split_part is not None and split_part not in parts:
            continue
        target.append(p)
    target = sorted(target)
    if len(target) != 1:
        die(
            f'Expected one best_model.pt for seed={model_seed} protocol={protocol} '
            f'split={split_seed} under {root}; got {target}'
        )
    return target[0]


def main_d3_checkpoint(model_seed: int) -> Path:
    return find_best_model(MAIN_D3_ROOT, model_seed, protocol='G2', split_seed=20260915)


def canonical_a_checkpoints() -> Dict[int, Path]:
    out={}
    for seed in MODEL_SEEDS_3:
        out[seed] = find_best_model(CANONICAL_A_ROOT, seed)
    return out


def find_threshold(table: Path, model_seed: int, protocol: str='G2', split_seed: int=20260915, mode: Optional[str]=None) -> float:
    df = pd.read_csv(table, sep='\t', dtype=str, keep_default_na=False)
    cols=list(df.columns)
    tc = resolve_col(cols, ['threshold','frozen_threshold'])
    mc = resolve_col(cols, ['model_seed','seed'])
    pc = resolve_col(cols, ['protocol'], required=False)
    sc = resolve_col(cols, ['split_seed','taxonomy_split_seed'], required=False)
    dc = resolve_col(cols, ['mode','ablation_mode'], required=False)
    method_c = resolve_col(cols, ['method','method_name','representation'], required=False)
    q = df.copy()
    q = q[pd.to_numeric(q[mc], errors='coerce') == model_seed]
    if pc: q = q[q[pc].astype(str) == protocol]
    if sc: q = q[pd.to_numeric(q[sc], errors='coerce') == split_seed]
    if mode and dc:
        q = q[q[dc].astype(str) == mode]
    elif mode and method_c:
        q = q[q[method_c].astype(str).str.upper() == str(mode).upper()]
    # Main frozen threshold tables can contain HGV + RawK6 + PCA rows.
    # If the caller did not name a mode and multiple rows remain, select the
    # HGV row explicitly rather than relying on row order.
    if len(q) > 1 and method_c:
        h = q[q[method_c].astype(str).str.contains('HGV', case=False, regex=False)]
        if len(h) == 1:
            q = h
    if len(q) != 1:
        die(f'Threshold lookup not unique: file={table} seed={model_seed} protocol={protocol} split={split_seed} mode={mode}; rows={len(q)}')
    return float(q.iloc[0][tc])


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float, sample_weight: Optional[np.ndarray]=None) -> Dict[str,float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.asarray(y, dtype=int); score=np.asarray(score,float)
    pred = (score >= threshold).astype(int)
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight,float)
    tp=float(w[(y==1)&(pred==1)].sum()); tn=float(w[(y==0)&(pred==0)].sum())
    fp=float(w[(y==0)&(pred==1)].sum()); fn=float(w[(y==1)&(pred==0)].sum())
    precision=tp/(tp+fp) if tp+fp else float('nan')
    recall=tp/(tp+fn) if tp+fn else float('nan')
    specificity=tn/(tn+fp) if tn+fp else float('nan')
    f1=2*precision*recall/(precision+recall) if precision+recall else float('nan')
    bacc=(recall+specificity)/2 if np.isfinite(recall) and np.isfinite(specificity) else float('nan')
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=(tp*tn-fp*fn)/den if den else float('nan')
    out={
        'AUROC': float(roc_auc_score(y,score,sample_weight=sample_weight)),
        'AUPRC': float(average_precision_score(y,score,sample_weight=sample_weight)),
        'MCC': mcc, 'F1': f1, 'BalancedAccuracy': bacc,
        'Precision': precision,'Recall':recall,'Specificity':specificity,
        'TP':tp,'FP':fp,'TN':tn,'FN':fn,
    }
    return out


def freeze_threshold(y: np.ndarray, s: np.ndarray) -> Tuple[float, Dict[str,float]]:
    y=np.asarray(y,dtype=np.int64); s=np.asarray(s,dtype=np.float64)
    if len(y)==0 or not np.isfinite(s).all() or set(np.unique(y)) != {0,1}:
        die('Invalid validation arrays for threshold freeze')
    order=np.argsort(-s,kind='mergesort')
    yy=y[order]; ss=s[order]
    candidates=[np.nextafter(ss[0], np.inf)]
    i=0
    while i < len(ss):
        current=ss[i]
        j=i
        while j+1 < len(ss) and ss[j+1] == current:
            j += 1
        if j+1 < len(ss):
            candidates.append((current + ss[j+1]) / 2.0)
        else:
            candidates.append(np.nextafter(current, -np.inf))
        i=j+1
    best=None
    for t in candidates:
        m=binary_metrics(y,s,float(t))
        key=(m['MCC'],m['BalancedAccuracy'],m['F1'],-abs(float(t)-0.5),-float(t))
        if best is None or key > best[0]:
            best=(key,float(t),m)
    return best[1], best[2]


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p=np.asarray(pvals,float)
    m=len(p)
    order=np.argsort(p)
    adj=np.empty(m,float)
    running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def spearman(x,y) -> Tuple[float,float]:
    from scipy.stats import spearmanr
    r,p=spearmanr(np.asarray(x,float),np.asarray(y,float),nan_policy='omit')
    return float(r),float(p)


def pearson(x,y) -> Tuple[float,float]:
    from scipy.stats import pearsonr
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return float('nan'),float('nan')
    r,p=pearsonr(x[m],y[m])
    return float(r),float(p)


def run_cmd(cmd: Sequence[str], timeout: Optional[int]=None) -> subprocess.CompletedProcess:
    return subprocess.run(list(map(str,cmd)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)


def tool_version(name: str, args: Sequence[str]=('--version',)) -> str:
    exe=shutil.which(name)
    if not exe: return 'MISSING'
    try:
        cp=run_cmd([exe,*args],timeout=30)
        txt=(cp.stdout+'\n'+cp.stderr).strip().splitlines()
        return txt[0] if txt else f'exit={cp.returncode}'
    except Exception as e:
        return f'ERROR:{e}'


def parse_skani_output(text: str) -> Tuple[float,float,float]:
    """Return ANI, AF_ref, AF_query; NaN if unreported."""
    lines=[x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith('#')]
    for line in reversed(lines):
        parts=re.split(r'\t+|\s+',line)
        nums=[]
        for z in parts:
            try: nums.append(float(z))
            except Exception: pass
        # skani data line usually has ANI then AF_ref AF_query among numeric values.
        plausible=[v for v in nums if 0 <= v <= 100]
        if plausible:
            ani=plausible[0]
            af1=plausible[1] if len(plausible)>1 else float('nan')
            af2=plausible[2] if len(plausible)>2 else float('nan')
            # AF may be fractions, preserve values.
            return float(ani),float(af1),float(af2)
    return float('nan'),float('nan'),float('nan')


def run_skani_pair(query: Path, ref: Path) -> Tuple[float,float,float,str]:
    exe=shutil.which('skani')
    if not exe: return float('nan'),float('nan'),float('nan'),'MISSING'
    attempts=[[exe,'dist',str(query),str(ref)],[exe,'dist','-q',str(query),'-r',str(ref)]]
    last=''
    for cmd in attempts:
        cp=run_cmd(cmd,timeout=300)
        last=(cp.stdout+'\n'+cp.stderr)
        ani,af1,af2=parse_skani_output(cp.stdout)
        if cp.returncode==0 and np.isfinite(ani):
            return ani,af1,af2,'OK'
    return float('nan'),float('nan'),float('nan'),f'FAIL:{last[:300]}'


def run_fastani_pair(query: Path, ref: Path, tmpout: Path) -> Tuple[float,float,float,str]:
    exe=shutil.which('fastANI') or shutil.which('fastani')
    if not exe: return float('nan'),float('nan'),float('nan'),'MISSING'
    tmpout.parent.mkdir(parents=True,exist_ok=True)
    try:
        if tmpout.exists(): tmpout.unlink()
        cp=run_cmd([exe,'-q',str(query),'-r',str(ref),'-o',str(tmpout)],timeout=300)
        if cp.returncode==0 and tmpout.exists() and tmpout.stat().st_size:
            line=tmpout.read_text(errors='ignore').strip().splitlines()[0]
            parts=line.split('\t')
            if len(parts)>=5:
                ani=float(parts[2]); mapped=float(parts[3]); total=float(parts[4]); af=mapped/total if total else float('nan')
                return ani,af,af,'OK'
        return float('nan'),float('nan'),float('nan'),f'FAIL:{(cp.stdout+cp.stderr)[:300]}'
    finally:
        try: tmpout.unlink()
        except Exception: pass


def get_fasta_map(manifest: Optional[pd.DataFrame]=None) -> Dict[str,Path]:
    if manifest is None: manifest=load_manifest()
    mc=manifest_columns(manifest)
    out={}
    for _,r in manifest.iterrows():
        p=normalize_fasta_path(str(r[mc['fasta']]))
        out[str(r[mc['accession']])] = p
    return out


def environment_snapshot() -> Dict[str,str]:
    import platform
    return {
        'python': sys.version.replace('\n',' '),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'torch': tool_version('python', ['-c','import torch; print(torch.__version__)']),
        'skani': tool_version('skani'),
        'fastANI': tool_version('fastANI'),
    }
