#!/usr/bin/env python3
import pandas as pd
from common import manifest_columns, infer_cohort

m=pd.DataFrame({
    'assembly_accession':['GCF_000000001.1','GCA_000000002.1','GCA_000000003.1'],
    'cohort':['A','B1','B2'],
    'subtype':['','','seen-genus'],
    'species':['s1','s1','s2'],
    'genus':['g1','g1','g1'],
    'family':['f1','f1','f1'],
    'fasta_path':['a.fa','b.fa','c.fa'],
})
mc=manifest_columns(m)
assert infer_cohort(m.iloc[0],mc)=='A'
meta=m.set_index(mc['accession'])
out={}
for acc,row in meta.iterrows():
    out[acc]=infer_cohort(row,mc,accession=acc)
assert out['GCF_000000001.1']=='A', out
assert out['GCA_000000002.1']=='B1', out
assert out['GCA_000000003.1']=='B2_seen_genus', out
print('INDEXED_COHORT_UNIT_TEST=PASS')
