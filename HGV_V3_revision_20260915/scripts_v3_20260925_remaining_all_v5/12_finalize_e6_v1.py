#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from common import *

MECH = E6ROOT / f'07_mechanism_geometry_v1_{DATE_TAG}'
STATS = BASE / f'17_statistics/E6_species_block_bootstrap_v1_{DATE_TAG}'
OUT = E6ROOT / f'08_final_freeze_v1_{DATE_TAG}'
PASS = 'E6_FINAL_FREEZE_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

required=[
    E6_TEST_FREEZE,
    MECH/'E6_MECHANISM_GEOMETRY_PASS.txt',
    STATS/'E6_SPECIES_BLOCK_STATS_PASS.txt',
    STATS/'E6_D3_PAIRED_CONTRASTS_HOLM.tsv',
]
for p in required:
    if not p.exists(): die(f'Missing prerequisite: {p}')

res=pd.read_csv(STATS/'E6_D3_PAIRED_CONTRASTS_HOLM.tsv',sep='\t')
rows=[]
for comp,g in res.groupby('comparison'):
    au=g[g.metric=='AUROC'].iloc[0]
    ap=g[g.metric=='AUPRC'].iloc[0]
    support=bool(au.direction_supported and ap.direction_supported and au.holm_significant and ap.holm_significant)
    rows.append({
        'comparison':comp,
        'AUROC_delta':au.observed_delta_mean_5seed,'AUROC_CI_low':au.ci95_low,'AUROC_CI_high':au.ci95_high,'AUROC_p_holm':au.p_holm,
        'AUPRC_delta':ap.observed_delta_mean_5seed,'AUPRC_CI_low':ap.ci95_low,'AUPRC_CI_high':ap.ci95_high,'AUPRC_p_holm':ap.p_holm,
        'supports_strict_improvement_claim':support,
    })
claim=pd.DataFrame(rows)
claim.to_csv(OUT/'E6_CLAIM_DECISION_TABLE.tsv',sep='\t',index=False)

all_support=bool(claim.supports_strict_improvement_claim.all())
# Important: no forced claim. Freeze evidence and state what wording is permitted.
if all_support:
    wording='E6 supports a statistically positive D3 effect against all pre-specified ablation comparators on both primary metrics under the frozen species-block bootstrap/Holm analysis.'
else:
    wording='E6 does not support an unconditional statement that D3 improves every ablation comparator on both primary metrics. Report the comparator-specific deltas/CIs/p-values and use geometry/mechanism wording where strict improvement is unsupported.'

report='\n'.join([
    'HGV-V3 E6 FINAL FREEZE v1',
    f'E6 locked test: {E6_TEST_ROOT}',
    f'Mechanism summary: {MECH}',
    f'Formal statistics: {STATS}',
    '',
    wording,
    '',
    'Immutable rules retained:',
    '- canonical G2 split_seed=20260915',
    '- D0-D5 5 model seeds',
    '- validation-derived thresholds only',
    '- no test/external retuning',
    '- historical E6 v1/v2 excluded from manuscript evidence',
])+'\n'
atomic_text(OUT/'E6_FINAL_REPORT.txt',report)

prov={
    'e6_test_freeze_sha256':sha256(E6_TEST_FREEZE),
    'e6_test_predictions_sha256':sha256(E6_PRED_LONG),
    'mechanism_pass_sha256':sha256(MECH/'E6_MECHANISM_GEOMETRY_PASS.txt'),
    'stats_pass_sha256':sha256(STATS/'E6_SPECIES_BLOCK_STATS_PASS.txt'),
    'claim_table_sha256':sha256(OUT/'E6_CLAIM_DECISION_TABLE.tsv'),
    'strict_improvement_all_comparators':all_support,
}
atomic_json(OUT/'E6_FINAL_PROVENANCE.json',prov)
atomic_text(OUT/PASS,'STATUS=PASS\nE6_FULL_EXPERIMENT=DONE\nPOST_TEST_RETUNING=NO\n')
write_sha256_manifest(OUT)
print('E6_FINAL_FREEZE=PASS')
print(wording)
