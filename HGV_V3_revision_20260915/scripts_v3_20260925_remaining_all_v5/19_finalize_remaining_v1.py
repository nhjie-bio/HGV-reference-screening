#!/usr/bin/env python3
from pathlib import Path
import json
from common import *

OUT = BASE / f'20_rebuttal/HGV_V3_remaining_completion_freeze_v1_{DATE_TAG}'
PASS = 'HGV_V3_REMAINING_COMPLETION_PASS.txt'
if not ensure_new_output(OUT, PASS):
    raise SystemExit(0)

stages={
    'E6_mechanism': E6ROOT/f'07_mechanism_geometry_v1_{DATE_TAG}'/'E6_MECHANISM_GEOMETRY_PASS.txt',
    'E6_statistics': BASE/f'17_statistics/E6_species_block_bootstrap_v1_{DATE_TAG}'/'E6_SPECIES_BLOCK_STATS_PASS.txt',
    'E6_final': E6ROOT/f'08_final_freeze_v1_{DATE_TAG}'/'E6_FINAL_FREEZE_PASS.txt',
    'E11_screening': BASE/f'15_reference_screening/E11_reference_screening_v1_{DATE_TAG}'/'E11_REFERENCE_SCREENING_PASS.txt',
    'E12_scalability': BASE/f'16_scalability/E12_scalability_v1_{DATE_TAG}'/'E12_SCALABILITY_PASS.txt',
    'E7_sensitivity': BASE/f'11_sensitivity/E7_sensitivity_v1_{DATE_TAG}'/'E7_SENSITIVITY_PASS.txt',
    'E8_fragmentation': BASE/f'12_fragmentation/E8_fragmentation_v1_{DATE_TAG}'/'E8_FRAGMENTATION_PASS.txt',
    'E9_explainability': BASE/f'13_explainability/E9_explainability_v1_{DATE_TAG}'/'E9_EXPLAINABILITY_PASS.txt',
    'E10_biological_fidelity': BASE/f'14_biological_validation/E10_biological_fidelity_v1_{DATE_TAG}'/'E10_BIOLOGICAL_FIDELITY_PASS.txt',
}
rows=[]
for name,p in stages.items():
    ok=p.exists(); rows.append({'block':name,'status':'PASS' if ok else 'MISSING','marker':str(p),'sha256':sha256(p) if ok else ''})
    if not ok: die(f'Missing completion marker: {p}')
import pandas as pd
pd.DataFrame(rows).to_csv(OUT/'HGV_V3_REMAINING_BLOCK_STATUS.tsv',sep='\t',index=False)

# Source-of-truth index, without copying large pair-level files.
source={
    'final_manifest':str(FINAL_MANIFEST),'final_manifest_sha256':sha256(FINAL_MANIFEST),
    'canonical_k6_feature':str(find_feature_npy()),'canonical_k6_feature_sha256':sha256(find_feature_npy()),
    'pair_root':str(PAIR_ROOT),'teacher_dashing2':str(TEACHER_DASHING),'teacher_dashing2_sha256':sha256(TEACHER_DASHING),
    'main_d3_root':str(MAIN_D3_ROOT),'main_threshold_file':str(MAIN_THRESHOLD_FILE),
    'canonical_A_root':str(CANONICAL_A_ROOT),'E6_root':str(E6ROOT),'E6_locked_test_root':str(E6_TEST_ROOT),
    'remaining_blocks':{k:str(v.parent) for k,v in stages.items()},
}
atomic_json(OUT/'HGV_V3_SOURCE_OF_TRUTH_INDEX.json',source)

atomic_text(OUT/'HGV_V3_COMPLETION_REPORT.txt','\n'.join([
    'HGV-V3 REMAINING EXPERIMENT COMPLETION FREEZE v1',
    'All remaining executable evidence blocks requested in the 2026-09-19 plan are complete:',
    '- E6 mechanism + formal species-block statistics + final freeze',
    '- E11 reference screening',
    '- E12 scalability',
    '- E7 sensitivity',
    '- E8 fragmentation robustness',
    '- E9 explainability/confound audit',
    '- E10 ANI/AF/taxonomic fidelity',
    '',
    'No frozen E0-E5/main-model inputs were modified by this pipeline.',
    'Next step is interpretation/figures/tables/rebuttal/manuscript, not additional model retuning.',
])+'\n')
atomic_text(OUT/PASS,'STATUS=PASS\nALL_REMAINING_BLOCKS_COMPLETE=YES\nFROZEN_INPUTS_MODIFIED=NO\n')
write_sha256_manifest(OUT)
print('HGV_V3_REMAINING_COMPLETION=PASS')
print(OUT)
