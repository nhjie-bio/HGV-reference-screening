#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
# Demonstrate why bracket notation is mandatory: df.mode is DataFrame.mode method.
df=pd.DataFrame({'mode':['D0','D3'],'model_seed':[1,1]})
if not callable(df.mode):
    raise SystemExit('Unexpected pandas behavior: DataFrame.mode is not callable')
if df[df['mode']=='D3'].shape[0] != 1:
    raise SystemExit('Bracket-mode selection failed')
for name in ['10_summarize_e6_metrics_geometry_v1.py','11_e6_species_block_statistics_v1.py']:
    s=(root/name).read_text()
    forbidden=['pred2.mode', 'df.mode', 'seed_df.mode']
    bad=[x for x in forbidden if x in s]
    if bad:
        raise SystemExit(f'Forbidden pandas attribute column access in {name}: {bad}')
print('PANDAS_MODE_COLUMN_UNIT_TEST=PASS')
