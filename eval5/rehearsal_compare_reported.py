"""Rehearsal (brief 02 6.43 마): stage-3 results on the stored predictions against the reported development values
(outputs/5_full: c25.csv, cipc.csv contrasts; m123.csv arm metrics) for the confirmatory contrasts and arms.
Usage: rehearsal_compare_reported.py ANALYSIS_DIR DATASET(MMPD-A|MCD-dev) POPULATION OUT.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]


def main():
    a, ds, pop, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    rows = []
    new = pd.read_csv(a / 'confirmatory.csv')
    old = pd.concat([pd.read_csv(WORK / 'outputs/5_full' / f) for f in ('c25.csv', 'cipc.csv')], ignore_index=True)
    old = old[old.dataset.eq(ds) & old.population.eq(pop) & old.layer.eq('HAR') & old.seeds.astype(str).eq('3')]
    m = new.merge(old, on=['backbone', 'contrast', 'metric'], suffixes=('', '_rep'))
    for c in ('point', 'ci_lo', 'ci_hi'):
        m['d_' + c] = (m[c] - m[c + '_rep']).abs()
    rows.append(dict(table='confirmatory contrasts', cells=len(m) * 3, max_abs_difference=float(m[['d_point', 'd_ci_lo', 'd_ci_hi']].max().max()),
                     rows_new=len(new), rows_reported_matched=len(m)))
    arms = pd.read_csv(a / 'arms.csv')
    arms = arms[arms.layer.eq('HAR')]
    om = pd.read_csv(WORK / 'outputs/5_full/m123.csv')
    om = om[om.dataset.eq(ds) & om.population.eq(pop) & om.layer.eq('HAR') & om.seeds.astype(str).eq('3')]
    k = arms.merge(om, on=['backbone', 'strategy', 'metric'], suffixes=('', '_rep'))
    for c in ('point', 'ci_lo', 'ci_hi'):
        k['d_' + c] = (k[c] - k[c + '_rep']).abs()
    rows.append(dict(table='arm metrics (M1-M3)', cells=len(k) * 3, max_abs_difference=float(k[['d_point', 'd_ci_lo', 'd_ci_hi']].max().max()),
                     rows_new=len(arms), rows_reported_matched=len(k), windows_new=int(arms.windows.max()), windows_reported=int(k.windows_rep.max()) if 'windows_rep' in k else ''))
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
