"""OPS2 7.13 2 (A9 item 3): the values from the stored October (0B) predictions (posthoc 7.01 ra2, made without the guards)
beside the E3 values of the guarded inference.

Why a separate step: the join inside a9_e2e3.py analyze looked up the October tables with the dataset names of the new run
('MMPD-B', 'MCD-rPPG test'), while ra2 names them 'MMPD' and 'MCD'; it matched no row and wrote no table. This script reads
only the finished aggregate outputs (E3_arms.csv, E3_contrasts.csv) and outputs/posthoc701/ra2 (ra2_arms.csv,
ra2_contrasts.csv); no prediction is read. Populations agree: MMPD-B Q*, MCD-rPPG test draft rule, layer HAR (both).
(0)-(0B) is the negated October (0B)-(0) (interval negated and swapped). VIPL-HR has no October values.
Writes <run>/E3_october_beside.csv (contrasts) and <run>/E3_october_beside_arms.csv; refuses to overwrite.
"""
import sys
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parents[2]
RA2 = W / 'outputs' / 'posthoc701' / 'ra2'
DS = {'MMPD-B': ('MMPD', 'Qstar'), 'MCD-rPPG test': ('MCD', 'draft_rule')}


def main(run=W / 'outputs' / 'a9_eval' / 'run', ra2=RA2):
    run, ra2 = Path(run), Path(ra2)
    outs = [run / 'E3_october_beside.csv', run / 'E3_october_beside_arms.csv']
    for o in outs:
        if o.exists():
            sys.exit(f'{o.name} exists')
    con = pd.read_csv(ra2 / 'ra2_contrasts.csv')
    arms = pd.read_csv(ra2 / 'ra2_arms.csv')
    con = con[con.layer.eq('HAR') & con.backbone.eq('RhythmFormer')]
    arms = arms[arms.layer.eq('HAR') & arms.backbone.eq('RhythmFormer')]
    new_c = pd.read_csv(run / 'E3_contrasts.csv')
    new_a = pd.read_csv(run / 'E3_arms.csv')
    rows = []
    for r in new_c.itertuples():
        if r.dataset not in DS:
            continue
        ds, pop = DS[r.dataset]
        q = con[con.dataset.eq(ds) & con.population.eq(pop) & con.metric.eq(r.metric)]
        if r.contrast == '(0)-(0B)':
            o, sign = q[q.contrast.eq('(0B)-(0)')], -1
        else:
            o, sign = q[q.contrast.eq(r.contrast)], 1
        if len(o) != 1:
            rows.append(dict(dataset=r.dataset, contrast=r.contrast, metric=r.metric, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             windows=r.windows, october_point=None, october_ci_lo=None, october_ci_hi=None, october_windows=None,
                             october_note='no October value' if len(o) == 0 else f'{len(o)} October rows'))
            continue
        o = o.iloc[0]
        lo, hi = (o.ci_lo, o.ci_hi) if sign == 1 else (-o.ci_hi, -o.ci_lo)
        rows.append(dict(dataset=r.dataset, contrast=r.contrast, metric=r.metric, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                         windows=r.windows, october_point=sign * o.point, october_ci_lo=lo, october_ci_hi=hi, october_windows=o.windows,
                         october_note='October (0B)-(0) negated' if sign == -1 else ''))
    arows = []
    for r in new_a.itertuples():
        if r.dataset not in DS:
            continue
        ds, pop = DS[r.dataset]
        o = arms[arms.dataset.eq(ds) & arms.population.eq(pop) & arms.strategy.eq('(0B)') & arms.metric.eq(r.metric)]
        arows.append(dict(dataset=r.dataset, arm='(0B)', metric=r.metric, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi, windows=r.windows,
                          october_point=o.point.iloc[0] if len(o) == 1 else None,
                          october_ci_lo=o.ci_lo.iloc[0] if len(o) == 1 else None,
                          october_ci_hi=o.ci_hi.iloc[0] if len(o) == 1 else None,
                          october_windows=o.windows.iloc[0] if len(o) == 1 else None))
    pd.DataFrame(rows).to_csv(outs[0], index=False)
    pd.DataFrame(arows).to_csv(outs[1], index=False)
    print('october beside contrasts', len(rows), 'with October value', sum(x['october_point'] is not None for x in rows),
          'arms', len(arows), 'with October value', sum(x['october_point'] is not None for x in arows), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
