"""OPS2 7.13: markdown tables (Korean report body) from the aggregate outputs of tools/a9_eval (values only; no new numbers).
Usage: report_tables.py RUN_DIR > draft.md
Prints, for each file present: arms and contrasts as 'point [ci_lo, ci_hi]' for MAE_high, MAE_low, MAE_all (bpm) and hit_high.
"""
import sys
from pathlib import Path

import pandas as pd

MET = [('MAE_high', 'MAE 고심박'), ('MAE_low', 'MAE 저심박'), ('MAE_all', 'MAE 전체'), ('hit_high', '적중률 고심박')]
BB = {'FactorizePhys (matched)': 'FactorizePhys'}


def fmt(r, m):
    if pd.isna(r.point):
        return '–'
    d = 3 if m == 'hit_high' else 1
    return f'{r.point:.{d}f} [{r.ci_lo:.{d}f}, {r.ci_hi:.{d}f}]'


def table(df, keys, head):
    out = ['| ' + ' | '.join(head + [h for _, h in MET]) + ' |', '|' + '---|' * (len(head) + len(MET))]
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        cells = [BB.get(str(x), str(x)) for x in k]
        for m, _ in MET:
            q = g[g.metric.eq(m)]
            cells.append(fmt(q.iloc[0], m) if len(q) else '–')
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def main(run):
    run = Path(run)
    spec = [('E2_P_arms.csv', ['backbone', 'arm', 'seeds'], ['백본', '팔', 'seed']),
            ('E2_P_contrasts.csv', ['backbone', 'name', 'contrast', 'seeds'], ['백본', '대비', '', 'seed']),
            ('E2_R_arms.csv', ['arm'], ['팔']),
            ('E2_R_contrasts.csv', ['name', 'contrast'], ['대비', '']),
            ('E2_R_interactions.csv', ['name', 'base_contrast'], ['대비', '']),
            ('E3_arms.csv', ['dataset', 'arm'], ['시험 세트', '팔']),
            ('E3_contrasts.csv', ['dataset', 'contrast'], ['시험 세트', '대비'])]
    for f, keys, head in spec:
        p = run / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        n = df.groupby(keys, sort=False)[['windows', 'subjects']].first().reset_index()
        print(f'### {f}\n')
        print(table(df, keys, head))
        print('\nwindows/subjects:', '; '.join(f"{' '.join(map(str, r[:len(keys)]))}: {r[-2]}/{r[-1]}" for r in n.itertuples(index=False)))
        print()


if __name__ == '__main__':
    main(sys.argv[1])
