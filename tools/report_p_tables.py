"""OPS2 7.04 / A6 items 3 and 8(d): report tables of the P analysis (tools/analysis_704b.py p), aggregates only. Markdown to
stdout: 8(d) replication table, C1-C4 high-HR and low-HR MAE differences on the new seeds and on all six seeds, arm values
(new seeds), verdict counts. Usage (main environment): report_p_tables.py P_ROOT   (P_ROOT = outputs/704/<root>/test/p)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(sys.argv[1])
SETS = {'MMPD': 'MMPD-B', 'MCD': 'MCD-test'}
BB = {'PhysNet': 'PhysNet', 'FactorizePhys (matched)': 'FactorizePhys', 'FactorizePhys': 'FactorizePhys', 'RhythmFormer': 'RhythmFormer'}
ORDER = ['PhysNet', 'FactorizePhys', 'RhythmFormer']
ARMS = ['(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-pc)']


def f(x, d=2):
    return '' if not np.isfinite(x) else f'{x:+.{d}f}'.replace('-', '−')


def ci(p, lo, hi):
    return f'{f(p)} [{f(lo)}, {f(hi)}]'


def bsort(df):
    df = df.copy()
    df['b'] = df.backbone.map(BB)
    return df[df.b.isin(ORDER)]


rep = bsort(pd.read_csv(P / 'replication_8d.csv'))
print('#### 8(d): 확증 대비의 새 시드 재현(고심박 MAE 차, 시드 20261148–50)\n')
print('| 시험 세트 | 백본 | 대비 | 시험 실행(시드 31–33) | 새 시드 | 판정 |\n|---|---|---|---|---|---|')
for ds in SETS:
    for b in ORDER:
        for r in rep[rep.dataset.eq(ds) & rep.b.eq(b)].sort_values('contrast').itertuples():
            print(f'| {SETS[ds]} | {b} | {r.contrast} | {ci(r.test_run_point, r.test_run_ci_lo, r.test_run_ci_hi)} | '
                  f'{ci(r.new_point, r.new_ci_lo, r.new_ci_hi)} | {r.verdict_8d} |')
print()
for metric, title in (('MAE_high', '고심박 MAE'), ('MAE_low', '저심박 MAE')):
    print(f'#### C1–C4, {title} 차 [95% 구간]: 새 시드(3) / 여섯 시드\n')
    print('| 시험 세트 | 백본 | 대비 | 새 시드 | 여섯 시드 |\n|---|---|---|---|---|')
    for ds in SETS:
        c = bsort(pd.read_csv(P / ds / 'contrasts.csv'))
        c = c[c.metric.eq(metric)]
        for b in ORDER:
            for name in ('C1', 'C2', 'C3', 'C4'):
                n = c[c.b.eq(b) & c.name.eq(name) & c.seeds.eq('new')]
                s = c[c.b.eq(b) & c.name.eq(name) & c.seeds.eq('six')]
                if not len(n):
                    continue
                n, s = n.iloc[0], s.iloc[0]
                print(f'| {SETS[ds]} | {b} | {name} | {ci(n.point, n.ci_lo, n.ci_hi)} | {ci(s.point, s.ci_lo, s.ci_hi)} |')
    print()
print('#### 팔별 값(새 시드): 고심박 MAE / 저심박 MAE / 고심박 적중률\n')
print('| 시험 세트 | 백본 | ' + ' | '.join(ARMS) + ' |\n|---|---|' + '---|' * len(ARMS))
for ds in SETS:
    a = bsort(pd.read_csv(P / ds / 'arms.csv'))
    a = a[a.seeds.eq('new')]
    for b in ORDER:
        x = a[a.b.eq(b)]
        if not len(x):
            continue
        cells = []
        for arm in ARMS:
            v = {m: x[x.arm.eq(arm) & x.metric.eq(m)].point for m in ('MAE_high', 'MAE_low', 'hit_high')}
            cells.append(' / '.join(f'{v[m].iloc[0]:.2f}' if len(v[m]) else '' for m in ('MAE_high', 'MAE_low', 'hit_high')))
        print(f'| {SETS[ds]} | {b} | ' + ' | '.join(cells) + ' |')
print()
print('#### 8(d) 판정 수\n')
print('| 백본 | replicated | not replicated | reversed | intervals only |\n|---|---|---|---|---|')
for b in ORDER + ['합계']:
    x = rep if b == '합계' else rep[rep.b.eq(b)]
    if not len(x):
        continue
    vc = x.verdict_8d.value_counts()
    print(f'| {b} | ' + ' | '.join(str(int(vc.get(k, 0))) for k in ('replicated', 'not replicated', 'reversed', 'intervals only')) + ' |')
