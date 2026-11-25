"""OPS2 7.07 (c2) report tables from cache/vipl707/stage_c2/VIPL/secondary_checkpoints.csv (aggregates; markdown to stdout).
Per group of secondary checkpoints (setting suffix and seed set; 'exploratory_arms' = the A4 exploratory arms with the
confirmatory seeds) and backbone: (ii-pc)-(i-simple) and (ii-lag)-(ii-pc) (PI 7.07h), high- and low-HR MAE, values only;
then the C1-C4 analogues of each group; then the groups and backbones where a requested contrast has no checkpoints.
Usage (main environment): report_c2_tables.py [CSV]
"""
import sys
from pathlib import Path  # 7.08

import numpy as np
import pandas as pd

p = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / 'cache' / 'vipl707' / 'stage_c2' / 'VIPL' / 'secondary_checkpoints.csv')
d = pd.read_csv(p)
d = d[d.layer.eq('HAR')] if 'layer' in d else d


def f(x):
    return '' if not np.isfinite(x) else f'{x:+.2f}'.replace('-', '−')


def ci(r):
    return f'{f(r.point)} [{f(r.ci_lo)}, {f(r.ci_hi)}]'


def base(c):  # '(ii-pc-mcc)-(i-simple-mcc)' -> 'ii-pc_vs_i-simple'
    a, b = c[1:-1].split(')-(')
    for x, y, n in (('ii-pc', 'i-simple', '(ii-pc)−(i-simple)'), ('ii-lag', 'ii-pc', '(ii-lag)−(ii-pc)')):
        if a.startswith(x) and b.startswith(y) and a[len(x):] == b[len(y):]:
            return n
    return None


d['requested'] = [base(c) for c in d.contrast]
req = d[d.requested.notna() & d.name.fillna('').eq('')]
print('#### (c2) 보조 체크포인트: 요청한 두 대비, 고심박 / 저심박 MAE 차 [95% 구간] (값만)\n')
print('| 묶음 | 백본 | 대비 | 창 | 피험자 | 고심박 MAE | 저심박 MAE |')
print('|---|---|---|---|---|---|---|')
for (an, bb, c), g in req.groupby(['analysis', 'backbone', 'contrast'], sort=True):
    h, lo = g[g.metric.eq('MAE_high')], g[g.metric.eq('MAE_low')]
    if not len(h):
        continue
    print(f'| {an} | {bb} | {c} | {int(h.iloc[0].windows)} | {int(h.iloc[0].subjects)} | {ci(h.iloc[0])} | {ci(lo.iloc[0]) if len(lo) else ""} |')
conf = d[d.name.fillna('').isin(['C1', 'C2', 'C3', 'C4']) & d.metric.eq('MAE_high')]
print('\n#### (c2) 묶음별 C1–C4 대응 대비, 고심박 MAE 차 [95% 구간] (값만)\n')
print('| 묶음 | 백본 | C1 | C2 | C3 | C4 |')
print('|---|---|---|---|---|---|')
for (an, bb), g in conf.groupby(['analysis', 'backbone'], sort=True):
    cells = []
    for n in ('C1', 'C2', 'C3', 'C4'):
        r = g[g.name.eq(n)]
        cells.append(ci(r.iloc[0]) if len(r) else '없음')
    print(f'| {an} | {bb} | ' + ' | '.join(cells) + ' |')
groups = sorted(set(zip(d.analysis, d.backbone)))
have = set(zip(req.analysis, req.backbone, req.requested))
print('\n#### 요청한 대비가 없는 묶음(체크포인트 없음)\n')
for an, bb in groups:
    miss = [n for n in ('(ii-pc)−(i-simple)', '(ii-lag)−(ii-pc)') if (an, bb, n) not in have]
    if miss:
        print(f'- {an}, {bb}: {", ".join(miss)}')
