"""OPS2 7.04 report tables (aggregates only) from outputs/704/analysis/<PART>/ (tools/analysis_704.py). Markdown to stdout.
Usage (main environment): report_704_tables.py PART [ROOT]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

part = sys.argv[1]
root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / 'outputs' / '704' / 'analysis'
P = root / part
SETS = {'MMPD': 'MMPD-B' if part == 'test' else 'MMPD-A', 'MCD': 'MCD-test' if part == 'test' else 'MCD-dev'}
ORDER = ['PhysNet', 'PhysNet-Std', 'PhysNet-MCC', 'PhysNet-Std-MCC', 'FactorizePhys', 'FactorizePhys-MCC']
NAMES = ['C1', 'C2', 'C3', 'C4', 'ii-pc_vs_i-simple', 'ii-lag_vs_ii-pc']
LABEL = {'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'ii-pc_vs_i-simple': '(ii-pc)−(i-simple)', 'ii-lag_vs_ii-pc': '(ii-lag)−(ii-pc)'}


def f(x, d=2):
    return '' if not np.isfinite(x) else f'{x:+.{d}f}'.replace('-', '−')


def ci(r, d=2):
    return f"{f(r.point, d)} [{f(r.ci_lo, d)}, {f(r.ci_hi, d)}]"


def mark(r):
    return ' ‡' if r.ci_hi < 0 else (' †' if r.ci_lo > 0 else '')


def stored_conf(ds):
    tag = {'MMPD': 'automatic', 'MCD': 'halves'}[ds]
    base = Path(__file__).resolve().parents[1] / 'cache' / ('test' if part == 'test' else 'rehearsal/part2') / 'analysis' / ds / tag
    st = pd.concat([pd.read_csv(base / 'confirmatory.csv'), pd.read_csv(base / 'secondary.csv')], ignore_index=True)
    pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
    return st[st.analysis.eq('primary') & st.layer.eq('HAR') & st.population.eq(pop)]


for metric, title in (('MAE_high', '고심박 MAE'), ('MAE_low', '저심박 MAE')):
    for ds in ('MMPD', 'MCD'):
        c = pd.read_csv(P / ds / 'contrasts.csv')
        c = c[c.metric.eq(metric)]
        st = stored_conf(ds)
        st = st[st.metric.eq(metric)]
        print(f'\n#### {SETS[ds]}: 설정 안 대비, {title} 차 [95% 구간]\n')
        print('| 대비 | ' + ' | '.join(ORDER) + ' |')
        print('|---|' + '---|' * len(ORDER))
        for n in NAMES:
            cells = []
            for s in ORDER:
                r = c[c.setting.eq(s) & c.name.eq(n)]
                if s in ('PhysNet', 'FactorizePhys') and len(r):  # reported value: stored stage (c)
                    q = st[st.backbone.eq(r.iloc[0].backbone) & st.contrast.eq(r.iloc[0].contrast)]
                    r = q if len(q) else r
                cells.append(ci(r.iloc[0]) + mark(r.iloc[0]) if len(r) else '없음')
            print(f'| {LABEL[n]} | ' + ' | '.join(cells) + ' |')
        w = c[c.setting.isin(ORDER[1:4] + ORDER[5:]) & c.name.eq('C1')]
        print(f'\n창 {sorted(set(w.windows))}, 피험자 {sorted(set(w.subjects))} (새 설정 C1 기준)')
for metric, title in (('MAE_high', '고심박 MAE'),):
    for ds in ('MMPD', 'MCD'):
        it = pd.read_csv(P / ds / 'interactions.csv')
        it = it[it.metric.eq(metric)]
        print(f'\n#### {SETS[ds]}: 상호작용, {title} [95% 구간]\n')
        cols = [('label', 'PhysNet-Std', 'PhysNet'), ('loss', 'PhysNet-MCC', 'PhysNet'), ('loss', 'PhysNet-Std-MCC', 'PhysNet-Std'),
                ('loss', 'FactorizePhys-MCC', 'FactorizePhys')]
        print('| 대비 | ' + ' | '.join(f'{x} − {y}' for _, x, y in cols) + ' |')
        print('|---|' + '---|' * len(cols))
        for n in NAMES:
            cells = []
            for k, x, y in cols:
                r = it[it.setting.eq(x) & it.reference_setting.eq(y) & it.name.eq(n)]
                cells.append(ci(r.iloc[0]) + mark(r.iloc[0]) if len(r) else '없음')
            print(f'| {LABEL[n]} | ' + ' | '.join(cells) + ' |')
        print(f'\n창 {sorted(set(it.windows))}, 피험자 {sorted(set(it.subjects))}')
for ds in ('MMPD', 'MCD'):
    a = pd.read_csv(P / ds / 'arms.csv')
    print(f'\n#### {SETS[ds]}: 팔별 값(세 시드), 고심박 MAE / 저심박 MAE / 고심박 적중률\n')
    print('| 팔 | ' + ' | '.join(ORDER) + ' |')
    print('|---|' + '---|' * len(ORDER))
    for base in ('(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-lag)', '(ii-pc)'):
        cells = []
        for s in ORDER:
            g = a[a.setting.eq(s) & a.base_arm.eq(base)].set_index('metric')
            cells.append('없음' if not len(g) else f"{g.point['MAE_high']:.2f} / {g.point['MAE_low']:.2f} / {g.point['hit_high']:.2f}")
        print(f'| {base} | ' + ' | '.join(cells) + ' |')
if (P / 'MCD' / 'm7_contrasts.csv').exists():
    m = pd.read_csv(P / 'MCD' / 'm7_contrasts.csv')
    m = m[m.metric.eq('M7_CCC')]
    print(f'\n#### {SETS["MCD"]}: M7 CCC 대비 [95% 구간]\n')
    print('| 대비 | ' + ' | '.join(ORDER) + ' |')
    print('|---|' + '---|' * len(ORDER))
    for n in NAMES:
        cells = []
        for s in ORDER:
            r = m[m.setting.eq(s) & m.name.eq(n)]
            cells.append(ci(r.iloc[0], 3) + mark(r.iloc[0]) if len(r) else '없음')
        print(f'| {LABEL[n]} | ' + ' | '.join(cells) + ' |')
if (P / 'statements.csv').exists():
    s = pd.read_csv(P / 'statements.csv')
    print('\n#### 문장 (고심박 MAE 구간)\n')
    for r in s.itertuples():
        if r.item == '9':
            print(f'- 9항: {r.statement} (점추정 해석 참고: absorbed {r.reference_point_reading_absorbed})')
            continue
        extra = ''
        if r.item == '8(b)':
            extra = (f"; 음의 Pearson {r.pearson_setting} {f(r.pearson_point)} [{f(r.pearson_ci_lo)}, {f(r.pearson_ci_hi)}];"
                     f" 손실 상호작용 {f(r.loss_interaction_point)} [{f(r.loss_interaction_ci_lo)}, {f(r.loss_interaction_ci_hi)}]"
                     f"{'; 점추정 해석이면 ' + r.kind_if_point_reading if r.readings_differ else ''}")
        print(f'- {r.item} {r.test_set} {r.setting} {r.contrast}: {f(r.point)} [{f(r.ci_lo)}, {f(r.ci_hi)}]{extra} → "{r.statement}"')
