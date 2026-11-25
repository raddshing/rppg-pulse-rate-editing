"""OPS2 7.04 / A6 item 9: report tables of the conditional RhythmFormer MCC analysis (tools/analysis_704b.py r), aggregates
only. Markdown to stdout: A6 8(b)/8(c) statements (test part), contrasts within setting (RhythmFormer = stored confirmatory
rows, RhythmFormer-MCC) with the loss interaction RhythmFormer-MCC - RhythmFormer, high-HR and low-HR MAE, arms, MCD-rPPG M7.
Usage (main environment): report_r_tables.py R_ROOT PART   (R_ROOT = outputs/704/analysis_b/<PART>/r; PART test|dev)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, part = Path(sys.argv[1]), sys.argv[2]
SETS = {'MMPD': 'MMPD-B' if part == 'test' else 'MMPD-A', 'MCD': 'MCD-test' if part == 'test' else 'MCD-dev'}
NAMES = ['C1', 'C2', 'C3', 'C4', 'ii-pc_vs_i-simple', 'ii-lag_vs_ii-pc']
LABEL = {'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'ii-pc_vs_i-simple': '(ii-pc)−(i-simple)', 'ii-lag_vs_ii-pc': '(ii-lag)−(ii-pc)'}
ARMS = ['(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-lag)', '(ii-pc)']


def f(x, d=2):
    return '' if not np.isfinite(x) else f'{x:+.{d}f}'.replace('-', '−')


def ci(r, d=2):
    return f'{f(r.point, d)} [{f(r.ci_lo, d)}, {f(r.ci_hi, d)}]'


def one(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k].eq(v)]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


if part == 'test':
    st = pd.read_csv(R / 'statements.csv')
    print('#### A6 8(b)·8(c) 문장(RhythmFormer-MCC, 고심박 MAE 차)\n')
    print('| 항 | 시험 세트 | 대비 | MCC 대비 | 음의 Pearson 대비 | 손실 상호작용 | 문장 |\n|---|---|---|---|---|---|---|')
    for r in st.itertuples():
        pe = f'{f(r.pearson_point)} [{f(r.pearson_ci_lo)}, {f(r.pearson_ci_hi)}]' if np.isfinite(r.pearson_point) else ''
        li = f'{f(r.loss_interaction_point)} [{f(r.loss_interaction_ci_lo)}, {f(r.loss_interaction_ci_hi)}]' \
            if np.isfinite(r.loss_interaction_point) else ''
        print(f'| {r.item} | {SETS[r.dataset]} | {str(r.contrast).replace(")-(", ")−(")} | {f(r.point)} [{f(r.ci_lo)}, {f(r.ci_hi)}] | {pe} | {li} | {r.statement} |')
    print()

for metric, title in (('MAE_high', '고심박 MAE'), ('MAE_low', '저심박 MAE')):
    for ds in SETS:
        c = pd.read_csv(R / ds / 'contrasts.csv')
        it = pd.read_csv(R / ds / 'interactions.csv')
        print(f'#### {SETS[ds]}: 설정 안 대비와 손실 상호작용, {title} 차 [95% 구간]\n')
        print('| 대비 | RhythmFormer | RhythmFormer-MCC | RhythmFormer-MCC − RhythmFormer |\n|---|---|---|---|')
        for n in NAMES:
            a = one(c, setting='RhythmFormer', name=n, metric=metric)
            b = one(c, setting='RhythmFormer-MCC', name=n, metric=metric)
            i = one(it, setting='RhythmFormer-MCC', name=n, metric=metric)
            print(f'| {LABEL[n]} | {ci(a)} | {ci(b)} | {ci(i)} |')
        w = one(c, setting='RhythmFormer-MCC', name='C1', metric=metric)
        wi = sorted(set(it[it.metric.eq(metric)].windows)), sorted(set(it[it.metric.eq(metric)].subjects))
        print(f'\n창 {int(w.windows)}, 피험자 {int(w.subjects)} (RhythmFormer-MCC C1); 상호작용 창 {wi[0]}, 피험자 {wi[1]}\n')

print('#### 팔별 값: 고심박 MAE / 저심박 MAE / 고심박 적중률\n')
print('| 세트 | 설정 | ' + ' | '.join(ARMS) + ' |\n|---|---|' + '---|' * len(ARMS))
for ds in SETS:
    a = pd.read_csv(R / ds / 'arms.csv')
    for s in ('RhythmFormer', 'RhythmFormer-MCC'):
        cells = []
        for arm in ARMS:
            arm_s = arm if s == 'RhythmFormer' else arm[:-1] + '-mcc)'
            v = [a[a.setting.eq(s) & a.arm.eq(arm_s) & a.metric.eq(m)].point for m in ('MAE_high', 'MAE_low', 'hit_high')]
            cells.append(' / '.join(f'{x.iloc[0]:.2f}' if len(x) else '' for x in v))
        print(f'| {SETS[ds]} | {s} | ' + ' | '.join(cells) + ' |')
print()

m7 = R / 'MCD' / 'm7_contrasts.csv'
if m7.exists():
    c = pd.read_csv(m7)
    it = pd.read_csv(R / 'MCD' / 'm7_interactions.csv')
    print(f'#### {SETS["MCD"]}: M7(피험자 단위 MAE) 대비와 손실 상호작용 [95% 구간]\n')
    print('| 대비 | RhythmFormer | RhythmFormer-MCC | RhythmFormer-MCC − RhythmFormer |\n|---|---|---|---|')
    for n in NAMES:
        a = one(c, setting='RhythmFormer', name=n, metric='M7_MAE')
        b = one(c, setting='RhythmFormer-MCC', name=n, metric='M7_MAE')
        i = one(it, setting='RhythmFormer-MCC', name=n, metric='M7_MAE')
        print(f'| {LABEL[n]} | {ci(a)} | {ci(b)} | {ci(i)} |')
    print(f'\n피험자 {int(one(c, setting="RhythmFormer-MCC", name="C1", metric="M7_MAE").subjects)}\n')
