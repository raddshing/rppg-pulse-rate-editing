"""OPS2 7.20 4: tables of the B1 report (A10) from the aggregate outputs of tools/a9_eval/b1_groups.py, and the report bundle.

tables [ROOT]: prints Markdown tables (Korean headers) from ROOT (default outputs/720/eval): windows and subjects per test set;
  arm values (high-HR, low-HR and all-window MAE, high-HR hit rate, signed errors) per backbone; the contrasts of A10 item 7
  (high-HR MAE Δ [95% CI] with the item 8 description; high-HR hit rate and all-window MAE Δ); the recovery ratio cells;
  the summary sentences; the repeat-pass identity (outputs/720/infer/determinism_*.csv). Writes ROOT/report/checks.csv.
bundle MD: zips the report MD, README.txt and the aggregate CSVs (no run_info.json) into reports/OPS2_720_B1.zip.
"""
import csv
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parents[2]
GROUPS = ('B1_PhysNet', 'B1_RhythmFormer', 'B1_FPm')
SETS = ('MMPD-B', 'MCD-rPPG test', 'VIPL-HR')
SHORT_SET = {'MMPD-B': 'MMPD-B', 'MCD-rPPG test': 'MCD-test', 'VIPL-HR': 'VIPL-HR'}
BB = {'B1_PhysNet': 'PhysNet', 'B1_FPm': 'FactorizePhys (matched)', 'B1_RhythmFormer': 'RhythmFormer'}
SHORT_BB = {'PhysNet': 'PhysNet', 'FactorizePhys (matched)': 'FactorizePhys', 'RhythmFormer': 'RhythmFormer'}
ARM_LABEL = {'(0-b1low)': '(0) U-low', '(i-simple-b1low)': '(i-simple)', '(i-b1low)': '(i)', '(i-pc-b1low)': '(i-pc)',
             '(0-b1fullm)': '(0) U-full-m'}
DESC_KO = {'not established': '확립되지 않음'}


def ci(p, lo, hi, d=1):
    f = f'{{:.{d}f}}'
    return f'{f.format(p)} [{f.format(lo)}, {f.format(hi)}]'


def load(root):
    a = pd.concat([pd.read_csv(root / g / 'run' / 'arms.csv') for g in GROUPS], ignore_index=True)
    c = pd.concat([pd.read_csv(root / g / 'run' / 'contrasts.csv') for g in GROUPS], ignore_index=True)
    d = pd.concat([pd.read_csv(root / g / 'run' / 'descriptions.csv') for g in GROUPS], ignore_index=True)
    r = pd.concat([pd.read_csv(root / g / 'run' / 'ratio.csv') for g in GROUPS], ignore_index=True)
    s = pd.read_csv(root / 'summary' / 'summary_sentences.csv')
    return a, c, d, r, s


def tables(root):
    a, c, d, r, s = load(root)
    out = []
    checks = []
    # windows per set (from the (0-b1low) arm rows of MAE_high)
    out.append('**창과 피험자 (시험 세트별, 세 백본 같은 창인지 확인)**\n')
    out.append('| 시험 세트 | 창 | 피험자 | 고심박 창 (ratio 표) |')
    out.append('|---|---|---|---|')
    for ds in SETS:
        q = a[a.dataset.eq(ds) & a.metric.eq('MAE_all')]
        rr = r[r.dataset.eq(ds)]
        w, sj = sorted(set(q.windows)), sorted(set(q.subjects))
        checks.append(dict(check=f'{ds}: windows equal over the 15 arm-backbone cells', value=';'.join(map(str, w)), passed=len(w) == 1))
        out.append(f'| {ds} | {"/".join(map(str, w))} | {"/".join(map(str, sj))} | {"/".join(map(str, sorted(set(rr.windows_high))))} |')
    out.append('')
    # arm values
    for g in GROUPS:
        bb = BB[g]
        out.append(f'**팔 값: {SHORT_BB[bb]}** (MAE bpm; 적중률은 ±3.52 bpm 안의 비율; 부호 오차 = 예측 − 기준)\n')
        out.append('| 시험 세트 | 팔 | MAE 고심박 | MAE 저심박 | MAE 전체 | 고심박 적중률 | 부호 오차 고심박 | 부호 오차 저심박 | 부호 오차 전체 |')
        out.append('|---|---|---|---|---|---|---|---|---|')
        for ds in SETS:
            for arm in ARM_LABEL:
                q = a[a.backbone.eq(bb) & a.dataset.eq(ds) & a.arm.eq(arm)].set_index('metric')
                if not len(q):
                    continue
                cell = lambda m, dd=1: ci(q.loc[m, 'point'], q.loc[m, 'ci_lo'], q.loc[m, 'ci_hi'], dd)  # noqa: E731
                out.append(f'| {SHORT_SET[ds]} | {ARM_LABEL[arm]} | {cell("MAE_high")} | {cell("MAE_low")} | {cell("MAE_all")} | '
                           f'{cell("hit_high", 3)} | {cell("signed_error_high")} | {cell("signed_error_low")} | {cell("signed_error_all")} |')
        out.append('')
    # contrasts: high-HR MAE with description
    out.append('**대비: MAE 고심박 Δ [95% 구간]과 A10 8항 서술**\n')
    out.append('| 시험 세트 | 대비 | ' + ' | '.join(SHORT_BB[BB[g]] for g in GROUPS) + ' |')
    out.append('|---|---|' + '---|' * len(GROUPS))
    names = list(dict.fromkeys(d.name))
    for ds in SETS:
        for nm in names:
            cells = []
            for g in GROUPS:
                q = d[d.dataset.eq(ds) & d.name.eq(nm) & d.backbone.eq(BB[g])]
                if len(q):
                    q = q.iloc[0]
                    cells.append(f'{ci(q.point, q.ci_lo, q.ci_hi)}<br>{DESC_KO.get(q.description, q.description)}')
                else:
                    cells.append('–')
            out.append(f'| {SHORT_SET[ds]} | {nm} | ' + ' | '.join(cells) + ' |')
    out.append('')
    for m, label, dd in (('hit_high', '고심박 적중률 Δ', 3), ('MAE_low', 'MAE 저심박 Δ', 1), ('MAE_all', 'MAE 전체 Δ', 1)):
        out.append(f'**대비: {label}**\n')
        out.append('| 시험 세트 | 대비 | ' + ' | '.join(SHORT_BB[BB[g]] for g in GROUPS) + ' |')
        out.append('|---|---|' + '---|' * len(GROUPS))
        for ds in SETS:
            for nm in names:
                cells = []
                for g in GROUPS:
                    q = c[c.dataset.eq(ds) & c.name.eq(nm) & c.backbone.eq(BB[g]) & c.metric.eq(m)]
                    cells.append(ci(q.iloc[0].point, q.iloc[0].ci_lo, q.iloc[0].ci_hi, dd) if len(q) else '–')
                out.append(f'| {SHORT_SET[ds]} | {nm} | ' + ' | '.join(cells) + ' |')
        out.append('')
    # ratio
    out.append('**회복 비율 [(i-simple) − (0)] / [(0, U-full-m) − (0, U-low)] (고심박 MAE, 복제마다 계산)**\n')
    out.append('| 시험 세트 | 백본 | 분자 (i-simple)−(0) | 분모 (0 full)−(0 low) | 분모 구간 0 아래 | 비율 [95% 백분위 구간] | 분자·분모 = 대비 |')
    out.append('|---|---|---|---|---|---|---|')
    pv = pd.concat([pd.read_csv(root / g / 'run' / 'ratio_parts_vs_contrasts.csv') for g in GROUPS], ignore_index=True) \
        if all((root / g / 'run' / 'ratio_parts_vs_contrasts.csv').exists() for g in GROUPS) else None
    for ds in SETS:
        for g in GROUPS:
            q = r[r.dataset.eq(ds) & r.backbone.eq(BB[g])]
            if not len(q):
                continue
            q = q.iloc[0]
            rat = ci(q.ratio_point, q.ratio_ci_lo, q.ratio_ci_hi, 2) if bool(q.denominator_interval_below_0) else '보고하지 않음'
            same = '–'
            if pv is not None:
                pp = pv[pv.dataset.eq(ds)]
                same = '예' if len(pp) and pp.equal.all() else '아니오'
            out.append(f'| {SHORT_SET[ds]} | {SHORT_BB[q.backbone]} | {ci(q.num_point, q.num_ci_lo, q.num_ci_hi)} | '
                       f'{ci(q.den_point, q.den_ci_lo, q.den_ci_hi)} | {"예" if q.denominator_interval_below_0 else "아니오"} | {rat} | {same} |')
            checks.append(dict(check=f'{ds} {q.backbone}: ratio numerator/denominator equal to boot.summarize', value=q.check_equal_to_boot_summarize,
                               passed=bool(q.check_equal_to_boot_summarize)))
    if pv is not None:
        checks.append(dict(check='ratio numerator/denominator equal to the separate contrasts (all cells)', value=int(pv.equal.sum()),
                           passed=bool(pv.equal.all())))
    out.append('')
    out.append('**요약 문장 (A10 8e)**\n')
    for t in s.itertuples():
        out.append(f'- "{t.sentence}"')
    out.append('')
    # repeat identity
    out.append('**반복 추론 (백본마다 (i-pc) seed 20261131 한 번 더)**\n')
    out.append('| 백본 | MMPD-B | MCD-test | VIPL-HR |')
    out.append('|---|---|---|---|')
    for g in GROUPS:
        cells = []
        for n in ('MMPD_B', 'MCD_test', 'VIPL'):
            f = W / 'outputs' / '720' / 'infer' / f'determinism_{g}_{n}.csv'
            if f.exists():
                q = pd.read_csv(f)
                cells.append(f'{int(q.identical.sum())}/{int(q.files.sum())}')
                checks.append(dict(check=f'{g} {n}: repeat files identical', value=f'{int(q.identical.sum())}/{int(q.files.sum())}',
                                   passed=bool((q.identical == q.files).all())))
            else:
                cells.append('–')
        out.append(f'| {SHORT_BB[BB[g]]} | ' + ' | '.join(cells) + ' |')
    out.append('')
    rep = root / 'report'
    rep.mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(rep / 'checks.csv', index=False)
    print('\n'.join(out))


README = """OPS2 7.20 B1 (A10) report bundle. Aggregate values only: no window-level predictions, recording lists or server paths.
- B1_arms.csv: arm values per backbone and test set (high-HR, low-HR and all-window MAE, signed errors, high-HR hit rate and
  the other metrics of the test run), with 95% intervals (crossed subject x seed bootstrap, B = 10,000, seed 20261110,
  seeds 20261131-20261133 paired by index, also between U-low and U-full-m runs).
- B1_contrasts.csv: the contrasts of A10 item 7 (all metrics).
- B1_descriptions.csv: high-HR MAE contrasts with the fixed descriptions of A10 item 8 (a-d).
- B1_ratio.csv: the recovery ratio of A10 item 7d (numerator, denominator, their intervals; the ratio and its percentile
  interval only where the denominator interval lies below 0) and the check against the bootstrap function.
- B1_ratio_parts_vs_contrasts.csv: numerator and denominator against the separately computed contrasts.
- B1_summary_sentences.csv: the summary sentences of A10 item 8e.
- B1_determinism.csv: repeat inference of one checkpoint per backbone, files identical per key.
- checks.csv: checks of the report tables.
Test sets: MMPD-B (Q*), MCD-rPPG test half (draft rule), VIPL-HR primary set (HAR).
"""


def bundle(md):
    root = W / 'outputs' / '720' / 'eval'
    zp = W / 'reports' / 'OPS2_720_B1.zip'
    assert not zp.exists(), f'{zp} exists; move it aside first'
    frames = {}
    for name in ('arms', 'contrasts', 'descriptions', 'ratio', 'ratio_parts_vs_contrasts'):
        frames[f'B1_{name}.csv'] = pd.concat([pd.read_csv(root / g / 'run' / f'{name}.csv') for g in GROUPS], ignore_index=True)
    frames['B1_summary_sentences.csv'] = pd.read_csv(root / 'summary' / 'summary_sentences.csv')
    det = []
    for g in GROUPS:
        for n in ('MMPD_B', 'MCD_test', 'VIPL'):
            q = pd.read_csv(W / 'outputs' / '720' / 'infer' / f'determinism_{g}_{n}.csv')
            q.insert(0, 'test_set', n)
            q.insert(0, 'group', g)
            det.append(q)
    frames['B1_determinism.csv'] = pd.concat(det, ignore_index=True)
    frames['checks.csv'] = pd.read_csv(root / 'report' / 'checks.csv')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, f in frames.items():
            z.writestr(n, f.to_csv(index=False))
        z.writestr('README.txt', README)
        z.write(md, Path(md).name)
    text = ''.join(zipfile.ZipFile(zp).read(n).decode('utf-8', 'replace') for n in zipfile.ZipFile(zp).namelist())
    # 'subjects' (counts) and 'subject x seed' are words of the tables; a recording ID has a number after 'subject'
    assert '/' + 'mnt/' not in text and '/' + 'home/' not in text and not re.search(r'subject\s*\d', text), \
        'server path or recording ID in the bundle'
    print(zp, zp.stat().st_size, 'bytes', sorted(zipfile.ZipFile(zp).namelist()))


if __name__ == '__main__':
    if sys.argv[1] == 'tables':
        tables(Path(sys.argv[2]) if len(sys.argv) > 2 else W / 'outputs' / '720' / 'eval')
    elif sys.argv[1] == 'bundle':
        bundle(sys.argv[2])
    else:
        sys.exit(__doc__)
