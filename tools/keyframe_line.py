"""Keyframe-line check (CPU, saved window predictions only, no new inference).

Line: predicted HR within 151.171875 bpm +/- 1 bin of the wide-band (HAR) grid (3.515625 bpm), i.e.
147.65625-154.6875 bpm. Models: matched FactorizePhys, PhysNet, RhythmFormer x (0), (i), (ii), (ii-pc) and (i-pc) when
its windows exist. Windows: the report-4 populations and common support (b1/analyze.py load_mmpd / load_mcd:
MMPD-A validation windows; MCD-dev draft-rule windows, high = agreement & reference >= 100, low = reference < 100).
Per dataset x backbone x strategy x seed:
- line fraction by reference-HR bin (<100, 100-120, >=120) and by set (high, low);
- high / low MAE and signed error on all windows and with the line windows excluded.
Seed mean = mean over seeds 20261131-33 of the per-seed values (as the report-4 point estimates).
Writes outputs/keyframe/line_by_bin.csv, line_by_set.csv (per seed and seed mean) and prints a summary.
Usage: keyframe_line.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import analyze  # noqa: E402

GRID = 3.515625
CENTER = 43 * GRID          # 151.171875
LO, HI = CENTER - GRID, CENTER + GRID
ARMS = ['(0)', '(i)', '(ii)', '(ii-pc)', '(i-pc)', '(ii-inv)', '(ii-lag)']  # (ii-inv)/(ii-lag): brief 02 6.18/6.19
ARMS += ['(i-rfl)', '(i-pc-rfl)', '(ii-rfl)', '(ii-pc-rfl)']  # loss swap: brief 02 6.37/6.42
ARMS += ['(i-simple)', '(ii-pc-g)', '(ii-pc-g80)', '(ii-pc-g2)', '(i-pc-g80)', '(sham-pc)', '(roi-pc)', '(ii-g)', '(ii-g2)']  # brief 02 6.43
BACKBONES = ['FactorizePhys (matched)', 'PhysNet', 'RhythmFormer']


def main():
    out = WORK / os.environ.get('OPS2_KEYFRAME_OUT', 'outputs/keyframe')  # 6.42: loss-swap runs write to outputs/lossswap/keyframe
    out.mkdir(parents=True, exist_ok=True)
    meta = analyze.runs()
    mm = analyze.load_mmpd(meta)
    mc, _ = analyze.load_mcd(meta)
    d = pd.concat([mm, mc], ignore_index=True)
    d = d[d.layer.eq('HAR') & d.backbone.isin(BACKBONES) & d.strategy.isin(ARMS) & d.seed.isin(analyze.MAIN)].copy()
    d['line'] = (d.predicted_HR >= LO - 1e-6) & (d.predicted_HR <= HI + 1e-6)
    d['bin'] = np.select([d.reference_HR < 100, d.reference_HR < 120], ['<100', '100-120'], '>=120')
    d['err'] = d.predicted_HR - d.reference_HR
    key = ['dataset', 'backbone', 'strategy', 'seed']
    by_bin = (d.groupby(key + ['bin']).agg(windows=('line', 'size'), line_windows=('line', 'sum'), line_fraction=('line', 'mean'))
              .reset_index())
    rows = []
    for k, g in d.groupby(key + ['set']):
        keep = g[~g.line]
        rows.append(dict(zip(key + ['set'], k), windows=len(g), line_windows=int(g.line.sum()), line_fraction=g.line.mean(),
                         MAE_all=g.err.abs().mean(), signed_all=g.err.mean(),
                         MAE_excl_line=keep.err.abs().mean() if len(keep) else np.nan,
                         signed_excl_line=keep.err.mean() if len(keep) else np.nan, windows_excl_line=len(keep)))
    by_set = pd.DataFrame(rows)
    for name, t, grp in (('line_by_bin', by_bin, ['bin']), ('line_by_set', by_set, ['set'])):
        g = ['dataset', 'backbone', 'strategy'] + grp
        num = [c for c in t.columns if c not in g + ['seed']]
        mean = t.groupby(g)[num].mean().reset_index().assign(seed='mean of 3')
        n = t.groupby(g).seed.nunique().reset_index(name='seeds')
        mean = mean.merge(n, on=g)
        full = pd.concat([t.assign(seeds=1), mean], ignore_index=True).sort_values(g + ['seed'], key=lambda s: s.astype(str))
        full.to_csv(out / f'{name}.csv', index=False)
    pd.set_option('display.width', 250)
    s = by_set.groupby(['dataset', 'backbone', 'strategy', 'set'])[['line_fraction', 'MAE_all', 'MAE_excl_line', 'signed_all', 'signed_excl_line']].mean()
    print(s.round(3).to_string())
    b = by_bin.groupby(['dataset', 'backbone', 'strategy', 'bin']).line_fraction.mean().unstack('bin')
    print(b.round(3).to_string())


if __name__ == '__main__':
    main()
