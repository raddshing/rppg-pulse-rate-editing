"""Common-window paired comparison for the keyframe line (brief 02 6.18; CPU, saved window predictions, no inference).

Line: predicted HR within 151.171875 +/- 1 HAR bin (147.65625-154.6875 bpm). Windows: report-4 populations and
common support (b1/analyze.py), wide band (HAR). For a pair a - b of the same backbone and every seed, the windows on
which a or b (same seed) is on the line are removed from both; on the remaining common windows the high/low MAE and
signed-error differences are computed (same-seed pairs, subject x seed crossed bootstrap, b1/boot.py, B = 10,000;
point = mean over seeds). The line-fraction difference a - b is computed on all windows of the pair (same windows,
same seed), by set (high, low, all) and reference-HR bin (<100, 100-120, >=120), with the same bootstrap.
Pairs: RhythmFormer (ii-pc) - (ii), (i), (0); added when the runs exist: (ii-pc) - (i-pc), (i-pc) - (i), (i-pc) - (0),
(ii-inv) - (ii-pc), (ii-inv) - (ii), (ii-inv) - (0). Other backbones with --backbones.
Writes outputs/keyframe/common_windows.csv (contrasts), common_windows_per_seed.csv (per-seed values, remaining
window counts) and line_diff.csv.
Usage: keyframe_common.py [--backbones RhythmFormer,...] [--b N]
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import analyze  # noqa: E402
import boot  # noqa: E402

GRID = 3.515625
LO, HI = 42 * GRID, 44 * GRID   # 147.65625, 154.6875 (151.171875 +/- 1 bin)
PAIRS = [('(ii-pc)', '(ii)'), ('(ii-pc)', '(i)'), ('(ii-pc)', '(0)'),
         ('(ii-pc)', '(i-pc)'), ('(i-pc)', '(i)'), ('(i-pc)', '(0)'),
         ('(ii-inv)', '(ii-pc)'), ('(ii-inv)', '(ii)'), ('(ii-inv)', '(0)'),
         ('(ii-lag)', '(ii-pc)'), ('(ii-lag)', '(ii)'), ('(ii-lag)', '(0)'), ('(ii-inv)', '(ii-lag)')]
PAIRS += [('(ii-pc-rfl)', '(ii-rfl)'), ('(i-pc-rfl)', '(i-rfl)'), ('(ii-pc-rfl)', '(i-pc-rfl)'), ('(ii-pc-rfl)', '(0)'),  # 6.37
          ('(i-pc)', '(i)'), ('(ii-pc)', '(i-pc)'),
          ('(i-rfl)', '(i)'), ('(i-pc-rfl)', '(i-pc)'), ('(ii-rfl)', '(ii)'), ('(ii-pc-rfl)', '(ii-pc)')]  # 6.42 loss effect
PAIRS += [('(i-simple)', '(i-pc)'), ('(i-simple)', '(i)'), ('(ii-pc)', '(i-simple)'), ('(i-simple)', '(0)'),  # 6.43 da
          ('(ii-pc-g80)', '(ii-pc)'), ('(ii-pc-g2)', '(ii-pc-g)'), ('(ii-pc-g)', '(ii-pc)'), ('(i-pc-g80)', '(i-pc)'),
          ('(sham-pc)', '(0)'), ('(ii-pc)', '(sham-pc)'), ('(roi-pc)', '(ii-pc)'), ('(roi-pc)', '(0)')]  # 6.43 ra
NAMES = {'RhythmFormer': 'RhythmFormer', 'FPm': 'FactorizePhys (matched)', 'PhysNet': 'PhysNet'}
KEEP = ['MAE_high', 'MAE_low', 'signed_error_high', 'signed_error_low']


def on_line(p):
    return (p >= LO - 1e-6) & (p <= HI + 1e-6)


def line_fn(W, r, p, s):
    L = on_line(p).astype(float)
    masks = {'all': np.ones(len(p), bool), 'high': s == 'high', 'low': s == 'low',
             'ref_lt100': r < 100, 'ref_100_120': (r >= 100) & (r < 120), 'ref_ge120': r >= 120}
    return {f'line_{k}': boot._wmean(W[:, m], L[m]) if m.any() else np.full(W.shape[0], np.nan) for k, m in masks.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='RhythmFormer')
    ap.add_argument('--b', type=int, default=boot.B)
    a = ap.parse_args()
    out = WORK / os.environ.get('OPS2_KEYFRAME_OUT', 'outputs/keyframe')  # 6.42: loss-swap runs write to outputs/lossswap/keyframe
    out.mkdir(parents=True, exist_ok=True)
    meta = analyze.runs()
    mm = analyze.load_mmpd(meta)
    mc, _ = analyze.load_mcd(meta)
    d = pd.concat([mm, mc], ignore_index=True)
    d = d[d.layer.eq('HAR') & d.seed.isin(analyze.MAIN)]
    rows, seed_rows, line_rows = [], [], []
    for bkey in a.backbones.split(','):
        bb = NAMES[bkey]
        for ds, dd in d[d.backbone.eq(bb)].groupby('dataset'):
            have = set(dd.strategy)
            for x, y in PAIRS:
                if x not in have or y not in have:
                    continue
                ga, gb = dd[dd.strategy.eq(x)], dd[dd.strategy.eq(y)]
                seeds = sorted(set(ga.seed) & set(gb.seed))
                if len(seeds) < 3:
                    continue
                k = ['window_id', 'seed']
                j = ga[k + ['predicted_HR']].merge(gb[k + ['predicted_HR']], on=k, suffixes=('_a', '_b'))
                assert len(j) == len(ga) == len(gb), (bb, ds, x, y)  # same windows for both runs of every seed
                drop = j[on_line(j.predicted_HR_a) | on_line(j.predicted_HR_b)][k]
                keyset = set(map(tuple, drop.to_numpy()))
                keep_a = ga[[t not in keyset for t in zip(ga.window_id, ga.seed)]]
                keep_b = gb[[t not in keyset for t in zip(gb.window_id, gb.seed)]]
                pt, ci = boot.contrast_ci(keep_a, keep_b, seeds, boot.m123, a.b)
                base = dict(dataset=ds, backbone=bb, contrast=f'{x}-{y}', seeds=len(seeds))
                for m in KEEP:
                    rows.append(dict(base, analysis='common windows without line', metric=m, point=pt[m],
                                     ci_lo=ci[m][0] if ci else np.nan, ci_hi=ci[m][1] if ci else np.nan))
                ptl, cil = boot.contrast_ci(ga, gb, seeds, line_fn, a.b)
                for m in ptl:
                    line_rows.append(dict(base, analysis='line fraction difference, same windows', metric=m, point=ptl[m],
                                          ci_lo=cil[m][0] if cil else np.nan, ci_hi=cil[m][1] if cil else np.nan))
                for sd in seeds:
                    ka, kb = keep_a[keep_a.seed.eq(sd)], keep_b[keep_b.seed.eq(sd)]
                    ea, eb = ka.predicted_HR - ka.reference_HR, kb.predicted_HR - kb.reference_HR
                    fa, fb = ga[ga.seed.eq(sd)], gb[gb.seed.eq(sd)]
                    r = dict(base, seed=sd, windows_all=len(fa), windows_removed=len(fa) - len(ka),
                             windows_common_high=int(ka.set.eq('high').sum()), windows_common_low=int(ka.set.eq('low').sum()))
                    for st in ('high', 'low'):
                        ma, mb = ka.set.eq(st).to_numpy(), kb.set.eq(st).to_numpy()
                        r[f'MAE_{st}_diff'] = ea[ma].abs().mean() - eb[mb].abs().mean()
                        r[f'signed_error_{st}_diff'] = ea[ma].mean() - eb[mb].mean()
                        r[f'line_{st}_a'] = on_line(fa[fa.set.eq(st)].predicted_HR).mean()
                        r[f'line_{st}_b'] = on_line(fb[fb.set.eq(st)].predicted_HR).mean()
                    seed_rows.append(r)
                print(ds, bb, f'{x}-{y}', {m: round(pt[m], 2) for m in KEEP}, 'removed per seed',
                      [r['windows_removed'] for r in seed_rows[-len(seeds):]], flush=True)
    pd.DataFrame(rows).to_csv(out / 'common_windows.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(out / 'common_windows_per_seed.csv', index=False)
    pd.DataFrame(line_rows).to_csv(out / 'line_diff.csv', index=False)


if __name__ == '__main__':
    main()
