"""Brief 02 6.45 나 3: presented label HR distribution of RhythmFormer (0B) (no training), beside (ii-pc) and (ii-pc-g80).

(0B): the exposure records of the three October runs (main project outputs/st30t_st30t_RhythmFormer_PURE_0B_s*_label_exposures
.csv, read only): label HR of every presented sample after the author augmentation (common HR rule on the chunk label).
(ii-pc): the OPS2 training exposure records (cache/train/<B>_ii-pc_s*/exposures.csv): original chunks with their reference
HR, pool items with their nominal target HR; RhythmFormer and PhysNet (and matched FactorizePhys).
(ii-pc-g80): the same (ii-pc) exposure sequence with the pool items' targets mapped by the (ii-pc-g80) rule
(h* < 100 -> 70 + (h* - 70)/3); the arm uses the same order, seed and draws as (ii-pc), so this is the sequence it is shown
(PhysNet and matched FactorizePhys, the backbones of that arm).
Bins: the 3.515625-bpm HR grid (bin = round(HR / 3.515625)) and bands < 70, 70-80, 80-100, 100-120, >= 120 bpm.
Writes outputs/label_dist/grid.csv and bands.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
import json  # 7.08
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
MAIN_OUT = Path(CFG['october_project']) / 'outputs'  # read-only
SEEDS = (20261131, 20261132, 20261133)
GRID = 3.515625
BANDS = [(-np.inf, 70, '<70'), (70, 80, '70-80'), (80, 100, '80-100'), (100, 120, '100-120'), (120, np.inf, '>=120')]


def g80(h):
    return np.where(h < 100, 70 + (h - 70) / 3, h)


def main():
    out = WORK / 'outputs' / 'label_dist'
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    sets[('RhythmFormer', '(0B)')] = pd.concat([pd.read_csv(MAIN_OUT / f'st30t_st30t_RhythmFormer_PURE_0B_s{s}_label_exposures.csv')
                                                 .assign(seed=s) for s in SEEDS]).label_HR.to_numpy(float)
    for b in ('RhythmFormer', 'PhysNet', 'FPm'):
        e = pd.concat([pd.read_csv(WORK / f'cache/train/{b}_ii-pc_s{s}/exposures.csv').assign(seed=s) for s in SEEDS])
        sets[(b, '(ii-pc)')] = e.label_HR.to_numpy(float)
        if b in ('PhysNet', 'FPm'):
            h = e.label_HR.to_numpy(float)
            sets[(b, '(ii-pc-g80)')] = np.where(e.kind.eq(1), g80(h), h)
    grid, bands = [], []
    for (b, arm), h in sets.items():
        k = np.round(h / GRID).astype(int)
        for v, n in zip(*np.unique(k, return_counts=True)):
            grid.append(dict(backbone=b, arm=arm, bin_center_bpm=v * GRID, exposures=int(n), share=n / len(h)))
        r = dict(backbone=b, arm=arm, exposures=len(h))
        for lo, hi, name in BANDS:
            r[name] = float(((h >= lo) & (h < hi)).mean())
        r['>=100'] = float((h >= 100).mean())
        bands.append(r)
    pd.DataFrame(grid).to_csv(out / 'grid.csv', index=False)
    pd.DataFrame(bands).to_csv(out / 'bands.csv', index=False)
    print(pd.DataFrame(bands).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
