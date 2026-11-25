"""Brief 02 6.54 나 5: the RhythmFormer (0) sigma-1.5 blur models were run twice under the guard: the rule-2 run (cache
gblur15_g3) and again in the remaining-arms run (cache hblur15_g3). Compare the two runs (saved arrays and readouts only).
Arrays: every recording file present in both caches, every run key present in both (the three (0) checkpoints): byte equality
and largest absolute difference. Readouts: outputs/guarded/blur_windows/mcd_dev_preds_gblur15_g3.csv.gz against
outputs/guarded/blur_windows_rest/mcd_dev_preds_hblur15_g3.csv.gz on the same run_id, window_id, layer: predicted_HR and
availability. Prints a summary; writes outputs/blur_guarded_rest/second_run_comparison.csv (per run).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
G, H = WORK / 'cache/mcd_dev_waveforms/gblur15_g3', WORK / 'cache/mcd_dev_waveforms/hblur15_g3'


def main():
    rows = {}
    files = sorted(set(os.listdir(G)) & set(os.listdir(H)))
    for fn in files:
        a, b = np.load(G / fn, allow_pickle=False), np.load(H / fn, allow_pickle=False)
        for k in sorted((set(a.files) & set(b.files)) - {'meta'}):
            x, y = a[k], b[k]
            r = rows.setdefault(k, dict(run_id=k, recordings=0, arrays_equal=0, shape_mismatch=0, max_abs_difference=0.0))
            r['recordings'] += 1
            if x.shape != y.shape:
                r['shape_mismatch'] += 1
                continue
            r['arrays_equal'] += int(x.tobytes() == y.tobytes())
            if x.size:
                d = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))) if np.isfinite(x).any() else 0.0
                r['max_abs_difference'] = max(r['max_abs_difference'], float(d))
    ra = pd.read_csv(WORK / 'outputs/guarded/blur_windows/mcd_dev_preds_gblur15_g3.csv.gz')
    rb = pd.read_csv(WORK / 'outputs/guarded/blur_windows_rest/mcd_dev_preds_hblur15_g3.csv.gz')
    k = ['run_id', 'window_id', 'layer']
    m = ra.merge(rb[rb.run_id.isin(ra.run_id.unique())], on=k, how='outer', suffixes=('_1', '_2'), indicator=True)
    for run, g in m.groupby('run_id'):
        both = g[g._merge.eq('both')]
        p1, p2 = both.predicted_HR_1.to_numpy(float), both.predicted_HR_2.to_numpy(float)
        same = (p1 == p2) | (np.isnan(p1) & np.isnan(p2))
        r = rows.setdefault(run, dict(run_id=run))
        r.update(readout_rows=len(g), readout_rows_one_side=int(g._merge.ne('both').sum()), readout_equal=int(same.sum()),
                 readout_max_abs_difference=float(np.nanmax(np.abs(p1 - p2))) if len(both) and np.isfinite(p1 - p2).any() else 0.0,
                 availability_differs=int((both.available_1.astype(bool) != both.available_2.astype(bool)).sum()))
    t = pd.DataFrame(list(rows.values()))
    out = WORK / 'outputs/blur_guarded_rest'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'second_run_comparison.csv', index=False)
    print('recording files in both', len(files))
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
