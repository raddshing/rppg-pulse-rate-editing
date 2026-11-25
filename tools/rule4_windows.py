"""Diagnostic rule 4 (brief 02 6.90 다 1): window HR of the rule-4 waveforms on the windows of the existing UBFC/PURE
evaluation (main study outputs/st15u_windows.csv, read only: 10 s windows, HAR and L0 layers).

Readout exactly as b1/mcd_windows.py: model frame grid = uniform 30 Hz grid over the endpoint-linear frame times used at
inference (src/st11x/pathways.uniform_frame_grid); a window starts at the first grid sample >= start_seconds
(complete_windows) and needs 300 predicted samples, otherwise unavailable; HR = grid._prepare(wave, diff_flag, band) +
periodogram peak (HAR 0.6-3.3 Hz, L0 0.75-2.5 Hz); diff_flag for PhysNet only.
Usage: rule4_windows.py CONDITION     Output: outputs/rule4/preds_<CONDITION>.csv.gz
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
sys.path.insert(0, str(V))
os.environ['ST9_NAMESPACE'] = 'st9g'
from src.eval import grid  # noqa: E402
from src.st11x.pathways import uniform_frame_grid, complete_windows  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
WINDOWS = Path(CFG['october_project']) / 'outputs' / 'st15u_windows.csv'
BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}
VIEW = {'st9t_PhysNet': 'PhysNet', 'st30t_FPm': 'FPm'}


def view_of(run):
    return next((v for k, v in VIEW.items() if run.startswith(k)), 'RhythmFormer')


def main():
    cond = sys.argv[1]
    cache = WORK / 'cache' / 'rule4' / 'waveforms' / cond / 'rule4'
    w = pd.read_csv(WINDOWS, usecols=['probe_dataset', 'recording_id', 'subject_id', 'window_id', 'start_seconds', 'layer'])
    out = []
    for rid, wins in w.groupby('recording_id'):
        z = np.load(cache / f'{rid}.npz')
        info = json.loads(str(z['meta']))
        n, span = int(info['decoded_frames']), float(info['span_seconds'])
        frame_grid = uniform_frame_grid(np.arange(n, dtype=np.float64) * span / (n - 1))
        bounds = {wid: complete_windows(frame_grid, [s])[0] for wid, s in
                  wins.drop_duplicates('window_id')[['window_id', 'start_seconds']].itertuples(index=False)}
        for run in [f for f in z.files if f != 'meta']:
            pred = np.asarray(z[run])
            diff = view_of(run) == 'PhysNet'
            for r in wins.itertuples(index=False):
                bd = bounds[r.window_id]
                start = bd['grid_start']
                base = dict(condition=cond, run_id=run, window_id=r.window_id, layer=r.layer)
                if not bd['available'] or start + 300 > len(pred):
                    out.append(dict(base, available=False, predicted_HR=np.nan))
                    continue
                band = BANDS[r.layer]
                pw = grid._prepare(pred[start:start + 300], diff, band, 'default', 30)
                out.append(dict(base, available=True, predicted_HR=grid._estimate(pw, 'periodogram', band, 30)))
    d = pd.DataFrame(out)
    d.to_csv(WORK / 'outputs' / 'rule4' / f'preds_{cond}.csv.gz', index=False)
    print(cond, len(d), 'rows', d.run_id.nunique(), 'runs', int(d.available.sum()), 'available', flush=True)


if __name__ == '__main__':
    main()
