"""Recording-level scores in the benchmark convention (ACCESS_OPS2_BRIEF_02 6.46 가 7, 6.54 다): one HR per recording from
the whole predicted waveform and the whole contact signal, narrow band 0.75-2.5 Hz; MAE, RMSE and Pearson r over recordings
per key, the MAE of the recordings with reference HR >= 100 bpm, and per arm the seed mean and sample SD. Descriptive.

October code, unchanged (archived tree, src/eval at commit 34dd3516c / 6ebbc9cd):
  recording interval   har.evaluate_arrays: the contiguous prediction prefix from frame 0 (packed chunks, recording unit)
  HR (setting 'L0')    published_settings.unit_settings: grid._prepare (cumulative sum for differenced outputs, detrend
                       lambda 100, first-order Butterworth 0.75-2.5 Hz, filtfilt) and grid._estimate 'periodogram'
                       (nfft = next power of two, maximum within 0.75-2.5 Hz)
Differences from the ST1 use of that code (stated in the report):
  - the contact signal is the raw contact PPG (stage-1 labels/), prepared without the cumulative sum (as the development
    readouts b1/mmpd_a.py and b1/mcd_windows.py); ST1 used the toolbox labels, which are differenced for PhysNet
  - MCD-rPPG: the interval is the prediction prefix intersected with the finite contact signal (mapped meta times); a gap
    inside it leaves the recording undefined
Recording sets: every recording with a stage-1 contact signal (MMPD: the input plan; MCD-rPPG: confirmed mapping).
Reference HR of a recording = the L0 HR of its contact signal over the same interval (high: >= 100 bpm).
"""
import numpy as np
import pandas as pd

BAND = (0.75, 2.5)
LENGTH = {'PhysNet': 128, 'FPm': 160, 'RhythmFormer': 160}


def recording_row(pred, label, view, diff):
    from src.eval import grid, har
    pred = np.asarray(pred)
    n = len(pred)
    L = LENGTH[view]
    assert n % L == 0 and len(label) >= n
    lab = np.asarray(label[:n], dtype=np.float64)
    rec = har.evaluate_arrays(pred, lab, np.arange(n // L) * L, L, 30, diff, len(label))['recording']
    start, end = int(rec['start_frame']), int(rec['end_frame'])
    fin = np.flatnonzero(np.isfinite(lab[start:end])) + start
    if not len(fin):
        return dict(status='no_contact_signal', start=start, end=end, frames=0)
    a, b = int(fin[0]), int(fin[-1]) + 1
    if b - a != len(fin):
        return dict(status='contact_gap_inside_interval', start=a, end=b, frames=b - a)
    p = grid._estimate(grid._prepare(pred[a:b], diff, BAND, 'default', 30), 'periodogram', BAND, 30)
    r = grid._estimate(grid._prepare(lab[a:b], False, BAND, 'default', 30), 'periodogram', BAND, 30)
    ok = bool(np.isfinite(p) and np.isfinite(r))
    return dict(status='ok' if ok else 'undefined_hr', start=a, end=b, frames=b - a, predicted_HR=p, reference_HR=r)


def run(dataset, s1, stage2_dirs, keys, bad, name_map):
    """keys: checkpoints table (key, backbone, arm, seed, variant, key_set, view); primary variants, defective keys excluded."""
    from pathlib import Path
    labels = {f.stem: f for f in sorted((Path(s1) / 'labels').glob('*.npy'))}
    rows = []
    for k in keys.itertuples():
        if k.variant not in ('std', 'fsam-on') or k.key in bad:
            continue
        src = next((Path(x) / dataset / 'pred' / k.key for x in stage2_dirs if (Path(x) / dataset / 'pred' / k.key).is_dir()), None)
        if src is None:
            continue
        diff = k.view == 'PhysNet' and not str(k.arm).endswith('-std')
        for rec, lf in labels.items():
            f = src / (rec + '.npy')
            base = dict(key=k.key, backbone=name_map.get(k.backbone, k.backbone), arm=k.arm, seed=int(k.seed), key_set=k.key_set,
                        recording_id=rec.replace('__', '/'))
            if not f.exists():
                rows.append(dict(base, status='no_prediction'))
                continue
            rows.append(dict(base, **recording_row(np.load(f), np.load(lf), k.view, diff)))
    rec = pd.DataFrame(rows)
    out = []
    for (bb, arm, seed, key, ks), g in rec.groupby(['backbone', 'arm', 'seed', 'key', 'key_set']):
        ok = g[g.status.eq('ok')]
        e = ok.predicted_HR - ok.reference_HR
        hi = ok[ok.reference_HR >= 100]
        out.append(dict(backbone=bb, arm=arm, seed=seed, key=key, key_set=ks, recordings=len(g), recordings_ok=len(ok),
                        MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                        pearson_r=float(np.corrcoef(ok.predicted_HR, ok.reference_HR)[0, 1]) if len(ok) > 2 else np.nan,
                        recordings_ge100=len(hi), MAE_ge100=float((hi.predicted_HR - hi.reference_HR).abs().mean()) if len(hi) else np.nan))
    per_key = pd.DataFrame(out)
    summ = []
    for (bb, arm), g in per_key.groupby(['backbone', 'arm']):
        r = dict(backbone=bb, arm=arm, seeds=len(g), key_set=g.key_set.iloc[0], recordings_ok_min=int(g.recordings_ok.min()),
                 recordings_ge100_min=int(g.recordings_ge100.min()))
        for m in ('MAE', 'RMSE', 'pearson_r', 'MAE_ge100'):
            r[m + '_mean'] = float(g[m].mean())
            r[m + '_sd'] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
        summ.append(r)
    return rec, per_key, pd.DataFrame(summ)
