"""Brief 02 6.48 다 2 / docs/STAGE5_DEV_DIAGNOSTICS.md rule 3 (descriptive): spectra of the mean absolute frame difference and
of the Laplacian energy in the face view of the development videos (CPU, no model).

Face view: the RhythmFormer network view (HC 1.5 box of the first frame, 128 x 128 INTER_AREA), as b1/mcd_infer.py.
Per decoded frame t: MAD(t) = mean |x_t - x_(t-1)| over pixels and channels (MAD(0) = MAD(1)); LAP(t) = mean of the squared
3x3 Laplacian (cv2.Laplacian, ksize 1) of the grey frame (mean of R, G, B). Both signals are placed on the recording's frame
times (MCD: endpoint-linear over the meta span, as the model input; MMPD: 30 fps), interpolated to the 30 Hz grid and given
a Welch spectrum as the 2026-11-11 entry (linear detrend, Hann, 300/150, 512 points, segment means removed).
MCD-rPPG development half (draft-rule recordings, the B9 set): power at the recording's key-frame frequency (VOP rate over
the span / 12, outputs/gop) relative to the median of the six neighbouring bins, and the signals folded at the key-frame
interval (mean by decoded frame number mod 12, as a fraction of the signal's SD).
MMPD-A (the 195 recordings of the report-4 windows): the same relative power at 2.0 and 3.0 Hz, by background flag
(tools/b2_flag_robust.py flags()).
Writes outputs/frame_spectra/{mcd,mmpd}_recordings.csv (no pixels; scalars per recording) and mean_spectra.csv.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
sys.path.insert(0, str(WORK / 'b9'))
import mcd_infer as M  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import json  # 7.08
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
MMPD = Path(CFG['mmpd_raw'])  # read-only, MMPD-A recordings only


def signals(view):
    import cv2
    v = view.astype(np.float64)
    mad = np.r_[0.0, np.abs(np.diff(v, axis=0)).mean(axis=(1, 2, 3))]
    mad[0] = mad[1]
    grey = v.mean(axis=3)
    lap = np.array([np.mean(cv2.Laplacian(g, cv2.CV_64F, ksize=1) ** 2) for g in grey])
    return mad, lap


def welch(sig, times):
    from scipy.signal import welch as w, detrend
    grid = np.arange(int(np.floor(times[-1] * 30 + 1e-9)) + 1) / 30.
    y = np.interp(grid, times, sig)
    f, p = w(detrend(y, type='linear'), fs=30, window='hann', nperseg=300, noverlap=150, nfft=512, detrend='constant')
    return f, p


def rel_power(f, p, hz):
    k = int(np.argmin(np.abs(f - hz)))
    nb = np.r_[p[max(k - 4, 0):k - 1], p[k + 2:k + 5]]
    return float(p[k - 1:k + 2].max() / np.median(nb)), float(p[k])


def mcd_job(args):
    rec, kf_hz = args
    import cv2
    cap = cv2.VideoCapture(str(M.MCD / rec['video']))
    frames, box, k = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if k == 0:
            box = np.asarray(M.LOADER.face_detection(rgb, 'HC', True, 1.5), dtype=int)
        a = rgb[max(box[1], 0):min(box[1] + box[3], rgb.shape[0]), max(box[0], 0):min(box[0] + box[2], rgb.shape[1])]
        frames.append(cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA))
        k += 1
    cap.release()
    view = np.asarray(frames)
    n = len(view)
    times = np.arange(n) * float(rec['span_seconds']) / (n - 1)
    out = dict(recording_id=rec['recording_id'], frames=n, keyframe_hz=kf_hz)
    spectra = {}
    for name, s in zip(('MAD', 'LAP'), signals(view)):
        f, p = welch(s, times)
        out[f'{name}_kf_rel'], out[f'{name}_kf_power'] = rel_power(f, p, kf_hz)
        fold = np.array([s[j::12].mean() for j in range(12)])
        out[f'{name}_fold_range_over_sd'] = float((fold.max() - fold.min()) / s.std())
        out[f'{name}_fold_phase0_minus_rest_over_sd'] = float((fold[0] - np.delete(fold, 0).mean()) / s.std())
        spectra[name] = p / p.sum()
    return out, spectra


def mmpd_job(rid):
    from scipy.io import loadmat
    import cv2
    v = loadmat(MMPD / (rid + '.mat'), variable_names=['video'])['video']
    v = np.asarray(v)
    u8 = np.clip(np.rint(v * 255 if v.max() <= 1.0 else v), 0, 255).astype(np.uint8)
    box = np.asarray(M.LOADER.face_detection(u8[0], 'HC', True, 1.5), dtype=int)
    view = np.asarray([cv2.resize(a[max(box[1], 0):min(box[1] + box[3], a.shape[0]), max(box[0], 0):min(box[0] + box[2], a.shape[1])],
                                  (128, 128), interpolation=cv2.INTER_AREA) for a in u8])
    times = np.arange(len(view)) / 30.
    out = dict(recording_id=rid, frames=len(view))
    spectra = {}
    for name, s in zip(('MAD', 'LAP'), signals(view)):
        f, p = welch(s, times)
        for hz in (2.0, 3.0):
            out[f'{name}_{hz:.0f}Hz_rel'], out[f'{name}_{hz:.0f}Hz_power'] = rel_power(f, p, hz)
        spectra[name] = p / p.sum()
    return out, spectra


def main():
    out = WORK / 'outputs' / 'frame_spectra'
    out.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get('OPS2_WORKERS', '2'))
    gop = pd.read_csv(WORK / 'outputs/gop/mcd_dev_gop.csv').set_index('recording_id')
    rows = [r for r in M.recordings() if r['draft_rule'] and isinstance(r.get('video'), str)]
    jobs = [(r, float(gop.loc[r['recording_id'], 'vop_fps_over_span'] / 12)) for r in rows]
    mean_spec = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=M.init_worker) as ex:
        res = list(ex.map(mcd_job, jobs))
        pd.DataFrame([r for r, _ in res]).to_csv(out / 'mcd_recordings.csv', index=False)
        for name in ('MAD', 'LAP'):
            mean_spec.append(dict(dataset='MCD-dev', signal=name, group='all',
                                  **{f'bin{k}': v for k, v in enumerate(np.mean([s[name] for _, s in res], axis=0))}))
        print('mcd done', len(res), flush=True)
        st = pd.read_csv(out / 'mmpd_status.csv')  # written in the main environment: report-4 recordings and background flags
        status = dict(zip(st.recording_id, st.background))
        rids = list(st.recording_id)
        res2 = list(ex.map(mmpd_job, rids))
    t = pd.DataFrame([r for r, _ in res2])
    t['background'] = t.recording_id.map(status)
    t.to_csv(out / 'mmpd_recordings.csv', index=False)
    for name in ('MAD', 'LAP'):
        for st in ('flagged', 'unflagged', 'no_background'):
            ss = [s[name] for (r, s) in res2 if status.get(r['recording_id']) == st]
            if ss:
                mean_spec.append(dict(dataset='MMPD-A', signal=name, group=st, **{f'bin{k}': v for k, v in enumerate(np.mean(ss, axis=0))}))
    pd.DataFrame(mean_spec).to_csv(out / 'mean_spectra.csv', index=False)
    print('mmpd done', len(res2))


if __name__ == '__main__':
    main()
