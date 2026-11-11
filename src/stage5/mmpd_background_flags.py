"""Stage-5 sensitivity analysis: MMPD recordings with a periodic background peak near 2.0 Hz or 3.0 Hz.

Fixed on MMPD-A (development data) and applied unchanged to MMPD-B. Per recording:
- frames: the provider .mat 'video', converted as at inference (round(255 x), uint8);
- face box: the PhysNet-view face box of the model input path (src/st9e/model_view.py ViewWorker), found on the first
  frame and used for every frame, clipped to the frame; a recording whose clipped box covers the whole frame has no
  background pixels and is left out of this analysis;
- background signal: per frame, the mean of the R, G and B values over the pixels outside the face box;
- spectrum: linear detrend of the whole signal, then the Welch average of Hann-windowed 300-sample (10 s) segments with
  150-sample overlap, nfft 512, at 30 frames/s, i.e. on the 512-point HR grid (3.515625 bpm per bin), normalized to unit
  power in 0.6-3.3 Hz;
- flag: a bin among 33-35 (116.02-123.05 bpm) or 50-52 (175.78-182.81 bpm) that is larger than both neighbours and at
  least twice the median of the six surrounding bins, three on each side.
The windows of flagged recordings (and of recordings without background pixels) are removed before the contrasts are
repeated (exclude_flagged).
Inputs come from the environment: PULSE_EDIT_MMPD_ROOT (provider .mat files, '<subject>/<recording>.mat') and the split
column of lists/mmpd_subjects_A_B.csv. No recording list or path is stored here.
Usage: python -m src.stage5.mmpd_background_flags SPLIT OUT_CSV   (SPLIT = A or B)
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st9e.model_view import ViewWorker  # noqa: E402

FS, NFFT, SEG, OVERLAP = 30, 512, 300, 150
BAND = (.6, 3.3)
PEAK_BINS = {'2Hz': (33, 34, 35), '3Hz': (50, 51, 52)}


def background_signal(frames, worker):
    _, box = worker.view(frames[:1], 'PhysNet')
    bx, by, bw, bh = map(int, np.asarray(box).reshape(-1, 4)[0])
    y0, y1, x0, x1 = max(by, 0), min(by + bh, frames.shape[1]), max(bx, 0), min(bx + bw, frames.shape[2])
    n_out = frames.shape[1] * frames.shape[2] - (y1 - y0) * (x1 - x0)
    if n_out == 0:
        return None, 0
    total = frames.reshape(len(frames), -1, 3).sum(axis=1, dtype=np.float64)
    inner = frames[:, y0:y1, x0:x1].reshape(len(frames), -1, 3).sum(axis=1, dtype=np.float64)
    return ((total - inner) / n_out).mean(axis=1), n_out


def spectrum(signal):
    f, p = welch(detrend(signal), fs=FS, nperseg=SEG, noverlap=OVERLAP, nfft=NFFT)
    band = (f >= BAND[0]) & (f <= BAND[1])
    bins = np.round(f[band] * 60 / (FS / NFFT * 60)).astype(int)
    return bins, p[band] / p[band].sum()


def flags(bins, power):
    hits = set()
    for i in range(1, len(power) - 1):
        base = np.median(np.r_[power[max(0, i - 3):i], power[i + 1:i + 4]])
        if power[i] > power[i - 1] and power[i] > power[i + 1] and base > 0 and power[i] / base >= 2:
            hits.add(int(bins[i]))
    return {k: bool(hits & set(v)) for k, v in PEAK_BINS.items()}


def exclude_flagged(rows, flag_rows):
    """Window rows (dicts with recording_id) without the windows of flagged or background-less recordings."""
    keep = {r['recording_id'] for r in flag_rows if r['background_pixels'] and not r['flagged']}
    return [r for r in rows if r['recording_id'] in keep]


def main():
    split, out = sys.argv[1], Path(sys.argv[2])
    mmpd = Path(os.environ['PULSE_EDIT_MMPD_ROOT'])
    with (ROOT / 'lists' / 'mmpd_subjects_A_B.csv').open(newline='') as f:
        subjects = sorted(r['subject_id'] for r in csv.DictReader(f) if r['split'] == split)
    worker = ViewWorker()
    rows = []
    try:
        for s in subjects:
            for mat in sorted((mmpd / s).glob('*.mat')):
                video = np.asarray(loadmat(mat, variable_names=['video'])['video'])
                frames = np.round(video * 255).astype(np.uint8)
                sig, n_out = background_signal(frames, worker)
                row = dict(recording_id=f'{s}/{mat.stem}', background_pixels=n_out, flagged_2Hz=False, flagged_3Hz=False)
                if sig is not None:
                    fl = flags(*spectrum(sig))
                    row.update(flagged_2Hz=fl['2Hz'], flagged_3Hz=fl['3Hz'])
                row['flagged'] = row['flagged_2Hz'] or row['flagged_3Hz']
                rows.append(row)
    finally:
        worker.close()
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
