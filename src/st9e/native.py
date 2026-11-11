"""Frozen native input and alignment functions, scoped to Part1 paths."""
from src.st9e.base import *
from src.st9e.signals import common_hr
from fractions import Fraction
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from scipy.signal import resample_poly, butter, filtfilt

def raw_gt(row, allowed):
    check('MMPD-A selected before raw contact read', row['subject_id'] in allowed)
    path = Path(row['label_path'])
    check('Declared original MMPD path', Path(PATHS['raw']['MMPD']).resolve() in path.resolve().parents)
    raw = loadmat(path, variable_names=['GT_ppg'])['GT_ppg'].T.reshape(-1)
    check('Raw30Hz contact frame alignment without resampling', len(raw) == int(row['frame_count']) and float(row['processing_fs']) == 30)
    return raw
def indices():
    source_hash('data/splits/mmpd_split_st1.json')
    source_hash('data/index/mmpd_index_st1.csv')
    source_hash('data/index/pure_index_st1.csv')
    allowed = set(json.loads((ROOT / 'data/splits/mmpd_split_st1.json').read_text())['A'])
    mm = {r['recording_id']: r for r in rows('data/index/mmpd_index_st1.csv') if r['subject_id'] in allowed}
    pure = {r['recording_id']: r for r in rows('data/index/pure_index_st1.csv')}
    return mm, pure, allowed

def contact(row, dataset, allowed):
    if dataset == 'MMPD':
        return raw_gt(row, allowed)
    check('Only authorized PURE label path', dataset == 'PURE' and
          Path(PATHS['raw']['PURE']).resolve() in Path(row['label_path']).resolve().parents)
    labels = json.loads(Path(row['label_path']).read_text())
    raw = np.asarray([v['Value']['waveform'] for v in labels['/FullPackage']], dtype=float)
    n = int(row['frame_count'])
    # Unchanged Stage1 PURE loader endpoint interpolation.
    return np.interp(np.linspace(1, len(raw), n), np.linspace(1, len(raw), len(raw)), raw)

def frame_files(row):
    root = Path(row['path'])
    check('Declared PURE source directory', Path(PATHS['raw']['PURE']).resolve() in root.resolve().parents)
    files = sorted(root.glob('*.png'))
    check('Frozen PURE source frame count', len(files) == int(row['frame_count']))
    return files

def read_pure(files, start, count=300):
    result = []
    for path in files[start:start + count]:
        bgr = cv2.imread(str(path))
        check('Original PURE frame decodes', bgr is not None)
        result.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    clip = np.asarray(result, dtype=np.uint8)
    check('Complete PURE frame interval', len(clip) == count)
    return clip

def load_recording(row, dataset, allowed):
    if dataset == 'PURE':
        return frame_files(row)
    check('A selected before original video access', row['subject_id'] in allowed)
    check('Declared original MMPD video path', Path(PATHS['raw']['MMPD']).resolve() in Path(row['path']).resolve().parents)
    video = loadmat(row['path'], variable_names=['video'])['video']
    check('Original MMPD normalized video shape', len(video) == int(row['frame_count']) and video.ndim == 4 and video.shape[-1] == 3)
    return video

def frame_window(source, dataset, start):
    if dataset == 'PURE':
        return read_pure(source, start)
    x = source[start:start + 300]
    check('Native MMPD window normalized range', len(x) == 300 and np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1)
    return np.rint(255 * x).astype(np.uint8)

def skin_mask(shape, points, roi):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for polygon in roi.values():
        xy = np.rint(points[polygon, :2]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(xy), 1)
    return mask.astype(bool)

def tracking_options():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarkerOptions(base_options=python.BaseOptions(
        model_asset_path=PATHS['face_landmarker_model'], delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_facial_transformation_matrixes=True)

def reflected_indices(index, n):
    z = np.asarray(index) % (2 * (n - 1))
    return np.where(z < n, z, 2 * (n - 1) - z).astype(int)

def parameters(scale):
    ratio = Fraction(1 / float(scale)).limit_denominator(64)
    return ratio.numerator, ratio.denominator

def resample(x, start, up, down, count=330, lead=15):
    margin = down * ((30 + down - 1) // down)
    # New B-range accelerations can have fewer than15 output context samples
    # under the inherited30-source-frame margin. Extend only those cases by
    # whole denominator blocks, retaining polyphase origin and reflection.
    if margin * up // down < lead:
        margin = down * ((lead + up - 1) // up + max((30 + down - 1) // down, 11))
    need = ((count + lead) * down + up - 1) // up + 2
    index = reflected_indices(np.arange(start - margin, start + need + margin), len(x))
    value = resample_poly(x[index].astype(np.float64), up, down, axis=0, window=('kaiser', 5.0))
    drop = margin * up // down
    check('Sufficient fixed rational-resampling leading context', drop >= lead)
    return value[drop - lead:drop - lead + count]
def editor_band(values):
    # ED1 changes only the inherited band, preserving its first-order filter.
    b, a = butter(1, [.6 / 15, 3.3 / 15], btype='bandpass')
    return filtfilt(b, a, values, axis=0)

def normalize(values):
    y = editor_band(values)
    check('Finite nonzero contact waveform for editor', np.isfinite(y).all() and y.std() > 0)
    return (y - y.mean()) / y.std()

def fit(clip, masks, gt, start):
    means = np.asarray([frame[m].mean(axis=0) for frame, m in zip(clip, masks)])
    normalized = means / means.mean(axis=0) - 1
    colors = editor_band(normalized)
    g = normalize(gt[reflected_indices(np.arange(start - 15, start + 315), len(gt))])
    fits = []
    for lag in range(-15, 16):
        wave = g[np.arange(300) + 15 - lag]
        wave = wave - wave.mean()
        centered = colors - colors.mean(axis=0)
        coef = (wave[:, None] * centered).sum(axis=0) / (wave**2).sum()
        residual = centered - wave[:, None] * coef
        total = (centered**2).sum(axis=0)
        r2 = np.divide(total - (residual**2).sum(axis=0), total, out=np.zeros(3), where=total > 0)
        fits.append((r2.sum(), lag, coef, r2))
    _, lag, coef, r2 = max(fits, key=lambda x: x[0])
    return dict(tau_frames=int(lag), a_R=float(coef[0]), a_G=float(coef[1]), a_B=float(coef[2]),
                R2_R=float(r2[0]), R2_G=float(r2[1]), R2_B=float(r2[2])), means
