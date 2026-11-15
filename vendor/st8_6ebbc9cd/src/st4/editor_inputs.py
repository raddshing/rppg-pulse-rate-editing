"""Fixed native uint8 clips, skin masks and contact paths for the CPU editor."""
from src.st4.base import *
from src.st4.ref_audit import raw_gt
from src.st4.signals import common_hr
from fractions import Fraction
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from scipy.signal import resample_poly


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


def target_cases():
    source_hash('outputs/st3_c3_targets.csv')
    source_hash('outputs/st3_c3_clips.csv')
    mm, pure, allowed = indices()
    targets = pd.read_csv(ROOT / 'outputs/st3_c3_targets.csv')
    clips = pd.read_csv(ROOT / 'outputs/st3_c3_clips.csv')
    excluded = set(clips[clips.variant.eq('identity') & clips.excluded.ne(0)].window_id)
    cases = [dict(dataset='MMPD', window_id=r.window_id, recording_id=r.recording_id,
                  window_index=int(r.window_index), nominal_original_HR=float(r.hr_label_bpm), target_HR=float(r.target_hr))
             for r in targets[targets.role.eq('c1R') & ~targets.window_id.isin(excluded)].itertuples()]
    check('MMPD editor exact162 fixed-box windows', len(cases) == 162)
    for rid, row in sorted(pure.items()):
        if not rid.endswith('-01'):
            continue
        gt = contact(row, 'PURE', allowed)
        for j in range(int(row['frame_count']) // 300):
            original = common_hr(gt[j * 300:(j + 1) * 300], 'HAR')
            cases.append(dict(dataset='PURE', window_id=f'PURE|{rid}|{j}', recording_id=rid,
                              window_index=j, nominal_original_HR=original, target_HR=original + 15))
    check('PURE editor exact60 task01 windows', sum(r['dataset'] == 'PURE' for r in cases) == 60)
    return sorted(cases, key=lambda r: (r['dataset'], r['window_id'])), mm, pure, allowed


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


def hc_box(clip, dataset):
    # Pipe the existing uint8 first-frame probe; no RGB frame file is written.
    probe = np.ascontiguousarray(clip[:1 if dataset == 'PURE' else 30])
    args = [str(Path(PATHS['environments']['repro']) / 'bin/python'), '-B', '-s', '-m',
            'src.st4.hc_probe', str(probe.shape[0]), str(probe.shape[1]), str(probe.shape[2])]
    child = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ledger('hc_probe', 'started', child.pid)
    stdout, stderr = child.communicate(probe.tobytes())
    check('Frozen repro Haar detector exit', child.returncode == 0)
    value = json.loads(stdout)
    check('Frozen repro OpenCV version', value['opencv_version'] == '4.5.2')
    box = value['box']
    if box is None:
        return None
    x, y, w, h = box
    left, top = max(0, int(x - .25 * w)), max(0, int(y - .25 * h))
    return [left, top, min(clip.shape[2], int(left + 1.5 * w)), min(clip.shape[1], int(top + 1.5 * h))]


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


def interpolate_masks(clip):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    source_hash('outputs/st3_roi_indices.json')
    roi = json.loads((ROOT / 'outputs/st3_roi_indices.json').read_text())
    points = np.full((300, 478, 2), np.nan, dtype=np.float32)
    h, w = clip.shape[1:3]
    with vision.FaceLandmarker.create_from_options(tracking_options()) as tracker:
        for i, rgb in enumerate(clip):
            result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), round(i * 1000 / 30))
            if result.face_landmarks:
                points[i] = [[p.x * w, p.y * h] for p in result.face_landmarks[0]]
    detected = np.isfinite(points[:, 0, 0])
    check('Every editor window has valid skin landmarks', detected.any())
    for k in range(points.reshape(300, -1).shape[1]):
        column = points.reshape(300, -1)[:, k]
        good = np.isfinite(column)
        column[:] = np.interp(np.arange(300), np.flatnonzero(good), column[good])
    masks = np.stack([skin_mask(clip.shape[1:], p, roi) for p in points])
    check('Every editor frame has nonempty skin mask', np.all(masks.sum(axis=(1, 2)) > 0))
    return masks, float(1 - detected.mean())


def geometry_file(window_id, mode='gate'):
    directory = CACHE / 'editor_geometry' / mode
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (hashlib.sha256(window_id.encode()).hexdigest() + '.npz')


def save_geometry(path, masks, box, clip):
    np.savez_compressed(path, packed_masks=np.packbits(masks.reshape(300, -1), axis=1),
                        mask_shape=np.array(masks.shape), box=np.asarray(box),
                        uint8_sha256=np.asarray(hashlib.sha256(clip.tobytes()).hexdigest()))


def read_geometry(path, clip):
    with np.load(path, allow_pickle=False) as data:
        check('Calibration and edited starting uint8 frames match exactly', hashlib.sha256(clip.tobytes()).hexdigest() == str(data['uint8_sha256']))
        shape = tuple(data['mask_shape'])
        return np.unpackbits(data['packed_masks'], axis=1, count=shape[1] * shape[2]).reshape(shape).astype(bool)


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
