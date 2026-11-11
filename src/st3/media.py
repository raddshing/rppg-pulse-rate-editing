"""Server-only pixels and geometry for Stage 3; no old artifact writes."""
from src.st3.base import *
import subprocess
import numpy as np
import cv2


def isolated_frames(path, expected_count=None):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(TMP))
    child = subprocess.Popen([str(Path(PATHS['environments']['repro'])/'bin/python'),
                              '-B', '-s', str(ROOT/'src/meta/decode_video.py'),
                              '--video', str(path)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env, cwd=ROOT)
    done = False
    try:
        header = json.loads(child.stdout.readline(4096))
        check('Fresh repro decoder', header['opencv_version'], '4.5.2')
        n, w, h = (int(header[k]) for k in ('frame_count', 'width', 'height'))
        if expected_count is not None:
            check('Original decoded frame count', n, expected_count)
        digest = hashlib.sha256()
        for i in range(n):
            data = bytearray()
            while len(data) < h*w*3:
                block = child.stdout.read(h*w*3-len(data))
                check('Complete server-only RGB frame', bool(block))
                data.extend(block)
            digest.update(data)
            yield np.frombuffer(data, dtype=np.uint8).reshape(h,w,3).copy()
        footer = json.loads(child.stdout.readline(4096))
        check('Repro RGB transport parity', digest.hexdigest(), footer['rgb_sha256'])
        check('No extra decoder bytes', child.stdout.read(1), b'')
        check('Repro decoder exit', child.wait(), 0)
        done = True
    finally:
        if not done and child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        child.stdout.close()
        child.stderr.close()


def frames(spec):
    if spec['kind'] == 'G':
        x = np.load(spec['path'], mmap_mode='r', allow_pickle=False)
        check('Generated frames uint8', x.dtype == np.uint8)
        check('Generated frame count', len(x), spec['n'])
        yield from x
    else:
        yield from isolated_frames(spec['path'], spec['n'])


def driver_indices(n_source, n_driver):
    """Exact fresh official generation rule: duplicated turn endpoints."""
    check('Positive source and driver lengths', n_source > 0 and n_driver > 0)
    segments = (np.arange(n_source, dtype=np.int64) // n_driver)
    local = np.arange(n_source, dtype=np.int64) % n_driver
    idx = np.where(segments % 2, n_driver-1-local, local)
    return idx, segments


def skin_mask(rgb_shape, points, indices):
    mask = np.zeros(rgb_shape[:2], dtype=np.uint8)
    for polygon in indices.values():
        xy = np.rint(points[polygon,:2]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(xy), 1)
    return mask


def skin_mean(rgb, points, indices):
    pixels = rgb[skin_mask(rgb.shape,points,indices).astype(bool)]
    check('Nonempty skin polygon', len(pixels)>0)
    return pixels.mean(axis=0)


def gt_aligned(row, dataset):
    if dataset in ('UBFC','MAUBFC'):
        wave = np.array([float(v) for v in Path(row['label_path']).open().readline().split()])
        check('UBFC GT frame alignment', len(wave), int(row['frame_count']))
        return wave
    if dataset == 'MMPD':
        check('A selected before contact PPG read', row['subject_id'] in SUBJECTS)
        from scipy.io import loadmat
        wave = loadmat(row['label_path'], variable_names=['GT_ppg'])['GT_ppg'].T.reshape(-1)
    elif dataset == 'PURE':
        labels = json.loads(Path(row['label_path']).read_text())
        wave = np.array([e['Value']['waveform'] for e in labels['/FullPackage']],dtype=float)
    else:
        raise ValueError('Unsupported GT source')
    n = int(row['frame_count'])
    return np.interp(np.linspace(1,len(wave),n),np.linspace(1,len(wave),len(wave)),wave)


def delete_temp(path):
    path = Path(path)
    check('Delete only new Stage 3 temporary paths', CACHE.resolve() in path.resolve().parents)
    record = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    dest = ROOT/'outputs/st3_deleted_temp.csv'
    with dest.open('a',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(record))
        if dest.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)
    path.unlink()
