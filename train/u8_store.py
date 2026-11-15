"""uint8 pool storage with load-time model preprocessing (brief 02, section 6.1).

Stored per pool item (instead of the preprocessed float32 tensor):
  view : the editor's uint8 output after stochastic rounding, cropped with the item's fixed
         face box and INTER_AREA-resized exactly as before (cv2 on uint8 frames returns
         integers, so this step adds no rounding). Only the frames the chunk needs are kept
         (brief 02 section 6.6): FactorizePhys and RhythmFormer the chunk frames, PhysNet the
         chunk frames plus the next frame (frame differences).
  stats: the 300-frame context statistics the archived normalization uses (scalars):
         PhysNet the float32 std of the frame-difference array (diff_normalize_data);
         RhythmFormer the float64 mean and the float64 std of the centred view (standardized_data).
         FactorizePhys (Raw input) has none.
  label: the raw float64 target label p(rt) for the 300-frame context.
Load-time preprocessing repeats the archived per-backbone arithmetic (src/st9g/preprocess.py)
on the kept frames with the stored statistics, then the float32 cast and the archived dataset
transpose. Equality with the stored (ii) tensors is checked by SHA (brief 02 section 6.5).
An older full-context form (all 300 frames, no stats) is still read for the first PhysNet check items.
Optional lossless compression, OPS2_POOL_COMPRESS: '0' .npy; '1' .npz (zlib); 'delta-zlib' / 'delta-lzma'
(brief 02 6.9): the uint8 view is stored as the first frame plus neighbour-frame differences modulo 256
(uint8), as .npy bytes compressed with the standard-library zlib or lzma; every write is decoded again
and must equal the original uint8 frames. Labels and statistics stay plain .npy in the delta modes.
"""
import hashlib
import io
import lzma
import os
import zlib
import sys
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parents[1] / 'vendor' / 'st8_6ebbc9cd'
if str(V) not in sys.path:
    sys.path.insert(0, str(V))
from src.st9g.preprocess import diff_normalize_data, diff_normalize_label, standardized_data, standardized_label  # noqa: E402

def save_array(path, arr):
    """Write arr to path(.npy) or, with OPS2_POOL_COMPRESS=1, path(.npz, zlib); return the written path."""
    path = Path(path)
    if os.environ.get('OPS2_POOL_COMPRESS', '0') == '1':
        out = path.with_suffix('.npz')
        assert not out.exists()
        np.savez_compressed(out, a=arr)
    else:
        out = path.with_suffix('.npy')
        assert not out.exists()
        np.save(out, arr)
    return out


def delta_encode(u8):
    """First frame, then frame-to-frame differences modulo 256 (uint8 arithmetic wraps)."""
    u8 = np.ascontiguousarray(u8)
    assert u8.dtype == np.uint8
    d = np.empty_like(u8)
    d[0] = u8[0]
    d[1:] = u8[1:] - u8[:-1]
    return d


def delta_decode(d):
    return np.cumsum(d, axis=0, dtype=np.uint8)


def _npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def encode_view(u8, method):
    """Bytes as written for a uint8 view under method ('delta-zlib' or 'delta-lzma')."""
    raw = _npy_bytes(delta_encode(u8))
    return zlib.compress(raw, 6) if method == 'delta-zlib' else lzma.compress(raw, preset=6)


def save_view(path, u8):
    """Save the kept uint8 frames; delta modes are decoded again and must equal the original frames."""
    mode = os.environ.get('OPS2_POOL_COMPRESS', '0')
    if mode not in ('delta-zlib', 'delta-lzma'):
        return save_array(path, u8)
    out = Path(path).with_name(Path(path).name + ('.npyd.zz' if mode == 'delta-zlib' else '.npyd.xz'))
    assert not out.exists()
    out.write_bytes(encode_view(u8, mode))
    if not np.array_equal(read_array(out), u8):
        raise RuntimeError('delta encoding did not restore the original uint8 frames: ' + out.name)
    return out


def read_array(path):
    path = str(path)
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=False) as z:
            return z['a']
    if path.endswith('.npyd.zz') or path.endswith('.npyd.xz'):
        data = Path(path).read_bytes()
        raw = zlib.decompress(data) if path.endswith('.zz') else lzma.decompress(data)
        return delta_decode(np.load(io.BytesIO(raw), allow_pickle=False))
    return np.load(path, allow_pickle=False)


def to_u8(view):
    """Exact uint8 form of an integer-valued float64 model view; refuses any non-integer value."""
    v = np.asarray(view)
    u = v.astype(np.uint8)
    if not (v.min() >= 0 and v.max() <= 255 and np.array_equal(u.astype(v.dtype), v)):
        raise ValueError('model view is not exactly uint8-representable')
    return u


def _diff_frames(v):
    """Frame-difference loop of diff_normalize_data (float64 arithmetic stored into float32)."""
    n, h, w, c = v.shape
    d = np.zeros((n - 1, h, w, c), dtype=np.float32)
    for j in range(n - 1):
        d[j, :, :, :] = (v[j + 1, :, :, :] - v[j, :, :, :]) / (v[j + 1, :, :, :] + v[j, :, :, :] + 1e-07)
    return d


def pack(backbone, view_u8, off, length):
    """(kept uint8 frames, context statistics or None) from the full 300-frame uint8 view."""
    v = np.asarray(view_u8)
    assert v.dtype == np.uint8 and len(v) == 300
    if backbone == 'FPm':
        return np.ascontiguousarray(v[off:off + length]), None
    full = v.astype(np.float64)
    if backbone == 'PhysNet':
        stats = np.asarray([np.std(_diff_frames(full))], dtype=np.float32)
        return np.ascontiguousarray(v[off:min(off + length + 1, 300)]), stats
    mean = np.mean(full)
    stats = np.asarray([mean, np.std(full - mean)], dtype=np.float64)
    return np.ascontiguousarray(v[off:off + length]), stats


def model_input(backbone, view_u8, raw_label, off, length, stats=None):
    """(x, y) float32 exactly as the archived generators saved them (before the dataset transpose)."""
    v = np.asarray(view_u8).astype(np.float64)
    label = np.asarray(raw_label)
    if backbone == 'PhysNet':
        y = diff_normalize_label(label)
        if stats is None:  # older full-context form
            x = diff_normalize_data(v)[off:off + length]
        else:
            x = _diff_frames(v) / np.float32(stats[0])
            if off + length == 300:
                x = np.append(x, np.zeros((1,) + x.shape[1:], dtype=np.float32), axis=0)
            x[np.isnan(x)] = 0
    elif backbone == 'RhythmFormer':
        y = standardized_label(label)
        if stats is None:
            x = standardized_data(v)[off:off + length]
        else:
            x = (v - np.float64(stats[0])) / np.float64(stats[1])
            x[np.isnan(x)] = 0
    else:
        x, y = v, standardized_label(label)
    assert len(x) == length
    lo = 0 if len(label) == length else off  # (i)/(i-pc) items store the chunk-length label itself (brief 02 6.12)
    return (np.ascontiguousarray(x, dtype=np.float32),
            np.ascontiguousarray(y[lo:lo + length], dtype=np.float32))


def npy_sha(arr):
    """SHA-256 of the bytes np.save would write (same as the manifest's file SHA for an .npy)."""
    b = io.BytesIO()
    np.save(b, arr)
    return hashlib.sha256(b.getvalue()).hexdigest()


def data_sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def item_input(backbone, r):
    stats = read_array(r['ops2_stats_path']) if r.get('ops2_stats_path') else None
    # ops2_view_offset: offset of the chunk within the stored view when it is not the context offset. (i-pc-g80) items
    # store the 128-frame chunk itself (full-chunk DiffNormalized, generator train/gen_ipc_g80.py, offset 0), while their
    # rows keep the (i) item's context_offset. Rows without the column (all other pools) are unchanged.
    off = r.get('ops2_view_offset')
    off = int(float(off)) if off not in (None, '') else int(r['context_offset'])
    return model_input(backbone, read_array(r['ops2_view_path']), read_array(r['ops2_rawlabel_path']),
                       off, int(r['chunk_length']), stats)


# ---------- training loader: archived datasets/streams, only the pool item source differs ----------
class U8Tensors:
    """Map-style dataset: archived StoredTensors (PhysNet, FPm) / RhythmTensors (RhythmFormer) item layout."""

    def __init__(self, backbone, original, pool):
        self.backbone, self.original, self.pool = backbone, original, pool

    def __len__(self):
        return len(self.original) + len(self.pool)

    def __getitem__(self, index):
        import torch
        kind, number = index
        r = (self.original if kind == 0 else self.pool)[number]
        if kind == 1 and r.get('ops2_view_path'):
            x, y = item_input(self.backbone, r)
        elif kind == 1 and r.get('ops2_label_path'):  # (i-pc) PhysNet/RhythmFormer: stored (i) input, new label file
            x = np.load(r['input_path'], allow_pickle=False)
            y = np.load(r['ops2_label_path'], allow_pickle=False)
        else:
            x = np.load(r['input_path'], allow_pickle=False)
            y = np.load(r['label_path'], allow_pickle=False)
        if self.backbone == 'RhythmFormer':  # src/st12c/rhythm_data.py RhythmTensors
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)
        else:  # src/st9t/data.py StoredTensors
            x = np.ascontiguousarray(x.transpose(3, 0, 1, 2), dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        if self.backbone == 'PhysNet' and os.environ.get('OPS2_LABEL_REPR') == 'std_cumsum':
            # brief 02 6.45 다 3 (PhysNet-Std): Standardized label = cumulative sum of the DiffNormalized label,
            # standardized over the chunk (equals the raw label one sample later, standardized; audit/std_label_check.py)
            c = np.cumsum(y.astype(np.float64))
            y = np.ascontiguousarray((c - c.mean()) / c.std(), dtype=np.float32)
        high = float(r['reference_HR'] if kind == 0 else r['target_HR']) >= 100
        return torch.from_numpy(x), torch.from_numpy(y), kind, number, int(high)


def loader(backbone, original, pool, seed, arm, steps, workers):
    import torch
    from functools import partial
    from src.st9s.streams import seed_worker
    from src.st9t.data import FixedBatches
    generator = torch.Generator()
    generator.manual_seed(seed)
    weights = None if backbone == 'RhythmFormer' else [float(r['LDS_weight']) for r in original]
    return torch.utils.data.DataLoader(U8Tensors(backbone, original, pool),
                                       batch_sampler=FixedBatches(len(original), len(pool), seed, weights, arm, steps),
                                       num_workers=workers, worker_init_fn=partial(seed_worker, seed=seed),
                                       generator=generator, pin_memory=True, prefetch_factor=2, persistent_workers=True)
