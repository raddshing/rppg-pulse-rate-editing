"""7.09b E1: UBFC original chunk tensor from the per-recording uint8 view (same arithmetic as train/gen_ubfc.original_tensor).

Light module for the training interpreter (numpy, train/u8_store.py and the archived transforms only).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import u8_store  # noqa: E402  (adds the vendor path)
from src.st9g.preprocess import diff_normalize_label, standardized_label  # noqa: E402


def original_tensor(backbone, view_u8, stats, gt, start, length):
    """(x, y) float32 of one original chunk from the whole-recording uint8 view, statistics and contact PPG."""
    n = len(view_u8)
    if backbone == 'PhysNet':
        v = view_u8[start:min(start + length + 1, n)].astype(np.float64)
        x = u8_store._diff_frames(v) / np.float32(stats[0])
        if start + length == n:
            x = np.append(x, np.zeros((1,) + x.shape[1:], dtype=np.float32), axis=0)
        x[np.isnan(x)] = 0
        y = diff_normalize_label(gt)[start:start + length]
    elif backbone == 'RhythmFormer':
        x = (view_u8[start:start + length].astype(np.float64) - np.float64(stats[0])) / np.float64(stats[1])
        x[np.isnan(x)] = 0
        y = standardized_label(gt)[start:start + length]
    else:
        x = view_u8[start:start + length].astype(np.float64)
        y = standardized_label(gt)[start:start + length]
    assert len(x) == length and len(y) == length
    return np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(y, dtype=np.float32)
