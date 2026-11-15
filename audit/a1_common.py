"""Shared setup for the 1a polarity/lag audit (PURE only, CPU only).

Imports the editor modules archived from commit 6ebbc9cd unchanged (vendor/),
with only src/st9/ops.py and src/mmpd_contract.py replaced by OPS2 stubs.
"""
import csv
import json
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
VENDOR = WORK / 'vendor' / 'st8_6ebbc9cd'
sys.path.insert(0, str(VENDOR))
os.environ['ST9_NAMESPACE'] = 'st9g'

from src.st9e.temporal import *  # noqa: E402,F401,F403  (editor, generator, native, inherited, signals)
from src.st9e.model_view import ViewWorker, crop_native  # noqa: E402,F401
from src.st9e.lag import lag_value  # noqa: E402,F401
from src.st9g.preprocess import diff_normalize_data  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = WORK / 'outputs' / '1a'
OUT.mkdir(parents=True, exist_ok=True)
LAGS = np.arange(-30, 31)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def pure_index():
    return {r['recording_id']: r for r in read_csv(VENDOR / 'data/index/pure_index_st1.csv')}


def direction():
    return json.loads((VENDOR / 'reports/st9g/st9g_pool_settings.json').read_text())['direction_RGB']


def tau_table():
    return {r['recording_id']: r for r in read_csv(VENDOR / 'outputs/st7b_tau_recordings.csv')}


def lag_curve(y, p, lags=LAGS):
    """Pearson correlation for each lag, same convention as the archived lag_value:
    lag>0 compares y(t) with p(t-lag)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        lag = int(lag)
        if lag > 0:
            a, b = y[lag:], p[:-lag]
        elif lag < 0:
            a, b = y[:lag], p[-lag:]
        else:
            a, b = y, p
        if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
            out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def summarize(c, lags=LAGS):
    c = np.asarray(c, float)
    finite = np.isfinite(c)
    if not finite.any():
        return dict(L_abs=None, sign=None, c_abs=np.nan, L_sig15=None, c_sig15=np.nan,
                    L_sig30=None, c_sig30=np.nan, margin=np.nan)
    i_abs = int(np.nanargmax(np.where(finite, np.abs(c), -np.inf)))
    s = 1 if c[i_abs] > 0 else -1
    in15 = (np.abs(lags) <= 15) & finite
    i15 = int(np.flatnonzero(in15)[np.argmax(c[in15])])
    i30 = int(np.nanargmax(np.where(finite, c, -np.inf)))
    opposite = finite & (np.sign(c) == -s)
    best_opposite = float(np.max(np.abs(c[opposite]))) if opposite.any() else 0.
    return dict(L_abs=int(lags[i_abs]), sign=s, c_abs=float(abs(c[i_abs])),
                L_sig15=int(lags[i15]), c_sig15=float(c[i15]), L_sig30=int(lags[i30]), c_sig30=float(c[i30]),
                margin=float(abs(c[i_abs]) - best_opposite))


def view_signal(view):
    return editor_band(green(view.mean(axis=(1, 2), dtype=np.float64)))


def write_rows(path, records):
    records = list(records)
    fields = []
    for r in records:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
