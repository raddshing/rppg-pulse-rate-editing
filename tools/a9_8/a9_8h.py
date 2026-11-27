"""OPS2 A9 item 8h (7.13 1): harmonic content of the edited pulse in the first PURE pool of (ii-pc). No face pixels are used.

Items: every item of the first PURE (ii-pc) pool of each backbone (OPS2 manifests cache/pools/<backbone>_polarity_explicit_all_
manifest.csv; PhysNet 2,972, RhythmFormer and matched FactorizePhys 2,384 items). For each item, the archived editor function
ppg_vectors (src/st9e/inherited.py, unchanged; as train/gen_pool.py via audit/editor_pc.injection_wave) gives, from the contact
PPG of the recording (src.st9e contact, PURE), the context start, the reference HR h0 of the chunk, the target h* and the
recording's tau: g* (the target waveform: contact PPG resampled by h*/h0 and filtered/normalized by the editor) and g0 (the
contact PPG of the chunk, filtered and normalized as g* but not resampled), 300 samples at 30 Hz.
Ratio: power within +-0.15 Hz of 2f over power within +-0.15 Hz of f; f = f* = h*/60 for g*, f = f0 = h0/60 for g0.
Power: scipy periodogram (Hann window, mean removed, nfft 4096, power spectrum), summed over the bins in each band.
Summary: per backbone and target interval (70-80, 80-90, ..., 130-140 bpm; by h*): items, median and quartiles of both ratios;
and over all items. Writes outputs/a9_8/8h/8h_harmonics.csv; item rows server-only in cache/a9_8/8h_items.csv.gz.
"""
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import periodogram  # noqa: E402

WORK = __import__('pathlib').Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'audit'))
import editor_pc as E  # noqa: E402  (a1_common: archived src.st9e helpers, contact, tau_table, pure_index)

OUT = WORK / 'outputs' / 'a9_8' / '8h'
BB = {'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer', 'FPm': 'FactorizePhys (matched)'}


def ratio(x, f):
    fr, p = periodogram(np.asarray(x, float), fs=30, window='hann', nfft=4096, detrend='constant', scaling='spectrum')
    p1 = p[np.abs(fr - f) <= .15].sum()
    p2 = p[np.abs(fr - 2 * f) <= .15].sum()
    return float(p2 / p1) if p1 > 0 else np.nan


def main():
    t0 = time.time()
    taus = E.tau_table()
    idx = E.pure_index()
    rows = []
    for b in BB:
        m = pd.read_csv(WORK / 'cache' / 'pools' / f'{b}_polarity_explicit_all_manifest.csv')
        m = m[m.generation_excluded.astype(str).ne('True')] if 'generation_excluded' in m else m
        for rid, g in m.groupby('recording_id'):
            gt = E.contact(idx[rid], 'PURE', None)
            tau = int(taus[rid]['tau_rec'])
            for r in g.itertuples():
                cs, h0, target = int(r.context_start), float(r.reference_HR), float(r.target_HR)
                g0, gstar = E.ppg_vectors(gt, cs, h0, target, tau)
                rows.append(dict(backbone=BB[b], recording_id=rid, chunk_id=r.chunk_id, augmentation_index=int(r.augmentation_index),
                                 h0=h0, target_HR=target, ratio_gstar=ratio(gstar, target / 60), ratio_g0=ratio(g0, h0 / 60)))
        print(b, len(m), round(time.time() - t0), flush=True)
    items = pd.DataFrame(rows)
    (WORK / 'cache' / 'a9_8').mkdir(parents=True, exist_ok=True)
    items.to_csv(WORK / 'cache' / 'a9_8' / '8h_items.csv.gz', index=False)
    items['interval'] = [f'{int(lo)}-{int(lo) + 10}' for lo in np.clip(np.floor(items.target_HR / 10) * 10, 70, 130)]
    out = []
    for (bb, iv), z in list(items.groupby(['backbone', 'interval'])) + [((bb, 'all'), z) for bb, z in items.groupby('backbone')]:
        r = dict(dataset='PURE (first pool, (ii-pc) items)', backbone=bb, target_interval=iv, items=len(z),
                 chunks=z.chunk_id.nunique(), recordings=z.recording_id.nunique(), target_min=float(z.target_HR.min()), target_max=float(z.target_HR.max()))
        for k, lab in (('ratio_gstar', 'gstar_2fstar_over_fstar'), ('ratio_g0', 'g0_2f0_over_f0')):
            v = z[k].dropna()
            r.update({f'{lab}_median': float(v.median()), f'{lab}_q1': float(v.quantile(.25)), f'{lab}_q3': float(v.quantile(.75)),
                      f'{lab}_nan': int(z[k].isna().sum())})
        out.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).sort_values(['backbone', 'target_interval']).to_csv(OUT / '8h_harmonics.csv', index=False)
    print('done', len(items), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
