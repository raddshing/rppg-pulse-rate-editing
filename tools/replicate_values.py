"""Diagnostic rule 6 (brief 02 6.67 마): paired difference in high-HR MAE between each replicate (i-simple-rep) and the
original (i-simple) run of the same seed, per backbone and seed, with the seed mean and range, on both development sets.

Inputs (read only): the guarded readouts of the original runs (outputs/guarded/1b, *_isimple_*) and of the replicates
(outputs/replicates/1b, *_isimplerep_*), produced by the same stage-2 inference code on the same GPUs (MMPD GPU1, MCD GPU0)
and the same readout code (b1/mmpd_a.py, b1/mcd_windows.py via tools/replicate_mcd_windows.py).
Populations
- MMPD-A 760: the draft's validation windows (in_Qstar of the dev readout); high = draft 'high' (as b1/analyze.py).
- MMPD-A 723: the automatic stage-1 Q* windows (rehearsal part 2 stage 1 populations.csv); high = reference HR >= 100 of
  that file, reference HR of that file (as eval5/stage3_analyze.py with --tag automatic). The dev readout equals the
  stage-3 readout for (i-simple) on every MMPD window and layer (checked before writing this tool).
- MCD-dev: draft rule (as b1/analyze.py load_mcd): front-webcam dev recordings with a confirmed mapping and >= 1 HAR window
  that is gap-free, reference-agreement and h_FFT >= 100; high = gap-free & agreement & h_FFT >= 100.
Support: per backbone, dataset and layer, the windows available in all six runs (three originals, three replicates), so every
seed is measured on the same windows. No resampling. Descriptive only; no contrast is re-judged with these values.
Usage: replicate_values.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
G = WORK / 'outputs' / 'guarded' / '1b'
R = WORK / 'outputs' / 'replicates' / '1b'
S1 = WORK / 'cache' / 'rehearsal' / 'part2' / 'stage1' / 'MMPD' / 'populations.csv'
BACKBONES = {'FPm': 'FactorizePhys (matched)', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
SEEDS = [20261131, 20261132, 20261133]


def seed_of(run_id):
    return run_id.str.extract(r'_s(\d+)$')[0].astype(int)


def common(d):
    """Windows (per layer) available in every run present."""
    n = d.run_id.nunique()
    ok = d[d.available.astype(bool)].groupby(['window_id', 'layer']).run_id.nunique()
    keep = set(ok[ok == n].index)
    return d[[k in keep for k in zip(d.window_id, d.layer)]]


def mmpd(b):
    parts = []
    for kind, f in (('original', G / f'mmpd_a_windows_isimple_{b}.csv.gz'), ('replicate', R / f'mmpd_a_windows_isimplerep_{b}.csv.gz')):
        x = pd.read_csv(f, dtype={'subject_id': str})
        x['kind'] = kind
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d['seed'] = seed_of(d.run_id)
    out = {}
    v = d[d.in_Qstar.astype(bool)].copy()
    v['set'] = np.where(v.high.astype(bool), 'high', 'low')
    out['MMPD-A 760'] = common(v)
    p = pd.read_csv(S1, dtype={'subject_id': str})
    p = p[p.in_Qstar.astype(bool)][['window_id', 'layer', 'reference_HR', 'high']]
    a = d.drop(columns=['reference_HR', 'in_Qstar', 'high']).merge(p, on=['window_id', 'layer'])
    a['set'] = np.where(a.high.astype(bool), 'high', 'low')
    out['MMPD-A 723'] = common(a)
    return out


def mcd(b):
    w = pd.read_csv(G / 'mcd_dev_windows.csv', dtype={'subject_id': str})
    recs = pd.read_csv(G / 'mcd_dev_recordings.csv', dtype={'subject_id': str})
    parts = []
    for kind, f in (('original', G / f'mcd_dev_preds_isimple_{b}.csv.gz'), ('replicate', R / f'mcd_dev_preds_isimplerep_{b}.csv.gz')):
        x = pd.read_csv(f)
        x['kind'] = kind
        parts.append(x)
    p = pd.concat(parts, ignore_index=True)
    p['seed'] = seed_of(p.run_id)
    p = p.merge(w[['window_id', 'layer', 'subject_id', 'recording_id', 'h_FFT', 'agreement', 'primary_included']],
                on=['window_id', 'layer'], how='left')
    assert p.subject_id.notna().all()
    p = p.rename(columns={'h_FFT': 'reference_HR'})
    p = p[p.primary_included.astype(bool) & np.isfinite(p.reference_HR)]
    confirmed = set(recs.recording_id[recs.mapping_confirmed.astype(bool)])
    har = w[w.layer.eq('HAR') & w.primary_included & w.agreement & w.h_FFT.ge(100)]
    d = p[p.recording_id.isin(set(har.recording_id) & confirmed)].copy()
    d['set'] = np.where(d.agreement.astype(bool) & (d.reference_HR >= 100), 'high', np.where(d.reference_HR < 100, 'low', ''))
    return {'MCD-dev': common(d[d.set.ne('')])}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for b, name in BACKBONES.items():
        for pop, d in {**mmpd(b), **mcd(b)}.items():
            for layer, g in d.groupby('layer'):
                h = g[g.set.eq('high')]
                for sd in SEEDS:
                    m = {}
                    for kind in ('original', 'replicate'):
                        x = h[h.seed.eq(sd) & h.kind.eq(kind)]
                        assert x.run_id.nunique() == 1, (b, pop, layer, sd, kind)
                        m[kind] = float(np.abs(x.predicted_HR - x.reference_HR).mean())
                        m[kind + '_windows'] = len(x)
                    assert m['original_windows'] == m['replicate_windows']
                    pr = h[h.seed.eq(sd)].pivot(index='window_id', columns='kind', values='predicted_HR')
                    rows.append(dict(backbone=name, population=pop, layer=layer, seed=sd, high_windows=m['original_windows'],
                                     high_subjects=h[h.seed.eq(sd) & h.kind.eq('original')].subject_id.nunique(),
                                     high_windows_different_prediction=int((pr.original != pr.replicate).sum()),
                                     MAE_high_original=m['original'], MAE_high_replicate=m['replicate'],
                                     difference_replicate_minus_original=m['replicate'] - m['original']))
    t = pd.DataFrame(rows)
    t.to_csv(out / 'replicate_per_seed.csv', index=False)
    s = (t.groupby(['backbone', 'population', 'layer'], sort=False)
         .agg(high_windows=('high_windows', 'first'), seeds=('seed', 'count'),
              mean_difference=('difference_replicate_minus_original', 'mean'),
              min_difference=('difference_replicate_minus_original', 'min'),
              max_difference=('difference_replicate_minus_original', 'max'),
              mean_abs_difference=('difference_replicate_minus_original', lambda x: float(np.abs(x).mean())))
         .reset_index())
    s.to_csv(out / 'replicate_summary.csv', index=False)
    print(f'replicate values: {len(t)} seed rows, {len(s)} summary rows')


if __name__ == '__main__':
    main()
