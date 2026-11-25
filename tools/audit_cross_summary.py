"""Brief 02 6.97 나: recording table and corpus summary of the cross-corpus natural audit (tools/audit_cross.py outputs).

Recording values follow audit/a1_analyze.py: the context curves of a recording are averaged (nanmean) and summarized
(audit/a1_common.py summarize: |corr| maximum over lags -30..+30 with its sign; signed maximum within -15..+15 and
within -30..+30); the recording HR is the median of the context HRs (common_hr 'HAR', 30 Hz assumed by the code) and
the half period is 900 / HR frames. Seconds use the recording's frame rate; source_HR_bpm = HR_code * fps / 30.
Corpus summary: recording bootstrap (2,000 resamples, seed 20261117) of the OLS slope, 95% percentile interval.
Also checks that the PURE re-run reproduces the 1a outputs.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
from a1_common import LAGS, summarize, write_rows, read_csv  # noqa: E402

OUTX = WORK / 'outputs' / 'audit_cross'
CORPORA = [('PURE', 'PURE'), ('UBFC', 'UBFC-rPPG'), ('MMPD', 'MMPD-A'), ('MCD', 'MCD-rPPG dev')]
B, SEED = 2000, 20261117


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def boot_slope(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    s = np.array([slope(x[i], y[i]) if np.ptp(x[i]) > 0 else np.nan for i in idx])
    return slope(x, y), float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))


def q(x):
    x = np.asarray(x, float)
    if not len(x):
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def recordings(corpus, label):
    ctx = read_csv(OUTX / f'{corpus}_contexts.csv')
    meta = {r['recording_id']: r for r in read_csv(OUTX / f'{corpus}_recordings_meta.csv')}
    z = np.load(OUTX / f'{corpus}_curves.npz')
    assert list(z['lags']) == list(LAGS)
    curves = dict(zip(z['keys'], z['curves']))
    out, serial = [], {}
    for k, rid in enumerate(sorted(meta), 1):
        m = meta[rid]
        rc = [r for r in ctx if r['recording_id'] == rid]
        ok = [r for r in rc if r['status'] == 'ok']
        public_id = rid if corpus in ('PURE', 'UBFC') else f'{label.split()[0]}_{k:03d}'
        serial[public_id] = rid
        row = dict(corpus=label, recording=public_id, fps=float(m['fps']) if m.get('fps') else np.nan,
                   contexts_ok=len(ok), contexts_mask_failure=sum(r['status'] == 'mask_failure' for r in rc),
                   contexts_ppg_unavailable=sum(r['status'] == 'ppg_unavailable' for r in rc), recording_status=m['status'])
        if not ok:
            out.append(row)
            continue
        fps = row['fps']
        hr = float(np.median([float(r['HR']) for r in ok]))
        half = 900. / hr
        s = summarize(np.nanmean([curves[f"{rid}|{r['context_index']}"] for r in ok], axis=0))
        row.update(HR_code_bpm=hr, source_HR_bpm=hr * fps / 30., half_period_frames=half, half_period_s=half / fps,
                   tau_signed_max_lag_frames=s['L_sig15'], tau_signed_max_lag_s=s['L_sig15'] / fps,
                   signed_max_corr=s['c_sig15'],
                   abs_max_lag_frames=s['L_abs'], abs_max_lag_s=s['L_abs'] / fps, abs_max_sign=s['sign'],
                   abs_max_corr=s['c_abs'],
                   signed_max_lag_pm30_frames=s['L_sig30'], signed_max_corr_pm30=s['c_sig30'],
                   abs_max_at_search_edge=abs(s['L_abs']) == 30, signed_max_at_search_edge=abs(s['L_sig15']) == 15)
        out.append(row)
    return out, serial


def summary(label, rows, rng):
    ok = [r for r in rows if r['contexts_ok'] > 0]
    neg = [r for r in ok if r['abs_max_sign'] == -1]
    pos = [r for r in ok if r['abs_max_sign'] == 1]
    maj, majs = (neg, -1) if len(neg) >= len(pos) else (pos, 1)
    half = [r['half_period_frames'] for r in ok]
    res = dict(corpus=label, recordings=len(ok), recordings_without_valid_context=len(rows) - len(ok),
               contexts_ok=sum(r['contexts_ok'] for r in rows),
               contexts_mask_failure=sum(r['contexts_mask_failure'] for r in rows),
               contexts_ppg_unavailable=sum(r['contexts_ppg_unavailable'] for r in rows),
               fps_median=float(np.median([r['fps'] for r in ok])), fps_min=float(min(r['fps'] for r in ok)),
               fps_max=float(max(r['fps'] for r in ok)),
               source_HR_median_bpm=float(np.median([r['source_HR_bpm'] for r in ok])),
               half_period_median_frames=float(np.median(half)),
               negative_abs_max_fraction=len(neg) / len(ok), negative_abs_max_recordings=len(neg),
               majority_sign=majs, majority_recordings=len(maj))
    for unit in ['frames', 's']:
        med, lo, hi = q([r[f'abs_max_lag_{unit}'] for r in maj])
        res.update({f'majority_abs_max_lag_median_{unit}': med, f'majority_abs_max_lag_q25_{unit}': lo,
                    f'majority_abs_max_lag_q75_{unit}': hi})
    for name, col in [('tau_signed_max_lag', 'tau_signed_max_lag'), ('abs_max_lag', 'abs_max_lag')]:
        for unit, hcol in [('frames', 'half_period_frames'), ('s', 'half_period_s')]:
            p, lo, hi = boot_slope([r[hcol] for r in ok], [r[f'{col}_{unit}'] for r in ok], rng)
            res.update({f'slope_{name}_vs_half_period_{unit}': p, f'slope_{name}_ci_lo_{unit}': lo,
                        f'slope_{name}_ci_hi_{unit}': hi})
    res['abs_max_corr_median'] = float(np.median([r['abs_max_corr'] for r in ok]))
    res['signed_max_corr_median'] = float(np.median([r['signed_max_corr'] for r in ok]))
    res['abs_max_at_search_edge_recordings'] = sum(bool(r['abs_max_at_search_edge']) for r in ok)
    res['signed_max_at_search_edge_recordings'] = sum(bool(r['signed_max_at_search_edge']) for r in ok)
    return res


def pure_check():
    new = {(r['recording_id'], r['context_index']): r for r in read_csv(OUTX / 'PURE_contexts.csv')}
    old = {(r['recording_id'], r['context_index']): r for r in read_csv(WORK / 'outputs/1a/natural_contexts.csv')
           if r['signal'] == 'face'}
    zn = np.load(OUTX / 'PURE_curves.npz')
    zo = np.load(WORK / 'outputs/1a/natural_curves.npz')
    cn = dict(zip(zn['keys'], zn['curves']))
    co = {k.rsplit('|', 1)[0]: c for k, c in zip(zo['keys'], zo['curves']) if k.endswith('|face')}
    res = dict(contexts_new=len(new), contexts_1a=len(old), same_keys=set(new) == set(old))
    diff = {}
    for f in ['HR', 'L_abs', 'sign', 'L_sig15', 'L_sig30']:
        diff[f] = sum(float(new[k][f]) != float(old[k][f]) for k in old if k in new)
    for f in ['c_abs', 'c_sig15', 'c_sig30', 'margin', 'mask_failure_fraction']:
        diff[f + '_max_abs_diff'] = max(abs(float(new[k][f]) - float(old[k][f])) for k in old if k in new)
    res['field_mismatch_counts'] = diff
    res['curve_max_abs_diff'] = float(max(np.nanmax(np.abs(cn[k] - co[k])) for k in co if k in cn))
    res['curve_nan_pattern_equal'] = all(np.array_equal(np.isnan(cn[k]), np.isnan(co[k])) for k in co if k in cn)
    return res


def main():
    rng = np.random.default_rng(SEED)
    allrows, summ, serials = [], [], {}
    for corpus, label in CORPORA:
        if not (OUTX / f'{corpus}_contexts.csv').exists():
            continue
        rows, serial = recordings(corpus, label)
        allrows += rows
        serials[label] = serial
        summ.append(summary(label, rows, rng))
    write_rows(OUTX / 'audit_cross_recordings.csv', allrows)
    write_rows(OUTX / 'audit_cross_summary.csv', summ)
    (OUTX / 'serial_to_recording_server_only.json').write_text(json.dumps(serials, indent=1))
    checks = {}
    if (OUTX / 'PURE_contexts.csv').exists():
        checks['PURE_reproduction_contexts'] = pure_check()
        a = {r['recording_id']: r for r in read_csv(WORK / 'outputs/facts_697/audit_PURE.csv')}
        mism = 0
        for r in [x for x in allrows if x['corpus'] == 'PURE']:
            o = a[r['recording']]
            mism += sum([int(o['tau_signed_max_lag_frames']) != r['tau_signed_max_lag_frames'],
                         int(o['abs_max_lag_frames']) != r['abs_max_lag_frames'], int(o['abs_max_sign']) != r['abs_max_sign'],
                         abs(float(o['abs_max_corr']) - r['abs_max_corr']) > 1e-12,
                         abs(float(o['signed_max_corr']) - r['signed_max_corr']) > 1e-12,
                         abs(float(o['source_HR_bpm']) - r['HR_code_bpm']) > 1e-12])
        checks['PURE_reproduction_recordings'] = dict(recordings=len(a), mismatched_fields=mism)
    (OUTX / 'checks.json').write_text(json.dumps(checks, indent=1, default=float))
    print(json.dumps(checks, indent=1, default=float))
    for s in summ:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == '__main__':
    main()
