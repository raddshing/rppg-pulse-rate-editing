"""Brief 02 6.97 가: export of analyses already done (no new model, no new data read).

1. audit_PURE.csv: the 59 PURE recordings of the 1a natural audit (outputs/1a/natural_recordings.csv, face-mask signal);
   slope checks against 1a summary.json; tau - (d - T0/2) summaries.
2. pc_items_phase.csv: the 120 items of outputs/2/pc_items.csv (PhysNet and FactorizePhys views), phase error of the
   injected component (edited minus A=0 twin) against the natural relation -g*(t - d), d = 2 frames.
3. m9_dev.csv: byte copy of outputs/guarded_v2/m9.csv (development analysis, guarded predictions, 11-14).
Exploratory, post hoc, training/development data only, outside the pre-registration.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / 'outputs' / 'facts_697'
OUT.mkdir(parents=True, exist_ok=True)
D = 2  # frames, natural relation -g*(t - d) (audit/editor_pc.py D_FRAMES)


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def write(p, recs):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)


def reg(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.linregress(x, y)
    t = stats.t.ppf(.975, len(x) - 2)
    return dict(n=len(x), slope=float(r.slope), ci_low=float(r.slope - t * r.stderr), ci_high=float(r.slope + t * r.stderr))


def q(x):
    x = np.asarray(x, float)
    return dict(n=len(x), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def wrap_deg(a):
    return (np.asarray(a, float) + 180.) % 360. - 180.


def circ(deg):
    z = np.exp(1j * np.deg2rad(np.asarray(deg, float)))
    m = z.mean()
    return float(np.rad2deg(np.angle(m))), float(abs(m))


def part1():
    nat = rows(WORK / 'outputs/1a/natural_recordings.csv')
    assert len(nat) == 59
    out = []
    for r in nat:
        out.append(dict(recording_id=r['recording_id'], n_contexts=int(r['n_contexts']),
                        source_HR_bpm=float(r['HR_median']), half_period_frames=float(r['half_period_frames']),
                        tau_signed_max_lag_frames=int(r['face_L_sig15']), signed_max_corr=float(r['face_c_sig15']),
                        abs_max_lag_frames=int(r['face_L_abs']), abs_max_sign=int(r['face_sign']),
                        abs_max_corr=float(r['face_c_abs']),
                        signed_max_lag_pm30_frames=int(r['face_L_sig30']), signed_max_corr_pm30=float(r['face_c_sig30']),
                        tau_rec_pool_frames=int(r['tau_rec_file'])))
    write(OUT / 'audit_PURE.csv', out)
    half = [r['half_period_frames'] for r in out]
    s = json.loads((WORK / 'outputs/1a/summary.json').read_text())
    ctx = [r for r in rows(WORK / 'outputs/1a/natural_contexts.csv') if r['signal'] == 'face']
    ch = [900. / float(r['HR']) for r in ctx]
    chk = dict(
        table_59_recordings=dict(
            tau_vs_half=reg(half, [r['tau_signed_max_lag_frames'] for r in out]),
            abs_lag_vs_half=reg(half, [r['abs_max_lag_frames'] for r in out]),
            tau_rec_pool_vs_half=reg(half, [r['tau_rec_pool_frames'] for r in out])),
        contexts_378=dict(
            tau_vs_half=reg(ch, [int(r['L_sig15']) for r in ctx]),
            abs_lag_vs_half=reg(ch, [int(r['L_abs']) for r in ctx])),
        summary_json_contexts=dict(tau=s['between_contexts_face']['Lsig15_vs_half_period']['ols_slope'],
                                   abs=s['between_contexts_face']['Labs_vs_half_period']['ols_slope']),
        summary_json_recordings=dict(tau=s['between_recordings']['face_Lsig15_vs_half_period']['ols_slope'],
                                     abs=s['between_recordings']['face_Labs_vs_half_period']['ols_slope']))
    neg = [r for r in out if r['abs_max_sign'] == -1]
    chk['tau_minus_d_minus_half'] = {
        'tau_signed_max_lag, all 59': q([r['tau_signed_max_lag_frames'] - (D - r['half_period_frames']) for r in out]),
        'tau_rec_pool, all 59': q([r['tau_rec_pool_frames'] - (D - r['half_period_frames']) for r in out]),
        'tau_signed_max_lag, negative-sign 49': q([r['tau_signed_max_lag_frames'] - (D - r['half_period_frames']) for r in neg]),
        'tau_rec_pool, negative-sign 49': q([r['tau_rec_pool_frames'] - (D - r['half_period_frames']) for r in neg])}
    chk['abs_lag_negative_sign_49'] = q([r['abs_max_lag_frames'] for r in neg])
    chk['abs_sign_counts'] = {'+1': sum(r['abs_max_sign'] == 1 for r in out), '-1': len(neg)}
    return chk


def part2():
    items = rows(WORK / 'outputs/2/pc_items.csv')
    assert len(items) == 480
    by = {}
    for r in items:
        key = (r['chunk_id'], r['augmentation_index'])
        by.setdefault(key, {})[(r['mode'], r['view'])] = r
    keys = list(by)  # file order
    assert len(keys) == 120
    out = []
    for n, k in enumerate(keys, 1):
        g = by[k]
        a = g[('ii', 'PhysNet_view')]
        tgt, orig, tau = float(a['target_HR']), float(a['original_HR']), int(a['tau_rec'])
        f = tgt / 60.
        nat = 180. - 360. * f * D / 30.  # phase of -g*(t - d) relative to g* (phase_at convention: -2 pi f L / 30)
        design_ii = 360. * f * (D - tau) / 30. - 180.  # +g*(t - tau) relative to the natural relation
        rec = dict(item=n, source_HR_bpm=orig, target_HR_bpm=tgt, r=float(a['r']),
                   target_band='<100' if tgt < 100 else ('100-120' if tgt < 120 else '>=120'),
                   tau_rec_frames=tau, design_phase_error_ii_deg=float(wrap_deg(design_ii)))
        for view, tag in [('PhysNet_view', 'PhysNet'), ('FactorizePhys_view', 'FactorizePhys')]:
            for mode, mtag in [('ii', 'ii'), ('ii-pc', 'iipc')]:
                ph = np.rad2deg(float(g[(mode, view)]['phase_at_target']))
                rec[f'phase_error_{mtag}_{tag}_deg'] = float(wrap_deg(ph - nat))
        out.append(rec)
    write(OUT / 'pc_items_phase.csv', out)
    summ = []
    for band in ['<100', '100-120', '>=120', 'all']:
        sel = [r for r in out if band == 'all' or r['target_band'] == band]
        row = dict(target_band=band, items=len(sel))
        for col in ['phase_error_ii_PhysNet_deg', 'phase_error_iipc_PhysNet_deg', 'phase_error_ii_FactorizePhys_deg',
                    'phase_error_iipc_FactorizePhys_deg', 'design_phase_error_ii_deg']:
            m, R = circ([r[col] for r in sel])
            row[col.replace('_deg', '') + '_circular_mean_deg'] = m
            row[col.replace('_deg', '') + '_resultant_length'] = R
            row[col.replace('_deg', '') + '_n_abs_gt_90'] = sum(abs(r[col]) > 90 for r in sel)
        summ.append(row)
    write(OUT / 'pc_items_phase_by_band.csv', summ)
    return summ


def part3():
    src = WORK / 'outputs/guarded_v2/m9.csv'
    dst = OUT / 'm9_dev.csv'
    shutil.copyfile(src, dst)
    h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (src, dst)]
    assert h[0] == h[1]
    return dict(source='outputs/guarded_v2/m9.csv', sha256=h[0], bytes=dst.stat().st_size,
                rows=sum(1 for _ in open(dst)) - 1)


def main():
    res = dict(part1=part1(), part2=part2(), part3=part3())
    (OUT / 'checks.json').write_text(json.dumps(res, indent=1))
    json.dump(res, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
