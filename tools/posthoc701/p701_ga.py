"""Brief 02 7.01 가: level-shift controls, offset and affine recalibration of arm (0) (post hoc; existing outputs only).

Fit (per backbone and seed, on the guarded development predictions of arm (0), wide band HAR, primary variant):
  windows: the development windows of the paper's development analysis (rehearsal part 2 step (c): MMPD-A automatic Q*
  windows, 723; MCD-rPPG development half draft-rule windows, the 1,449 windows of the constant predictor 97.78, of which the
  windows with a valid prediction (confirmatory support and the key's own valid windows) are used); no selection by
  reference HR.
  offset  b = mean(reference - predicted)                       -> predicted + b
  affine  reference = a * predicted + c (least squares, polyfit)  -> a * predicted + c
Application: the same seed's (0) test predictions (MMPD-A fit on MMPD-B, MCD-dev fit on the MCD-rPPG test half), window HR
values only (no spectrum is read again). The calibrated arm keeps the windows and validity of (0).
Metrics (test step (c) code unchanged: eval5/stage3_analyze.py arm_rows / contrast_rows, b1/boot.py crossed bootstrap,
B = 10,000, default_rng(20261110)): arms (0), (0)+offset, (0)+affine, (i-simple), (i-pc), (ii-pc); MCD-rPPG also MAE over
the verified windows (MAE_verified) and the within-person change of the M7 population (eval5/families.py fam_m7: M7_MAE,
M7_slope, M7_CCC). Contrasts (same-seed pairs, same windows): (ii-pc) and (i-simple) minus (0)+offset and (0)+affine;
(ii-pc)-(0) and (i-simple)-(0) are recomputed as a check against the test outputs. No grades (post hoc).
Writes outputs/posthoc701/ga/{ga_fits,ga_arms,ga_contrasts,ga_m7,ga_checks}.csv.
"""
import sys
import time

from p701_common import (B, DEV, SA, SEEDS, SRC_DEV, SRC_TEST, TEST, load, main_population, np, pd, primary_har, stamp,
                         write)

ARMS = ['(0)', '(0)+offset', '(0)+affine', '(i-simple)', '(i-pc)', '(ii-pc)']
CONTRASTS = [('(ii-pc)', '(0)+offset'), ('(ii-pc)', '(0)+affine'), ('(i-simple)', '(0)+offset'), ('(i-simple)', '(0)+affine'),
             ('(ii-pc)', '(0)'), ('(i-simple)', '(0)')]
NAMES = {'MMPD': ('MMPD-A', 'MMPD-B'), 'MCD': ('MCD-rPPG dev', 'MCD-rPPG test')}


def fits(ds):
    d, bad = load(DEV, ds)
    z = primary_har(d)
    z = z[z.population.eq(main_population(ds)) & z.arm.eq('(0)')]
    rows = []
    for (bb, sd), g in z.groupby(['backbone', 'seed']):
        n_pop = g.window_id.nunique()
        g = SA.restrict(g)
        r, p = g.reference_HR.to_numpy(float), g.predicted_HR.to_numpy(float)
        assert np.isfinite(r).all() and np.isfinite(p).all()
        b = float(np.mean(r - p))
        a, c = (float(v) for v in np.polyfit(p, r, 1))
        rows.append(dict(dataset_fit=NAMES[ds][0], dataset_applied=NAMES[ds][1], backbone=bb, seed=int(sd), population_windows=n_pop,
                         fit_windows=len(g), high_windows=int((g.set == 'high').sum()), subjects=g.subject_id.nunique(),
                         offset_b=b, affine_a=a, affine_c=c, mean_reference=float(r.mean()), mean_predicted=float(p.mean()),
                         dev_MAE_0=float(np.abs(p - r).mean()), dev_MAE_offset=float(np.abs(p + b - r).mean()),
                         dev_MAE_affine=float(np.abs(a * p + c - r).mean()), excluded_keys=len(bad), source=SRC_DEV))
    return pd.DataFrame(rows)


def calibrated(prim, fit):
    z = prim[prim.arm.eq('(0)')]
    out = []
    for kind in ('offset', 'affine'):
        parts = []
        for (bb, sd), g in z.groupby(['backbone', 'seed']):
            f = fit[fit.backbone.eq(bb) & fit.seed.eq(sd)].iloc[0]
            g = g.copy()
            g['predicted_HR'] = g.predicted_HR + f.offset_b if kind == 'offset' else f.affine_a * g.predicted_HR + f.affine_c
            parts.append(g)
        y = pd.concat(parts)
        y['arm'] = y['strategy'] = f'(0)+{kind}'
        y['key'] = y['run_id'] = y.key + '|' + kind
        out.append(y)
    return pd.concat(out)


def checks(ds, arms, con, m7):
    """Uncalibrated arms and contrasts against the test step (c) outputs (same code; must be equal)."""
    out = []
    folder = TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves')
    ref = pd.read_csv(folder / 'arms.csv')
    ref = ref[ref.layer.eq('HAR')]
    for r in arms[arms.strategy.isin(['(0)', '(i-simple)', '(i-pc)', '(ii-pc)'])].itertuples():
        o = ref[ref.backbone.eq(r.backbone) & ref.strategy.eq(r.strategy) & ref.metric.eq(r.metric)]
        if len(o):
            o = o.iloc[0]
            out.append(dict(dataset=ds, table='arms', backbone=r.backbone, item=r.strategy, metric=r.metric,
                            max_abs_difference=float(np.nanmax(np.abs([r.point - o.point, r.ci_lo - o.ci_lo, r.ci_hi - o.ci_hi])))))
    for name, f in (('confirmatory', 'confirmatory.csv'), ('secondary', 'secondary.csv')):
        ref = pd.read_csv(folder / f)
        ref = ref[ref.layer.eq('HAR')]
        for r in con[con.contrast.isin(['(ii-pc)-(0)', '(i-simple)-(0)'])].itertuples():
            o = ref[ref.backbone.eq(r.backbone) & ref.contrast.eq(r.contrast) & ref.metric.eq(r.metric)]
            if len(o):
                o = o.iloc[0]
                out.append(dict(dataset=ds, table=name, backbone=r.backbone, item=r.contrast, metric=r.metric,
                                max_abs_difference=float(np.nanmax(np.abs([r.point - o.point, r.ci_lo - o.ci_lo, r.ci_hi - o.ci_hi])))))
    if m7 is not None:
        ref = pd.read_csv(folder / 'm7.csv')
        ref = ref[ref.layer.eq('HAR')]
        for r in m7[m7.strategy.isin(['(0)', '(i-simple)', '(i-pc)', '(ii-pc)'])].itertuples():
            o = ref[ref.backbone.eq(r.backbone) & ref.strategy.eq(r.strategy) & ref.metric.eq(r.metric)]
            if len(o):
                o = o.iloc[0]
                out.append(dict(dataset=ds, table='m7', backbone=r.backbone, item=r.strategy, metric=r.metric,
                                max_abs_difference=float(np.nanmax(np.abs([r.point - o.point, r.ci_lo - o.ci_lo, r.ci_hi - o.ci_hi])))))
    return out


def main():
    t0 = time.time()
    datasets = sys.argv[1:] or ['MMPD', 'MCD']
    all_fit, all_arms, all_con, all_m7, all_chk = [], [], [], [], []
    for ds in datasets:
        fit = fits(ds)
        all_fit.append(fit)
        print(ds, 'fits done', round(time.time() - t0), flush=True)
        d, bad = load(TEST, ds)
        prim = primary_har(d)
        prim = pd.concat([prim, calibrated(prim, fit)], ignore_index=True)
        prim = prim[prim.arm.isin(ARMS)]
        pw = prim[prim.population.eq(main_population(ds))]
        arms = pd.DataFrame(SA.arm_rows(pw, B, 'posthoc'))
        arms['dataset'] = NAMES[ds][1]
        arms['source'] = np.where(arms.strategy.str.contains(r'\+'), f'{SRC_TEST}; calibration from {SRC_DEV}', SRC_TEST)
        print(ds, 'arms done', round(time.time() - t0), flush=True)
        con = pd.DataFrame(SA.contrast_rows(pw, [('', a, c, 'values only') for a, c in CONTRASTS], B, 'posthoc'))
        con['dataset'] = NAMES[ds][1]
        con['source'] = np.where(con.contrast.str.contains(r'\+'), f'{SRC_TEST}; calibration from {SRC_DEV}', SRC_TEST)
        con['check_only'] = ~con.contrast.str.contains(r'\+')
        print(ds, 'contrasts done', round(time.time() - t0), flush=True)
        m7 = None
        if ds == 'MCD':
            m7 = pd.DataFrame(SA.AN.fam_m7(SA.restrict(prim[prim.population.eq('M7_both_states')]), [], B))
            m7['dataset'] = NAMES[ds][1]  # fam_m7 writes a fixed 'MCD-dev' label
            m7['source'] = np.where(m7.strategy.str.contains(r'\+'), f'{SRC_TEST}; calibration from {SRC_DEV}', SRC_TEST)
            all_m7.append(m7)
            print(ds, 'm7 done', round(time.time() - t0), flush=True)
        all_arms.append(arms)
        all_con.append(con)
        all_chk += checks(ds, arms, con, m7)
    write(pd.concat(all_fit), 'ga_fits.csv', 'ga')
    write(pd.concat(all_arms), 'ga_arms.csv', 'ga')
    write(pd.concat(all_con), 'ga_contrasts.csv', 'ga')
    if all_m7:
        write(pd.concat(all_m7), 'ga_m7.csv', 'ga')
    chk = pd.DataFrame(all_chk)
    write(chk, 'ga_checks.csv', 'ga')
    print('checks', len(chk), 'max abs difference', chk.max_abs_difference.max() if len(chk) else None)
    stamp('ga', datasets=datasets, seconds=time.time() - t0, checks=len(chk),
          checks_max_abs_difference=float(chk.max_abs_difference.max()) if len(chk) else None)


if __name__ == '__main__':
    main()
