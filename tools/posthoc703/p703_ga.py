"""Brief 02 7.03 가: MAE-optimal level calibration of arm (0) as a diagnostic control (post hoc; existing outputs only).

Fit data: as 7.01 가 (tools/posthoc701/p701_ga.py fits, same lines): per backbone and seed, the guarded development
predictions of arm (0), wide band (HAR), primary variant; MMPD-A automatic Q* windows (723) for MMPD-B, MCD-rPPG development
half draft-rule windows (1,449, of which those with a valid prediction) for the MCD-rPPG test half; no selection by HR.
  (1) median offset  b = median(reference - predicted)                         -> predicted + b
  (2) LAD affine     (a, c) = argmin sum |reference - (a predicted + c)|       -> a predicted + c
      (linear program, scipy.optimize.linprog HiGHS; a vertex solution, i.e. a line through at least two fit points)
  7.01, reproduced with the same lines: mean offset b = mean(reference - predicted); least-squares affine (numpy polyfit).
Application: the same seed's (0) test predictions (window HR values only), windows and validity of (0) kept.
Metrics (test step (c) code unchanged: eval5/stage3_analyze.py arm_rows / contrast_rows, eval5/families.py fam_m7,
b1/boot.py crossed bootstrap B = 10,000, default_rng(20261110)): arms (0), the four calibrated (0), (i-simple), (i-pc),
(ii-pc); contrasts (ii-pc) and (i-simple) minus each calibrated (0) (same-seed pairs, same windows); MCD-rPPG test half:
within-person change (M7_MAE, M7_slope, M7_CCC). No grades.
Checks: mean-offset and least-squares rows against the stored 7.01 outputs (fits, arms, contrasts, m7); uncalibrated arms
and contrasts against the test step (c) outputs (7.01 checks function).
Writes outputs/posthoc703/ga/{ga_fits,ga_arms,ga_contrasts,ga_m7,ga_checks}.csv.
"""
import sys
import time

from scipy import sparse
from scipy.optimize import linprog

from p703_common import B, DEV, NAMES, SA, SRC_DEV, SRC_TEST, TEST, WORK, load, main_population, np, pd, primary_har, stamp, write
sys.path.insert(0, str(WORK / 'tools' / 'posthoc701'))
import p701_ga as G1  # noqa: E402  (7.01 checks against the test step (c) outputs)

KINDS = ['offset', 'affine', 'median', 'LAD']
ARMS = ['(0)'] + [f'(0)+{k}' for k in KINDS] + ['(i-simple)', '(i-pc)', '(ii-pc)']
CONTRASTS = [(a, f'(0)+{k}') for k in KINDS for a in ('(ii-pc)', '(i-simple)')] + [('(ii-pc)', '(0)'), ('(i-simple)', '(0)')]
FIT_NAMES = {'MMPD': ('MMPD-A', 'MMPD-B'), 'MCD': ('MCD-rPPG dev', 'MCD-rPPG test')}
OLD = WORK / 'outputs' / 'posthoc701' / 'ga'


def lad(p, r):
    n = len(p)
    A = sparse.hstack([sparse.csr_matrix(np.c_[p, np.ones(n)]), sparse.identity(n), -sparse.identity(n)]).tocsr()
    c = np.r_[0., 0., np.ones(2 * n)]
    res = linprog(c, A_eq=A, b_eq=r, bounds=[(None, None)] * 2 + [(0, None)] * (2 * n), method='highs')
    assert res.status == 0, res.message
    a, b = float(res.x[0]), float(res.x[1])
    return a, b, float(res.fun), res.message


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
        bm = float(np.median(r - p))
        la, lc, obj, msg = lad(p, r)
        res = r - (la * p + lc)
        rows.append(dict(dataset_fit=FIT_NAMES[ds][0], dataset_applied=FIT_NAMES[ds][1], backbone=bb, seed=int(sd),
                         population_windows=n_pop, fit_windows=len(g), high_windows=int((g.set == 'high').sum()),
                         subjects=g.subject_id.nunique(), offset_b=b, affine_a=a, affine_c=c, median_b=bm, LAD_a=la, LAD_c=lc,
                         mean_reference=float(r.mean()), mean_predicted=float(p.mean()),
                         median_reference=float(np.median(r)), median_predicted=float(np.median(p)),
                         dev_MAE_0=float(np.abs(p - r).mean()), dev_MAE_offset=float(np.abs(p + b - r).mean()),
                         dev_MAE_affine=float(np.abs(a * p + c - r).mean()), dev_MAE_median=float(np.abs(p + bm - r).mean()),
                         dev_MAE_LAD=float(np.abs(la * p + lc - r).mean()), LAD_objective_over_n=obj / len(g),
                         LAD_zero_residual_points=int((np.abs(res) < 1e-7).sum()), LAD_solver=f'linprog highs: {msg}',
                         excluded_keys=len(bad), source=SRC_DEV))
    return pd.DataFrame(rows)


def calibrated(prim, fit):
    z = prim[prim.arm.eq('(0)')]
    out = []
    for kind in KINDS:
        parts = []
        for (bb, sd), g in z.groupby(['backbone', 'seed']):
            f = fit[fit.backbone.eq(bb) & fit.seed.eq(sd)].iloc[0]
            g = g.copy()
            if kind == 'offset':
                g['predicted_HR'] = g.predicted_HR + f.offset_b
            elif kind == 'affine':
                g['predicted_HR'] = f.affine_a * g.predicted_HR + f.affine_c
            elif kind == 'median':
                g['predicted_HR'] = g.predicted_HR + f.median_b
            else:
                g['predicted_HR'] = f.LAD_a * g.predicted_HR + f.LAD_c
            parts.append(g)
        y = pd.concat(parts)
        y['arm'] = y['strategy'] = f'(0)+{kind}'
        y['key'] = y['run_id'] = y.key + '|' + kind
        out.append(y)
    return pd.concat(out)


def diff(new, old, on, table):
    m = new.merge(old, on=on, suffixes=('', '_701'))
    vals = [c for c in ('point', 'ci_lo', 'ci_hi', 'offset_b', 'affine_a', 'affine_c') if c in new.columns and f'{c}_701' in m.columns]
    mx = float(np.nanmax(np.abs(np.concatenate([(m[c] - m[f'{c}_701']).to_numpy(float) for c in vals])))) if len(m) else np.nan
    return dict(table=table, rows_compared=len(m), rows_701=len(old), max_abs_difference=mx)


def main():
    t0 = time.time()
    datasets = sys.argv[1:] or ['MMPD', 'MCD']
    all_fit, all_arms, all_con, all_m7, chk = [], [], [], [], []
    for ds in datasets:
        fit = fits(ds)
        all_fit.append(fit)
        print(ds, 'fits done', round(time.time() - t0), flush=True)
        d, bad = load(TEST, ds)
        prim = primary_har(d)
        prim = pd.concat([prim, calibrated(prim, fit)], ignore_index=True)
        prim = prim[prim.arm.isin(ARMS)]
        pw = prim[prim.population.eq(main_population(ds))]
        src = lambda s: np.where(s.str.contains(r'\+'), f'{SRC_TEST}; calibration from {SRC_DEV}', SRC_TEST)  # noqa: E731
        arms = pd.DataFrame(SA.arm_rows(pw, B, 'posthoc'))
        arms['dataset'] = NAMES[ds]
        arms['source'] = src(arms.strategy)
        print(ds, 'arms done', round(time.time() - t0), flush=True)
        con = pd.DataFrame(SA.contrast_rows(pw, [('', a, c, 'values only') for a, c in CONTRASTS], B, 'posthoc'))
        con['dataset'] = NAMES[ds]
        con['source'] = src(con.contrast)
        con['check_only'] = ~con.contrast.str.contains(r'\+')
        print(ds, 'contrasts done', round(time.time() - t0), flush=True)
        m7 = None
        if ds == 'MCD':
            m7 = pd.DataFrame(SA.AN.fam_m7(SA.restrict(prim[prim.population.eq('M7_both_states')]), [], B))
            m7['dataset'] = NAMES[ds]  # fam_m7 writes a fixed 'MCD-dev' label
            m7['source'] = src(m7.strategy)
            all_m7.append(m7)
            print(ds, 'm7 done', round(time.time() - t0), flush=True)
        all_arms.append(arms)
        all_con.append(con)
        chk += [{**c, 'dataset': NAMES[ds], 'comparison': 'test step (c) outputs'} for c in G1.checks(ds, arms, con, m7)]
    fit, arms, con = pd.concat(all_fit), pd.concat(all_arms), pd.concat(all_con)
    m7 = pd.concat(all_m7) if all_m7 else None
    write(fit, 'ga_fits.csv', 'ga')
    write(arms, 'ga_arms.csv', 'ga')
    write(con, 'ga_contrasts.csv', 'ga')
    if m7 is not None:
        write(m7, 'ga_m7.csv', 'ga')
    # reproduction of 7.01 (mean offset, least squares): same code, same draws -> equal values
    of = pd.read_csv(OLD / 'ga_fits.csv')
    chk.append(dict(comparison='7.01 ga_fits.csv', **diff(fit, of[['dataset_fit', 'backbone', 'seed', 'offset_b', 'affine_a', 'affine_c']],
                                                       ['dataset_fit', 'backbone', 'seed'], 'fits (offset_b, affine_a, affine_c)')))
    oa = pd.read_csv(OLD / 'ga_arms.csv')
    chk.append(dict(comparison='7.01 ga_arms.csv', **diff(arms[arms.strategy.isin(oa.strategy.unique())], oa[['dataset', 'backbone', 'strategy', 'metric', 'point', 'ci_lo', 'ci_hi']],
                                                       ['dataset', 'backbone', 'strategy', 'metric'], 'arms')))
    oc = pd.read_csv(OLD / 'ga_contrasts.csv')
    chk.append(dict(comparison='7.01 ga_contrasts.csv', **diff(con[con.contrast.isin(oc.contrast.unique())], oc[['dataset', 'backbone', 'contrast', 'metric', 'point', 'ci_lo', 'ci_hi']],
                                                            ['dataset', 'backbone', 'contrast', 'metric'], 'contrasts')))
    if m7 is not None:
        om = pd.read_csv(OLD / 'ga_m7.csv')
        mm = m7[m7.strategy.isin(om.strategy.unique())].assign(seeds=lambda x: x.seeds.astype(str))
        chk.append(dict(comparison='7.01 ga_m7.csv', **diff(mm, om[['dataset', 'backbone', 'strategy', 'seeds', 'metric', 'point', 'ci_lo', 'ci_hi']].assign(seeds=lambda x: x.seeds.astype(str)),
                                                         ['dataset', 'backbone', 'strategy', 'seeds', 'metric'], 'm7')))
    chk = pd.DataFrame(chk).assign(source=f'{SRC_TEST} (recomputed here vs stored tables)')
    write(chk, 'ga_checks.csv', 'ga')
    print(chk.groupby('comparison').agg(rows=('max_abs_difference', 'size'), mx=('max_abs_difference', 'max')))
    stamp('ga', datasets=datasets, seconds=time.time() - t0, checks=len(chk),
          checks_max_abs_difference=float(chk.max_abs_difference.max()))


if __name__ == '__main__':
    main()
