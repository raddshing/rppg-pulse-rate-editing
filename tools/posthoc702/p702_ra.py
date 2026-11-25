"""Brief 02 7.02 라: multiple-comparison sensitivity of the 24 confirmatory contrasts (post hoc; saved test step (c) outputs only).

C1-C4 x three backbones x two test sets, high-HR MAE difference, windows and pairing of the test step (c) confirmatory analysis
(eval5/stage3_analyze.py contrast_rows: confirmatory support, windows valid for every key of both arms, same-seed pairs).
Crossed subject x seed bootstrap of b1/boot.py (draws default_rng(20261110) on the numeric subject order) with B = 100,000:
  Bonferroni interval: two-sided 1 - 0.05/24 (99.79%), percentiles 100*(0.05/48) and 100*(1 - 0.05/48)
  95% interval of the same 100,000 replicates (for reference)
Grades (test rule, eval5/stage3_analyze.py grade): T1 upper < 0; T2 point < 0 and the interval contains 0; T3 point >= 0 and the
interval contains 0; T4 lower > 0. The original grade is the test step (c) value (95%, B = 10,000, confirmatory.csv); the
same contrast is recomputed with B = 10,000 as a check against confirmatory.csv.
Writes outputs/posthoc702/ra/{ra_bonferroni,ra_checks}.csv.
"""
import warnings

from p702_common import (CONF, NAMES, SA, SEEDS, SRC_TEST, TEST, boot, load, main_population, np, pd, primary_har, stamp, time, write)

B_BIG = 100_000
M = 24
Q_LO, Q_HI = 100 * 0.05 / (2 * M), 100 * (1 - 0.05 / (2 * M))


def replicates(ga, gc, b, chunk=500):
    """b1/boot.py contrast_ci / summarize, MAE_high only, returning the point and every replicate value."""
    subjects = boot.numeric_subject_order(set(ga.subject_id) | set(gc.subject_id))
    sa, sc = boot.Sample(ga, subjects, SEEDS), boot.Sample(gc, subjects, SEEDS)

    def f(W):
        va, vc = boot.per_seed(sa, boot.m1, W), boot.per_seed(sc, boot.m1, W)
        return {sd: va[sd]['MAE_high'] - vc[sd]['MAE_high'] for sd in SEEDS}
    full = f(np.ones((1, len(subjects))))
    point = float(np.nanmean([full[sd][0] for sd in SEEDS]))
    isub, iseed = boot.draws(len(subjects), len(SEEDS), b)
    reps = np.empty(b)
    for a in range(0, b, chunk):
        Wsub = boot.subject_weights(isub[a:a + chunk], len(subjects))
        vals = f(Wsub)
        stack = np.stack([vals[sd] for sd in SEEDS])
        chosen = stack[iseed[a:a + chunk].T, np.arange(Wsub.shape[0])[None, :]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            reps[a:a + chunk] = np.nanmean(chosen, axis=0)
    return point, reps, len(subjects)


def main():
    t0 = time.time()
    rows, chk = [], []
    for ds in ('MMPD', 'MCD'):
        d, bad = load(TEST, ds)
        pw = primary_har(d)
        pw = SA.restrict(pw[pw.population.eq(main_population(ds))])
        ref = pd.read_csv(TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves') / 'confirmatory.csv')
        ref = ref[ref.layer.eq('HAR') & ref.metric.eq('MAE_high')]
        for bb in sorted(pw.backbone.unique()):
            for name, a, c in CONF:
                ga, gc = pw[pw.backbone.eq(bb) & pw.strategy.eq(a)], pw[pw.backbone.eq(bb) & pw.strategy.eq(c)]
                ga, gc, nw = SA.pair_windows(ga[ga.seed.isin(SEEDS)], gc[gc.seed.isin(SEEDS)])
                o = ref[ref.backbone.eq(bb) & ref.name.eq(name)].iloc[0]
                p10, r10, ns = replicates(ga, gc, boot.B)
                lo10, hi10 = float(np.nanpercentile(r10, 2.5)), float(np.nanpercentile(r10, 97.5))
                chk.append(dict(dataset=NAMES[ds], backbone=bb, name=name, windows=nw, windows_test=int(o.windows),
                                max_abs_difference=float(np.max(np.abs([p10 - o.point, lo10 - o.ci_lo, hi10 - o.ci_hi]))),
                                grade_recomputed=SA.grade(p10, lo10, hi10), grade_test=o.grade))
                pt, reps, ns = replicates(ga, gc, B_BIG)
                lo95, hi95 = float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))
                lob, hib = float(np.nanpercentile(reps, Q_LO)), float(np.nanpercentile(reps, Q_HI))
                rows.append(dict(dataset=NAMES[ds], backbone=bb, name=name, contrast=f'{a}-{c}', metric='MAE_high', windows=nw,
                                 windows_high=int(ga[ga.seed.eq(SEEDS[0])].set.eq('high').sum()), subjects=ns, point=pt,
                                 ci95_lo_test=o.ci_lo, ci95_hi_test=o.ci_hi, grade_95_test=o.grade,
                                 ci95_lo_B100000=lo95, ci95_hi_B100000=hi95, grade_95_B100000=SA.grade(pt, lo95, hi95),
                                 bonferroni_lo=lob, bonferroni_hi=hib, grade_bonferroni=SA.grade(pt, lob, hib),
                                 grade_changed=SA.grade(pt, lob, hib) != o.grade, nan_replicates=int(np.isnan(reps).sum()),
                                 percentiles=f'{Q_LO:.6f};{Q_HI:.6f}', B=B_BIG, source=SRC_TEST))
                print(ds, bb, name, 'done', round(time.time() - t0), flush=True)
    chk = pd.DataFrame(chk).assign(source=f'{SRC_TEST} (recomputed here vs stored test step (c) confirmatory.csv)')
    write(pd.DataFrame(rows), 'ra_bonferroni.csv', 'ra')
    write(chk, 'ra_checks.csv', 'ra')
    print('checks', len(chk), 'max', chk.max_abs_difference.max())
    stamp('ra', seconds=time.time() - t0, B_big=B_BIG, contrasts=M, percentiles=[Q_LO, Q_HI], checks=len(chk),
          checks_max_abs_difference=float(chk.max_abs_difference.max()))


if __name__ == '__main__':
    main()
