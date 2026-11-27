"""OPS2 A9 items 8a and 8e (7.13 1, second): VIPL-HR primary analysis without the gt_HR 255 windows, and reference agreement.

8a. The primary analysis of eval5/vipl_c.py stage (c1) (confirmatory arms (0), (i), (i-pc), (ii), (ii-pc), seeds
    20261131-20261133, primary variants; primary set, HAR, verified windows without the 200-ms flag, reference = gt_HR mean;
    stage3_analyze.contrast_rows unchanged: confirmatory support, same-seed pairs, crossed bootstrap B = 10,000) without the
    two windows whose mean gt_HR is 255.0 bpm (one recording pair, p66 v6 source1/source3, window 2; both in the primary
    analysis as high-HR windows). C1-C4, every metric of the contrast family; intervals, no tiers (A9 header). The values of
    stage (c1) (with the two windows) are joined for comparison.
8e. Windows whose mean gt_HR and BVP reference at 60 Hz (A8: eval5/vipl_a8.py refs, BVP placed at 60 Hz, common_hr HAR)
    differ by more than 10 and by more than 20 bpm. Populations: (1) the windows of the primary analysis (confirmatory
    support), (2) the A8 windows (primary set, within the 60-Hz BVP record, no 200-ms flag) with a 60-Hz reference; each
    overall and among high-HR windows (gt_HR mean >= 100; and, for (2), 60-Hz BVP reference >= 100).
Writes outputs/a9_8/8a/{8a_contrasts,8a_windows}.csv, outputs/a9_8/8e/8e_agreement.csv.
"""
import time

import a9_common as C
from a9_common import B, SA, np, pd
import vipl_c as VC


def main():
    t0 = time.time()
    d, refs, choice, bad = C.vipl_frame()
    prim = C.primary(d)
    pw = prim[prim.arm.isin(VC.CONF_ARMS) & prim.seed.isin(VC.CONF_SEEDS)]
    drop = set(refs[refs.gt_HR_mean.eq(255.0)].window_id)
    conf = [(n, a, c, 'graded') for n, a, c in SA.CONF]
    rows = SA.contrast_rows(pw[~pw.window_id.isin(drop)], conf, B, 'without_gtHR_255')
    c1 = pd.read_csv(C.VO / 'stage_c1' / 'VIPL' / 'confirmatory.csv')
    c1 = c1[c1.layer.eq('HAR')].set_index(['backbone', 'contrast', 'metric'])
    out = []
    for r in rows:
        r = dict(r)
        r.pop('grade', None)
        o = c1.loc[(r['backbone'], r['contrast'], r['metric'])]
        out.append(dict(r, windows_with=int(o.windows), point_with=o.point, ci_lo_with=o.ci_lo, ci_hi_with=o.ci_hi,
                        source=f'{C.SRC_VIPL}; *_with: stage (c1) confirmatory.csv'))
    C.write(pd.DataFrame(out), '8a_contrasts.csv', '8a')
    w = refs[refs.window_id.isin(drop)][['scene', 'source', 'window_index', 'gt_HR_mean', 'h_FFT_HAR', 'verified', 'gap200']]
    sup = SA.restrict(pw)
    w = w.assign(in_primary_analysis=[x in set(sup.window_id) for x in refs[refs.window_id.isin(drop)].window_id],
                 window=w.window_index + 1, BVP_reference_per_record_rate=w.h_FFT_HAR).drop(columns=['window_index', 'h_FFT_HAR'])
    C.write(w, '8a_windows.csv', '8a')
    print('8a done', round(time.time() - t0), flush=True)
    # 8e
    a8 = pd.read_csv(C.VO / 'A8' / 'references_60hz.csv', dtype={'subject_id': str})
    sup_w = set(sup.window_id)
    agree = []
    def rows_for(name, z, high_rule):
        z = z[np.isfinite(z.gt_HR_mean) & np.isfinite(z.reference_HR_60Hz)]
        diff = (z.gt_HR_mean - z.reference_HR_60Hz).abs()
        subsets = [('all windows', np.ones(len(z), bool))] + [(f'high-HR ({lab})', m) for lab, m in high_rule(z)]
        for sub, m in subsets:
            dd, zz = diff[m], z[m]
            agree.append(dict(dataset='VIPL-HR', population=name, subset=sub, windows=int(m.sum()), subjects=zz.subject_id.nunique(),
                              diff_gt_10bpm=int((dd > 10).sum()), diff_gt_20bpm=int((dd > 20).sum()),
                              share_gt_10bpm=float((dd > 10).mean()) if len(dd) else np.nan,
                              share_gt_20bpm=float((dd > 20).mean()) if len(dd) else np.nan,
                              subjects_gt_10bpm=zz[dd > 10].subject_id.nunique(), subjects_gt_20bpm=zz[dd > 20].subject_id.nunique(),
                              median_abs_difference=float(dd.median()) if len(dd) else np.nan,
                              median_signed_difference_gtHR_minus_BVP60=float((zz.gt_HR_mean - zz.reference_HR_60Hz).median()) if len(dd) else np.nan,
                              source='VIPL-HR stage (a) references, A8 60-Hz references'))
    p1 = a8[a8.window_id.isin(sup_w)]
    rows_for('primary analysis windows (confirmatory support)', p1, lambda z: [('gt_HR mean >= 100', z.gt_HR_mean.to_numpy() >= 100)])
    p2 = a8[a8.A8_window.astype(bool)]
    rows_for('A8 windows (primary set, within the 60-Hz BVP record, no 200-ms flag)', p2,
             lambda z: [('gt_HR mean >= 100', z.gt_HR_mean.to_numpy() >= 100), ('60-Hz BVP reference >= 100', z.reference_HR_60Hz.to_numpy() >= 100)])
    p3 = a8[a8.A8_window.astype(bool) & a8.A8_verified.astype(bool)]
    rows_for('A8 verified windows (60-Hz BVP: no defect, three estimates agree)', p3,
             lambda z: [('gt_HR mean >= 100', z.gt_HR_mean.to_numpy() >= 100), ('60-Hz BVP reference >= 100', z.reference_HR_60Hz.to_numpy() >= 100)])
    C.write(pd.DataFrame(agree), '8e_agreement.csv', '8e')
    C.stamp('8a', seconds=time.time() - t0, dropped=len(drop))
    C.stamp('8e', windows_primary=len(p1))
    print('8e done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
