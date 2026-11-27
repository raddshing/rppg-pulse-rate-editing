"""OPS2 A9 item 8f (7.13 1): subject-weighted MAE on the three test sets.

Test sets and windows: MMPD-B (Q* windows) and the MCD-rPPG test half (draft-rule windows) of the test step (c) primary
analysis (tools/posthoc701/p701_common.load, eval5/stage3_analyze.py, wide band HAR, primary variants), and the VIPL-HR primary
analysis (eval5/vipl_c.py: primary set, HAR, verified, no 200-ms flag, reference = gt_HR mean); confirmatory support and valid
windows; per arm the windows valid for every key of the arm, per contrast those valid for every key of both arms.
Subject-weighted MAE: per seed, the mean over subjects of the subject's MAE (high: subjects with a high-HR window, their
high-HR windows; low and all likewise), i.e. every subject weighted equally. Intervals: the crossed subject x seed bootstrap of
b1/boot.py (B = 10,000, default_rng(20261110)) applied to the per-subject MAE (a resampled subject counts as often as drawn).
Per arm ((0), (i), (i-pc), (ii), (ii-pc), (i-simple), (ii-lag); seeds 20261131-20261133) and for C1-C4 (same-seed pairs).
Share of subjects whose high-HR MAE is lower with (ii-pc) than with (0): per seed over the subjects with high-HR windows in
the C1 windows, seed mean, the same bootstrap.
The window-weighted MAE of the same windows is computed by the same code as a check (stored arm and contrast tables).
Writes outputs/a9_8/8f/{8f_arms,8f_contrasts,8f_share,8f_checks}.csv.
"""
import time

import a9_common as C
from a9_common import B, SA, SEEDS, boot, np, pd
from p701_common import TEST, load, main_population, primary_har

ARMS = ['(0)', '(i)', '(i-pc)', '(ii)', '(ii-pc)', '(i-simple)', '(ii-lag)']
NAMES = {'MMPD': 'MMPD-B', 'MCD': 'MCD-rPPG test', 'VIPL': 'VIPL-HR'}


def subject_rows(g):
    """Per seed and subject: MAE on high, low and all windows (rows with set = high/low/all, value in predicted_HR)."""
    e = g.assign(err=(g.predicted_HR - g.reference_HR).abs())
    parts = []
    for name, m in (('high', e.set.eq('high')), ('low', e.set.eq('low')), ('all', np.ones(len(e), bool))):
        a = e[m].groupby(['seed', 'subject_id']).agg(predicted_HR=('err', 'mean'), windows=('err', 'size')).reset_index()
        parts.append(a.assign(set=name, reference_HR=0.0, window_id=a.subject_id + '|' + name))
    return pd.concat(parts, ignore_index=True)


def m_sw(W, r, p, s):
    return {f'SW_MAE_{n}': (boot._wmean(W[:, s == n], p[s == n]) if (s == n).any() else np.full(len(W), np.nan))
            for n in ('high', 'low', 'all')}


def m_share(W, r, p, s):
    return {'share_subjects_lower_high_MAE': boot._wmean(W, p) if len(p) else np.full(len(W), np.nan)}


def frames():
    for ds in ('MMPD', 'MCD'):
        d, bad = load(TEST, ds)
        pw = primary_har(d)
        yield NAMES[ds], SA.restrict(pw[pw.population.eq(main_population(ds))])
    d, refs, choice, bad = C.vipl_frame()
    yield 'VIPL-HR', SA.restrict(C.primary(d))


def main():
    t0 = time.time()
    arms, cons, share, chk = [], [], [], []
    for ds, r in frames():
        r = r[r.seed.isin(SEEDS) & r.strategy.isin(ARMS)]
        for bb in sorted(r.backbone.unique()):
            for st in ARMS:
                g, nw = C.arm_windows(r[r.backbone.eq(bb) & r.strategy.eq(st)])
                if g.empty:
                    continue
                sr = subject_rows(g)
                pt, ci = boot.point_ci(sr, SEEDS, m_sw, B)
                one = g[g.seed.eq(SEEDS[0])]
                for k in pt:
                    n = k.rsplit('_', 1)[1]
                    arms.append(dict(dataset=ds, backbone=bb, strategy=st, windows=nw, subjects=int(sr[sr.seed.eq(SEEDS[0]) & sr.set.eq(n)].subject_id.nunique()),
                                     windows_in_set=int(one.set.eq(n).sum()) if n != 'all' else len(one), metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1]))
            for name, a, c in SA.CONF:
                ga, gc = r[r.backbone.eq(bb) & r.strategy.eq(a)], r[r.backbone.eq(bb) & r.strategy.eq(c)]
                ga, gc, nw = SA.pair_windows(ga, gc)
                sa, sc = subject_rows(ga), subject_rows(gc)
                pt, ci = boot.contrast_ci(sa, sc, SEEDS, m_sw, B)
                for k in pt:
                    n = k.rsplit('_', 1)[1]
                    cons.append(dict(dataset=ds, backbone=bb, name=name, contrast=f'{a}-{c}', windows=nw,
                                     subjects=int(sa[sa.seed.eq(SEEDS[0]) & sa.set.eq(n)].subject_id.nunique()), metric=k,
                                     point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1]))
                ptw, ciw = boot.contrast_ci(ga, gc, SEEDS, boot.m123, B)
                chk.append(dict(dataset=ds, backbone=bb, item=f'{name} window-weighted MAE_high (check value)', windows=nw,
                                point=ptw['MAE_high'], ci_lo=ciw['MAE_high'][0], ci_hi=ciw['MAE_high'][1]))
                if name == 'C1':
                    hs, h0 = sa[sa.set.eq('high')], sc[sc.set.eq('high')]
                    m = hs.merge(h0, on=['seed', 'subject_id'], suffixes=('', '_0'))
                    z = m.assign(predicted_HR=(m.predicted_HR < m.predicted_HR_0).astype(float), set='high')
                    pt, ci = boot.point_ci(z, SEEDS, m_share, B)
                    per_seed = z.groupby('seed').predicted_HR.agg(['mean', 'sum', 'size'])
                    share.append(dict(dataset=ds, backbone=bb, contrast='(ii-pc) vs (0), high-HR MAE per subject', windows=nw,
                                      subjects=int(per_seed['size'].iloc[0]), point=pt['share_subjects_lower_high_MAE'],
                                      ci_lo=ci['share_subjects_lower_high_MAE'][0], ci_hi=ci['share_subjects_lower_high_MAE'][1],
                                      per_seed=';'.join(f'{sd}:{int(x["sum"])}/{int(x["size"])}' for sd, x in per_seed.iterrows())))
            print(ds, bb, round(time.time() - t0), flush=True)
    C.write(pd.DataFrame(arms), '8f_arms.csv', '8f')
    C.write(pd.DataFrame(cons), '8f_contrasts.csv', '8f')
    C.write(pd.DataFrame(share), '8f_share.csv', '8f')
    C.write(pd.DataFrame(chk), '8f_checks.csv', '8f')
    C.stamp('8f', seconds=time.time() - t0)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
