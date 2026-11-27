"""OPS2 A9 item 8g (7.13 1, first): VIPL-HR arm-level values of every checkpoint evaluated on it when A9 was committed
(stages (b1), (b2): 178 checkpoints, seeds 20261131-20261133), and an oracle constant predictor.

Windows: the primary analysis of eval5/vipl_c.py (primary set, wide band HAR, primary variants, verified windows without
the 200-ms flag, reference = gt_HR mean; confirmatory support and each key's valid windows); per arm the windows valid for
every key of the arm (stage3_analyze.arm_rows). high = reference >= 100 bpm, low = < 100.
Measures (tools/posthoc702/p702_ga.m_hit, unchanged): MAE high/low/all; hit rate (|readout - reference| <= 3.515625 bpm,
one bin) and half-HR rate (|readout - reference/2| <= 3.515625 bpm) on high/low/all windows. Per arm: seed mean and 95%
interval (crossed subject x seed bootstrap, B = 10,000, default_rng(20261110)). Per checkpoint: the plain window means
on the arm's windows (no interval).
Oracle constant predictor: every window of the primary analysis (confirmatory support) is given the mean, or the median,
of the reference HR of those windows; the same measures, with a subject bootstrap interval (same draws, one 'seed').
Check: MAE of the confirmatory arms against stage (c1) arms.csv.
Recording-level scores in the benchmark convention are written by a9_8g_rec.py (they need the BVP record).
Writes outputs/a9_8/8g/{8g_arms,8g_checkpoints,8g_constant,8g_checks}.csv.
"""
import time

import a9_common as C
from a9_common import B, SA, SEEDS, boot, np, pd
from p702_ga import m_hit


def main():
    t0 = time.time()
    d, refs, choice, bad = C.vipl_frame()
    pw = SA.restrict(C.primary(d))
    arms, per_key, chk = [], [], []
    c1 = pd.read_csv(C.VO / 'stage_c1' / 'VIPL' / 'arms.csv')
    c1 = c1[c1.layer.eq('HAR') & c1.population.eq('primary')]
    for (bb, st), g in pw.groupby(['backbone', 'strategy']):
        if set(SEEDS) - set(g.seed):
            continue
        g, nw = C.arm_windows(g)
        pt, ci = boot.point_ci(g, SEEDS, m_hit, B)
        one = g[g.seed.eq(SEEDS[0])]
        for k in pt:
            arms.append(dict(dataset='VIPL-HR', population='primary', backbone=bb, strategy=st, key_set=g.key_set.iloc[0], windows=nw,
                             windows_high=int(one.set.eq('high').sum()), windows_low=int(one.set.eq('low').sum()),
                             subjects=g.subject_id.nunique(), metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], source=C.SRC_VIPL))
            o = c1[c1.backbone.eq(bb) & c1.strategy.eq(st) & c1.metric.eq(k)]
            if len(o):
                o = o.iloc[0]
                chk.append(dict(table='stage (c1) arms.csv', backbone=bb, strategy=st, metric=k, windows=nw, windows_c1=int(o.windows),
                                max_abs_difference=float(np.nanmax(np.abs([pt[k] - o.point, ci[k][0] - o.ci_lo, ci[k][1] - o.ci_hi])))))
        for (key, sd), x in g.groupby(['key', 'seed']):
            r, p = x.reference_HR.to_numpy(), x.predicted_HR.to_numpy()
            hi, lo = x.set.eq('high').to_numpy(), x.set.eq('low').to_numpy()
            row = dict(dataset='VIPL-HR', backbone=bb, strategy=st, seed=int(sd), key=key, windows=len(x), windows_high=int(hi.sum()),
                       windows_low=int(lo.sum()), subjects=x.subject_id.nunique())
            for name, m in (('high', hi), ('low', lo), ('all', np.ones(len(x), bool))):
                row[f'MAE_{name}'] = float(np.abs(p[m] - r[m]).mean()) if m.any() else np.nan
                row[f'hit_rate_{name}'] = float(C.hit(r[m], p[m]).mean()) if m.any() else np.nan
                row[f'half_HR_rate_{name}'] = float(C.half(r[m], p[m]).mean()) if m.any() else np.nan
            per_key.append(row)
        print(bb, st, round(time.time() - t0), flush=True)
    # oracle constant predictor on the windows of the primary analysis
    w = pw.drop_duplicates('window_id')[['window_id', 'subject_id', 'reference_HR', 'set']]
    const = []
    for name, c in (('mean', float(w.reference_HR.mean())), ('median', float(w.reference_HR.median()))):
        f = w.assign(predicted_HR=c, seed=0, key='constant')
        pt, ci = boot.point_ci(f, [0], m_hit, B)
        for k in pt:
            const.append(dict(dataset='VIPL-HR', population='primary', predictor=f'constant {name} reference HR', constant_bpm=c,
                              windows=len(w), windows_high=int(w.set.eq('high').sum()), windows_low=int(w.set.eq('low').sum()),
                              subjects=w.subject_id.nunique(), metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], source=C.SRC_VIPL))
    chk = pd.DataFrame(chk)
    C.write(pd.DataFrame(arms), '8g_arms.csv', '8g')
    C.write(pd.DataFrame(per_key), '8g_checkpoints.csv', '8g')
    C.write(pd.DataFrame(const), '8g_constant.csv', '8g')
    C.write(chk, '8g_checks.csv', '8g')
    C.stamp('8g', part='window level', seconds=time.time() - t0, reference=choice, defective_keys=sorted(bad),
            checks=len(chk), checks_max_abs_difference=float(chk.max_abs_difference.max()) if len(chk) else None)
    print('done', round(time.time() - t0), 'checks', len(chk), chk.max_abs_difference.max() if len(chk) else None)


if __name__ == '__main__':
    main()
