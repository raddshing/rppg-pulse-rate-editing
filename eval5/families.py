"""Stage-5 evaluation: metric families of the development analyses, copied unchanged from the OPS2 development analysis
module (b1/analyze.py: constants, models, seedsets, rowify, fam_m4, m7_table, fam_m7, fam_m9, boot_wrap, circular_ci), so that
the evaluation script does not depend on the development workspace. Estimation uses evaluation/boot.py (same as b1/boot.py).
"""
import numpy as np
import pandas as pd

import boot

MAIN = [20261131, 20261132, 20261133]
FIVE = MAIN + [20261143, 20261144]
GRID = 3.515625
PURE_LABEL_MEAN = 70.80799077181209   # mean reference_HR of the 596 PURE01-08 160-frame original training chunks
PURE_LABEL_MEAN_128 = 70.80932452893674  # same over the 743 PhysNet 128-frame chunks (reported with M5)
BACKBONE = {'FPm': 'FactorizePhys (matched)', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
M8 = [('(ii)', '(0)'), ('(i)', '(0)'), ('(ii)', '(i)'), ('(iii)', '(i)'), ('(iv)', '(0)'), ('(vii)', '(0)'),
      ('(ii)', '(iv)'), ('(ii)', '(vii)'), ('(iv-d)', '(ii)'), ('(iv-d)', '(0)'), ('(ii-f)', '(ii)'),
      ('(ii-g)', '(ii)'), ('(ii-g2)', '(ii-g)'), ('(iv-d)', '(ii-g2)'), ('(0B)', '(0)'), ('(0B)', '(ii)')]
C25 = [('(ii-pc)', '(ii)'), ('(ii-pc)', '(i)'), ('(ii-pc)', '(0)')]
CINV = [('(ii-inv)', '(ii-pc)'), ('(ii-inv)', '(ii)'), ('(ii-inv)', '(0)')]  # brief 02 6.18
CLAG = [('(ii-lag)', '(ii-pc)'), ('(ii-lag)', '(ii)'), ('(ii-lag)', '(0)'), ('(ii-inv)', '(ii-lag)')]  # brief 02 6.19
CIPC = [('(i-pc)', '(i)'), ('(i-pc)', '(0)'), ('(ii-pc)', '(i-pc)')]  # brief 02 6.12
CSIMPLE = [('(i-simple)', '(i-pc)'), ('(i-simple)', '(i)'), ('(ii-pc)', '(i-simple)'), ('(i-simple)', '(0)')]  # brief 02 6.43 다
CLAB = [('(ii-pc-g80)', '(ii-pc)'), ('(ii-pc-g2)', '(ii-pc-g)'), ('(ii-pc-g)', '(ii-pc)'), ('(i-pc-g80)', '(i-pc)'),
        ('(sham-pc)', '(0)'), ('(ii-pc)', '(sham-pc)'), ('(roi-pc)', '(ii-pc)'), ('(roi-pc)', '(0)'),
        ('(ii-g)', '(ii)'), ('(ii-g2)', '(ii-g)')]  # brief 02 6.43 라 (last two: October reference arms)
# brief 02 6.37/6.42 loss swap (PhysNet with the RhythmFormer author loss): contrasts inside the loss-swap arms, the same
# contrasts under the standard loss, and the loss effect on each arm (same arm, same seed)
CRFL = [('(ii-pc-rfl)', '(ii-rfl)'), ('(i-pc-rfl)', '(i-rfl)'), ('(ii-pc-rfl)', '(i-pc-rfl)'), ('(ii-pc-rfl)', '(0)')]
CSTD = [('(ii-pc)', '(ii)'), ('(i-pc)', '(i)'), ('(ii-pc)', '(i-pc)'), ('(ii-pc)', '(0)')]
CLOSS = [('(i-rfl)', '(i)'), ('(i-pc-rfl)', '(i-pc)'), ('(ii-rfl)', '(ii)'), ('(ii-pc-rfl)', '(ii-pc)')]
PHASE_BINS = [('lt100', -np.inf, 100), ('100-120', 100, 120), ('ge120', 120, np.inf)]


def models(d, backbones):
    return [(b, s) for (b, s), _ in d.groupby(['backbone', 'strategy']) if not backbones or b in backbones]


def seedsets(g):
    have = set(g.seed)
    out = [('3', MAIN)] if set(MAIN) <= have else []
    if set(FIVE) <= have:
        out.append(('5', FIVE))
    return out


def rowify(base, point, ci):
    return [dict(base, metric=k, point=v, ci_lo=ci.get(k, (np.nan, np.nan))[0], ci_hi=ci.get(k, (np.nan, np.nan))[1])
            for k, v in point.items()]


def fam_m4(d, backbones):
    rows = []
    for (ds, pop, layer, bb, st, sd), g in d.groupby(['dataset', 'population', 'layer', 'backbone', 'strategy', 'seed']):
        if backbones and bb not in backbones:
            continue
        for lo in range(40, 180, 10):
            x = g[(g.reference_HR >= lo) & (g.reference_HR < lo + 10)]
            if x.empty:
                continue
            e = x.predicted_HR - x.reference_HR
            rows.append(dict(dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st, seed=sd, bin=f'{lo}-{lo + 10}',
                             windows=len(x), subjects=x.subject_id.nunique(), MAE=e.abs().mean(), signed_error=e.mean(),
                             hit_pm1bin=(e.abs() <= GRID + 1e-6).mean(),
                             half_HR_error=((x.predicted_HR - x.reference_HR / 2).abs() <= GRID + 1e-6).mean()))
    r = pd.DataFrame(rows)
    r = r[r.seed.isin(MAIN)]
    key = ['dataset', 'population', 'layer', 'backbone', 'strategy', 'bin']
    agg = r.groupby(key).agg(seeds=('seed', 'nunique'), windows=('windows', 'first'), subjects=('subjects', 'first'),
                             MAE=('MAE', 'mean'), signed_error=('signed_error', 'mean'), hit_pm1bin=('hit_pm1bin', 'mean'),
                             half_HR_error=('half_HR_error', 'mean')).reset_index()
    return agg


def m7_table(g):
    """Per run and subject: reference and predicted delta (after - before) of recording means."""
    rec = g.groupby(['run_id', 'seed', 'subject_id', 'state']).agg(r=('reference_HR', 'mean'), p=('predicted_HR', 'mean')).reset_index()
    w = rec.pivot_table(index=['run_id', 'seed', 'subject_id'], columns='state', values=['r', 'p'])
    t = pd.DataFrame(dict(reference_HR=w[('r', 'after')] - w[('r', 'before')], predicted_HR=w[('p', 'after')] - w[('p', 'before')])).reset_index()
    t['window_id'] = t.subject_id
    t['set'] = 'all'
    return t.dropna()


def fam_m7(m7, backbones, b):
    out = []
    for layer, dd in m7.groupby('layer'):
        for bb, st in models(dd, backbones):
            g = m7_table(dd[dd.backbone.eq(bb) & dd.strategy.eq(st)])
            for tag, seeds in seedsets(g):
                x = g[g.seed.isin(seeds)]
                pt, ci = boot.point_ci(x, seeds, boot.m7, b)
                out += rowify(dict(dataset='MCD-dev', population='M7_both_states', layer=layer, backbone=bb, strategy=st, seeds=tag,
                                   subjects=x.subject_id.nunique()), pt, ci)
    return out


def fam_m9(d, backbones, b):
    """Output phase at the reference fundamental: (0) phi itself; others delta phi = wrap(phi - phi(0))."""
    out = []
    for (ds, pop, layer), dd in d.groupby(['dataset', 'population', 'layer']):
        dd = dd[np.isfinite(dd.phase)]
        for bb in sorted(dd.backbone.unique()):
            if backbones and bb not in backbones:
                continue
            g0 = dd[dd.backbone.eq(bb) & dd.strategy.eq('(0)') & dd.seed.isin(MAIN)][['window_id', 'seed', 'phase']].rename(columns={'phase': 'phase0'})
            for st in sorted(dd[dd.backbone.eq(bb)].strategy.unique()):
                g = dd[dd.backbone.eq(bb) & dd.strategy.eq(st) & dd.seed.isin(MAIN)]
                if set(g.seed) != set(MAIN):
                    continue
                if st == '(0)':
                    g = g.assign(angle=g.phase)
                else:
                    g = g.merge(g0, on=['window_id', 'seed'])
                    g = g.assign(angle=boot_wrap(g.phase - g.phase0))
                for name, lo, hi in PHASE_BINS:
                    x = g[(g.reference_HR >= lo) & (g.reference_HR < hi)]
                    if x.empty:
                        continue
                    res = circular_ci(x, b)
                    out.append(dict(dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st,
                                    quantity='phi' if st == '(0)' else 'delta_phi_vs_(0)', HR_bin=name,
                                    windows=len(x) // 3, subjects=x.subject_id.nunique(), **res,
                                    pearson_lag0_median=float(np.nanmedian(x.pearson_lag0))))
    return out


def boot_wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def circular_ci(x, b):
    subjects = boot.numeric_subject_order(x.subject_id)
    idx = {s: i for i, s in enumerate(subjects)}
    per = {}
    for sd in MAIN:
        y = x[x.seed == sd]
        per[sd] = (np.array([idx[s] for s in y.subject_id]), np.exp(1j * y.angle.to_numpy(float)))

    def z(W):
        return {sd: (W[:, si] @ e) / W[:, si].sum(1) for sd, (si, e) in per.items()}
    full = z(np.ones((1, len(subjects))))
    zp = np.mean([full[sd][0] for sd in MAIN])
    res = dict(circular_mean=float(np.angle(zp)), resultant_length=float(abs(zp)))
    if b == 0:
        return res
    isub, iseed = boot.draws(len(subjects), len(MAIN), b)
    zr = np.empty(b, complex)
    for a in range(0, b, 1000):
        W = boot.subject_weights(isub[a:a + 1000], len(subjects))
        with np.errstate(invalid='ignore', divide='ignore'):
            v = z(W)
        stack = np.stack([v[sd] for sd in MAIN])
        zr[a:a + 1000] = np.nanmean(stack[iseed[a:a + 1000].T, np.arange(W.shape[0])[None, :]], axis=0)
    dev = boot_wrap(np.angle(zr) - res['circular_mean'])
    res.update(circular_mean_ci_lo=float(res['circular_mean'] + np.nanpercentile(dev, 2.5)),
               circular_mean_ci_hi=float(res['circular_mean'] + np.nanpercentile(dev, 97.5)),
               resultant_length_ci_lo=float(np.nanpercentile(abs(zr), 2.5)), resultant_length_ci_hi=float(np.nanpercentile(abs(zr), 97.5)))
    return res


