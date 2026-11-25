"""Brief 02 7.02 가: hit rates, half-HR rates and bin-balanced MAE on the test sets (post hoc; saved test step (c) outputs only).

Windows: the test step (c) primary analysis (eval5/stage3_analyze.py, unchanged): wide band (HAR), primary variant,
MMPD-B Q* windows (high = reference HR >= 100), MCD-rPPG test half draft-rule windows (high = agreement and >= 100,
low = < 100); confirmatory support and valid windows.
Hit: |readout - reference| <= 3.515625 + 1e-6 bpm; half HR: |readout - reference/2| <= 3.515625 + 1e-6 bpm (the test code's
hit_pm1bin and half_HR_error, eval5/families.py fam_m4). Rates are window means on the high, low and all windows.
Arms (0), (i), (ii), (i-pc), (ii-pc), (i-simple), (ii-lag) x three backbones: seed mean and 95% interval, windows valid for
every key of the arm (eval5/stage3_analyze.py arm_rows). Contrasts (front - back; positive = the front arm hits more):
C1-C4, (ii-pc)-(i-simple), (i-simple)-(0), (i-pc)-(0), (ii-lag)-(ii-pc); same-seed pairs on the windows valid for every
key of both arms (pair_windows); crossed subject x seed bootstrap (b1/boot.py, B = 10,000, default_rng(20261110)).
The same bootstrap function also gives the MAE of the same windows, compared with the test outputs (arms.csv,
confirmatory.csv, secondary.csv) as a check.
Bin-balanced MAE (descriptive, no interval): reference HR bins of 10 bpm ([40, 50), [50, 60), ...); per arm and seed the MAE of
each bin, seed mean per bin; mean over the bins with >= 20 windows: all such bins, and the high-HR bins (>= 100 bpm) only.
Writes outputs/posthoc702/ga/{ga_arms,ga_contrasts,ga_balanced,ga_balanced_bins,ga_checks}.csv.
"""
import sys

from p702_common import (B, CONF, NAMES, SA, SEEDS, SRC_TEST, TEST, TOL, GRID, boot, half, hit, load, main_population, np, pd,
                         primary_har, stamp, time, write)

ARMS = ['(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)', '(i-simple)', '(ii-lag)']
PAIRS = [(n, a, c) for n, a, c in CONF] + [('', '(ii-pc)', '(i-simple)'), ('', '(i-simple)', '(0)'), ('', '(i-pc)', '(0)'),
                                           ('', '(ii-lag)', '(ii-pc)')]
MIN_BIN = 20


def m_hit(W, r, p, s):
    out = {}
    for name, mask in [('high', s == 'high'), ('low', s == 'low'), ('all', np.ones(len(r), bool))]:
        Wm, rr, pp = W[:, mask], r[mask], p[mask]
        if not mask.any():
            for k in ('hit_rate', 'half_HR_rate', 'MAE'):
                out[f'{k}_{name}'] = np.full(len(W), np.nan)
            continue
        out[f'hit_rate_{name}'] = boot._wmean(Wm, hit(rr, pp))
        out[f'half_HR_rate_{name}'] = boot._wmean(Wm, half(rr, pp))
        out[f'MAE_{name}'] = boot.mae(Wm, rr, pp)
    return out


def arm_windows(g):
    g = g[g.seed.isin(SEEDS)]
    k = g.key.nunique()
    v = g.groupby(['window_id', 'layer']).key.nunique()
    w = set(v[v == k].index)
    return g[[x in w for x in zip(g.window_id, g.layer)]], len(w)


def balanced(g, nw):
    """Per 10-bpm reference bin: seed-mean MAE; means over the bins with >= MIN_BIN windows."""
    g = g.assign(bin_lo=(np.floor(g.reference_HR / 10) * 10).astype(int))
    per = (g.assign(err=(g.predicted_HR - g.reference_HR).abs()).groupby(['bin_lo', 'seed']).err.mean()
           .groupby('bin_lo').mean())
    counts = g[g.seed.eq(SEEDS[0])].groupby('bin_lo').size()
    rows = [dict(bin=f'{lo}-{lo + 10}', bin_lo=lo, windows=int(counts[lo]), MAE_seed_mean=float(per[lo]),
                 used=bool(counts[lo] >= MIN_BIN), high_bin=bool(lo >= 100)) for lo in per.index]
    used = [r for r in rows if r['used']]
    hi = [r for r in used if r['high_bin']]
    summary = dict(windows=nw, bins_used_all=len(used), bins_used_high=len(hi),
                   balanced_MAE_all=float(np.mean([r['MAE_seed_mean'] for r in used])) if used else np.nan,
                   balanced_MAE_high=float(np.mean([r['MAE_seed_mean'] for r in hi])) if hi else np.nan,
                   bins_all=';'.join(f"{r['bin']}:{r['windows']}" for r in used),
                   bins_high=';'.join(f"{r['bin']}:{r['windows']}" for r in hi),
                   windows_in_used_bins_all=sum(r['windows'] for r in used), windows_in_used_bins_high=sum(r['windows'] for r in hi),
                   bins_dropped=';'.join(f"{r['bin']}:{r['windows']}" for r in rows if not r['used']))
    return summary, rows


def main():
    t0 = time.time()
    datasets = sys.argv[1:] or ['MMPD', 'MCD']
    arms_out, con_out, bal_out, bins_out, chk = [], [], [], [], []
    for ds in datasets:
        d, bad = load(TEST, ds)
        pw = primary_har(d)
        pw = pw[pw.population.eq(main_population(ds))]
        r = SA.restrict(pw)
        assert (r.set.eq('high') == (r.reference_HR >= 100)).all(), 'high set differs from reference >= 100'
        # definition check: the test code's per-bin hit and half-HR rates (fam_m4 on all primary windows of the population)
        m4 = pd.read_csv(TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves') / 'm4_bins.csv')
        m4 = m4[m4.layer.eq('HAR')].set_index(['backbone', 'strategy', 'bin'])
        own = SA.AN.fam_m4(r, []).set_index(['backbone', 'strategy', 'bin'])
        mine = []
        for (bb, st, sd), g in r[r.seed.isin(SEEDS)].groupby(['backbone', 'strategy', 'seed']):
            g = g.assign(bin_lo=(np.floor(g.reference_HR / 10) * 10).astype(int))
            for lo, x in g.groupby('bin_lo'):
                if 40 <= lo < 180:
                    mine.append(dict(backbone=bb, strategy=st, seed=sd, bin=f'{lo}-{lo + 10}', h=hit(x.reference_HR.to_numpy(), x.predicted_HR.to_numpy()).mean(),
                                     f=half(x.reference_HR.to_numpy(), x.predicted_HR.to_numpy()).mean()))
        mine = pd.DataFrame(mine).groupby(['backbone', 'strategy', 'bin'])[['h', 'f']].mean()
        j = m4.join(own[['hit_pm1bin', 'half_HR_error']], rsuffix='_own').join(mine)
        chk.append(dict(dataset=NAMES[ds], table='m4_bins (hit_pm1bin, half_HR_error)', rows=len(j),
                        max_abs_difference=float(np.nanmax(np.abs(np.r_[j.hit_pm1bin - j.hit_pm1bin_own, j.half_HR_error - j.half_HR_error_own,
                                                                          j.hit_pm1bin - j.h, j.half_HR_error - j.f])))))
        ref_arms = pd.read_csv(TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves') / 'arms.csv')
        ref_arms = ref_arms[ref_arms.layer.eq('HAR')]
        for bb in sorted(r.backbone.unique()):
            for st in ARMS:
                g, nw = arm_windows(r[r.backbone.eq(bb) & r.strategy.eq(st)])
                pt, ci = boot.point_ci(g, SEEDS, m_hit, B)
                one = g[g.seed.eq(SEEDS[0])]
                for k in pt:
                    arms_out.append(dict(dataset=NAMES[ds], backbone=bb, strategy=st, windows=nw, windows_high=int(one.set.eq('high').sum()),
                                         windows_low=int(one.set.eq('low').sum()), subjects=g.subject_id.nunique(), metric=k,
                                         point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], source=SRC_TEST))
                    if k.startswith('MAE'):
                        o = ref_arms[ref_arms.backbone.eq(bb) & ref_arms.strategy.eq(st) & ref_arms.metric.eq(k)].iloc[0]
                        chk.append(dict(dataset=NAMES[ds], table='arms', backbone=bb, item=st, metric=k, windows=nw, windows_test=int(o.windows),
                                        max_abs_difference=float(np.nanmax(np.abs([pt[k] - o.point, ci[k][0] - o.ci_lo, ci[k][1] - o.ci_hi])))))
                s, rows = balanced(g, nw)
                bal_out.append(dict(dataset=NAMES[ds], backbone=bb, strategy=st, **s, source=SRC_TEST))
                bins_out += [dict(dataset=NAMES[ds], backbone=bb, strategy=st, **x, source=SRC_TEST) for x in rows]
            print(ds, bb, 'arms done', round(time.time() - t0), flush=True)
        refc = pd.concat([pd.read_csv(TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves') / f) for f in ('confirmatory.csv', 'secondary.csv')])
        refc = refc[refc.layer.eq('HAR')]
        for bb in sorted(r.backbone.unique()):
            for name, a, c in PAIRS:
                ga, gc = r[r.backbone.eq(bb) & r.strategy.eq(a)], r[r.backbone.eq(bb) & r.strategy.eq(c)]
                ga, gc, nw = SA.pair_windows(ga[ga.seed.isin(SEEDS)], gc[gc.seed.isin(SEEDS)])
                pt, ci = boot.contrast_ci(ga, gc, SEEDS, m_hit, B)
                one = ga[ga.seed.eq(SEEDS[0])]
                for k in pt:
                    con_out.append(dict(dataset=NAMES[ds], backbone=bb, name=name, contrast=f'{a}-{c}', windows=nw,
                                        windows_high=int(one.set.eq('high').sum()), windows_low=int(one.set.eq('low').sum()),
                                        subjects=ga.subject_id.nunique(), metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], source=SRC_TEST))
                    if k.startswith('MAE'):
                        o = refc[refc.backbone.eq(bb) & refc.contrast.eq(f'{a}-{c}') & refc.metric.eq(k)].iloc[0]
                        chk.append(dict(dataset=NAMES[ds], table='confirmatory' if name else 'secondary', backbone=bb, item=f'{a}-{c}', metric=k,
                                        windows=nw, windows_test=int(o.windows),
                                        max_abs_difference=float(np.nanmax(np.abs([pt[k] - o.point, ci[k][0] - o.ci_lo, ci[k][1] - o.ci_hi])))))
            print(ds, bb, 'contrasts done', round(time.time() - t0), flush=True)
    chk = pd.DataFrame(chk).assign(source=f'{SRC_TEST} (recomputed here vs stored test step (c) tables)')
    write(pd.DataFrame(arms_out), 'ga_arms.csv', 'ga')
    write(pd.DataFrame(con_out), 'ga_contrasts.csv', 'ga')
    write(pd.DataFrame(bal_out), 'ga_balanced.csv', 'ga')
    write(pd.DataFrame(bins_out), 'ga_balanced_bins.csv', 'ga')
    write(chk, 'ga_checks.csv', 'ga')
    print('checks', len(chk), 'max abs difference', chk.max_abs_difference.max())
    stamp('ga', datasets=datasets, seconds=time.time() - t0, checks=len(chk), checks_max_abs_difference=float(chk.max_abs_difference.max()),
          hit_rule=f'|readout - reference| <= {GRID} + {TOL}', min_bin_windows=MIN_BIN)


if __name__ == '__main__':
    main()
