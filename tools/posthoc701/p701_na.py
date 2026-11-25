"""Brief 02 7.01 나: contrasts by motion condition on MMPD-B (post hoc; saved test step (c) outputs only).

Labels and cells: the supplementary S18 subgroup analysis (eval5/stage3_analyze.py mmpd_labels and subgroup_rows,
unchanged): the MMPDLoader motion label of the original data (activity: Stationary, Rotation, Talking, Walking) from the
sort keys of the test input plan (st30e_B); Q* windows, wide band (HAR), primary variant, confirmatory support and valid
windows; H = reference HR >= 100. Per backbone and condition:
  high cell = condition & H; all cell = condition (H and L).
  Contrasts (ii-pc)-(i-simple), (ii-pc)-(i-pc), (ii-pc)-(0): same-seed pairs on the windows valid for both arms
  (pair_windows), MAE difference with the crossed subject x seed bootstrap (b1/boot.py contrast_ci, B = 10,000,
  default_rng(20261110), the subjects of the cell in numeric order), as S18 (MAE over the cell's windows). The same
  estimates are computed by a copy of b1/boot.py summarize that also counts replicates with a NaN value (nan_replicates;
  a resample that draws no subject of one part), which the percentile interval ignores as in b1/boot.py.
  Metric by rule: high-HR MAE (the high cell) if the condition has >= 20 high-HR windows, otherwise all-window MAE (the all
  cell); both are written, the one chosen by the rule is marked.
  Interaction (condition - Stationary): [contrast in condition] - [contrast in Stationary] on the same resample (the
  subjects of both cells, numeric order, same draws as b1/boot.py); high-HR version when both have >= 20 high-HR windows.
Writes outputs/posthoc701/na/{na_contrasts,na_interactions,na_counts,na_checks}.csv.
"""
import time

from p701_common import (B, SA, SEEDS, SRC_TEST, TEST, boot, contrast_nan, load, np, pd, primary_har, stamp, summarize_nan,
                         write)

CONTRASTS = [('(ii-pc)', '(i-simple)'), ('(ii-pc)', '(i-pc)'), ('(ii-pc)', '(0)')]
ACTIVITIES = ['Stationary', 'Rotation', 'Talking', 'Walking']
MIN_HIGH = 20


def cell_pair(cell, a, c):
    ga = cell[cell.strategy.eq(a) & cell.seed.isin(SEEDS)]
    gc = cell[cell.strategy.eq(c) & cell.seed.isin(SEEDS)]
    if ga.empty or gc.empty or set(SEEDS) - set(ga.seed) or set(SEEDS) - set(gc.seed):
        return None
    ga, gc, nw = SA.pair_windows(ga, gc)
    return ga.assign(set='all'), gc.assign(set='all'), nw


def interaction(parts):
    """parts: [ga_cond, gc_cond, ga_stat, gc_stat] -> (contrast_cond - contrast_stat) on one resample."""
    subjects = boot.numeric_subject_order(set().union(*[set(q.subject_id) for q in parts]))
    smp = [boot.Sample(q, subjects, SEEDS) for q in parts]

    def f(W):
        v = [boot.per_seed(s, boot.m1, W) for s in smp]
        return {sd: {k: (v[0][sd][k] - v[1][sd][k]) - (v[2][sd][k] - v[3][sd][k]) for k in ('MAE_all',)} for sd in SEEDS}
    pt, ci, nan = summarize_nan(f, subjects, SEEDS, B)
    return pt['MAE_all'], ci['MAE_all'], len(subjects), nan['MAE_all']


def main():
    t0 = time.time()
    d, bad = load(TEST, 'MMPD')
    pw = primary_har(d)
    pw = pw[pw.population.eq('Qstar')]
    labels = SA.mmpd_labels(TEST / 'inputs', 'st30e_B')
    r = SA.restrict(pw).merge(labels, on='recording_id')
    assert (r.set.eq('high') == (r.reference_HR >= 100)).all()
    r['group'] = np.where(r.reference_HR >= 100, 'H', 'L')
    counts, rows, inter, chk = [], [], [], []
    for bb in sorted(r.backbone.unique()):
        x = r[r.backbone.eq(bb)]
        cells = {}
        for act in ACTIVITIES:
            for a, c in CONTRASTS:
                full = cell_pair(x[x.activity.eq(act)], a, c)
                high = cell_pair(x[x.activity.eq(act) & x.group.eq('H')], a, c)
                cells[(act, a, c)] = (full, high)
                nw_all = full[2] if full else 0
                nw_high = high[2] if high else 0
                rule = 'MAE_high' if nw_high >= MIN_HIGH else 'MAE_all'
                counts.append(dict(backbone=bb, activity=act, contrast=f'{a}-{c}', windows=nw_all, high_windows=nw_high,
                                   subjects=full[0].subject_id.nunique() if full else 0,
                                   subjects_high=high[0].subject_id.nunique() if high else 0,
                                   recordings=full[0].recording_id.nunique() if full else 0, metric_by_rule=rule))
                for metric, cp in (('MAE_high', high), ('MAE_all', full)):
                    if not cp or cp[2] == 0:
                        continue
                    ga, gc, nw = cp
                    pt, ci, nan = contrast_nan(ga, gc, SEEDS, boot.m1, B)
                    rows.append(dict(backbone=bb, activity=act, contrast=f'{a}-{c}', metric=metric, windows=nw,
                                     subjects=ga.subject_id.nunique(), point=pt['MAE_all'], ci_lo=ci['MAE_all'][0],
                                     ci_hi=ci['MAE_all'][1], nan_replicates=nan['MAE_all'], selected_by_rule=metric == rule,
                                     source=SRC_TEST))
            print(bb, act, 'done', round(time.time() - t0), flush=True)
        for act in ACTIVITIES[1:]:
            for a, c in CONTRASTS:
                (fc, hc), (fs, hs) = cells[(act, a, c)], cells[('Stationary', a, c)]
                both_high = bool(hc and hs and hc[2] >= MIN_HIGH and hs[2] >= MIN_HIGH)
                for metric, pc, ps in (('MAE_high', hc, hs), ('MAE_all', fc, fs)):
                    if not pc or not ps or pc[2] == 0 or ps[2] == 0:
                        continue
                    pt, ci, ns, nan = interaction([pc[0], pc[1], ps[0], ps[1]])
                    inter.append(dict(backbone=bb, activity=act, reference='Stationary', contrast=f'{a}-{c}', metric=metric,
                                      windows_condition=pc[2], windows_stationary=ps[2],
                                      subjects_condition=pc[0].subject_id.nunique(), subjects_stationary=ps[0].subject_id.nunique(),
                                      subjects=ns, point=pt, ci_lo=ci[0], ci_hi=ci[1], nan_replicates=nan,
                                      selected_by_rule=(metric == 'MAE_high') == both_high, source=SRC_TEST))
        print(bb, 'interactions done', round(time.time() - t0), flush=True)
    rows, inter, counts = pd.DataFrame(rows), pd.DataFrame(inter), pd.DataFrame(counts)
    # check: (ii-pc)-(0) and (ii-pc)-(i-pc) on the high cell equal the S18 table of the test run (activity x H)
    s18 = pd.read_csv(TEST / 'analysis' / 'MMPD' / 'automatic' / 'subgroup_contrasts.csv')
    s18 = s18[s18.dimension.eq('activity') & s18.group.eq('H')]
    for q in rows[rows.metric.eq('MAE_high')].itertuples():
        o = s18[s18.backbone.eq(q.backbone) & s18.category.eq(q.activity) & s18.contrast.eq(q.contrast)]
        if len(o):
            o = o.iloc[0]
            diff = [q.point - o.MAE_difference] + ([q.ci_lo - o.ci_lo, q.ci_hi - o.ci_hi] if bool(o.interval_reported) else [])
            chk.append(dict(backbone=q.backbone, activity=q.activity, contrast=q.contrast, windows=q.windows, windows_s18=int(o.windows),
                            interval_in_s18=bool(o.interval_reported), max_abs_difference=float(np.nanmax(np.abs(diff)))))
    chk = pd.DataFrame(chk)
    write(rows, 'na_contrasts.csv', 'na')
    write(inter, 'na_interactions.csv', 'na')
    write(counts, 'na_counts.csv', 'na')
    write(chk, 'na_checks.csv', 'na')
    print('S18 checks', len(chk), 'max abs difference', chk.max_abs_difference.max() if len(chk) else None)
    stamp('na', seconds=time.time() - t0, excluded_keys=sorted(bad), s18_checks=len(chk),
          s18_max_abs_difference=float(chk.max_abs_difference.max()) if len(chk) else None)


if __name__ == '__main__':
    main()
