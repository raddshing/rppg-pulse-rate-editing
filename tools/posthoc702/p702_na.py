"""Brief 02 7.02 나: paired contrasts of the within-person change on the MCD-rPPG test half (post hoc; saved test step (c)
outputs only).

Definition (m7, unchanged): test step (c) M7 population (both states), wide band (HAR), primary variant, confirmatory support
and each key's valid windows (eval5/stage3_analyze.py: restrict(prim[population == 'M7_both_states'])); per run and subject the
reference and predicted change (after - before) of the recording means (eval5/families.py m7_table); M7_MAE (MAE of the
changes), M7_slope (OLS slope of predicted on reference change), M7_CCC (b1/boot.py m7).
Contrasts (same seed, same resample): one bootstrap function evaluates both arms on the same subject x seed draw (b1/boot.py
draws, B = 10,000, default_rng(20261110), subjects of the population in numeric order) and returns the two arm values and
their difference. Subjects: those with a change for every key of both arms.
Populations: all subjects of the M7 population (149); subjects whose post-exercise reference HR is >= 100 bpm (the mean
reference HR of the subject's after-state windows of the M7 population inside the confirmatory support; arm independent).
Writes outputs/posthoc702/na/{na_contrasts,na_subjects,na_checks}.csv.
"""
from p702_common import (B, SA, SEEDS, SRC_TEST, TEST, boot, load, np, pd, stamp, summarize_nan, time, write)

ARMS = ['(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)', '(i-simple)', '(ii-lag)']
PAIRS = [('(ii-pc)', '(0)'), ('(i-simple)', '(0)'), ('(ii-pc)', '(ii)'), ('(i-pc)', '(i)'), ('(ii-pc)', '(i-simple)'),
         ('(ii-lag)', '(ii-pc)')]
METRICS = ['M7_MAE', 'M7_slope', 'M7_CCC']


def paired(ta, tc, subjects):
    subjects = boot.numeric_subject_order(subjects)
    sa = boot.Sample(ta[ta.subject_id.isin(subjects)], subjects, SEEDS)
    sc = boot.Sample(tc[tc.subject_id.isin(subjects)], subjects, SEEDS)

    def f(W):
        va, vc = boot.per_seed(sa, boot.m7, W), boot.per_seed(sc, boot.m7, W)
        return {sd: {**{f'A|{k}': va[sd][k] for k in METRICS}, **{f'B|{k}': vc[sd][k] for k in METRICS},
                     **{f'diff|{k}': va[sd][k] - vc[sd][k] for k in METRICS}} for sd in SEEDS}
    return summarize_nan(f, subjects, SEEDS, B)


def main():
    t0 = time.time()
    d, bad = load(TEST, 'MCD')
    prim = d[d.variant.isin(SA.PRIMARY)]
    m7 = SA.restrict(prim[prim.population.eq('M7_both_states')])
    m7 = m7[m7.layer.eq('HAR')]
    # post-exercise reference HR per subject (population windows in the support, arm independent)
    pop = prim[prim.population.eq('M7_both_states') & prim.layer.eq('HAR') & prim.in_support]
    pop = pop.drop_duplicates(['window_id', 'layer'])
    after = pop[pop.state.eq('after')].groupby('subject_id').reference_HR.mean()
    before = pop[pop.state.eq('before')].groupby('subject_id').reference_HR.mean()
    subj = pd.DataFrame(dict(after_reference_HR=after, before_reference_HR=before)).reset_index()
    subj['after_ge_100'] = subj.after_reference_HR >= 100
    tables = {}
    for bb in sorted(m7.backbone.unique()):
        for st in ARMS:
            t = SA.AN.m7_table(m7[m7.backbone.eq(bb) & m7.strategy.eq(st)])
            tables[(bb, st)] = t[t.seed.isin(SEEDS)]
    ref = pd.read_csv(TEST / 'analysis' / 'MCD' / 'halves' / 'm7.csv')
    ref = ref[ref.layer.eq('HAR')]
    rows, chk = [], []
    groups = {'all subjects': set(subj.subject_id), 'post-exercise reference HR >= 100': set(subj.subject_id[subj.after_ge_100])}
    for bb in sorted(m7.backbone.unique()):
        for a, c in PAIRS:
            ta, tc = tables[(bb, a)], tables[(bb, c)]
            ka, kc = ta.groupby('subject_id').seed.nunique(), tc.groupby('subject_id').seed.nunique()
            both = set(ka[ka == len(SEEDS)].index) & set(kc[kc == len(SEEDS)].index)
            for gname, members in groups.items():
                s = both & members
                pt, ci, nan = paired(ta, tc, s)
                for part, label in (('A', a), ('B', c), ('diff', f'{a}-{c}')):
                    for k in METRICS:
                        q = f'{part}|{k}'
                        rows.append(dict(dataset='MCD-rPPG test', population=gname, backbone=bb, contrast=f'{a}-{c}', quantity=part,
                                         item=label, subjects=len(s), metric=k, point=pt[q], ci_lo=ci[q][0], ci_hi=ci[q][1],
                                         nan_replicates=nan[q], source=SRC_TEST))
                if gname == 'all subjects':
                    for part, st in (('A', a), ('B', c)):
                        for k in METRICS:
                            o = ref[ref.backbone.eq(bb) & ref.strategy.eq(st) & ref.metric.eq(k) & ref.seeds.astype(str).eq('3')]
                            q = f'{part}|{k}'
                            if len(o) and int(o.iloc[0].subjects) == len(s):
                                # the arm alone on the same subjects and the same draws equals the m7 table of the test run
                                chk.append(dict(table='m7', backbone=bb, contrast=f'{a}-{c}', item=st, metric=k, subjects=len(s),
                                                max_abs_difference=float(np.nanmax(np.abs([pt[q] - o.iloc[0].point, ci[q][0] - o.iloc[0].ci_lo,
                                                                                            ci[q][1] - o.iloc[0].ci_hi])))))
            print(bb, a, c, 'done', round(time.time() - t0), flush=True)
    chk = pd.DataFrame(chk).assign(source=f'{SRC_TEST} (recomputed here vs stored test step (c) m7.csv)')
    write(pd.DataFrame(rows), 'na_contrasts.csv', 'na')
    cnt = pd.DataFrame([dict(population=g, subjects=len(m),
                             after_reference_HR_median=float(subj[subj.subject_id.isin(m)].after_reference_HR.median()),
                             after_reference_HR_min=float(subj[subj.subject_id.isin(m)].after_reference_HR.min()),
                             after_reference_HR_max=float(subj[subj.subject_id.isin(m)].after_reference_HR.max()),
                             before_reference_HR_median=float(subj[subj.subject_id.isin(m)].before_reference_HR.median()),
                             source=SRC_TEST) for g, m in groups.items()])
    write(cnt, 'na_subjects.csv', 'na')
    write(chk, 'na_checks.csv', 'na')
    print('checks', len(chk), 'max', chk.max_abs_difference.max() if len(chk) else None)
    stamp('na', seconds=time.time() - t0, checks=len(chk), checks_max_abs_difference=float(chk.max_abs_difference.max()) if len(chk) else None,
          subjects=cnt.to_dict('records'))


if __name__ == '__main__':
    main()
