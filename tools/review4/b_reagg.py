"""OPS2 7.19 B: stored-output re-aggregations (no training, no network inference).

b1  E1 (A9 item 1): (ii-lag-ubfc) - (ii-ubfc) per backbone on MMPD-B (Q*), the MCD-rPPG test half (draft rule) and the
    VIPL-HR primary set (HAR): high-HR, low-HR and all-window MAE and high-HR hit rate, with the A9 item 9 intervals
    (crossed subject x seed bootstrap, B = 10,000, default_rng(20261110), seeds 20261131-33 paired). Frames, readouts and
    jobs are those of tools/a9_eval/a9_groups.py (frames, AB.sel, analysis_704.job_contrast); beside each value the stored
    contrast (ii-lag) - (ii) with PURE training (test run secondary.csv; VIPL-HR stage (c2) exploratory_arms).
b2  Per-seed signs: high-HR MAE difference of C1-C4 and (ii-lag) - (ii-pc) per seed, PURE training, on the three test sets.
    Seeds: 20261131-33 and the new 20261148-50 where both arms exist on all six (C1-C4); (ii-lag) has no new seeds, so
    (ii-lag) - (ii-pc) uses the three original seeds. Windows: S3.pair_windows over all seeds used (as the stored six-seed
    analyses); per-seed value = the summand of the bootstrap point (boot.per_seed with unit weights), so the mean over seeds
    must equal the stored point (checked: MMPD-B/MCD-rPPG outputs/704/analysis_b/test/p 'six', VIPL-HR E2_P 'all six', and
    the stored secondary (ii-lag)-(ii-pc)). Frames: MMPD-B/MCD-rPPG analysis_704b.load('test') (test run plus the new-seed
    passes), VIPL-HR as tools/a9_eval/a9_e2e3.py E2 (stored keys plus the E2 pass).
b3  PURE tau slope: pool tau (tau_rec, search -15..-4 frames) of the 59 recordings against the half period (outputs/facts_697/
    audit_PURE.csv), OLS slope (summary.json of 1a: -0.387) with the subject-cluster bootstrap of tools/posthoc701/p701_da.py
    (boot_ci, B = 10,000, default_rng(20261110)) and its t and recording-bootstrap intervals, as da_slopes.csv.
b5  Generation time and storage per pool item, (ii-pc) and (i-simple), PURE (second pool, 7.04/7.07) and UBFC-rPPG (E1), from
    the generation manifests and run records.
Writes outputs/719/B/<n>/ (refuses to overwrite).
"""
import json
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[2]
OUT = W / 'outputs' / '719' / 'B'
sys.path.insert(0, str(W / 'tools'))
sys.path.insert(0, str(W / 'tools' / 'a9_eval'))
sys.path.insert(0, str(W / 'tools' / 'a9_8'))


def outdir(n):
    o = OUT / n
    if o.exists() and any(o.iterdir()):
        sys.exit(f'{o} exists and is not empty')
    o.mkdir(parents=True, exist_ok=True)
    return o


def cmd_b1(workers, b):
    import a9_groups as G
    from a9_groups import A, AB, CONF, pd
    A.setup()
    o = outdir('1')
    t0 = time.time()
    rows = []
    for g in ('E1_PhysNet', 'E1_FPm', 'E1_RhythmFormer'):
        fr, bad = G.frames(g)
        bb = G.BB[g]
        jobs = []
        for ds, f in fr.items():
            ga, gc = AB.sel(f, bb, '(ii-lag-ubfc)', CONF), AB.sel(f, bb, '(ii-ubfc)', CONF)
            assert ga is not None and gc is not None, (g, ds)
            jobs.append(('contrast', CONF, (dict(dataset=G.NAMES[ds], experiment='E1', backbone=bb, base_contrast='(ii-lag)-(ii)',
                                                 contrast='(ii-lag-ubfc)-(ii-ubfc)'), ga, gc, b)))
        res = AB.execute(jobs, workers, f'b1 {g}')
        rows += res['contrast']
    df = pd.DataFrame(rows)
    df.to_csv(o / 'E1_iilag_minus_ii_all_metrics.csv', index=False)
    ref = G.pure_reference()
    keep = ('MAE_high', 'MAE_low', 'MAE_all', 'hit_high')
    s = []
    for r in df[df.metric.isin(keep)].itertuples():
        pr = ref.get((r.dataset, r.backbone, '(ii-lag)-(ii)', r.metric))
        s.append(dict(dataset=r.dataset, backbone=r.backbone, contrast='(ii-lag) - (ii), UBFC-rPPG training', metric=r.metric,
                      windows=r.windows, subjects=r.subjects, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                      pure_point=pr[0] if pr else '', pure_ci_lo=pr[1] if pr else '', pure_ci_hi=pr[2] if pr else ''))
    pd.DataFrame(s).to_csv(o / 'E1_iilag_minus_ii.csv', index=False)
    (o / 'run_info.json').write_text(json.dumps(dict(command='b1', b=b, seconds=time.time() - t0, rows=len(s)), indent=1))
    print('b1 done', len(s), round(time.time() - t0), flush=True)


def per_seed_diff(A, ga, gc, seeds):
    boot, S3 = A.boot, A.S3
    ga, gc, nw = S3.pair_windows(ga, gc)
    subjects = boot.numeric_subject_order(set(ga.subject_id) | set(gc.subject_id))
    one = A.np.ones((1, len(subjects)))
    va = boot.per_seed(boot.Sample(ga, subjects, seeds), A.m_all, one)
    vc = boot.per_seed(boot.Sample(gc, subjects, seeds), A.m_all, one)
    nh = len(set(map(tuple, ga[ga.set.eq('high')][['window_id', 'layer']].to_numpy())))
    return {sd: float(va[sd]['MAE_high'][0] - vc[sd]['MAE_high'][0]) for sd in seeds}, nw, nh, ga.subject_id.nunique()


def cmd_b2():
    import analysis_704 as A
    import analysis_704b as AB
    import a9_common as C
    import a9_e2e3 as E
    pd, np = C.pd, C.np
    A.setup()
    o = outdir('2')
    t0 = time.time()
    CONF, NEW = E.CONF, E.NEW
    old, _, _, _ = C.vipl_frame()
    new, _ = E.vipl_new_frame()
    vf = C.SA.restrict(pd.concat([old, new], ignore_index=True).pipe(C.primary)).drop_duplicates(['key', 'window_id', 'layer'])
    frames = {'VIPL-HR': vf}
    for ds in ('MMPD', 'MCD'):
        frames[E.NAMES[ds]] = AB.load('test', ds)[0]
    pairs = list(A.PAIRS[:4]) + [('(ii-lag)-(ii-pc)', '(ii-lag)', '(ii-pc)')]
    rows, summ = [], []
    for ds, f in frames.items():
        for bb in E.BACK:
            for name, a, c in pairs:
                seeds = CONF + NEW
                ga, gc = AB.sel(f, bb, a, seeds), AB.sel(f, bb, c, seeds)
                if ga is None or gc is None:
                    seeds = CONF
                    ga, gc = AB.sel(f, bb, a, seeds), AB.sel(f, bb, c, seeds)
                if ga is None or gc is None:
                    continue
                d, nw, nh, ns = per_seed_diff(A, ga, gc, seeds)
                for sd in seeds:
                    rows.append(dict(dataset=ds, backbone=bb, contrast=name, arms=f'{a}-{c}', seed=sd,
                                     seed_set='original' if sd in CONF else 'new', windows=nw, high_windows=nh, subjects=ns,
                                     MAE_high_difference=d[sd]))
                v = np.array([d[sd] for sd in seeds])
                summ.append(dict(dataset=ds, backbone=bb, contrast=name, arms=f'{a}-{c}', seeds=len(seeds),
                                 seeds_used=('20261131-33, 20261148-50' if len(seeds) == 6 else '20261131-33'),
                                 negative_seeds=int((v < 0).sum()), positive_seeds=int((v > 0).sum()), zero_seeds=int((v == 0).sum()),
                                 negative_original=int(sum(d[sd] < 0 for sd in CONF)),
                                 negative_new=int(sum(d[sd] < 0 for sd in seeds if sd not in CONF)) if len(seeds) == 6 else '',
                                 mean_over_seeds=float(v.mean()), min=float(v.min()), max=float(v.max()), windows=nw,
                                 high_windows=nh, subjects=ns))
    rows, summ = pd.DataFrame(rows), pd.DataFrame(summ)
    # checks: mean over seeds equals the stored bootstrap point
    checks = []
    st6 = {}
    for ds, name in (('MMPD', 'MMPD-B'), ('MCD', 'MCD-rPPG test')):
        p = pd.read_csv(W / 'outputs' / '704' / 'analysis_b' / 'test' / 'p' / ds / 'contrasts.csv')
        for r in p[p.seeds.eq('six') & p.metric.eq('MAE_high')].itertuples():
            st6[(name, r.backbone, r.name)] = r.point
        sec = AB.stored('test', ds)
        for r in sec[sec.contrast.eq('(ii-lag)-(ii-pc)') & sec.metric.eq('MAE_high')].itertuples():
            st6[(name, r.backbone, '(ii-lag)-(ii-pc)')] = r.point
    e2 = pd.read_csv(W / 'outputs' / 'a9_eval' / 'run' / 'E2_P_contrasts.csv')
    for r in e2[e2.seeds.eq('all six') & e2.metric.eq('MAE_high')].itertuples():
        st6[('VIPL-HR', r.backbone, r.name)] = r.point
    c2 = pd.read_csv(C.VC2 / 'secondary_checkpoints.csv')
    c2 = c2[c2.analysis.eq('exploratory_arms') & c2.layer.eq('HAR') & c2.population.eq('primary') &
            c2.contrast.eq('(ii-lag)-(ii-pc)') & c2.metric.eq('MAE_high')]
    for r in c2.itertuples():
        st6[('VIPL-HR', r.backbone, '(ii-lag)-(ii-pc)')] = r.point
    for r in summ.itertuples():
        sp = st6.get((r.dataset, r.backbone, r.contrast))
        checks.append(dict(dataset=r.dataset, backbone=r.backbone, contrast=r.contrast, seeds=r.seeds, mean_over_seeds=r.mean_over_seeds,
                           stored_point=sp if sp is not None else '', abs_difference=abs(r.mean_over_seeds - sp) if sp is not None else '',
                           equal_1e9=(abs(r.mean_over_seeds - sp) < 1e-9) if sp is not None else ''))
    rows.to_csv(o / 'per_seed_MAE_high_differences.csv', index=False)
    summ.to_csv(o / 'per_seed_sign_counts.csv', index=False)
    pd.DataFrame(checks).to_csv(o / 'checks_mean_equals_stored_point.csv', index=False)
    (o / 'run_info.json').write_text(json.dumps(dict(command='b2', seconds=time.time() - t0, rows=len(rows)), indent=1))
    print('b2 done', len(rows), len(summ), round(time.time() - t0), flush=True)
    print(pd.DataFrame(checks).to_string())


def cmd_b3():
    sys.path.insert(0, str(W / 'tools' / 'posthoc701'))
    import p701_da as DA
    import pandas as pd
    o = outdir('3')
    rec = pd.read_csv(W / 'outputs' / 'facts_697' / 'audit_PURE.csv', dtype={'recording_id': str})
    nat = pd.read_csv(W / 'outputs' / '1a' / 'natural_recordings.csv', dtype={'recording_id': str})
    assert len(rec) == 59
    rec['subject'] = rec.recording_id.str.split('-').str[0]
    j = rec.merge(nat[['recording_id', 'tau_rec_file', 'half_period_frames']], on='recording_id', suffixes=('', '_1a'))
    same_tau = bool((j.tau_rec_pool_frames == j.tau_rec_file).all())
    same_half = bool((j.half_period_frames - j.half_period_frames_1a).abs().max() < 1e-12)
    x, y = rec.half_period_frames.to_numpy(float), rec.tau_rec_pool_frames.to_numpy(float)
    pt, tlo, thi = DA.t_ci(x, y)
    slo, shi, sdrop = DA.boot_ci(x, y, rec.subject.tolist())
    rlo, rhi, rdrop = DA.boot_ci(x, y, rec.recording_id.tolist())
    s1a = json.loads((W / 'outputs' / '1a' / 'summary.json').read_text())['between_recordings']['tau_rec_vs_half_period']
    da = pd.read_csv(W / 'outputs' / 'posthoc701' / 'da' / 'da_slopes.csv')
    rows = [dict(level='recording (59)', slope='pool tau (tau_rec, search -15..-4) vs half period, frames', n=len(rec),
                 subjects=rec.subject.nunique(), point=pt, subject_cluster_ci_lo=slo, subject_cluster_ci_hi=shi,
                 subject_cluster_dropped_replicates=sdrop, t_ci_lo=tlo, t_ci_hi=thi, recording_boot10000_ci_lo=rlo,
                 recording_boot10000_ci_hi=rhi, recording_boot10000_dropped_replicates=rdrop)]
    rows += da.iloc[:1].to_dict('records')
    pd.DataFrame(rows).to_csv(o / 'tau_slopes.csv', index=False)
    checks = [dict(check='pool tau of audit_PURE.csv equals tau_rec of 1a natural_recordings.csv (59)', passed=same_tau),
              dict(check='half periods equal', passed=same_half),
              dict(check='OLS point equals 1a summary.json ols_slope', value=f"{pt} vs {s1a['ols_slope']}",
                   passed=abs(pt - s1a['ols_slope']) < 1e-12),
              dict(check='t interval equals 1a summary.json slope_ci', value=f"{tlo}, {thi} vs {s1a['slope_ci_low']}, {s1a['slope_ci_high']}",
                   passed=abs(tlo - s1a['slope_ci_low']) < 1e-12 and abs(thi - s1a['slope_ci_high']) < 1e-12)]
    xr, yr = rec.half_period_frames.to_numpy(float), rec.tau_signed_max_lag_frames.to_numpy(float)
    lo2, hi2, _ = DA.boot_ci(xr, yr, rec.subject.tolist())
    checks.append(dict(check='same code reproduces the da_slopes.csv subject-cluster interval of -1.02',
                       value=f'{lo2}, {hi2}', passed=abs(lo2 - da.iloc[0].subject_cluster_ci_lo) < 1e-12 and abs(hi2 - da.iloc[0].subject_cluster_ci_hi) < 1e-12))
    pd.DataFrame(checks).to_csv(o / 'checks.csv', index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(pd.DataFrame(checks).to_string())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['b1', 'b2', 'b3', 'b5'])
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--b', type=int, default=10000)
    a = ap.parse_args()
    if a.cmd == 'b1':
        cmd_b1(a.workers, a.b)
    elif a.cmd == 'b2':
        cmd_b2()
    elif a.cmd == 'b3':
        cmd_b3()
    else:
        cmd_b5()



def cmd_b5():
    """Generation time and storage per pool item, (ii-pc) and (i-simple), PURE second pool and UBFC-rPPG E1.

    Time per item from the generation manifests (ops2_CPU_seconds = wall seconds of the worker, time.time()):
      family (i) (one clip per item shared by (i), (i-pc), (i-simple); clock per item, cumulative over its three labels):
        time per item = max over the item's three rows;
      family (ii) (clock per chunk, cumulative over the chunk's items and over all modes generated in the same run):
        time per item and arm = chunk total (max over the chunk's rows in all modes of the run) / rows of the chunk.
    Run level: wall seconds of the run (generation log) x workers / (items x modes of the run), where the worker count is
    recorded. Bytes per item: stored view + raw label (+ statistics) files of the arm (the (i) clip is shared by three arms).
    PURE also gives the CPU_seconds column that the pool manifests carry from the October generation manifests ((i) and
    (ii-h) source items; for PhysNet equal to outputs/st9g_pool_PhysNet_{i,ii-h}_manifest.csv).
    """
    import pandas as pd
    o = outdir('5')
    P2, UB = W / 'cache' / 'pools_p2', W / 'cache' / 'ubfc709' / 'pools'
    rows = []
    runs = {  # (corpus, backbone, family): (wall seconds, workers, modes, source)
        ('PURE', 'PhysNet', 'i'): (3153, 4, 3, 'p2_gen_PhysNet_i.log'), ('PURE', 'FPm', 'i'): (2774, 4, 3, 'p2_gen_FPm_i.log'),
        ('PURE', 'RhythmFormer', 'i'): (3877, 4, 3, 'p2_gen_RhythmFormer_i.log'),
        ('PURE', 'PhysNet', 'ii'): (23472, 4, 2, 'p2_gen_PhysNet_ii.log'), ('PURE', 'FPm', 'ii'): (20568, 4, 2, 'p2_gen_FPm_ii.log'),
        ('PURE', 'RhythmFormer', 'ii'): (16987, 4, 2, 'p2_gen_RhythmFormer_ii.log'),
        ('UBFC-rPPG', 'PhysNet', 'i'): (5765, 3, 3, 'ubfc_prep.out'), ('UBFC-rPPG', 'FPm', 'i'): (6044, 3, 3, 'ubfc_prep.out'),
        ('UBFC-rPPG', 'FPm', 'ii'): (66701, 4, 3, 'ubfc_edit_FPm.log (HDD volume)'),
        ('UBFC-rPPG', 'RhythmFormer', 'ii'): (29931, 4, 2, 'ubfc_edit_RhythmFormer_part1.log'),
    }
    for b in ('PhysNet', 'FPm', 'RhythmFormer'):
        fams = {
            ('PURE', 'i'): [P2 / f'{b}_p2_i_family_manifest.csv'],
            ('PURE', 'ii'): [P2 / f'{b}_p2_ii_family_manifest.csv'],
            ('UBFC-rPPG', 'i'): [UB / f'{b}_{a}_all_manifest.csv' for a in ('i', 'i-pc', 'i-simple')],
            ('UBFC-rPPG', 'ii'): [UB / f'{b}_{a}_source_all_manifest.csv' for a in
                                  (('ii', 'ii-pc') if b == 'RhythmFormer' else ('ii', 'ii-pc', 'ii-lag'))],
        }
        for (corpus, fam), files in fams.items():
            d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
            d = d[d.get('generation_excluded', pd.Series(False, index=d.index)).astype(str).ne('True')]
            arm = '(i-simple)' if fam == 'i' else '(ii-pc)'
            mine = d[d.arm.eq(arm)]
            if fam == 'i':
                t = d.groupby(['chunk_order', 'augmentation_index']).ops2_CPU_seconds.max()
                per = t.to_numpy()
                label = 'item (one clip and three labels)'
            else:
                ct = d.groupby('chunk_order').ops2_CPU_seconds.max()
                n = d.groupby('chunk_order').size()
                per = (ct / n).reindex(mine.chunk_order).to_numpy()
                label = f'item and arm (chunk total / rows; {d["mode"].nunique()} modes in the run)'
            by = mine.ops2_view_bytes.astype(float)
            for c in ('ops2_rawlabel_bytes', 'ops2_stats_bytes'):
                if c in mine:
                    by = by + mine[c].fillna(0).astype(float)
            run = runs.get((corpus, b, fam))
            r = dict(corpus=corpus, backbone=b, arm=arm, family=fam, items=len(mine), time_unit=label,
                     manifest_seconds_per_unit_median=float(pd.Series(per).median()),
                     manifest_seconds_per_unit_mean=float(pd.Series(per).mean()),
                     bytes_per_item_median=float(by.median()), bytes_per_item_mean=float(by.mean()), bytes_total=float(by.sum()),
                     bytes_columns='view' + (' + raw label' if 'ops2_rawlabel_bytes' in mine else '')
                                   + (' + statistics' if 'ops2_stats_bytes' in mine else '')
                                   + ('' if 'ops2_rawlabel_bytes' in mine else ' (raw label and statistics bytes not recorded)'))
            if run:
                wall, workers, modes, src = run
                r.update(run_wall_seconds=wall, run_workers=workers, run_modes=modes, run_items=len(mine),
                         run_worker_seconds_per_item_and_arm=wall * workers / (len(mine) * (modes if fam == 'ii' else 1)),
                         run_source=src)
            if corpus == 'PURE' and 'CPU_seconds' in mine:
                r.update(october_manifest_CPU_seconds_median=float(mine.CPU_seconds.astype(float).median()))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(o / 'pool_item_time_storage.csv', index=False)
    rat = []
    for (corpus, b), g in df.groupby(['corpus', 'backbone']):
        a, s = g[g.arm.eq('(ii-pc)')].iloc[0], g[g.arm.eq('(i-simple)')].iloc[0]
        rr = dict(corpus=corpus, backbone=b,
                  manifest_time_ratio_median=a.manifest_seconds_per_unit_median / s.manifest_seconds_per_unit_median,
                  manifest_time_ratio_mean=a.manifest_seconds_per_unit_mean / s.manifest_seconds_per_unit_mean,
                  bytes_ratio_median=a.bytes_per_item_median / s.bytes_per_item_median)
        if pd.notna(a.get('run_worker_seconds_per_item_and_arm')) and pd.notna(s.get('run_worker_seconds_per_item_and_arm')):
            rr['run_time_ratio'] = a.run_worker_seconds_per_item_and_arm / s.run_worker_seconds_per_item_and_arm
        if corpus == 'PURE':
            rr['october_CPU_seconds_ratio_median'] = a.october_manifest_CPU_seconds_median / s.october_manifest_CPU_seconds_median
        rat.append(rr)
    pd.DataFrame(rat).to_csv(o / 'ratios_iipc_over_isimple.csv', index=False)
    print(df.T.to_string())
    print(pd.DataFrame(rat).to_string())


if __name__ == '__main__':
    main()
