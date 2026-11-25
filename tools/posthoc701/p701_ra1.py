"""Brief 02 7.01 라-1: loss swap results completed before A4 (PhysNet trained with the RhythmFormer author loss), development
data only, saved guarded predictions only (no new run).

Predictions: the guarded development readouts (outputs/guarded/1b: rehearsal part-1 stage-2 predictions and the guarded
re-inference rounds, including rfl_PhysNet and rfli_PhysNet; the source of the paper's development values, guarded_v2).
Populations and support: b1/analyze.py load_mmpd / load_mcd unchanged (common support of all runs; MMPD-A 760 validation
windows; MCD-rPPG development half draft rule), plus MMPD-A 723 = the same windows restricted to the automatic Q* rule of the
development stage 1 (rehearsal part 2; reference HR and high flag of that rule), with the same common-support rule.
Arms: PhysNet (0), (i), (i-pc), (ii), (ii-pc) (standard loss) and (i-rfl), (i-pc-rfl), (ii-rfl), (ii-pc-rfl) (RhythmFormer
author loss); RhythmFormer (0), (i), (i-pc), (ii), (ii-pc) (its standard = author loss). Seeds 20261131-33, wide band HAR.
Estimates (b1/boot.py crossed subject x seed bootstrap, B = 10,000, default_rng(20261110), the subjects of the dataset):
  arms         MAE high / low / all and signed error (boot.point_ci, m1)
  loss swap    in one bootstrap function (tools/lossswap_interaction.py logic): D RF loss = (ii-pc-rfl)-(ii-rfl),
               (i-pc-rfl)-(i-rfl), (ii-pc-rfl)-(i-pc-rfl); D standard = the same under the standard loss; interaction
               I = D(RF loss) - D(standard); loss effect L_X = X-rfl - X
  analysis A   in one bootstrap function (tools/analysis_a.py logic): C2 = (ii-pc)-(ii), C3 = (i-pc)-(i), C4 = (ii-pc)-(i-pc)
               of RhythmFormer and of PhysNet with the RhythmFormer loss, D_k = C_k(RhythmFormer) - C_k(PhysNet RF loss)
MCD-rPPG key-frame line fractions (147.66-154.69 bpm): copied from the guarded development analysis guarded_v2
(outputs/guarded_v2/keyframe, tools/keyframe_line.py and keyframe_common.py on the same guarded readouts), not recomputed.
Writes outputs/posthoc701/ra1/{ra1_arms,ra1_lossswap,ra1_analysis_a,ra1_keyframe_line,ra1_keyframe_diff,ra1_counts,ra1_checks}.csv.
"""
import os
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
os.environ['OPS2_OUT1B'] = str(WORK / 'outputs' / 'guarded' / '1b')  # before b1/analyze.py is imported
from p701_common import DEV, SRC_DEV, np, pd, stamp, summarize_nan, write  # noqa: E402
import analyze  # noqa: E402  (b1/analyze.py, OUT1B = guarded readouts)
import boot  # noqa: E402

SEEDS = analyze.MAIN
STD = ['(i)', '(i-pc)', '(ii)', '(ii-pc)']
RFL = [a[:-1] + '-rfl)' for a in STD]
DELTAS = [('(ii-pc)', '(ii)'), ('(i-pc)', '(i)'), ('(ii-pc)', '(i-pc)')]
C = {'C2': ('(ii-pc)', '(ii)'), 'C3': ('(i-pc)', '(i)'), 'C4': ('(ii-pc)', '(i-pc)')}
SRC = f'{SRC_DEV} (guarded readouts)'


def load_mmpd_723(meta):
    """analyze.load_mmpd with the automatic Q* rule of the development stage 1 (723 HAR windows) instead of the 760 roster."""
    parts = [pd.read_csv(analyze.OUT1B / 'mmpd_a_windows.csv.gz', dtype={'subject_id': str})]
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for pre in ('iipc', 'ipc', 'iinv', 'ilag', 'isimple', 'rfl', 'rfli') + tuple('lab-' + a for a in analyze.model_sets.LAB_ARMS):
            f = analyze.OUT1B / f'mmpd_a_windows_{pre}_{b}.csv.gz'
            if f.exists():
                parts.append(pd.read_csv(f, dtype={'subject_id': str}))
    d = analyze.attach(pd.concat(parts, ignore_index=True), meta)
    auto = pd.read_csv(DEV / 'stage1' / 'MMPD' / 'populations.csv')[['window_id', 'layer', 'reference_HR', 'in_Qstar', 'high']]
    d = d.merge(auto.rename(columns={'reference_HR': 'ref_auto', 'in_Qstar': 'q_auto', 'high': 'high_auto'}), on=['window_id', 'layer'], how='left')
    q = d.q_auto.fillna(False).astype(bool)
    diff = float((d.reference_HR[q] - d.ref_auto[q]).abs().max())
    d = d[q].copy()
    d['reference_HR'], d['high'], d['in_Qstar'] = d.ref_auto, d.high_auto, True
    d = analyze.support(d.drop(columns=['ref_auto', 'q_auto', 'high_auto']))
    d['set'] = np.where(d.high.astype(bool), 'high', 'low')
    d['dataset'], d['population'] = 'MMPD-A 723', 'automatic Q*'
    return d, diff


def arm_rows(dd, ds):
    out = []
    for (bb, st), g in dd.groupby(['backbone', 'strategy']):
        if not ((bb == 'PhysNet' and st in ['(0)'] + STD + RFL) or (bb == 'RhythmFormer' and st in ['(0)'] + STD)):
            continue
        pt, ci = boot.point_ci(g, SEEDS, boot.m1)
        one = g[g.seed.eq(SEEDS[0])]
        for k in pt:
            out.append(dict(dataset=ds, backbone=bb, arm=st, metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], windows=len(one),
                            high_windows=int((one.set == 'high').sum()), subjects=g.subject_id.nunique(), source=SRC))
    return out


def lossswap(dd, ds):
    d = dd[dd.backbone.eq('PhysNet')]
    names = STD + RFL
    subjects = boot.numeric_subject_order(set(d.subject_id))
    smp = {n: boot.Sample(d[d.strategy.eq(n)], subjects, SEEDS) for n in names}
    nwin = {n: len(s.per[SEEDS[0]][1]) for n, s in smp.items()}
    assert len(set(nwin.values())) == 1, nwin

    def f(W):
        v = {n: boot.per_seed(smp[n], boot.m1, W) for n in names}
        res = {}
        for sd in SEEDS:
            r = {}
            for x in STD:
                xr = x[:-1] + '-rfl)'
                for k in ('MAE_high', 'MAE_low'):
                    r[f'loss effect|{xr}-{x}|{k}'] = v[xr][sd][k] - v[x][sd][k]
            for a, b in DELTAS:
                ar, br = a[:-1] + '-rfl)', b[:-1] + '-rfl)'
                for k in ('MAE_high', 'MAE_low'):
                    dstd, drf = v[a][sd][k] - v[b][sd][k], v[ar][sd][k] - v[br][sd][k]
                    r[f'D standard|{a}-{b}|{k}'] = dstd
                    r[f'D RF loss|{ar}-{br}|{k}'] = drf
                    r[f'interaction I = D(RF loss) - D(standard)|[{a}-{b}]|{k}'] = drf - dstd
            res[sd] = r
        return res
    pt, ci, nan = summarize_nan(f, subjects, SEEDS)
    full = f(np.ones((1, len(subjects))))
    out = []
    for k in pt:
        q, c, m = k.split('|')
        out.append(dict(dataset=ds, quantity=q, contrast=c, metric=m, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], nan_replicates=nan[k],
                        **{f'seed_{sd}': float(full[sd][k][0]) for sd in SEEDS}, windows=list(nwin.values())[0], subjects=len(subjects),
                        source=SRC))
    return out


def analysis_a(dd, ds):
    subjects = boot.numeric_subject_order(set(dd.subject_id))
    smp = {}
    for a in STD:
        for key, (bb, st) in ((('RF', a), ('RhythmFormer', a)), (('PN', a), ('PhysNet', a[:-1] + '-rfl)'))):
            sub = dd[dd.backbone.eq(bb) & dd.strategy.eq(st)]
            assert len(sub), (ds, bb, st)
            smp[key] = boot.Sample(sub, subjects, SEEDS)
    nwin = {k: len(s.per[SEEDS[0]][1]) for k, s in smp.items()}
    assert len(set(nwin.values())) == 1, nwin

    def f(W):
        v = {k: boot.per_seed(s, boot.m1, W) for k, s in smp.items()}
        res = {}
        for sd in SEEDS:
            r = {}
            for c, (x, y) in C.items():
                for m in ('MAE_high', 'MAE_low'):
                    rf = v[('RF', x)][sd][m] - v[('RF', y)][sd][m]
                    pn = v[('PN', x)][sd][m] - v[('PN', y)][sd][m]
                    r[f'{c}|RhythmFormer|{m}'] = rf
                    r[f'{c}|PhysNet RF loss|{m}'] = pn
                    r[f'{c}|D = RhythmFormer - PhysNet RF loss|{m}'] = rf - pn
            res[sd] = r
        return res
    pt, ci, nan = summarize_nan(f, subjects, SEEDS)
    full = f(np.ones((1, len(subjects))))
    out = []
    for k in pt:
        c, q, m = k.split('|')
        out.append(dict(dataset=ds, contrast=f'{c} = {C[c][0]}-{C[c][1]}', quantity=q, metric=m, point=pt[k], ci_lo=ci[k][0],
                        ci_hi=ci[k][1], nan_replicates=nan[k], **{f'seed_{sd}': float(full[sd][k][0]) for sd in SEEDS},
                        windows=list(nwin.values())[0], subjects=len(subjects), source=SRC))
    return out


def checks(ls, arms):
    """760 / MCD values against the guarded development analysis guarded_v2 (crfl, c25, cipc, closs, m123)."""
    g = WORK / 'outputs' / 'guarded_v2'
    ref = pd.concat([pd.read_csv(g / f) for f in ('crfl.csv', 'c25.csv', 'cipc.csv', 'closs.csv')])
    ref = ref[ref.layer.eq('HAR') & ref.backbone.eq('PhysNet') & ref.seeds.astype(str).eq('3') & ref.population.ne('M7_both_states')]
    name = {'MMPD-A': 'MMPD-A 760', 'MCD-dev': 'MCD-dev'}
    out = []
    for r in ls.itertuples():
        if r.quantity.startswith('interaction'):
            continue
        o = ref[ref.dataset.map(name).eq(r.dataset) & ref.contrast.eq(r.contrast) & ref.metric.eq(r.metric)]
        if len(o):
            o = o.iloc[0]
            out.append(dict(table='guarded_v2 ' + str(o.family), dataset=r.dataset, item=r.contrast, metric=r.metric,
                            max_abs_difference=float(np.nanmax(np.abs([r.point - o.point, r.ci_lo - o.ci_lo, r.ci_hi - o.ci_hi])))))
    m = pd.read_csv(g / 'm123.csv')
    m = m[m.layer.eq('HAR') & m.seeds.astype(str).eq('3') & m.metric.isin(['MAE_high', 'MAE_low', 'MAE_all'])]
    for r in arms.itertuples():
        o = m[m.dataset.map(name).eq(r.dataset) & m.backbone.eq(r.backbone) & m.strategy.eq(r.arm) & m.metric.eq(r.metric)]
        if len(o):
            o = o.iloc[0]
            out.append(dict(table='guarded_v2 m123', dataset=r.dataset, item=f'{r.backbone} {r.arm}', metric=r.metric,
                            max_abs_difference=float(np.nanmax(np.abs([r.point - o.point, r.ci_lo - o.ci_lo, r.ci_hi - o.ci_hi])))))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    meta = analyze.runs()
    mm = analyze.load_mmpd(meta).assign(dataset='MMPD-A 760')
    mc, _ = analyze.load_mcd(meta)
    m723, refdiff = load_mmpd_723(meta)
    print('loaded', round(time.time() - t0), 'reference HR max difference 760 file vs automatic rule on the 723:', refdiff, flush=True)
    arms, ls, aa, counts = [], [], [], []
    for d in (m723, mm, mc):
        ds = d.dataset.iloc[0]
        dd = d[d.layer.eq('HAR') & d.seed.isin(SEEDS) & d.backbone.isin(['PhysNet', 'RhythmFormer'])]
        one = d[d.layer.eq('HAR') & d.run_id.eq(d.run_id.iloc[0])]
        counts.append(dict(dataset=ds, windows=len(one), high_windows=int((one.set == 'high').sum()), low_windows=int((one.set == 'low').sum()),
                           subjects=one.subject_id.nunique(), recordings=one.recording_id.nunique(), runs_in_common_support=d.run_id.nunique(),
                           reference_HR_max_difference_vs_760_file=refdiff if ds.endswith('723') else np.nan))
        arms += arm_rows(dd, ds)
        ls += lossswap(dd, ds)
        aa += analysis_a(dd, ds)
        print(ds, 'done', round(time.time() - t0), flush=True)
    arms, ls, aa = pd.DataFrame(arms), pd.DataFrame(ls), pd.DataFrame(aa)
    k = WORK / 'outputs' / 'guarded_v2' / 'keyframe'
    line = pd.read_csv(k / 'line_by_set.csv')
    line = line[line.dataset.eq('MCD-dev') & line.backbone.eq('PhysNet') & line.seeds.eq(3) & line.strategy.isin(['(0)'] + STD + RFL)]
    diff = pd.read_csv(k / 'line_diff.csv')
    diff = diff[diff.dataset.eq('MCD-dev') & diff.backbone.eq('PhysNet') & diff.contrast.isin(
        [f'{x[:-1]}-rfl)-{x}' for x in STD] + [f'{a[:-1]}-rfl)-{b[:-1]}-rfl)' for a, b in DELTAS] + [f'{a}-{b}' for a, b in DELTAS])]
    line = line.assign(source=f'{SRC}; copied from guarded_v2 keyframe/line_by_set.csv')
    diff = diff.assign(source=f'{SRC}; copied from guarded_v2 keyframe/line_diff.csv')
    chk = checks(ls, arms)
    write(arms, 'ra1_arms.csv', 'ra1')
    write(ls, 'ra1_lossswap.csv', 'ra1')
    write(aa, 'ra1_analysis_a.csv', 'ra1')
    write(line, 'ra1_keyframe_line.csv', 'ra1')
    write(diff, 'ra1_keyframe_diff.csv', 'ra1')
    write(pd.DataFrame(counts), 'ra1_counts.csv', 'ra1')
    write(chk, 'ra1_checks.csv', 'ra1')
    print('checks', len(chk), 'max abs difference', chk.max_abs_difference.max() if len(chk) else None)
    stamp('ra1', seconds=time.time() - t0, checks=len(chk),
          checks_max_abs_difference=float(chk.max_abs_difference.max()) if len(chk) else None)


if __name__ == '__main__':
    sys.exit(main())
