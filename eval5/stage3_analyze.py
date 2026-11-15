"""Step (c) of the Stage-5 evaluation: the analysis, run once after the stage-1 and stage-2 hashes are committed and step (b)
(eval5/stage3_check.py) is complete (ACCESS_OPS2_BRIEF_02 6.43 마, 6.45 나 4, 6.46 가 4/6/7, 6.48 마).

Data: stage-1 populations (references), step-(b) readouts, validity and window support; for the notch rules the stage-2
prediction files. Estimates and intervals use the code of the development analyses unchanged (b1/boot.py: crossed subject
x seed bootstrap, 10,000 resamples, draws default_rng(20261110) on the numeric subject order; b1/analyze.py families).
Window sets:
  confirmatory contrasts: the confirmatory support (step b); exploratory contrasts and sensitivity variants: the support
  intersected with the windows where every key of both arms of the contrast is valid (6.46 가 4, 6.48 마 2).
  MMPD: Q* windows (in_Qstar of the population file); high = reference HR >= 100. MCD-rPPG: draft-rule windows (high =
  agreement and >= 100, low = < 100) and M7 (both states).
Outputs (--out/<dataset>/<population>/):
  confirmatory.csv     C1-C4 per backbone, all metrics of the contrast family (high/low/all MAE, signed error, slope,
                       Spearman (7g: rank-correlation difference), Kendall, Pearson, AUROC), grades on MAE_high and MAE_low:
                       T1 upper < 0; T2 point < 0 and the interval contains 0; T3 point >= 0 and the interval contains 0;
                       T4 lower > 0
  secondary.csv        named secondary and exploratory contrasts, graded or 'values only' (A4 item 3), equivalence rule of
                       (i-simple)-(i-pc) (95% interval of the high-HR MAE difference within +-3.52 bpm on both datasets)
  MAE_verified         (6.70 나, A4 item 5) in confirmatory.csv, secondary.csv and arms.csv: MAE over all verified windows
                       (MMPD: every Q* window, the same windows as MAE_all; MCD-rPPG: the population windows whose three
                       reference estimates agree, high and low), same subjects, seeds and resampling; contrasts values only
  sensitivity.csv      C1-C4 under each sensitivity (narrow band; FactorizePhys FSAM off; FactorizePhys batch 1; A2
                       background exclusion (MMPD); October and main-lobe notch rules (MCD); RhythmFormer blur (MCD, if
                       its keys are present)); grade_changes.csv: grade change and largest absolute change per contrast
  arms.csv             M1-M3 per arm (seeds 20261131-33), ge120.csv (windows >= 120 bpm), m4_bins.csv (per 10-bpm reference
                       bin: MAE, signed error, +-1-bin hit rate, half-HR error rate), per_subject.csv, constants.csv,
                       line.csv (MCD key-frame line fraction 147.66-154.69 bpm by set), attraction.csv (MMPD background
                       bins 33-35 / 50-52, windows with reference outside those bins, flagged / unflagged recordings),
                       m7.csv (MCD within-person change: MAE, slope, CCC; and the contrasts on it)
  benchmark.csv        recording-level scores in the benchmark convention (eval5/benchmark.py; 6.46 가 7, 6.54 다): per arm the
                       seed mean and sample SD of MAE, RMSE, Pearson r and the MAE of recordings with reference HR >= 100;
                       benchmark_per_key.csv, benchmark_recordings.csv (per key and recording); written when stage 1 has
                       contact signals (labels/)
  analysis_hashes.txt, run.json (operator, start, end), log in the caller.
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'b1'))  # boot.py in the development workspace
sys.path.insert(0, str(HERE.parent))  # boot.py in the public package (evaluation/boot.py, identical)
import stage2_predict as S2  # noqa: E402
import boot  # noqa: E402
import families as AN  # noqa: E402  (the development analysis families, copied unchanged)

NAME = {'FactorizePhys-matched': 'FactorizePhys (matched)', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
PRIMARY = ('std', 'fsam-on')
SEEDS = [20261131, 20261132, 20261133]
GRID = 3.515625
CONF = [('C1', '(ii-pc)', '(0)'), ('C2', '(ii-pc)', '(ii)'), ('C3', '(i-pc)', '(i)'), ('C4', '(ii-pc)', '(i-pc)')]
SECONDARY = [('(i-pc)', '(0)', 'graded'), ('(i-simple)', '(0)', 'graded'), ('(i)', '(0)', 'values only'), ('(ii)', '(0)', 'values only'),
             ('(ii-pc)', '(i-simple)', 'graded'), ('(i-simple)', '(i-pc)', 'graded'),
             ('(i)', '(i-simple)', 'graded'), ('(ii)', '(i-simple)', 'values only'),  # brief 02 6.67 가
             ('(ii-lag)', '(ii-pc)', 'graded'),
             ('(ii-lag)', '(ii)', 'graded')] + [(a, c, 'graded') for a, c in AN.CLAB if a not in ('(ii-g)', '(ii-g2)')]
KEEP = ['MAE_high', 'MAE_low', 'MAE_all', 'signed_error_high', 'signed_error_low', 'signed_error_all',
        'slope_high', 'spearman_high', 'kendall_high', 'pearson_high', 'AUROC']
LINE = (147.65625, 154.6875)
BG = set(range(33, 36)) | set(range(50, 53))


def verified(g):
    """Brief 02 6.70 나, 6.71 가 (A4 item 5): windows with a verified reference, as the verification of Section 3, whatever the
    HR. MMPD: the Q* windows (no signal defect, three reference estimates agree), i.e. every window of the population;
    MCD-rPPG: the windows of the population (gap-free primary windows of stage 1) whose three reference estimates agree
    ('agreement'), high and low (the low set of the draft rule also holds windows without agreement). MAE_all is kept."""
    if 'agreement' in g:
        return g.agreement.astype(str).str.lower().isin(['true', '1']).to_numpy()
    return np.ones(len(g), bool)


def m_verified(W, r, p, s):
    """MAE over the verified windows (marked 'v' in the set column of a copy; same windows, subjects and resampling as m123)."""
    m = s == 'v'
    return dict(MAE_verified=boot.mae(W[:, m], r[m], p[m]) if m.any() else np.full(len(W), np.nan))


def mark_verified(g):
    return g.assign(set=np.where(verified(g), 'v', 'x'))


def values_only(bb, a, c):
    """A4 item 3 (6.48 바): values without grades or sentences."""
    if (a, c) in (('(i)', '(0)'), ('(ii)', '(0)'), ('(i-pc-g80)', '(i-pc)'), ('(roi-pc)', '(0)')):
        return True
    if bb == 'RhythmFormer' and (a, c) in (('(ii-lag)', '(ii-pc)'), ('(ii-lag)', '(ii)')):  # the two RhythmFormer mechanism contrasts
        return True
    if bb == 'FactorizePhys (matched)' and (a, c) in [(x, y) for x, y in AN.CLAB]:
        return True
    return False


def grade(pt, lo, hi):
    if not np.isfinite([pt, lo, hi]).all():
        return ''
    if hi < 0:
        return 'T1'
    if lo > 0:
        return 'T4'
    return 'T2' if pt < 0 else 'T3'


def load(dataset, s1, chk, keys, population):
    pops = pd.read_csv(s1 / 'populations.csv', dtype={'subject_id': str})
    rd = pd.read_csv(chk / 'readouts.csv.gz')
    sup = pd.read_csv(chk / 'support.csv', dtype={'subject_id': str})
    bad = set(pd.read_csv(chk / 'defects.csv').key.dropna())
    support = set(map(tuple, sup[sup.in_support][['window_id', 'layer']].to_numpy()))
    if dataset == 'MMPD':
        if population != 'automatic':  # rehearsal: an October A population file (window_id, layer, reference_HR, in_Qstar, high)
            o = pd.read_csv(population, dtype={'subject_id': str})
            pops = pops.drop(columns=['reference_HR', 'in_Qstar', 'high']).merge(o[['window_id', 'layer', 'reference_HR', 'in_Qstar', 'high']],
                                                                                 on=['window_id', 'layer'])
        pops = pops[pops.in_Qstar.astype(bool)].copy()
        pops['set'] = np.where(pops.high.astype(bool), 'high', 'low')
        pops['population'] = 'Qstar'
    else:
        refs = pd.read_csv(s1 / 'references.csv', dtype={'subject_id': str})[['window_id', 'layer', 'start_seconds']]
        pops = pops.merge(refs, on=['window_id', 'layer'])
    d = rd.merge(keys, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'] = dataset
    d['backbone'] = d.backbone.map(NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    return d, bad


def restrict(d):
    """Confirmatory support and the key's own valid windows."""
    return d[d.in_support & d.valid]


def pair_windows(ga, gc):
    """Windows valid for every key of both arms (inside the support)."""
    ka, kc = ga.key.nunique(), gc.key.nunique()
    va = ga.groupby(['window_id', 'layer']).key.nunique()
    vc = gc.groupby(['window_id', 'layer']).key.nunique()
    w = set(va[va == ka].index) & set(vc[vc == kc].index)
    pick = lambda g: g[[x in w for x in zip(g.window_id, g.layer)]]  # noqa: E731
    return pick(ga), pick(gc), len(w)


def contrast_rows(d, pairs, b, label):
    out = []
    for (ds, pop, layer), dd in d.groupby(['dataset', 'population', 'layer']):
        dd = restrict(dd)
        for bb in sorted(dd.backbone.unique()):
            for item in pairs:
                name, a, c, kind = item
                ga, gc = dd[dd.backbone.eq(bb) & dd.strategy.eq(a)], dd[dd.backbone.eq(bb) & dd.strategy.eq(c)]
                if ga.empty or gc.empty or set(SEEDS) - set(ga.seed) or set(SEEDS) - set(gc.seed):
                    continue
                ga, gc, nw = pair_windows(ga[ga.seed.isin(SEEDS)], gc[gc.seed.isin(SEEDS)])
                pt, ci = boot.contrast_ci(ga, gc, SEEDS, boot.m123, b)
                vo = kind == 'values only' or values_only(bb, a, c)
                for k in KEEP:
                    lo, hi = ci.get(k, (np.nan, np.nan))
                    out.append(dict(analysis=label, dataset=ds, population=pop, layer=layer, backbone=bb, name=name, contrast=f'{a}-{c}',
                                    windows=nw // 1, subjects=ga.subject_id.nunique(), metric=k, point=pt.get(k, np.nan), ci_lo=lo, ci_hi=hi,
                                    grade='' if vo or k not in ('MAE_high', 'MAE_low') else grade(pt.get(k, np.nan), lo, hi),
                                    values_only=vo))
                # 6.70 나 (A4 item 5): MAE over all verified windows, values only
                ptv, civ = boot.contrast_ci(mark_verified(ga), mark_verified(gc), SEEDS, m_verified, b)
                lo, hi = civ.get('MAE_verified', (np.nan, np.nan))
                nv = len(set(map(tuple, ga[verified(ga)][['window_id', 'layer']].to_numpy())))
                out.append(dict(analysis=label, dataset=ds, population=pop, layer=layer, backbone=bb, name=name, contrast=f'{a}-{c}',
                                windows=nv, subjects=ga.subject_id.nunique(), metric='MAE_verified', point=ptv.get('MAE_verified', np.nan),
                                ci_lo=lo, ci_hi=hi, grade='', values_only=True))
    return out


BINS = [('70-80', 70, 80), ('80-90', 80, 90), ('90-100', 90, 100), ('100-110', 100, 110), ('110-120', 110, 120), ('ge120', 120, np.inf)]
LAB_PAIRS = [(a, c) for a, c in AN.CLAB]
INTERACTION = (('(ii-pc-g2)', '(ii-pc-g)'), ('(ii-pc-g80)', '(ii-pc)'))


def m_bin(W, r, p, s):
    """MAE and signed error over all windows given (the bin's windows)."""
    return dict(MAE=boot.mae(W, r, p), signed_error=boot.bias(W, r, p))


def interaction(x, fn, b):
    """[(ii-pc-g2)-(ii-pc-g)] - [(ii-pc-g80)-(ii-pc)] on the same subject x seed resample; windows valid for all four arms."""
    arms = [s for pair in INTERACTION for s in pair]
    parts = [x[x.strategy.eq(a)] for a in arms]
    if not all(len(q) and not set(SEEDS) - set(q.seed) for q in parts):
        return None
    ks = [q.key.nunique() for q in parts]
    vs = [q.groupby(['window_id', 'layer']).key.nunique() for q in parts]
    w = set.intersection(*[set(v[v == k].index) for v, k in zip(vs, ks)])
    if not w:
        return None
    parts = [q[[t in w for t in zip(q.window_id, q.layer)]] for q in parts]
    subjects = boot.numeric_subject_order(set().union(*[set(q.subject_id) for q in parts]))
    smp = [boot.Sample(q, subjects, SEEDS) for q in parts]

    def f(W):
        v = [boot.per_seed(sm, fn, W) for sm in smp]
        return {sd: {k: (v[0][sd][k] - v[1][sd][k]) - (v[2][sd][k] - v[3][sd][k]) for k in v[0][sd]} for sd in SEEDS}
    pt, ci = boot.summarize(f, subjects, SEEDS, b)
    return pt, ci, len(w), len(subjects)


def bin_contrasts(pw, b):
    """6.45 나 3: per reference-HR bin paired differences (MAE, signed error) of the label contrasts and the interaction
    [(ii-pc-g2)-(ii-pc-g)] - [(ii-pc-g80)-(ii-pc)], the four arms on the same resample (same subject and seed draws)."""
    out = []
    dd = restrict(pw)
    for bb in sorted(dd.backbone.unique()):
        g = dd[dd.backbone.eq(bb) & dd.seed.isin(SEEDS)]
        for name, lo, hi in BINS:
            x = g[(g.reference_HR >= lo) & (g.reference_HR < hi)]
            for a, c in LAB_PAIRS:
                ga, gc = x[x.strategy.eq(a)], x[x.strategy.eq(c)]
                if ga.empty or gc.empty or set(SEEDS) - set(ga.seed) or set(SEEDS) - set(gc.seed):
                    continue
                ga, gc, nw = pair_windows(ga, gc)
                if nw == 0:
                    continue
                pt, ci = boot.contrast_ci(ga, gc, SEEDS, m_bin, b)
                for k in pt:
                    out.append(dict(backbone=bb, contrast=f'{a}-{c}', HR_bin=name, windows=nw, subjects=ga.subject_id.nunique(), metric=k,
                                    point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], values_only=values_only(bb, a, c)))
            res = interaction(x, m_bin, b)
            if res:
                pt, ci, nw, ns = res
                for k in pt:
                    out.append(dict(backbone=bb, contrast='interaction [(ii-pc-g2)-(ii-pc-g)]-[(ii-pc-g80)-(ii-pc)]', HR_bin=name, windows=nw,
                                    subjects=ns, metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], values_only=bb == 'FactorizePhys (matched)'))
        res = interaction(g, boot.m1, b)
        if res:
            pt, ci, nw, ns = res
            for k in ('MAE_high', 'MAE_low', 'signed_error_high', 'signed_error_low'):
                out.append(dict(backbone=bb, contrast='interaction [(ii-pc-g2)-(ii-pc-g)]-[(ii-pc-g80)-(ii-pc)]', HR_bin='set', windows=nw,
                                subjects=ns, metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], values_only=bb == 'FactorizePhys (matched)'))
    return out


def mmpd_labels(plan_dir, prefix):
    """October st17/descriptive.py metadata_B: the input plan's sort keys carry the MMPDLoader labels
    (_L<light>_MO<motion>_E<exercise>_S<skin>_); skin group 3-4 / 5-6, lighting, activity."""
    import re
    plan = pd.read_csv(Path(plan_dir) / f'{prefix}_FPm_input_plan.csv')
    out = []
    for rid, part in plan.groupby('recording_id', sort=False):
        values = {tuple(re.search(r'_L([1-4])_MO([1-4])_E([12])_S([3-6])_', v).groups()) for v in part.sort_key}
        assert len(values) == 1
        light, motion, exercise, skin = next(iter(values))
        out.append(dict(recording_id=rid, skin_group='3-4' if int(skin) <= 4 else '5-6',
                        lighting={'1': 'LED-low', '2': 'LED-high', '3': 'Incandescent', '4': 'Nature'}[light],
                        activity={'1': 'Stationary', '2': 'Rotation', '3': 'Talking', '4': 'Walking'}[motion]))
    return pd.DataFrame(out)


def subgroup_rows(pw, labels, b):
    """October st17 descriptive subgroups (skin group, lighting, activity) on the Q* wide-band windows: per arm the seed-mean
    MAE of the high (H) and low (L) windows, and the contrasts C1-C4 and (ii)-(0), (i)-(0), (ii)-(i) with intervals when the
    cell has at least 10 windows (crossed bootstrap, b1/boot.py)."""
    d = restrict(pw).merge(labels, on='recording_id')
    d['group'] = np.where(d.reference_HR >= 100, 'H', 'L')
    metrics, contrasts = [], []
    pairs = [(n, a, c) for n, a, c in CONF] + [('', '(ii)', '(0)'), ('', '(i)', '(0)'), ('', '(ii)', '(i)')]
    for dim in ('skin_group', 'lighting', 'activity'):
        for (cat, grp), cell in d.groupby([dim, 'group']):
            for (bb, st, sd), run in cell.groupby(['backbone', 'strategy', 'seed']):
                metrics.append(dict(dimension=dim, category=cat, group=grp, backbone=bb, strategy=st, seed=sd, windows=len(run),
                                    subjects=run.subject_id.nunique(), MAE=float((run.predicted_HR - run.reference_HR).abs().mean())))
            for bb in sorted(cell.backbone.unique()):
                for name, a, c in pairs:
                    ga, gc = cell[cell.backbone.eq(bb) & cell.strategy.eq(a)], cell[cell.backbone.eq(bb) & cell.strategy.eq(c)]
                    if ga.empty or gc.empty or set(SEEDS) - set(ga.seed) or set(SEEDS) - set(gc.seed):
                        continue
                    ga, gc, nw = pair_windows(ga[ga.seed.isin(SEEDS)], gc[gc.seed.isin(SEEDS)])
                    enough = nw >= 10
                    pt, ci = boot.contrast_ci(ga.assign(set='all'), gc.assign(set='all'), SEEDS, boot.m1, b if enough else 0)
                    contrasts.append(dict(dimension=dim, category=cat, group=grp, backbone=bb, name=name, contrast=f'{a}-{c}', windows=nw,
                                          subjects=ga.subject_id.nunique(), MAE_difference=pt['MAE_all'],
                                          ci_lo=ci.get('MAE_all', (np.nan, np.nan))[0] if enough else np.nan,
                                          ci_hi=ci.get('MAE_all', (np.nan, np.nan))[1] if enough else np.nan,
                                          interval_reported=enough, no_hypothesis_test=True))
    m = pd.DataFrame(metrics)
    seed_mean = m[m.seed.isin(SEEDS)].groupby(['dimension', 'category', 'group', 'backbone', 'strategy']).agg(
        seeds=('seed', 'nunique'), windows=('windows', 'first'), subjects=('subjects', 'first'), MAE=('MAE', 'mean')).reset_index()
    return m, seed_mean, pd.DataFrame(contrasts)


def common_window_rows(pw, b):
    """MCD common-window sensitivity (tools/keyframe_common.py): for each confirmatory pair and seed, the windows on which
    either arm (same seed) is read on the key-frame line (147.66-154.69 bpm) are removed from both; contrasts on the rest."""
    out = []
    dd = restrict(pw)
    for bb in sorted(dd.backbone.unique()):
        for name, a, c in CONF:
            ga, gc = dd[dd.backbone.eq(bb) & dd.strategy.eq(a)], dd[dd.backbone.eq(bb) & dd.strategy.eq(c)]
            if ga.empty or gc.empty or set(SEEDS) - set(ga.seed) or set(SEEDS) - set(gc.seed):
                continue
            ga, gc, _ = pair_windows(ga[ga.seed.isin(SEEDS)], gc[gc.seed.isin(SEEDS)])
            keep_a, keep_c, removed = [], [], {}
            for sd in SEEDS:
                xa, xc = ga[ga.seed.eq(sd)], gc[gc.seed.eq(sd)]
                on = set(xa.window_id[(xa.predicted_HR >= LINE[0]) & (xa.predicted_HR <= LINE[1])]) | \
                    set(xc.window_id[(xc.predicted_HR >= LINE[0]) & (xc.predicted_HR <= LINE[1])])
                removed[sd] = len(on)
                keep_a.append(xa[~xa.window_id.isin(on)])
                keep_c.append(xc[~xc.window_id.isin(on)])
            pt, ci = boot.contrast_ci(pd.concat(keep_a), pd.concat(keep_c), SEEDS, boot.m1, b)
            for k in ('MAE_high', 'MAE_low', 'signed_error_high', 'signed_error_low'):
                out.append(dict(backbone=bb, name=name, contrast=f'{a}-{c}', metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1],
                                grade=grade(pt[k], *ci[k]) if k.startswith('MAE') else '',
                                **{f'removed_s{sd}': removed[sd] for sd in SEEDS}))
    return out


def arm_rows(d, b, label):
    out = []
    for (ds, pop, layer), dd in d.groupby(['dataset', 'population', 'layer']):
        dd = restrict(dd)
        for (bb, st), g in dd.groupby(['backbone', 'strategy']):
            g = g[g.seed.isin(SEEDS)]
            if set(SEEDS) - set(g.seed):
                continue
            k = g.key.nunique()
            v = g.groupby(['window_id', 'layer']).key.nunique()
            w = set(v[v == k].index)
            g = g[[x in w for x in zip(g.window_id, g.layer)]]
            pt, ci = boot.point_ci(g, SEEDS, boot.m123, b)
            for m in KEEP:
                lo, hi = ci.get(m, (np.nan, np.nan))
                out.append(dict(analysis=label, dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st, windows=len(w),
                                subjects=g.subject_id.nunique(), metric=m, point=pt.get(m, np.nan), ci_lo=lo, ci_hi=hi))
            # 6.70 나 (A4 item 5): MAE over all verified windows
            ptv, civ = boot.point_ci(mark_verified(g), SEEDS, m_verified, b)
            lo, hi = civ.get('MAE_verified', (np.nan, np.nan))
            out.append(dict(analysis=label, dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st,
                            windows=len(set(map(tuple, g[verified(g)][['window_id', 'layer']].to_numpy()))),
                            subjects=g.subject_id.nunique(), metric='MAE_verified', point=ptv.get('MAE_verified', np.nan), ci_lo=lo, ci_hi=hi))
    return out


def notch_readouts(d, stage2_dirs, keys):
    """MCD: HR under the October (2.45-2.55 Hz, bins 42-43) and main-lobe (centres 2.40-2.60 Hz, bins 41-44) rules, the
    periodogram of the primary readout (detrend False) with those bins set to zero (tools/mcd_notch_variants.py)."""
    from scipy.signal import periodogram
    from src.eval import grid
    from src.st11x.pathways import uniform_frame_grid, complete_windows
    out = {}
    refs = None
    for folder in stage2_dirs:
        s2 = Path(folder) / 'MCD'
        recs = pd.read_csv(s2 / 'recordings.csv').set_index('recording_id')
        ck = pd.read_csv(s2 / 'checkpoints.csv')
        for r in ck[ck.status.eq('used')].itertuples():
            g = d[d.key.eq(r.key)]
            if g.empty:
                continue
            diff = r.view == 'PhysNet' and '-std' not in r.arm
            for rid, w in g.groupby('recording_id'):
                pred = np.load(s2 / 'pred' / r.key / (rid + '.npy'))  # float32 as stored (tools/mcd_notch_variants.py)
                n, span = int(recs.loc[rid, 'decoded_frames']), float(recs.loc[rid, 'span_seconds'])
                fg = uniform_frame_grid(np.arange(n, dtype=np.float64) * span / (n - 1))
                for x in w.itertuples():
                    bd = complete_windows(fg, [x.start_seconds])[0]
                    s = bd['grid_start']
                    if not bd['available'] or s + 300 > len(pred):
                        continue
                    band = AN_BANDS[x.layer]
                    pw = grid._prepare(pred[s:s + 300], diff, band, 'default', 30)
                    f, p0 = periodogram(pw, fs=30, nfft=512, detrend=False)
                    inside = (f >= band[0]) & (f <= band[1])
                    for rule, bins in (('october', [42, 43]), ('mainlobe', [41, 42, 43, 44])):
                        p = p0.copy()
                        p[bins] = 0
                        out[(r.key, x.window_id, x.layer, rule)] = float(f[inside][np.argmax(p[inside])] * 60)
    return out


AN_BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=['MMPD', 'MCD'])
    ap.add_argument('--stage1', required=True), ap.add_argument('--check', required=True)
    ap.add_argument('--stage2', nargs='+', required=True)
    ap.add_argument('--population', default='automatic', help="MMPD: 'automatic' (stage 1) or an October A population CSV (rehearsal)")
    ap.add_argument('--tag', default='automatic')
    ap.add_argument('--b', type=int, default=boot.B)
    ap.add_argument('--vendor', required=True), ap.add_argument('--out', required=True)
    ap.add_argument('--constants', help='JSON {"mean": .., "median": ..} of the development part (6.46 가 7)')
    ap.add_argument('--plan-dir', help='MMPD: input plan folder (sort keys -> subgroup labels)'), ap.add_argument('--plan-prefix', default='st30e_A')
    args = ap.parse_args()
    vendor = Path(args.vendor).resolve()
    sys.path.insert(0, str(vendor))
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    s1, chk = Path(args.stage1) / args.dataset, Path(args.check) / args.dataset
    keys = pd.concat([pd.read_csv(Path(x) / args.dataset / 'checkpoints.csv') for x in args.stage2], ignore_index=True)
    keys = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    d, bad = load(args.dataset, s1, chk, keys, args.population)
    out = Path(args.out) / args.dataset / args.tag
    out.mkdir(parents=True, exist_ok=True)
    prim = d[d.variant.isin(PRIMARY)]
    wide = prim[prim.layer.eq('HAR')]
    main_pop = 'Qstar' if args.dataset == 'MMPD' else 'draft_rule'
    pw = wide[wide.population.eq(main_pop)]
    conf = [(n, a, c, 'graded') for n, a, c in CONF]
    rows = contrast_rows(pw, conf, args.b, 'primary')
    pd.DataFrame(rows).to_csv(out / 'confirmatory.csv', index=False)
    sec = contrast_rows(pw, [('', a, c, k) for a, c, k in SECONDARY], args.b, 'primary')
    eq = [r for r in sec if r['contrast'] == '(i-simple)-(i-pc)' and r['metric'] == 'MAE_high']
    for r in eq:
        r['equivalence_within_3.52'] = bool(-3.52 < r['ci_lo'] and r['ci_hi'] < 3.52)
    pd.DataFrame(sec).to_csv(out / 'secondary.csv', index=False)
    pd.DataFrame(bin_contrasts(pw, args.b)).to_csv(out / 'label_bins.csv', index=False)
    # sensitivities
    sens = contrast_rows(prim[prim.layer.eq('L0') & prim.population.eq(main_pop)], conf, args.b, 'narrow_band')
    for var in ('fsam-off', 'fsam-on-batch1'):
        z = d[d.variant.eq(var) & d.layer.eq('HAR') & d.population.eq(main_pop)]
        if len(z):
            sens += contrast_rows(z, conf, args.b, var.replace('fsam-on-batch1', 'batch1'))
    blur = d[d.variant.str.startswith('blur') & d.layer.eq('HAR') & d.population.eq(main_pop)]
    if len(blur):
        sens += contrast_rows(blur, conf, args.b, 'blur15')
    if args.dataset == 'MMPD':
        bg = pd.read_csv(s1 / 'bg_flags.csv') if (s1 / 'bg_flags.csv').exists() else None
        if bg is not None:
            keep = set(bg.recording_id[bg.background_flag.eq('unflagged')])
            sens += contrast_rows(pw[pw.recording_id.isin(keep)], conf, args.b, 'background_excluded')
    else:
        nr = notch_readouts(pw, args.stage2, keys)
        for rule in ('october', 'mainlobe'):
            z = pw.copy()
            z['predicted_HR'] = [nr.get((k, w, l, rule), np.nan) for k, w, l in zip(z.key, z.window_id, z.layer)]
            z['valid'] = z.valid & np.isfinite(z.predicted_HR)
            sens += contrast_rows(z, conf, args.b, f'notch_{rule}')
    pd.DataFrame(sens).to_csv(out / 'sensitivity.csv', index=False)
    base = pd.DataFrame(rows)
    ch = []
    if len(sens):
        s = pd.DataFrame(sens)
        k = ['dataset', 'layer', 'backbone', 'name', 'metric']
        for an, g in s.groupby('analysis'):
            m = base[base.metric.isin(['MAE_high', 'MAE_low'])].merge(g[g.metric.isin(['MAE_high', 'MAE_low'])], on=['dataset', 'backbone', 'name', 'metric'], suffixes=('', '_s'))
            for r in m.itertuples():
                ch.append(dict(analysis=an, dataset=r.dataset, backbone=r.backbone, name=r.name, metric=r.metric, grade=r.grade,
                               grade_sensitivity=r.grade_s, changed=r.grade != r.grade_s, point=r.point, point_sensitivity=r.point_s,
                               abs_change=abs(r.point_s - r.point)))
    pd.DataFrame(ch).to_csv(out / 'grade_changes.csv', index=False)
    # arms, >= 120, bins, per subject, line, attraction
    arms = arm_rows(prim[prim.population.eq(main_pop)], args.b, 'primary')
    pd.DataFrame(arms).to_csv(out / 'arms.csv', index=False)
    g120 = pw[pw.reference_HR >= 120].copy()
    pd.DataFrame(arm_rows(g120, args.b, 'ge120') + contrast_rows(g120, conf, args.b, 'ge120')).to_csv(out / 'ge120.csv', index=False)
    pd.DataFrame(AN.fam_m4(restrict(prim[prim.population.eq(main_pop)]), [])).to_csv(out / 'm4_bins.csv', index=False)
    r = restrict(pw)
    ps = (r.assign(err=(r.predicted_HR - r.reference_HR).abs()).groupby(['backbone', 'strategy', 'subject_id', 'set', 'seed']).err.mean()
          .groupby(['backbone', 'strategy', 'subject_id', 'set']).mean().rename('MAE_seed_mean').reset_index())
    ps.to_csv(out / 'per_subject.csv', index=False)
    rr = restrict(prim[prim.population.eq(main_pop)])
    line = rr.assign(in_line=(rr.predicted_HR >= LINE[0]) & (rr.predicted_HR <= LINE[1])).groupby(
        ['layer', 'backbone', 'strategy', 'set', 'seed']).in_line.mean().reset_index()
    line.to_csv(out / 'line.csv', index=False)
    if args.dataset == 'MMPD' and args.plan_dir:
        m, sm, sc = subgroup_rows(pw, mmpd_labels(args.plan_dir, args.plan_prefix), args.b)
        m.to_csv(out / 'subgroup_seed_metrics.csv', index=False)
        sm.to_csv(out / 'subgroup_metrics.csv', index=False)
        sc.to_csv(out / 'subgroup_contrasts.csv', index=False)
    if args.dataset == 'MMPD' and (s1 / 'bg_flags.csv').exists():
        bg = pd.read_csv(s1 / 'bg_flags.csv').set_index('recording_id').background_flag
        z = r.assign(flag=r.recording_id.map(bg), ref_bin=np.round(r.reference_HR / GRID).astype(int), pred_bin=np.round(r.predicted_HR / GRID))
        z = z[~z.ref_bin.isin(BG)]
        att = z.assign(in_bg=z.pred_bin.isin(BG)).groupby(['backbone', 'strategy', 'flag', 'seed']).in_bg.agg(['mean', 'size']).reset_index()
        att.to_csv(out / 'attraction.csv', index=False)
    # constant predictors (6.46 가 7 / 6.48 라: development-part reference mean and median, same code path)
    if args.constants:
        c = json.loads(Path(args.constants).read_text())
        one = r[r.key.eq(r.key.iloc[0])][['window_id', 'subject_id', 'reference_HR', 'set']]
        crow = []
        for kind in ('mean', 'median'):
            t = one.assign(predicted_HR=float(c[kind]), seed=0)
            pt, ci = boot.point_ci(t, [0], boot.m1, args.b)
            for m, v in pt.items():
                crow.append(dict(constant=kind, value=c[kind], metric=m, point=v, ci_lo=ci.get(m, (np.nan, np.nan))[0], ci_hi=ci.get(m, (np.nan, np.nan))[1],
                                 windows=len(t), subjects=t.subject_id.nunique()))
        pd.DataFrame(crow).to_csv(out / 'constants.csv', index=False)
    if args.dataset == 'MCD':
        pd.DataFrame(common_window_rows(pw, args.b)).to_csv(out / 'common_windows.csv', index=False)
        m7 = restrict(prim[prim.population.eq('M7_both_states')])
        pd.DataFrame(AN.fam_m7(m7, [], args.b)).to_csv(out / 'm7.csv', index=False)
    if (s1 / 'labels.csv').exists():  # recording-level benchmark scores (descriptive)
        import benchmark
        brec, bkey, bsum = benchmark.run(args.dataset, s1, args.stage2, keys, bad, NAME)
        brec.to_csv(out / 'benchmark_recordings.csv', index=False)
        bkey.to_csv(out / 'benchmark_per_key.csv', index=False)
        bsum.to_csv(out / 'benchmark.csv', index=False)
    names = sorted(p.name for p in out.iterdir() if p.suffix == '.csv')
    (out / 'analysis_hashes.txt').write_text(''.join(f'{S2.file_sha(out / n)}  {args.dataset}/{args.tag}/{n}\n' for n in names))
    run = dict(step='c', dataset=args.dataset, population=args.population if args.population == 'automatic' else Path(args.population).name,
               started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(), b=args.b,
               excluded_keys=sorted(bad), windows_primary=int(restrict(pw).drop_duplicates(['window_id', 'layer']).shape[0]))
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
