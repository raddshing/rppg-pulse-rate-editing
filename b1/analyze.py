"""1b metrics M1-M9 and 2.5 (ii-pc) contrasts on MMPD-A and MCD-dev (brief 01 1b, brief 02 1, 2.5, 6.2).

Populations
- MMPD-A: the draft's 760 validation windows (in_Qstar; HAR 760/141 high, L0 758/140 high); high = draft 'high'.
- MCD-dev draft rule (M1-M6, M8 window metrics, M9): front-webcam dev recordings with a confirmed mapping and
  >= 1 HAR window that is gap-free (primary_included), reference-agreement and h_FFT >= 100 (draft
  st11x cohort_selection: primary = confirmed & agreement & h >= 100, low = h < 100). In those recordings,
  per layer: high = gap-free & agreement & h_FFT >= 100, low = gap-free & h_FFT < 100; other windows unused.
- MCD-dev M7: every dev subject with confirmed before and after recordings; per recording the mean of the
  reference h_FFT and of the predicted HR over the gap-free finite-reference windows; delta = after - before.
- Common support: windows predicted by every model in the run set (draft 'all models predicted').
Estimates: seed mean (seeds 20261131-33; 5-seed strategies also with 20261143/44 separately) and 95%
subject x seed crossed bootstrap, 10,000 replicates (b1/boot.py).
Usage: analyze.py [--b N] [--backbones a,b] [--families m123,m4,m5,m6,m7,m8,m9,c25,cipc,cinv,clag] [--tag name]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import boot  # noqa: E402
import model_sets  # noqa: E402

OUT1B = Path(os.environ.get('OPS2_OUT1B', WORK / 'outputs' / '1b'))  # 6.54: guarded readouts (eval5/stage2_to_dev.py)
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


def runs():
    out = {r['run_id']: r for r in model_sets.roster()}
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for r in ((model_sets.iipc(b) or []) + (model_sets.ipc(b) or []) + (model_sets.iinv(b) or []) + (model_sets.ilag(b) or [])
                  + (model_sets.isimple(b) or []) + sum((model_sets.lab(b, a) or [] for a in model_sets.LAB_ARMS), [])
                  + (model_sets.rfl(b) or [] if b == 'PhysNet' else []) + (model_sets.rfli(b) or [] if b == 'PhysNet' else [])):
            out[r['run_id']] = dict(r, backbone=BACKBONE[b])
    return out


def attach(df, meta):
    df = df[df.run_id.isin(meta)].copy()
    for k in ('backbone', 'strategy', 'seed'):
        df[k] = df.run_id.map(lambda x: meta[x][k])
    return df


def support(df):
    """Windows (per layer) available for every run present."""
    n_runs = df.run_id.nunique()
    ok = df[df.available].groupby(['window_id', 'layer']).run_id.nunique()
    keep = ok[ok == n_runs].index
    return df.set_index(['window_id', 'layer']).loc[lambda d: d.index.isin(keep)].reset_index()


def load_mmpd(meta):
    parts = [pd.read_csv(OUT1B / 'mmpd_a_windows.csv.gz', dtype={'subject_id': str})]
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for pre in ('iipc', 'ipc', 'iinv', 'ilag', 'isimple', 'rfl', 'rfli') + tuple('lab-' + a for a in model_sets.LAB_ARMS):
            f = OUT1B / f'mmpd_a_windows_{pre}_{b}.csv.gz'
            if f.exists():
                parts.append(pd.read_csv(f, dtype={'subject_id': str}))
    d = attach(pd.concat(parts, ignore_index=True), meta)
    d = d[d.in_Qstar.astype(bool)]
    d = support(d)
    d['set'] = np.where(d.high.astype(bool), 'high', 'low')
    d['dataset'], d['population'] = 'MMPD-A', 'validation'
    return d


def load_mcd(meta):
    w = pd.read_csv(OUT1B / 'mcd_dev_windows.csv', dtype={'subject_id': str})
    recs = pd.read_csv(OUT1B / 'mcd_dev_recordings.csv', dtype={'subject_id': str})
    parts = [pd.read_csv(f) for f in sorted(OUT1B.glob('mcd_dev_preds_*.csv.gz'))]
    p = attach(pd.concat(parts, ignore_index=True), meta)
    p = p.merge(w[['window_id', 'layer', 'subject_id', 'state', 'recording_id', 'h_FFT', 'agreement', 'primary_included']],
                on=['window_id', 'layer'], how='left')
    assert p.subject_id.notna().all()
    p = p.rename(columns={'h_FFT': 'reference_HR'})
    p = p[p.primary_included.astype(bool) & np.isfinite(p.reference_HR)]
    confirmed = set(recs.recording_id[recs.mapping_confirmed.astype(bool)])
    har = w[w.layer.eq('HAR') & w.primary_included & w.agreement & w.h_FFT.ge(100)]
    draft_recs = set(har.recording_id) & confirmed
    d = p[p.recording_id.isin(draft_recs)].copy()
    d['set'] = np.where(d.agreement.astype(bool) & (d.reference_HR >= 100), 'high', np.where(d.reference_HR < 100, 'low', ''))
    d = support(d[d.set.ne('')])
    d['dataset'], d['population'] = 'MCD-dev', 'draft_rule'
    st = recs[recs.mapping_confirmed.astype(bool)].groupby('subject_id').state.nunique()
    both = set(st[st == 2].index)
    m7 = support(p[p.subject_id.isin(both) & p.recording_id.isin(confirmed)].copy())
    m7['set'] = np.where(m7.reference_HR >= 100, 'high', 'low')
    m7['dataset'], m7['population'] = 'MCD-dev', 'M7_both_states'
    return d, m7


def counts(d):
    rows = []
    for (ds, pop, layer), g in d.groupby(['dataset', 'population', 'layer']):
        one = g[g.run_id.eq(g.run_id.iloc[0])]
        rows.append(dict(dataset=ds, population=pop, layer=layer, runs=g.run_id.nunique(), windows=len(one),
                         high=int((one.set == 'high').sum()), low=int((one.set == 'low').sum()),
                         recordings=one.recording_id.nunique(), subjects=one.subject_id.nunique(),
                         subjects_high=one[one.set == 'high'].subject_id.nunique()))
    return rows


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


# ---------------- families ----------------
def fam_m123(d, backbones, b):
    out = []
    for (ds, pop, layer), dd in d.groupby(['dataset', 'population', 'layer']):
        for bb, st in models(dd, backbones):
            g = dd[dd.backbone.eq(bb) & dd.strategy.eq(st)]
            for tag, seeds in seedsets(g):
                x = g[g.seed.isin(seeds)]
                pt, ci = boot.point_ci(x, seeds, boot.m123, b)
                out += rowify(dict(dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st, seeds=tag,
                                   windows=len(x) // len(seeds), subjects=x.subject_id.nunique()), pt, ci)
    return out


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


def fam_m5(mm, mc, b):
    """Constant predictors: MMPD-A leave-one-subject-out; MCD-dev with the full MMPD-A value."""
    out = []
    for layer in ('HAR', 'L0'):
        a = mm[mm.layer.eq(layer)]
        a = a[a.run_id.eq(a.run_id.iloc[0])][['window_id', 'subject_id', 'reference_HR', 'set']]
        full = dict(dev_mean=a.reference_HR.mean(), dev_median=a.reference_HR.median())
        targets = [('MMPD-A', 'validation', a)]
        c = mc[mc.layer.eq(layer)]
        targets.append(('MCD-dev', 'draft_rule', c[c.run_id.eq(c.run_id.iloc[0])][['window_id', 'subject_id', 'reference_HR', 'set']]))
        for ds, pop, t in targets:
            for kind in ('dev_mean', 'dev_median', 'PURE_label_mean'):
                t = t.copy()
                if kind == 'PURE_label_mean':
                    t['predicted_HR'] = PURE_LABEL_MEAN
                elif ds == 'MMPD-A':
                    f = np.mean if kind == 'dev_mean' else np.median
                    t['predicted_HR'] = t.subject_id.map({s: f(a.reference_HR[a.subject_id != s]) for s in a.subject_id.unique()})
                else:
                    t['predicted_HR'] = full[kind]
                t['seed'] = 0
                pt, ci = boot.point_ci(t, [0], boot.m1, b)
                const = full.get(kind, PURE_LABEL_MEAN) if ds == 'MCD-dev' or kind == 'PURE_label_mean' else np.nan
                out += rowify(dict(dataset=ds, population=pop, layer=layer, constant=kind, fit='leave-one-subject-out' if ds == 'MMPD-A' and kind != 'PURE_label_mean' else 'fixed',
                                   constant_value=const, windows=len(t), subjects=t.subject_id.nunique()), pt, ci)
    return out


def recal(fit_p, fit_r, apply_p, kind):
    if kind == 'linear':
        b1, b0 = np.polyfit(fit_p, fit_r, 1)
        return b0 + b1 * apply_p
    return IsotonicRegression(increasing=True, out_of_bounds='clip').fit(fit_p, fit_r).predict(apply_p)


def fam_m6(mm, mc, backbones, b):
    out = []
    for layer in ('HAR', 'L0'):
        A = mm[mm.layer.eq(layer)]
        C = mc[mc.layer.eq(layer)]
        for bb, st in models(A, backbones):
            ga, gc = A[A.backbone.eq(bb) & A.strategy.eq(st)], C[C.backbone.eq(bb) & C.strategy.eq(st)]
            for kind in ('linear', 'isotonic'):
                ra, rc = [], []
                for run, x in ga.groupby('run_id'):
                    x = x.copy()
                    vals = np.empty(len(x))
                    for s in x.subject_id.unique():
                        m = x.subject_id.eq(s).to_numpy()
                        vals[m] = recal(x.predicted_HR[~m].to_numpy(), x.reference_HR[~m].to_numpy(), x.predicted_HR[m].to_numpy(), kind)
                    ra.append(x.assign(predicted_HR=vals))
                    y = gc[gc.run_id.eq(run)].copy()
                    if len(y):
                        rc.append(y.assign(predicted_HR=recal(x.predicted_HR.to_numpy(), x.reference_HR.to_numpy(), y.predicted_HR.to_numpy(), kind)))
                for ds, pop, parts in [('MMPD-A', 'validation', ra), ('MCD-dev', 'draft_rule', rc)]:
                    if not parts:
                        continue
                    g = pd.concat(parts)
                    for tag, seeds in seedsets(g):
                        x = g[g.seed.isin(seeds)]
                        pt, ci = boot.point_ci(x, seeds, boot.m1, b)
                        out += rowify(dict(dataset=ds, population=pop, layer=layer, backbone=bb, strategy=st, seeds=tag,
                                           recalibration=kind, fit='MMPD-A leave-one-subject-out' if ds == 'MMPD-A' else 'MMPD-A all',
                                           windows=len(x) // len(seeds), subjects=x.subject_id.nunique()), pt, ci)
    return out


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


def contrasts(d, m7, pairs, backbones, b, family):
    out = []
    keep = ['MAE_high', 'MAE_low', 'MAE_all', 'signed_error_high', 'signed_error_low', 'signed_error_all',
            'slope_high', 'spearman_high', 'kendall_high', 'pearson_high', 'AUROC']
    if family == 'M8':
        keep = keep[:6] + ['spearman_high', 'AUROC']
    for (ds, pop, layer), dd in d.groupby(['dataset', 'population', 'layer']):
        for bb in sorted(dd.backbone.unique()):
            if backbones and bb not in backbones:
                continue
            for a, c in pairs:
                ga, gc = dd[dd.backbone.eq(bb) & dd.strategy.eq(a)], dd[dd.backbone.eq(bb) & dd.strategy.eq(c)]
                if ga.empty or gc.empty:
                    continue
                for tag, seeds in seedsets(ga):
                    if not set(seeds) <= set(gc.seed):
                        continue
                    pt, ci = boot.contrast_ci(ga[ga.seed.isin(seeds)], gc[gc.seed.isin(seeds)], seeds, boot.m123, b)
                    pt = {k: v for k, v in pt.items() if k in keep}
                    out += rowify(dict(family=family, dataset=ds, population=pop, layer=layer, backbone=bb, contrast=f'{a}-{c}', seeds=tag), pt, ci)
    for layer, dd in m7.groupby('layer'):
        for bb in sorted(dd.backbone.unique()):
            if backbones and bb not in backbones:
                continue
            for a, c in pairs:
                ga, gc = dd[dd.backbone.eq(bb) & dd.strategy.eq(a)], dd[dd.backbone.eq(bb) & dd.strategy.eq(c)]
                if ga.empty or gc.empty:
                    continue
                ta, tc = m7_table(ga), m7_table(gc)
                for tag, seeds in seedsets(ta):
                    if not set(seeds) <= set(tc.seed):
                        continue
                    pt, ci = boot.contrast_ci(ta[ta.seed.isin(seeds)], tc[tc.seed.isin(seeds)], seeds, boot.m7, b)
                    out += rowify(dict(family=family, dataset='MCD-dev', population='M7_both_states', layer=layer, backbone=bb,
                                       contrast=f'{a}-{c}', seeds=tag), pt, ci)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=boot.B)
    ap.add_argument('--backbones', default='')
    ap.add_argument('--families', default='m123,m4,m5,m6,m7,m8,m9,c25')
    ap.add_argument('--tag', default='1b')
    a = ap.parse_args()
    backbones = [x for x in a.backbones.split(',') if x]
    fams = a.families.split(',')
    dest = WORK / 'outputs' / a.tag
    dest.mkdir(parents=True, exist_ok=True)
    meta = runs()
    t0 = time.time()
    mm = load_mmpd(meta)
    mc, m7 = load_mcd(meta)
    d = pd.concat([mm, mc], ignore_index=True)
    pd.DataFrame(counts(pd.concat([d, m7], ignore_index=True))).to_csv(dest / 'counts.csv', index=False)
    print('loaded', len(d), len(m7), f'{time.time() - t0:.0f}s', flush=True)
    jobs = dict(m123=lambda: fam_m123(d, backbones, a.b), m4=lambda: fam_m4(d, backbones),
                m5=lambda: fam_m5(mm, mc, a.b), m6=lambda: fam_m6(mm, mc, backbones, a.b),
                m7=lambda: fam_m7(m7, backbones, a.b), m8=lambda: contrasts(d, m7, M8, backbones, a.b, 'M8'),
                m9=lambda: fam_m9(d, backbones, a.b), c25=lambda: contrasts(d, m7, C25, backbones, a.b, '2.5'),
                cipc=lambda: contrasts(d, m7, CIPC, backbones, a.b, '6.12'),
                cinv=lambda: contrasts(d, m7, CINV, backbones, a.b, '6.18'),
                clag=lambda: contrasts(d, m7, CLAG, backbones, a.b, '6.19'),
                crfl=lambda: contrasts(d, m7, CRFL, backbones, a.b, '6.37'),
                cstd=lambda: contrasts(d, m7, CSTD, backbones, a.b, '6.37'),
                closs=lambda: contrasts(d, m7, CLOSS, backbones, a.b, '6.42'),
                csimple=lambda: contrasts(d, m7, CSIMPLE, backbones, a.b, '6.43da'),
                clab=lambda: contrasts(d, m7, CLAB, backbones, a.b, '6.43ra'))
    for f in fams:
        res = jobs[f]()
        pd.DataFrame(res).to_csv(dest / f'{f}.csv', index=False)
        print(f, len(res), f'{time.time() - t0:.0f}s', flush=True)
    (dest / 'run.json').write_text(json.dumps(dict(args=vars(a), seconds=time.time() - t0, runs=sorted(set(d.run_id))), indent=1))


if __name__ == '__main__':
    main()
