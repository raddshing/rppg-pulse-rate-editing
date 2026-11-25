"""Brief 02 7.01 다: PURE natural audit statistics with subject clusters (training corpus only; saved audit outputs only).

Inputs: the 59-recording audit table of 6.97 가 (outputs/facts_697/audit_PURE.csv, from outputs/1a/natural_recordings.csv)
and the 378 face contexts of 1a (outputs/1a/natural_contexts.csv, signal 'face'). Subject = the two digits before '-'.
Slopes (OLS, frames): signed maximum lag within -15..+15 (tau) and lag of the |corr| maximum within -30..+30, each against
the half period 900 / HR frames; recording level (59 rows; HR = median of the context HRs) and context level (378 rows;
context HR), as in 6.97 가-1.
Intervals: subject-cluster bootstrap (10 subjects resampled with replacement, all recordings / contexts of a drawn subject
kept with its multiplicity), B = 10,000, numpy default_rng(20261110); beside it the t interval of 6.97 가-1, the recording
bootstrap of 6.97 나 (B = 2,000, seed 20261117, copied from outputs/audit_cross/audit_cross_summary.csv) and a recording
bootstrap with B = 10,000 and default_rng(20261110). 95% percentile intervals; replicates whose half periods are all equal
are dropped (counted).
d and negative recordings by subject subset (01-08, 09-10, all): recordings whose |corr| maximum is negative, and the
median and quartiles (numpy percentile, linear) of the |corr|-maximum lag over those recordings (d) and over all recordings.
Writes outputs/posthoc701/da/{da_slopes,da_d_by_subjects,da_subjects}.csv.
"""
import csv
import time

import numpy as np
import pandas as pd
from scipy import stats

from p701_common import WORK, stamp, write

B, SEED = 10000, 20261110


def wslope(W, x, y):
    n = W.sum(1)
    mx, my = (W @ x) / n, (W @ y) / n
    with np.errstate(invalid='ignore', divide='ignore'):
        vx = (W @ (x * x)) / n - mx * mx
        s = ((W @ (x * y)) / n - mx * my) / vx
    return np.where(vx > 1e-12 * np.maximum(1, mx * mx), s, np.nan)


def boot_ci(x, y, groups):
    """Percentile interval of the OLS slope over B resamples of the groups (each group drawn with replacement)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    labels = sorted(set(groups))
    g = np.array([labels.index(v) for v in groups])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(labels), size=(B, len(labels)))
    mult = np.zeros((B, len(labels)))
    np.add.at(mult, (np.repeat(np.arange(B), len(labels)), idx.ravel()), 1)
    s = wslope(mult[:, g], x, y)
    ok = np.isfinite(s)
    return float(np.percentile(s[ok], 2.5)), float(np.percentile(s[ok], 97.5)), int((~ok).sum())


def t_ci(x, y):
    r = stats.linregress(np.asarray(x, float), np.asarray(y, float))
    t = stats.t.ppf(.975, len(x) - 2)
    return float(r.slope), float(r.slope - t * r.stderr), float(r.slope + t * r.stderr)


def q(x):
    x = np.asarray(x, float)
    if not len(x):
        return dict(median=np.nan, q25=np.nan, q75=np.nan)
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)))


def main():
    t0 = time.time()
    rec = pd.read_csv(WORK / 'outputs/facts_697/audit_PURE.csv', dtype={'recording_id': str})
    assert len(rec) == 59
    rec['subject'] = rec.recording_id.str.split('-').str[0]
    ctx = pd.read_csv(WORK / 'outputs/1a/natural_contexts.csv', dtype={'recording_id': str, 'subject_id': str})
    ctx = ctx[ctx.signal.eq('face')].copy()
    assert len(ctx) == 378
    ctx['subject'] = ctx.recording_id.str.split('-').str[0]
    assert (ctx.subject == ctx.subject_id.str.zfill(2)).all()
    ctx['half'] = 900. / ctx.HR.astype(float)
    prior = {r['corpus']: r for r in csv.DictReader(open(WORK / 'outputs/audit_cross/audit_cross_summary.csv'))}['PURE']
    rows = []
    for level, df, xcol, ycols in (('recording (59)', rec, 'half_period_frames',
                                    {'tau': 'tau_signed_max_lag_frames', 'abs': 'abs_max_lag_frames'}),
                                   ('context (378)', ctx, 'half', {'tau': 'L_sig15', 'abs': 'L_abs'})):
        for name, ycol in ycols.items():
            x, y = df[xcol].to_numpy(float), df[ycol].to_numpy(float)
            pt, tlo, thi = t_ci(x, y)
            slo, shi, sdrop = boot_ci(x, y, df.subject.tolist())
            r = dict(level=level, slope=('signed maximum lag (tau, -15..+15)' if name == 'tau' else '|corr| maximum lag (-30..+30)')
                     + ' vs half period, frames', n=len(df), subjects=df.subject.nunique(), point=pt,
                     subject_cluster_ci_lo=slo, subject_cluster_ci_hi=shi, subject_cluster_dropped_replicates=sdrop,
                     t_ci_lo=tlo, t_ci_hi=thi)
            if level.startswith('recording'):
                rlo, rhi, rdrop = boot_ci(x, y, df.recording_id.tolist())
                key = 'tau_signed_max_lag' if name == 'tau' else 'abs_max_lag'
                r.update(recording_boot10000_ci_lo=rlo, recording_boot10000_ci_hi=rhi, recording_boot10000_dropped_replicates=rdrop,
                         recording_boot2000_697na_point=float(prior[f'slope_{key}_vs_half_period_frames']),
                         recording_boot2000_697na_ci_lo=float(prior[f'slope_{key}_ci_lo_frames']),
                         recording_boot2000_697na_ci_hi=float(prior[f'slope_{key}_ci_hi_frames']))
            else:
                rlo, rhi, rdrop = boot_ci(x, y, df.recording_id.tolist())
                r.update(recording_cluster_boot10000_ci_lo=rlo, recording_cluster_boot10000_ci_hi=rhi,
                         recording_cluster_boot10000_dropped_replicates=rdrop)
            rows.append(r)
    slopes = pd.DataFrame(rows)
    dd = []
    for label, keep in (('01-08', [f'{i:02d}' for i in range(1, 9)]), ('09-10', ['09', '10']), ('all', sorted(rec.subject.unique()))):
        g = rec[rec.subject.isin(keep)]
        neg = g[g.abs_max_sign.eq(-1)]
        dd.append(dict(subjects=label, subject_count=g.subject.nunique(), recordings=len(g), negative_recordings=len(neg),
                       positive_recordings=int(g.abs_max_sign.eq(1).sum()), negative_fraction=len(neg) / len(g),
                       **{f'd_negative_{k}': v for k, v in q(neg.abs_max_lag_frames).items()},
                       **{f'abs_max_lag_all_{k}': v for k, v in q(g.abs_max_lag_frames).items()}))
    dd = pd.DataFrame(dd)
    expected = {f'{s:02d}-{t:02d}' for s in range(1, 11) for t in range(1, 7)}
    subj = rec.groupby('subject').agg(recordings=('recording_id', 'size'), recording_ids=('recording_id', lambda s: ' '.join(sorted(s))),
                                      negative_recordings=('abs_max_sign', lambda s: int((s == -1).sum()))).reset_index()
    subj['contexts'] = subj.subject.map(ctx.groupby('subject').size())
    subj['PURE_recordings_not_in_audit'] = subj.subject.map(lambda s: ' '.join(sorted(r for r in expected - set(rec.recording_id)
                                                                                     if r.startswith(s + '-'))))
    write(slopes, 'da_slopes.csv', 'da')
    write(dd, 'da_d_by_subjects.csv', 'da')
    write(subj, 'da_subjects.csv', 'da')
    print(slopes.round(3).to_string(index=False))
    print(dd.round(3).to_string(index=False))
    print(subj.to_string(index=False))
    stamp('da', seconds=time.time() - t0, seed=SEED, b_cluster=B)


if __name__ == '__main__':
    main()
