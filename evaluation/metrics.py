"""1b metrics M1-M9 with subject x seed crossed bootstrap (brief 01 1b, brief 02 1b M9).

Input: long table W with one row per (run, window, layer):
  dataset, run_id, backbone, strategy, seed, window_id, subject_id, recording_id, state, layer,
  reference_HR, predicted_HR, set ('high' or 'low'), phase, pearson_lag0.
Bootstrap (as the draft's st11e/st12e setup): rng = default_rng(20261110); subject indices
integers(0, n_subjects, (B, n_subjects)) then seed indices integers(0, n_seeds, (B, n_seeds));
windows are weighted by subject multiplicity; each replicate averages the metric over the
resampled seeds. Percentile 2.5/97.5.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

GRID = 3.515625
B = 10000
HR_BINS = list(range(40, 181, 10))


def boot_indices(n_sub, n_seed, b=B):
    rng = np.random.default_rng(20261110)
    return rng.integers(0, n_sub, size=(b, n_sub)), rng.integers(0, n_seed, size=(b, n_seed))


# ---------- per-sample metric functions (arrays of reference r, prediction p, set s) ----------
def m1(r, p, s):
    e = p - r
    out = {}
    for name, mask in [('high', s == 'high'), ('low', s == 'low'), ('all', np.ones(len(r), bool))]:
        out[f'MAE_{name}'] = float(np.mean(np.abs(e[mask]))) if mask.any() else np.nan
        out[f'signed_error_{name}'] = float(np.mean(e[mask])) if mask.any() else np.nan
    return out


def m2(r, p, s):
    h = s == 'high'
    rr, pp = r[h], p[h]
    if h.sum() < 3 or np.std(rr) == 0 or np.std(pp) == 0:
        return dict(slope_high=np.nan, spearman_high=np.nan, kendall_high=np.nan, pearson_high=np.nan)
    return dict(slope_high=float(np.polyfit(rr, pp, 1)[0]), spearman_high=float(stats.spearmanr(rr, pp)[0]),
                kendall_high=float(stats.kendalltau(rr, pp, variant='b')[0]), pearson_high=float(np.corrcoef(rr, pp)[0, 1]))


def m3(r, p, s):
    y = (s == 'high').astype(int)
    return dict(AUROC=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan)


def all_metrics(r, p, s):
    out = m1(r, p, s)
    out.update(m2(r, p, s))
    out.update(m3(r, p, s))
    return out


def seed_arrays(df, seeds):
    """Per-seed arrays aligned on the same window order (common support already applied)."""
    subj = sorted(df.subject_id.unique())
    out = {}
    for sd in seeds:
        d = df[df.seed == sd].sort_values(['subject_id', 'window_id'])
        out[sd] = d
    return subj, out


def point_and_ci(df, seeds, fn, b=B):
    """Seed-mean point estimate and crossed-bootstrap CI for metric dict fn(r, p, s)."""
    subj, per = seed_arrays(df, seeds)
    by_subject = {sd: {k: g for k, g in d.groupby('subject_id')} for sd, d in per.items()}
    point_each = [fn(d.reference_HR.to_numpy(), d.predicted_HR.to_numpy(), d.set.to_numpy()) for d in per.values()]
    keys = list(point_each[0])
    point = {k: float(np.nanmean([x[k] for x in point_each])) for k in keys}
    if b == 0:
        return point, {}
    isub, iseed = boot_indices(len(subj), len(seeds), b)
    draws = np.full((b, len(keys)), np.nan)
    arrays = {sd: {k: (g.reference_HR.to_numpy(), g.predicted_HR.to_numpy(), g.set.to_numpy())
                   for k, g in by_subject[sd].items()} for sd in seeds}
    for i in range(b):
        chosen = [subj[j] for j in isub[i]]
        vals = []
        for j in iseed[i]:
            sd = seeds[j]
            parts = [arrays[sd][s] for s in chosen if s in arrays[sd]]
            r = np.concatenate([x[0] for x in parts])
            p = np.concatenate([x[1] for x in parts])
            st = np.concatenate([x[2] for x in parts])
            res = fn(r, p, st)
            vals.append([res[k] for k in keys])
        draws[i] = np.nanmean(np.asarray(vals, float), axis=0)
    ci = {k: (float(np.nanpercentile(draws[:, n], 2.5)), float(np.nanpercentile(draws[:, n], 97.5))) for n, k in enumerate(keys)}
    return point, ci


def contrast(da, db, seeds, fn, b=B):
    """Paired same-seed contrast a - b with the same crossed bootstrap."""
    subj = sorted(set(da.subject_id) | set(db.subject_id))
    keys = None
    arr = {}
    for tag, d in [('a', da), ('b', db)]:
        for sd in seeds:
            for s, g in d[d.seed == sd].groupby('subject_id'):
                arr[(tag, sd, s)] = (g.reference_HR.to_numpy(), g.predicted_HR.to_numpy(), g.set.to_numpy())

    def stat(chosen, seed_list):
        vals = []
        for sd in seed_list:
            res = {}
            for tag in 'ab':
                parts = [arr[(tag, sd, s)] for s in chosen if (tag, sd, s) in arr]
                res[tag] = fn(np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]),
                              np.concatenate([x[2] for x in parts]))
            vals.append({k: res['a'][k] - res['b'][k] for k in res['a']})
        ks = list(vals[0])
        return ks, [float(np.nanmean([v[k] for v in vals])) for k in ks]

    keys, point = stat(subj, seeds)
    if b == 0:
        return dict(zip(keys, point)), {}
    isub, iseed = boot_indices(len(subj), len(seeds), b)
    draws = np.array([stat([subj[j] for j in isub[i]], [seeds[j] for j in iseed[i]])[1] for i in range(b)])
    ci = {k: (float(np.nanpercentile(draws[:, n], 2.5)), float(np.nanpercentile(draws[:, n], 97.5))) for n, k in enumerate(keys)}
    return dict(zip(keys, point)), ci


def m4_bins(df):
    rows = []
    for (run, sd), d in df.groupby(['run_id', 'seed']):
        edges = HR_BINS
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = d[(d.reference_HR >= lo) & (d.reference_HR < hi)]
            if g.empty:
                continue
            e = g.predicted_HR - g.reference_HR
            rows.append(dict(run_id=run, seed=sd, bin=f'{lo}-{hi}', windows=len(g), subjects=g.subject_id.nunique(),
                             MAE=float(e.abs().mean()), signed_error=float(e.mean()),
                             hit_pm1bin=float((e.abs() <= GRID + 1e-6).mean()),
                             half_HR_error=float(((g.predicted_HR - g.reference_HR / 2).abs() <= GRID + 1e-6).mean())))
    return pd.DataFrame(rows)


def recalibrate(fit_r, fit_p, apply_p, kind):
    if kind == 'linear':
        b, a = np.polyfit(fit_p, fit_r, 1)
        return a + b * apply_p
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(fit_p, fit_r)
    return iso.predict(apply_p)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    sx, sy = x.var(), y.var()
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (sx + sy + (x.mean() - y.mean()) ** 2))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def circ(a):
    z = np.mean(np.exp(1j * np.asarray(a, float)))
    return float(np.angle(z)), float(abs(z))
