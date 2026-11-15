"""Vectorized subject x seed crossed bootstrap (same draws and estimands as b1/metrics.py loops).

Draws (draft st11x paired_bootstrap_indices): subjects in numeric order, rng = default_rng(20261110),
subject indices integers(0, n_sub, (B, n_sub)) then seed indices integers(0, n_seed, (B, n_seed)).
A replicate weights every window by its subject's multiplicity; the metric is computed per seed on the
weighted sample (identical to concatenating the resampled subjects) and averaged over the drawn seeds
(nan ignored). Point estimate: mean over seeds of the metric on all windows.
Weighted forms reproduce the concatenated-sample statistics exactly: MAE/bias (means), OLS slope and
Pearson (moments), Spearman (Pearson of weighted mid-ranks), Kendall tau-b (pair counts with ties,
including the tied copies of a duplicated window), AUROC (Mann-Whitney with half ties), CCC.
"""
import re
import warnings

import numpy as np

B = 10000


def numeric_subject_order(subjects):
    items = list(set(map(str, subjects)))

    def key(x):
        m = re.fullmatch(r'\D*(\d+)', x)
        return (0, int(m.group(1)), x) if m else (1, x)
    return sorted(items, key=key)


def draws(n_sub, n_seed, b=B):
    rng = np.random.default_rng(20261110)
    return rng.integers(0, n_sub, size=(b, n_sub)), rng.integers(0, n_seed, size=(b, n_seed))


def subject_weights(isub, n_sub):
    """(B, n_sub) multiplicity of every subject in each replicate; row 0 of the result is not special."""
    w = np.zeros((isub.shape[0], n_sub))
    np.add.at(w, (np.repeat(np.arange(isub.shape[0]), isub.shape[1]), isub.ravel()), 1)
    return w


# ---------------- weighted statistics; W is (R, n) window weights (R replicates), x/y length n ----------------
def _wmean(W, x):
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (W @ x) / n


def _wcov(W, x, y):
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = (W @ x) / n, (W @ y) / n
        return (W @ (x * y)) / n - mx * my


def mae(W, r, p):
    return _wmean(W, np.abs(p - r))


def bias(W, r, p):
    return _wmean(W, p - r)


def slope(W, r, p):
    """OLS slope of p on r (np.polyfit(r, p, 1)[0]) on the weighted sample."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return _wcov(W, r, p) / _wcov(W, r, r)


def pearson(W, x, y):
    with np.errstate(invalid='ignore', divide='ignore'):
        return _wcov(W, x, y) / np.sqrt(_wcov(W, x, x) * _wcov(W, y, y))


def ccc(W, x, y):
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = _wmean(W, x), _wmean(W, y)
        return 2 * _wcov(W, x, y) / (_wcov(W, x, x) + _wcov(W, y, y) + (mx - my) ** 2)


def _midranks(W, x):
    lt = (x[None, :] < x[:, None]).astype(float)   # lt[i, j] = x_j < x_i
    eq = (x[None, :] == x[:, None]).astype(float)
    return W @ lt.T + (W @ eq.T + 1) / 2


def spearman(W, x, y):
    return pearson_rows(W, _midranks(W, x), _midranks(W, y))


def pearson_rows(W, X, Y):
    """Weighted Pearson where X, Y are (R, n) per-replicate values."""
    n = W.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = (W * X).sum(1) / n, (W * Y).sum(1) / n
        cxy = (W * X * Y).sum(1) / n - mx * my
        cxx = (W * X * X).sum(1) / n - mx * mx
        cyy = (W * Y * Y).sum(1) / n - my * my
        return cxy / np.sqrt(cxx * cyy)


def kendall_b(W, x, y):
    sx = np.sign(x[:, None] - x[None, :])
    sy = np.sign(y[:, None] - y[None, :])
    c = sx * sy
    ex = (sx == 0).astype(float)
    ey = (sy == 0).astype(float)
    N = W.sum(1)
    n0 = N * (N - 1) / 2
    s = ((W @ c) * W).sum(1) / 2
    n1 = (((W @ ex) * W).sum(1) - N) / 2
    n2 = (((W @ ey) * W).sum(1) - N) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / np.sqrt((n0 - n1) * (n0 - n2))


def auroc(W, p, y):
    """P(score of a positive > score of a negative) + half ties; y boolean."""
    pos, neg = np.where(y)[0], np.where(~y)[0]
    m = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    wp, wn = W[:, pos], W[:, neg]
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((wp @ m) * wn).sum(1) / (wp.sum(1) * wn.sum(1))


# ---------------- metric families ----------------
def m1(W, r, p, s):
    out = {}
    for name, mask in [('high', s == 'high'), ('low', s == 'low'), ('all', np.ones(len(r), bool))]:
        Wm = W[:, mask]
        out[f'MAE_{name}'] = mae(Wm, r[mask], p[mask]) if mask.any() else np.full(len(W), np.nan)
        out[f'signed_error_{name}'] = bias(Wm, r[mask], p[mask]) if mask.any() else np.full(len(W), np.nan)
    return out


def m2(W, r, p, s):
    h = s == 'high'
    Wh, rr, pp = W[:, h], r[h], p[h]
    out = dict(slope_high=slope(Wh, rr, pp), spearman_high=spearman(Wh, rr, pp),
               kendall_high=kendall_b(Wh, rr, pp), pearson_high=pearson(Wh, rr, pp))
    bad = (Wh.sum(1) < 3) | ~(_wcov(Wh, rr, rr) > 1e-12) | ~(_wcov(Wh, pp, pp) > 1e-12)
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    return out


def m3(W, r, p, s):
    y = s == 'high'
    a = auroc(W, p, y)
    bad = ~((W[:, y].sum(1) > 0) & (W[:, ~y].sum(1) > 0))
    return dict(AUROC=np.where(bad, np.nan, a))


def m123(W, r, p, s):
    out = m1(W, r, p, s)
    out.update(m2(W, r, p, s))
    out.update(m3(W, r, p, s))
    return out


def m7(W, dref, dpred, s=None):
    return dict(M7_MAE=mae(W, dref, dpred), M7_slope=slope(W, dref, dpred), M7_CCC=ccc(W, dref, dpred))


# ---------------- engine ----------------
class Sample:
    """Windows of one model (all seeds) on a fixed support: arrays per seed aligned on subject order."""

    def __init__(self, df, subjects, seeds, value='predicted_HR'):
        self.subjects, self.seeds = subjects, seeds
        index = {s: i for i, s in enumerate(subjects)}
        self.per = {}
        for sd in seeds:
            d = df[df.seed == sd].sort_values(['subject_id', 'window_id'])
            self.per[sd] = (np.array([index[x] for x in d.subject_id]), d.reference_HR.to_numpy(float),
                            d[value].to_numpy(float), d.set.to_numpy())


def per_seed(sample, fn, Wsub):
    """{seed: {metric: (R,) values}} for subject weights Wsub (R, n_sub)."""
    out = {}
    for sd, (si, r, p, s) in sample.per.items():
        out[sd] = fn(Wsub[:, si], r, p, s)
    return out


def summarize(samples_fn, subjects, seeds, b=B, chunk=500):
    """samples_fn(Wsub) -> {seed: {metric: (R,)}}. Returns point dict and (lo, hi) dict."""
    full = samples_fn(np.ones((1, len(subjects))))
    keys = list(next(iter(full.values())))
    point = {k: float(np.nanmean([full[sd][k][0] for sd in seeds])) for k in keys}
    if b == 0:
        return point, {}
    isub, iseed = draws(len(subjects), len(seeds), b)
    reps = {k: np.empty(b) for k in keys}
    for a in range(0, b, chunk):
        Wsub = subject_weights(isub[a:a + chunk], len(subjects))
        vals = samples_fn(Wsub)
        for k in keys:
            stack = np.stack([vals[sd][k] for sd in seeds])            # (n_seed, R)
            chosen = stack[iseed[a:a + chunk].T, np.arange(Wsub.shape[0])[None, :]]  # (n_seed draws, R)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # all-nan columns stay nan
                reps[k][a:a + chunk] = np.nanmean(chosen, axis=0)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in reps.items()}
    return point, ci


def point_ci(df, seeds, fn, b=B):
    subjects = numeric_subject_order(df.subject_id)
    smp = Sample(df, subjects, seeds)
    return summarize(lambda W: per_seed(smp, fn, W), subjects, seeds, b)


def contrast_ci(da, db, seeds, fn, b=B):
    """Same-seed paired difference a - b on the same windows."""
    subjects = numeric_subject_order(set(da.subject_id) | set(db.subject_id))
    sa, sb = Sample(da, subjects, seeds), Sample(db, subjects, seeds)

    def f(W):
        va, vb = per_seed(sa, fn, W), per_seed(sb, fn, W)
        return {sd: {k: va[sd][k] - vb[sd][k] for k in va[sd]} for sd in seeds}
    return summarize(f, subjects, seeds, b)
