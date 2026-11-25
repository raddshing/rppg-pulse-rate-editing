"""Brief 02 7.01: post-hoc analyses on existing outputs only (no training, no inference, no re-inference; CPU, one thread).

Shared loaders. Test data: the saved stage-2 predictions, step-(b) readouts and stage-1 references of the test run step (c)
(cache/test). Development data: the guarded predictions of rehearsal part 2 (cache/rehearsal/part2), the source of the
development values of the paper (6.93 나: rehearsal (c) equals the guarded development analysis to the last digit).
Windows, validity, support, populations and the crossed subject x seed bootstrap (b1/boot.py, B = 10,000, draws
default_rng(20261110) on the numeric subject order) are those of eval5/stage3_analyze.py, imported unchanged.
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WORK = Path(__file__).resolve().parents[2]
for p in (WORK / 'b1', WORK / 'eval5'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import stage3_analyze as SA  # noqa: E402
import boot  # noqa: E402

TEST = WORK / 'cache' / 'test'
DEV = WORK / 'cache' / 'rehearsal' / 'part2'
VENDOR = WORK / 'vendor' / 'st8_6ebbc9cd'
OUT = WORK / 'outputs' / 'posthoc701'
SEEDS = SA.SEEDS
B = boot.B
SRC_TEST = 'test (c) outputs'
SRC_DEV = 'development guarded predictions'
SRC_OCT = 'October stored predictions (before guards)'


def summarize_nan(samples_fn, subjects, seeds, b=B, chunk=500):
    """b1/boot.py summarize, same draws and estimates, plus the number of replicates whose value is NaN (a resample without
    any window of a part); those replicates are ignored by the percentile interval, as in b1/boot.py."""
    import warnings
    full = samples_fn(np.ones((1, len(subjects))))
    ks = list(next(iter(full.values())))
    point = {k: float(np.nanmean([full[sd][k][0] for sd in seeds])) for k in ks}
    isub, iseed = boot.draws(len(subjects), len(seeds), b)
    reps = {k: np.empty(b) for k in ks}
    for a in range(0, b, chunk):
        Wsub = boot.subject_weights(isub[a:a + chunk], len(subjects))
        vals = samples_fn(Wsub)
        for k in ks:
            stack = np.stack([vals[sd][k] for sd in seeds])
            chosen = stack[iseed[a:a + chunk].T, np.arange(Wsub.shape[0])[None, :]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reps[k][a:a + chunk] = np.nanmean(chosen, axis=0)
    with np.errstate(all='ignore'):
        ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in reps.items()}
    return point, ci, {k: int(np.isnan(v).sum()) for k, v in reps.items()}


def contrast_nan(da, db, seeds, fn, b=B):
    """b1/boot.py contrast_ci with the NaN-replicate count of summarize_nan."""
    subjects = boot.numeric_subject_order(set(da.subject_id) | set(db.subject_id))
    sa, sb = boot.Sample(da, subjects, seeds), boot.Sample(db, subjects, seeds)

    def f(W):
        va, vb = boot.per_seed(sa, fn, W), boot.per_seed(sb, fn, W)
        return {sd: {k: va[sd][k] - vb[sd][k] for k in va[sd]} for sd in seeds}
    return summarize_nan(f, subjects, seeds, b)


def keys(root, dataset):
    k = pd.concat([pd.read_csv(root / x / dataset / 'checkpoints.csv') for x in ('stage2_conf', 'stage2_expl')], ignore_index=True)
    return k[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})


def load(root, dataset):
    """Window table of every key (eval5/stage3_analyze.py load, population 'automatic')."""
    return SA.load(dataset, root / 'stage1' / dataset, root / 'check' / dataset, keys(root, dataset), 'automatic')


def primary_har(d):
    return d[d.variant.isin(SA.PRIMARY) & d.layer.eq('HAR')]


def main_population(dataset):
    return 'Qstar' if dataset == 'MMPD' else 'draft_rule'


def write(df, name, sub):
    out = OUT / sub
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False)
    return out / name


def stamp(sub, **kw):
    out = OUT / sub
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run.json').write_text(json.dumps(dict(ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), b=B, seeds=SEEDS, **kw), indent=1, default=str))
