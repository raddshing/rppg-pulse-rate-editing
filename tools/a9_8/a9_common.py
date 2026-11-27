"""OPS2 A9 item 8 (7.13 1): stored-output analyses on the predictions that existed when A9 was committed. Shared loaders.

No inference and no re-inference; CPU, one thread (OMP/BLAS 1), run with nice 19 so that the pool generators keep the CPU.
VIPL-HR: the readouts of stage (c2) (every key of stages (b1) and (b2), read by eval5/vipl_c.py), the stage (a) references,
the populations of eval5/vipl_c.populations (unchanged: reference = gt_HR mean, verified by the BVP; high = reference >= 100),
the defects and support of (c2); the analysis frame is built as in vipl_c.main (stage3_analyze columns).
Test sets (MMPD-B, MCD-rPPG test half): the loaders of tools/posthoc701/p701_common.py (test step (c) outputs, unchanged).
Statistics: b1/boot.py (crossed subject x seed bootstrap, B = 10,000, draws default_rng(20261110)), seeds paired by index.
Outputs: outputs/a9_8/<part>/ (aggregate only).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
for p in (HERE.parent / 'posthoc703', HERE.parent / 'posthoc702', HERE.parent / 'posthoc701', WORK / 'b1', WORK / 'eval5'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import p702_common as P2  # noqa: E402  (p701_common: one thread, paths, test loaders)
from p702_common import B, SA, SEEDS, boot, half, hit, np, pd  # noqa: E402,F401

VO = WORK / 'cache' / 'vipl707'
VSA, VC2 = VO / 'stage_a' / 'VIPL', VO / 'stage_c2' / 'VIPL'
OUT = WORK / 'outputs' / 'a9_8'
SRC_VIPL = 'VIPL-HR stage (c2) readouts (keys of stages (b1), (b2))'


def vipl_keys():
    k = pd.concat([pd.read_csv(VO / x / 'VIPL' / 'checkpoints.csv') for x in ('stage_b_confirmatory', 'stage_b_exploratory')],
                  ignore_index=True)
    return k


def vipl_frame():
    """The analysis frame of eval5/vipl_c.py main (both layers, every population), from the stored (c2) outputs."""
    import vipl_c as VC
    refs = pd.read_csv(VSA / 'references.csv', dtype={'subject_id': str})
    choice = json.loads((VSA / 'bvp_rate.json').read_text())['choice']
    rd = pd.read_csv(VC2 / 'readouts.csv.gz')
    keys = vipl_keys()
    bad = set(pd.read_csv(VC2 / 'defects.csv').key)
    sup = pd.read_csv(VC2 / 'support.csv')
    support = set(zip(sup.window_id, sup.layer))
    kk = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    pops = VC.populations(refs, choice)
    d = rd.merge(kk, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'] = 'VIPL-HR'
    d['backbone'] = d.backbone.map(SA.NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    return d, refs, choice, bad


def primary(d, population='primary', layer='HAR'):
    return d[d.variant.isin(SA.PRIMARY) & d.layer.eq(layer) & d.population.eq(population)]


def arm_windows(g, seeds=SEEDS):
    """Windows valid for every key of the arm (inside the support), as stage3_analyze.arm_rows."""
    g = g[g.seed.isin(seeds)]
    k = g.key.nunique()
    v = g.groupby(['window_id', 'layer']).key.nunique()
    w = set(v[v == k].index)
    return g[[x in w for x in zip(g.window_id, g.layer)]], len(w)


def write(df, name, sub):
    out = OUT / sub
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False)
    text = (out / name).read_text()
    assert '/' + 'home/' not in text and '/' + 'mnt/' not in text, 'server path in an output'
    return out / name


def stamp(sub, **kw):
    out = OUT / sub
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_info.json').write_text(json.dumps(dict(ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), b=B, seeds=SEEDS, **kw),
                                                  indent=1, default=str))
