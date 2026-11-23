"""Registered evaluation on VIPL-HR (amendment A7): stages (c1) and (c2), the analyses.

Stage (c1) (--part primary, --stage-b the primary predictions of stage (b1)): every analysis of the primary checkpoints.
Stage (c2) (--part remaining, --stage-b the outputs of (b1) and (b2)): the analyses of the secondary checkpoints, which use
the primary checkpoints as the reference of each group; readouts, defects and support are recomputed over all keys.

Uses the evaluation code of commit 19ecf5d unchanged (eval5/stage3_check.py: hash check and HR readout of the predictions;
eval5/stage3_analyze.py with b1/boot.py and the development families: crossed subject x seed bootstrap, B = 10,000, draws
default_rng(20261110) on the numeric subject order; contrast rows, grades T1-T4, arm rows, bins). Added for VIPL-HR:
  - readout: every output key on every window of stage (a), the prediction on the same 30-Hz grid from the first frame
    (window j = samples 300 j .. 300 j + 299), float32 as stored (as the MCD-rPPG path); wide (HAR) and narrow (L0) band;
    PhysNet keys with DiffNormalized output read as differenced (stage3_check rule: PhysNet and not '-std').
  - defects: load failures of stage (b), and a key not valid on more than half of the windows; support = windows valid for
    every non-defective primary-set confirmatory key (the 45 checkpoints, primary variants).
  - populations: 'primary' (source1 and source3; verified windows, i.e. no signal defect and the three estimates agree;
    windows with an interval > 200 ms excluded), 'primary_with_gaps' (sensitivity, item 4), 'primary_gtHR' (sensitivity,
    item 5: the gt_HR mean replaces the reference), 'secondary' (source2); high = reference >= 100, low = < 100.
  - item 10: C1-C4 graded on the primary set (MAE_high and the low-HR grade of each contrast); everything else values only.
    Summary: the number k of backbones with C1 in T1.
  - item 9: within-subject change from v1 (rest) to v7 (after exercise) per subject and device with both recordings
    (additional recordings excluded), on the windows of the primary analysis: per run and (subject, device) the change of
    the mean reference and of the mean prediction; MAE, slope and CCC of the change (boot.m7) per arm, and C1-C4 on them;
    crossed bootstrap over subjects and seeds (as for the within-person change on MCD-rPPG).
  - secondary checkpoints: the exploratory arms of the 2026-11-15 entry with the test run's secondary contrasts; the arms of
    later settings (arm suffix, e.g. '-std', '-mcc', '-std-mcc') and the confirmatory arms with other seeds are grouped by
    suffix and seed set; per group the C1-C4 analogues, (ii-pc) - (i-simple) and (ii-lag) - (ii-pc). All values only.
Outputs (--out/VIPL/): readouts.csv.gz, defects.csv, support.csv, analysis_hashes.txt, run.json, and
  (c1) confirmatory.csv, sensitivity.csv, secondary_set.csv, arms.csv, ge120.csv, m4_bins.csv, within_subject.csv,
       summary.json;  (c2) secondary_checkpoints.csv.
Usage: vipl_c.py --part primary|remaining --stage-a A --stage-b B [B ...] --vendor V --out OUT [--b 10000] [--workers N]
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage2_predict as S2  # noqa: E402
import stage3_check as SC  # noqa: E402
import stage3_analyze as S3  # noqa: E402

CONF_ARMS = ('(0)', '(i)', '(i-pc)', '(ii)', '(ii-pc)')
CONF_SEEDS = [20261131, 20261132, 20261133]
LAYERS = ('HAR', 'L0')


def readout_job(args):
    keydir, key, diff, recs = args
    SC.vendor_imports(SC.VENDOR_ARG)
    out = []
    for rid, wins in recs:
        p = keydir / (rid + '.npy')
        pred = np.load(p) if p.exists() else None  # float32 as stored
        for wid, j in wins:
            for layer in LAYERS:
                if pred is None or 300 * (j + 1) > len(pred):
                    out.append((key, wid, layer, np.nan, False))
                    continue
                h = SC.read_hr(pred[300 * j:300 * (j + 1)], diff, layer)
                out.append((key, wid, layer, h, bool(np.isfinite(h))))
    return out


def populations(refs, choice):
    """Window rows per population and layer (reference, high/low)."""
    gt_ref = choice.startswith('gt_HR')
    rows = []
    for layer in LAYERS:
        ref = refs.gt_HR_mean if gt_ref else refs[f'h_FFT_{layer}']
        ok = refs.verified.astype(bool) if layer == 'HAR' else refs.Qstar_L0.astype(bool)
        base = refs.assign(layer=layer, reference_HR=ref)
        for pop, m, r in (('primary', ok & refs.set.eq('primary') & ~refs.gap200.astype(bool), ref),
                          ('primary_with_gaps', ok & refs.set.eq('primary'), ref),
                          ('primary_gtHR', ok & refs.set.eq('primary') & ~refs.gap200.astype(bool), refs.gt_HR_mean),
                          ('secondary', ok & refs.set.eq('secondary'), ref)):
            z = base[m].assign(reference_HR=r[m], population=pop)
            z = z[np.isfinite(z.reference_HR)]
            rows.append(z.assign(set=np.where(z.reference_HR >= 100, 'high', 'low')))
    return pd.concat(rows, ignore_index=True)


def setting_of(arm):
    """Arm suffix that names a secondary setting: '(ii-pc-std-mcc)' -> ('(ii-pc)', '-std-mcc')."""
    a = arm.strip('()')
    for base in ('ii-pc', 'ii-lag', 'i-simple', 'i-pc', 'ii', 'i', '0'):
        if a == base or a.startswith(base + '-'):
            return f'({base})', a[len(base):]
    return arm, ''


def change_table(g):
    """Item 9: per run and (subject, device) with both v1 and v7 (additional recordings v1-2/v7-2 excluded), the change
    (v7 - v1) of the mean reference and of the mean prediction over the windows given (as b1/analyze.m7_table, with the
    device as a second level); subject_id stays the bootstrap unit, window_id names the (subject, device) pair."""
    z = g[g.scene.isin(['v1', 'v7'])].assign(state=lambda x: np.where(x.scene.eq('v1'), 'before', 'after'))
    rec = z.groupby(['run_id', 'seed', 'subject_id', 'source', 'state']).agg(r=('reference_HR', 'mean'), p=('predicted_HR', 'mean')).reset_index()
    w = rec.pivot_table(index=['run_id', 'seed', 'subject_id', 'source'], columns='state', values=['r', 'p'])
    if ('r', 'after') not in w or ('r', 'before') not in w:
        return pd.DataFrame(columns=['run_id', 'seed', 'subject_id', 'source', 'reference_HR', 'predicted_HR', 'window_id', 'set'])
    t = pd.DataFrame(dict(reference_HR=w[('r', 'after')] - w[('r', 'before')], predicted_HR=w[('p', 'after')] - w[('p', 'before')])).reset_index().dropna()
    t['window_id'] = t.subject_id + '|' + t.source
    t['set'] = 'all'
    return t


def within_subject(d, b):
    """Item 9: per arm the change measures (MAE, slope, CCC; boot.m7) and, per backbone, C1-C4 on them (paired by seed,
    on the (subject, device) pairs present for every key of both arms); intervals from the crossed bootstrap over subjects
    and seeds (boot.point_ci / boot.contrast_ci, as for the within-person change on MCD-rPPG)."""
    out = []
    tables = {}
    for (layer, bb, st), g in d.groupby(['layer', 'backbone', 'strategy']):
        g = g[g.seed.isin(S3.SEEDS)]
        if set(S3.SEEDS) - set(g.seed):
            continue
        t = change_table(g)
        if t.empty:
            continue
        tables[(layer, bb, st)] = t
        pt, ci = S3.boot.point_ci(t, S3.SEEDS, S3.boot.m7, b)
        for m, v in pt.items():
            lo, hi = ci.get(m, (np.nan, np.nan))
            out.append(dict(kind='arm', layer=layer, backbone=bb, strategy=st, pairs=t[['subject_id', 'source']].drop_duplicates().shape[0],
                            subjects=t.subject_id.nunique(), metric=m, point=v, ci_lo=lo, ci_hi=hi))
    for (layer, bb, st), ta in sorted(tables.items()):
        for name, a, c in S3.CONF:
            if st != a or (layer, bb, c) not in tables:
                continue
            tc = tables[(layer, bb, c)]
            keep = set(ta.groupby('window_id').seed.nunique().loc[lambda x: x == len(S3.SEEDS)].index) & \
                set(tc.groupby('window_id').seed.nunique().loc[lambda x: x == len(S3.SEEDS)].index)
            xa, xc = ta[ta.window_id.isin(keep)], tc[tc.window_id.isin(keep)]
            pt, ci = S3.boot.contrast_ci(xa, xc, S3.SEEDS, S3.boot.m7, b)
            for m, v in pt.items():
                lo, hi = ci.get(m, (np.nan, np.nan))
                out.append(dict(kind='contrast', layer=layer, backbone=bb, strategy=f'{name} {a}-{c}', pairs=len(keep),
                                subjects=xa.subject_id.nunique(), metric=m, point=v, ci_lo=lo, ci_hi=hi))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage-a', required=True), ap.add_argument('--stage-b', nargs='+', required=True)
    ap.add_argument('--vendor', required=True), ap.add_argument('--out', required=True)
    ap.add_argument('--b', type=int, default=S3.boot.B), ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--part', required=True, choices=['primary', 'remaining'])
    args = ap.parse_args()
    vendor = str(Path(args.vendor).resolve())
    SC.init(vendor)
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    sa = Path(args.stage_a).resolve() / 'VIPL'
    sbs = [Path(x).resolve() / 'VIPL' for x in args.stage_b]
    n_hash = SC.verify_hashes(sa, 'stage_a_hashes.txt') + sum(SC.verify_hashes(x, 'stage2_hashes.txt') for x in sbs)
    out = Path(args.out).resolve() / 'VIPL'
    out.mkdir(parents=True, exist_ok=True)
    refs = pd.read_csv(sa / 'references.csv', dtype={'subject_id': str})
    choice = json.loads((sa / 'bvp_rate.json').read_text())['choice']
    keys = pd.concat([pd.read_csv(x / 'checkpoints.csv').assign(folder=str(x)) for x in sbs], ignore_index=True)
    sets = set(keys['set'])
    if (args.part == 'primary' and sets != {'confirmatory'}) or (args.part == 'remaining' and sets != {'confirmatory', 'exploratory'}):
        sys.exit(f'--part {args.part}: stage (b) outputs of sets {sorted(sets)}')
    # readouts
    recs = [(rid, list(zip(g.window_id, g.window_index.astype(int)))) for rid, g in refs.groupby('recording_id')]
    jobs = []
    for r in keys[keys.status.eq('used')].itertuples():
        diff = r.view == 'PhysNet' and '-std' not in r.arm
        jobs.append((Path(r.folder) / 'pred' / r.key, r.key, diff, recs))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=SC.init, initargs=(vendor,)) as ex:
        rd = pd.DataFrame([x for part in ex.map(readout_job, jobs) for x in part], columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])
    rd.to_csv(out / 'readouts.csv.gz', index=False, compression=dict(method='gzip', mtime=0))  # fixed gzip header: same bytes on every run
    # defects and support
    bad = {r.key: r.reason for r in keys[keys.status.ne('used')].itertuples()}
    share = rd[rd.layer.eq('HAR')].groupby('key').valid.mean()
    bad.update({k: 'not valid on more than half of the windows' for k, v in share.items() if v < 0.5})
    pd.DataFrame([dict(key=k, reason=v) for k, v in sorted(bad.items())], columns=['key', 'reason']).to_csv(out / 'defects.csv', index=False)
    kk = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    conf_keys = set(kk[kk.arm.isin(CONF_ARMS) & kk.seed.isin(CONF_SEEDS) & kk.variant.isin(S3.PRIMARY)].key) - set(bad)
    v = rd[rd.key.isin(conf_keys)].groupby(['window_id', 'layer']).valid.agg(['sum', 'size'])
    support = set(v[(v['sum'] == len(conf_keys))].index)
    pd.DataFrame([dict(window_id=w, layer=l, in_support=True) for w, l in sorted(support)]).to_csv(out / 'support.csv', index=False)
    # analysis frame (the columns of stage3_analyze.load)
    pops = populations(refs, choice)
    d = rd.merge(kk, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(set(bad))].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'] = 'VIPL-HR'
    d['backbone'] = d.backbone.map(S3.NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    prim = d[d.variant.isin(S3.PRIMARY)]
    conf_d = prim[prim.arm.isin(CONF_ARMS) & prim.seed.isin(CONF_SEEDS)]
    pw = conf_d[conf_d.layer.eq('HAR') & conf_d.population.eq('primary')]
    conf = [(n, a, c, 'graded') for n, a, c in S3.CONF]
    if args.part == 'remaining':
        secondary_checkpoints(prim, pw, args.b, out)
        finish(out, args, started, n_hash, bad, support, None)
        return
    rows = S3.contrast_rows(pw, conf, args.b, 'primary')
    pd.DataFrame(rows).to_csv(out / 'confirmatory.csv', index=False)
    sens = []
    for pop in ('primary_with_gaps', 'primary_gtHR'):
        sens += S3.contrast_rows(conf_d[conf_d.layer.eq('HAR') & conf_d.population.eq(pop)], conf, args.b, pop)
    sens += S3.contrast_rows(conf_d[conf_d.layer.eq('L0') & conf_d.population.eq('primary')], conf, args.b, 'narrow_band')
    for r in sens:
        r['grade'] = ''
    pd.DataFrame(sens).to_csv(out / 'sensitivity.csv', index=False)
    sec = S3.contrast_rows(conf_d[conf_d.layer.eq('HAR') & conf_d.population.eq('secondary')], conf, args.b, 'secondary_set')
    for r in sec:
        r['grade'] = ''
    pd.DataFrame(sec).to_csv(out / 'secondary_set.csv', index=False)
    arms = S3.arm_rows(conf_d[conf_d.population.eq('primary')], args.b, 'primary')
    pd.DataFrame(arms).to_csv(out / 'arms.csv', index=False)
    g120 = pw[pw.reference_HR >= 120].copy()
    pd.DataFrame(S3.arm_rows(g120, args.b, 'ge120') + S3.contrast_rows(g120, conf, args.b, 'ge120')).to_csv(out / 'ge120.csv', index=False)
    pd.DataFrame(S3.AN.fam_m4(S3.restrict(conf_d[conf_d.population.eq('primary')]), [])).to_csv(out / 'm4_bins.csv', index=False)
    pd.DataFrame(within_subject(S3.restrict(conf_d[conf_d.population.eq('primary')]), args.b)).to_csv(out / 'within_subject.csv', index=False)
    c1 = [r for r in rows if r['name'] == 'C1' and r['metric'] == 'MAE_high']
    low = {r['backbone']: r['grade'] for r in rows if r['name'] == 'C1' and r['metric'] == 'MAE_low'}
    summary = dict(k=sum(r['grade'] == 'T1' for r in c1), backbones={r['backbone']: dict(C1_high=r['grade'], C1_low=low.get(r['backbone'], '')) for r in c1},
                   bvp_rate=choice)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    finish(out, args, started, n_hash, bad, support, summary)


def secondary_checkpoints(prim, pw, b, out):
    """Stage (c2): groups by setting suffix and seed set, values only."""
    other = prim[~(prim.arm.isin(CONF_ARMS) & prim.seed.isin(CONF_SEEDS)) & prim.layer.eq('HAR')]
    other = pd.concat([other, pw[pw.population.eq('primary')]])  # confirmatory arms as the reference of each group
    sc = []
    groups = {}
    for arm, seeds in other.groupby('arm').seed.unique().items():
        base, suf = setting_of(arm)
        groups.setdefault((suf, tuple(sorted(seeds))), set()).add(base)
    for (suf, seeds), bases in sorted(groups.items()):
        if suf == '' and list(seeds) == CONF_SEEDS:
            continue
        S3.SEEDS = list(seeds)
        z = other[other.population.eq('primary')]
        name = lambda b: f'({b.strip("()")}{suf})'  # noqa: E731
        pairs = [(n, name(a), name(c), 'values only') for n, a, c in S3.CONF] + \
                [('', name('(ii-pc)'), name('(i-simple)'), 'values only'), ('', name('(ii-lag)'), name('(ii-pc)'), 'values only')]
        for r in S3.contrast_rows(z, pairs, b, f'setting{suf or "-confirmatory-arms"}_seeds_{seeds[0]}-{seeds[-1]}'):
            r['grade'] = ''
            sc.append(r)
    S3.SEEDS = CONF_SEEDS
    # exploratory arms of the 2026-11-15 entry (A4 manifest, confirmatory seeds): the test run's secondary contrasts
    z = prim[prim.seed.isin(CONF_SEEDS) & prim.layer.eq('HAR') & prim.population.eq('primary')]
    for r in S3.contrast_rows(z, [('', a, c, 'values only') for a, c, k in S3.SECONDARY], b, 'exploratory_arms'):
        r['grade'] = ''
        sc.append(r)
    pd.DataFrame(sc).to_csv(out / 'secondary_checkpoints.csv', index=False)


def finish(out, args, started, n_hash, bad, support, summary):
    names = sorted(p.name for p in out.iterdir() if p.suffix in ('.csv', '.json', '.gz') and p.name != 'run.json')
    (out / 'analysis_hashes.txt').write_text(''.join(f'{S2.file_sha(out / n)}  VIPL/{n}\n' for n in names))
    run = dict(stage='c1' if args.part == 'primary' else 'c2', part=args.part, started=started,
               ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(), b=args.b, hashes_checked=n_hash,
               excluded_keys=sorted(bad), support_windows=len(support), summary=summary)
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    print(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
