"""OPS2 7.13 2: evaluation of E2 and E3 (A9 items 2, 3 and 7), run only after the SHA-256 lists of their prediction files are
committed (A9 item 7; tools/pub709e_predsha.py). Values with 95% intervals, no tiers, no statements (A9 header).

Code reused unchanged: eval5/vipl_c.py (readout_job, populations: VIPL-HR primary analysis of the first entry of 2026-11-23),
eval5/stage3_check.py (step (b) of the test run), eval5/stage3_analyze.py (load, restrict, pair_windows), tools/analysis_704.py
(job_arm, job_contrast, job_interaction: M1-M3 of the test run, MAE over verified windows, high-HR hit rate; crossed subject x
seed bootstrap of b1/boot.py, B = 10,000, default_rng(20261110), seeds paired by index), tools/analysis_704b.py (execute, sel,
cmd_check, load with its folders replaced).

Commands (main environment):
  vipl_readouts [--folder F --out O]  readouts of every key of the E2/E3 VIPL-HR pass (cache/infer709/vipl_main) on every
                stage (a) window, as stage (c) (vipl_c.readout_job); stage2 hashes verified first; defects as vipl_c
                (load failure, valid on less than half of the windows). --folder/--out: dry test on stored folders only.
  test_check    step (b) of the test run over the test-run folders (confirmatory, exploratory) and the E3 passes
                (cache/infer709/E3/MMPD_B, MCD_test); the stored readouts and support must be reproduced.
  analyze       E2 on VIPL-HR (primary set, HAR): P (A6 item 3) per backbone, arms on the new seeds 20261148-50, C1-C4 on
                the new seeds and on all six seeds; R (A6 item 9), RhythmFormer-MCC (seeds 20261131-33): arms, the
                contrasts of A6 item 7 and the loss interaction RhythmFormer-MCC - RhythmFormer. E3: (0B) on MMPD-B (Q*),
                the MCD-rPPG test half (draft rule) and VIPL-HR (primary): arm values, (ii-pc)-(0B), (i-simple)-(0B),
                (0)-(0B); beside them the values from the stored October predictions (outputs/posthoc701/ra2, made without
                the guards; (0)-(0B) is the negated (0B)-(0)).
Outputs: outputs/a9_eval/{E2,E3}/*.csv (aggregate); server-only readouts in cache/a9_eval/.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'tools'))
sys.path.insert(0, str(W / 'tools' / 'a9_8'))
import analysis_704 as A  # noqa: E402
import analysis_704b as AB  # noqa: E402
import a9_common as C  # noqa: E402
from a9_common import np, pd  # noqa: E402

CACHE = W / 'cache' / 'a9_eval'
OUT = W / 'outputs' / 'a9_eval'
VIPL_MAIN = W / 'cache' / 'infer709' / 'vipl_main'
E3 = {'MMPD': W / 'cache' / 'infer709' / 'E3' / 'MMPD_B', 'MCD': W / 'cache' / 'infer709' / 'E3' / 'MCD_test'}
CONF, NEW = [20261131, 20261132, 20261133], [20261148, 20261149, 20261150]
ALL6 = CONF + NEW
BACK = ('PhysNet', 'FactorizePhys (matched)', 'RhythmFormer')
RF = 'RhythmFormer'
P_ARMS = ('(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-pc)')
E3_PAIRS = [('(ii-pc)', '(0B)'), ('(i-simple)', '(0B)'), ('(0)', '(0B)')]
NAMES = {'MMPD': 'MMPD-B', 'MCD': 'MCD-rPPG test', 'VIPL': 'VIPL-HR'}


def cmd_vipl_readouts(workers, folder=VIPL_MAIN, out=CACHE / 'vipl'):
    import vipl_c as VC
    import stage3_check as SC
    vendor = str(A.VENDOR)
    SC.init(vendor)
    sb = Path(folder) / 'VIPL'
    n = SC.verify_hashes(sb, 'stage2_hashes.txt')
    refs = pd.read_csv(C.VSA / 'references.csv', dtype={'subject_id': str})
    recs = [(rid, list(zip(g.window_id, g.window_index.astype(int)))) for rid, g in refs.groupby('recording_id')]
    keys = pd.read_csv(sb / 'checkpoints.csv')
    jobs = [(sb / 'pred' / r.key, r.key, r.view == 'PhysNet' and '-std' not in r.arm, recs) for r in keys[keys.status.eq('used')].itertuples()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=SC.init, initargs=(vendor,)) as ex:
        rd = pd.DataFrame([x for part in ex.map(VC.readout_job, jobs) for x in part], columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rd.to_csv(out / 'readouts.csv.gz', index=False)
    bad = {r.key: r.reason for r in keys[keys.status.ne('used')].itertuples()}
    share = rd[rd.layer.eq('HAR')].groupby('key').valid.mean()
    bad.update({k: 'not valid on more than half of the windows' for k, v in share.items() if v < 0.5})
    pd.DataFrame([dict(key=k, reason=v) for k, v in sorted(bad.items())], columns=['key', 'reason']).to_csv(out / 'defects.csv', index=False)
    keys.to_csv(out / 'checkpoints.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps(dict(folder=str(folder), hashes_verified=n, keys=len(jobs), defective=sorted(bad),
                                                       seconds=time.time() - t0), indent=1))
    print('vipl readouts', len(jobs), 'keys', len(rd), 'rows', 'defective', len(bad), round(time.time() - t0), flush=True)


def vipl_new_frame(rdir=CACHE / 'vipl'):
    """Frame of the new keys built as vipl_c.main / a9_common.vipl_frame (support: the stored confirmatory support)."""
    import vipl_c as VC
    refs = pd.read_csv(C.VSA / 'references.csv', dtype={'subject_id': str})
    choice = json.loads((C.VSA / 'bvp_rate.json').read_text())['choice']
    rd = pd.read_csv(Path(rdir) / 'readouts.csv.gz')
    keys = pd.read_csv(Path(rdir) / 'checkpoints.csv')
    bad = set(pd.read_csv(Path(rdir) / 'defects.csv').key)
    sup = pd.read_csv(C.VC2 / 'support.csv')
    support = set(zip(sup.window_id, sup.layer))
    kk = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    d = rd.merge(kk, on='key').merge(VC.populations(refs, choice), on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'], d['backbone'], d['strategy'], d['run_id'], d['available'] = 'VIPL-HR', d.backbone.map(C.SA.NAME), d.arm, d.key, d.valid
    return d, sorted(bad)


def test_folders(part, ds):
    root = A.PARTS[part]['root']
    return [root / 'stage2_conf', root / 'stage2_expl', E3[ds]]


def cmd_test_check(workers):
    AB.CHECK = CACHE / 'test_check'
    AB.stage2_folders = test_folders
    for ds in E3:
        assert (E3[ds] / ds / 'stage2_hashes.txt').exists(), f'E3 pass {ds} not finished'
    AB.cmd_check('test', workers)


def ra2_beside():
    """October stored-prediction values of (0B) (posthoc 7.01 ra2) in the E3 contrast orientation."""
    p = W / 'outputs' / 'posthoc701' / 'ra2'
    arms = pd.read_csv(p / 'ra2_arms.csv')
    con = pd.read_csv(p / 'ra2_contrasts.csv')
    return arms, con


def cmd_analyze(workers, b, dry=False):
    A.setup()
    t0 = time.time()
    out = OUT / ('dry' if dry else 'run')
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    old, _, _, bad_old = C.vipl_frame()
    if dry:  # validation only: the stored keys stand in for the new pass (NEW := CONF seeds of the stored keys)
        new, bad_new = vipl_new_frame(CACHE / 'vipl_dry')
        global NEW, ALL6
        NEW = CONF
    else:
        new, bad_new = vipl_new_frame()
    vf = C.SA.restrict(pd.concat([old, new], ignore_index=True).pipe(C.primary))
    vf = vf.drop_duplicates(['key', 'window_id', 'layer'])
    jobs = []
    # E2 P (A6 item 3) on VIPL-HR
    for bb in BACK:
        for base in P_ARMS:
            g = AB.sel(vf, bb, base, NEW)
            if g is not None:
                jobs.append(('arm', NEW, (dict(experiment='E2 P', dataset='VIPL-HR', backbone=bb, arm=base, seeds='new 20261148-50'), g, b)))
        for name, a, c in A.PAIRS[:4]:
            for lab, seeds in (('new 20261148-50', NEW),) + ((('all six', CONF + NEW),) if not dry else ()):
                ga, gc = AB.sel(vf, bb, a, seeds), AB.sel(vf, bb, c, seeds)
                if ga is not None and gc is not None:
                    jobs.append(('contrast', seeds, (dict(experiment='E2 P', dataset='VIPL-HR', backbone=bb, name=name, contrast=f'{a}-{c}', seeds=lab), ga, gc, b)))
    # E2 R (A6 item 9) on VIPL-HR
    for base in A.BASES:
        g = AB.sel(vf, RF, A.arm_name(base, '-mcc'), CONF)
        if g is not None:
            jobs.append(('arm', CONF, (dict(experiment='E2 R', dataset='VIPL-HR', backbone=RF, setting='RhythmFormer-MCC', arm=A.arm_name(base, '-mcc')), g, b)))
    for name, a, c in A.PAIRS:
        ga, gc = AB.sel(vf, RF, A.arm_name(a, '-mcc'), CONF), AB.sel(vf, RF, A.arm_name(c, '-mcc'), CONF)
        if ga is not None and gc is not None:
            jobs.append(('contrast', CONF, (dict(experiment='E2 R', dataset='VIPL-HR', backbone=RF, setting='RhythmFormer-MCC', name=name,
                                                 base_contrast=f'{a}-{c}', contrast=f'{A.arm_name(a, "-mcc")}-{A.arm_name(c, "-mcc")}'), ga, gc, b)))
        parts = [AB.sel(vf, RF, A.arm_name(a, '-mcc'), CONF), AB.sel(vf, RF, A.arm_name(c, '-mcc'), CONF), AB.sel(vf, RF, a, CONF), AB.sel(vf, RF, c, CONF)]
        if all(q is not None for q in parts):
            jobs.append(('interaction', CONF, (dict(experiment='E2 R', dataset='VIPL-HR', interaction='loss', setting='RhythmFormer-MCC',
                                                    reference_setting='RhythmFormer', backbone=RF, name=name, base_contrast=f'{a}-{c}'), parts, b)))
    # E3 on VIPL-HR and on the test sets
    frames = {'VIPL': vf}
    if not dry:
        AB.CHECK = CACHE / 'test_check'
        AB.stage2_folders = test_folders
        for ds in ('MMPD', 'MCD'):
            frames[ds] = AB.load('test', ds)[0]
    for ds, f in frames.items():
        g = AB.sel(f, RF, '(0B)', CONF)
        if g is not None:
            jobs.append(('arm', CONF, (dict(experiment='E3', dataset=NAMES[ds], backbone=RF, arm='(0B)'), g, b)))
        for a, c in E3_PAIRS:
            ga, gc = AB.sel(f, RF, a, CONF), AB.sel(f, RF, c, CONF)
            if ga is not None and gc is not None:
                jobs.append(('contrast', CONF, (dict(experiment='E3', dataset=NAMES[ds], backbone=RF, contrast=f'{a}-{c}'), ga, gc, b)))
    print('jobs', len(jobs), round(time.time() - t0), flush=True)
    res = AB.execute(jobs, workers, 'a9 e2e3')
    out.mkdir(parents=True)
    allrows = [dict(kind=k, **r) for k, rows in res.items() for r in rows]
    df = pd.DataFrame(allrows)
    for exp, sub in (('E2 P', 'E2_P'), ('E2 R', 'E2_R'), ('E3', 'E3')):
        x = df[df.experiment.eq(exp)]
        for kind in ('arm', 'contrast', 'interaction'):
            y = x[x.kind.eq(kind)].dropna(axis=1, how='all')
            if len(y):
                y.to_csv(out / f'{sub}_{kind}s.csv', index=False)
    if not dry:
        arms, con = ra2_beside()
        e3 = df[df.experiment.eq('E3') & df.kind.eq('contrast')].copy()
        rows = []
        for r in e3.itertuples():
            ds = {'MMPD-B': 'MMPD-B', 'MCD-rPPG test': 'MCD-rPPG test'}.get(r.dataset)
            o, sign = None, 1
            if ds:
                q = con[con.dataset.eq(ds) & con.metric.eq(r.metric)]
                if r.contrast == '(0)-(0B)':
                    o, sign = q[q.contrast.eq('(0B)-(0)')], -1
                else:
                    o = q[q.contrast.eq(r.contrast)]
            if o is not None and len(o):
                o = o.iloc[0]
                lo, hi = (o.ci_lo, o.ci_hi) if sign == 1 else (-o.ci_hi, -o.ci_lo)
                rows.append(dict(dataset=r.dataset, contrast=r.contrast, metric=r.metric, october_point=sign * o.point, october_ci_lo=lo,
                                 october_ci_hi=hi, october_windows=o.windows))
        if rows:
            pd.DataFrame(rows).to_csv(out / 'E3_october_beside.csv', index=False)
    info = dict(seconds=time.time() - t0, jobs=len(jobs), b=b, dry=dry, defective_new=bad_new, defective_old=sorted(bad_old))
    (out / 'run_info.json').write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['vipl_readouts', 'test_check', 'analyze'])
    ap.add_argument('--workers', type=int, default=2), ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--folder'), ap.add_argument('--out'), ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'vipl_readouts':
        if a.folder:
            cmd_vipl_readouts(a.workers, a.folder, a.out)
        else:
            cmd_vipl_readouts(a.workers)
    elif a.cmd == 'test_check':
        cmd_test_check(a.workers)
    else:
        cmd_analyze(a.workers, a.b, a.dry)


if __name__ == '__main__':
    main()
