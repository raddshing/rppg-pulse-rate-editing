"""OPS2 7.13 3-4: evaluation of the A9 groups E467 (items 4-6) and E1_<backbone> (item 1), run only after the SHA-256 lists of
their prediction files are committed (tools/pub713_predsha.py; A9 item 7). Same code as tools/a9_eval/a9_e2e3.py (VIPL-HR
readouts of eval5/vipl_c.py; step (b) of the test run; tools/analysis_704.py job_arm / job_contrast: M1-M3 of the test run,
MAE over verified windows, high-HR hit rate; crossed subject x seed bootstrap, B = 10,000, default_rng(20261110), seeds
20261131-20261133 paired by index). Test sets: MMPD-B (Q*), MCD-rPPG test half (draft rule), VIPL-HR primary set (HAR).

E467: E4 (os) per backbone: arm values, (os)-(0), (ii-pc)-(os), (i-simple)-(os). E6 RhythmFormer (ii-pc, second pool):
  arm values, (ii-pc-pool2)-(ii-pc) and C1 with the second pool (ii-pc-pool2)-(0). E7 FactorizePhys (ii-a) for a = 0.25, 0.5,
  0.75: arm values (high-HR MAE and hit rate among them), (ii-a)-(ii-pc) per a, and the least-squares slope of the seed-paired
  high-HR MAE on a over the five levels ((ii-pc) a = 0, (ii-a), (ii-lag) a = 1; windows valid for every key of the five arms;
  the same bootstrap) with the A9 sentence: increased only if the lower bound > 0, decreased only if the upper bound < 0,
  otherwise no trend established; and the mean absolute phase error at f* of each level: over the items of the first
  FactorizePhys (ii-pc) pool, |wrap(2 pi f* a (tau + T*/2 - d) / 30)| with T* = 1800/h* frames, d = 2, tau of the recording.
E1_<b>: arms (x-ubfc) values; C1-C4, (ii-lag)-(ii-pc), (i-simple)-(0), (ii-pc)-(i-simple) with UBFC-rPPG training; each
  described against the same contrast with PURE training on the same test set (stored: test run confirmatory.csv and
  secondary.csv; VIPL-HR stage (c1) confirmatory.csv and (c2) secondary_checkpoints.csv 'exploratory_arms'): 'in the same
  direction as with PURE training' if its interval excludes 0 on the side of the PURE point estimate, 'in the opposite
  direction' if it excludes 0 on the other side, 'not established' otherwise; '' (interval only) if the PURE interval included 0.
  Descriptions for high-HR MAE (the primary measure; also given for low-HR and all-window MAE). Summary sentence per set.
Commands: vipl_readouts GROUP | test_check GROUP | analyze GROUP [--dry] (main environment).
Outputs: outputs/a9_eval/<GROUP>/*.csv (aggregate); server-only readouts in cache/a9_eval/<GROUP>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a9_e2e3 as E  # noqa: E402
from a9_e2e3 import A, AB, C, CONF, np, pd, W  # noqa: E402

I = W / 'cache' / 'infer709'
GROUPS = ('E467', 'E1_PhysNet', 'E1_FPm', 'E1_RhythmFormer')
BB = {'E1_PhysNet': 'PhysNet', 'E1_FPm': 'FactorizePhys (matched)', 'E1_RhythmFormer': 'RhythmFormer'}
ALPHA = [('(ii-pc)', 0.0), ('(ii-a0.25)', 0.25), ('(ii-a0.5)', 0.5), ('(ii-a0.75)', 0.75), ('(ii-lag)', 1.0)]
E1_PAIRS = list(A.PAIRS[:4]) + [('', '(ii-lag)', '(ii-pc)'), ('', '(i-simple)', '(0)'), ('', '(ii-pc)', '(i-simple)')]
NAMES = {'MMPD': 'MMPD-B', 'MCD': 'MCD-rPPG test', 'VIPL': 'VIPL-HR'}


def ubfc(arm):
    return f'({arm.strip("()")}-ubfc)'


def folders_for(g):
    def f(part, ds):
        root = A.PARTS[part]['root']
        return [root / 'stage2_conf', root / 'stage2_expl', I / g / {'MMPD': 'MMPD_B', 'MCD': 'MCD_test'}[ds]]
    return f


def cmd_test_check(g, workers):
    AB.CHECK = E.CACHE / g / 'test_check'
    AB.stage2_folders = folders_for(g)
    AB.cmd_check('test', workers)


def frames(g, dry=False):
    old, _, _, bad_old = C.vipl_frame()
    new, bad_new = E.vipl_new_frame(E.CACHE / g / ('vipl_dry' if dry else 'vipl'))
    out = {'VIPL': C.SA.restrict(pd.concat([old, new], ignore_index=True).pipe(C.primary)).drop_duplicates(['key', 'window_id', 'layer'])}
    if not dry:
        AB.CHECK = E.CACHE / g / 'test_check'
        AB.stage2_folders = folders_for(g)
        for ds in ('MMPD', 'MCD'):
            out[ds] = AB.load('test', ds)[0]
    return out, bad_new


def slope_ci(parts, alphas, b):
    boot = A.boot
    subjects = boot.numeric_subject_order(set().union(*[set(q.subject_id) for q in parts]))
    smp = [boot.Sample(q, subjects, CONF) for q in parts]
    x = np.asarray(alphas, float)
    xc = x - x.mean()

    def fn(Wt, r, p, s):
        m = s == 'high'
        return {'MAE_high': boot.mae(Wt[:, m], r[m], p[m])}

    def f(Wt):
        v = [boot.per_seed(sm, fn, Wt) for sm in smp]
        out = {}
        for sd in CONF:
            Y = np.stack([v[i][sd]['MAE_high'] for i in range(len(parts))])
            out[sd] = {'slope_MAE_high_per_unit_alpha': (xc[:, None] * (Y - Y.mean(0))).sum(0) / (xc ** 2).sum()}
        return out
    return boot.summarize(f, subjects, CONF, b), len(subjects)


def phase_errors():
    """Mean absolute phase error at f* of each E7 level over the items of the first FactorizePhys (ii-pc) pool."""
    sys.path.insert(0, str(W / 'audit'))
    import a1_common as AC
    taus = AC.tau_table()
    m = pd.read_csv(W / 'cache' / 'pools' / 'FPm_polarity_explicit_all_manifest.csv')
    tau = m.recording_id.map(lambda r: float(taus[r]['tau_rec'])).to_numpy()
    h = m.target_HR.to_numpy(float)
    rows = []
    for arm, a in ALPHA:
        delay = a * (tau + (1800.0 / h) / 2 - 2)
        ph = np.angle(np.exp(1j * 2 * np.pi * (h / 60) * delay / 30))
        rows.append(dict(arm=arm, alpha=a, items=len(m), mean_abs_phase_error_deg=float(np.degrees(np.abs(ph)).mean()),
                         median_abs_phase_error_deg=float(np.median(np.degrees(np.abs(ph))))))
    return rows


def pure_reference():
    """Stored contrasts with PURE training: (dataset, backbone, contrast, metric) -> (point, lo, hi)."""
    ref = {}
    for ds in ('MMPD', 'MCD'):
        st = AB.stored('test', ds)
        for r in st.itertuples():
            ref[(NAMES[ds], r.backbone, r.contrast, r.metric)] = (r.point, r.ci_lo, r.ci_hi)
    c1 = pd.read_csv(C.VO / 'stage_c1' / 'VIPL' / 'confirmatory.csv')
    c2 = pd.read_csv(C.VC2 / 'secondary_checkpoints.csv')
    c2 = c2[c2.analysis.eq('exploratory_arms')]
    for t in (c1[c1.layer.eq('HAR') & c1.population.eq('primary')], c2[c2.layer.eq('HAR') & c2.population.eq('primary')]):
        for r in t.itertuples():
            ref[('VIPL-HR', r.backbone, r.contrast, r.metric)] = (r.point, r.ci_lo, r.ci_hi)
    return ref


def describe(point, lo, hi, ref):
    if ref is None or not np.isfinite(ref[1]) or (ref[1] <= 0 <= ref[2]):
        return ''
    side = np.sign(ref[0])
    if lo > 0 or hi < 0:
        return 'in the same direction as with PURE training' if np.sign(point) == side else 'in the opposite direction'
    return 'not established'


def cmd_analyze(g, workers, b, dry=False):
    A.setup()
    t0 = time.time()
    out = W / 'outputs' / 'a9_eval' / g / ('dry' if dry else 'run')
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    fr, bad_new = frames(g, dry)
    jobs, slopes = [], []
    for ds, f in fr.items():
        base = dict(dataset=NAMES[ds])
        if g == 'E467':
            for bb in E.BACK:
                gg = AB.sel(f, bb, '(os)', CONF)
                if gg is not None:
                    jobs.append(('arm', CONF, (dict(base, experiment='E4', backbone=bb, arm='(os)'), gg, b)))
                for a, c in (('(os)', '(0)'), ('(ii-pc)', '(os)'), ('(i-simple)', '(os)')):
                    ga, gc = AB.sel(f, bb, a, CONF), AB.sel(f, bb, c, CONF)
                    if ga is not None and gc is not None:
                        jobs.append(('contrast', CONF, (dict(base, experiment='E4', backbone=bb, contrast=f'{a}-{c}'), ga, gc, b)))
            gg = AB.sel(f, E.RF, '(ii-pc-pool2)', CONF)
            if gg is not None:
                jobs.append(('arm', CONF, (dict(base, experiment='E6', backbone=E.RF, arm='(ii-pc-pool2)'), gg, b)))
            for name, a, c in (('pool', '(ii-pc-pool2)', '(ii-pc)'), ('C1 second pool', '(ii-pc-pool2)', '(0)')):
                ga, gc = AB.sel(f, E.RF, a, CONF), AB.sel(f, E.RF, c, CONF)
                if ga is not None and gc is not None:
                    jobs.append(('contrast', CONF, (dict(base, experiment='E6', backbone=E.RF, name=name, contrast=f'{a}-{c}'), ga, gc, b)))
            fp = 'FactorizePhys (matched)'
            for arm, a in ALPHA[1:4]:
                gg = AB.sel(f, fp, arm, CONF)
                if gg is not None:
                    jobs.append(('arm', CONF, (dict(base, experiment='E7', backbone=fp, arm=arm, alpha=a), gg, b)))
                ga, gc = AB.sel(f, fp, arm, CONF), AB.sel(f, fp, '(ii-pc)', CONF)
                if ga is not None and gc is not None:
                    jobs.append(('contrast', CONF, (dict(base, experiment='E7', backbone=fp, alpha=a, contrast=f'{arm}-(ii-pc)'), ga, gc, b)))
            parts = [AB.sel(f, fp, arm, CONF) for arm, _ in ALPHA]
            if all(q is not None for q in parts):
                parts, nw = A.four(parts)
                (pt, ci), ns = slope_ci(parts, [a for _, a in ALPHA], b)
                k = 'slope_MAE_high_per_unit_alpha'
                lo, hi = ci[k]
                st = ('the high-HR error increased with the misalignment fraction' if lo > 0 else
                      'the high-HR error decreased with the misalignment fraction' if hi < 0 else 'no trend was established')
                slopes.append(dict(base, experiment='E7', backbone=fp, levels=';'.join(f'{x}={a}' for x, a in ALPHA), windows=nw,
                                   windows_high=int(parts[0][parts[0].seed.eq(CONF[0])].set.eq('high').sum()), subjects=ns,
                                   metric=k, point=pt[k], ci_lo=lo, ci_hi=hi, statement=st))
        else:
            bb = BB[g]
            for base_arm in ('(0)', '(i-simple)', '(i)', '(i-pc)', '(ii)', '(ii-pc)', '(ii-lag)'):
                gg = AB.sel(f, bb, ubfc(base_arm), CONF)
                if gg is not None:
                    jobs.append(('arm', CONF, (dict(base, experiment='E1', backbone=bb, arm=ubfc(base_arm)), gg, b)))
            for name, a, c in E1_PAIRS:
                ga, gc = AB.sel(f, bb, ubfc(a), CONF), AB.sel(f, bb, ubfc(c), CONF)
                if ga is not None and gc is not None:
                    jobs.append(('contrast', CONF, (dict(base, experiment='E1', backbone=bb, name=name, base_contrast=f'{a}-{c}',
                                                         contrast=f'{ubfc(a)}-{ubfc(c)}'), ga, gc, b)))
    print('jobs', len(jobs), round(time.time() - t0), flush=True)
    res = AB.execute(jobs, workers, f'a9 {g}')
    out.mkdir(parents=True)
    df = pd.DataFrame([dict(kind=k, **r) for k, rows in res.items() for r in rows])
    for kind in ('arm', 'contrast'):
        y = df[df.kind.eq(kind)].dropna(axis=1, how='all')
        if len(y):
            y.to_csv(out / f'{kind}s.csv', index=False)
    if g == 'E467':
        pd.DataFrame(slopes).to_csv(out / 'E7_slope.csv', index=False)
        pd.DataFrame(phase_errors()).to_csv(out / 'E7_phase_error.csv', index=False)
    else:
        ref = pure_reference()
        con = df[df.kind.eq('contrast')].copy()
        rows = []
        for r in con.itertuples():
            if r.metric not in ('MAE_high', 'MAE_low', 'MAE_all'):
                continue
            pr = ref.get((r.dataset, r.backbone, r.base_contrast, r.metric))
            rows.append(dict(dataset=r.dataset, backbone=r.backbone, name=r.name, contrast=r.contrast, metric=r.metric,
                             windows=r.windows, subjects=r.subjects, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             pure_point=pr[0] if pr else np.nan, pure_ci_lo=pr[1] if pr else np.nan, pure_ci_hi=pr[2] if pr else np.nan,
                             description=describe(r.point, r.ci_lo, r.ci_hi, pr)))
        pd.DataFrame(rows).to_csv(out / 'E1_descriptions.csv', index=False)
    info = dict(group=g, seconds=time.time() - t0, jobs=len(jobs), b=b, dry=dry, defective_new=bad_new)
    (out / 'run_info.json').write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['vipl_readouts', 'test_check', 'analyze'])
    ap.add_argument('group', choices=GROUPS)
    ap.add_argument('--workers', type=int, default=2), ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'vipl_readouts':
        E.cmd_vipl_readouts(a.workers, I / a.group / 'vipl_main', E.CACHE / a.group / 'vipl')
    elif a.cmd == 'test_check':
        cmd_test_check(a.group, a.workers)
    else:
        cmd_analyze(a.group, a.workers, a.b, a.dry)


if __name__ == '__main__':
    main()
