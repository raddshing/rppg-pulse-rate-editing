"""OPS2 7.20 4: evaluation of the A10 (B1) groups B1_PhysNet, B1_FPm and B1_RhythmFormer, run only after the SHA-256 lists of
their prediction files are committed (tools/pub720_predsha.py; A10 item 6c). The A9 evaluation code is imported unchanged
(tools/a9_eval/a9_groups.py and a9_e2e3.py: VIPL-HR readouts of eval5/vipl_c.py, step (b) of the test run, loading;
tools/analysis_704.py job_arm / job_contrast: M1-M3 of the test run, MAE over verified windows, high-HR hit rate; crossed
subject x seed bootstrap of b1/boot.py, B = 10,000, default_rng(20261110), seeds 20261131-20261133 paired by index, also
between U-low and U-full-m runs). Test sets as for E1: MMPD-B (Q*), MCD-rPPG test half (draft rule), VIPL-HR primary set
(HAR). This file adds the inference folders of A10 (cache/infer720/) and the contrasts of A10 items 7 and 8:

  arms: (0-b1low), (i-simple-b1low), (i-b1low), (i-pc-b1low) on U-low and (0-b1fullm) on U-full-m: values (job_arm).
  contrasts (job_contrast): C3 = (i-pc)-(i); (i)-(i-simple); (i-simple)-(0); (i-pc)-(0) (all on U-low);
    (0, U-full-m)-(0, U-low).
  recovery ratio (item 7d): [(i-simple)-(0)] / [(0, U-full-m)-(0, U-low)] of the high-HR MAE, the two differences and their
    ratio computed within each bootstrap replicate (the same draws as boot.summarize), on the windows valid for every key
    of the three arms (analysis_704.four). Its percentile interval is reported only if the interval of the denominator
    lies below 0; otherwise the two differences are reported without the ratio. Check: numerator and denominator (point
    and interval) equal boot.summarize on the same windows.
  descriptions (item 8a-d) for the high-HR MAE, with the fixed texts; summary sentences (item 8e) by the command summary
    over the three groups.

Commands (main environment): vipl_readouts GROUP | test_check GROUP | analyze GROUP [--dry] | summary
Outputs: outputs/720/eval/<GROUP>/run/{arms,contrasts,ratio,descriptions}.csv (aggregate) and outputs/720/eval/summary/;
server-only readouts in cache/a9_eval/<GROUP>/.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'a9_eval'))  # 7.21: published in tools/a10/; a9_e2e3.py and a9_groups.py are in tools/a9_eval/
import a9_e2e3 as E  # noqa: E402
import a9_groups as G9  # noqa: E402
from a9_e2e3 import A, AB, C, CONF, np, pd, W  # noqa: E402

I = W / 'cache' / 'infer720'
OUT = W / 'outputs' / '720' / 'eval'
GROUPS = ('B1_PhysNet', 'B1_FPm', 'B1_RhythmFormer')
BB = {'B1_PhysNet': 'PhysNet', 'B1_FPm': 'FactorizePhys (matched)', 'B1_RhythmFormer': 'RhythmFormer'}
NAMES = G9.NAMES
ARMS = ('(0-b1low)', '(i-simple-b1low)', '(i-b1low)', '(i-pc-b1low)', '(0-b1fullm)')
# name, a, c (contrast a - c), description key
PAIRS = [('C3', '(i-pc-b1low)', '(i-b1low)', 'C3'),
         ('(i)-(i-simple)', '(i-b1low)', '(i-simple-b1low)', 'shift'),
         ('(i-simple)-(0)', '(i-simple-b1low)', '(0-b1low)', 'synth'),
         ('(i-pc)-(0)', '(i-pc-b1low)', '(0-b1low)', 'synth'),
         ('(0, U-full-m)-(0, U-low)', '(0-b1fullm)', '(0-b1low)', 'recorded')]
SHORT = {'(i-simple-b1low)': '(i-simple)', '(i-pc-b1low)': '(i-pc)'}
RATIO_ARMS = ('(i-simple-b1low)', '(0-b1low)', '(0-b1fullm)')   # numerator arm, (0, U-low), (0, U-full-m)
PHASE = '64–65°'


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
    """As a9_groups.frames, with the inference folders of A10."""
    old, _, _, bad_old = C.vipl_frame()
    new, bad_new = E.vipl_new_frame(E.CACHE / g / ('vipl_dry' if dry else 'vipl'))
    out = {'VIPL': C.SA.restrict(pd.concat([old, new], ignore_index=True).pipe(C.primary)).drop_duplicates(['key', 'window_id', 'layer'])}
    if not dry:
        AB.CHECK = E.CACHE / g / 'test_check'
        AB.stage2_folders = folders_for(g)
        for ds in ('MMPD', 'MCD'):
            out[ds] = AB.load('test', ds)[0]
    return out, bad_new


def describe(key, arm, lo, hi):
    """A10 item 8 (a-d), fixed texts."""
    below, above = hi < 0, lo > 0
    if key == 'C3':
        return ('correcting the delay rule lowered the high-HR error of resampling' if below else 'raised it' if above
                else 'not established')
    if key == 'shift':
        return ('the delay-rule shift raised the high-HR error relative to plain resampling' if above else 'lowered it' if below
                else 'not established')
    if key == 'synth':
        return f'{SHORT[arm]} lowered the high-HR error' if below else 'raised it' if above else 'not established'
    if key == 'recorded':
        return ('recorded high heart rates lowered the high-HR error' if below else 'raised it' if above
                else 'not established')
    raise KeyError(key)


def summarize_reps(f, subjects, seeds, b):
    """boot.summarize, also returning the replicate values (same draws, same seed choice, same nan handling)."""
    boot = A.boot
    full = f(np.ones((1, len(subjects))))
    keys = list(next(iter(full.values())))
    point = {k: float(np.nanmean([full[sd][k][0] for sd in seeds])) for k in keys}
    isub, iseed = boot.draws(len(subjects), len(seeds), b)
    reps = {k: np.empty(b) for k in keys}
    chunk = 500
    for a in range(0, b, chunk):
        Wsub = boot.subject_weights(isub[a:a + chunk], len(subjects))
        vals = f(Wsub)
        for k in keys:
            stack = np.stack([vals[sd][k] for sd in seeds])
            chosen = stack[iseed[a:a + chunk].T, np.arange(Wsub.shape[0])[None, :]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reps[k][a:a + chunk] = np.nanmean(chosen, axis=0)
    return point, reps


def pct(v):
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def ratio_job(base, simple, low, full, b):
    """A10 item 7d on the high-HR MAE: [(i-simple)-(0)] / [(0, U-full-m)-(0, U-low)] within each replicate."""
    boot = A.boot
    parts, nw = A.four([simple, low, full])
    subjects = boot.numeric_subject_order(set().union(*[set(q.subject_id) for q in parts]))
    smp = [boot.Sample(q, subjects, CONF) for q in parts]

    def fn(Wt, r, p, s):
        m = s == 'high'
        return {'MAE_high': boot.mae(Wt[:, m], r[m], p[m])}

    def f(Wt):
        v = [boot.per_seed(sm, fn, Wt) for sm in smp]
        return {sd: {'num': v[0][sd]['MAE_high'] - v[1][sd]['MAE_high'],
                     'den': v[2][sd]['MAE_high'] - v[1][sd]['MAE_high']} for sd in CONF}
    pt, reps = summarize_reps(f, subjects, CONF, b)
    chk_pt, chk_ci = boot.summarize(f, subjects, CONF, b)  # the unchanged function on the same windows
    num_ci, den_ci = pct(reps['num']), pct(reps['den'])
    same = all(abs(chk_pt[k] - pt[k]) < 1e-12 for k in ('num', 'den')) and \
        all(np.allclose(chk_ci[k], c, rtol=0, atol=1e-12) for k, c in (('num', num_ci), ('den', den_ci)))
    reported = den_ci[1] < 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = reps['num'] / reps['den']
    r_ci = pct(rr) if reported else (np.nan, np.nan)
    windows_high = int(parts[0][parts[0].seed.eq(CONF[0])].set.eq('high').sum())
    return dict(base, windows=nw, windows_high=windows_high, subjects=len(subjects), metric='MAE_high',
                numerator='(i-simple)-(0)', num_point=pt['num'], num_ci_lo=num_ci[0], num_ci_hi=num_ci[1],
                denominator='(0, U-full-m)-(0, U-low)', den_point=pt['den'], den_ci_lo=den_ci[0], den_ci_hi=den_ci[1],
                denominator_interval_below_0=reported,
                ratio_point=pt['num'] / pt['den'] if reported else np.nan, ratio_ci_lo=r_ci[0], ratio_ci_hi=r_ci[1],
                replicates_nonfinite_ratio=int((~np.isfinite(rr)).sum()) if reported else np.nan,
                check_equal_to_boot_summarize=bool(same))


def cmd_analyze(g, workers, b, dry=False):
    A.setup()
    t0 = time.time()
    out = OUT / g / ('dry' if dry else 'run')
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    fr, bad_new = frames(g, dry)
    bb = BB[g]
    jobs, ratios = [], []
    for ds, f in fr.items():
        base = dict(dataset=NAMES[ds], experiment='B1', backbone=bb)
        for arm in ARMS:
            gg = AB.sel(f, bb, arm, CONF)
            if gg is not None:
                jobs.append(('arm', CONF, (dict(base, arm=arm), gg, b)))
        for name, a, c, _ in PAIRS:
            ga, gc = AB.sel(f, bb, a, CONF), AB.sel(f, bb, c, CONF)
            if ga is not None and gc is not None:
                jobs.append(('contrast', CONF, (dict(base, name=name, contrast=f'{a}-{c}'), ga, gc, b)))
        three = [AB.sel(f, bb, x, CONF) for x in RATIO_ARMS]
        if all(q is not None for q in three):
            ratios.append((base, *three))
    print('jobs', len(jobs), 'ratio cells', len(ratios), round(time.time() - t0), flush=True)
    res = AB.execute(jobs, workers, f'b1 {g}')
    A.SEEDS = list(CONF)
    rat = []
    for base, s, lo, fu in ratios:
        rat.append(ratio_job(base, s, lo, fu, b))
        print('ratio', base['dataset'], 'denominator interval below 0:', rat[-1]['denominator_interval_below_0'],
              'check', rat[-1]['check_equal_to_boot_summarize'], round(time.time() - t0), flush=True)
    out.mkdir(parents=True)
    df = pd.DataFrame([dict(kind=k, **r) for k, rows in res.items() for r in rows])
    for kind in ('arm', 'contrast'):
        y = df[df.kind.eq(kind)].dropna(axis=1, how='all')
        if len(y):
            y.to_csv(out / f'{kind}s.csv', index=False)
    rt = pd.DataFrame(rat)
    rt.to_csv(out / 'ratio.csv', index=False)
    con = df[df.kind.eq('contrast') & df.metric.eq('MAE_high')]
    rows = []
    for name, a, c, key in PAIRS:
        for r in con[con.name.eq(name)].itertuples():
            rows.append(dict(dataset=r.dataset, backbone=r.backbone, name=name, contrast=r.contrast, metric=r.metric,
                             windows=r.windows, subjects=r.subjects, point=r.point, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                             description=describe(key, a, r.ci_lo, r.ci_hi)))
    pd.DataFrame(rows).to_csv(out / 'descriptions.csv', index=False)
    # the ratio's numerator and denominator against the separate contrasts (equal when the windows are the same)
    cmp = []
    for r in rt.itertuples():
        for nm, p_, lo_, hi_ in (('(i-simple)-(0)', r.num_point, r.num_ci_lo, r.num_ci_hi),
                                 ('(0, U-full-m)-(0, U-low)', r.den_point, r.den_ci_lo, r.den_ci_hi)):
            q = con[con.name.eq(nm) & con.dataset.eq(r.dataset)]
            if len(q):
                q = q.iloc[0]
                cmp.append(dict(dataset=r.dataset, part=nm, windows_ratio=r.windows, windows_contrast=int(q.windows),
                                equal=bool(abs(q.point - p_) < 1e-12 and abs(q.ci_lo - lo_) < 1e-12 and abs(q.ci_hi - hi_) < 1e-12)))
    pd.DataFrame(cmp).to_csv(out / 'ratio_parts_vs_contrasts.csv', index=False)
    info = dict(group=g, seconds=time.time() - t0, jobs=len(jobs), ratio_cells=len(rat), b=b, dry=dry, defective_new=bad_new)
    (out / 'run_info.json').write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, default=str), flush=True)


def fmt(x):
    return f'{x:.2f}'


def cmd_summary():
    out = OUT / 'summary'
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    desc = pd.concat([pd.read_csv(OUT / g / 'run' / 'descriptions.csv') for g in GROUPS], ignore_index=True)
    rat = pd.concat([pd.read_csv(OUT / g / 'run' / 'ratio.csv') for g in GROUPS], ignore_index=True)
    sent = []
    for ds in ('VIPL-HR', 'MMPD-B', 'MCD-rPPG test'):
        c3 = desc[desc.name.eq('C3') & desc.dataset.eq(ds)]
        k = int((c3.ci_hi < 0).sum())
        sent.append(dict(kind='C3', dataset=ds, backbone='', backbones=len(c3), k=k,
                         sentence=(f'With training on UBFC-rPPG chunks below 100 bpm, correcting the delay rule lowered the high-HR '
                                   f'error of resampling (C3 interval below 0) for {k} of three backbones on {ds}, at a mean '
                                   f'phase error of {PHASE}.')))
    for r in rat.itertuples():
        if r.denominator_interval_below_0:
            sent.append(dict(kind='recovery', dataset=r.dataset, backbone=r.backbone, backbones=1, k=np.nan,
                             sentence=(f'Plain resampling recovered {fmt(r.ratio_point)} [{fmt(r.ratio_ci_lo)}, {fmt(r.ratio_ci_hi)}] '
                                       f'of the high-HR gain from recorded high heart rates of the same subjects for {r.backbone} '
                                       f'on {r.dataset},')))
    out.mkdir(parents=True)
    pd.DataFrame(sent).to_csv(out / 'summary_sentences.csv', index=False)
    desc.to_csv(out / 'descriptions_all.csv', index=False)
    rat.to_csv(out / 'ratio_all.csv', index=False)
    print(pd.DataFrame(sent).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['vipl_readouts', 'test_check', 'analyze', 'summary'])
    ap.add_argument('group', nargs='?', choices=GROUPS)
    ap.add_argument('--workers', type=int, default=2), ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'summary':
        cmd_summary()
    elif a.cmd == 'vipl_readouts':
        E.cmd_vipl_readouts(a.workers, I / a.group / 'vipl_main', E.CACHE / a.group / 'vipl')
    elif a.cmd == 'test_check':
        cmd_test_check(a.group, a.workers)
    else:
        cmd_analyze(a.group, a.workers, a.b, a.dry)


if __name__ == '__main__':
    main()
