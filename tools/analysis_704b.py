"""OPS2 brief 02 7.04 / A6 items 3, 8(d) and 9 (PI 7.07m item 2): analysis of the P runs (new seeds, second pool) and of the
conditional RhythmFormer MCC runs (R), with tools/analysis_704.py imported unchanged (its bytes are those of the
delivered 7.04 analysis): the same step-(b) code, loading, window support and pairing, crossed subject x seed bootstrap
(B = 10,000, draws default_rng(20261110)), metrics (M1-M3 of the test run, MAE over verified windows, high-HR hit rate)
and M7. Values only, no tiers (A6: post hoc).

Commands (main environment):
  check PART            step (b) of the test run (eval5/stage3_check.py, unchanged) over the stage-2 folders of the test run,
                        the 7.04 inference folder and the P/R inference folders that exist (cache/train704/infer_<group>/
                        <pass>); then the comparison with the stored step-(b) outputs (support equal, every stored key's
                        readouts equal; otherwise exit 1). Written to cache/train704/analysis_b/<PART>/check/<dataset>/.
  r PART                R (A6 item 9): settings RhythmFormer (confirmatory arms with the exploratory (i-simple), (ii-lag))
                        and RhythmFormer-MCC (arms '(x-mcc)'), seeds 20261131-33: arms, the contrasts of A6 item 7, the loss
                        interaction RhythmFormer-MCC - RhythmFormer, MCD-rPPG M7; test part: statements 8(b) (C2, C3; the
                        negative Pearson contrast 'lies below 0' = upper bound < 0, PI 7.07i item 3) and 8(c). The
                        RhythmFormer contrasts are recomputed only to check the stored stage (c) rows (reported: stored).
  p                     P (A6 items 3 and 8(d)), test part: per backbone (PhysNet, FactorizePhys, RhythmFormer), arms on
                        the new seeds 20261148-50; C1-C4 on the new seeds and on all six seeds (paired by seed); 8(d): a
                        confirmatory contrast whose test-run high-HR interval excluded 0 is replicated if its new-seed
                        interval excludes 0 in the same direction, reversed if in the opposite one, not replicated
                        otherwise; for the other confirmatory contrasts only the intervals are given.
Test switches (validation on existing outputs; never for reported values): p --dry-seeds uses the confirmatory seeds as
the 'new' seeds (C1-C4 must equal the stored rows); r --dry-backbone PhysNet maps RhythmFormer(-MCC) to PhysNet(-MCC)
(must equal the 7.04 PhysNet-MCC rows).
Outputs: outputs/704/analysis_b/<PART>/{r,p}/... ; analysis_hashes.txt. No run.json in report bundles (PI 7.07m item 3).
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_704 as A  # noqa: E402

W = A.W
CHECK = W / 'cache' / 'train704' / 'analysis_b'
OUTR = W / 'outputs' / '704' / 'analysis_b'
GROUPS = {'test': {'MMPD': ['P_PhysNet_FPm/MMPD_B', 'P_RhythmFormer/MMPD_B', 'R/MMPD_B'],
                   'MCD': ['P_PhysNet_FPm/MCD_test', 'P_RhythmFormer/MCD_test', 'R/MCD_test']},
          'dev': {'MMPD': ['R/MMPD_A'], 'MCD': ['R/MCD_dev']}}
CONF = [20261131, 20261132, 20261133]
NEW = [20261148, 20261149, 20261150]
RF = 'RhythmFormer'
NAMES = {'PhysNet': 'PhysNet', 'FPm': 'FactorizePhys (matched)', 'RhythmFormer': 'RhythmFormer'}


def extra_folders(part, ds):
    out = []
    for g in GROUPS[part][ds]:
        grp, pas = g.split('/')
        f = W / 'cache' / 'train704' / f'infer_{grp}' / pas
        if (f / ds / 'checkpoints.csv').exists() and (f / ds / 'stage2_hashes.txt').exists():  # finished passes only
            out.append(f)
    return out


def stage2_folders(part, ds):
    P = A.PARTS[part]
    return [P['root'] / 'stage2_conf', P['root'] / 'stage2_expl', P['infer'][ds]] + extra_folders(part, ds)


def cmd_check(part, workers):
    P = A.PARTS[part]
    out = CHECK / part / 'check'
    res = []
    for ds in ('MMPD', 'MCD'):
        if (out / ds).exists():
            sys.exit(f'{out / ds} exists; move it aside first')
        cmd = ['nice', '-n', '16', 'taskset', '-c', '0-13', A.PY_MAIN, '-B', '-s', str(A.EVAL / 'stage3_check.py'), '--dataset', ds,
               '--stage1', str(P['root'] / 'stage1'), '--stage2'] + [str(x) for x in stage2_folders(part, ds)] + \
              ['--vendor', str(A.VENDOR), '--out', str(out), '--workers', str(workers)]
        print('step (b):', ' '.join(cmd[6:]), flush=True)
        e = subprocess.run(cmd).returncode
        if e:
            sys.exit(f'stage3_check {ds} exit {e}')
        old, new = P['root'] / 'check' / ds, out / ds
        so, sn = pd.read_csv(old / 'support.csv', dtype=str), pd.read_csv(new / 'support.csv', dtype=str)
        ro, rn = pd.read_csv(old / 'readouts.csv.gz'), pd.read_csv(new / 'readouts.csv.gz')
        m = ro.merge(rn, on=['key', 'window_id', 'layer'], how='left', suffixes=('_stored', '_new'), indicator=True)
        same = ((m.predicted_HR_stored == m.predicted_HR_new) | (m.predicted_HR_stored.isna() & m.predicted_HR_new.isna())) & (m.valid_stored == m.valid_new)
        new_keys = sorted(set(rn.key) - set(ro.key))
        dn = pd.read_csv(new / 'defects.csv')
        r = dict(part=part, dataset=ds, folders=len(stage2_folders(part, ds)), stored_keys=ro.key.nunique(), stored_rows=len(ro),
                 stored_rows_missing_now=int((m._merge != 'both').sum()), stored_rows_equal=int(same.sum()), support_equal=bool(so.equals(sn)),
                 new_keys=len(new_keys), defective_new_keys=int(dn.key.isin(new_keys).sum()))
        r['passed'] = r['support_equal'] and r['stored_rows_equal'] == r['stored_rows'] and r['stored_rows_missing_now'] == 0
        print(json.dumps(r), flush=True)
        res.append(r)
    pd.DataFrame(res).to_csv(out / 'comparison_with_stored.csv', index=False)
    sys.exit(0 if all(r['passed'] for r in res) else 1)


def load(part, ds):
    chk = CHECK / part / 'check'
    if not pd.read_csv(chk / 'comparison_with_stored.csv').passed.all():
        sys.exit('step (b) comparison did not pass')
    keys = pd.concat([pd.read_csv(f / ds / 'checkpoints.csv') for f in stage2_folders(part, ds)], ignore_index=True)
    keys = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    d, bad = A.S3.load(ds, A.PARTS[part]['root'] / 'stage1' / ds, chk / ds, keys, 'automatic')
    prim = d[d.variant.isin(A.S3.PRIMARY) & d.layer.eq('HAR')]
    main_pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
    pw = A.S3.restrict(prim[prim.population.eq(main_pop)])
    m7 = A.S3.restrict(prim[prim.population.eq('M7_both_states')]) if ds == 'MCD' else None
    return pw, m7, sorted(bad)


def run_job(a):
    kind, seeds, payload = a
    A.SEEDS = list(seeds)
    return kind, getattr(A, f'job_{kind}')(payload)


def execute(jobs, workers, label):
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=A.init_worker) as ex:
        for n, (kind, rows) in enumerate(ex.map(run_job, jobs), 1):
            res.setdefault(kind, []).extend(rows)
            if n % 10 == 0 or n == len(jobs):
                print(f'{label} job {n}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    return res


def sel(frame, bb, arm, seeds):
    g = frame[frame.backbone.eq(bb) & frame.strategy.eq(arm) & frame.seed.isin(seeds)]
    return g if len(g) and not set(seeds) - set(g.seed) else None


def stored(part, ds):
    conf, sec, arms, m7 = A.stored_rows(part, ds)
    pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
    st = pd.concat([conf, sec], ignore_index=True)
    return st[st.analysis.eq('primary') & st.layer.eq('HAR') & st.population.eq(pop)]


def write(out, name, rows):
    if rows:
        pd.DataFrame(rows).to_csv(out / name, index=False)


def finish(out, info):
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob('*.csv'))
    (out / 'analysis_hashes.txt').write_text(''.join(f'{A.file_sha(out / n)}  {n}\n' for n in names))
    (out / 'run_info.json').write_text(json.dumps(info, indent=1, default=str))  # not for report bundles (operator-free anyway)
    print(json.dumps(info, indent=1, default=str), flush=True)


# ---------------------------------------------------------------------------------------------------------------- R
def cmd_r(part, b, workers, dry_backbone=None):
    A.setup()
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    bb_key = dry_backbone or RF
    bb = NAMES[bb_key]
    settings = {'RhythmFormer': '', 'RhythmFormer-MCC': '-mcc'}
    out = OUTR / part / ('r' if not dry_backbone else f'r_dry_{dry_backbone}')
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    allres = {}
    for ds in ('MMPD', 'MCD'):
        pw, m7, bad = load(part, ds)
        jobs = []
        for s, suf in settings.items():
            for base in A.BASES:
                arm = A.arm_name(base, suf)
                g = sel(pw, bb, arm, CONF)
                if g is None:
                    continue
                jobs.append(('arm', CONF, (dict(dataset=ds, setting=s, backbone=bb, arm=arm, base_arm=base), g, b)))
                if m7 is not None and sel(m7, bb, arm, CONF) is not None:
                    jobs.append(('m7_arm', CONF, (dict(dataset=ds, setting=s, backbone=bb, arm=arm, base_arm=base), sel(m7, bb, arm, CONF), b)))
            for name, a, c in A.PAIRS:
                ga, gc = sel(pw, bb, A.arm_name(a, suf), CONF), sel(pw, bb, A.arm_name(c, suf), CONF)
                if ga is None or gc is None:
                    continue
                base = dict(dataset=ds, setting=s, backbone=bb, name=name, base_contrast=f'{a}-{c}', contrast=f'{A.arm_name(a, suf)}-{A.arm_name(c, suf)}')
                jobs.append(('contrast', CONF, (base, ga, gc, b)))
                if m7 is not None:
                    ma, mc = sel(m7, bb, A.arm_name(a, suf), CONF), sel(m7, bb, A.arm_name(c, suf), CONF)
                    if ma is not None and mc is not None:
                        jobs.append(('m7_contrast', CONF, (base, ma, mc, b)))
        for name, a, c in A.PAIRS:
            parts = [sel(pw, bb, A.arm_name(a, '-mcc'), CONF), sel(pw, bb, A.arm_name(c, '-mcc'), CONF), sel(pw, bb, a, CONF), sel(pw, bb, c, CONF)]
            if any(q is None for q in parts):
                continue
            base = dict(dataset=ds, interaction='loss', setting='RhythmFormer-MCC', reference_setting='RhythmFormer', backbone=bb, name=name, base_contrast=f'{a}-{c}')
            jobs.append(('interaction', CONF, (base, parts, b)))
            if m7 is not None:
                mp_ = [sel(m7, bb, A.arm_name(a, '-mcc'), CONF), sel(m7, bb, A.arm_name(c, '-mcc'), CONF), sel(m7, bb, a, CONF), sel(m7, bb, c, CONF)]
                if all(q is not None for q in mp_):
                    jobs.append(('m7_interaction', CONF, (base, mp_, b)))
        res = execute(jobs, workers, f'r {part} {ds}')
        res['excluded_keys'] = bad
        allres[ds] = res
    out.mkdir(parents=True)
    repro = []
    for ds, res in allres.items():
        dsd = out / ds
        dsd.mkdir()
        for kind, fn in (('arm', 'arms'), ('contrast', 'contrasts'), ('interaction', 'interactions'), ('m7_arm', 'm7_arms'),
                         ('m7_contrast', 'm7_contrasts'), ('m7_interaction', 'm7_interactions')):
            write(dsd, f'{fn}.csv', res.get(kind))
        st = stored(part, ds)
        for r in [x for x in res.get('contrast', []) if x['setting'] == 'RhythmFormer' and x['metric'] != 'hit_high']:
            s = st[st.backbone.eq(r['backbone']) & st.contrast.eq(r['contrast']) & st.metric.eq(r['metric'])]
            if len(s):
                s = s.iloc[0]
                repro.append(dict(dataset=ds, item=r['contrast'], metric=r['metric'], windows_equal=int(s.windows) == int(r['windows']),
                                  max_abs_difference=float(np.nanmax(np.abs([s.point - r['point'], s.ci_lo - r['ci_lo'], s.ci_hi - r['ci_hi']])))))
    write(out, 'reproduction.csv', repro)
    st_rows = statements_r(part, allres, bb) if part == 'test' else []
    write(out, 'statements.csv', st_rows)
    rp = pd.DataFrame(repro)
    finish(out, dict(command='r', part=part, dry_backbone=dry_backbone or '', started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), b=b,
                     reproduction=dict(rows=len(rp), all_windows_equal=bool(rp.windows_equal.all()) if len(rp) else None,
                                       max_abs_difference=float(rp.max_abs_difference.max()) if len(rp) else None),
                     statements=len(st_rows)))


def statements_r(part, allres, bb):
    out = []
    bname = 'RhythmFormer' if bb == 'RhythmFormer' else bb.split(' ')[0]
    for ds, res in allres.items():
        setname = A.PARTS[part]['name'][ds]
        st = stored(part, ds)
        st = st[st.metric.eq('MAE_high') & st.backbone.eq(bb)]
        c = pd.DataFrame(res['contrast'])
        c = c[c.metric.eq('MAE_high') & c.setting.eq('RhythmFormer-MCC')]
        it = pd.DataFrame(res['interaction'])
        it = it[it.metric.eq('MAE_high')]

        def get_mcc(name):
            s = c[c.name.eq(name)]
            return (s.iloc[0].point, s.iloc[0].ci_lo, s.iloc[0].ci_hi) if len(s) else None

        def get_ref(name):
            a, cc = [(x, y) for n, x, y in A.PAIRS if n == name][0]
            s = st[st.contrast.eq(f'{a}-{cc}')]
            return (s.iloc[0].point, s.iloc[0].ci_lo, s.iloc[0].ci_hi) if len(s) else None
        for name, what in (('C2', 'editing'), ('C3', 'time resampling')):
            v, ref = get_mcc(name), get_ref(name)
            ii = it[it.name.eq(name)]
            if not v or not ref or not len(ii):
                continue
            lo_int = ii.iloc[0].ci_lo
            if v[2] < 0:
                s, kind = f'With the MCC loss, correcting the alignment of {what} still lowered the high-HR error of {bname} on {setname}.', 'still lowered'
            elif v[1] > 0:
                s, kind = f'With the MCC loss, the corrected {what} had the higher high-HR error for {bname} on {setname}.', 'higher'
            elif ref[2] < 0 and lo_int > 0:
                s, kind = f'With the MCC loss, the cost of misaligned {what} was absorbed for {bname} on {setname}.', 'absorbed'
            else:
                s, kind = f'With the MCC loss, no effect of correcting the alignment of {what} was established for {bname} on {setname}.', 'no effect'
            out.append(dict(dataset=ds, test_set=setname, item='8(b)', setting='RhythmFormer-MCC', contrast=name, point=v[0], ci_lo=v[1], ci_hi=v[2],
                            pearson_point=ref[0], pearson_ci_lo=ref[1], pearson_ci_hi=ref[2], pearson_source='stored stage (c)',
                            loss_interaction_point=ii.iloc[0].point, loss_interaction_ci_lo=lo_int, loss_interaction_ci_hi=ii.iloc[0].ci_hi,
                            kind=kind, statement=s))
        v = get_mcc('ii-pc_vs_i-simple')
        if v:
            s = ('editing had a lower high-HR error than time resampling without a label shift' if v[2] < 0 else
                 'time resampling without a label shift had the lower high-HR error' if v[1] > 0 else 'no difference between the two was established')
            out.append(dict(dataset=ds, test_set=setname, item='8(c)', setting='RhythmFormer-MCC', contrast='(ii-pc)-(i-simple)', point=v[0], ci_lo=v[1], ci_hi=v[2], statement=s))
    return out


# ---------------------------------------------------------------------------------------------------------------- P
def cmd_p(b, workers, dry=False):
    A.setup()
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    new = CONF if dry else NEW
    six = CONF if dry else CONF + NEW
    out = OUTR / 'test' / ('p' if not dry else 'p_dry')
    if out.exists():
        sys.exit(f'{out} exists; move it aside first')
    allres = {}
    for ds in ('MMPD', 'MCD'):
        pw, m7, bad = load('test', ds)
        jobs = []
        for key, bb in NAMES.items():
            for base in ('(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-pc)'):
                g = sel(pw, bb, base, new)
                if g is not None:
                    jobs.append(('arm', new, (dict(dataset=ds, seeds='new', backbone=bb, arm=base), g, b)))
            for tag, seeds in (('new', new), ('six', six)):
                for name, a, c in A.PAIRS[:4]:
                    ga, gc = sel(pw, bb, a, seeds), sel(pw, bb, c, seeds)
                    if ga is None or gc is None:
                        continue
                    jobs.append(('contrast', seeds, (dict(dataset=ds, seeds=tag, backbone=bb, name=name, contrast=f'{a}-{c}'), ga, gc, b)))
        res = execute(jobs, workers, f'p {ds}')
        res['excluded_keys'] = bad
        allres[ds] = res
    out.mkdir(parents=True)
    rows8d, repro = [], []
    for ds, res in allres.items():
        dsd = out / ds
        dsd.mkdir()
        write(dsd, 'arms.csv', res.get('arm'))
        write(dsd, 'contrasts.csv', res.get('contrast'))
        st = stored('test', ds)
        c = pd.DataFrame(res.get('contrast', []))
        for r in c[c.metric.eq('MAE_high') & c.seeds.eq('new')].itertuples():
            s = st[st.backbone.eq(r.backbone) & st.contrast.eq(r.contrast) & st.metric.eq('MAE_high')]
            if not len(s):
                continue
            s = s.iloc[0]
            excl = s.ci_hi < 0 or s.ci_lo > 0
            if excl:
                same = (s.ci_hi < 0 and r.ci_hi < 0) or (s.ci_lo > 0 and r.ci_lo > 0)
                opp = (s.ci_hi < 0 and r.ci_lo > 0) or (s.ci_lo > 0 and r.ci_hi < 0)
                verdict = 'replicated' if same else ('reversed' if opp else 'not replicated')
            else:
                verdict = 'intervals only'
            rows8d.append(dict(dataset=ds, test_set=A.PARTS['test']['name'][ds], backbone=r.backbone, contrast=r.name,
                               test_run_point=s.point, test_run_ci_lo=s.ci_lo, test_run_ci_hi=s.ci_hi, test_run_excluded_0=excl,
                               new_point=r.point, new_ci_lo=r.ci_lo, new_ci_hi=r.ci_hi, verdict_8d=verdict))
            if dry:
                repro.append(dict(dataset=ds, item=f'{r.backbone} {r.contrast}', windows_equal=int(s.windows) == int(r.windows),
                                  max_abs_difference=float(np.nanmax(np.abs([s.point - r.point, s.ci_lo - r.ci_lo, s.ci_hi - r.ci_hi])))))
    write(out, 'replication_8d.csv', rows8d)
    write(out, 'dry_reproduction.csv', repro)
    finish(out, dict(command='p', dry=dry, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), b=b, new_seeds=new, six_seeds=six,
                     rows_8d=len(rows8d), dry_max_abs_difference=max([r['max_abs_difference'] for r in repro], default=None)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['check', 'r', 'p'])
    ap.add_argument('part', nargs='?', default='test', choices=list(A.PARTS))
    ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dry-seeds', action='store_true')
    ap.add_argument('--dry-backbone', choices=['PhysNet', 'FPm'])
    ap.add_argument('--check-root', help='step-(b) root (<root>/<PART>/check); test runs: the 7.04 one')
    ap.add_argument('--out-root', help='output root; test runs elsewhere')
    a = ap.parse_args()
    global CHECK, OUTR
    if a.check_root:
        CHECK = Path(a.check_root)
    if a.out_root:
        OUTR = Path(a.out_root)
    if a.command == 'check':
        cmd_check(a.part, a.workers)
    elif a.command == 'r':
        cmd_r(a.part, a.b, a.workers, a.dry_backbone)
    else:
        cmd_p(a.b, a.workers, a.dry_seeds)


if __name__ == '__main__':
    main()
