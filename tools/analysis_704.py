"""OPS2 brief 02 7.04 / amendment A6 (2026-11-20 entry), items 5-9: analysis of the post hoc block (settings of items 1-2
and rule 5). Item 3 (P, new seeds and second pool) is analysed later, after its training and inference.

Data parts: test = MMPD-B and the MCD-rPPG test half (folders of the test run); dev = MMPD-A and the MCD-rPPG development
half (folders of the rehearsal). Two commands per part:

  check PART     step (b) of the test run, unchanged (eval5/stage3_check.py, no October comparison), over the stage-2
                 folders of the test run (confirmatory, exploratory) and the 7.04 guarded-inference folder: readouts of
                 every key, defects, window support. Then the new outputs are compared with the step-(b) outputs stored
                 by the test run: support.csv must be equal, and every stored key's readouts must be equal (value and
                 validity); otherwise exit 1. Written to cache/train704/analysis/<PART>/check/<dataset>/.

  analyze PART   A6 items 6-8 with the test-run analysis functions unchanged (eval5/stage3_analyze.py: load, confirmatory
                 support and the key's own valid windows, pairing on windows valid for every key of both arms, crossed
                 subject x seed bootstrap of b1/boot.py, B = 10,000, draws default_rng(20261110)); seeds paired by index.
                 Primary band (HAR) on the main population (MMPD: Q*, MCD-rPPG: draft rule); MCD-rPPG M7 on the windows
                 of both states. Values only, no tiers (A6: post hoc).
    settings     PhysNet: confirmatory arms with the exploratory (i-simple), (ii-lag) of the A4 manifest; PhysNet-Std:
                 A6 item 1 with the rule-5 (ii), (ii-lag), (ii-pc); PhysNet-MCC, PhysNet-Std-MCC, FactorizePhys-MCC: A6
                 item 2; FactorizePhys: confirmatory arms with the exploratory (i-simple), (ii-lag).
    arms         per setting and arm: M1-M3 of the test run (arms.csv metrics), MAE over verified windows, and the
                 high-HR hit rate (|readout - reference| <= 3.515625 bpm, as the +-1-bin hit of the M4 family).
    contrasts    per setting (A6 item 7): C1-C4 analogues, (ii-pc)-(i-simple), (ii-lag)-(ii-pc).
    interactions label: PhysNet-Std - PhysNet; loss: PhysNet-MCC - PhysNet, PhysNet-Std-MCC - PhysNet-Std,
                 FactorizePhys-MCC - FactorizePhys; each contrast, [a - c] in the setting minus [a - c] in the reference
                 setting, on the windows valid for every key of the four arms, same subject x seed resample.
    M7           MCD-rPPG: per arm (families.m7_table: per run and subject, change of the recording means from before to
                 after), contrasts and interactions on the subjects present for every key of the arms concerned.
    confirmatory For the two confirmatory settings the arms, contrasts and M7 arms are recomputed only to check that the
                 stored stage-(c) rows are reproduced (reproduction.csv); the reported confirmatory values are the stored
                 ones (A6 item 5). The extra (exploratory-arm) contrasts are taken from the stored secondary.csv the same way.
    statements   test part only: 8(a)-(c) per setting and test set from the high-HR MAE intervals, and the item 9
                 condition (an 8(b) statement 'absorbed' for PhysNet or FactorizePhys). In 8(b), "the same contrast with
                 the negative Pearson loss lies below 0" means that the upper bound of its 95% interval is below 0 (T1);
                 fixed by the PI before the results (7.07i item 3, 11-24). Statements and item 9 use this reading only;
                 the column kind_if_point_reading (point estimate below 0) is for reference only.
Outputs: outputs/704/analysis/<PART>/<dataset>/ arms.csv, contrasts.csv, interactions.csv, m7_arms.csv, m7_contrasts.csv,
m7_interactions.csv, reproduction.csv; outputs/704/analysis/<PART>/statements.csv (test); analysis_hashes.txt, run.json.
Usage (main environment): analysis_704.py check PART | analyze PART [--b 10000] [--workers 4]
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import getpass  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

W = Path(__file__).resolve().parents[1]
EVAL = W / 'eval5'
VENDOR = W / 'vendor' / 'st8_6ebbc9cd'
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
PY_MAIN = CFG['python_main']
INFER = W / 'cache' / 'train704' / 'infer'
PARTS = {
    'test': dict(root=W / 'cache' / 'test', infer={'MMPD': INFER / 'MMPD_B', 'MCD': INFER / 'MCD_test'},
                 tag={'MMPD': 'automatic', 'MCD': 'halves'}, name={'MMPD': 'MMPD-B', 'MCD': 'the MCD-rPPG test half'}),
    'dev': dict(root=W / 'cache' / 'rehearsal' / 'part2', infer={'MMPD': INFER / 'MMPD_A', 'MCD': INFER / 'MCD_dev'},
                tag={'MMPD': 'automatic', 'MCD': 'halves'}, name={'MMPD': 'MMPD-A', 'MCD': 'the MCD-rPPG development half'}),
}
SEEDS = [20261131, 20261132, 20261133]
GRID = 3.515625
PN, FP = 'PhysNet', 'FactorizePhys (matched)'
SETTINGS = {'PhysNet': (PN, ''), 'PhysNet-Std': (PN, '-std'), 'PhysNet-MCC': (PN, '-mcc'), 'PhysNet-Std-MCC': (PN, '-std-mcc'),
            'FactorizePhys': (FP, ''), 'FactorizePhys-MCC': (FP, '-mcc')}
CONFIRMATORY = ('PhysNet', 'FactorizePhys')
BASES = ('(0)', '(i)', '(i-pc)', '(i-simple)', '(ii)', '(ii-lag)', '(ii-pc)')
PAIRS = [('C1', '(ii-pc)', '(0)'), ('C2', '(ii-pc)', '(ii)'), ('C3', '(i-pc)', '(i)'), ('C4', '(ii-pc)', '(i-pc)'),
         ('ii-pc_vs_i-simple', '(ii-pc)', '(i-simple)'), ('ii-lag_vs_ii-pc', '(ii-lag)', '(ii-pc)')]
INTERACTIONS = [('label', 'PhysNet-Std', 'PhysNet'), ('loss', 'PhysNet-MCC', 'PhysNet'),
                ('loss', 'PhysNet-Std-MCC', 'PhysNet-Std'), ('loss', 'FactorizePhys-MCC', 'FactorizePhys')]
S3 = boot = AN = None


def setup():
    global S3, boot, AN
    for p in (str(VENDOR), str(W / 'b1'), str(EVAL)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    import stage3_analyze  # noqa: E402
    import boot as b_  # noqa: E402
    import families  # noqa: E402
    S3, boot, AN = stage3_analyze, b_, families


def arm_name(base, suf):
    return f'({base.strip("()")}{suf})'


def m_all(Wt, r, p, s):
    """The test run's M1-M3 (boot.m123) and the high-HR hit rate."""
    out = boot.m123(Wt, r, p, s)
    h = s == 'high'
    out['hit_high'] = boot._wmean(Wt[:, h], (np.abs(p[h] - r[h]) <= GRID + 1e-6).astype(float)) if h.any() else np.full(len(Wt), np.nan)
    return out


def keep():
    return list(S3.KEEP) + ['hit_high']


def rows_of(base, pt, ci, metrics):
    return [dict(base, metric=k, point=pt.get(k, np.nan), ci_lo=ci.get(k, (np.nan, np.nan))[0], ci_hi=ci.get(k, (np.nan, np.nan))[1])
            for k in metrics]


# ---------------- jobs (run in worker processes) ----------------
def job_arm(a):
    base, g, b = a
    k = g.key.nunique()
    v = g.groupby(['window_id', 'layer']).key.nunique()
    w = set(v[v == k].index)
    g = g[[x in w for x in zip(g.window_id, g.layer)]]
    pt, ci = boot.point_ci(g, SEEDS, m_all, b)
    out = rows_of(dict(base, windows=len(w), subjects=g.subject_id.nunique()), pt, ci, keep())
    ptv, civ = boot.point_ci(S3.mark_verified(g), SEEDS, S3.m_verified, b)
    nv = len(set(map(tuple, g[S3.verified(g)][['window_id', 'layer']].to_numpy())))
    out += rows_of(dict(base, windows=nv, subjects=g.subject_id.nunique()), ptv, civ, ['MAE_verified'])
    return out


def job_contrast(a):
    base, ga, gc, b = a
    ga, gc, nw = S3.pair_windows(ga, gc)
    pt, ci = boot.contrast_ci(ga, gc, SEEDS, m_all, b)
    out = rows_of(dict(base, windows=nw, subjects=ga.subject_id.nunique()), pt, ci, keep())
    ptv, civ = boot.contrast_ci(S3.mark_verified(ga), S3.mark_verified(gc), SEEDS, S3.m_verified, b)
    nv = len(set(map(tuple, ga[S3.verified(ga)][['window_id', 'layer']].to_numpy())))
    out += rows_of(dict(base, windows=nv, subjects=ga.subject_id.nunique()), ptv, civ, ['MAE_verified'])
    return out


def four(parts):
    """Windows valid for every key of the four arms."""
    ks = [q.key.nunique() for q in parts]
    vs = [q.groupby(['window_id', 'layer']).key.nunique() for q in parts]
    w = set.intersection(*[set(v[v == k].index) for v, k in zip(vs, ks)])
    return [q[[t in w for t in zip(q.window_id, q.layer)]] for q in parts], len(w)


def diff4(parts, fn, b):
    subjects = boot.numeric_subject_order(set().union(*[set(q.subject_id) for q in parts]))
    smp = [boot.Sample(q, subjects, SEEDS) for q in parts]

    def f(Wt):
        v = [boot.per_seed(sm, fn, Wt) for sm in smp]
        return {sd: {k: (v[0][sd][k] - v[1][sd][k]) - (v[2][sd][k] - v[3][sd][k]) for k in v[0][sd]} for sd in SEEDS}
    return boot.summarize(f, subjects, SEEDS, b), len(subjects)


def job_interaction(a):
    base, parts, b = a
    parts, nw = four(parts)
    if not nw:
        return [dict(base, windows=0, subjects=0, metric='', point=np.nan, ci_lo=np.nan, ci_hi=np.nan)]
    (pt, ci), ns = diff4(parts, m_all, b)
    out = rows_of(dict(base, windows=nw, subjects=ns), pt, ci, keep())
    (ptv, civ), _ = diff4([S3.mark_verified(q) for q in parts], S3.m_verified, b)
    nv = len(set(map(tuple, parts[0][S3.verified(parts[0])][['window_id', 'layer']].to_numpy())))
    out += rows_of(dict(base, windows=nv, subjects=ns), ptv, civ, ['MAE_verified'])
    return out


def m7_tab(g):
    t = AN.m7_table(g)
    return t.assign(key=t.run_id, layer='HAR')


def job_m7_arm(a):
    """As families.fam_m7 of the test run (no pairing across seeds)."""
    base, g, b = a
    t = AN.m7_table(g)
    t = t[t.seed.isin(SEEDS)]
    pt, ci = boot.point_ci(t, SEEDS, boot.m7, b)
    return rows_of(dict(base, subjects=t.subject_id.nunique()), pt, ci, ['M7_MAE', 'M7_slope', 'M7_CCC'])


def job_m7_contrast(a):
    base, ga, gc, b = a
    ta, tc, n = S3.pair_windows(m7_tab(ga), m7_tab(gc))
    pt, ci = boot.contrast_ci(ta, tc, SEEDS, boot.m7, b)
    return rows_of(dict(base, subjects=n), pt, ci, ['M7_MAE', 'M7_slope', 'M7_CCC'])


def job_m7_interaction(a):
    base, parts, b = a
    parts, n = four([m7_tab(q) for q in parts])
    if not n:
        return [dict(base, subjects=0, metric='', point=np.nan, ci_lo=np.nan, ci_hi=np.nan)]
    (pt, ci), _ = diff4(parts, boot.m7, b)
    return rows_of(dict(base, subjects=n), pt, ci, ['M7_MAE', 'M7_slope', 'M7_CCC'])


def run_job(a):
    kind, payload = a
    return kind, globals()[f'job_{kind}'](payload)


def init_worker():
    setup()


# ---------------- commands ----------------
def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(8 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def cmd_check(part, workers, root):
    P = PARTS[part]
    out = root / part / 'check'
    res = []
    for ds in ('MMPD', 'MCD'):
        if (out / ds).exists():
            sys.exit(f'{out / ds} exists; move it aside first')
        cmd = ['nice', '-n', '16', 'taskset', '-c', '0-13', PY_MAIN, '-B', '-s', str(EVAL / 'stage3_check.py'), '--dataset', ds,
               '--stage1', str(P['root'] / 'stage1'), '--stage2', str(P['root'] / 'stage2_conf'), str(P['root'] / 'stage2_expl'),
               str(P['infer'][ds]), '--vendor', str(VENDOR), '--out', str(out), '--workers', str(workers)]
        print('step (b):', ' '.join(cmd[6:]), flush=True)
        e = subprocess.run(cmd).returncode
        if e:
            sys.exit(f'stage3_check {ds} exit {e}')
        old, new = P['root'] / 'check' / ds, out / ds
        so = pd.read_csv(old / 'support.csv', dtype=str)
        sn = pd.read_csv(new / 'support.csv', dtype=str)
        support_equal = so.equals(sn)
        ro = pd.read_csv(old / 'readouts.csv.gz')
        rn = pd.read_csv(new / 'readouts.csv.gz')
        m = ro.merge(rn, on=['key', 'window_id', 'layer'], how='left', suffixes=('_stored', '_new'), indicator=True)
        same_val = (m.predicted_HR_stored == m.predicted_HR_new) | (m.predicted_HR_stored.isna() & m.predicted_HR_new.isna())
        same = same_val & (m.valid_stored == m.valid_new)
        new_keys = sorted(set(rn.key) - set(ro.key))
        dn = pd.read_csv(new / 'defects.csv')
        r = dict(part=part, dataset=ds, stored_keys=ro.key.nunique(), stored_rows=len(ro), stored_rows_missing_now=int((m._merge != 'both').sum()),
                 stored_rows_equal=int(same.sum()), support_equal=bool(support_equal), support_windows=int((sn.in_support == 'True').sum()),
                 new_keys=len(new_keys), new_rows=int(rn.key.isin(new_keys).sum()), new_valid_rows=int(rn[rn.key.isin(new_keys)].valid.sum()),
                 defective_new_keys=int(dn.key.isin(new_keys).sum()))
        r['passed'] = r['support_equal'] and r['stored_rows_equal'] == r['stored_rows'] and r['stored_rows_missing_now'] == 0
        print(json.dumps(r), flush=True)
        res.append(r)
    pd.DataFrame(res).to_csv(out / 'comparison_with_stored.csv', index=False)
    sys.exit(0 if all(r['passed'] for r in res) else 1)


def stored_rows(part, ds):
    a = PARTS[part]['root'] / 'analysis' / ds / PARTS[part]['tag'][ds]
    conf = pd.read_csv(a / 'confirmatory.csv')
    sec = pd.read_csv(a / 'secondary.csv')
    arms = pd.read_csv(a / 'arms.csv')
    m7 = pd.read_csv(a / 'm7.csv') if (a / 'm7.csv').exists() else None
    return conf, sec, arms, m7


def cmd_analyze(part, b, workers, root, out_root):
    setup()
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    P = PARTS[part]
    chk = root / part / 'check'
    cmp_ = pd.read_csv(chk / 'comparison_with_stored.csv')
    if not cmp_.passed.all():
        sys.exit('step (b) comparison with the stored outputs did not pass; no analysis')
    outp = out_root / part
    if outp.exists():
        sys.exit(f'{outp} exists; move it aside first')
    allres, info = {}, {}
    for ds in ('MMPD', 'MCD'):
        keys = pd.concat([pd.read_csv(f / ds / 'checkpoints.csv') for f in (P['root'] / 'stage2_conf', P['root'] / 'stage2_expl', P['infer'][ds])],
                         ignore_index=True)
        keys = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
        d, bad = S3.load(ds, P['root'] / 'stage1' / ds, chk / ds, keys, 'automatic')
        prim = d[d.variant.isin(S3.PRIMARY) & d.layer.eq('HAR') & d.seed.isin(SEEDS)]
        main_pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
        pw = S3.restrict(prim[prim.population.eq(main_pop)])
        m7 = S3.restrict(prim[prim.population.eq('M7_both_states')]) if ds == 'MCD' else None
        info[ds] = dict(excluded_keys=sorted(bad), windows_primary=int(pw.drop_duplicates(['window_id', 'layer']).shape[0]))

        def sel(frame, setting, base):
            bb, suf = SETTINGS[setting]
            g = frame[frame.backbone.eq(bb) & frame.strategy.eq(arm_name(base, suf))]
            return g if len(g) and not set(SEEDS) - set(g.seed) else None
        jobs, absent = [], []
        for setting, (bb, suf) in SETTINGS.items():
            for base in BASES:
                g = sel(pw, setting, base)
                if g is None:
                    absent.append(dict(dataset=ds, setting=setting, arm=arm_name(base, suf), what='arm'))
                    continue
                jobs.append(('arm', (dict(dataset=ds, setting=setting, backbone=bb, arm=arm_name(base, suf), base_arm=base), g, b)))
                if m7 is not None and sel(m7, setting, base) is not None:
                    jobs.append(('m7_arm', (dict(dataset=ds, setting=setting, backbone=bb, arm=arm_name(base, suf), base_arm=base),
                                            sel(m7, setting, base), b)))
            for name, a, c in PAIRS:
                ga, gc = sel(pw, setting, a), sel(pw, setting, c)
                base = dict(dataset=ds, setting=setting, backbone=bb, name=name, base_contrast=f'{a}-{c}', contrast=f'{arm_name(a, suf)}-{arm_name(c, suf)}')
                if ga is None or gc is None:
                    absent.append(dict(dataset=ds, setting=setting, arm=base['contrast'], what='contrast'))
                    continue
                jobs.append(('contrast', (base, ga, gc, b)))
                if m7 is not None and sel(m7, setting, a) is not None and sel(m7, setting, c) is not None:
                    jobs.append(('m7_contrast', (base, sel(m7, setting, a), sel(m7, setting, c), b)))
        for kind, x, y in INTERACTIONS:
            for name, a, c in PAIRS:
                parts = [sel(pw, x, a), sel(pw, x, c), sel(pw, y, a), sel(pw, y, c)]
                base = dict(dataset=ds, interaction=kind, setting=x, reference_setting=y, backbone=SETTINGS[x][0], name=name, base_contrast=f'{a}-{c}')
                if any(q is None for q in parts):
                    absent.append(dict(dataset=ds, setting=f'{x} - {y}', arm=base['base_contrast'], what='interaction'))
                    continue
                jobs.append(('interaction', (base, parts, b)))
                if m7 is not None and all(sel(m7, s_, t_) is not None for s_ in (x, y) for t_ in (a, c)):
                    jobs.append(('m7_interaction', (base, [sel(m7, x, a), sel(m7, x, c), sel(m7, y, a), sel(m7, y, c)], b)))
        res = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as ex:
            for n, (kind, rows) in enumerate(ex.map(run_job, jobs), 1):
                res.setdefault(kind, []).extend(rows)
                if n % 10 == 0 or n == len(jobs):
                    print(f'{part} {ds} job {n}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
        res['absent'] = absent
        allres[ds] = res
    # ---------------- reproduction of the stored confirmatory rows; reported values ----------------
    outp.mkdir(parents=True)
    repro = []
    for ds, res in allres.items():
        conf, sec, arms_s, m7_s = stored_rows(part, ds)
        main_pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
        st = pd.concat([conf, sec], ignore_index=True)
        st = st[st.analysis.eq('primary') & st.layer.eq('HAR') & st.population.eq(main_pop)]
        new = pd.DataFrame(res['contrast'])
        for r in new[new.setting.isin(CONFIRMATORY) & new.metric.ne('hit_high')].itertuples():
            s = st[st.backbone.eq(r.backbone) & st.contrast.eq(r.contrast) & st.metric.eq(r.metric)]
            if not len(s):
                repro.append(dict(dataset=ds, kind='contrast', setting=r.setting, item=r.contrast, metric=r.metric, stored=False))
                continue
            s = s.iloc[0]
            repro.append(dict(dataset=ds, kind='contrast', setting=r.setting, item=r.contrast, metric=r.metric, stored=True,
                              windows_equal=int(s.windows) == int(r.windows), max_abs_difference=float(np.nanmax(np.abs(
                                  np.array([s.point - r.point, s.ci_lo - r.ci_lo, s.ci_hi - r.ci_hi], float))) if np.isfinite([s.point, r.point]).all() else np.nan)))
        newa = pd.DataFrame(res['arm'])
        sa = arms_s[arms_s.analysis.eq('primary') & arms_s.layer.eq('HAR') & arms_s.population.eq(main_pop)]
        for r in newa[newa.setting.isin(CONFIRMATORY)].itertuples():
            s = sa[sa.backbone.eq(r.backbone) & sa.strategy.eq(r.arm) & sa.metric.eq(r.metric)]
            if not len(s):
                continue
            s = s.iloc[0]
            repro.append(dict(dataset=ds, kind='arm', setting=r.setting, item=r.arm, metric=r.metric, stored=True,
                              windows_equal=int(s.windows) == int(r.windows),
                              max_abs_difference=float(np.nanmax(np.abs(np.array([s.point - r.point, s.ci_lo - r.ci_lo, s.ci_hi - r.ci_hi], float))))))
        if m7_s is not None and 'm7_arm' in res:
            n7 = pd.DataFrame(res['m7_arm'])
            s7 = m7_s[m7_s.layer.eq('HAR') & m7_s.seeds.astype(str).eq('3')]
            for r in n7[n7.setting.isin(CONFIRMATORY)].itertuples():
                s = s7[s7.backbone.eq(r.backbone) & s7.strategy.eq(r.arm) & s7.metric.eq(r.metric)]
                if not len(s):
                    continue
                s = s.iloc[0]
                repro.append(dict(dataset=ds, kind='m7_arm', setting=r.setting, item=r.arm, metric=r.metric, stored=True,
                                  windows_equal=int(s.subjects) == int(r.subjects),
                                  max_abs_difference=float(np.nanmax(np.abs(np.array([s.point - r.point, s.ci_lo - r.ci_lo, s.ci_hi - r.ci_hi], float))))))
        dsd = outp / ds
        dsd.mkdir()
        for kind, fn in (('arm', 'arms'), ('contrast', 'contrasts'), ('interaction', 'interactions'), ('m7_arm', 'm7_arms'),
                         ('m7_contrast', 'm7_contrasts'), ('m7_interaction', 'm7_interactions'), ('absent', 'absent')):
            if res.get(kind):
                df = pd.DataFrame(res[kind])
                if kind in ('arm', 'contrast', 'm7_arm', 'm7_contrast'):
                    df['source'] = np.where(df.setting.isin(CONFIRMATORY), 'recomputed (reported values: stored stage (c))', '7.04')
                df.to_csv(dsd / f'{fn}.csv', index=False)
    rp = pd.DataFrame(repro)
    rp.to_csv(outp / 'reproduction.csv', index=False)
    # ---------------- statements (test part) ----------------
    st_rows = []
    if part == 'test':
        st_rows = statements(allres)
        pd.DataFrame(st_rows).to_csv(outp / 'statements.csv', index=False)
    names = sorted(p.relative_to(outp).as_posix() for p in outp.rglob('*.csv'))
    (outp / 'analysis_hashes.txt').write_text(''.join(f'{file_sha(outp / n)}  {n}\n' for n in names))
    det = {}
    for f in sorted((W / 'outputs' / '704' / 'infer').glob('determinism_*.csv')):
        x = pd.read_csv(f)
        det[f.stem] = dict(files=int(x.files.sum()), identical=int(x.identical.sum()))
    run = dict(step='7.04 analysis', part=part, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(),
               b=b, seeds=SEEDS, datasets=info, reproduction=dict(rows=len(rp), all_windows_equal=bool(rp.windows_equal.all()) if len(rp) else None,
                                                                  max_abs_difference=float(rp.max_abs_difference.max()) if len(rp) else None),
               determinism=det, statements=len(st_rows))
    (outp / 'run.json').write_text(json.dumps(run, indent=1, default=str))
    print(json.dumps(run, indent=1, default=str))


def statements(allres):
    out = []
    for ds, res in allres.items():
        setname = PARTS['test']['name'][ds]
        conf, sec, _, _ = stored_rows('test', ds)
        main_pop = 'Qstar' if ds == 'MMPD' else 'draft_rule'
        stored = pd.concat([conf, sec], ignore_index=True)
        stored = stored[stored.analysis.eq('primary') & stored.layer.eq('HAR') & stored.population.eq(main_pop) & stored.metric.eq('MAE_high')]
        c = pd.DataFrame(res['contrast'])
        c = c[c.metric.eq('MAE_high')]
        it = pd.DataFrame(res['interaction'])
        it = it[it.metric.eq('MAE_high')]

        def get(setting, name):
            if setting in CONFIRMATORY:
                bb, _ = SETTINGS[setting]
                a, cc = [(x, y) for n, x, y in PAIRS if n == name][0]
                s = stored[stored.backbone.eq(bb) & stored.contrast.eq(f'{a}-{cc}')]
                return (s.iloc[0].point, s.iloc[0].ci_lo, s.iloc[0].ci_hi, 'stored stage (c)') if len(s) else None
            s = c[c.setting.eq(setting) & c.name.eq(name)]
            return (s.iloc[0].point, s.iloc[0].ci_lo, s.iloc[0].ci_hi, '7.04') if len(s) else None

        def base(item, setting, name, v):
            return dict(dataset=ds, test_set=setname, item=item, setting=setting, contrast=name, point=v[0], ci_lo=v[1], ci_hi=v[2], source=v[3])
        v = get('PhysNet-Std', 'C1')
        if v:
            s = (f'With standardized labels, aligned editing lowered the high-HR error of PhysNet against no augmentation on {setname}.'
                 if v[2] < 0 else f'With standardized labels, no gain of aligned editing over no augmentation was established for PhysNet on {setname}.')
            out.append(dict(base('8(a)', 'PhysNet-Std', 'C1', v), statement=s))
        for x, y in (('PhysNet-MCC', 'PhysNet'), ('PhysNet-Std-MCC', 'PhysNet-Std'), ('FactorizePhys-MCC', 'FactorizePhys')):
            bbname = 'PhysNet' if x.startswith('PhysNet') else 'FactorizePhys'
            for name, what in (('C2', 'editing'), ('C3', 'time resampling')):
                v, ref = get(x, name), get(y, name)
                ii = it[it.setting.eq(x) & it.reference_setting.eq(y) & it.name.eq(name)]
                if not v or ref is None or not len(ii):
                    continue
                lo_int = ii.iloc[0].ci_lo

                def say(below):
                    if v[2] < 0:
                        return f'With the MCC loss, correcting the alignment of {what} still lowered the high-HR error of {bbname} on {setname}.', 'still lowered'
                    if v[1] > 0:
                        return f'With the MCC loss, the corrected {what} had the higher high-HR error for {bbname} on {setname}.', 'higher'
                    if below and lo_int > 0:
                        return f'With the MCC loss, the cost of misaligned {what} was absorbed for {bbname} on {setname}.', 'absorbed'
                    return f'With the MCC loss, no effect of correcting the alignment of {what} was established for {bbname} on {setname}.', 'no effect'
                s, kind = say(ref[2] < 0)
                s_pt, kind_pt = say(ref[0] < 0)
                out.append(dict(base('8(b)', x, name, v), statement=s, kind=kind, pearson_setting=y, pearson_point=ref[0], pearson_ci_lo=ref[1],
                                pearson_ci_hi=ref[2], pearson_source=ref[3], loss_interaction_point=ii.iloc[0].point, loss_interaction_ci_lo=lo_int,
                                loss_interaction_ci_hi=ii.iloc[0].ci_hi, kind_if_point_reading=kind_pt, readings_differ=kind != kind_pt))
        for x in ('PhysNet-Std', 'PhysNet-MCC', 'PhysNet-Std-MCC', 'FactorizePhys-MCC'):
            v = get(x, 'ii-pc_vs_i-simple')
            if not v:
                continue
            s = ('editing had a lower high-HR error than time resampling without a label shift' if v[2] < 0 else
                 'time resampling without a label shift had the lower high-HR error' if v[1] > 0 else 'no difference between the two was established')
            out.append(dict(base('8(c)', x, '(ii-pc)-(i-simple)', v), statement=s))
    absorbed = [r for r in out if r['item'] == '8(b)' and r['kind'] == 'absorbed']
    absorbed_pt = [r for r in out if r['item'] == '8(b)' and r['kind_if_point_reading'] == 'absorbed']
    out.append(dict(item='9', kind='required' if absorbed else 'not required',
                    statement=f"conditional RhythmFormer MCC runs: {'required' if absorbed else 'not required'} "
                              f"({len(absorbed)} 'absorbed' statements under 8(b), interval reading of 7.07i item 3)",
                    reference_point_reading_absorbed=len(absorbed_pt)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['check', 'analyze'])
    ap.add_argument('part', choices=list(PARTS))
    ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check-root', default=str(W / 'cache' / 'train704' / 'analysis'), help='step-(b) outputs (<root>/<PART>/check)')
    ap.add_argument('--out-root', default=str(W / 'outputs' / '704' / 'analysis'), help='analysis outputs (<root>/<PART>); trial runs elsewhere')
    a = ap.parse_args()
    if a.command == 'check':
        cmd_check(a.part, a.workers, Path(a.check_root))
    else:
        cmd_analyze(a.part, a.b, a.workers, Path(a.check_root), Path(a.out_root))


if __name__ == '__main__':
    main()
