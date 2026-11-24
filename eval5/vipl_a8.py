"""A8 (entry of 2026-11-23 written after stage (a), commit efa380c): sensitivity analysis on VIPL-HR, computed after stage
(c2) by separate code. The A7 stage code is imported unchanged (eval5/vipl.py, eval5/vipl_c.py, eval5/stage3_analyze.py,
eval5/stage3_check.py); nothing of it is modified.

Reference: the BVP read at 60 Hz, as in the first comparison of A7 item 6: eval5/vipl.windows_for with fs = 60 (BVP placed
on the 30-Hz axis from the first frame, the readout used for the test sets, the verification of MMPD: no signal defect and
the three model-independent estimates agree). Verified windows with a reference >= 100 bpm are high-HR, < 100 bpm low-HR.
Windows: the stage (a) windows of the primary set (source1, source3), less those that do not lie within the BVP record at
60 Hz (not returned by windows_for at 60 Hz) and those excluded by the 200-ms rule of A7 item 4 (gap200).
Checkpoints and predictions: the 45 confirmatory (primary) keys of stage (b1); their readouts are those of stage (c1)
(readouts.csv.gz: the same per-window readout), the window support that of (c1) (windows valid for every primary key).
Measures and contrasts of the primary analysis of (c1), with intervals and without tiers: C1-C4 (stage3_analyze.contrast_rows),
arms (arm_rows), windows >= 120 bpm, M4 bins (families.fam_m4), within-subject change (vipl_c.within_subject).
Machine-specific locations are arguments: --zip-dir (folder of the 14 original zips), --vipl-out (output root of the
A7 stages: stage_a, stage_b_confirmatory, stage_c1), --vendor (archived source tree), --work (server-only folder for the
window rows), --out (folder of the aggregate outputs).
Commands:
  refs [--limit N]   (environment of stage (a)) 60-Hz references of the stage (a) windows; reads time.txt, wave.csv and
                     gt_HR.csv from the zips (frame counts from stage (a) recordings.csv; no video decoding) and no
                     prediction. Writes the window rows to <work>/references_60hz.csv (server only, window level) and the
                     counts to <out>/counts.csv.
  analyze            (environment of stage (c)) after the (c2) hashes are committed: the analysis above, <out>/*.csv,
                     analysis_hashes.txt.
This file is the published form of the code that produced the reported A8 outputs: the machine paths, which were
constants there, are arguments here; the computation is unchanged (outputs reproduced byte for byte, except the start
and end times in counts.csv and analysis_windows.csv).
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VENDOR = ZIPS = SA = C1 = B1 = SRV = OUT = None
FS = 60.0
G = {}


def configure(a):
    global VENDOR, ZIPS, SA, C1, B1, SRV, OUT
    VENDOR, ZIPS = str(Path(a.vendor).resolve()), str(Path(a.zip_dir).resolve())
    v7 = Path(a.vipl_out).resolve()
    SA, C1, B1 = v7 / 'stage_a' / 'VIPL', v7 / 'stage_c1' / 'VIPL', v7 / 'stage_b_confirmatory' / 'VIPL'
    SRV, OUT = Path(a.work).resolve(), Path(a.out).resolve()


def init(vendor):
    import vipl
    vipl.init_worker(vendor)
    from src.st11x.future import frozen_reference_functions
    src = Path(vendor) / 'src/st3/c4.py'  # the defect map exactly as stage (a) (vipl.run_a)
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'defect_map')
    ctx = dict(np=np, pd=pd, write_rows=lambda *a, **k: None, update=lambda *a, **k: None)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ctx)
    G.update(vipl=vipl, helpers=frozen_reference_functions(), defect_map=ctx['defect_map'])


def job(args):
    rec, row = args
    vipl = G['vipl']
    t, reason = vipl.frame_times(rec, int(row['decoded_frames']))
    assert t is not None, (rec['recording_id'], reason)
    wave = vipl.column(vipl.read(rec, 'wave.csv'), 'Wave')
    hr = vipl.column(vipl.read(rec, 'gt_HR.csv'), 'HR') if 'gt_HR.csv' in rec['members'] else None
    excl, rows = vipl.windows_for(row, dict(t=t, wave=wave, hr=hr), FS, G['helpers'], G['defect_map'])
    return rec['recording_id'], excl, rows


def cmd_refs(limit, workers):
    import vipl
    sa_recs = pd.read_csv(SA / 'recordings.csv', dtype={'subject_id': str})
    sa_win = pd.read_csv(SA / 'references.csv', dtype={'subject_id': str})
    prim = sa_win[sa_win.set.eq('primary')]
    # every recording that stage (a) passed to windows_for (vipl.a_job returned data): the 60-Hz comparison of A7 item 6
    # (bvp_rate.json, first row) is reproduced from them as a check of this path; A8 uses the primary-set windows only
    first = ('video not decodable', 'no time.txt', 'time.txt not one line per frame', 'frame times not strictly increasing', 'no BVP record')
    use = set(sa_recs[~sa_recs.exclusion.fillna('').isin(first)].recording_id)
    by = {r['recording_id']: r for r in sa_recs.to_dict('records')}
    recs = [r for r in vipl.index(ZIPS) if r['recording_id'] in use]
    assert len(recs) == len(use), (len(recs), len(use))
    if limit:
        recs = recs[:limit]
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    rows, excl = [], {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(VENDOR,)) as ex:
        for n, (rid, e, rr) in enumerate(ex.map(job, [(r, by[r['recording_id']]) for r in recs]), 1):
            excl[rid] = e
            rows += rr
            if n % 100 == 0 or n == len(recs):
                print(f'A8 refs {n}/{len(recs)}', flush=True)
    at60 = pd.DataFrame(rows)
    v60 = at60[at60.verified.astype(bool) & np.isfinite(at60.gt_HR_mean) & (at60.gt_HR_mean > 0)]
    r60, n60 = float(np.median(v60.h_FFT_HAR / v60.gt_HR_mean)), len(v60)
    stored = json.loads((SA / 'bvp_rate.json').read_text())['comparisons'][0]
    at60 = at60.rename(columns={'h_FFT_HAR': 'reference_HR_60Hz', 'verified': 'verified_60Hz'})
    keep = ['window_id', 'reference_HR_60Hz', 'h_FFT_L0', 'reference_class', 'verified_60Hz', 'defect_any']
    sub = prim[prim.recording_id.isin([r['recording_id'] for r in recs])]
    m = sub.merge(at60[keep].rename(columns={'h_FFT_L0': 'h_FFT_L0_60Hz', 'reference_class': 'reference_class_60Hz', 'defect_any': 'defect_any_60Hz'}),
                  on='window_id', how='left', indicator=True)
    m['within_60Hz_record'] = m._merge.eq('both')
    m['excluded_200ms'] = m.gap200.astype(bool)
    m['A8_window'] = m.within_60Hz_record & ~m.excluded_200ms
    v = m.A8_window & m.verified_60Hz.fillna(False).astype(bool)
    m['A8_verified'] = v
    m['A8_high'] = v & (m.reference_HR_60Hz >= 100)
    m['A8_low'] = v & (m.reference_HR_60Hz < 100)
    SRV.mkdir(parents=True, exist_ok=True)
    m.drop(columns=['_merge']).to_csv(SRV / ('references_60hz.csv' if not limit else f'references_60hz_limit{limit}.csv'), index=False)
    a = m[~m.excluded_200ms]  # the windows of the primary analysis before the 60-Hz rule
    p = sub[~sub.gap200.astype(bool)]
    counts = dict(stage_a_primary_windows=len(sub), stage_a_primary_subjects=sub.subject_id.nunique(),
                  excluded_200ms=int(m.excluded_200ms.sum()),
                  excluded_not_within_60Hz_record=int((~m.excluded_200ms & ~m.within_60Hz_record).sum()),
                  excluded_both=int((m.excluded_200ms & ~m.within_60Hz_record).sum()),
                  A8_windows=int(m.A8_window.sum()), A8_recordings=int(m[m.A8_window].recording_id.nunique()),
                  A8_verified=int(v.sum()), A8_high_windows=int(m.A8_high.sum()), A8_high_subjects=int(m[m.A8_high].subject_id.nunique()),
                  A8_low_windows=int(m.A8_low.sum()), A8_low_subjects=int(m[m.A8_low].subject_id.nunique()),
                  primary_analysis_verified=int(p.verified.astype(bool).sum()), primary_analysis_high_windows=int(p.high.astype(bool).sum()),
                  primary_analysis_high_subjects=int(p[p.high.astype(bool)].subject_id.nunique()),
                  recordings_read=len(recs), recordings_excluded_at_60Hz=sum(bool(e) for e in excl.values()),
                  check_60Hz_median_ratio=r60, check_60Hz_windows=n60, stage_a_60Hz_median_ratio=stored['median_ratio'],
                  stage_a_60Hz_windows=stored['windows'], check_equal=bool(r60 == stored['median_ratio'] and n60 == stored['windows']),
                  started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), limit=limit or '')
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([counts]).to_csv(OUT / ('counts.csv' if not limit else f'counts_limit{limit}.csv'), index=False)
    print(json.dumps(counts, indent=1), flush=True)


def cmd_analyze(b, workers):
    import stage3_check as SC
    SC.init(VENDOR)
    import stage3_analyze as S3
    import vipl_c
    assert (SRV / 'references_60hz.csv').exists(), 'run refs first'
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    m = pd.read_csv(SRV / 'references_60hz.csv', dtype={'subject_id': str})
    a8 = m[m.A8_verified.astype(bool)].copy()
    pops = a8.assign(layer='HAR', reference_HR=a8.reference_HR_60Hz, population='A8_60Hz',
                     set=np.where(a8.reference_HR_60Hz >= 100, 'high', 'low'))
    pops = pops[['window_id', 'layer', 'recording_id', 'subject_id', 'scene', 'source', 'reference_HR', 'population', 'set']]
    rd = pd.read_csv(C1 / 'readouts.csv.gz')
    keys = pd.read_csv(B1 / 'checkpoints.csv')
    kk = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    bad = set(pd.read_csv(C1 / 'defects.csv').key.dropna())
    sup = pd.read_csv(C1 / 'support.csv')
    support = set(zip(sup.window_id, sup.layer))
    d = rd.merge(kk, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(x, y) in support for x, y in zip(d.window_id, d.layer)]
    d['dataset'] = 'VIPL-HR'
    d['backbone'] = d.backbone.map(S3.NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    conf_d = d[d.variant.isin(S3.PRIMARY) & d.arm.isin(vipl_c.CONF_ARMS) & d.seed.isin(vipl_c.CONF_SEEDS)]
    assert conf_d.key.nunique() == 45, conf_d.key.nunique()
    conf = [(n, x, c, 'values only') for n, x, c in S3.CONF]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = S3.contrast_rows(conf_d, conf, b, 'A8_60Hz')
    for r in rows:
        r['grade'] = ''
    pd.DataFrame(rows).to_csv(OUT / 'contrasts.csv', index=False)
    pd.DataFrame(S3.arm_rows(conf_d, b, 'A8_60Hz')).to_csv(OUT / 'arms.csv', index=False)
    g120 = conf_d[conf_d.reference_HR >= 120].copy()
    g = S3.arm_rows(g120, b, 'ge120') + S3.contrast_rows(g120, conf, b, 'ge120')
    for r in g:
        r['grade'] = ''
    pd.DataFrame(g).to_csv(OUT / 'ge120.csv', index=False)
    pd.DataFrame(S3.AN.fam_m4(S3.restrict(conf_d), [])).to_csv(OUT / 'm4_bins.csv', index=False)
    pd.DataFrame(vipl_c.within_subject(S3.restrict(conf_d), b)).to_csv(OUT / 'within_subject.csv', index=False)
    w = pops[[(x, 'HAR') in support for x in pops.window_id]]
    info = dict(windows_in_support=len(w), high_windows=int(w.set.eq('high').sum()), high_subjects=int(w[w.set.eq('high')].subject_id.nunique()),
                low_windows=int(w.set.eq('low').sum()), subjects=int(w.subject_id.nunique()), keys=int(conf_d.key.nunique()), b=b,
                started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    pd.DataFrame([info]).to_csv(OUT / 'analysis_windows.csv', index=False)
    names = sorted(p.name for p in OUT.iterdir() if p.suffix == '.csv')
    (OUT / 'analysis_hashes.txt').write_text(''.join(f'{SC.S2.file_sha(OUT / n)}  A8/{n}\n' for n in names))
    print(json.dumps(info, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('command', choices=['refs', 'analyze'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--b', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=4)
    for k in ('--zip-dir', '--vipl-out', '--vendor', '--work', '--out'):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    configure(a)
    if a.command == 'refs':
        cmd_refs(a.limit, a.workers)
    else:
        cmd_analyze(a.b, a.workers)


if __name__ == '__main__':
    main()
