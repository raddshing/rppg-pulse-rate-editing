"""Step (b) of the Stage-5 evaluation, between stage 2 and the analysis (ACCESS_OPS2_BRIEF_02 6.48 마 3-4, 6.46 가 4, 8):
hashes, validity (automatic defect marks, window support) and the comparison of the original arms' MMPD HR readouts with the
stored October MMPD predictions. Nothing else is computed here (line fractions, background-bin shares etc. are step (c)).

Inputs: stage-1 folders (references, populations), stage-2 folders (one or more; e.g. confirmatory, exploratory), the
confirmatory manifests. Per dataset:
  1. hashes: every file listed in stage1_hashes.txt / stage2_hashes.txt is re-hashed and must match (a mismatch stops).
  2. HR readout of every output key on every window of the populations (the development readouts b1/mmpd_a.py (waveform cast
     to float64) and b1/mcd_windows.py (float32 as stored), i.e. grid._prepare + 512-point
     periodogram peak; wide band 0.6-3.3 Hz (HAR) and narrow band 0.75-2.5 Hz (L0); MCD-rPPG windows on the uniform 30 Hz grid
     of the endpoint-linear frame times, complete_windows). A window is valid for a key if the prediction covers it and the
     readout is finite.
  3. automatic defects (until this step only): manifest hash mismatch, load failure (stage-2 checkpoints.csv), and an output
     that is not valid on more than half of the dataset's population windows. A defective key is excluded from the analyses
     that use it, with the reason; nothing is retrained. --exclude KEY=REASON adds a recorded exclusion (defective-checkpoint
     path, 6.46 가 8).
  4. window support: confirmatory = windows valid for every non-defective confirmatory primary key (PhysNet/RhythmFormer 'std',
     FactorizePhys 'fsam-on'; the 45 manifest checkpoints); exploratory keys and sensitivity variants (FSAM off, batch 1,
     blur) = the confirmatory support intersected with the key's own valid windows (6.48 마 2). Removed counts are written.
  5. MMPD: HR readout of the original arms (0), (i), (ii) against the stored October predictions (--october-mmpd: CSV with
     backbone, arm, seed, window_id, layer, predicted_HR).
     MCD-rPPG (brief 02 6.67 다 5, technical): the same comparison against the October predictions of the MCD-rPPG half
     (--october-mcd, same CSV columns). Both datasets: every window whose two readouts differ (unequal finite values, or finite
     on one side only) is listed in october_differences.csv; tools/rerun_differing.py re-runs the guarded inference of the MCD
     recordings concerned and compares its predictions byte for byte with these stage-2 predictions.
     1% rule (6.70 다, both datasets): if, for any checkpoint, the windows with a different readout are more than 1% of
     that checkpoint's compared windows, october_stop.json is written and the check ends with exit code 3 after writing all
     its outputs; step (c) does not start until the cause has been found and reported (eval5/drive/step.sh: a release
     note october_stop_release.txt beside it, written by OPS2 only after the cause report has been sent and the command has
     instructed to proceed, with the instruction number, its time and a summary of the cause; 6.71 나).
Outputs (--out/<dataset>/): readouts.csv.gz (key, window_id, layer, predicted_HR, valid), defects.csv, support.csv (per
population, layer, set: windows and the confirmatory support), support_removed.csv (per key: windows removed),
october_comparison.csv and october_differences.csv (with --october-mmpd / --october-mcd), check_hashes.txt, run.json.
"""
import os  # noqa: E402
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse  # noqa: E402
import getpass  # noqa: E402
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
import stage2_predict as S2  # noqa: E402

BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}
PRIMARY = ('std', 'fsam-on')
VENDOR = None


def vendor_imports(vendor):
    global VENDOR
    VENDOR = Path(vendor).resolve()
    os.environ.setdefault('ST9_NAMESPACE', 'st9g')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


def verify_hashes(folder, name):
    """Every 'SHA  relative' line of the hash file must match the file (relative to the folder's parent)."""
    bad = []
    n = 0
    for line in (folder / name).read_text().splitlines():
        sha, rel = line.split('  ', 1)
        n += 1
        if S2.file_sha(folder.parent / rel) != sha:
            bad.append(rel)
    if bad:
        sys.exit(f'HASH MISMATCH in {folder / name}: {len(bad)} files, e.g. {bad[:3]}')
    return n


def read_hr(pred, diff, layer):
    from src.eval import grid
    band = BANDS[layer]
    return float(grid._estimate(grid._prepare(pred, diff, band, 'default', 30), 'periodogram', band, 30))


def mmpd_job(args):
    keydir, key, diff, windows = args
    vendor_imports(VENDOR_ARG)
    out = []
    cache = {}
    for rid, j, wid in windows:
        if rid not in cache:
            p = keydir / (rid.replace('/', '__') + '.npy')
            cache = {rid: np.load(p).astype(np.float64) if p.exists() else None}  # float64 (b1/mmpd_a.py casts the waveform)
        pred = cache[rid]
        for layer in BANDS:
            if pred is None or 300 * (j + 1) > len(pred):
                out.append((key, wid, layer, np.nan, False))
                continue
            h = read_hr(pred[300 * j:300 * (j + 1)], diff, layer)
            out.append((key, wid, layer, h, bool(np.isfinite(h))))
    return out


def mcd_job(args):
    keydir, key, diff, recs = args
    vendor_imports(VENDOR_ARG)
    from src.st11x.pathways import uniform_frame_grid, complete_windows
    out = []
    for rid, n, span, wins in recs:
        p = keydir / (rid + '.npy')
        pred = np.load(p) if p.exists() else None  # float32 as stored (b1/mcd_windows.py reads the waveform without a cast)
        frame_grid = uniform_frame_grid(np.arange(n, dtype=np.float64) * span / (n - 1))
        for wid, layer, start_s in wins:
            bd = complete_windows(frame_grid, [start_s])[0]
            s = bd['grid_start']
            if pred is None or not bd['available'] or s + 300 > len(pred):
                out.append((key, wid, layer, np.nan, False))
                continue
            h = read_hr(pred[s:s + 300], diff, layer)
            out.append((key, wid, layer, h, bool(np.isfinite(h))))
    return out


VENDOR_ARG = None


def init(vendor):
    global VENDOR_ARG
    VENDOR_ARG = vendor
    vendor_imports(vendor)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=['MMPD', 'MCD'])
    ap.add_argument('--stage1', required=True, help='stage-1 output folder (containing <dataset>/)')
    ap.add_argument('--stage2', nargs='+', required=True, help='stage-2 output folders (containing <dataset>/)')
    ap.add_argument('--october-mmpd', help='MMPD: stored October predictions of the original arms (CSV)')
    ap.add_argument('--october-mcd', help='MCD-rPPG: October predictions of the original arms (CSV, same columns; 6.67 다 5)')
    ap.add_argument('--exclude', nargs='*', default=[], help='KEY=REASON (recorded exclusion)')
    ap.add_argument('--vendor', required=True), ap.add_argument('--out', required=True), ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    init(str(Path(args.vendor).resolve()))
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    s1 = Path(args.stage1) / args.dataset
    nh = {'stage1': verify_hashes(s1, 'stage1_hashes.txt')}
    keys, recs2 = [], {}
    for d in args.stage2:
        s2 = Path(d) / args.dataset
        nh[str(Path(d).name)] = verify_hashes(s2, 'stage2_hashes.txt')
        ck = pd.read_csv(s2 / 'checkpoints.csv')
        ck['folder'] = str(s2)
        keys.append(ck)
        recs2[str(s2)] = pd.read_csv(s2 / 'recordings.csv').set_index('recording_id')
    ck = pd.concat(keys, ignore_index=True)
    assert ck.key.is_unique, 'an output key appears in two stage-2 folders'
    pops = pd.read_csv(s1 / 'populations.csv', dtype={'subject_id': str})
    jobs = []
    for r in ck[ck.status.eq('used')].itertuples():
        kd = Path(r.folder) / 'pred' / r.key
        diff = r.view == 'PhysNet' and '-std' not in r.arm
        if args.dataset == 'MMPD':
            w = pops.drop_duplicates('window_id')
            jobs.append((kd, r.key, diff, [(x.recording_id, int(x.window_index), x.window_id) for x in w.itertuples()]))
        else:
            refs = pd.read_csv(s1 / 'references.csv', dtype={'subject_id': str})
            refs = refs[refs.window_id.isin(set(pops.window_id))]
            rr = recs2[r.folder]
            items = []
            for rid, g in refs.groupby('recording_id'):
                items.append((rid, int(rr.loc[rid, 'decoded_frames']), float(rr.loc[rid, 'span_seconds']),
                              [(x.window_id, x.layer, float(x.start_seconds)) for x in g.itertuples()]))
            jobs.append((kd, r.key, diff, items))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(VENDOR_ARG,)) as ex:
        for n, res in enumerate(ex.map(mmpd_job if args.dataset == 'MMPD' else mcd_job, jobs), 1):
            rows += res
            if n % 10 == 0 or n == len(jobs):
                print(f'readout {n}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    rd = pd.DataFrame(rows, columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])
    if args.dataset == 'MCD':  # MCD windows are per layer in stage 1: keep only the layer of the reference row
        lay = pops[['window_id', 'layer']].drop_duplicates()
        rd = rd.merge(lay, on=['window_id', 'layer'])
    rd.to_csv(out / 'readouts.csv.gz', index=False)
    # defects
    defects = [dict(key=r.key, status='defective', reason=r.reason) for r in ck[ck.status.ne('used')].itertuples()]
    popw = pops[['window_id', 'layer']].drop_duplicates()
    v = rd.merge(popw, on=['window_id', 'layer'])
    frac = v.groupby('key').valid.mean()
    for k, f in frac.items():
        if f < 0.5:
            defects.append(dict(key=k, status='defective', reason=f'valid on {f:.3f} of the population windows (< half)'))
    for e in args.exclude:
        k, reason = e.split('=', 1)
        defects.append(dict(key=k, status='excluded', reason=reason))
    bad = {d['key'] for d in defects}
    S2.write_csv(out / 'defects.csv', defects or [dict(key='', status='', reason='')])
    # support
    conf = ck[ck.set.eq('confirmatory') & ck.variant.isin(PRIMARY) & ~ck.key.isin(bad)].key.tolist()
    ok = v[v.valid & v.key.isin(conf)].groupby(['window_id', 'layer']).key.nunique()
    support = set(ok[ok == len(conf)].index)
    pw = pops.copy()
    pw['in_support'] = [(a, b) in support for a, b in zip(pw.window_id, pw.layer)]
    pw.to_csv(out / 'support.csv', index=False)
    removed = []
    for k in sorted(set(ck.key) - bad):
        own = set(map(tuple, v[v.valid & v.key.eq(k)][['window_id', 'layer']].to_numpy()))
        removed.append(dict(key=k, set=ck.set[ck.key.eq(k)].iloc[0], variant=ck.variant[ck.key.eq(k)].iloc[0],
                            population_windows=len(popw), support_windows=len(support), own_valid=len(own & set(map(tuple, popw.to_numpy()))),
                            support_and_own=len(support & own), removed_from_support=len(support - own)))
    S2.write_csv(out / 'support_removed.csv', removed)
    # October MMPD comparison
    comp, diffs = [], []
    october = args.october_mmpd if args.dataset == 'MMPD' else args.october_mcd
    if october:
        o = pd.read_csv(october)
        for r in ck[ck.arm.isin(['(0)', '(i)', '(ii)']) & ck.variant.isin(PRIMARY)].itertuples():
            a = o[o.backbone.eq(r.backbone) & o.arm.eq(r.arm) & o.seed.eq(r.seed)][['window_id', 'layer', 'predicted_HR']]
            b = rd[rd.key.eq(r.key)][['window_id', 'layer', 'predicted_HR']]
            m = a.merge(b, on=['window_id', 'layer'], suffixes=('_october', '_new'))
            both = np.isfinite(m.predicted_HR_october) & np.isfinite(m.predicted_HR_new)
            comp.append(dict(key=r.key, backbone=r.backbone, arm=r.arm, seed=r.seed, windows=len(m), both_finite=int(both.sum()),
                             equal=int((m.predicted_HR_october[both] == m.predicted_HR_new[both]).sum()),
                             max_abs_difference=float((m.predicted_HR_october[both] - m.predicted_HR_new[both]).abs().max()) if both.any() else np.nan,
                             finite_only_october=int((np.isfinite(m.predicted_HR_october) & ~np.isfinite(m.predicted_HR_new)).sum()),
                             finite_only_new=int((~np.isfinite(m.predicted_HR_october) & np.isfinite(m.predicted_HR_new)).sum())))
            fo, fn = np.isfinite(m.predicted_HR_october), np.isfinite(m.predicted_HR_new)
            dm = m[(fo & fn & (m.predicted_HR_october != m.predicted_HR_new)) | (fo != fn)]
            comp[-1].update(different_windows=len(dm), fraction_different=len(dm) / len(m) if len(m) else 0.0,
                            over_1_percent=bool(len(m) and len(dm) / len(m) > 0.01))
            diffs += [dict(key=r.key, backbone=r.backbone, arm=r.arm, seed=r.seed, window_id=x.window_id, layer=x.layer,
                           predicted_HR_october=x.predicted_HR_october, predicted_HR_new=x.predicted_HR_new) for x in dm.itertuples()]
        S2.write_csv(out / 'october_comparison.csv', comp)
        over = [c for c in comp if c['over_1_percent']]
        stop = out / 'october_stop.json'
        if over:
            stop.write_text(json.dumps(dict(rule='windows with a different readout > 1% of the checkpoint windows (6.70 다)',
                                            keys=[dict(key=c['key'], windows=c['windows'], different=c['different_windows'],
                                                       fraction=c['fraction_different']) for c in over]), indent=1))
            print(f'1% RULE: {len(over)} checkpoint(s) above 1% different windows; step (c) must not start ({stop.name})', flush=True)
        elif stop.exists():
            stop.unlink()
        S2.write_csv(out / 'october_differences.csv', diffs or [dict(key='', backbone='', arm='', seed='', window_id='', layer='',
                                                                    predicted_HR_october='', predicted_HR_new='')])
    names = sorted(p.name for p in out.iterdir() if p.suffix in ('.csv', '.gz'))
    (out / 'check_hashes.txt').write_text(''.join(f'{S2.file_sha(out / n)}  {args.dataset}/{n}\n' for n in names))
    run = dict(step='b', dataset=args.dataset, started=started, ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), operator=getpass.getuser(),
               hash_files_verified=nh, keys=len(ck), defective=len(bad), confirmatory_primary_keys=len(conf),
               support_windows=len(support), population_windows=len(popw), october_compared=len(comp), october_different_windows=len(diffs),
               october_stop=bool(comp) and any(c['over_1_percent'] for c in comp))
    (out / 'run.json').write_text(json.dumps(run, indent=1))
    if run['october_stop']:
        sys.exit(3)
    print(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
