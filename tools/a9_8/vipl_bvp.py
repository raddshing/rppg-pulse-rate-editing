"""OPS2 A9 items 8g (recording level) and 8c (phase): the BVP record of every primary-set recording of VIPL-HR stage (a)
placed on the 30-Hz axis from the first frame at 60 Hz, as in A8 (eval5/vipl_a8.py job: vipl.read, vipl.column,
src.st11x.pathways.reference_30hz(wave, 60)); no video is decoded and no prediction is read.
Server-only cache (window-level physiological signals of VIPL-HR, never exported): cache/a9_8/bvp60/<recording>.npz with
c30 (float64, BVP on the 30-Hz grid at 60 Hz) and gt_HR (the gt_HR.csv rows, one per second, NaN-free or empty).
Recordings: stage (a) recordings of the primary set with windows (recordings.csv exclusion empty).
Usage (main environment): vipl_bvp.py [--workers N] [--limit N]
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'eval5'))
SA = WORK / 'cache' / 'vipl707' / 'stage_a' / 'VIPL'
CACHE = WORK / 'cache' / 'a9_8' / 'bvp60'
FS = 60.0
G = {}


def paths():
    sh = lambda v, f: subprocess.run(['bash', '-c', f'. {WORK}/eval5/drive/{f}; printf %s "${v}"'], capture_output=True, text=True).stdout  # noqa: E731
    return sh('VIPL_ZIPS', 'local_paths_vipl.sh'), sh('VENDOR', 'local_paths.sh')


def init(vendor):
    import vipl
    vipl.init_worker(vendor)
    G['vipl'] = vipl


def job(rec):
    vipl = G['vipl']
    from src.st11x.pathways import reference_30hz
    wave = vipl.column(vipl.read(rec, 'wave.csv'), 'Wave')
    hr = vipl.column(vipl.read(rec, 'gt_HR.csv'), 'HR') if 'gt_HR.csv' in rec['members'] else np.zeros(0)
    c30 = np.asarray(reference_30hz(wave, FS), dtype=np.float64)
    np.savez(CACHE / f"{rec['recording_id']}.npz", c30=c30, gt_HR=np.asarray(hr, dtype=np.float64))
    return rec['recording_id'], len(c30), len(hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2), ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    zips, vendor = paths()
    assert zips and vendor
    import vipl
    recs_a = pd.read_csv(SA / 'recordings.csv', dtype={'subject_id': str})
    use = set(recs_a[recs_a.set.eq('primary') & recs_a.exclusion.isna() & (recs_a.windows > 0)].recording_id)
    recs = [r for r in vipl.index(zips) if r['recording_id'] in use]
    assert len(recs) == len(use), (len(recs), len(use))
    recs = [r for r in recs if not (CACHE / f"{r['recording_id']}.npz").exists()]
    if a.limit:
        recs = recs[:a.limit]
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'vipl bvp: {len(use)} primary recordings with windows, {len(recs)} to read', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(vendor,)) as ex:
        for n, (rid, n30, nhr) in enumerate(ex.map(job, recs), 1):
            if n % 100 == 0 or n == len(recs):
                print(f'vipl bvp {n}/{len(recs)} {time.time() - t0:.0f}s', flush=True)
    print('done', len(list(CACHE.glob('*.npz'))), 'files', flush=True)


if __name__ == '__main__':
    main()
