"""OPS2 A9 item 8g, recording level (7.13 1): VIPL-HR recording-level scores in the benchmark convention for every checkpoint
evaluated on VIPL-HR when A9 was committed (stages (b1), (b2); 177 keys of primary variants, seeds 20261131-20261133).

Benchmark convention (eval5/benchmark.recording_row, unchanged; the October code src/eval/har.evaluate_arrays,
grid._prepare and grid._estimate 'periodogram', band 0.75-2.5 Hz): one HR per recording from the whole predicted waveform
(the packed prediction of stage (b) on the 30-Hz grid from the first frame, float32 as stored) and one from the whole
contact signal over the same interval (the prediction prefix of the recording unit intersected with the finite contact
signal; a gap inside it leaves the recording undefined).
Contact signal: the BVP record placed on the 30-Hz axis from the first frame at 60 Hz (as A8; cache of tools/a9_8/vipl_bvp.py),
padded with NaN where the prediction is longer than the BVP. Two reference HR per recording: (1) the benchmark reference,
the L0 HR of the BVP over the interval; (2) the mean of the gt_HR rows (one per second from the first frame) whose second
lies fully inside the interval (the reference of the primary analysis is the window mean of gt_HR).
Recordings: the primary set (source1, source3) recordings with at least one stage (a) window (1,408).
Per key: recordings, recordings with both HR defined, MAE, RMSE, Pearson r, and the MAE of recordings with reference >= 100;
per arm: seed mean and sample SD (as benchmark.run). Oracle constant predictor: the mean, or the median, of the recording
references of each backbone's (0) arm, seed 20261131 (its intervals), given to every one of those recordings.
Writes outputs/a9_8/8g/{8g_recording_keys,8g_recording_arms,8g_recording_constant}.csv; recording rows server-only in
cache/a9_8/8g_recordings.csv.gz.
"""
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import a9_common as C
from a9_common import np, pd
import stage3_check as SC
import benchmark as BM

BVP = C.WORK / 'cache' / 'a9_8' / 'bvp60'
VENDOR = str(C.P2.VENDOR)


def job(args):
    key, folder, view, diff, rids = args
    SC.vendor_imports(VENDOR)
    from src.eval import grid
    out = []
    for rid in rids:
        f = f'{folder}/pred/{key}/{rid}.npy'
        try:
            pred = np.load(f)
        except FileNotFoundError:
            out.append(dict(key=key, recording_id=rid, status='no_prediction'))
            continue
        z = np.load(BVP / f'{rid}.npz')
        c30, gt = z['c30'], z['gt_HR']
        lab = np.full(max(len(pred), len(c30)), np.nan)
        lab[:len(c30)] = c30
        row = dict(key=key, recording_id=rid, **BM.recording_row(pred, lab, view, diff))
        if row['status'] == 'ok':
            a, b = row['start'], row['end']
            secs = [i for i in range(len(gt)) if 30 * i >= a and 30 * (i + 1) <= b]
            row['reference_gtHR'] = float(np.mean(gt[secs])) if secs else np.nan
        out.append(row)
    return out


def scores(ok, ref):
    ok = ok[np.isfinite(ok[ref])]
    e = ok.predicted_HR - ok[ref]
    hi = ok[ok[ref] >= 100]
    return dict(recordings_ok=len(ok), MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                pearson_r=float(np.corrcoef(ok.predicted_HR, ok[ref])[0, 1]) if len(ok) > 2 and ok.predicted_HR.std() > 0 else np.nan,
                recordings_ge100=len(hi), MAE_ge100=float((hi.predicted_HR - hi[ref]).abs().mean()) if len(hi) else np.nan)


def main():
    t0 = time.time()
    SC.init(VENDOR)
    keys = C.vipl_keys()
    bad = set(pd.read_csv(C.VC2 / 'defects.csv').key)
    keys = keys[keys.status.eq('used') & keys.variant.isin(C.SA.PRIMARY) & ~keys.key.isin(bad)]
    rids = sorted(p.stem for p in BVP.glob('*.npz'))
    jobs = []
    for k in keys.itertuples():
        folder = str(C.VO / ('stage_b_confirmatory' if k.set == 'confirmatory' else 'stage_b_exploratory') / 'VIPL')
        diff = k.view == 'PhysNet' and '-std' not in k.arm
        jobs.append((k.key, folder, k.view, diff, rids))
    rows = []
    with ProcessPoolExecutor(max_workers=int(__import__('os').environ.get('A9_WORKERS', '1')), mp_context=mp.get_context('spawn'), initializer=SC.init, initargs=(VENDOR,)) as ex:
        for n, part in enumerate(ex.map(job, jobs), 1):
            rows += part
            if n % 10 == 0 or n == len(jobs):
                print('keys', n, '/', len(jobs), round(time.time() - t0), flush=True)
    rec = pd.DataFrame(rows).merge(keys[['key', 'backbone', 'arm', 'seed', 'set']], on='key')
    rec['backbone'] = rec.backbone.map(C.SA.NAME)
    rec.rename(columns={'reference_HR': 'reference_BVP60_L0'}).to_csv(C.WORK / 'cache' / 'a9_8' / '8g_recordings.csv.gz', index=False)
    rec = rec.rename(columns={'reference_HR': 'reference_BVP60_L0'})
    per_key = []
    for (bb, arm, seed, key, ks), g in rec.groupby(['backbone', 'arm', 'seed', 'key', 'set']):
        ok = g[g.status.eq('ok')]
        for ref in ('reference_BVP60_L0', 'reference_gtHR'):
            per_key.append(dict(dataset='VIPL-HR', backbone=bb, strategy=arm, seed=int(seed), key=key, key_set=ks, reference=ref,
                                recordings=len(g), status_counts=';'.join(f'{s}:{n}' for s, n in g.status.value_counts().items()),
                                **scores(ok, ref)))
    per_key = pd.DataFrame(per_key)
    arms = []
    for (bb, arm, ref), g in per_key.groupby(['backbone', 'strategy', 'reference']):
        r = dict(dataset='VIPL-HR', backbone=bb, strategy=arm, reference=ref, seeds=len(g), key_set=g.key_set.iloc[0],
                 recordings_ok_min=int(g.recordings_ok.min()), recordings_ok_max=int(g.recordings_ok.max()),
                 recordings_ge100_min=int(g.recordings_ge100.min()), recordings_ge100_max=int(g.recordings_ge100.max()))
        for m in ('MAE', 'RMSE', 'pearson_r', 'MAE_ge100'):
            r[m + '_mean'] = float(g[m].mean())
            r[m + '_sd'] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
        arms.append(r)
    const = []
    for bb, g in rec[rec.arm.eq('(0)') & rec.seed.eq(20261131) & rec.status.eq('ok')].groupby('backbone'):
        for ref in ('reference_BVP60_L0', 'reference_gtHR'):
            z = g[np.isfinite(g[ref])]
            for name, c in (('mean', float(z[ref].mean())), ('median', float(z[ref].median()))):
                s = scores(z.assign(predicted_HR=c), ref)
                s['pearson_r'] = np.nan  # a constant has no correlation
                const.append(dict(dataset='VIPL-HR', intervals_of=f'{bb} (0) seed 20261131', reference=ref,
                                  predictor=f'constant {name} reference HR', constant_bpm=c, **s))
    C.write(per_key.drop(columns=['key']), '8g_recording_keys.csv', '8g')
    C.write(pd.DataFrame(arms), '8g_recording_arms.csv', '8g')
    C.write(pd.DataFrame(const), '8g_recording_constant.csv', '8g')
    C.stamp('8g_recording', seconds=time.time() - t0, recordings=len(rids), keys=len(jobs))
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
