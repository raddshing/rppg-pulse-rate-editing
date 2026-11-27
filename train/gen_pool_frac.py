"""7.09b E7: pools of the partial-misalignment arms (FactorizePhys first pool items; train/gen_pool.py unchanged).

Runs gen_pool.job with its injection function replaced, in each worker process, by audit/editor_frac.injection_wave, which
adds modes lag_frac_<alpha> and leaves every other mode unchanged. Items, contexts, targets, notch, masks, calibration,
amplitude, rounding generator ([20261202, h] for FactorizePhys) and label are those of the first-pool (ii-pc)/(ii-lag) items
(same manifest, same per-item key, a fresh rounding generator per mode). All alphas of one run are made in one pass.
Check mode: MODES may include polarity_explicit and lag_shift; their tensors are compared with the frozen first-pool
manifests (cache/pools/<BACKBONE>_<mode>_all_manifest.csv).
Usage: gen_pool_frac.py BACKBONE MODES OUT_ROOT [ITEM_LIST.csv]   (OPS2_WORKERS, OPS2_POOL_COMPRESS as gen_pool.py)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'audit'))
import gen_pool as GP  # noqa: E402


def init_worker(backbone):
    GP.init_worker(backbone)
    import editor_frac
    GP.injection_wave = editor_frac.injection_wave  # gen_pool.job looks the name up in its own module


def main():
    backbone, modes, out_root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    items = GP.read_csv(GP.VENDOR / GP.MANIFEST[backbone])
    items = [r for r in items if str(r.get('generation_excluded', 'False')) != 'True']
    tag = 'all'
    if len(sys.argv) > 4:
        keep = {(r['chunk_id'], r['augmentation_index']) for r in GP.read_csv(sys.argv[4])}
        items = [r for r in items if (r['chunk_id'], r['augmentation_index']) in keep]
        tag = Path(sys.argv[4]).stem
    by_rec = {}
    for r in items:
        by_rec.setdefault(r['recording_id'], []).append(r)
    workers = int(os.environ.get('OPS2_WORKERS', '4'))
    jobs = [(backbone, modes, str(out_root), rid, its) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    started, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                             initargs=(backbone,)) as ex:
        for n, out in enumerate(ex.map(GP.job, jobs), 1):
            rows += out
            print(f'frac {backbone} {n}/{len(jobs)} recordings, {len(rows)} rows, {time.time() - started:.0f}s', flush=True)
    for m in modes.split(','):
        mr = [r for r in rows if r['mode'] == m]
        GP.write_rows(out_root / f'{backbone}_{m}_{tag}_manifest.csv', mr)
        line = f'done {m} {len(mr)}'
        ref = Path(__file__).resolve().parents[1] / 'cache' / 'pools' / f'{backbone}_{m}_all_manifest.csv'
        if m in ('polarity_explicit', 'lag_shift') and ref.exists():
            old = {(q['chunk_id'], q['augmentation_index']): q for q in GP.read_csv(ref)}
            eq = sum(r['ops2_input_sha256'] == old[(r['chunk_id'], r['augmentation_index'])]['ops2_input_sha256'] for r in mr)
            line += f' input_sha_equal_first_pool {eq}/{len(mr)}'
        print(line, flush=True)


if __name__ == '__main__':
    main()
