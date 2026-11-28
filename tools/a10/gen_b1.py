"""OPS2 7.19 A (B1): family-(i) pools of the B1 U-low items with train/gen_ubfc.py job_family_i unchanged.

The only differences from the A9 item 1 (E1) family-(i) build are the plan (cache/b1_719/plan/<backbone>_items.csv, written
by tools/b1_719/b1_plan.py: the U-low items and their targets) and the key of the rounding generator, 20261160 instead of
20261159 (gen_ubfc.KEY, set in every worker before its first item; generator default_rng([20261160, j, h])). Arms (i),
(i-pc) with d = 1 and (i-simple) share one generated clip per item, as in E1.
Usage: gen_b1.py BACKBONE OUT_ROOT   (OPS2_WORKERS)
Writes OUT_ROOT/<backbone>/i/<recording>/... and OUT_ROOT/<backbone>_{i,i-pc,i-simple}_all_manifest.csv (refuses to overwrite).
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'train'))  # 7.21: published in tools/a10/; gen_ubfc.py is in train/
import gen_ubfc as G  # noqa: E402

PLAN = G.WORK / 'cache' / 'b1_719' / 'plan'
KEY_B1 = 20261160


def init_worker(backbone):
    G.KEY = KEY_B1
    G.init_worker(backbone)
    print(f'b1 worker {os.getpid()} rounding key {G.KEY}', flush=True)


def main():
    backbone, out_root = sys.argv[1], Path(sys.argv[2])
    assert backbone in G.J
    outs = [out_root / f'{backbone}_{arm}_all_manifest.csv' for arm in ('i', 'i-pc', 'i-simple')]
    for o in outs:
        if o.exists():
            sys.exit(f'{o} exists')
    rows = G.read_csv(PLAN / f'{backbone}_items.csv')
    assert all(r['rounding_key'] == f"[{KEY_B1},{G.J[backbone]},h]" for r in rows)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r['recording_id'], []).append(r)
    jobs = [(backbone, str(out_root), rid, its) for rid, its in sorted(by_rec.items(), key=lambda kv: -len(kv[1]))]
    started, out = time.time(), []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '3')), mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(backbone,)) as ex:
        for i, o in enumerate(ex.map(G.job_family_i, jobs), 1):
            out += o
            print(f'b1 {backbone} family_i {i}/{len(jobs)} recordings, {len(out)} rows, {time.time() - started:.0f}s', flush=True)
    for (mode, arm), path in zip((('original', 'i'), ('d', 'i-pc'), ('zero', 'i-simple')), outs):
        mr = [r for r in out if r.get('mode') == mode]
        excl = [r for r in out if r.get('generation_excluded') is True]
        G.write_rows(path, mr + excl)
        print(f'done {backbone} ({arm}) items {len(mr)} excluded {len(excl)}', flush=True)


if __name__ == '__main__':
    main()
