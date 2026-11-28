"""OPS2 7.19 (B1, after A10): training on a B1 chunk set of UBFC-rPPG, train/train_ubfc.py unchanged except the originals.

Originals: the chunks of cache/b1_719/plan/<list>_<set>_chunks.csv (set U_low or U_full_m; list PhysNet for PhysNet,
RhythmFormer for RhythmFormer and FactorizePhys), taken from the regenerated originals
cache/b1_719/pools/<backbone>_original_all_manifest.csv (train/gen_ubfc.py originals, tensors equal to E1), sorted by
chunk order. Pool: a B1 family-(i) manifest (cache/b1_719/pools/<backbone>_{i,i-pc,i-simple}_all_manifest.csv) or
ORIGINAL for arm (0). Everything else (configuration, initial weights, optimizer, loss, batch stream: arm (0) 4
originals, augmented arms 2 originals + 2 pool items; steps PhysNet 5,580, RhythmFormer 4,470, FactorizePhys 1,490) is
that of train/train_ubfc.py and train/train_704.py, unchanged.
The arguments are read only in the main process: the data-loader workers (spawn) run this file again as __mp_main__ with
the argument list already rewritten for train_ubfc.py, and only need the module imports (7.20: the first trial version
read them at import and its workers stopped there).
Usage: train_b1.py BACKBONE SEED U_low|U_full_m POOL_MANIFEST|ORIGINAL OUT_DIR [MAX_STEPS]
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / 'train'  # 7.21: published in tools/a10/; train_ubfc.py is in train/, the cache folder at the root
B1 = HERE.parent / 'cache' / 'b1_719'
LIST = {'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer', 'FPm': 'RhythmFormer'}
sys.path.insert(0, str(HERE))
import train_ubfc as TU  # noqa: E402

CHUNKS = None


def setup_b1(backbone_, pool_manifest):
    config, _, _, key, init_file, arm = TU.train_pc.setup_ii(backbone_)
    with open(CHUNKS, newline='') as f:
        wanted = {r['chunk_id'] for r in csv.DictReader(f)}
    allrows = TU.train_pc.rows(B1 / 'pools' / f'{backbone_}_original_all_manifest.csv')
    original = sorted([r for r in allrows if r['chunk_id'] in wanted], key=lambda r: int(r['chunk_order']))
    assert len(original) == len(wanted), (len(original), len(wanted))
    if pool_manifest == 'ORIGINAL':
        return config, original, [], key, init_file, '0'
    pool = [r for r in TU.train_pc.rows(pool_manifest) if str(r.get('generation_excluded', 'False')) != 'True']
    assert {r['chunk_id'] for r in pool} <= wanted, 'pool items outside the chunk set'
    pool.sort(key=lambda r: int(r['pool_index']))
    return config, original, pool, key, init_file, 'ii'


def main():
    global CHUNKS
    backbone, seed, chunk_set, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
    assert backbone in LIST and chunk_set in ('U_low', 'U_full_m'), sys.argv
    CHUNKS = B1 / 'plan' / f'{LIST[backbone]}_{chunk_set}_chunks.csv'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'settings_b1.json').write_text(json.dumps(dict(backbone=backbone, seed=int(seed), chunk_set=chunk_set,
                                                          chunk_list=CHUNKS.name, pool_manifest=Path(manifest).name), indent=1))
    sys.argv = [sys.argv[0], backbone, seed, manifest, str(out)] + sys.argv[6:7]  # the argument list of train_ubfc.py
    TU.setup_ubfc = setup_b1
    TU.POOLS = B1 / 'pools'
    TU.main()


if __name__ == '__main__':
    main()
