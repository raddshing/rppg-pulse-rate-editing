"""7.09b E1: training on UBFC-rPPG, a wrapper around train/train_704.py (train_pc.py and u8_store.py unchanged).

Only the training corpus changes (7.09b 가-3): configuration, initial weights (same seeds), optimizer, scheduler, loss,
batch stream (arm (0): 4 originals; augmented arms: 2 originals + 2 pool items) and step counts are those of the PURE
runs (PhysNet 5,580, RhythmFormer 4,470, FactorizePhys 1,490 steps).
Originals: cache/ubfc709/pools/<backbone>_original_all_manifest.csv (train/gen_ubfc.py originals), sorted by chunk order;
their tensors are rebuilt at load time from the per-recording uint8 view, statistics and contact PPG
(gen_ubfc.original_tensor; whole-recording normalization as the October UBFC originals). Recording views are cached per
data-loader worker. Pool: a pool manifest of train/gen_ubfc.py (u8 storage; train/u8_store.item_input), sorted by
pool_index, or ORIGINAL for arm (0).
Usage: train_ubfc.py BACKBONE SEED POOL_MANIFEST|ORIGINAL OUT_DIR [MAX_STEPS]
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARM0 = sys.argv[3] == 'ORIGINAL' if len(sys.argv) > 3 else False
os.environ.pop('OPS2_704_ARM', None)
import train_704  # noqa: E402
import numpy as np  # noqa: E402
import u8_store  # noqa: E402

train_pc = train_704.train_pc
POOLS = HERE.parent / 'cache' / 'ubfc709' / 'pools'
_VIEWS = {}


def original_tensor(backbone, r):
    import ubfc_tensor
    key = r['ubfc_view_path']
    if key not in _VIEWS:
        _VIEWS[key] = (u8_store.read_array(key), u8_store.read_array(r['ubfc_contact_path']),
                       u8_store.read_array(r['ubfc_stats_path']) if r['ubfc_stats_path'] else None)
    view, gt, stats = _VIEWS[key]
    return ubfc_tensor.original_tensor(backbone, view, stats, gt, int(r['start_frame']), int(r['chunk_length']))


class UBFCTensors(u8_store.U8Tensors):
    """u8_store.U8Tensors with UBFC originals (kind 0) rebuilt from the recording views; pool items unchanged."""

    def __getitem__(self, index):
        import torch
        kind, number = index
        if kind == 1:
            return super().__getitem__(index)
        r = self.original[number]
        x, y = original_tensor(self.backbone, r)
        if self.backbone == 'RhythmFormer':
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)
        else:
            x = np.ascontiguousarray(x.transpose(3, 0, 1, 2), dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), kind, number, int(float(r['reference_HR']) >= 100)


def setup_ubfc(backbone, pool_manifest):
    config, _, _, key, init_file, arm = train_pc.setup_ii(backbone)
    original = sorted(train_pc.rows(POOLS / f'{backbone}_original_all_manifest.csv'), key=lambda r: int(r['chunk_order']))
    assert [int(r['chunk_order']) for r in original] == list(range(len(original)))
    if pool_manifest == 'ORIGINAL':
        return config, original, [], key, init_file, '0'
    pool = [r for r in train_pc.rows(pool_manifest) if str(r.get('generation_excluded', 'False')) != 'True']
    pool.sort(key=lambda r: int(r['pool_index']))
    return config, original, pool, key, init_file, 'ii'


def main():
    backbone, seed, manifest, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    u8_store.U8Tensors = UBFCTensors  # u8_store.loader builds its dataset by this name
    state = setup_ubfc(backbone, manifest)

    def setup(backbone_, arm_kind=None):
        return state
    config, original, pool, *_ = state
    (out / 'settings_ubfc.json').write_text(json.dumps(dict(
        backbone=backbone, seed=seed, corpus='UBFC-rPPG DATASET_2 (42 recordings)', originals=len(original),
        originals_ge100=sum(float(r['reference_HR']) >= 100 for r in original), pool_manifest=manifest, pool_items=len(pool),
        steps=int(config['epochs']) * int(config['steps_per_epoch']), stream_arm=state[5]), indent=1))
    train_704.setup = setup
    # the pool manifest is passed as ORIGINAL so that train_pc does not merge a PURE pool by key; the UBFC pool rows
    # already carry their file paths (ops2_view_path, ops2_rawlabel_path)
    sys.argv = [sys.argv[0], backbone, str(seed), 'ORIGINAL', str(out)] + sys.argv[5:6]
    train_704.main()


if __name__ == '__main__':
    main()
