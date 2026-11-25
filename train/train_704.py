"""Brief 02 7.04/7.04a (A6 items 1-4): wrapper around train/train_pc.py for the post hoc runs.

train/train_pc.py is not changed (its bytes stay those of docs/A4_CODE_SHA256.txt). This wrapper imports it and makes,
before its main() starts and inside this process only, the substitutions that a run needs:
  - arm (0) (OPS2_704_ARM=0): train_pc.setup returns the same configuration, originals, initial-weight record and key as
    for (ii), with an empty pool and stream arm '0' (four originals per step, src/st9s/streams.SourceStreams.batch, as in
    the October runs: st9t 'batch: 0/vii four original; augmented two original/two augmented').
    Every other arm keeps train_pc.setup (OPS2_ARM=i or ii) and its pool manifest argument, as in the earlier OPS2 runs.
  - loss (OPS2_704_LOSS=mcc): the criterion returned by src.st9s.backend.initialize (PhysNet/FactorizePhys negative
    Pearson) is replaced by train/mcc_loss.MCCLoss (Fs 30, band 0.6-3.3 Hz). The archived step is unchanged: batch
    standardization of label and prediction, then criterion(pred, label). OPS2_704_LOSS unset or 'pearson': unchanged.
  - labels: OPS2_LABEL_REPR=std_cumsum (existing switch of train/u8_store.py, rule 5) for PhysNet-Std.
  - new seeds (A6 item 3: 20261148-20261150): train_pc.main checks the initial weights against the recorded same-seed
    digest; the records have none for new seeds. For these seeds only, the digest of a model built by the same
    initialize(...) with that seed on the run's device is computed first and added to the record read by main (process-local); the check
    then runs as for the recorded seeds. Recorded seeds are checked against the October records unchanged.
Writes OUT_DIR/settings_704.json before training (every setting that a run uses, for the diff against its reference
run) and OUT_DIR/loss_704.json after it. RhythmFormer (7.07, A6 item 3 P): confirmatory loss only (OPS2_704_LOSS unset);
its initial weights come from src.st12c.rhythm_backend_v2.initialize(config, seed), as in train_pc.main. The MCC loss
for RhythmFormer (conditional item 9) is not handled here.
Usage: train_704.py BACKBONE SEED POOL_MANIFEST|ORIGINAL OUT_DIR [MAX_STEPS]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_pc  # noqa: E402  (its top level sets the thread variables, the vendor path and the working directory)
import json  # noqa: E402

NEW_SEEDS = (20261148, 20261149, 20261150)
ARM = os.environ.get('OPS2_704_ARM', '')
LOSS = os.environ.get('OPS2_704_LOSS', 'pearson')
assert ARM in ('', '0'), 'OPS2_704_ARM is 0 or unset (other arms use OPS2_ARM and the pool manifest)'
assert LOSS in ('pearson', 'mcc')
_setup = train_pc.setup


def setup(backbone, arm_kind=None):
    if ARM == '0':
        config, original, pool, key, init_file, arm = train_pc.setup_ii(backbone)
        return config, original, [], key, init_file, '0'
    return _setup(backbone, arm_kind)


def settings(backbone, seed, manifest, out):
    config, original, pool, key, init_file, arm = setup(backbone)
    return dict(backbone=backbone, seed=seed, stream_arm=arm, OPS2_ARM=os.environ.get('OPS2_ARM', 'ii'),
                pool_manifest=manifest, pool_items=len(pool), originals=len(original),
                label_representation=os.environ.get('OPS2_LABEL_REPR', '') or ('DiffNormalized' if backbone == 'PhysNet' else 'Standardized'),
                loss='MCC (train/mcc_loss.MCCLoss, 0.6-3.3 Hz, batch mean)' if LOSS == 'mcc' else
                ('archived RhythmFormer loss (src/st12c/rhythm_backend_v2)' if backbone == 'RhythmFormer' else 'archived negative Pearson'),
                learning_rate=str(config['learning_rate']), epochs=int(config['epochs']), steps_per_epoch=int(config['steps_per_epoch']),
                batch='4 originals' if arm == '0' else '2 originals + 2 pool items', initial_weight_record=f'{init_file} {key}|{seed}',
                new_seed=seed in NEW_SEEDS, data_workers=os.environ.get('OPS2_DATA_WORKERS', '4'),
                threads=os.environ.get('OPS2_THREADS', 'one'), lr_factor=os.environ.get('OPS2_LR_FACTOR', ''),
                partial_pool=os.environ.get('OPS2_PARTIAL_POOL', ''), out_dir=str(out.name))


def main():
    backbone, seed, manifest, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    assert backbone in ('PhysNet', 'FPm') or (backbone == 'RhythmFormer' and LOSS == 'pearson'), 'RhythmFormer: confirmatory loss only'
    out.mkdir(parents=True, exist_ok=True)
    train_pc.setup = setup
    import torch  # noqa: F401
    import src.st9s.backend as backend
    _initialize = backend.initialize
    if LOSS == 'mcc':
        from mcc_loss import MCCLoss

        def initialize(name, settings_, seed_, device='cpu'):
            model, criterion, optimizer, scheduler = _initialize(name, settings_, seed_, device)
            return model, MCCLoss(Fs=30), optimizer, scheduler
        backend.initialize = initialize
    s = settings(backbone, seed, manifest, out)
    if seed in NEW_SEEDS:
        config, _, _, key, init_file, _ = setup(backbone)
        if backbone == 'RhythmFormer':
            from src.st12c.rhythm_backend_v2 import initialize as rhythm_initialize
            model = rhythm_initialize(config, seed)[0]  # as train_pc.main for RhythmFormer
        else:
            name = 'FactorizePhys' if backbone == 'FPm' else 'PhysNet'
            model = _initialize(name, config, seed, 'cuda:0')[0]  # same device as the run (FactorizePhys builds some tensors on its device)
        digest = backend.state_digest(model)
        del model
        record = json.loads((train_pc.V / init_file).read_text())
        assert f'{key}|{seed}' not in record
        _loads = train_pc.json.loads

        class _Json:
            dumps = staticmethod(json.dumps)

            @staticmethod
            def loads(text, *a, **k):
                d = _loads(text, *a, **k)
                if isinstance(d, dict) and d.keys() == record.keys():
                    d = dict(d, **{f'{key}|{seed}': {'sha256': digest, 'source': 'train_704 new seed (same initialize)'}})
                return d
        train_pc.json = _Json
        s['new_seed_initial_weights_sha256'] = digest
    (out / 'settings_704.json').write_text(json.dumps(s, indent=1))
    train_pc.main()
    (out / 'loss_704.json').write_text(json.dumps(dict(loss=s['loss'], label_representation=s['label_representation'],
                                                       stream_arm=s['stream_arm']), indent=1))


if __name__ == '__main__':
    main()
