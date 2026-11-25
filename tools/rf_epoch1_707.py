"""7.07a: RhythmFormer trainer check (additional check outside A6 item 4), epoch-1 tensors.

The first one-epoch runs (run_707_rf_check2.sh, cache/train704/rfcheck_<arm>) trained 149 steps but kept no state:
train/train_pc.py (A4 code, unchanged) saves a checkpoint only for a complete run. This script runs the same one epoch
again (same command as train/train_704.py RhythmFormer 20261131 ORIGINAL OUT 149) and, inside this process only, keeps
the model, optimizer and scheduler that src.st12c.rhythm_backend_v2.initialize returns to train_pc.main; after the
149 steps it saves their states to OUT/epoch_1_check.pth. Nothing else in the run changes.

  run ARM OUT        ARM 0 | i | ii (environment as for train_704: OPS2_704_ARM=0 or OPS2_ARM=i|ii); OUT absolute
  compare            compares OUT=cache/train704/rfepoch1_<arm> with the October epoch-1 checkpoints (model tensors:
                     largest absolute difference; optimizer and scheduler states: equal or not) and writes
                     outputs/707/rf_trainer_check_epoch1.csv. Epoch-1 loss: the OPS2 mean over the 149 steps; the
                     October records keep the final loss only (no per-step or per-epoch loss), so no loss difference.
"""
import csv
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'train'))
sys.path.insert(0, str(W / 'tools'))
HOLD = {}


def run(arm, out):
    import train_704  # imports train_pc (vendor path and working directory as in every run)
    import src.st12c.rhythm_backend_v2 as rb
    _initialize = rb.initialize

    def initialize(config, seed, *a, **k):
        objs = _initialize(config, seed, *a, **k)
        HOLD['objs'] = objs
        return objs
    rb.initialize = initialize
    sys.argv = ['train_704.py', 'RhythmFormer', '20261131', 'ORIGINAL', str(out), '149']
    train_704.main()
    import torch
    model, criterion, optimizer, scheduler = HOLD['objs']
    res = json.loads((out / 'result.json').read_text())
    assert res['status'] == 'partial_check' and res['steps'] == 149, res
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=1,
                    step=149, arm=arm, seed=20261131, initial_weights_sha256=res['initial_weights_sha256']),
               out / 'epoch_1_check.pth')
    print(f'saved epoch_1_check.pth arm {arm}', flush=True)


def compare():
    from trainer_check_704 import tensor_diff
    idx = {(r['backbone'], r['arm'], r['seed']): r['path'] for r in csv.DictReader(open(W / 'cache/test/checkpoint_index.csv'))}
    out = []
    for a in ('0', 'i', 'ii'):
        d = W / 'cache/train704' / f'rfepoch1_{a}'
        res = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
        october = Path(idx[('RhythmFormer', f'({a})', '20261131')]).with_name('epoch_1.pth')
        steps = list(csv.DictReader(open(d / 'steps.csv'))) if (d / 'steps.csv').exists() else []
        rec = dict(arm=f'({a})', steps=res.get('steps', ''), initial_matches_record=res.get('initial_matches_record', ''),
                   ops2_epoch1_mean_loss=sum(float(s['loss']) for s in steps) / len(steps) if steps else '',
                   october_epoch1_loss='not recorded (final loss only)', saved=(d / 'epoch_1_check.pth').exists())
        if rec['saved']:
            rec.update(tensor_diff(str(d / 'epoch_1_check.pth'), str(october)))
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(W / 'outputs/707/rf_trainer_check_epoch1.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in out for k in r)), lineterminator='\n')
        w.writeheader()
        w.writerows(out)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2], Path(sys.argv[3]))
    else:
        compare()
