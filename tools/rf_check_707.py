"""7.07 (RhythmFormer P): trainer check of the OPS2 code for RhythmFormer arms (0), (i), (ii), seed 20261131, one epoch.
The runs of run_707_rf_check.sh (train/train_704.py, MAX_STEPS = steps per epoch) are compared with the October epoch-1
checkpoints of the confirmatory runs (model, optimizer and scheduler tensors; tools/trainer_check_704.tensor_diff).
Writes outputs/707/rf_trainer_check.csv."""
import csv
import json
import sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
from trainer_check_704 import tensor_diff  # noqa: E402
idx = {(r['backbone'], r['arm'], r['seed']): r['path'] for r in csv.DictReader(open(W / 'cache/test/checkpoint_index.csv'))}
out = []
for a in ('0', 'i', 'ii'):
    d = W / 'cache/train704' / f'rfcheck_{a}'
    res = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
    st = json.loads((d / 'settings_704.json').read_text()) if (d / 'settings_704.json').exists() else {}
    october = Path(idx[('RhythmFormer', f'({a})', '20261131')]).with_name('epoch_1.pth')
    rec = dict(arm=f'({a})', status=res.get('status', 'absent'), steps=res.get('steps', ''), steps_per_epoch=st.get('steps_per_epoch', ''),
               initial_matches_record=res.get('initial_matches_record', ''))
    if res.get('status') == 'complete':
        rec.update(tensor_diff(str(d / 'final_epoch.pth'), str(october)))
    out.append(rec)
    print(json.dumps(rec))
with open(W / 'outputs/707/rf_trainer_check.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in out for k in r}), lineterminator='\n'); w.writeheader(); w.writerows(out)
