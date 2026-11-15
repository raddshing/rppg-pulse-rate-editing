"""1b model roster: the 88 PURE-trained draft models (brief 01 table + 5-seed extras)."""
import csv
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
BULK = Path('<MACHINE_PATH>')  # read-only checkpoints
VIEW = {'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer', 'FactorizePhys': 'FactorizePhys',
        'FactorizePhys (matched)': 'FPm'}
NS_DIR = {'st9t': 'st9t', 'st12c': 'st12c/checkpoints', 'st12v': 'st12v/checkpoints', 'st13x': 'st13x/checkpoints',
          'st16v': 'st16v/checkpoints', 'st30t': 'st30t/checkpoints', 'st31t': 'st31t/checkpoints', 'st32t': 'st32t/checkpoints'}


def roster():
    out = []
    with (WORK / 'outputs' / 'stage0_checkpoints.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['training_corpus'] != 'PURE':
                continue
            base = BULK / NS_DIR[r['namespace']] / r['run_name']
            finals = sorted(base.rglob('final_epoch.pth'))
            assert finals, r['run_name']
            seed = int(r['seed'])
            out.append(dict(run_id=r['run_name'], backbone=r['backbone'], view=VIEW[r['backbone']],
                            strategy=r['strategy'], seed=seed, five_seed_extra=seed in (20261143, 20261144),
                            checkpoint=str(finals[-1]), namespace=r['namespace']))
    assert len(out) == 88, len(out)
    return out
