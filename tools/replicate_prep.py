"""Brief 02 6.67 마 (diagnostic rule 6): checkpoint index and manifest of the nine (i-simple) replicate runs for
eval5/stage2_predict.py, in the format of tools/rehearsal_index.py (index: set,backbone,arm,seed,path,run_id,source;
manifest: kind,backbone,arm,seed,n,sha256). Arm name '(i-simple-rep)', so the stage-2 keys differ from the original runs'.
Writes cache/replicates/checkpoint_index.csv and cache/replicates/manifest_replicates.csv; exit 3 unless all nine
replicate runs are complete. Usage: replicate_prep.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
NAME = {'FPm': 'FactorizePhys-matched', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
SEEDS = (20261131, 20261132, 20261133)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    idx, man = [], []
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for s in SEEDS:
            d = WORK / 'cache' / 'train' / f'{b}_i-simple-rep_s{s}'
            try:
                res = json.loads((d / 'result.json').read_text())
            except (OSError, ValueError):
                sys.exit(f'{d.name}: no result')
            if res.get('status') != 'complete' or res.get('learning_rate_factor') != '1.000001':
                sys.exit(f'{d.name}: not a complete replicate run')
            h = sha(d / 'final_epoch.pth')
            assert h == res['checkpoint_sha256']
            idx.append(dict(set='exploratory', backbone=NAME[b], arm='(i-simple-rep)', seed=s, path=str(d / 'final_epoch.pth'),
                            run_id=d.name, source='OPS2 replicate (diagnostic rule 6)'))
            man.append(dict(kind='checkpoint', backbone=NAME[b], arm='(i-simple-rep)', seed=s, n='', sha256=h))
    out = WORK / 'cache' / 'replicates'
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('checkpoint_index.csv', idx), ('manifest_replicates.csv', man)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print('replicate index and manifest: 9 checkpoints')


if __name__ == '__main__':
    main()
