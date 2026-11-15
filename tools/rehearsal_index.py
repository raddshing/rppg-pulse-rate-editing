"""Rehearsal (brief 02 6.43 마, 6.53 나): checkpoint index for eval5/stage2_predict.py and a draft A4 exploratory manifest.

checkpoint_index.csv (server-local, not committed): backbone, arm, seed, path, run_id (the run_id stored in the checkpoint,
checked by the inference server), source. Manifest names: backbone FactorizePhys-matched / PhysNet / RhythmFormer, arm '(..)'.
  confirmatory: October (0)/(i)/(ii) from the October roster (b1/roster.py); (ii-pc), (i-pc) from the OPS2 runs.
  exploratory: (i-simple), (ii-lag), label/processing arms (6.43 라) of the OPS2 runs whose three seeds completed.
manifest_A4_draft.csv: kind,backbone,arm,seed,n,sha256 for the exploratory checkpoints present now (draft; A4 itself is
written from the final list). Every confirmatory row of the committed manifests must have an index entry.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
from roster import roster  # noqa: E402

NAME = {'FactorizePhys (matched)': 'FactorizePhys-matched', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
OPS2 = {'FactorizePhys-matched': 'FPm', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
SEEDS = (20261131, 20261132, 20261133)
EXPL = {'FactorizePhys-matched': ('i-simple', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'sham-pc'),
        'PhysNet': ('i-simple', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'i-pc-g80', 'sham-pc', 'roi-pc'),
        'RhythmFormer': ('i-simple', 'ii-lag')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ops2_run(backbone, arm, seed):
    d = WORK / 'cache' / 'train' / f'{OPS2[backbone]}_{arm}_s{seed}'
    try:
        ok = json.loads((d / 'result.json').read_text()).get('status') == 'complete'
    except (OSError, ValueError):
        ok = False
    return (str(d / 'final_epoch.pth'), d.name) if ok else None


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / 'outputs' / 'rehearsal'  # optional: snapshot folder of one pass
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in roster():
        b = NAME.get(r['backbone'])
        if b and r['strategy'] in ('(0)', '(i)', '(ii)') and r['seed'] in SEEDS:
            rows.append(dict(set='confirmatory', backbone=b, arm=r['strategy'], seed=r['seed'], path=r['checkpoint'], run_id=r['run_id'], source='October'))
    for b in OPS2:
        for arm in ('ii-pc', 'i-pc'):
            for s in SEEDS:
                x = ops2_run(b, arm, s)
                assert x, (b, arm, s)
                rows.append(dict(set='confirmatory', backbone=b, arm=f'({arm})', seed=s, path=x[0], run_id=x[1], source='OPS2'))
    a4 = []
    for b, arms in EXPL.items():
        for arm in arms:
            runs = [ops2_run(b, arm, s) for s in SEEDS]
            if not all(runs):
                print('not complete (not indexed):', b, arm, flush=True)
                continue
            for s, x in zip(SEEDS, runs):
                rows.append(dict(set='exploratory', backbone=b, arm=f'({arm})', seed=s, path=x[0], run_id=x[1], source='OPS2'))
                a4.append(dict(kind='checkpoint', backbone=b, arm=f'({arm})', seed=s, n='', sha256=sha(x[0])))
    man = []
    for f in ('PREREGISTRATION_STAGE5_MANIFEST.csv', 'PREREGISTRATION_STAGE5_MANIFEST_A1.csv'):
        man += [r for r in csv.DictReader(open(WORK / 'pub' / 'docs' / f, newline='')) if r['kind'] == 'checkpoint']
    have = {(r['backbone'], r['arm'], int(r['seed'])) for r in rows if r['set'] == 'confirmatory'}
    need = {(r['backbone'], r['arm'], int(r['seed'])) for r in man}
    assert need == have, (need ^ have)
    with (out / 'checkpoint_index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / 'manifest_A4_draft.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'backbone', 'arm', 'seed', 'n', 'sha256'], lineterminator='\n')
        w.writeheader()
        w.writerows(a4)
    print('confirmatory', len(have), 'manifest rows', len(man), 'exploratory indexed', len(a4))


if __name__ == '__main__':
    main()
