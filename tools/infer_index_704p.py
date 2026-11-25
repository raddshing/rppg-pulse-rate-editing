"""Brief 02 7.04 / A6 items 3, 5 and 9: manifest and checkpoint index of the P and R checkpoints for guarded inference
(eval5/stage2_predict.py, as tools/infer_index_704.py for the 84 checkpoints of items 1-2 and rule 5).

Groups:
  P_PhysNet_FPm   A6 item 3: PhysNet and FactorizePhys, arms (0), (i), (i-pc), (i-simple), (ii), (ii-pc), seeds 20261148-50
                  (36 checkpoints, cache/train704/P_<backbone>_<arm>_s<seed>); test sets only (A6 item 5).
  P_RhythmFormer  the same for RhythmFormer (18): (0) from run_707_P_RF0.sh, the pool arms from run_707_RF.sh.
  R               A6 item 9: RhythmFormer with the MCC time term, arms (0), (i), (i-pc), (i-simple), (ii), (ii-lag), (ii-pc),
                  seeds 20261131-33 (21, cache/train704/R_RhythmFormer_<arm>_s<seed>); named '(x-mcc)' as the M runs;
                  test sets and MMPD-A, MCD-dev (as items 1-2).
P arms keep the confirmatory arm names; their keys differ from the confirmatory keys by the seed.
A checkpoint must have result.json status complete and final_epoch.pth; otherwise the tool exits 1 and writes nothing.
Writes <OUT>/manifest.csv (kind,backbone,arm,seed,n,sha256) and <OUT>/checkpoint_index.csv (set,backbone,arm,seed,path,
run_id,source). Usage: infer_index_704p.py GROUP OUT_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

C = Path(__file__).resolve().parents[1] / 'cache' / 'train704'
BB = {'PhysNet': 'PhysNet', 'FPm': 'FactorizePhys-matched', 'RhythmFormer': 'RhythmFormer'}
P_SEEDS = (20261148, 20261149, 20261150)
SEEDS = (20261131, 20261132, 20261133)
P_ARMS = ('0', 'i', 'i-pc', 'i-simple', 'ii', 'ii-pc')
R_ARMS = ('0', 'i', 'i-pc', 'i-simple', 'ii', 'ii-lag', 'ii-pc')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def planned(group):
    rows = []
    if group in ('P_PhysNet_FPm', 'P_RhythmFormer'):
        for b in (('PhysNet', 'FPm') if group == 'P_PhysNet_FPm' else ('RhythmFormer',)):
            for a in P_ARMS:
                for s in P_SEEDS:
                    rows.append((b, f'({a})', s, C / f'P_{b}_{a}_s{s}', 'A6 item 3 (P)'))
    elif group == 'R':
        for a in R_ARMS:
            for s in SEEDS:
                rows.append(('RhythmFormer', f'({a}-mcc)', s, C / f'R_RhythmFormer_{a}_s{s}', 'A6 item 9 (R)'))
    else:
        sys.exit(f'unknown group {group}')
    return rows


def main():
    group, out = sys.argv[1], Path(sys.argv[2])
    man, idx, missing = [], [], []
    plan = planned(group)
    for b, a, s, d, src in plan:
        res = d / 'result.json'
        ok = res.exists() and json.loads(res.read_text()).get('status') == 'complete' and (d / 'final_epoch.pth').is_file()
        if not ok:
            missing.append(d.name)
            continue
        man.append(dict(kind='checkpoint', backbone=BB[b], arm=a, seed=s, n='', sha256=sha(d / 'final_epoch.pth')))
        idx.append(dict(set='exploratory', backbone=BB[b], arm=a, seed=s, path=str(d / 'final_epoch.pth'), run_id=d.name, source=src))
    if missing:
        print('incomplete:', len(missing), missing[:5])
        sys.exit(1)
    assert len(man) == len(plan) and len({(r['backbone'], r['arm'], r['seed']) for r in man}) == len(plan)
    assert len({r['sha256'] for r in man}) == len(man), 'two checkpoints with the same SHA-256'
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('manifest.csv', man), ('checkpoint_index.csv', idx)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(dict(group=group, checkpoints=len(man))))


if __name__ == '__main__':
    main()
