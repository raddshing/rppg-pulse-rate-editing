"""Brief 02 7.04 라 / A6 item 5: manifest and checkpoint index of the checkpoints evaluated by guarded inference.

The 84 checkpoints of A6 items 1 and 2 and of rule 5: batch S (PhysNet-Std, arms (0), (i), (i-pc), (i-simple)), rule 5
(PhysNet-Std (ii), (ii-lag), (ii-pc)), M1 (PhysNet-MCC), M2 (PhysNet-Std-MCC), M3 (FactorizePhys-MCC), seeds 20261131-33.
Arm names carry the setting as in tools/a7_files_706.py: '(x-std)', '(x-mcc)', '(x-std-mcc)'. The retrained checkpoints of
A6 item 4 are not evaluated (tensors equal to the confirmatory ones). P (item 3) is indexed later, after its training.
A checkpoint must have result.json status complete and final_epoch.pth; otherwise the tool exits 1 and writes nothing.
Writes <out>/manifest_704.csv (kind,backbone,arm,seed,n,sha256) and <out>/checkpoint_index_704.csv
(set,backbone,arm,seed,path,run_id,source). Usage: infer_index_704.py OUT_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
T, C = WORK / 'cache' / 'train', WORK / 'cache' / 'train704'
SEEDS = (20261131, 20261132, 20261133)
SEVEN = ('0', 'i', 'i-pc', 'i-simple', 'ii', 'ii-lag', 'ii-pc')
BB = {'PhysNet': 'PhysNet', 'FPm': 'FactorizePhys-matched'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def planned():
    rows = []
    for a in ('0', 'i', 'i-pc', 'i-simple'):
        for s in SEEDS:
            rows.append(('PhysNet', f'({a}-std)', s, C / f'S_PhysNet-Std_{a}_s{s}', 'A6 item 1 (S)'))
    for a in ('ii', 'ii-lag', 'ii-pc'):
        for s in SEEDS:
            rows.append(('PhysNet', f'({a}-std)', s, T / f'PhysNet_{a}-std_s{s}', 'rule 5'))
    for a in SEVEN:
        for s in SEEDS:
            rows.append(('PhysNet', f'({a}-mcc)', s, C / f'M1_PhysNet-MCC_{a}_s{s}', 'A6 item 2 (M1)'))
            rows.append(('PhysNet', f'({a}-std-mcc)', s, C / f'M2_PhysNet-Std-MCC_{a}_s{s}', 'A6 item 2 (M2)'))
            rows.append(('FPm', f'({a}-mcc)', s, C / f'M3_FPm-MCC_{a}_s{s}', 'A6 item 2 (M3)'))
    return rows


def main():
    out = Path(sys.argv[1])
    man, idx, missing = [], [], []
    for b, a, s, d, src in planned():
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
    assert len(man) == 84 and len({(r['backbone'], r['arm'], r['seed']) for r in man}) == 84
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('manifest_704.csv', man), ('checkpoint_index_704.csv', idx)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    by = {}
    for r in idx:
        by[r['source']] = by.get(r['source'], 0) + 1
    print(json.dumps(dict(checkpoints=len(man), by_source=by)))


if __name__ == '__main__':
    main()
