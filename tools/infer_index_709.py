"""7.09d / A9 item 7: manifest and checkpoint index of the A9 checkpoints for guarded inference (as tools/infer_index_704p.py).

Groups:
  E467  E4 (os): PhysNet, FactorizePhys, RhythmFormer x seeds 20261131-33 (cache/train709/E4_<b>_0-os_s<seed>), arm '(os)';
        E6: RhythmFormer (ii-pc) on the second pool, seeds 20261131-33 (E6_RhythmFormer_ii-pc_s<seed>), arm '(ii-pc-pool2)';
        E7: FactorizePhys (ii-alpha), alpha 0.25, 0.5, 0.75 x seeds 20261131-33 (E7_FPm_lagfrac<alpha>_s<seed>),
        arms '(ii-a0.25)', '(ii-a0.5)', '(ii-a0.75)'.  21 checkpoints.
  E1    UBFC-rPPG training: arms (0), (i), (i-pc), (i-simple), (ii), (ii-pc), (ii-lag) x 3 backbones x seeds 20261131-33
        (U_<b>_<arm>_s<seed>), arms named '(<arm>-ubfc)'.  63 checkpoints.
The arm names keep the keys distinct from the confirmatory checkpoints with the same seeds. A checkpoint must have
result.json status complete and final_epoch.pth; otherwise the tool exits 1 and writes nothing.
Writes <OUT>/manifest.csv (kind,backbone,arm,seed,n,sha256) and <OUT>/checkpoint_index.csv (set,backbone,arm,seed,path,
run_id,source). Usage: infer_index_709.py GROUP OUT_DIR
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

C = Path(__file__).resolve().parents[1] / 'cache' / 'train709'
BB = {'PhysNet': 'PhysNet', 'FPm': 'FactorizePhys-matched', 'RhythmFormer': 'RhythmFormer'}
SEEDS = (20261131, 20261132, 20261133)
E1_ARMS = ('0', 'i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def planned(group):
    rows = []
    if group == 'E467':
        for b in ('PhysNet', 'FPm', 'RhythmFormer'):
            for s in SEEDS:
                rows.append((b, '(os)', s, C / f'E4_{b}_0-os_s{s}', 'A9 item 4 (E4)'))
        for s in SEEDS:
            rows.append(('RhythmFormer', '(ii-pc-pool2)', s, C / f'E6_RhythmFormer_ii-pc_s{s}', 'A9 item 5 (E6)'))
        for x in ('0.25', '0.5', '0.75'):
            for s in SEEDS:
                rows.append(('FPm', f'(ii-a{x})', s, C / f'E7_FPm_lagfrac{x}_s{s}', 'A9 item 6 (E7)'))
    elif group == 'E1':
        for b in ('PhysNet', 'FPm', 'RhythmFormer'):
            for a in E1_ARMS:
                for s in SEEDS:
                    rows.append((b, f'({a}-ubfc)', s, C / f'U_{b}_{a}_s{s}', 'A9 item 1 (E1)'))
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
