"""OPS2 7.20 2: regenerated UBFC-rPPG originals (cache/b1_719/pools, train/gen_ubfc.py originals) against those of A9
item 1 (cache/ubfc709/pools manifests): per chunk, the input and label SHA-256 of the rebuilt chunk tensor and the
whole-recording check. Writes outputs/719/A10/originals_check.csv (refuses to overwrite); exit 1 if any differs."""
import csv
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[2]
OUT = W / 'outputs' / '719' / 'A10' / 'originals_check.csv'


def rd(p):
    with open(p, newline='') as f:
        return {r['chunk_id']: r for r in csv.DictReader(f)}


def main():
    if OUT.exists():
        sys.exit('exists')
    rows, ok = [], True
    for b, name in (('PhysNet', 'PhysNet'), ('RhythmFormer', 'RhythmFormer'), ('FPm', 'FactorizePhys')):
        new = rd(W / 'cache' / 'b1_719' / 'pools' / f'{b}_original_all_manifest.csv')
        old = rd(W / 'cache' / 'ubfc709' / 'pools' / f'{b}_original_all_manifest.csv')
        eq = sum(c in new and new[c]['ops2_input_sha256'] == old[c]['ops2_input_sha256']
                 and new[c]['ops2_label_sha256'] == old[c]['ops2_label_sha256'] for c in old)
        whole = sum(new[c]['load_equals_whole_recording'] == 'True' for c in new)
        rows.append(dict(backbone=name, chunks=len(old), regenerated=len(new), equal=eq, load_equals_whole_recording=whole,
                         passed=eq == len(old) == len(new) == whole))
        ok &= rows[-1]['passed']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
