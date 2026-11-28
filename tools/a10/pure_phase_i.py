"""OPS2 7.20 2: the (i) phase-error measure of the A10 design table, |wrap(2 pi (tau - d)(r - 1) / T0)|, T0 = 1800 / h0,
r = h* / h0, on the first PURE pool (manifests only): the (i-pc) items of cache/pools_i/<backbone>_i_d_all_manifest.csv
(target HR, reference HR, tau_rec of the recording, d = 2 frames), PhysNet list (2,972 items) and RhythmFormer list
(2,384 items, the same items and targets as FactorizePhys). Mean and median, all items and by target interval.
Writes outputs/719/A10/pure_first_pool_phase_i.csv (refuses to overwrite)."""
import csv
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'tools' / 'a10'))
import b1_design as D  # noqa: E402

OUT = W / 'outputs' / '719' / 'A10' / 'pure_first_pool_phase_i.csv'


def load(b):
    rs = [r for r in csv.DictReader(open(W / 'cache' / 'pools_i' / f'{b}_i_d_all_manifest.csv'))
          if str(r.get('generation_excluded', 'False')) != 'True']
    assert all(r['d'] == '2' for r in rs)
    return rs


def main():
    if OUT.exists():
        sys.exit('exists')
    rf, fp = load('RhythmFormer'), load('FPm')
    key = lambda r: (r['chunk_id'], r['augmentation_index'], r['target_HR'], r['reference_HR'], r['tau_rec'])
    same = sorted(map(key, rf)) == sorted(map(key, fp))
    rows = []
    for name, rs in (('PhysNet', load('PhysNet')), ('RhythmFormer/FactorizePhys', rf)):
        h, h0, tau = (np.array([float(r[c]) for r in rs]) for c in ('target_HR', 'reference_HR', 'tau_rec'))
        ph = D.phase_deg(h, h0, tau, d=2)
        for lab, lo, hi in [('all', 0, 1e9)] + D.BINS:
            m = (h >= lo) & (h < hi)
            rows.append(dict(corpus='PURE first pool', list=name, d=2, targets=lab, items=int(m.sum()),
                             mean_phase_error_deg=float(ph[m].mean()), median_phase_error_deg=float(np.median(ph[m])),
                             rf_fpm_items_identical=same))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
