"""OPS2 7.19 B4: tables of the UBFC-rPPG editor check with d = 1, in the format of B02 (tools/t709b_validity_report.py:
cells() and the October gate rule, reused unchanged).

Reads outputs/719/B/4/2u_d1 (audit/a2_validity_ubfc_d1.py) and, for comparison, outputs/2u (B02 f0/2f0 run, d = 2) and
outputs/709b/ubfc_validity_cells.csv. Writes outputs/719/B/4/:
  ubfc_validity_cells_d1.csv          every cell (mode, view, reader, band, target interval), B02 columns, d of (ii-pc)
  ubfc_validity_gate_d1.csv           October rule per mode (PhysNet view, HAR, POS/CHROM x 70-100/100-140)
  d1_vs_d2_decision_cells.csv         the four decision cells of (ii-pc) with d = 1 and d = 2 side by side
  checks.csv                          (ii) readouts equal to those of outputs/2u (d is not used by (ii)); baselines equal
"""
import sys
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'tools'))
import t709b_validity_report as T  # noqa: E402

OUT = W / 'outputs' / '719' / 'B' / '4'


def main():
    for n in ('ubfc_validity_cells_d1.csv', 'ubfc_validity_gate_d1.csv', 'd1_vs_d2_decision_cells.csv', 'checks.csv'):
        if (OUT / n).exists():
            sys.exit(f'{n} exists')
    c, r = T.cells('719/B/4/2u_d1', 'f0,2f0')
    c.insert(2, 'd_of_ii_pc', [1 if m == 'ii-pc' else '' for m in c['mode']])
    c.to_csv(OUT / 'ubfc_validity_cells_d1.csv', index=False)
    gates = []
    for mode, g in c[c.decision_cell].groupby('mode'):
        gates.append(dict(notch='f0,2f0', mode=mode, d_of_ii_pc=1 if mode == 'ii-pc' else '', cells=len(g),
                          B1_all=bool(g.B1.all()), B2_all=bool(g.B2.all()), passed=bool(g.B1.all() and g.B2.all()),
                          cells_failing_B1=';'.join(f"{x.reader} {x.range}" for x in g.itertuples() if not x.B1)))
    pd.DataFrame(gates).to_csv(OUT / 'ubfc_validity_gate_d1.csv', index=False)
    old = pd.read_csv(W / 'outputs' / '709b' / 'ubfc_validity_cells.csv')
    old = old[old.notch.eq('f0,2f0') & old.decision_cell]
    new = c[c.decision_cell]
    key = ['mode', 'model_view', 'reader', 'layer', 'range']
    j = new.merge(old, on=key, suffixes=('_d1', '_d2'))
    cols = key + ['windows_d1', 'unedited_hit_rate_d1', 'target_hit_rate_d1', 'target_hit_rate_d2', 'original_hit_after_rate_d1',
                  'original_hit_after_rate_d2', 'B1_d1', 'B1_d2', 'B2_d1', 'B2_d2']
    j[cols].to_csv(OUT / 'd1_vs_d2_decision_cells.csv', index=False)
    checks = []
    o = pd.read_csv(W / 'outputs' / '2u' / 'validity360_ubfc_readouts.csv')
    k = ['window_id', 'model_view', 'reader', 'layer', 'range_low', 'mode']
    for mode in ('ii', 'ii-pc'):
        a, b = r[r['mode'].eq(mode)], o[o['mode'].eq(mode)]
        m = a.merge(b, on=k, suffixes=('', '_d2'))
        eq = ((m.edited_HR - m.edited_HR_d2).abs() < 1e-9) | (m.edited_HR.isna() & m.edited_HR_d2.isna())
        checks.append(dict(check=f'({mode}) readouts: rows matched, edited HR equal to the d = 2 run (outputs/2u)',
                           rows=len(a), matched=len(m), equal=int(eq.sum()),
                           expected='all equal' if mode == 'ii' else 'not required (d differs)', passed=bool(eq.all()) if mode == 'ii' else ''))
    bn = pd.read_csv(OUT / '2u_d1' / 'validity360_ubfc_baselines.csv')
    bo = pd.read_csv(W / 'outputs' / '2u' / 'validity360_ubfc_baselines.csv')
    checks.append(dict(check='baselines (unedited readouts) identical to outputs/2u', rows=len(bn), matched=len(bo),
                       equal=int(bn.equals(bo)), expected='identical', passed=bool(bn.equals(bo))))
    pd.DataFrame(checks).to_csv(OUT / 'checks.csv', index=False)
    print(pd.DataFrame(gates).to_string())
    print(j[cols].to_string())
    print(pd.DataFrame(checks).to_string())


if __name__ == '__main__':
    main()
