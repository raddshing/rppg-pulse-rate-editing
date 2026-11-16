"""Test step (a), A4 item 4(a): compare the MMPD-B stage-1 sets regenerated from the raw data with the sets of the October
evaluation (st13b outputs), as the rehearsal compared MMPD-A (eval5/rehearsal_compare_stage1_mmpd.py). Read only.
The October files use CRLF line endings and another row order, so the comparison is by key and value, not by bytes.
1. references.csv vs st13b_reference_audit.csv, keyed by window_id: the window list, then every column (reference HRs
   h_FFT_HAR and h_FFT_L0 first). Floats are compared exactly; NaN equals NaN.
2. populations.csv vs st13b_qstar.csv, keyed by (window_id, layer): every common column.
3. defect_segments.csv vs st13b_defect_segments.csv: the set of rows.
Only counts and equality are written; no value is printed.
Usage: test_a_compare_mmpd_b.py STAGE1_MMPD_DIR OCTOBER_OUTPUTS_DIR OUT_CSV
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def same(a, b):
    """Elementwise equality of two aligned Series: floats exactly (NaN == NaN), other types as strings."""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        return (x == y) | (np.isnan(x) & np.isnan(y))
    return (a.astype(str).to_numpy() == b.astype(str).to_numpy())


def compare(new, old, key, label, rows):
    rows.append(dict(table=label, item='rows', new=len(new), october=len(old)))
    kn, ko = set(map(tuple, new[key].to_numpy())), set(map(tuple, old[key].to_numpy()))
    rows.append(dict(table=label, item='keys ' + '+'.join(key), new=len(kn), october=len(ko), equal=int(len(kn & ko)),
                     only_new=len(kn - ko), only_october=len(ko - kn), result='same' if kn == ko else 'different'))
    j = new.merge(old, on=key, how='inner', suffixes=('', '__oct'))
    cols = [c for c in new.columns if c not in key and c in old.columns]
    first = [c for c in ('h_FFT_HAR', 'h_FFT_L0', 'reference_HR') if c in cols]
    for c in first + [c for c in cols if c not in first]:
        eq = same(j[c], j[c + '__oct'])
        rows.append(dict(table=label, item=c, compared=len(j), equal=int(eq.sum()), different=int((~eq).sum()),
                         result='same' if eq.all() else 'different'))
    for c in old.columns:
        if c not in new.columns:
            rows.append(dict(table=label, item=c, result='only in October file (not compared)'))
    for c in new.columns:
        if c not in old.columns:
            rows.append(dict(table=label, item=c, result='only in new file (not compared)'))


def main():
    s1, oct_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    compare(pd.read_csv(s1 / 'references.csv'), pd.read_csv(oct_dir / 'st13b_reference_audit.csv'), ['window_id'], 'references', rows)
    compare(pd.read_csv(s1 / 'populations.csv'), pd.read_csv(oct_dir / 'st13b_qstar.csv'), ['window_id', 'layer'], 'populations', rows)
    n, o = pd.read_csv(s1 / 'defect_segments.csv'), pd.read_csv(oct_dir / 'st13b_defect_segments.csv')
    sn, so = set(map(tuple, n.astype(str).to_numpy())), set(map(tuple, o.astype(str).to_numpy()))
    rows.append(dict(table='defect_segments', item='row set', new=len(n), october=len(o), equal=len(sn & so),
                     only_new=len(sn - so), only_october=len(so - sn), result='same' if sn == so and len(n) == len(o) else 'different'))
    t = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
