"""7.09b 가-4: the two UBFC 360-window checks with the decision rule of the October UBFC gate (src/st31u/gate.py).

October rule: the four PhysNet-view HAR cells (POS, CHROM) x (70-100, 100-140 bpm) must all have B1 (target-hit rate >=
unedited hit rate - 0.12) and B2 (native-HR hit rate after editing <= unedited rate / 2). L0 cells are applicable only
with >= 30 windows. Reads outputs/2u (notch f0, 2f0) and outputs/2u_target (notch f0, 2f0, f*, 2f*); writes
outputs/709b/ubfc_validity_cells.csv (all cells, both settings, both modes) and ubfc_validity_gate.csv, and compares the
(ii) readouts of the PhysNet and RhythmFormer views of the f0/2f0 run with the October G-U readouts.
"""
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parents[1] / 'outputs'
import json  # 7.19
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
OCT = Path(CFG['october_project']) / 'outputs'


def cells(folder, notch):
    r = pd.read_csv(W / folder / 'validity360_ubfc_readouts.csv')
    b = pd.read_csv(W / folder / 'validity360_ubfc_baselines.csv').drop_duplicates(['window_id', 'model_view', 'reader', 'layer'])
    base = b.groupby(['model_view', 'reader', 'layer']).baseline_hit.mean()
    rows = []
    for (mode, view, reader, layer, lo, hi), g in r.groupby(['mode', 'model_view', 'reader', 'layer', 'range_low', 'range_high']):
        rate = float(base.loc[(view, reader, layer)])
        t, o, n = float(g.target_hit.mean()), float(g.original_hit_after.mean()), len(g)
        applicable = layer == 'HAR' or n >= 30
        rows.append(dict(notch=notch, mode=mode, model_view=view, reader=reader, layer=layer, range=f'{lo}-{hi}', windows=n,
                         unedited_hit_rate=rate, target_hit_rate=t, original_hit_after_rate=o, applicable=applicable,
                         B1=(t >= rate - .12) if applicable else '', B2=(o <= rate / 2) if applicable else '',
                         decision_cell=view == 'PhysNet' and layer == 'HAR' and lo in (70, 100),
                         nonfinite_edited=int((~pd.to_numeric(g.edited_HR, errors='coerce').notna()).sum())))
    return pd.DataFrame(rows), r


def main():
    out = W / '709b'
    out.mkdir(exist_ok=True)
    parts, gates = [], []
    readouts = {}
    for folder, notch in (('2u', 'f0,2f0'), ('2u_target', 'f0,2f0,f*,2f*')):
        if not (W / folder / 'validity360_ubfc_readouts.csv').exists():
            continue
        c, r = cells(folder, notch)
        readouts[notch] = r
        parts.append(c)
        for mode, g in c[c.decision_cell].groupby('mode'):
            gates.append(dict(notch=notch, mode=mode, cells=len(g), B1_all=bool(g.B1.all()), B2_all=bool(g.B2.all()),
                              passed=bool(g.B1.all() and g.B2.all()),
                              cells_failing_B1=';'.join(f"{x.reader} {x.range}" for x in g.itertuples() if not x.B1)))
    allc = pd.concat(parts, ignore_index=True)
    allc.to_csv(out / 'ubfc_validity_cells.csv', index=False)
    pd.DataFrame(gates).to_csv(out / 'ubfc_validity_gate.csv', index=False)
    print(pd.DataFrame(gates).to_string())
    show = allc[allc.layer.eq('HAR') & allc.model_view.eq('PhysNet')]
    print(show.pivot_table(index=['reader', 'range'], columns=['notch', 'mode'], values='target_hit_rate').round(3).to_string())
    if 'f0,2f0' in readouts:
        g = pd.read_csv(OCT / 'st31u_GU_readouts.csv')
        r = readouts['f0,2f0']
        j = r[r['mode'].eq('ii')].merge(g, on=['window_id', 'model_view', 'reader', 'layer', 'range_low'], suffixes=('', '_oct'))
        same = ((j.edited_HR - j.edited_HR_oct).abs() < 1e-9) & ((j.baseline_HR - j.baseline_HR_oct).abs() < 1e-9)
        print('October G-U rows', len(g), 'matched', len(j), 'equal edited and baseline HR', int(same.sum()))
        pd.DataFrame([dict(october_rows=len(g), matched=len(j), equal=int(same.sum()))]).to_csv(out / 'ubfc_validity_vs_october.csv', index=False)


if __name__ == '__main__':
    main()
