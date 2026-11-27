"""OPS2 7.17 4: HR readouts of the E1_PhysNet MCD-test repeat passes, run only after the prediction SHA commit and the
evaluation (A9 item 7). Key PhysNet__ii-pc-ubfc__s20261131__std; three prediction folders: the main pass (used by the
evaluation), the first repeat (moved aside; one file differed), the repeat inferred again (7.17 1). The readout is that of
step (b): eval5/stage3_check.py mcd_job (float32 as stored, uniform 30 Hz frame grid, complete windows, 512-point
periodogram peak, HAR and L0 bands), on every population window of the test-run stage 1, layer of the reference row.
Checks: the main-pass readouts recomputed here equal the evaluation's step-(b) readouts of this key. Writes the aggregate
outputs/717/repeat_readouts.csv (no identifiers): per comparison (first repeat vs main, repeat again vs main), scope (the
recording whose first-repeat file differed; all recordings), layer: windows, windows valid in both, windows with equal
valid flag, windows with equal predicted HR (or both missing), largest absolute HR difference. Usage (main env):
repeat_readouts_717.py
"""
import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'eval5'))
import stage3_check as S3  # noqa: E402

KEY = 'PhysNet__ii-pc-ubfc__s20261131__std'
I = W / 'cache' / 'infer709' / 'E1_PhysNet'
F = {'main': I / 'MCD_test' / 'MCD', 'first repeat': I / 'det_MCD_test_first_20261127' / 'MCD', 'repeat again': I / 'det_MCD_test' / 'MCD'}
S1 = W / 'cache' / 'test' / 'stage1' / 'MCD'
EVAL = W / 'cache' / 'a9_eval' / 'E1_PhysNet' / 'test_check' / 'test' / 'check' / 'MCD' / 'readouts.csv.gz'
OUT = W / 'outputs' / '717' / 'repeat_readouts.csv'


def main():
    assert not OUT.exists(), OUT
    S3.init(str(W / 'vendor' / 'st8_6ebbc9cd'))
    pops = pd.read_csv(S1 / 'populations.csv', dtype={'subject_id': str})
    refs = pd.read_csv(S1 / 'references.csv', dtype={'subject_id': str})
    refs = refs[refs.window_id.isin(set(pops.window_id))]
    lay = pops[['window_id', 'layer']].drop_duplicates()
    ck = pd.read_csv(F['main'] / 'checkpoints.csv').set_index('key').loc[KEY]
    diff = ck.view == 'PhysNet' and '-std' not in ck.arm
    rd = {}
    for name, fo in F.items():
        rr = pd.read_csv(fo / 'recordings.csv').set_index('recording_id')
        items = [(rid, int(rr.loc[rid, 'decoded_frames']), float(rr.loc[rid, 'span_seconds']),
                  [(x.window_id, x.layer, float(x.start_seconds)) for x in g.itertuples()]) for rid, g in refs.groupby('recording_id')]
        d = pd.DataFrame(S3.mcd_job((fo / 'pred' / KEY, KEY, diff, items)), columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])
        rd[name] = d.merge(lay, on=['window_id', 'layer'])
        print(name, 'rows', len(rd[name]), flush=True)
    ev = pd.read_csv(EVAL)
    ev = ev[ev.key.eq(KEY)]
    m = ev.merge(rd['main'], on=['key', 'window_id', 'layer'], suffixes=('_eval', '_here'))
    same = ((m.predicted_HR_eval == m.predicted_HR_here) | (m.predicted_HR_eval.isna() & m.predicted_HR_here.isna())) & (m.valid_eval == m.valid_here)
    print('main recomputed vs evaluation: rows', len(ev), 'matched', len(m), 'equal', int(same.sum()), flush=True)
    assert len(m) == len(ev) == len(rd['main']) and same.all()
    kd = F['main'] / 'pred' / KEY
    bad = [p.stem for p in sorted((F['first repeat'] / 'pred' / KEY).iterdir()) if not filecmp.cmp(p, kd / p.name, shallow=False)]
    assert len(bad) == 1, bad
    rows = [dict(comparison='main pass recomputed here vs evaluation step (b)', scope='all recordings', layer='all',
                 windows=len(ev), windows_valid_both=int((m.valid_eval & m.valid_here).sum()), windows_valid_equal=int((m.valid_eval == m.valid_here).sum()),
                 windows_HR_equal=int(same.sum()), max_abs_HR_difference_bpm=0.0)]
    for other in ('first repeat', 'repeat again'):
        x = rd['main'].merge(rd[other], on=['key', 'window_id', 'layer'], suffixes=('_m', '_o'))
        x['rec'] = x.window_id.astype(str).str.split('|').str[0]
        for scope, sub in (('recording whose first-repeat file differed', x[x.rec.eq(bad[0])]), ('all recordings', x)):
            for layer, g in [('all', sub)] + list(sub.groupby('layer')):
                eq = (g.predicted_HR_m == g.predicted_HR_o) | (g.predicted_HR_m.isna() & g.predicted_HR_o.isna())
                dd = (g.predicted_HR_m - g.predicted_HR_o).abs()
                rows.append(dict(comparison=f'{other} vs main pass', scope=scope, layer=layer, windows=len(g),
                                 windows_valid_both=int((g.valid_m & g.valid_o).sum()), windows_valid_equal=int((g.valid_m == g.valid_o).sum()),
                                 windows_HR_equal=int(eq.sum()), max_abs_HR_difference_bpm=float(dd.max()) if dd.notna().any() else np.nan))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == '__main__':
    main()
