"""Amendment A5 (ACCESS_OPS2_BRIEF_02 6.78): tables of the stored October predictions for the comparison between test
stages (b) and (c), in the format of eval5/stage3_check.py --october-mmpd / --october-mcd (backbone, arm, seed, window_id,
layer, predicted_HR; the column names, backbone names and layer values of the rehearsal tables).

Checkpoints: arms (0), (i), (ii) of docs/PREREGISTRATION_STAGE5_MANIFEST.csv (27; three backbones x three arms x three seeds).
Stored October waveforms (prediction_waveform .npz; only the arrays 'pred' and 'diff_flag' and the October grid length
'uniform_grid_samples' are used):
  PhysNet, RhythmFormer  outputs/st13_sealed_predictions_manifest.csv (only the columns dataset, model, recording_id,
                         file_kind, path and checkpoint_sha256 are read; rows are matched to the manifest checkpoints by
                         checkpoint_sha256)
  FactorizePhys-matched  the October ST6 evaluation (bulk st30e/<run>/B, /M), runs of b1/roster.py; their checkpoint files
                         must have the manifest SHA-256 (these waveforms are not in the st13 manifest)
No October window-HR file, reference signal or reference HR is read, and no value of the tables is printed: only row counts,
missing counts and SHA-256.
Windows and HR readout are those of eval5/stage3_check.py for the new predictions (mmpd_job, mcd_job and read_hr, imported
unchanged; each stored 'pred' array is written unchanged to a temporary .npy in the layout these functions read):
  MMPD   windows j = 0..4 of every recording (window_id MMPD|<recording>|<j>), samples 300j..300j+300 of the waveform cast
         to float64; wide band (HAR) and narrow band (L0).
  MCD    window j starts at 10 j s on the uniform 30 Hz grid of the October inference ('uniform_grid_samples' points;
         pathways.uniform_frame_grid / complete_windows inside mcd_job) and needs 300 samples; window_id <recording>|<j>
         for j = 0 .. floor(grid / 300), i.e. every window that can lie on the grid plus one more (missing). The stage-1
         window list of the test half (which needs the contact signals) is a subset; extra windows are not compared.
  Recordings: MMPD-B, all recordings of the stored October predictions; MCD-rPPG, the stored recordings of the subjects
  of the test-half list of the manifest (row mcd_test_subjects; the list file must have its SHA-256).
Row order: checkpoints in b1/roster.py order, recordings sorted, layer (HAR, L0), window index.

Modes (roots of the October project and of its bulk store are arguments; no server path is written to any output):
  sources OUT.txt                    SHA-256 of every stored file that 'test' reads (relative names: outputs/..., bulk/...)
  test MMPD.csv MCD.csv SOURCES.txt  checks every source against SOURCES.txt, then writes the two tables
  dev-mmpd-a OUT.csv                 development check 1: MMPD-A from the October draft waveforms read by b1/mmpd_a.py
                                     (windows of the same 220 recordings; must reproduce the rehearsal table byte for byte)
  dev-mcd OUT.csv                    development check 2: the stored October MCD-rPPG waveforms of development-half subjects
Usage: october_test_tables.py MODE ... --october-root DIR --bulk-root DIR [--subjects LIST] [--workers N]
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'eval5'))
sys.path.insert(0, str(WORK / 'b1'))
import stage3_check as S3  # noqa: E402  (window construction and HR readout of the new predictions)
from roster import roster  # noqa: E402

DOCS = WORK / 'docs' if (WORK / 'docs').is_dir() else WORK / 'pub' / 'docs'
VENDOR = WORK / 'vendor' / 'st8_6ebbc9cd'
ARMS = ('(0)', '(i)', '(ii)')
NAME = {'FactorizePhys (matched)': 'FactorizePhys-matched', 'PhysNet': 'PhysNet', 'RhythmFormer': 'RhythmFormer'}
LAYERS = list(S3.BANDS)
COLUMNS = ['backbone', 'arm', 'seed', 'window_id', 'layer', 'predicted_HR']
ST13 = 'outputs/st13_sealed_predictions_manifest.csv'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def checkpoints():
    """(backbone, arm, seed) -> manifest SHA-256, in b1/roster.py order, with the roster run."""
    man = pd.read_csv(DOCS / 'PREREGISTRATION_STAGE5_MANIFEST.csv')
    man = man[man.kind.eq('checkpoint') & man.arm.isin(ARMS)]
    want = {(r.backbone, r.arm, int(r.seed)): r.sha256 for r in man.itertuples()}
    out = []
    for r in roster():
        k = (NAME.get(r['backbone']), r['strategy'], int(r['seed']))
        if k in want:
            out.append(dict(backbone=k[0], arm=k[1], seed=k[2], sha256=want[k], run=r))
    assert len(out) == len(want) == 27, (len(out), len(want))
    return out


def subjects(path, row):
    """Subject list file of the manifest row (mcd_test_subjects / mcd_dev_subjects), checked against its SHA-256."""
    man = pd.read_csv(DOCS / 'PREREGISTRATION_STAGE5_MANIFEST.csv')
    expect = man[man.kind.eq(row)].sha256.iloc[0]
    assert sha(path) == expect, f'{path} is not the {row} list of the manifest'
    return set(Path(path).read_text().split())


def rel(path, a):
    """Relative name of a stored file (outputs/... under the October project, bulk/... under its bulk store)."""
    p = Path(path)
    for label, root in (('bulk', a.bulk_root), ('', a.october_root)):
        try:
            r = p.relative_to(root)
            return f'{label}/{r}' if label else str(r)
        except ValueError:
            continue
    raise ValueError(f'{path} is outside the October roots')


def plan(a, mode):
    """Per checkpoint: {'MMPD': {recording: npz}, 'MCD': {recording: npz}} and the files read for the selection."""
    ck = checkpoints()
    read = []
    if mode == 'dev-mmpd-a':
        ref = pd.read_csv(VENDOR / 'outputs' / 'st30a_new_A_reference.csv', usecols=['recording_id'])  # recording IDs only
        recs = sorted(ref.recording_id.unique())
        for c in ck:
            d = draft_a_dir(c['run'], a)
            c['MMPD'] = {r: d / (r.replace('/', '__') + '.npz') for r in recs}
            c['MCD'] = {}
        return ck, read
    keep = subjects(a.subjects, 'mcd_test_subjects' if mode in ('test', 'sources') else 'mcd_dev_subjects')
    st13 = Path(a.october_root) / ST13
    read.append(st13)
    m = pd.read_csv(st13, usecols=['dataset', 'model', 'recording_id', 'file_kind', 'path', 'checkpoint_sha256'])
    m = m[m.file_kind.eq('prediction_waveform')]
    b_recs = sorted(m[m.dataset.eq('B')].recording_id.unique())
    m_recs = sorted(r for r in m[m.dataset.eq('M')].recording_id.unique() if r.split('_')[0] in keep)
    for c in ck:
        if c['backbone'] == 'FactorizePhys-matched':  # October ST6 evaluation (not in the st13 manifest)
            cp = Path(c['run']['checkpoint'])
            read.append(cp)
            assert sha(cp) == c['sha256'], f'checkpoint {cp.name} of {c["run"]["run_id"]} is not the manifest checkpoint'
            base = Path(a.bulk_root) / 'st30e' / c['run']['run_id']
            c['MMPD'] = {r: base / 'B' / (r.replace('/', '__') + '.npz') for r in b_recs}
            c['MCD'] = {r: base / 'M' / (r + '.npz') for r in m_recs}
        else:
            s = m[m.checkpoint_sha256.eq(c['sha256'])]
            assert set(s.model.str.lower()) == {c['backbone'].lower()}, (c['backbone'], set(s.model))
            c['MMPD'] = {r.recording_id: Path(r.path) for r in s[s.dataset.eq('B')].itertuples()}
            c['MCD'] = {r.recording_id: Path(r.path) for r in s[s.dataset.eq('M')].itertuples() if r.recording_id in m_recs}
        if mode == 'dev-mcd':
            c['MMPD'] = {}
        else:
            assert sorted(c['MMPD']) == b_recs, f'{c["backbone"]} {c["arm"]} {c["seed"]}: MMPD-B recordings differ'
        assert sorted(c['MCD']) == m_recs, f'{c["backbone"]} {c["arm"]} {c["seed"]}: MCD-rPPG recordings differ'
        for f in list(c['MMPD'].values()) + list(c['MCD'].values()):
            assert f.is_file(), f'missing {f.name}'
    return ck, read


def draft_a_dir(run, a):
    """October draft MMPD-A waveform folder of an original-arm run (as b1/mmpd_a.py wave_dir)."""
    rid, ns = run['run_id'], run['namespace']
    out = Path(a.october_root) / 'outputs'
    if ns == 'st9t':
        return sorted((out / 'st12e_waveforms' / rid.replace('st9t_', 'st12e_')).glob('attempt*'))[-1]
    if ns in ('st12c', 'st13x'):
        return sorted((out / f'{ns}_waveforms' / rid.replace(ns + '_', ns + '_eval_', 1)).glob('attempt*'))[-1]
    if ns == 'st30t':
        return Path(a.bulk_root) / 'st30e' / rid / 'A'
    raise ValueError(f'no draft MMPD-A folder rule for {rid}')


def job(spec):
    """One checkpoint and dataset: stored 'pred' arrays to temporary .npy files, then stage3_check's job."""
    c, ds, tmp_root = spec
    files = c[ds]
    tmp = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        diffs, grids = set(), {}
        for r, f in files.items():
            with np.load(f) as z:
                pred = z['pred']
                diffs.add(bool(z['diff_flag']))
                if ds == 'MCD':
                    grids[r] = int(z['uniform_grid_samples'])
                np.save(tmp / ((r.replace('/', '__') if ds == 'MMPD' else r) + '.npy'), pred)  # unchanged dtype and values
        assert len(diffs) == 1, 'diff_flag differs between recordings of one checkpoint'
        diff = diffs.pop()
        assert diff == (c['backbone'] == 'PhysNet'), f'stored diff_flag {diff} is not the stage3_check rule'
        key = f'{c["backbone"]}|{c["arm"]}|{c["seed"]}'
        if ds == 'MMPD':
            windows = [(r, j, f'MMPD|{r}|{j}') for r in sorted(files) for j in range(5)]
            res = S3.mmpd_job((tmp, key, diff, windows))
        else:
            recs = []
            for r in sorted(files):
                g = grids[r]  # October grid length: n = g frames over span (g - 0.5) / 30 gives a grid of g points
                wins = [(f'{r}|{j}', layer, 10.0 * j) for j in range(g // 300 + 1) for layer in LAYERS]
                recs.append((r, g, (g - 0.5) / 30, wins))
            res = S3.mcd_job((tmp, key, diff, recs))
        rows = [dict(backbone=c['backbone'], arm=c['arm'], seed=c['seed'], window_id=w, layer=layer, predicted_HR=h)
                for _, w, layer, h, _ in res]
        return ds, c['order'], rows
    finally:
        shutil.rmtree(tmp)


def tables(ck, a, datasets):
    tmp_root = Path(a.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(ck):
        c['order'] = i
    specs = [({k: c[k] for k in ('backbone', 'arm', 'seed', 'order', ds)}, ds, str(tmp_root)) for ds in datasets for c in ck]
    out = {ds: {} for ds in datasets}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=S3.init,
                             initargs=(str(VENDOR),)) as ex:
        for n, (ds, order, rows) in enumerate(ex.map(job, specs), 1):
            out[ds][order] = rows
            print(f'{ds} {n}/{len(specs)}', flush=True)
    result = {}
    for ds in datasets:
        d = pd.DataFrame([r for o in sorted(out[ds]) for r in out[ds][o]], columns=COLUMNS)
        d['j'] = d.window_id.str.rsplit('|', n=1).str[1].astype(int)
        d['rec'] = d.window_id.str.rsplit('|', n=1).str[0]
        d['li'] = d.layer.map({x: i for i, x in enumerate(LAYERS)})
        d['o'] = d.groupby(['backbone', 'arm', 'seed'], sort=False).ngroup()
        d = d.sort_values(['o', 'rec', 'li', 'j'], kind='mergesort')[COLUMNS].reset_index(drop=True)
        result[ds] = d
    return result


def summary(name, d, path):
    g = d.groupby(['backbone', 'arm', 'seed'], sort=False).agg(rows=('predicted_HR', 'size'),
                                                               missing=('predicted_HR', lambda s: int(s.isna().sum())))
    print(f'{name}: rows {len(d)}, missing {int(d.predicted_HR.isna().sum())}, windows {d.window_id.nunique()}, '
          f'checkpoints {len(g)}, SHA-256 {sha(path)}')
    print(g.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sources', 'test', 'dev-mmpd-a', 'dev-mcd'])
    ap.add_argument('outputs', nargs='+')
    ap.add_argument('--october-root', required=True), ap.add_argument('--bulk-root', required=True)
    ap.add_argument('--subjects', help='MCD-rPPG subject list (test half for sources/test, development half for dev-mcd)')
    ap.add_argument('--workers', type=int, default=3), ap.add_argument('--tmp', default=str(WORK / 'cache' / 'october_tables_tmp'))
    a = ap.parse_args()
    a.october_root, a.bulk_root = str(Path(a.october_root).resolve()), str(Path(a.bulk_root).resolve())
    S3.init(str(VENDOR))
    ck, read = plan(a, a.mode)
    files = read + [f for c in ck for ds in ('MMPD', 'MCD') for f in c[ds].values()]
    if a.mode == 'sources':
        lines = sorted(f'{sha(f)}  {rel(f, a)}' for f in files)
        Path(a.outputs[0]).write_text('\n'.join(lines) + '\n')
        print(f'sources: {len(lines)} files ({len(read)} manifest/checkpoint, {len(files) - len(read)} waveforms)')
        return
    if a.mode == 'test':
        listed = dict(reversed(x.split('  ', 1)) for x in Path(a.outputs[2]).read_text().splitlines())
        here = {rel(f, a): f for f in files}
        assert set(here) == set(listed), f'files to read differ from the source list: {len(set(here) ^ set(listed))}'
        bad = [r for r, f in here.items() if sha(f) != listed[r]]
        assert not bad, f'{len(bad)} sources differ from the source list, e.g. {bad[:3]}'
        print(f'sources checked: {len(here)} files equal the source list')
        t = tables(ck, a, ['MMPD', 'MCD'])
        for ds, path in zip(('MMPD', 'MCD'), a.outputs[:2]):
            t[ds].to_csv(path, index=False)
            summary(f'October {ds} test table', t[ds], path)
        return
    ds = 'MMPD' if a.mode == 'dev-mmpd-a' else 'MCD'
    t = tables(ck, a, [ds])[ds]
    t.to_csv(a.outputs[0], index=False)
    summary(f'{a.mode} table', t, a.outputs[0])


if __name__ == '__main__':
    main()
