"""Brief 02 7.01 라-2: RhythmFormer (0B) (author built-in augmentation), October checkpoints and October stored predictions,
read again under the current window rule (no inference, no re-inference).

(0B) predictions: the stored October waveforms of the three October runs st30t_RhythmFormer_PURE_0B_s2026113{1,2,3}
(bulk st30e/<run>/B and /M .npz; arrays 'pred' and 'diff_flag', and the identifying fields; their checkpoint SHA-256 is
checked against the October training results). These predictions were made before the guards of the stage-5 test run.
MCD-rPPG: only the stored recordings of test-half subjects (manifest list mcd_test_subjects, SHA-256 checked).
Window rule and readout: those of the test run (eval5/stage3_check.py read_hr, unchanged): MMPD-B windows j of the stage-1
population (samples 300j..300j+300, waveform cast to float64); MCD-rPPG test windows on the uniform 30 Hz grid of the
endpoint-linear frame times of the test recording (decoded frames and span of the test stage 2, start of the stage-1
reference window; complete_windows), waveform float32 as stored. A window is valid if the waveform covers it and the readout
is finite. As a check, MCD readouts are also made on the October grid ('uniform_grid_samples' points, as
tools/october_test_tables.py) and compared.
Populations, support, metrics and contrasts: the test analysis (eval5/stage3_analyze.py load/restrict/pair_windows/
arm_rows/contrast_rows, b1/boot.py crossed bootstrap, B = 10,000, default_rng(20261110)), primary HAR, Q* (MMPD-B) and draft
rule (MCD-rPPG test half); (0B) enters as an exploratory key (confirmatory support and its own valid windows). Contrasts:
(0B)-(0), (ii-pc)-(0B), (i-simple)-(0B) (same seed, windows valid for every key of both arms), values only.
Recording-level scores: eval5/benchmark.py recording_row (whole waveform, narrow band) with the stage-1 contact signals of the
test run; the other arms from the test step (c) benchmark_recordings.csv; per key MAE, RMSE, Pearson r, MAE of recordings
with reference >= 100; seed mean and sample SD; on each key's own recordings and on the recordings scored for (0B) as well.
Writes outputs/posthoc701/ra2/{ra2_settings,ra2_coverage,ra2_arms,ra2_arms_common,ra2_contrasts,ra2_benchmark_per_key,
ra2_benchmark,ra2_checks}.csv.
"""
import json
import time
from pathlib import Path

from p701_common import (B, SA, SEEDS, SRC_OCT, SRC_TEST, TEST, VENDOR, WORK, keys, np, pd, primary_har, stamp, write)
import stage3_check as S3  # noqa: E402
import benchmark  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parents[2] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
BULK = Path(CFG['october_bulk']) / 'st30e'
OCT_RESULTS = Path(CFG['october_project']) / 'reports' / 'st30t'
RUN = 'st30t_RhythmFormer_PURE_0B_s{}'
ARMS = ['(0B)', '(0)', '(ii-pc)', '(i-simple)']
CONTRASTS = [('(0B)', '(0)'), ('(ii-pc)', '(0B)'), ('(i-simple)', '(0B)')]
OKEY = 'RhythmFormer|(0B)|{}'


def october(ds_letter, seed, keep=None):
    run = RUN.format(seed)
    expect = json.loads((OCT_RESULTS / f'{run}_result.json').read_text())['checkpoint_sha256']
    out = {}
    for f in sorted((BULK / run / ds_letter).glob('*.npz')):
        with np.load(f) as z:
            rid = str(z['recording_id'])
            if keep is not None and rid.split('_')[0] not in keep:
                continue
            assert str(z['checkpoint_sha256']) == expect and str(z['run_id']) == run and not bool(z['diff_flag'])
            out[rid] = (z['pred'].copy(), int(z['uniform_grid_samples']))
    return out, expect


def readouts_mmpd(preds, key, pops):
    rows = []
    for x in pops.drop_duplicates('window_id').itertuples():
        p = preds.get(x.recording_id)
        j = int(x.window_index)
        for layer in S3.BANDS:
            if p is None or 300 * (j + 1) > len(p[0]):
                rows.append((key, x.window_id, layer, np.nan, False))
                continue
            h = S3.read_hr(p[0].astype(np.float64)[300 * j:300 * (j + 1)], False, layer)
            rows.append((key, x.window_id, layer, h, bool(np.isfinite(h))))
    return pd.DataFrame(rows, columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])


def readouts_mcd(preds, key, refs, recs, grid='test'):
    from src.st11x.pathways import uniform_frame_grid, complete_windows
    rows = []
    for rid, g in refs.groupby('recording_id'):
        p = preds.get(rid)
        if grid == 'test':
            n, span = int(recs.loc[rid, 'decoded_frames']), float(recs.loc[rid, 'span_seconds'])
        elif p is not None:
            n, span = p[1], (p[1] - 0.5) / 30
        if p is not None or grid == 'test':
            fg = uniform_frame_grid(np.arange(n, dtype=np.float64) * span / (n - 1))
        for x in g.itertuples():
            if p is None:
                rows.append((key, x.window_id, x.layer, np.nan, False))
                continue
            bd = complete_windows(fg, [float(x.start_seconds)])[0]
            s = bd['grid_start']
            if not bd['available'] or s + 300 > len(p[0]):
                rows.append((key, x.window_id, x.layer, np.nan, False))
                continue
            h = S3.read_hr(p[0][s:s + 300], False, x.layer)
            rows.append((key, x.window_id, x.layer, h, bool(np.isfinite(h))))
    return pd.DataFrame(rows, columns=['key', 'window_id', 'layer', 'predicted_HR', 'valid'])


def load_with(dataset, extra_rd, extra_keys):
    """eval5/stage3_analyze.py load (population 'automatic') with extra readouts and keys added."""
    s1, chk = TEST / 'stage1' / dataset, TEST / 'check' / dataset
    pops = pd.read_csv(s1 / 'populations.csv', dtype={'subject_id': str})
    rd = pd.concat([pd.read_csv(chk / 'readouts.csv.gz'), extra_rd], ignore_index=True)
    sup = pd.read_csv(chk / 'support.csv', dtype={'subject_id': str})
    bad = set(pd.read_csv(chk / 'defects.csv').key.dropna())
    support = set(map(tuple, sup[sup.in_support][['window_id', 'layer']].to_numpy()))
    if dataset == 'MMPD':
        pops = pops[pops.in_Qstar.astype(bool)].copy()
        pops['set'] = np.where(pops.high.astype(bool), 'high', 'low')
        pops['population'] = 'Qstar'
    else:
        refs = pd.read_csv(s1 / 'references.csv', dtype={'subject_id': str})[['window_id', 'layer', 'start_seconds']]
        pops = pops.merge(refs, on=['window_id', 'layer'])
    k = pd.concat([keys(TEST, dataset), extra_keys], ignore_index=True)
    d = rd.merge(k, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'] = dataset
    d['backbone'] = d.backbone.map(SA.NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    return d, bad


def summarize_bench(per_key, label):
    out = []
    for arm, g in per_key.groupby('arm'):
        r = dict(recordings_set=label, arm=arm, seeds=len(g), recordings_ok_min=int(g.recordings_ok.min()),
                 recordings_ge100_min=int(g.recordings_ge100.min()))
        for m in ('MAE', 'RMSE', 'pearson_r', 'MAE_ge100'):
            r[m + '_mean'] = float(g[m].mean())
            r[m + '_sd'] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
        r['source'] = SRC_OCT if arm == '(0B)' else SRC_TEST
        out.append(r)
    return out


def per_key(rec):
    out = []
    for (arm, seed), g in rec.groupby(['arm', 'seed']):
        ok = g[g.status.eq('ok')]
        e = ok.predicted_HR - ok.reference_HR
        hi = ok[ok.reference_HR >= 100]
        out.append(dict(arm=arm, seed=seed, recordings=len(g), recordings_ok=len(ok), MAE=float(e.abs().mean()),
                        RMSE=float(np.sqrt((e ** 2).mean())),
                        pearson_r=float(np.corrcoef(ok.predicted_HR, ok.reference_HR)[0, 1]) if len(ok) > 2 else np.nan,
                        recordings_ge100=len(hi), MAE_ge100=float((hi.predicted_HR - hi.reference_HR).abs().mean()) if len(hi) else np.nan))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    S3.init(str(VENDOR))
    man = pd.read_csv(WORK / 'pub' / 'docs' / 'PREREGISTRATION_STAGE5_MANIFEST.csv')
    sub_file = WORK / 'outputs' / 'rehearsal' / 'sealed_mcd_test_subjects.txt'
    import hashlib
    assert hashlib.sha256(sub_file.read_bytes()).hexdigest() == man[man.kind.eq('mcd_test_subjects')].sha256.iloc[0]
    test_subjects = set(sub_file.read_text().split())
    settings, coverage, arms_all, arms_common, contrasts, bench_k, bench_s, chk = [], [], [], [], [], [], [], []
    for ds, letter in (('MMPD', 'B'), ('MCD', 'M')):
        s1 = TEST / 'stage1' / ds
        pops = pd.read_csv(s1 / 'populations.csv', dtype={'subject_id': str})
        rds, extra = [], []
        preds_by_seed = {}
        for sd in SEEDS:
            preds, sha = october(letter, sd, None if ds == 'MMPD' else test_subjects)
            preds_by_seed[sd] = preds
            settings.append(dict(dataset=ds, seed=sd, run=RUN.format(sd), checkpoint_sha256=sha, stored_recordings=len(preds),
                                 source=SRC_OCT))
            key = OKEY.format(sd)
            if ds == 'MMPD':
                rd = readouts_mmpd(preds, key, pops)
            else:
                refs = pd.read_csv(s1 / 'references.csv', dtype={'subject_id': str})
                refs = refs[refs.window_id.isin(set(pops.window_id))]
                recs = pd.read_csv(TEST / 'stage2_conf' / 'MCD' / 'recordings.csv').set_index('recording_id')
                rd = readouts_mcd(preds, key, refs, recs, 'test')
                rd = rd.merge(pops[['window_id', 'layer']].drop_duplicates(), on=['window_id', 'layer'])
                ro = readouts_mcd(preds, key, refs, recs, 'october').merge(pops[['window_id', 'layer']].drop_duplicates(),
                                                                         on=['window_id', 'layer'])
                m = rd.merge(ro, on=['window_id', 'layer'], suffixes=('', '_oct'))
                same = (m.valid == m.valid_oct) & ((m.predicted_HR == m.predicted_HR_oct) | (~m.valid & ~m.valid_oct))
                chk.append(dict(check='MCD (0B) readout, test grid vs October grid', dataset=ds, seed=sd, windows=len(m),
                                valid_test_grid=int(m.valid.sum()), valid_october_grid=int(m.valid_oct.sum()), differing=int((~same).sum())))
            rds.append(rd)
            extra.append(dict(key=key, backbone='RhythmFormer', arm='(0B)', seed=sd, variant='std', key_set='october', view='RhythmFormer'))
            print(ds, sd, 'readout done', round(time.time() - t0), flush=True)
        d, bad = load_with(ds, pd.concat(rds, ignore_index=True), pd.DataFrame(extra))
        pw = primary_har(d)
        pw = pw[pw.population.eq('Qstar' if ds == 'MMPD' else 'draft_rule') & pw.backbone.eq('RhythmFormer') & pw.arm.isin(ARMS)]
        r = SA.restrict(pw[pw.arm.ne('(0B)')])
        prim_windows = r.drop_duplicates(['window_id', 'layer'])
        for sd in SEEDS:
            z = pw[pw.key.eq(OKEY.format(sd)) & pw.in_support]
            v = set(z.window_id[z.valid])
            coverage.append(dict(dataset=ds, seed=sd, primary_windows=len(prim_windows), primary_high=int((prim_windows.set == 'high').sum()),
                                 windows_with_0B=len(v & set(prim_windows.window_id)),
                                 high_with_0B=int(prim_windows[prim_windows.set.eq('high')].window_id.isin(v).sum()),
                                 windows_without_0B=len(set(prim_windows.window_id) - v),
                                 high_without_0B=int((~prim_windows[prim_windows.set.eq('high')].window_id.isin(v)).sum()),
                                 recordings_primary=prim_windows.recording_id.nunique(),
                                 recordings_without_0B_prediction=len(set(prim_windows.recording_id) - set(preds_by_seed[sd])),
                                 source=SRC_OCT))
        a = pd.DataFrame(SA.arm_rows(pw, B, 'posthoc'))
        a['dataset'] = ds
        a['source'] = np.where(a.strategy.eq('(0B)'), SRC_OCT, SRC_TEST)
        arms_all.append(a)
        # all four arms on the windows valid for (0B) in every seed (and for the other arms)
        k = pw.groupby('strategy').key.nunique()
        v = SA.restrict(pw)
        cnt = v.groupby(['window_id', 'layer']).key.nunique()
        common = set(cnt[cnt == int(k.sum())].index)
        pc = pw[[t in common for t in zip(pw.window_id, pw.layer)]]
        a2 = pd.DataFrame(SA.arm_rows(pc, B, 'posthoc common windows'))
        a2['dataset'] = ds
        a2['source'] = np.where(a2.strategy.eq('(0B)'), SRC_OCT, SRC_TEST)
        arms_common.append(a2)
        print(ds, 'arms done', round(time.time() - t0), flush=True)
        c = pd.DataFrame(SA.contrast_rows(pw, [('', x, y, 'values only') for x, y in CONTRASTS], B, 'posthoc'))
        c['dataset'] = ds
        c['source'] = f'{SRC_TEST}; (0B): {SRC_OCT}'
        contrasts.append(c)
        print(ds, 'contrasts done', round(time.time() - t0), flush=True)
        # recording-level scores
        labels = {f.stem.replace('__', '/'): f for f in sorted((s1 / 'labels').glob('*.npy'))}
        brow = []
        for sd in SEEDS:
            for rid, lf in labels.items():
                p = preds_by_seed[sd].get(rid)
                base = dict(arm='(0B)', seed=sd, recording_id=rid)
                if p is None:
                    brow.append(dict(base, status='no_prediction'))
                    continue
                try:
                    brow.append(dict(base, **benchmark.recording_row(p[0], np.load(lf), 'RhythmFormer', False)))
                except AssertionError:
                    brow.append(dict(base, status='length_mismatch'))
        rec0b = pd.DataFrame(brow)
        folder = TEST / 'analysis' / ds / ('automatic' if ds == 'MMPD' else 'halves')
        bt = pd.read_csv(folder / 'benchmark_recordings.csv')
        bt = bt[bt.backbone.eq('RhythmFormer') & bt.arm.isin(ARMS[1:]) & bt.seed.isin(SEEDS)]
        rec = pd.concat([rec0b, bt[['arm', 'seed', 'recording_id', 'status', 'predicted_HR', 'reference_HR']]], ignore_index=True)
        # check: our recording_row on the stored test (0) predictions equals the test benchmark rows
        s2 = TEST / 'stage2_conf' / ds
        ck = pd.read_csv(s2 / 'checkpoints.csv')
        nd = 0
        nchk = 0
        for kk in ck[ck.backbone.eq('RhythmFormer') & ck.arm.eq('(0)') & ck.variant.eq('std')].itertuples():
            ref = bt[bt.arm.eq('(0)') & bt.seed.eq(kk.seed)].set_index('recording_id')
            for rid, lf in list(labels.items()):
                f = s2 / 'pred' / kk.key / (rid.replace('/', '__') + '.npy')
                if not f.exists() or rid not in ref.index:
                    continue
                x = benchmark.recording_row(np.load(f), np.load(lf), 'RhythmFormer', False)
                o = ref.loc[rid]
                nchk += 1
                nd += int(x['status'] != o.status or (x['status'] == 'ok' and (x['predicted_HR'] != o.predicted_HR or x['reference_HR'] != o.reference_HR)))
        chk.append(dict(check='recording_row on stored test (0) predictions vs test benchmark_recordings', dataset=ds, seed='all',
                        windows=nchk, differing=nd))
        pk = per_key(rec)
        pk['recordings_set'] = 'own recordings of each key'
        own0b = rec0b[rec0b.status.eq('ok')][['seed', 'recording_id']]
        pk2 = per_key(rec.merge(own0b, on=['seed', 'recording_id']))
        pk2['recordings_set'] = 'recordings scored for (0B) of the same seed'
        for t in (pk, pk2):
            t['dataset'] = ds
            t['source'] = np.where(t.arm.eq('(0B)'), SRC_OCT, SRC_TEST)
        bench_k += [pk, pk2]
        for t, lab in ((pk, 'own recordings of each key'), (pk2, 'recordings scored for (0B) of the same seed')):
            s = pd.DataFrame(summarize_bench(t, lab))
            s['dataset'] = ds
            bench_s.append(s)
        st = rec0b.status.value_counts().to_dict()
        chk.append(dict(check='(0B) recording_row status counts', dataset=ds, seed='all', windows=len(rec0b), differing=0,
                        **{f'status_{k2}': v2 for k2, v2 in st.items()}))
        print(ds, 'benchmark done', round(time.time() - t0), flush=True)
    write(pd.DataFrame(settings), 'ra2_settings.csv', 'ra2')
    write(pd.DataFrame(coverage), 'ra2_coverage.csv', 'ra2')
    write(pd.concat(arms_all), 'ra2_arms.csv', 'ra2')
    write(pd.concat(arms_common), 'ra2_arms_common.csv', 'ra2')
    write(pd.concat(contrasts), 'ra2_contrasts.csv', 'ra2')
    write(pd.concat(bench_k), 'ra2_benchmark_per_key.csv', 'ra2')
    write(pd.concat(bench_s), 'ra2_benchmark.csv', 'ra2')
    write(pd.DataFrame(chk), 'ra2_checks.csv', 'ra2')
    print(pd.DataFrame(chk).to_string(index=False))
    print(pd.DataFrame(coverage).to_string(index=False))
    stamp('ra2', seconds=time.time() - t0)


if __name__ == '__main__':
    main()
