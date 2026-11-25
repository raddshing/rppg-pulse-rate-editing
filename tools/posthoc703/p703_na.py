"""Brief 02 7.03 나: output SNR of the saved test predictions (post hoc; saved test step (c) stage-2 waveforms only,
no re-inference).

Windows and waveforms: the test step (c) primary analysis windows (wide band HAR, primary variant; MMPD-B Q*, MCD-rPPG test
half draft rule; confirmatory support and each key's valid windows, eval5/stage3_analyze.py restrict), and for each key and
window the saved stage-2 prediction segment read exactly as step (b) reads it (eval5/stage3_check.py mmpd_job / mcd_job:
MMPD 300 samples from 300 x window_index, float64; MCD 300 samples from the uniform-grid start of complete_windows,
float32 as stored). The readout preparation is that of step (b) (src/eval/grid.py _prepare: PhysNet differential output
accumulated, detrend lambda 100, first-order Butterworth 0.6-3.3 Hz, filtfilt); the HR read from it must equal the stored
step (b) readout (check).
SNR (dB, the official rPPG-Toolbox definition as archived in src/eval/har.py snr, unchanged): 512-point periodogram of the
prepared segment; numerator = power within +-0.1 Hz of the reference HR and of twice the reference HR (not clipped to the
band); denominator = the remaining power within 0.6-3.3 Hz; reference HR = the stage-1 window reference of the analysis.
Arms (0), (i), (ii), (i-pc), (ii-pc), (i-simple), (ii-lag) x three backbones: seed mean of the window-mean SNR on the high,
low and all windows, 95% crossed subject x seed bootstrap (b1/boot.py, B = 10,000, default_rng(20261110)), windows valid
for every key of the arm (as 7.02 가). Contrasts C1-C4 and (ii-pc)-(i-simple): same-seed pairs on the windows valid for
every key of both arms (pair_windows). No grades.
Writes outputs/posthoc703/na/{na_arms,na_contrasts,na_checks}.csv (+ server-only window SNR table in cache/posthoc703).
"""
import sys
import time

from p703_common import B, CONF, NAMES, SA, SEEDS, SRC_TEST, TEST, VENDOR, WORK, boot, load, main_population, np, pd, primary_har, stamp, write
sys.path.insert(0, str(WORK / 'eval5'))
import stage3_check as S3  # noqa: E402

ARMS = ['(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)', '(i-simple)', '(ii-lag)']
PAIRS = list(CONF) + [('', '(ii-pc)', '(i-simple)')]
BAND = (.6, 3.3)
CACHE = WORK / 'cache' / 'posthoc703' / 'na'


def key_dir(ds, key):
    for f in ('stage2_conf', 'stage2_expl'):
        p = TEST / f / ds / 'pred' / key
        if p.is_dir():
            return f, p
    raise FileNotFoundError(key)


def window_values(ds, g, recs):
    """(HR, SNR) for every window row of one key, read as step (b) reads it."""
    from src.eval import grid, har
    from src.st11x.pathways import uniform_frame_grid, complete_windows
    folder, kd = key_dir(ds, g.key.iloc[0])
    diff = g.view.iloc[0] == 'PhysNet' and '-std' not in g.arm.iloc[0]
    out = []
    for rid, x in g.groupby('recording_id', sort=False):
        if ds == 'MMPD':
            pred = np.load(kd / (rid.replace('/', '__') + '.npy')).astype(np.float64)
        else:
            pred = np.load(kd / (rid + '.npy'))
            n, span = int(recs[folder].loc[rid, 'decoded_frames']), float(recs[folder].loc[rid, 'span_seconds'])
            fg = uniform_frame_grid(np.arange(n, dtype=np.float64) * span / (n - 1))
        for w in x.itertuples():
            if ds == 'MMPD':
                s = 300 * int(w.window_index)
            else:
                bd = complete_windows(fg, [float(w.start_seconds)])[0]
                assert bd['available']
                s = bd['grid_start']
            seg = pred[s:s + 300]
            assert len(seg) == 300
            prep = grid._prepare(seg, diff, BAND, 'default', 30)
            hr = float(grid._estimate(prep, 'periodogram', BAND, 30))
            out.append((w.Index, hr, float(har.snr(prep, w.reference_HR, 30., BAND[0], BAND[1]))))
    return out


def m_snr(W, r, p, s):
    """p carries the window SNR (dB)."""
    out = {}
    for name, mask in [('high', s == 'high'), ('low', s == 'low'), ('all', np.ones(len(r), bool))]:
        out[f'SNR_{name}'] = boot._wmean(W[:, mask], p[mask]) if mask.any() else np.full(len(W), np.nan)
    return out


def arm_windows(g):
    k = g.key.nunique()
    v = g.groupby(['window_id', 'layer']).key.nunique()
    w = set(v[v == k].index)
    return g[[x in w for x in zip(g.window_id, g.layer)]], len(w)


def main():
    t0 = time.time()
    S3.init(str(VENDOR))
    datasets = sys.argv[1:] or ['MMPD', 'MCD']
    arms_out, con_out, chk, wins = [], [], [], []
    for ds in datasets:
        d, bad = load(TEST, ds)
        pw = primary_har(d)
        pw = pw[pw.population.eq(main_population(ds))]
        r = SA.restrict(pw)
        r = r[r.strategy.isin(ARMS) & r.seed.isin(SEEDS)].copy()
        recs = {f: pd.read_csv(TEST / f / ds / 'recordings.csv').set_index('recording_id') for f in ('stage2_conf', 'stage2_expl')}
        r['HR_recomputed'], r['SNR'] = np.nan, np.nan
        for k, (key, g) in enumerate(r.groupby('key'), 1):
            for idx, hr, snr in window_values(ds, g, recs):
                r.at[idx, 'HR_recomputed'] = hr
                r.at[idx, 'SNR'] = snr
            if k % 10 == 0:
                print(ds, k, 'keys', round(time.time() - t0), flush=True)
        eq = (r.HR_recomputed == r.predicted_HR)
        nonfinite = int((~np.isfinite(r.SNR)).sum())
        chk.append(dict(dataset=NAMES[ds], check='readout HR recomputed from the saved waveform vs stored step (b) readout',
                        rows=len(r), equal=int(eq.sum()), max_abs_difference=float((r.HR_recomputed - r.predicted_HR).abs().max()),
                        nonfinite_SNR=nonfinite, keys=r.key.nunique()))
        assert nonfinite == 0, nonfinite
        wins.append(r[['key', 'backbone', 'strategy', 'seed', 'window_id', 'subject_id', 'set', 'reference_HR', 'predicted_HR', 'SNR']]
                    .assign(dataset=NAMES[ds]))
        s = r.copy()
        s['predicted_HR'] = s.SNR  # the bootstrap engine reads the 'predicted_HR' column
        for bb in sorted(s.backbone.unique()):
            for st in ARMS:
                g, nw = arm_windows(s[s.backbone.eq(bb) & s.strategy.eq(st)])
                pt, ci = boot.point_ci(g, SEEDS, m_snr, B)
                one = g[g.seed.eq(SEEDS[0])]
                for m in pt:
                    arms_out.append(dict(dataset=NAMES[ds], backbone=bb, strategy=st, windows=nw, windows_high=int(one.set.eq('high').sum()),
                                         windows_low=int(one.set.eq('low').sum()), subjects=g.subject_id.nunique(), metric=m,
                                         point=pt[m], ci_lo=ci[m][0], ci_hi=ci[m][1], source=SRC_TEST))
            for name, a, c in PAIRS:
                ga, gc = s[s.backbone.eq(bb) & s.strategy.eq(a)], s[s.backbone.eq(bb) & s.strategy.eq(c)]
                ga, gc, nw = SA.pair_windows(ga, gc)
                pt, ci = boot.contrast_ci(ga, gc, SEEDS, m_snr, B)
                one = ga[ga.seed.eq(SEEDS[0])]
                for m in pt:
                    con_out.append(dict(dataset=NAMES[ds], backbone=bb, name=name, contrast=f'{a}-{c}', windows=nw,
                                        windows_high=int(one.set.eq('high').sum()), windows_low=int(one.set.eq('low').sum()),
                                        subjects=ga.subject_id.nunique(), metric=m, point=pt[m], ci_lo=ci[m][0], ci_hi=ci[m][1],
                                        source=SRC_TEST))
            print(ds, bb, 'done', round(time.time() - t0), flush=True)
    arms, con = pd.DataFrame(arms_out), pd.DataFrame(con_out)
    # window counts against 7.02 가 (same window rule)
    old = pd.read_csv(WORK / 'outputs' / 'posthoc702' / 'ga' / 'ga_arms.csv')
    old = old[old.metric.eq('hit_rate_all')][['dataset', 'backbone', 'strategy', 'windows', 'windows_high', 'windows_low']]
    m = arms[arms.metric.eq('SNR_all')].merge(old, on=['dataset', 'backbone', 'strategy'], suffixes=('', '_702'))
    chk.append(dict(dataset='both', check='arm window counts vs 7.02 ga_arms.csv', rows=len(m),
                    equal=int(((m.windows == m.windows_702) & (m.windows_high == m.windows_high_702) & (m.windows_low == m.windows_low_702)).sum()),
                    max_abs_difference=float((m.windows - m.windows_702).abs().max())))
    old = pd.read_csv(WORK / 'outputs' / 'posthoc702' / 'ga' / 'ga_contrasts.csv')
    old = old[old.metric.eq('hit_rate_all')][['dataset', 'backbone', 'contrast', 'windows']]
    m = con[con.metric.eq('SNR_all')].merge(old, on=['dataset', 'backbone', 'contrast'], suffixes=('', '_702'))
    chk.append(dict(dataset='both', check='contrast window counts vs 7.02 ga_contrasts.csv', rows=len(m),
                    equal=int((m.windows == m.windows_702).sum()), max_abs_difference=float((m.windows - m.windows_702).abs().max())))
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat(wins).to_csv(CACHE / 'window_snr_server_only.csv.gz', index=False)
    write(arms, 'na_arms.csv', 'na')
    write(con, 'na_contrasts.csv', 'na')
    chk = pd.DataFrame(chk).assign(source=f'{SRC_TEST} (recomputed here vs stored tables)')
    write(chk, 'na_checks.csv', 'na')
    print(chk.to_string())
    stamp('na', datasets=datasets, seconds=time.time() - t0, checks=chk.to_dict('records'),
          snr_definition='src/eval/har.py snr on grid._prepare(segment, diff, (0.6, 3.3), default, 30); reference = stage-1 window HR')


if __name__ == '__main__':
    main()
