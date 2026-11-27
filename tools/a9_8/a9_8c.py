"""OPS2 A9 item 8c (7.13 1): VIPL-HR PhysNet outputs, SNR at the reference HR and phase relative to the BVP.

Windows and keys: every PhysNet arm evaluated on VIPL-HR when A9 was committed (stages (b1), (b2); seeds 20261131-20261133),
the primary analysis windows (eval5/vipl_c.py: primary set, HAR, verified, no 200-ms flag, reference = gt_HR mean;
confirmatory support and the key's valid windows), per arm the windows valid for every key of the arm.
Waveform: the stored stage (b) prediction of the window, samples 300 j .. 300 j + 299 of the 30-Hz grid from the first frame,
float32 as stored, prepared as the readout of stage (c) (eval5/stage3_check.read_hr: src/eval/grid._prepare, PhysNet
DiffNormalized output accumulated, detrend lambda 100, first-order Butterworth 0.6-3.3 Hz, filtfilt); the HR read from it
must equal the stored (c2) readout (check).
SNR (dB): the rPPG-Toolbox evaluation SNR as archived (src/eval/har.snr, unchanged; as tools/posthoc703/p703_na.py): 512-point
periodogram of the prepared window; power within +-0.1 Hz of the reference HR and of twice it, over the remaining power in
0.6-3.3 Hz; reference HR = the window reference of the primary analysis (gt_HR mean).
Phase: the BVP of the recording placed on the 30-Hz axis from the first frame at 60 Hz (as the A8 entry, the second entry
of 2026-11-23; cache from tools/a9_8/vipl_bvp.py), the same window, prepared as the prediction without accumulation
(grid._prepare, 0.6-3.3 Hz, zero-phase filtering). At the reference frequency f = reference HR / 60 the Hann-windowed DFT
coefficient of each signal; phase = arg(P conj(Q)) (prediction minus BVP; positive = the prediction leads), in degrees.
Windows outside the 60-Hz BVP record have no phase (counted).
Per arm (overall, high, low; overall and by scene v1-v9): SNR seed mean with 95% crossed bootstrap interval (b1/boot.py
point_ci with p703_na.m_snr); phase circular mean (the direction of the mean unit vector, averaged over seeds as the
bootstrap averages a metric) with the resultant length, and a 95% interval from the same crossed bootstrap draws
(B = 10,000, default_rng(20261110)): percentiles of the replicate directions measured from the point direction.
Writes outputs/a9_8/8c/{8c_snr,8c_phase,8c_checks}.csv; window values server-only in cache/a9_8/8c_windows.csv.gz.
"""
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import a9_common as C
from a9_common import B, SA, SEEDS, boot, np, pd
import stage3_check as SC
from p703_na import m_snr

BAND = (.6, 3.3)
BVP = C.WORK / 'cache' / 'a9_8' / 'bvp60'
VENDOR = str(C.P2.VENDOR)


def dft(x, f_hz):
    n = np.arange(len(x))
    return np.sum(np.hanning(len(x)) * x * np.exp(-2j * np.pi * f_hz * n / 30.0))


def key_job(args):
    key, folder, diff, rows = args
    SC.vendor_imports(VENDOR)
    from src.eval import grid, har
    out = []
    for rid, wins in rows:
        pred = np.load(f'{folder}/pred/{key}/{rid}.npy')
        bp = BVP / f'{rid}.npz'
        bv = np.load(bp)['c30'] if bp.exists() else None
        for idx, j, ref in wins:
            seg = pred[300 * j:300 * j + 300]
            prep = grid._prepare(seg, diff, BAND, 'default', 30)
            hr = float(grid._estimate(prep, 'periodogram', BAND, 30))
            snr = float(har.snr(prep, ref, 30., BAND[0], BAND[1]))
            ph = np.nan
            if bv is not None and 300 * (j + 1) <= len(bv):
                q = grid._prepare(bv[300 * j:300 * j + 300], False, BAND, 'default', 30)
                ph = float(np.angle(dft(prep, ref / 60.0) * np.conj(dft(q, ref / 60.0))))
            out.append((idx, hr, snr, ph))
    return out


def m_cs(W, r, p, s):
    out = {}
    for name, mask in [('high', s == 'high'), ('low', s == 'low'), ('all', np.ones(len(r), bool))]:
        for k, f in (('c', np.cos), ('s', np.sin)):
            out[f'{k}_{name}'] = boot._wmean(W[:, mask], f(p[mask])) if mask.any() else np.full(len(W), np.nan)
    return out


def circ(g, seeds=SEEDS, b=B, chunk=500):
    """Circular mean (deg), resultant length and 95% interval of the phase (column 'phase', radians)."""
    subjects = boot.numeric_subject_order(g.subject_id)
    smp = boot.Sample(g.assign(predicted_HR=g.phase), subjects, seeds)
    full = boot.per_seed(smp, m_cs, np.ones((1, len(subjects))))
    res = {}
    isub, iseed = boot.draws(len(subjects), len(seeds), b)
    reps = {n: [np.empty(b), np.empty(b)] for n in ('high', 'low', 'all')}
    for a in range(0, b, chunk):
        Wsub = boot.subject_weights(isub[a:a + chunk], len(subjects))
        vals = boot.per_seed(smp, m_cs, Wsub)
        for n in reps:
            for i, k in enumerate(('c', 's')):
                stack = np.stack([vals[sd][f'{k}_{n}'] for sd in seeds])
                chosen = stack[iseed[a:a + chunk].T, np.arange(Wsub.shape[0])[None, :]]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    reps[n][i][a:a + chunk] = np.nanmean(chosen, axis=0)
    for n in reps:
        c = float(np.nanmean([full[sd][f'c_{n}'][0] for sd in seeds]))
        s = float(np.nanmean([full[sd][f's_{n}'][0] for sd in seeds]))
        pt = np.arctan2(s, c)
        d = np.angle(np.exp(1j * (np.arctan2(reps[n][1], reps[n][0]) - pt)))
        with np.errstate(all='ignore'):
            lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        res[n] = dict(point_deg=float(np.degrees(pt)), ci_lo_deg=float(np.degrees(pt + lo)), ci_hi_deg=float(np.degrees(pt + hi)),
                      resultant_length=float(np.hypot(c, s)))
    return res


def main():
    t0 = time.time()
    SC.init(VENDOR)
    d, refs, choice, bad = C.vipl_frame()
    pw = SA.restrict(C.primary(d))
    pw = pw[pw.backbone.eq('PhysNet') & pw.seed.isin(SEEDS)].copy()
    keys = C.vipl_keys()
    folder = {r.key: str(C.VO / ('stage_b_confirmatory' if r.set == 'confirmatory' else 'stage_b_exploratory') / 'VIPL')
              for r in keys.itertuples()}
    # arm windows first (only the windows used are read)
    parts = []
    for st, g in pw.groupby('strategy'):
        g, nw = C.arm_windows(g)
        parts.append(g)
    pw = pd.concat(parts)
    jobs = []
    for key, g in pw.groupby('key'):
        diff = g.view.iloc[0] == 'PhysNet' and '-std' not in g.arm.iloc[0]
        rows = [(rid, [(i, int(j), float(r)) for i, j, r in zip(x.index, x.window_index, x.reference_HR)]) for rid, x in g.groupby('recording_id')]
        jobs.append((key, folder[key], diff, rows))
    pw['HR_recomputed'], pw['SNR'], pw['phase'] = np.nan, np.nan, np.nan
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=SC.init, initargs=(VENDOR,)) as ex:
        for n, part in enumerate(ex.map(key_job, jobs), 1):
            for idx, hr, snr, ph in part:
                pw.at[idx, 'HR_recomputed'], pw.at[idx, 'SNR'], pw.at[idx, 'phase'] = hr, snr, ph
            if n % 10 == 0:
                print('keys', n, '/', len(jobs), round(time.time() - t0), flush=True)
    chk = [dict(check='readout HR recomputed from the stored waveform vs stage (c2) readout', rows=len(pw),
                equal=int((pw.HR_recomputed == pw.predicted_HR).sum()), max_abs_difference=float((pw.HR_recomputed - pw.predicted_HR).abs().max()),
                nonfinite_SNR=int((~np.isfinite(pw.SNR)).sum()), windows_without_phase=int(pw.drop_duplicates('window_id').phase.isna().sum()),
                windows=pw.window_id.nunique(), keys=pw.key.nunique())]
    out_dir = C.WORK / 'cache' / 'a9_8'
    pw[['key', 'strategy', 'seed', 'window_id', 'subject_id', 'scene', 'set', 'reference_HR', 'predicted_HR', 'SNR', 'phase']].to_csv(
        out_dir / '8c_windows.csv.gz', index=False)
    snr_rows, ph_rows = [], []
    for st, g in pw.groupby('strategy'):
        for scope, z in [('all scenes', g)] + [(f'scene {s}', x) for s, x in sorted(g.groupby('scene'))]:
            one = z[z.seed.eq(SEEDS[0])]
            base = dict(dataset='VIPL-HR', backbone='PhysNet', strategy=st, scope=scope, windows=len(one),
                        windows_high=int(one.set.eq('high').sum()), windows_low=int(one.set.eq('low').sum()), subjects=z.subject_id.nunique())
            pt, ci = boot.point_ci(z.assign(predicted_HR=z.SNR), SEEDS, m_snr, B)
            for k in pt:
                snr_rows.append(dict(base, metric=k, point=pt[k], ci_lo=ci[k][0], ci_hi=ci[k][1], source=C.SRC_VIPL))
            zp = z[np.isfinite(z.phase)]
            onep = zp[zp.seed.eq(SEEDS[0])]
            for n, v in circ(zp).items():
                ph_rows.append(dict(base, windows_with_phase=int(len(onep) if n == 'all' else onep.set.eq(n).sum()),
                                    metric=f'phase_{n}', **v, source=C.SRC_VIPL + '; BVP at 60 Hz'))
        print(st, round(time.time() - t0), flush=True)
    C.write(pd.DataFrame(snr_rows), '8c_snr.csv', '8c')
    C.write(pd.DataFrame(ph_rows), '8c_phase.csv', '8c')
    C.write(pd.DataFrame(chk), '8c_checks.csv', '8c')
    C.stamp('8c', seconds=time.time() - t0, checks=chk)
    print('done', round(time.time() - t0), chk, flush=True)


if __name__ == '__main__':
    main()
