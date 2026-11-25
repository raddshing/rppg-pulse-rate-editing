"""Brief 02 6.48 다 1 / docs/STAGE5_DEV_DIAGNOSTICS.md rule 2: judgement numbers of the RhythmFormer blur re-inference (CPU,
saved predictions only).

Windows: MCD-rPPG development half, draft-rule windows of b1/analyze.py load_mcd (wide band HAR; high = agreement and
reference >= 100, low = < 100), restricted to the recordings of the blur re-inference (b9/blur_infer.py: the draft-rule
recordings) and, per checkpoint, to the windows available both without and with the filter.
Readouts: without filter = outputs/1b/mcd_dev_preds_<group>.csv.gz (the standard RhythmFormer inference); with filter =
b1/mcd_windows.py on cache/mcd_dev_waveforms/blur<10|15|20>_<group> (outputs/blur/windows/).
Per arm x sigma x seed (high-HR windows unless stated):
  line fraction without / with the filter (window read within 147.65625-154.6875 bpm), relative fall = 1 - with/without;
  line windows = windows read on the line by the same checkpoint without or with the filter; high-HR MAE outside the line
  windows without / with the filter and its change; high-HR and low-HR MAE on all windows without / with the filter.
Seed mean = mean over seeds 20261131-33 of the per-seed values; the relative fall of the seed mean uses the seed-mean fractions.
Judgement (rule 2; sigma 1.5, arm (0), seed mean): 'consistent' if the line fraction falls by >= 50% and the outside-line
high-HR MAE changes by < 1 bpm (absolute); 'rejected' if it falls by < 20%; otherwise 'open'.
Writes outputs/blur/judgement_per_seed.csv, judgement_seed_mean.csv, judgement.json.
Guarded judgement (brief 02 6.54 나 5): OPS2_BLUR_BASE (folder of the unfiltered readouts mcd_dev_preds_<group>.csv.gz, e.g.
the guarded readouts of the rehearsal part-1 pass A predictions), OPS2_BLUR_WINDOWS (folder of the filtered readouts),
OPS2_BLUR_PREFIX (tag prefix of the filtered folders, e.g. 'g' -> gblur15_<group>), OPS2_BLUR_LOG (infer_log.csv of the
filtered run) and OPS2_BLUR_OUT (output folder). Defaults = the original paths.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import analyze  # noqa: E402

LINE = (147.65625, 154.6875)
GROUPS = {'g3': None, 'iipc_RhythmFormer': '(ii-pc)', 'ipc_RhythmFormer': '(i-pc)', 'iinv_RhythmFormer': '(ii-inv)', 'ilag_RhythmFormer': '(ii-lag)'}
SIGMA = {'blur10': 1.0, 'blur15': 1.5, 'blur20': 2.0}
SEEDS = [20261131, 20261132, 20261133]


def on_line(x):
    return (x >= LINE[0]) & (x <= LINE[1])


def main():
    meta = analyze.runs()
    w = pd.read_csv(WORK / 'outputs/1b/mcd_dev_windows.csv', dtype={'subject_id': str})
    recs = pd.read_csv(WORK / 'outputs/1b/mcd_dev_recordings.csv', dtype={'subject_id': str})
    confirmed = set(recs.recording_id[recs.mapping_confirmed.astype(bool)])
    har = w[w.layer.eq('HAR') & w.primary_included & w.agreement & w.h_FFT.ge(100)]
    draft = set(har.recording_id) & confirmed
    pop = w[w.layer.eq('HAR') & w.primary_included & np.isfinite(w.h_FFT) & w.recording_id.isin(draft)].copy()
    pop['set'] = np.where(pop.agreement & (pop.h_FFT >= 100), 'high', np.where(pop.h_FFT < 100, 'low', ''))
    pop = pop[pop.set.ne('')][['window_id', 'layer', 'recording_id', 'subject_id', 'h_FFT', 'set']]
    blur_recs = set(pd.read_csv(os.environ.get('OPS2_BLUR_LOG', WORK / 'outputs/blur/infer_log.csv')).query("status == 'done'").recording_id)
    pop = pop[pop.recording_id.isin(blur_recs)]
    rows = []
    for g, arm in GROUPS.items():
        fb = Path(os.environ.get('OPS2_BLUR_BASE', WORK / 'outputs/1b')) / f'mcd_dev_preds_{g}.csv.gz'
        if not fb.exists():
            continue
        base = pd.read_csv(fb)
        base = base[base.layer.eq('HAR')]
        for tag, sigma in SIGMA.items():
            f = Path(os.environ.get('OPS2_BLUR_WINDOWS', WORK / 'outputs/blur/windows')) / f"mcd_dev_preds_{os.environ.get('OPS2_BLUR_PREFIX', '')}{tag}_{g}.csv.gz"
            if not f.exists():
                continue
            bl = pd.read_csv(f)
            bl = bl[bl.layer.eq('HAR')]
            for run in sorted(bl.run_id.unique()):
                m = meta.get(run)
                if m is None or m['backbone'] != 'RhythmFormer' or m['seed'] not in SEEDS:
                    continue
                st = m['strategy']
                if arm and st != arm:
                    continue
                a = base[base.run_id.eq(run)][['window_id', 'predicted_HR', 'available']].rename(columns={'predicted_HR': 'p0', 'available': 'a0'})
                b = bl[bl.run_id.eq(run)][['window_id', 'predicted_HR', 'available']].rename(columns={'predicted_HR': 'p1', 'available': 'a1'})
                x = pop.merge(a, on='window_id').merge(b, on='window_id')
                x = x[x.a0.astype(bool) & x.a1.astype(bool) & np.isfinite(x.p0) & np.isfinite(x.p1)]
                h, lo = x[x.set.eq('high')], x[x.set.eq('low')]
                line = on_line(h.p0) | on_line(h.p1)
                out = h[~line]
                r = dict(arm=st, sigma=sigma, seed=m['seed'], run_id=run, high_windows=len(h), low_windows=len(lo),
                         line_fraction_without=float(on_line(h.p0).mean()), line_fraction_with=float(on_line(h.p1).mean()),
                         line_windows_either=int(line.sum()), outside_windows=len(out),
                         MAE_outside_without=float((out.p0 - out.h_FFT).abs().mean()), MAE_outside_with=float((out.p1 - out.h_FFT).abs().mean()),
                         MAE_high_without=float((h.p0 - h.h_FFT).abs().mean()), MAE_high_with=float((h.p1 - h.h_FFT).abs().mean()),
                         MAE_low_without=float((lo.p0 - lo.h_FFT).abs().mean()), MAE_low_with=float((lo.p1 - lo.h_FFT).abs().mean()))
                r['relative_fall'] = 1 - r['line_fraction_with'] / r['line_fraction_without'] if r['line_fraction_without'] > 0 else np.nan
                r['MAE_outside_change'] = r['MAE_outside_with'] - r['MAE_outside_without']
                rows.append(r)
    t = pd.DataFrame(rows)
    out = Path(os.environ.get('OPS2_BLUR_OUT', WORK / 'outputs' / 'blur'))
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'judgement_per_seed.csv', index=False)
    num = [c for c in t.columns if c not in ('arm', 'sigma', 'seed', 'run_id')]
    sm = t.groupby(['arm', 'sigma'])[num].mean().reset_index()
    sm['seeds'] = t.groupby(['arm', 'sigma']).seed.nunique().to_numpy()
    sm['relative_fall_of_seed_mean'] = 1 - sm.line_fraction_with / sm.line_fraction_without
    sm.to_csv(out / 'judgement_seed_mean.csv', index=False)
    j = sm[sm.arm.eq('(0)') & sm.sigma.eq(1.5)]
    res = dict(rule='docs/STAGE5_DEV_DIAGNOSTICS.md rule 2', judged='RhythmFormer (0), sigma 1.5, seed mean')
    if len(j) and int(j.seeds.iloc[0]) == 3:
        fall, change = float(j.relative_fall_of_seed_mean.iloc[0]), float(j.MAE_outside_change.iloc[0])
        verdict = 'consistent' if fall >= 0.5 and abs(change) < 1 else 'rejected' if fall < 0.2 else 'open'
        res.update(line_fraction_without=float(j.line_fraction_without.iloc[0]), line_fraction_with=float(j.line_fraction_with.iloc[0]),
                   relative_fall=fall, MAE_outside_change=change, first_condition=bool(fall >= 0.5 and abs(change) < 1), outcome=verdict,
                   line_removal_procedure_fixed=bool(fall >= 0.5 and abs(change) < 1))
    else:
        res['outcome'] = 'not computable (missing seeds)'
    (out / 'judgement.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(sm.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
