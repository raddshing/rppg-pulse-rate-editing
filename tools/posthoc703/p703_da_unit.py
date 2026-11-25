"""Brief 02 7.03 다: synthetic unit test of the sign and delay conventions of p703_da_common.harmonic_phase (no data).

p(t) = sin(2 pi f0 t) + 0.5 sin(2 pi 2 f0 t + 0.7) + 0.2 sin(2 pi 3 f0 t + 1.9) + noise_p;
v(t) = s p_clean(t - d / fs) + noise_v (independent white noise, SD 0.3 each), 60 s, rng default_rng(20261110).
Cases: f0 in {0.8, 1.23, 1.6, 2.07, 2.6} Hz (1.23 and 2.07 off the Welch grid), s in {+1, -1}, d in {0, 2, 5} frames,
fs in {30, 23.3} Hz, filter on/off. Expected: decision 'positive' for s = +1 and 'negative' for s = -1; d_h ~ d frames.
Writes outputs/posthoc703/da/da_unit_test.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p703_da_common import harmonic_phase  # noqa: E402

OUT = HERE.parents[1] / 'outputs' / 'posthoc703' / 'da'


def clean(t, f0):
    w = 2 * np.pi * f0
    return np.sin(w * t) + 0.5 * np.sin(2 * w * t + 0.7) + 0.2 * np.sin(3 * w * t + 1.9)


def main():
    rng = np.random.default_rng(20261110)
    rows = []
    for fs in (30.0, 23.3):
        t = np.arange(int(60 * fs)) / fs
        for f0 in (0.8, 1.23, 1.6, 2.07, 2.6):
            for s in (1, -1):
                for d in (0, 2, 5):
                    p = clean(t, f0) + 0.3 * rng.standard_normal(len(t))
                    v = s * clean(t - d / fs, f0) + 0.3 * rng.standard_normal(len(t))
                    for filt in (True, False):
                        r = harmonic_phase(v, p, fs, filtered=filt)
                        rows.append(dict(fs=fs, f0_true_Hz=f0, s=s, d_frames=d, filtered=filt, **r,
                                         expected='positive' if s > 0 else 'negative',
                                         d_h_error_frames=r['d_h_frames'] - d,
                                         f0_detected_minus_true_Hz=r['f0_Hz'] - f0))
    df = pd.DataFrame(rows)
    df['decision_ok'] = df.decision.eq(df.expected)
    df['source'] = 'synthetic signals (no data)'
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'da_unit_test.csv', index=False)
    print('cases', len(df), 'decision ok', int(df.decision_ok.sum()), 'max |d_h - d| frames', float(df.d_h_error_frames.abs().max()))
    print(df.groupby(['fs', 'filtered']).agg(ok=('decision_ok', 'sum'), n=('decision_ok', 'size'),
                                             max_dh_err=('d_h_error_frames', lambda x: x.abs().max()),
                                             min_g1=('gamma2_f0', 'min'), min_g2=('gamma2_2f0', 'min'),
                                             max_absdev=('Delta_deg', lambda x: (180 - x.abs()).where(df.loc[x.index, 's'] < 0, x.abs()).max())))
    print(df[~df.decision_ok][['fs', 'f0_true_Hz', 's', 'd_frames', 'filtered', 'decision', 'Delta_deg', 'gamma2_f0', 'gamma2_2f0']])


if __name__ == '__main__':
    main()
