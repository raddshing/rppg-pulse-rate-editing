"""Brief 02 7.03 다: harmonic phase of the face pulsation against the contact PPG (definitions fixed in the instruction).

Per recording: v = face-mask weighted G / mean(G) - 1 over the recording (the audit's signal, 6.97 나), p = the contact PPG
per frame of the audit reader; both through the audit filter (archived st9e editor_band: first-order 0.6-3.3 Hz zero-phase,
coefficients for 30 Hz, as in the audit; in the ideal case the same real gain on both signals leaves the cross-spectrum
phase and the coherence unchanged; the measured effect of the filter on the decisions is in da_checks.csv). Welch (fs = recording frame rate, 10 s Hann segments: nperseg = round(10 fs), 50% overlap,
nfft = nperseg, constant detrend, mean average):
  S = E[V conj(P)] (scipy csd(p, v) = conj(P) V), S_vv, S_pp
  f0 = frequency of the maximum of S_pp in 0.7-3.0 Hz; 2 f0 = bin 2 k0 (exact on the Welch grid)
  phi1 = arg S(f0), phi2 = arg S(2 f0); gamma2(f) = |S|^2 / (S_vv S_pp)
  Delta = wrap(phi2 - 2 phi1)  (pure delay cancels; pi for negative, 0 for positive polarity)
  d_h = wrap(phi1 - phi2) / (2 pi f0) seconds, x fs frames (video later than the PPG: positive)
Decision (fixed before the data): only if gamma2(f0) >= 0.5 and gamma2(2 f0) >= 0.25; |Delta| >= 120 deg -> negative,
|Delta| <= 60 deg -> positive, otherwise undecidable. wrap() maps to (-pi, pi].
"""
import numpy as np
from scipy.signal import butter, csd, filtfilt, welch

F_LO, F_HI = 0.7, 3.0
G1, G2 = 0.5, 0.25
NEG, POS = 120.0, 60.0


def editor_band(values):
    """Archived st9e native.editor_band (first-order 0.6-3.3 Hz at 30 Hz, filtfilt), copied to avoid importing the vendor
    tree here; equality with the archived function is checked in p703_da_phase.py."""
    b, a = butter(1, [.6 / 15, 3.3 / 15], btype='bandpass')
    return filtfilt(b, a, values, axis=0)


def wrap(x):
    return float(np.angle(np.exp(1j * x)))


def harmonic_phase(v, p, fs, filtered=True):
    v, p = np.asarray(v, np.float64), np.asarray(p, np.float64)
    if filtered:
        v, p = editor_band(v), editor_band(p)
    nper = int(round(10 * fs))
    kw = dict(fs=fs, window='hann', nperseg=nper, noverlap=nper // 2, nfft=nper, detrend='constant', scaling='density',
              average='mean')
    f, spp = welch(p, **kw)
    _, svv = welch(v, **kw)
    _, s = csd(p, v, **kw)
    band = np.flatnonzero((f >= F_LO) & (f <= F_HI))
    k0 = int(band[np.argmax(spp[band])])
    k2 = 2 * k0
    f0 = float(f[k0])
    assert abs(f[k2] - 2 * f0) < 1e-9
    g = np.abs(s) ** 2 / (svv * spp)
    phi1, phi2 = float(np.angle(s[k0])), float(np.angle(s[k2]))
    delta = wrap(phi2 - 2 * phi1)
    dh_s = wrap(phi1 - phi2) / (2 * np.pi * f0)
    g1, g2 = float(g[k0]), float(g[k2])
    if g1 >= G1 and g2 >= G2:
        ad = abs(np.degrees(delta))
        decision = 'negative' if ad >= NEG else 'positive' if ad <= POS else 'undecidable (intermediate Delta)'
    else:
        decision = 'undecidable (coherence)'
    return dict(f0_Hz=f0, f0_bpm=60 * f0, nperseg=nper, segments_length_s=nper / fs, gamma2_f0=g1, gamma2_2f0=g2,
                phi1_deg=float(np.degrees(phi1)), phi2_deg=float(np.degrees(phi2)), Delta_deg=float(np.degrees(delta)),
                d_h_s=float(dh_s), d_h_frames=float(dh_s * fs), decision=decision)
