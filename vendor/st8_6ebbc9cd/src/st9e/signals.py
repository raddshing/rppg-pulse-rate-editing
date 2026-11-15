"""Fixed independent reference estimators; no model or batch-dependent labels."""
from src.st9e.base import BANDS, GRID
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt, find_peaks, periodogram
from src.eval import grid


def filter4(values, band, kind='bandpass'):
    return sosfiltfilt(butter(4, band, btype=kind, fs=30, output='sos'), np.asarray(values, dtype=np.float64), axis=0)


def common_hr(values, layer):
    values = np.asarray(values)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    # Explicit current R_c/common-HR exception to the new-estimator filter rule.
    band = BANDS[layer]
    return grid._estimate(grid._prepare(values, False, band, 'default', 30), 'periodogram', band, 30)


def parabola(y, peak):
    den = y[peak - 1] - 2 * y[peak] + y[peak + 1]
    delta = .5 * (y[peak - 1] - y[peak + 1]) / den if den else 0.
    return float(peak + np.clip(delta, -.5, .5))


def classify(h_fft, h_acf, h_ibi):
    if not np.isfinite([h_fft, h_acf, h_ibi]).all():
        return 'unavailable'
    if abs(h_acf - h_fft) <= GRID + 1e-6 and abs(h_ibi - h_fft) <= 2 * GRID + 1e-6:
        return 'agreement'
    if abs(h_acf - h_fft / 2) <= GRID + 1e-6 or abs(h_ibi - h_fft / 2) <= 2 * GRID + 1e-6:
        return 'harmonic_suspect'
    return 'disagreement'


def reference_estimators(raw, h_fft):
    raw = np.asarray(raw, dtype=float)
    result = dict(h_ACF=np.nan, h_IBI=np.nan, ACF_ratio=np.nan, SHR=np.nan, SQI_t=np.nan,
                  HR_slope_bpm_per_s=np.nan, HR_instant_sd_bpm=np.nan, n_IBI_peaks=0)
    if len(raw) != 300 or not np.isfinite(raw).all():
        return result
    acf_signal = filter4(raw, [.5, 8.])
    # 120 Hz grid restricted to the observed support: no endpoint extrapolation.
    interpolated = CubicSpline(np.arange(300), acf_signal)(np.arange(1197) / 4)
    interpolated -= interpolated.mean()
    acf = np.correlate(interpolated, interpolated, mode='full')[len(interpolated) - 1:] / len(interpolated)
    if acf[0] > 0:
        acf /= acf[0]
        lo, hi = int(np.ceil(.30 * 120)), int(np.floor(1.67 * 120))
        peaks = find_peaks(acf[lo:hi + 1])[0] + lo
        if len(peaks):
            best = int(peaks[np.argmax(acf[peaks])])
            result['h_ACF'] = 60 * 120 / parabola(acf, best)
        if np.isfinite(h_fft) and h_fft > 0:
            lag = 60 / h_fft * 120
            num, den = np.interp([lag, 2 * lag], np.arange(len(acf)), acf)
            result['ACF_ratio'] = num / den if den != 0 else np.nan
    ibi_signal = filter4(raw, [.6, 3.3])
    peaks = find_peaks(ibi_signal, distance=9, prominence=.3 * np.std(ibi_signal, ddof=0))[0]
    result['n_IBI_peaks'] = len(peaks)
    if len(peaks) >= 6:
        locations = np.asarray([parabola(ibi_signal, int(p)) / 30 for p in peaks])
        intervals = np.diff(locations)
        instants = 60 / intervals
        midpoints = (locations[1:] + locations[:-1]) / 2
        result['h_IBI'] = 60 / np.median(intervals)
        result['HR_slope_bpm_per_s'] = np.polyfit(midpoints, instants, 1)[0]
        result['HR_instant_sd_bpm'] = np.std(instants, ddof=0)
        lowpass = filter4(raw, 8., 'lowpass')
        boundaries = (locations[:-1] + locations[1:]) / 2
        beats = np.array([np.interp(np.linspace(a, b, 101), np.arange(300) / 30, lowpass)
                          for a, b in zip(boundaries[:-1], boundaries[1:])])
        template = beats.mean(axis=0)
        corrs = [np.corrcoef(beat, template)[0, 1] for beat in beats
                 if np.std(beat) > 0 and np.std(template) > 0]
        if corrs:
            result['SQI_t'] = np.median(corrs)
    if np.isfinite(h_fft) and h_fft / 2 >= 36:
        f, power = periodogram(ibi_signal, fs=30, nfft=512, detrend=False)
        inband = (f >= .6) & (f <= 3.3)
        num = power[inband & (np.abs(f * 60 - h_fft / 2) <= GRID + 1e-6)].sum()
        den = power[inband & (np.abs(f * 60 - h_fft) <= GRID + 1e-6)].sum()
        result['SHR'] = num / den if den > 0 else np.nan
    return result
