"""ED1/ED2, F/S, independent signal checks; no learned-model execution."""
from src.st4.base import *
from src.st4.signals import common_hr, filter4
from src.st4.editor_fit import normalize, editor_band
from src.st4.editor_inputs import reflected_indices, parameters, resample
from src.st3.readers import pos as inherited_pos, chrom as inherited_chrom
from src.eval.har import detrend
import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt, windows, periodogram, find_peaks


def ppg_vectors(gt, start, nominal_hr, nominal_target, tau):
    original = normalize(gt[reflected_indices(np.arange(start - 15, start + 315), len(gt))])
    up, down = parameters(nominal_target / nominal_hr)
    target = normalize(resample(gt, start, up, down))
    positions = np.arange(300) + 15 - int(tau)
    return original[positions], target[positions]


def skin_means(clip, masks):
    return np.asarray([frame[mask].mean(axis=0, dtype=np.float64) for frame, mask in zip(clip, masks)])


def green(rgb):
    return rgb[:, 1] / rgb[:, 1].mean() - 1


def band_reader_pos(rgb):
    return filter4(inherited_pos(rgb, filtered=False), [.6, 3.3])


def band_reader_chrom(rgb):
    length = math.ceil(1.6 * 30)
    length += length % 2
    half = length // 2
    nwin = math.floor((len(rgb) - half) / half)
    output = np.zeros(half * (nwin + 1))
    sos = butter(4, [.6, 3.3], btype='bandpass', fs=30, output='sos')
    for i in range(nwin):
        start, end = i * half, i * half + length
        x = rgb[start:end] / rgb[start:end].mean(axis=0)
        xf = sosfiltfilt(sos, 3 * x[:, 0] - 2 * x[:, 1])
        yf = sosfiltfilt(sos, 1.5 * x[:, 0] + x[:, 1] - 1.5 * x[:, 2])
        wave = (xf - xf.std() / yf.std() * yf) * windows.hann(length)
        output[start:start + half] += wave[:half]
        output[start + half:end] = wave[half:]
    return output


def signals(clip, masks, brange=False):
    skin = skin_means(clip, masks)
    rgb = skin if brange else clip.mean(axis=(1, 2), dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(POS=band_reader_pos(rgb) if brange else inherited_pos(rgb),
                    CHROM=band_reader_chrom(rgb) if brange else inherited_chrom(rgb), GREEN=green(skin))


def power(values):
    if not np.isfinite(values).all():
        return np.fft.rfftfreq(512, 1 / 30), np.full(257, np.nan)
    return periodogram(values, fs=30, nfft=512, detrend=False)


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def interval_power(f, p, bpm, half_width=GRID, broad=False):
    mask = np.abs(f * 60 - bpm) <= half_width + 1e-6
    if broad:
        mask &= (f >= .6) & (f <= 3.3)
    return float(p[mask].sum())


def independent_validity(before, after, original, target):
    f, p0 = power(before)
    _, p1 = power(after)
    denom = interval_power(f, p0, original, broad=True)
    tgt = interval_power(f, p1, target, broad=True)
    residual = interval_power(f, p1, original, broad=True)
    broad_idx = np.flatnonzero((f >= .6) & (f <= 3.3))
    local = find_peaks(p1[broad_idx])[0]
    candidates = [int(broad_idx[k]) for k in local
                  if abs(f[broad_idx[k]] * 60 - target) > GRID + 1e-6
                  and abs(f[broad_idx[k]] * 60 - 2 * target) > GRID + 1e-6]
    if candidates:
        peak = max(candidates, key=lambda k: interval_power(f, p1, f[k] * 60, broad=True))
        competitor = interval_power(f, p1, f[peak] * 60, broad=True)
        kappa = ratio(tgt, competitor)
        comp_bpm = float(f[peak] * 60)
    else:
        kappa, comp_bpm = np.inf, np.nan
    rt, rr = ratio(tgt, denom), ratio(residual, denom)
    if not np.isfinite([rt, rr]).all() or np.isnan(kappa):
        classification = 'unavailable'
    elif kappa > 1:
        classification = 'success' if rr <= .25 else 'competition'
    elif abs(comp_bpm - original) <= GRID + 1e-6:
        classification = 'failure'
    else:
        classification = 'uncertain'
    return dict(rho_tgt=rt, rho_res=rr, kappa=float(kappa), competitor_HR=comp_bpm,
                validity_class=classification, original_band_power=denom)


def nonpulsatile(before, after, original_HR, target_HR, g0, condition):
    f, p0 = power(before)
    _, p1 = power(after)
    original_bands = (np.abs(f - original_HR / 60) <= .15) | (np.abs(f - 2 * original_HR / 60) <= .15)
    outside = (f > 0) & ~original_bands
    if condition != 'sham':
        outside &= (np.abs(f - target_HR / 60) > .15) & (np.abs(f - 2 * target_HR / 60) > .15)
    nu_out = ratio(p1[outside].sum(), p0[outside].sum())
    harmonic = filter4(g0, [2 * original_HR / 60 - .15, 2 * original_HR / 60 + .15])
    design = np.column_stack([np.ones(len(g0)), g0, harmonic])
    r0 = before - design @ np.linalg.lstsq(design, before, rcond=None)[0]
    r1 = after - design @ np.linalg.lstsq(design, after, rcond=None)[0]
    _, e0 = power(r0)
    _, e1 = power(r1)
    nu_in = ratio(e1[original_bands].sum(), e0[original_bands].sum())
    broad = (f >= .6) & (f <= 3.3)
    removed = ratio(e0[original_bands].sum() - e1[original_bands].sum(), p0[broad].sum())
    return dict(nu_out=nu_out, nu_in=nu_in, nonpulsatile_removal_fraction=removed)


def edit_float(clip, masks, g0, goal, fit, direction, original_HR, editor):
    check('Editor starts from native uint8', clip.dtype == np.uint8 and len(clip) == 300)
    means = skin_means(clip, masks)
    mu = clip.mean(axis=0, dtype=np.float64)
    out = clip.astype(np.float32)
    if editor == 'ED1':
        coef = np.array([float(fit['a_' + k]) for k in 'RGB'])
        for i in range(300):
            delta = mu * coef * (goal[i] - g0[i])
            out[i] = np.where(masks[i, :, :, None], np.clip(clip[i].astype(np.float64) + delta, 0, 255), clip[i]).astype(np.float32)
        return out, dict(injection_A=np.nan, original_R=np.nan, notch_original_band_removed_fraction=np.nan)
    v = np.asarray(direction, dtype=float)
    normalized = green(means)
    band = [original_HR / 60 - .15, original_HR / 60 + .15]
    R = float(np.sqrt(np.mean(filter4(normalized, band)**2)))
    A = R / (v[1] * np.sqrt(np.mean(goal**2)))
    check('ED2 injection scale finite', np.isfinite(A))
    filters = [butter(4, [multiple * original_HR / 60 - .15, multiple * original_HR / 60 + .15],
                     btype='bandstop', fs=30, output='sos') for multiple in (1, 2)]
    mask_flat = masks.reshape(300, -1)
    pixel_indices = np.flatnonzero(mask_flat.any(axis=0))
    original_flat = clip.reshape(300, -1, 3)
    out_flat = out.reshape(300, -1, 3)
    mu_flat = mu.reshape(-1, 3)
    notch_skin_sum = np.zeros((300, 3))
    for start in range(0, len(pixel_indices), 2048):
        index = pixel_indices[start:start + 2048]
        block = original_flat[:, index].astype(np.float64)
        filtered = block
        for sos in filters:
            filtered = sosfiltfilt(sos, filtered, axis=0)
        active = mask_flat[:, index, None]
        notch_skin_sum += (filtered * active).sum(axis=1)
        injected = filtered + mu_flat[index][None, :, :] * A * v[None, None, :] * goal[:, None, None]
        out_flat[:, index] = np.where(active, np.clip(injected, 0, 255), block).astype(np.float32)
    notch_skin = notch_skin_sum / masks.sum(axis=(1, 2))[:, None]
    f, p0 = power(normalized)
    _, pn = power(green(notch_skin))
    original_band = np.abs(f - original_HR / 60) <= .15
    removed = ratio(p0[original_band].sum() - pn[original_band].sum(), p0[original_band].sum())
    return out, dict(injection_A=float(A), original_R=R, notch_original_band_removed_fraction=removed)


def stochastic_round(value, window_id, combination, condition):
    h = int(hashlib.sha256(f'ST4|SR|{window_id}|{combination}|{condition}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([20261101, h])
    output = np.empty(value.shape, dtype=np.uint8)
    for i in range(len(value)):
        output[i] = np.floor(value[i].astype(np.float64) + rng.random(value.shape[1:])).astype(np.uint8)
    return output


def amplitude(before, after, masks):
    delta = after[masks].astype(np.float64) - before[masks].astype(np.float64)
    channel = np.abs(delta).reshape(-1)
    pixel = np.abs(delta).max(axis=1)
    changed = int((pixel != 0).sum())
    output = dict(n_skin_pixels=len(pixel), n_changed_skin_pixels=changed,
                  changed_skin_fraction=changed / len(pixel))
    for kind, value in [('channel_absolute', channel), ('pixel_max_absolute', pixel)]:
        q = np.quantile(value, [.25, .5, .75])
        output.update({kind + '_p25': float(q[0]), kind + '_median': float(q[1]),
                       kind + '_p75': float(q[2]), kind + '_max': float(value.max())})
    return output


def measure_case(case, clip, masks, gt, fit, direction, brange=False):
    start = int(case['window_index']) * 300
    original = {layer: common_hr(gt[start:start + 300], layer) for layer in BANDS}
    nominal = float(case.get('nominal_original_HR', case.get('original_HAR', original['HAR'])))
    g0, target = ppg_vectors(gt, start, nominal, float(case['target_HR']), int(fit['tau_frames']))
    before = signals(clip, masks, brange)
    target_ref = {layer: common_hr(target, layer) for layer in BANDS}
    overlap = abs(float(case['target_HR']) - 2 * nominal) <= GRID + 1e-6
    base = dict(dataset=case['dataset'], window_id=case['window_id'], population='B_range_gate' if brange else 'fixed_editor_gate',
                ground_truth_path='independent_raw_contact_common_HR;target=injected_shifted_waveform',
                harmonic_overlap=overlap, nominal_target_HR=float(case['target_HR']))
    if brange:
        base.update(task=case['task'], range_low=int(case['range_low']), range_high=int(case['range_high']))
    result = {name: [] for name in ('readout', 'validity', 'sham', 'nonpulsatile', 'amplitude')}
    for condition in (['tgt'] if brange else ['tgt', 'sham']):
        goal = target if condition == 'tgt' else g0
        goal_ref = target_ref if condition == 'tgt' else original
        for editor in ('ED1', 'ED2'):
            value, diagnostic = edit_float(clip, masks, g0, goal, fit, direction, original['HAR'], editor)
            for route in (['S'] if brange else ['F', 'S']):
                combination = editor + '-' + route
                seed_condition = f'tgt-{base["range_low"]}-{base["range_high"]}' if brange else condition
                edited = stochastic_round(value, case['window_id'], combination, seed_condition) if route == 'S' else value
                if editor == 'ED1' and condition == 'sham':
                    check('ED1 F/S sham exact uint8 identity', np.array_equal(edited, clip))
                measured = signals(edited, masks, brange)
                identity = dict(**base, combination=combination, condition=condition)
                if not brange:
                    result['amplitude'].append(dict(**identity, **amplitude(clip, edited, masks), **diagnostic))
                result['nonpulsatile'].append(dict(**identity, **nonpulsatile(before['GREEN'], measured['GREEN'], original['HAR'], goal_ref['HAR'], g0, condition)))
                for reader in ('POS', 'CHROM'):
                    for layer in BANDS:
                        if brange and layer == 'L0' and not 45 <= float(case['target_HR']) <= 150:
                            continue
                        hb = common_hr(before[reader], layer)
                        ha = common_hr(measured[reader], layer)
                        result['readout'].append(dict(**identity, reader=reader, layer=layer,
                            original_HR=original[layer], target_HR=goal_ref[layer], baseline_HR=hb, edited_HR=ha,
                            baseline_hit=bool(abs(hb - original[layer]) <= GRID + 1e-6),
                            target_hit=bool(abs(ha - goal_ref[layer]) <= GRID + 1e-6),
                            original_hit_after=bool(abs(ha - original[layer]) <= GRID + 1e-6),
                            target_absolute_error=abs(ha - goal_ref[layer])))
                        if condition == 'sham':
                            f, p0 = power(before[reader]); _, p1 = power(measured[reader])
                            result['sham'].append(dict(**identity, reader=reader, layer=layer,
                                rho_sham=ratio(interval_power(f, p1, original[layer]), interval_power(f, p0, original[layer])),
                                original_hit_after=bool(abs(ha - original[layer]) <= GRID + 1e-6)))
                if condition == 'tgt':
                    for signal in ('POS', 'CHROM', 'GREEN'):
                        result['validity'].append(dict(**identity, signal=signal,
                            **independent_validity(before[signal], measured[signal], original['HAR'], target_ref['HAR'])))
                if route == 'S':
                    del edited
            del value
    return result
