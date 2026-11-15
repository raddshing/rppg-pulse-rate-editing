"""Editor with an injection-mode argument (brief 02, section 2.1).

`edited` below is the archived st9e/editor.py `edited` (commit 6ebbc9cd) with one
added keyword argument `inject`: the waveform that is injected. The amplitude A is
always computed from `goal` (the original g*(t - tau)), so A is identical to (ii).
With inject=None (default) the function is the original editor.

`ii_context` mirrors the archived pool path (st9g/pool_worker.py, arm ii-h) and adds
`injection` ('original', 'polarity_explicit' or 'polarity_inverted'). For 'polarity_explicit'
(ii-pc) the injected term is s * g*(t - d) with s = -1 and d = D_FRAMES for every item; for
'polarity_inverted' (ii-inv, brief 02 6.18) it is s * g*(t - d) with s = +1 and the same d. For 'lag_shift'
(ii-lag, brief 02 6.19) it is -g*(t - tau_rec - T*/2) with T* = 1800/h* frames (the item's target period); the
fractional delay is linear interpolation of the same target waveform g* (archived ppg_vectors context, 330 samples).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_common import *  # noqa: F401,F403

S_SIGN = -1
S_INV = +1  # (ii-inv): constant inverted polarity, same delay d
SIGN = {'polarity_explicit': S_SIGN, 'polarity_inverted': S_INV}
D_FRAMES = 2  # median |corr|-max lag of the 49 negative-sign recordings (1a natural_recordings.csv)


def edited(clip, notched, face, roi, g0, goal, direction, original, combination, pulse, w, window_id, condition,
           rounding_rng=None, inject=None):
    v = np.asarray(direction, float)
    mu = clip.mean(axis=0, dtype=np.float64)
    if combination == 'ED2AN-S':
        original_roi = green(weighted_rgb(clip, roi))
        R = float(np.sqrt(np.mean(filter4(original_roi, [original / 60 - .15, original / 60 + .15])**2)))
        A = R / (v[1] * np.sqrt(np.mean(goal**2)))
    else:
        R = pulse
        A = R / (v[1] * np.sqrt(np.mean(goal**2)) * w)
    check('Finite nonnegative editor amplitude', np.isfinite(A) and A >= 0)
    wave = goal if inject is None else np.asarray(inject, dtype=np.float64)  # OPS2: injection-mode argument
    value = np.empty(clip.shape, np.float32)
    n_clipped_values = n_clipped_pixels = 0
    for i in range(300):
        if combination == 'ED2W-S':
            x = notched[i].astype(np.float64) * (1 + face[i, :, :, None] * A * v * wave[i])
        elif combination == 'ED2Wadd-S':
            x = notched[i].astype(np.float64) + face[i, :, :, None] * mu * A * v * wave[i]
        else:
            x = notched[i].astype(np.float64) + roi[i, :, :, None] * mu * A * v * wave[i]
        clipped = (x < 0) | (x > 255)
        n_clipped_values += int(clipped.sum())
        n_clipped_pixels += int(clipped.any(axis=2).sum())
        value[i] = np.clip(x, 0, 255).astype(np.float32)
    if rounding_rng is None:
        result = stochastic_round(value, window_id, combination, condition)
    else:
        result = np.empty(value.shape, np.uint8)
        for i, frame in enumerate(value):
            result[i] = np.floor(frame.astype(np.float64) + rounding_rng.random(frame.shape)).astype(np.uint8)
    return result, dict(injection_A=float(A), R_pulse=float(pulse), injection_R=float(R), w=float(w),
                        n_clipped_channel_values=n_clipped_values, n_clipped_pixel_times=n_clipped_pixels)


def lag_shift_wave(gt, start, original, target, tau):
    """-g*(t - tau - T*/2): the archived ppg_vectors target waveform, read at a fractional delay."""
    from src.st9e.native import normalize, parameters, resample  # the helpers of the archived ppg_vectors
    up, down = parameters(target / original)
    tg = normalize(resample(gt, start, up, down))
    pos = np.arange(300) + 15 - int(tau) - (1800.0 / target) / 2
    check('Lag-shift positions inside the target context', pos.min() >= 0 and pos.max() <= len(tg) - 1)
    return -np.interp(pos, np.arange(len(tg)), tg)


def injection_wave(gt, start, original, target, tau, injection):
    """Returns (g0, goal used for A, injected waveform)."""
    g0, gstar = ppg_vectors(gt, start, original, target, tau)
    if injection == 'original':
        return g0, gstar, None
    if injection == 'lag_shift':
        return g0, gstar, lag_shift_wave(gt, start, original, target, tau)
    check('Known injection mode', injection in SIGN)
    _, gd = ppg_vectors(gt, start, original, target, D_FRAMES)
    return g0, gstar, SIGN[injection] * gd


def ii_context(clip, face, roi, notched, physview, physbox, gt, start, original, target, tau, direction,
               rounding_h, window_id, injection='original'):
    """Pool path of st9g/pool_worker.py (arm ii-h) with the injection-mode argument."""
    g0, gstar, inject = injection_wave(gt, start, original, target, tau, injection)
    pulse, w, beta = calibration(physview, crop_native(notched, physbox), crop_native(face, physbox), g0, direction)
    output, diag = edited(clip, notched, face, roi, g0, gstar, direction, original, 'ED2W-S', pulse, w,
                          window_id, 'ii-h', rounding_rng=np.random.default_rng([20261137, int(rounding_h)]), inject=inject)
    label = raw_target_label(gt, start, original, target)
    return output, label, dict(diag, beta=beta, pulse=pulse)
