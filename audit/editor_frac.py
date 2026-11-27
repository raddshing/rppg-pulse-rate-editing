"""7.09b E7: partial misalignment, injection modes between (ii-pc) and (ii-lag) (audit/editor_pc.py unchanged).

Mode 'lag_frac_<alpha>' injects -g*(t - d - alpha (tau + T*/2 - d)), with d = D_FRAMES (2) and T* = 1800/h* frames:
alpha = 0 gives the (ii-pc) term -g*(t - d), alpha = 1 the (ii-lag) term -g*(t - tau - T*/2). The fractional delay is read
from the same 330-sample target context by linear interpolation, as lag_shift_wave does. The amplitude A, g0, g*(t - tau),
notch, masks, calibration, rounding and label are those of the other modes (gen_pool.job), so only the injected term differs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import editor_pc as E  # noqa: E402
import numpy as np  # noqa: E402

PREFIX = 'lag_frac_'


def alpha_of(mode):
    a = float(mode[len(PREFIX):])
    assert 0.0 <= a <= 1.0
    return a


def lag_frac_wave(gt, start, original, target, tau, alpha, d=E.D_FRAMES):
    from src.st9e.native import normalize, parameters, resample  # the helpers of lag_shift_wave
    up, down = parameters(target / original)
    tg = normalize(resample(gt, start, up, down))
    delay = d + alpha * (int(tau) + (1800.0 / target) / 2 - d)
    pos = np.arange(300) + 15 - delay
    E.check('Partial-lag positions inside the target context', pos.min() >= 0 and pos.max() <= len(tg) - 1)
    return -np.interp(pos, np.arange(len(tg)), tg)


def injection_wave(gt, start, original, target, tau, injection):
    """editor_pc.injection_wave with the added modes lag_frac_<alpha>; all other modes unchanged."""
    if isinstance(injection, str) and injection.startswith(PREFIX):
        g0, gstar = E.ppg_vectors(gt, start, original, target, tau)
        return g0, gstar, lag_frac_wave(gt, start, original, target, tau, alpha_of(injection))
    return E.injection_wave(gt, start, original, target, tau, injection)
