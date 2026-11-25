"""Brief 02 7.02: post-hoc analyses 2 on existing outputs only (no training, no inference, no re-inference; CPU, one thread).

Same rules and loaders as 7.01 (tools/posthoc701/p701_common.py, imported unchanged): test data are the saved stage-2
predictions, step-(b) readouts and stage-1 references of the test run step (c) (cache/test); windows, validity, support,
populations and the crossed subject x seed bootstrap (b1/boot.py: B = 10,000, draws default_rng(20261110) on the numeric
subject order) are those of eval5/stage3_analyze.py. Outputs go to outputs/posthoc702/<part>/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'posthoc701'))
import p701_common as C  # noqa: E402

from p701_common import (B, DEV, SA, SEEDS, SRC_DEV, SRC_TEST, TEST, VENDOR, WORK, boot, contrast_nan, json, load,  # noqa: E402,F401
                         main_population, np, pd, primary_har, summarize_nan, time)

C.OUT = WORK / 'outputs' / 'posthoc702'
OUT = C.OUT
write, stamp = C.write, C.stamp
GRID = SA.GRID          # 3.515625 bpm, one bin of the 512-point periodogram at 30 Hz
TOL = 1e-6              # the hit rule of the test code (eval5/families.py fam_m4: |e| <= GRID + 1e-6)
SRC_POOL = 'training pools (labels)'
SRC_B9 = 'development outputs (B9, 11-12)'
NAMES = {'MMPD': 'MMPD-B', 'MCD': 'MCD-rPPG test'}
CONF = SA.CONF  # C1-C4


def hit(r, p):
    """Readout within one bin of the reference (fam_m4 hit_pm1bin)."""
    return (np.abs(p - r) <= GRID + TOL).astype(float)


def half(r, p):
    """Readout within one bin of half the reference (fam_m4 half_HR_error)."""
    return (np.abs(p - r / 2) <= GRID + TOL).astype(float)
