"""Brief 02 7.03: post-hoc analyses 3 on existing outputs and training/development data only (no training, no inference,
no re-inference; CPU, one thread per process, nice 19).

Same rules and loaders as 7.01/7.02 (tools/posthoc701/p701_common.py, imported unchanged): test data are the saved stage-2
predictions, step-(b) readouts and stage-1 references of the test run step (c) (cache/test); development data are the
guarded predictions of rehearsal part 2; windows, validity, support, populations and the crossed subject x seed bootstrap
(b1/boot.py: B = 10,000, draws default_rng(20261110) on the numeric subject order) are those of eval5/stage3_analyze.py.
Outputs go to outputs/posthoc703/<part>/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'posthoc701'))
import p701_common as C  # noqa: E402

from p701_common import (B, DEV, SA, SEEDS, SRC_DEV, SRC_TEST, TEST, VENDOR, WORK, boot, json, load,  # noqa: E402,F401
                         main_population, np, pd, primary_har, summarize_nan, time)

C.OUT = WORK / 'outputs' / 'posthoc703'
OUT = C.OUT
write, stamp = C.write, C.stamp
NAMES = {'MMPD': 'MMPD-B', 'MCD': 'MCD-rPPG test'}
CONF = SA.CONF  # C1-C4
