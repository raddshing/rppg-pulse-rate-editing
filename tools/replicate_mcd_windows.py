"""Brief 02 6.67 마 (diagnostic rule 6): b1/mcd_windows.py unchanged, for the replicate groups isimplerep_<B>; its run -> view
map (views(): roster and all_trained()) is extended by the replicate runs (b1/model_sets.isimplerep), which are kept out of
all_trained(). Usage: replicate_mcd_windows.py GROUP...   (env as b1/mcd_windows.py: OPS2_MCD_CACHE, OPS2_WINDOWS_OUT)
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'b1'))
import mcd_windows  # noqa: E402
import model_sets  # noqa: E402

_views = mcd_windows.views


def views():
    out = _views()
    for b in ('FPm', 'PhysNet', 'RhythmFormer'):
        for r in model_sets.isimplerep(b) or []:
            out[r['run_id']] = r['view']
    return out


mcd_windows.views = views

if __name__ == '__main__':
    mcd_windows.main()
