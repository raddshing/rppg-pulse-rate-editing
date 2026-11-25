"""Diagnostic rule 1 (brief 02 6.89 다 3, 6.90 라): run one of the unchanged development inference/readout scripts
(b1/mmpd_infer.py, b1/mcd_infer.py, b1/mmpd_a.py, b1/mcd_windows.py) for the Pearson-only RhythmFormer runs.

b1/model_sets.py is in the A4 hash list and stays unchanged. Its lab groups ('lab-<arm>_<backbone>', the three trained runs
in cache/train/<backbone>_<arm>_s*) accept only the arms of LAB_ARMS; here the two rule-1 arms 'ii-pearson' and
'ii-pc-pearson' are added to that tuple in this process only, before the script's main() runs. Nothing else changes:
the script's module code, checkpoints, windows, readout and RNG handling are those of the standard runs.
The target script is imported as a module (its own thread settings run first) so that its worker functions stay importable
in spawned workers; its main() is then called with the given arguments.
Usage: rule1_run.py SCRIPT(mmpd_infer|mcd_infer|mmpd_a|mcd_windows) ARGS...
"""
import importlib
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
RULE1_ARMS = ('ii-pearson', 'ii-pc-pearson')


def main():
    name, args = sys.argv[1], sys.argv[2:]
    assert name in ('mmpd_infer', 'mcd_infer', 'mmpd_a', 'mcd_windows'), name
    sys.path.insert(0, str(WORK / 'b1'))
    mod = importlib.import_module(name)  # module code first (thread settings precede numpy/torch in the inference scripts)
    import model_sets
    assert Path(model_sets.__file__).resolve() == (WORK / 'b1' / 'model_sets.py').resolve()
    model_sets.LAB_ARMS = tuple(model_sets.LAB_ARMS) + tuple(a for a in RULE1_ARMS if a not in model_sets.LAB_ARMS)
    for g in args:
        if g.startswith('lab-'):
            arm, backbone = g[4:].rsplit('_', 1)
            assert arm in RULE1_ARMS and backbone == 'RhythmFormer', g
    sys.argv = [str(WORK / 'b1' / f'{name}.py')] + args
    mod.main()


if __name__ == '__main__':
    main()
