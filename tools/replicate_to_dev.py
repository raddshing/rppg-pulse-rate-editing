"""Brief 02 6.67 마 (diagnostic rule 6): stage-2 predictions of the (i-simple) replicate runs written in the development
layout, by eval5/stage2_to_dev.py unchanged except for its two lookups: the run of a key (view, '(i-simple-rep)', seed) is
ops2_<B>_i-simple-rep_s<seed> (b1/model_sets.isimplerep; not in the b1/analyze.py registry) and its MCD group is
isimplerep_<B>. Same array formats, same root layout. Usage: replicate_to_dev.py DATASET ROOT STAGE2_DIR
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'eval5'))
sys.path.insert(0, str(WORK / 'b1'))
import stage2_to_dev  # noqa: E402

VIEWS = ('FPm', 'PhysNet', 'RhythmFormer')
SEEDS = (20261131, 20261132, 20261133)


def registry():
    return {(v, '(i-simple-rep)', s): [f'ops2_{v}_i-simple-rep_s{s}'] for v in VIEWS for s in SEEDS}


def mcd_group_of(run_ids):
    return {r: [f"isimplerep_{r.split('_')[1]}"] for r in run_ids}


stage2_to_dev.registry = registry
stage2_to_dev.mcd_group_of = mcd_group_of

if __name__ == '__main__':
    stage2_to_dev.main()
