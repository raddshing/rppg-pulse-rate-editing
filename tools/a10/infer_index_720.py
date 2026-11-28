"""OPS2 7.20 4: manifest and checkpoint index of the 15 B1 checkpoints (A10) of one backbone, for guarded inference (A10
item 6). Plan: U-low arms (0), (i-simple), (i), (i-pc) and U-full-m (0), seeds 20261131-33, from cache/train720/
B1_<backbone>_<arm>_s<seed>; arm names (<arm>-b1low) and (0-b1fullm) keep the keys distinct from every other checkpoint
with the same seeds. Checks and output format: tools/infer_index_709.py main (unchanged; imported).
Usage: infer_index_720.py BACKBONE OUT_DIR   (BACKBONE: PhysNet, FPm or RhythmFormer)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 7.21: published in tools/a10/; infer_index_709.py is in tools/
import infer_index_709 as T  # noqa: E402

C = Path(__file__).resolve().parents[2] / 'cache' / 'train720'
ARMS = (('0-low', '(0-b1low)'), ('i-simple', '(i-simple-b1low)'), ('i', '(i-b1low)'), ('i-pc', '(i-pc-b1low)'),
        ('0-fullm', '(0-b1fullm)'))
b, out = sys.argv[1], sys.argv[2]
assert b in T.BB, b
T.planned = lambda group: [(b, name, s, C / f'B1_{b}_{a}_s{s}', 'A10 item 5 (B1)') for a, name in ARMS for s in T.SEEDS]
sys.argv = [sys.argv[0], 'B1', out]
T.main()
