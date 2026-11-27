"""OPS2 7.11 나: manifest and checkpoint index of the 21 E1 checkpoints of one backbone (7 arms x 3 seeds), so that E1
inference can run per backbone. The plan, checks and output format are those of tools/infer_index_709.py group E1
(unchanged; imported), restricted to one backbone.
Usage: infer_index_711.py BACKBONE OUT_DIR   (BACKBONE: PhysNet, FPm or RhythmFormer)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_index_709 as T  # noqa: E402

b, out = sys.argv[1], sys.argv[2]
assert b in T.BB, b
_all = T.planned('E1')
T.planned = lambda group: [r for r in _all if r[0] == b]
assert len(T.planned('E1')) == 21
sys.argv = [sys.argv[0], 'E1', out]
T.main()
