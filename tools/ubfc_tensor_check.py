"""7.09b E1: tensors of the UBFC originals and family-i items rebuilt in the training interpreter (repro, numpy 1.22).

Originals: train/ubfc_tensor.original_tensor (the function train/train_ubfc.py uses) against ops2_input_sha256 /
ops2_label_sha256 of the generator and against the October UBFC SHA-256 (PhysNet, RhythmFormer). Family i: train/u8_store
.item_input (the training loader path) against the generator SHA. Writes outputs/709b/ubfc_tensor_check.csv (one row per
manifest). Usage (repro interpreter): ubfc_tensor_check.py MANIFEST [MANIFEST ...]
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
import numpy as np  # noqa: E402
import u8_store  # noqa: E402
import ubfc_tensor  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'outputs' / '709b' / 'ubfc_tensor_check.csv'


def main():
    res = []
    for man in sys.argv[1:]:
        rows = [r for r in csv.DictReader(open(man)) if str(r.get('generation_excluded', 'False')) != 'True']
        name = Path(man).name
        backbone = name.split('_')[0]
        cache = {}
        eq_x = eq_y = oct_x = oct_y = n_oct = 0
        for r in rows:
            if 'ubfc_view_path' in r:
                k = r['ubfc_view_path']
                if k not in cache:
                    cache = {k: (u8_store.read_array(k), u8_store.read_array(r['ubfc_contact_path']),
                                 u8_store.read_array(r['ubfc_stats_path']) if r['ubfc_stats_path'] else None)}
                v, gt, st = cache[k]
                x, y = ubfc_tensor.original_tensor(backbone, v, st, gt, int(r['start_frame']), int(r['chunk_length']))
            else:
                x, y = u8_store.item_input(backbone, r)
            sx, sy = u8_store.npy_sha(x), u8_store.npy_sha(y)
            eq_x += sx == r['ops2_input_sha256']
            eq_y += sy == r['ops2_label_sha256']
            if r.get('october_input_sha256'):
                n_oct += 1
                oct_x += sx == r['october_input_sha256']
                oct_y += sy == r['october_label_sha256']
        res.append(dict(manifest=name, numpy=np.__version__, rows=len(rows), input_equal_generator=eq_x,
                        label_equal_generator=eq_y, rows_with_october_sha=n_oct, input_equal_october=oct_x,
                        label_equal_october=oct_y))
        print(res[-1], flush=True)
    new = not OUT.exists()
    with open(OUT, 'a', newline='') as f:
        w = csv.DictWriter(f, list(res[0]))
        if new:
            w.writeheader()
        w.writerows(res)


if __name__ == '__main__':
    main()
