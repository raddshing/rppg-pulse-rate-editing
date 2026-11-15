"""Determinism check of stage 2 (ACCESS_OPS2_BRIEF_02 6.46 가 1): the same stage-2 command run twice; every prediction file
of the second run is compared with the file of the same output key and recording of the first run.

Per output key: files compared, files byte-identical, largest absolute difference of the predictions, and, when a file
differs, whether the predictions rounded to 1e-4 (np.round(x, 4) as float32) are identical; the SHA-256 of the rounded
predictions of each run is then given per key (sorted recording names, one array after another).
Usage: compare_runs.py RUN1_DATASET_DIR RUN2_DATASET_DIR OUT.csv
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for kd in sorted((b / 'pred').iterdir()):
        files = sorted(kd.glob('*.npy'))
        same = 0
        maxd = 0.0
        ha, hb = hashlib.sha256(), hashlib.sha256()
        missing = 0
        for f in files:
            g = a / 'pred' / kd.name / f.name
            if not g.exists():
                missing += 1
                continue
            if f.read_bytes() == g.read_bytes():
                same += 1
            x, y = np.load(g), np.load(f)
            if x.shape != y.shape:
                maxd = float('inf')
                continue
            maxd = max(maxd, float(np.max(np.abs(x.astype(np.float64) - y))) if x.size else 0.0)
            ha.update(np.round(x, 4).astype(np.float32).tobytes())
            hb.update(np.round(y, 4).astype(np.float32).tobytes())
        rows.append(dict(key=kd.name, files=len(files), missing_in_run1=missing, byte_identical=same,
                         max_abs_difference=maxd, rounded_1e4_sha256_run1=ha.hexdigest(), rounded_1e4_sha256_run2=hb.hexdigest(),
                         rounded_identical=ha.hexdigest() == hb.hexdigest()))
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('keys', len(rows), 'files', sum(r['files'] for r in rows), 'byte-identical', sum(r['byte_identical'] for r in rows),
          'max abs difference', max(r['max_abs_difference'] for r in rows))


if __name__ == '__main__':
    main()
