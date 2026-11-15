"""Brief 02 6.67 다 5: between steps (b) and (c), MCD-rPPG. The recordings that have a window whose HR readout differs from the
October prediction (eval5/stage3_check.py october_differences.csv) are inferred again under the same guards with the same
stage-2 command (eval5/stage2_predict.py, restricted to those recordings with --recordings), into a separate folder; every
prediction file of the rerun is compared byte for byte with the step-(b) file of the same key and recording.
No differing window: nothing is run (recorded). Writes OUT/rerun_recordings.csv (the recording list given to stage 2),
OUT/rerun_comparison.csv (key, recording, identical, max_abs_difference) and OUT/rerun_summary.json.
Usage: rerun_differing.py --check CHECK_DATASET_DIR --stage2 STAGE2_DATASET_DIR --out OUT -- STAGE2_ARGS...
  (STAGE2_ARGS: the step-(b) arguments of eval5/stage2_predict.py without --recordings and --out)
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]


def main():
    argv = sys.argv[1:]
    cut = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', required=True), ap.add_argument('--stage2', required=True), ap.add_argument('--out', required=True)
    a = ap.parse_args(argv[:cut])
    s2args = argv[cut + 1:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.check) / 'october_differences.csv')
    d = d[d.window_id.notna() & d.window_id.astype(str).ne('')]
    recs = sorted({str(w).split('|')[0] for w in d.window_id})
    with open(out / 'rerun_recordings.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['recording_id'])
        w.writerows([[r] for r in recs])
    summary = dict(differing_windows=len(d), recordings=len(recs))
    if not recs:
        summary['result'] = 'no differing window; nothing re-run'
        (out / 'rerun_summary.json').write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
        return
    cmd = [sys.executable, '-B', '-s', str(WORK / 'eval5/stage2_predict.py')] + s2args + ['--recordings', str(out / 'rerun_recordings.csv'),
                                                                                        '--out', str(out / 'stage2')]
    rc = subprocess.run(cmd, cwd=str(WORK)).returncode
    summary['stage2_exit'] = rc
    rows = []
    if rc == 0:
        ds = 'MCD'
        for kd in sorted((out / 'stage2' / ds / 'pred').iterdir()):
            for f in sorted(kd.glob('*.npy')):
                g = Path(a.stage2) / 'pred' / kd.name / f.name
                if not g.exists():
                    rows.append(dict(key=kd.name, recording_id=f.stem, identical=False, max_abs_difference='', note='no step-(b) file'))
                    continue
                same = f.read_bytes() == g.read_bytes()
                x, y = np.load(f), np.load(g)
                md = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape else float('nan')
                rows.append(dict(key=kd.name, recording_id=f.stem, identical=same, max_abs_difference=md, note=''))
    pd.DataFrame(rows, columns=['key', 'recording_id', 'identical', 'max_abs_difference', 'note']).to_csv(out / 'rerun_comparison.csv', index=False)
    r = pd.DataFrame(rows)
    summary.update(files=len(r), identical=int(r.identical.sum()) if len(r) else 0,
                   result='all identical' if len(r) and r.identical.all() else ('differences' if len(r) else 'rerun failed'))
    (out / 'rerun_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    sys.exit(0 if summary['result'] in ('all identical',) else 3)


if __name__ == '__main__':
    main()
