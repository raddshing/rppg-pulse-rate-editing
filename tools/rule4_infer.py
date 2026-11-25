"""Diagnostic rule 4 (brief 02 6.89 다 2, 6.90 다): inference on the re-encoded and as-distributed UBFC-rPPG / PURE 09-10 videos
through the MCD-rPPG development input path of b1/mcd_infer.py, unchanged (its job(): HC 1.5 face box on frame 0 for the 72 and
128 views, INTER_AREA, endpoint-linear frame times over the given span, interpolation to 30 Hz, profile transform, full
non-overlapping chunks; inference client with the allocator-OOM guard of audit/infer_client.py; model RNG reset per recording).

What this wrapper sets, nothing else:
- recordings: the 54 of the existing UBFC/PURE evaluation (main study outputs/st15u_recordings.csv, read only);
- video: condition gop10/gop12/gop15/nokey -> cache/rule4/enc/<condition>/<recording>.avi; condition distributed ->
  the original UBFC vid.avi, or the PURE recording folder, whose PNG frames are read with cv2.imread in name (= time) order
  through a VideoCapture-like reader (frames BGR as VideoCapture returns them);
- span_seconds = (frames - 1) / 30: the frames are placed at the existing evaluation's processing rate of 30 Hz, the rate
  at which the videos were encoded (so the 30 Hz interpolation leaves the frames unchanged);
- models: the 21 checkpoints of rule 4 (seeds 20261131-33): RhythmFormer (0), (i), (ii), (i-pc), (ii-pc), PhysNet (0),
  FactorizePhys matched (FPm) (0); one group 'rule4';
- output: cache/rule4/waveforms/<condition>/rule4/<recording>.npz (OPS2_MCD_CACHE), log outputs/rule4/infer_log_<condition>.csv.
Usage: OPS2_GPU=<gpu> rule4_infer.py CONDITION
"""
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
COND = sys.argv[1] if __name__ == '__main__' else os.environ['OPS2_RULE4_COND']
os.environ['OPS2_RULE4_COND'] = COND
os.environ['OPS2_MCD_CACHE'] = str(WORK / 'cache' / 'rule4' / 'waveforms' / COND)
sys.path.insert(0, str(WORK / 'b1'))
import mcd_infer  # noqa: E402  (sets the thread environment first, as in b1)

import csv  # noqa: E402
import time  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait  # noqa: E402

import numpy as np  # noqa: E402

import json  # 7.08
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
UBFC = Path(CFG['ubfc2_raw'])
PURE = Path(CFG['pure_raw'])
LIST = Path(CFG['october_project']) / 'outputs' / 'st15u_recordings.csv'
ENC = WORK / 'cache' / 'rule4' / 'enc'
RUNS = {('RhythmFormer', s) for s in ('(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)')} | {('PhysNet', '(0)'), ('FactorizePhys (matched)', '(0)')}
SEEDS = (20261131, 20261132, 20261133)


class PngSequence:
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob('*.png'))
        self.i = 0

    def isOpened(self):
        return bool(self.files)

    def read(self):
        import cv2
        if self.i >= len(self.files):
            return False, None
        frame = cv2.imread(str(self.files[self.i]), cv2.IMREAD_COLOR)
        self.i += 1
        return frame is not None, frame

    def release(self):
        pass


def init_worker():
    mcd_infer.init_worker()
    import cv2
    original = cv2.VideoCapture
    cv2.VideoCapture = lambda p, *a: PngSequence(p) if os.path.isdir(p) else original(p, *a)


def models():
    import analyze
    out = [dict(r) for k, r in sorted(analyze.runs().items()) if (r['backbone'], r['strategy']) in RUNS and int(r['seed']) in SEEDS]
    assert len(out) == 21, len(out)
    return out


def recordings():
    rows = []
    with LIST.open(newline='') as f:
        for r in csv.DictReader(f):
            rid, n = r['recording_id'], int(r['frame_count'])
            if COND == 'distributed':
                video = str(UBFC / rid / 'vid.avi') if r['probe_dataset'] == 'U' else str(PURE / rid)
            else:
                video = str(ENC / COND / f'{rid}.avi')
            rows.append(dict(recording_id=rid, probe_dataset=r['probe_dataset'], frame_count=n, video=video,
                             span_seconds=(n - 1) / 30.0, draft_rule=False))
    return rows


def main():
    rows = recordings()
    ms = models()
    cache = Path(os.environ['OPS2_MCD_CACHE']) / 'rule4'
    log = WORK / 'outputs' / 'rule4' / f'infer_log_{COND}.csv'
    fields = ['recording_id', 'frame_count', 'decoded_frames', 'span_seconds', 'grid_samples', 'seconds', 'status', 'finished']
    if not log.exists():
        with log.open('w', newline='') as f:
            csv.writer(f).writerow(fields)
    todo = [r for r in rows if not (cache / (r['recording_id'] + '.npz')).exists()]
    print(COND, 'recordings', len(rows), 'to do', len(todo), 'models', len(ms), flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '3')), mp_context=mp.get_context('spawn'),
                             initializer=init_worker) as ex:
        futs = {ex.submit(mcd_infer.job, (r, {'rule4': ms})): r for r in todo}
        for n, fut in enumerate(futs, 1):
            r = futs[fut]
            try:
                m = fut.result()
            except Exception as e:  # recorded, not retried
                m = dict(status='failed: ' + repr(e)[:300])
            with log.open('a', newline='') as f:
                csv.writer(f).writerow([r['recording_id'], r['frame_count'], m.get('decoded_frames'), m.get('span_seconds'),
                                        m.get('grid_samples'), m.get('seconds'), m.get('status'), time.strftime('%FT%T%z')])
            print(f"{COND} {n}/{len(todo)} {r['recording_id']} {m.get('status')} {time.time() - started:.0f}s", flush=True)
    print(COND, 'done', flush=True)


if __name__ == '__main__':
    main()
