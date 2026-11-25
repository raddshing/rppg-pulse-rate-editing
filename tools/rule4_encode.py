"""Diagnostic rule 4 (docs/STAGE5_DEV_DIAGNOSTICS.md; brief 02 6.89 다 2): re-encoding of UBFC-rPPG and PURE 09-10.

Recordings: the 54 of the existing UBFC/PURE evaluation list (main study outputs/st15u_recordings.csv, read only):
UBFC-rPPG DATASET_2 subject* (42; vid.avi, raw bgr24) and PURE 09-01..10-06 (12; PNG frames, names in time order).
Encoder: ffmpeg 7.0.2 (static build in the main environment's imageio_ffmpeg), MPEG-4 Part 2 encoder 'mpeg4', fourcc FMP4
(as the MCD-rPPG bitstreams), no B-frames (-bf 0), scene-cut key frames disabled (-sc_threshold 1000000000), one thread;
rate control with -qmin 1 -lmin 1 so that low-complexity recordings can reach the target bit rate (with the defaults PURE 09-01
reached about 0.39 of it in a test).
Frames are encoded 1:1 (no frame dropped or repeated) at a nominal 30 fps (the processing rate of the existing evaluation).
Bit rate: bits per pixel of the MCD-rPPG development videos (median, outputs/rule4/mcd_dev_bpp.json) x 640 x 480 x 30.
Conditions: gop10, gop12, gop15 (-g N), nokey (-g 100000: no periodic key frame within a recording). The as-distributed
control is the original video and is not encoded.
For the GOP check by tools/mcd_gop.py (unchanged) each condition folder also gets db.csv and meta files in the MCD-rPPG
layout: patient_id = recording, step = condition, view front, camera FullHDwebcam, meta timestamps at 1/30 s.
Outputs (server only; face-derived): cache/rule4/enc/<condition>/; log outputs/rule4/encode_log.csv.
Usage: rule4_encode.py CONDITION[,CONDITION] [RECORDING[,RECORDING]]
"""
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
FFMPEG = CFG['ffmpeg']
UBFC = Path(CFG['ubfc2_raw'])
PURE = Path(CFG['pure_raw'])
LIST = Path(CFG['october_project']) / 'outputs' / 'st15u_recordings.csv'
ENC = WORK / 'cache' / 'rule4' / 'enc'
FPS = 30
GOP = {'gop10': 10, 'gop12': 12, 'gop15': 15, 'nokey': 100000}


def recordings():
    with LIST.open(newline='') as f:
        return [(r['probe_dataset'], r['recording_id'], int(r['frame_count'])) for r in csv.DictReader(f)]


def source_args(ds, rec):
    if ds == 'U':
        return ['-r', str(FPS), '-i', str(UBFC / rec / 'vid.avi')]
    return ['-framerate', str(FPS), '-pattern_type', 'glob', '-i', str(PURE / rec / '*.png')]


def encode(job):
    ds, rec, n, cond, kbps = job
    out = ENC / cond / f'{rec}.avi'
    cmd = [FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', *source_args(ds, rec), '-fps_mode', 'passthrough',
           '-an', '-c:v', 'mpeg4', '-vtag', 'FMP4', '-bf', '0', '-g', str(GOP[cond]), '-sc_threshold', '1000000000',
           '-b:v', f'{kbps}k', '-qmin', '1', '-lmin', '1', '-pix_fmt', 'yuv420p', '-threads', '1', str(out)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    meta = ENC / cond / f'{rec}_meta.txt'
    base = datetime(2026, 1, 1)
    meta.write_text(''.join(f'{i} {(base + timedelta(seconds=i / FPS)).isoformat(sep=" ", timespec="microseconds")}\n'
                            for i in range(n)))
    return dict(dataset=ds, recording_id=rec, condition=cond, frames_source=n, exit=p.returncode, stderr=p.stderr.strip()[-300:],
                bytes=out.stat().st_size if out.exists() else 0, seconds=round(time.time() - t0, 2),
                command=' '.join(c if not c.startswith('/') else Path(c).name for c in cmd))


def main():
    conds = sys.argv[1].split(',')
    only = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
    bpp = json.loads((WORK / 'outputs' / 'rule4' / 'mcd_dev_bpp.json').read_text())['bits_per_pixel_median']
    kbps = round(bpp * 640 * 480 * FPS / 1000, 3)
    recs = [r for r in recordings() if only is None or r[1] in only]
    jobs = []
    for c in conds:
        (ENC / c).mkdir(parents=True, exist_ok=True)
        with (ENC / c / 'db.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['patient_id', 'step', 'view', 'camera', 'video', 'meta'])
            for ds, rec, n in recs:
                w.writerow([rec, c, 'front', 'FullHDwebcam', f'{rec}.avi', f'{rec}_meta.txt'])
        with (ENC / c / 'list.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['recording_id'])
            for ds, rec, n in recs:
                w.writerow([f'{rec}_{c}'])
        jobs += [(ds, rec, n, c, kbps) for ds, rec, n in recs]
    log = WORK / 'outputs' / 'rule4' / 'encode_log.csv'
    new = not log.exists()
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '3'))) as ex, log.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'recording_id', 'condition', 'frames_source', 'exit', 'stderr', 'bytes',
                                          'seconds', 'command', 'target_kbps'])
        if new:
            w.writeheader()
        for r in ex.map(encode, jobs):
            w.writerow(dict(r, target_kbps=kbps))
            f.flush()
            print(r['condition'], r['recording_id'], 'exit', r['exit'], r['bytes'], flush=True)


if __name__ == '__main__':
    main()
