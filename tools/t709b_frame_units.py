"""7.09b 나-1: frame units of the UBFC-rPPG pipeline (training data only; no video pixels are read).

For each of the 42 recordings: the contact PPG (ground_truth.txt line 1) has one value per video frame (index frame count),
and its timestamps (line 3) give the real frame rate. Every chunk h0 of the October UBFC chunk list is recomputed with the
readout used by the editor and the pools (common_hr on 300 frames, HAR band, fs = 30 frame units) and compared with the
stored h0. Writes outputs/709b/frame_units_recordings.csv and frame_units_chunks.csv and prints a summary.
Usage: python -B t709b_frame_units.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'audit'))
from a1_common import common_hr  # noqa: E402  (src.st9e.signals.common_hr, fs 30)

import json  # 7.19
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
OCT = Path(CFG['october_project'])
OUT = WORK / 'outputs' / '709b'


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = {r['recording_id']: r for r in rows(OCT / 'data/index/ubfc_index_st1.csv')}
    chunks = rows(OCT / 'outputs/st30r_ubfc_chunks.csv')
    rec_out, chunk_out = [], []
    gts = {}
    for rid, r in sorted(index.items()):
        lines = open(r['label_path']).read().splitlines()
        ppg = np.asarray([float(x) for x in lines[0].split()])
        ts = np.asarray([float(x) for x in lines[2].split()])
        gts[rid] = ppg
        d = np.diff(ts)
        fps_ts = (len(ts) - 1) / (ts[-1] - ts[0])
        rec_out.append(dict(recording_id=rid, frame_count=int(r['frame_count']), ppg_values=len(ppg), timestamps=len(ts),
                            ppg_equals_frames=len(ppg) == int(r['frame_count']), fps_container=r['fps'],
                            fps_timestamps=fps_ts, median_frame_interval_s=float(np.median(d)),
                            nonincreasing_intervals=int((d <= 0).sum()), factor_30_over_fps=30.0 / fps_ts,
                            anomaly_flags=r['anomaly_flags']))
    for c in chunks:
        gt = gts[c['recording_id']]
        cs = int(c['context_start'])
        h = float(common_hr(gt[cs:cs + 300], 'HAR'))
        chunk_out.append(dict(view=c['view'], chunk_id=c['chunk_id'], context_start=cs, quality_pass=c['quality_pass'],
                              h0_stored=float(c['h0']), h0_recomputed_fs30=h, abs_difference=abs(h - float(c['h0'])),
                              context_inside_recording=cs + 300 <= len(gt)))
    for name, data in [('frame_units_recordings.csv', rec_out), ('frame_units_chunks.csv', chunk_out)]:
        with open(OUT / name, 'w', newline='') as f:
            w = csv.DictWriter(f, list(data[0]))
            w.writeheader()
            w.writerows(data)
    f = np.array([r['fps_timestamps'] for r in rec_out])
    diff = np.array([r['abs_difference'] for r in chunk_out])
    print('recordings', len(rec_out), 'ppg==frames', sum(r['ppg_equals_frames'] for r in rec_out),
          'fps(timestamps) median %.3f min %.3f max %.3f' % (np.median(f), f.min(), f.max()),
          'below 29:', int((f < 29).sum()), 'below 25:', int((f < 25).sum()),
          'with non-increasing intervals:', sum(r['nonincreasing_intervals'] > 0 for r in rec_out))
    print('chunks', len(chunk_out), 'h0 equal (|diff|<1e-9):', int((diff < 1e-9).sum()), 'max diff', float(diff.max()),
          'contexts inside:', sum(c['context_inside_recording'] for c in chunk_out))


if __name__ == '__main__':
    main()
