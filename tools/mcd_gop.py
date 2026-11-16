"""MCD-rPPG GOP check (CPU; no pixel decoding, no installation). Brief 02 6.45 나 1 (development half, 2026-11-13);
brief 02 6.82 나: the recording list, the data root and the output path became arguments; the measurement is unchanged.

Usage: mcd_gop.py RECORDING_LIST DATA_ROOT OUT_CSV
Recordings: the MCD-rPPG index (DATA_ROOT/db.csv) rows are first reduced to the recordings of RECORDING_LIST (column
recording_id = <patient_id>_<step>; rows of other recordings are discarded before any use), then the front-view
FullHDwebcam video of every listed recording is read.
Bitstream: RIFF AVI demultiplexing (vendor src/st5c/avi_packets.py chunk rules); in every video packet each MPEG-4
Part 2 VOP start code 00 00 01 B6 is followed by vop_coding_type (2 bits: 0 I, 1 P, 2 B, 3 S). Frame types are taken
in packet order; empty packets (dropped frames) are counted separately.
Per recording: frame-type counts, B-frame presence, GOP lengths (frames from one I-VOP to the next), whether the
I-interval is constant (all complete GOPs equal), the AVI header frame rate, the frame rate from the recording's meta
timestamps ((n - 1) / span), the bytes of the video packets and the bitrate over the timestamp span.
Writes OUT_CSV (per recording) and summary.json in the same folder.
"""
import json
import os
import struct
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
MCD = None  # read-only original data root (argument DATA_ROOT)
TYPES = 'IPBS'


def video_packets(path):
    data = Path(path).read_bytes()
    streams, packets = [], []

    def chunks(begin, end):
        pos = begin
        while pos + 8 <= end:
            tag = data[pos:pos + 4]
            size = struct.unpack_from('<I', data, pos + 4)[0]
            stop = min(pos + 8 + size, end)
            yield tag, pos + 8, stop
            pos = pos + 8 + size + (size & 1)

    def walk(begin, end, in_movi=False):
        for tag, start, stop in chunks(begin, end):
            if tag in (b'RIFF', b'LIST'):
                kind = data[start:start + 4]
                if kind == b'strl':
                    vals = {t: data[a:b] for t, a, b in chunks(start + 4, stop)}
                    sh, sf = vals.get(b'strh', b''), vals.get(b'strf', b'')
                    s = dict(kind=sh[:4], scale=struct.unpack_from('<I', sh, 20)[0] if len(sh) >= 28 else 0,
                             rate=struct.unpack_from('<I', sh, 24)[0] if len(sh) >= 28 else 0,
                             codec=sf[16:20] if sh[:4] == b'vids' and len(sf) >= 20 else sh[4:8])
                    streams.append(s)
                elif kind in (b'AVI ', b'AVIX', b'hdrl', b'movi', b'rec '):
                    walk(start + 4, stop, in_movi or kind in (b'movi', b'rec '))
            elif in_movi and tag[2:] in (b'dc', b'db'):
                try:
                    packets.append((int(tag[:2], 16), start, stop))
                except ValueError:
                    pass
    assert data[:4] == b'RIFF' and data[8:12] == b'AVI '
    walk(0, len(data))
    vid = [i for i, s in enumerate(streams) if s['kind'] == b'vids']
    assert len(vid) == 1
    st = streams[vid[0]]
    return data, st, [(a, b) for sid, a, b in packets if sid == vid[0]]


def job(rec):
    from src.st5c.mcd_packets import timestamp_table
    data, st, pk = video_packets(MCD / rec['video'])
    types, empty, multi, nbytes = [], 0, 0, 0
    for a, b in pk:
        nbytes += b - a
        if b <= a:
            empty += 1
            continue
        p = data[a:b]
        locs = []
        i = p.find(b'\x00\x00\x01\xb6')
        while i >= 0:
            locs.append(i)
            i = p.find(b'\x00\x00\x01\xb6', i + 4)
        multi += int(len(locs) > 1)
        for i in locs:
            if i + 4 < len(p):
                types.append(TYPES[p[i + 4] >> 6])
    seq = ''.join(types)
    ipos = [i for i, t in enumerate(seq) if t == 'I']
    gops = np.diff(ipos).tolist()
    _, stamps = timestamp_table(MCD / rec['meta'])
    t = (np.asarray(stamps, float) - float(stamps[0])) / 1e6
    span = float(t[-1]) if len(t) > 1 else np.nan
    c = Counter(seq)
    return dict(recording_id=rec['recording_id'], subject_id=rec['subject_id'], codec=st['codec'].decode('ascii', 'replace'),
                video_packets=len(pk), empty_packets=empty, packets_with_several_vops=multi, vops=len(seq),
                I=c.get('I', 0), P=c.get('P', 0), B=c.get('B', 0), S=c.get('S', 0), has_B=c.get('B', 0) > 0,
                first_type=seq[:1], gops_complete=len(gops),
                gop_mode=Counter(gops).most_common(1)[0][0] if gops else np.nan,
                gop_min=min(gops) if gops else np.nan, gop_max=max(gops) if gops else np.nan,
                gop_constant=bool(gops) and len(set(gops)) == 1, gop_lengths=' '.join(str(g) for g in sorted(set(gops))),
                tail_after_last_I=len(seq) - ipos[-1] if ipos else np.nan,
                header_fps=st['rate'] / st['scale'] if st['scale'] else np.nan,
                meta_frames=len(t), meta_span_s=span, meta_fps=(len(t) - 1) / span if span else np.nan,
                video_bytes=nbytes, bitrate_kbps=nbytes * 8 / span / 1000 if span else np.nan,
                vop_fps_over_span=(len(seq) - 1) / span if span else np.nan,
                keyframe_hz_from_meta=((len(t) - 1) / span) / (Counter(gops).most_common(1)[0][0]) if gops and span else np.nan)


def init(root):
    global MCD
    MCD = Path(root)
    sys.path.insert(0, str(V))


def main():
    import csv
    listed, root, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out = out_csv.parent
    out.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(listed, dtype=str)
    wanted = set(recs.recording_id)
    video, meta, subject = {}, {}, {}
    with (root / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            rid = f"{r['patient_id']}_{r['step']}"
            if rid not in wanted:  # rows of recordings outside the list are discarded before any use
                continue
            if r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                video[rid] = r['video']
                meta[rid] = r['meta']
                subject[rid] = r['patient_id']
    rows = [dict(recording_id=rid, subject_id=subject[rid], video=video[rid], meta=meta[rid])
            for rid in recs.recording_id if rid in video]
    print('listed recordings', len(recs), 'with front FullHD video', len(rows), flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get('OPS2_WORKERS', '3')), initializer=init, initargs=(str(root),)) as ex:
        for n, r in enumerate(ex.map(job, rows), 1):
            res.append(r)
            if n % 50 == 0 or n == len(rows):
                print(f'gop {n}/{len(rows)}', flush=True)
    d = pd.DataFrame(res)
    d.to_csv(out_csv, index=False)
    summ = dict(recordings=len(d), codecs=d.codec.value_counts().to_dict(),
                recordings_with_B=int(d.has_B.sum()), recordings_with_S=int((d.S > 0).sum()),
                gop_mode_by_recording=d.gop_mode.value_counts().to_dict(),
                gop_min=float(d.gop_min.min()), gop_max=float(d.gop_max.max()),
                share_constant_I_interval=float(d.gop_constant.mean()),
                first_frame_types=d.first_type.value_counts().to_dict(),
                empty_packets_total=int(d.empty_packets.sum()), packets_with_several_vops_total=int(d.packets_with_several_vops.sum()),
                header_fps=d.header_fps.value_counts().to_dict(),
                meta_fps_median=float(d.meta_fps.median()), meta_fps_min=float(d.meta_fps.min()), meta_fps_max=float(d.meta_fps.max()),
                bitrate_kbps_median=float(d.bitrate_kbps.median()), bitrate_kbps_min=float(d.bitrate_kbps.min()),
                bitrate_kbps_max=float(d.bitrate_kbps.max()),
                keyframe_hz_median=float(d.keyframe_hz_from_meta.median()),
                keyframe_bpm_median=float(d.keyframe_hz_from_meta.median() * 60),
                vop_fps_over_span_median=float(d.vop_fps_over_span.median()),
                keyframe_bpm_on_vop_grid_median=float((d.vop_fps_over_span / d.gop_mode).median() * 60),
                meta_frames_minus_vops_median=float((d.meta_frames - d.vops).median()))
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == '__main__':
    main()
