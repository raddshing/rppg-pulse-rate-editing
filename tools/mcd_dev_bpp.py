"""Rule 4 (6.89 다 2): bits per pixel of the MCD-rPPG development videos (CPU; AVI headers only, no pixel decoding).

For every recording of outputs/gop/mcd_dev_gop.csv (the 404 front FullHDwebcam development videos measured by tools/mcd_gop.py),
read the video stream format chunk (strf, BITMAPINFOHEADER biWidth/biHeight) with the demultiplexer of tools/mcd_gop.py.
bits per pixel = video_bytes * 8 / (vops * width * height)   (bits per coded pixel per frame; video_bytes and vops from mcd_gop.py)
Writes outputs/rule4/mcd_dev_bpp.csv and outputs/rule4/mcd_dev_bpp.json (median).
Usage: mcd_dev_bpp.py DATA_ROOT
"""
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parents[1]


def dims(path):
    data = Path(path).read_bytes()[:65536]
    i = data.find(b'strf')
    while i >= 0:
        size = struct.unpack_from('<I', data, i + 4)[0]
        if size >= 40:
            return struct.unpack_from('<i', data, i + 12)[0], abs(struct.unpack_from('<i', data, i + 16)[0])
        i = data.find(b'strf', i + 4)
    raise ValueError('no video strf')


def main():
    root = Path(sys.argv[1])
    g = pd.read_csv(WORK / 'outputs' / 'gop' / 'mcd_dev_gop.csv', dtype={'subject_id': str})
    wanted = set(g.recording_id)
    video = {}
    with (root / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            rid = f"{r['patient_id']}_{r['step']}"
            if rid in wanted and r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                video[rid] = r['video']
    w, h = zip(*(dims(root / video[rid]) for rid in g.recording_id))
    g['width'], g['height'] = w, h
    g['bits_per_pixel'] = g.video_bytes * 8 / (g.vops * g.width * g.height)
    out = WORK / 'outputs' / 'rule4'
    g[['recording_id', 'width', 'height', 'vops', 'video_bytes', 'header_fps', 'meta_fps', 'bitrate_kbps', 'bits_per_pixel']].to_csv(out / 'mcd_dev_bpp.csv', index=False)
    s = dict(recordings=len(g), sizes=g.groupby(['width', 'height']).size().rename('n').reset_index().to_dict('records'),
             bits_per_pixel_median=float(np.median(g.bits_per_pixel)), bits_per_pixel_min=float(g.bits_per_pixel.min()),
             bits_per_pixel_max=float(g.bits_per_pixel.max()), bitrate_kbps_median=float(np.median(g.bitrate_kbps)))
    (out / 'mcd_dev_bpp.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    main()
