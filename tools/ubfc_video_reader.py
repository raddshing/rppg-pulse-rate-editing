"""Original UBFC-rPPG frame pipe (7.09a, UBFC editor validity check).

The body is that of the October src/st9s/video_reader.py (commit 237e07107^, used by the October UBFC gate G-U), run in
the same frozen decoding environment (OpenCV 4.5.2, repro interpreter); only the allowed source root is fixed here
instead of being read from the October paths.yaml. Frames never leave the server: they go to the parent through a pipe.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
allowed = Path(CFG['ubfc2_raw']).resolve()
output = sys.stdout.buffer
for line in sys.stdin.buffer:
    request = json.loads(line)
    if request.get('close'):
        break
    path = Path(request['path']).resolve()
    if allowed not in path.parents:
        raise RuntimeError('Only authorized original UBFC decoding')
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, request['start'])
    frames = []
    for _ in range(request['count']):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('Incomplete original UBFC frame interval')
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    value = np.ascontiguousarray(frames, dtype=np.uint8)
    output.write((json.dumps(dict(shape=list(value.shape), opencv_version=cv2.__version__)) + '\n').encode())
    output.write(value.tobytes())
    output.flush()
