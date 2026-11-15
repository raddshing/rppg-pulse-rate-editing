"""Client for infer_server.py; also the RhythmFormer CPU view worker client (draft st15u/views.py)."""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
REPRO = '<MACHINE_PATH>'  # read-only use, no installs
LOGS = WORK / 'logs'
LOGS.mkdir(exist_ok=True)
GPU = os.environ.get('OPS2_GPU', '1')


class Infer:
    def __init__(self, load):
        self.log = (LOGS / 'infer_server.log').open('ab')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
        self.p = subprocess.Popen([REPRO, '-B', '-s', str(WORK / 'audit' / 'infer_server.py')], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, env=env)
        self.loaded = self._send(dict(load=load))['loaded']

    def _send(self, header, payload=None):
        self.p.stdin.write((json.dumps(header) + '\n').encode())
        if payload is not None:
            self.p.stdin.write(payload)
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('inference server ended')
        return json.loads(line)

    def predict(self, view, x, rng='reset'):
        x = np.ascontiguousarray(x, dtype=np.float32)
        meta = self._send(dict(predict=view, shape=list(x.shape), rng=rng), x.tobytes())
        n = int(np.prod(meta['shape'])) * 4
        raw = self.p.stdout.read(n)
        assert len(raw) == n
        if meta.get('num_ooms'):  # brief 02 6.54 나 2: no output after an allocator OOM is used (cuDNN silently changes plan)
            msg = f"GPU memory shortage: allocator OOM count {meta['num_ooms']} in the inference server; run stopped, outputs of this request not returned"
            with (LOGS / 'gpu_oom_events.log').open('a') as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} pid {os.getpid()} {msg}\n")
            raise SystemExit(msg)  # not an Exception: per-recording failure handlers do not catch it, the whole run stops
        return meta['keys'], np.frombuffer(raw, np.float32).reshape(meta['shape'])

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(b'{"close": true}\n')
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=120)
        self.log.close()


class RhythmView:
    """Draft RhythmViewWorker (st15u/views.py) using the archived st12c/rhythm_view_worker.py."""

    def __init__(self):
        self.log = (LOGS / 'rhythm_view_worker.log').open('ab')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        self.p = subprocess.Popen([REPRO, '-B', str(V / 'src' / 'st12c' / 'rhythm_view_worker.py')], cwd=V,
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)

    def view(self, frames, box=None):
        a = np.ascontiguousarray(frames, dtype=np.uint8)
        self.p.stdin.write((json.dumps(dict(shape=list(a.shape), reuse_box=None if box is None else list(map(int, box)))) + '\n').encode())
        self.p.stdin.write(a.tobytes())
        self.p.stdin.flush()
        meta = json.loads(self.p.stdout.readline())
        assert meta['opencv_version'] == '4.5.2' and meta['GPU_used'] is False
        n = int(np.prod(meta['shape'])) * 8
        raw = self.p.stdout.read(n)
        assert len(raw) == n
        return np.frombuffer(raw, np.float64).reshape(meta['shape']), np.asarray(meta['boxes'], int)

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write(b'{"close":true}\n')
            self.p.stdin.flush()
            self.p.stdin.close()
            self.p.wait(timeout=60)
        self.log.close()
