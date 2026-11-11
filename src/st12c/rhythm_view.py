"""Persistent local CPU preprocessing pipe for the frozen RhythmFormer view."""
from src.st12.ops import *
import numpy as np


class RhythmViewWorker:
    def __init__(self):
        self.log=(REPORTS/'st12c_rhythm_view_worker.log').open('a')
        self.process=subprocess.Popen([str(Path(PATHS['environments']['repro'])/'bin/python'),'-B',
            str(ROOT/'src/st12c/rhythm_view_worker.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        ledger('rhythm_view_worker','started',self.process.pid)
    def view(self,frames,box=None):
        p=self.process;array=np.ascontiguousarray(frames,dtype=np.uint8)
        p.stdin.write((json.dumps(dict(shape=list(array.shape),reuse_box=None if box is None else list(map(int,box))))+'\n').encode())
        p.stdin.write(array.tobytes());p.stdin.flush();line=p.stdout.readline()
        check('Rhythm CPU view header present',bool(line));meta=json.loads(line)
        check('Frozen CPU OpenCV4.5.2 view',meta['opencv_version']=='4.5.2' and meta['GPU_used'] is False)
        raw=p.stdout.read(int(np.prod(meta['shape']))*8);check('Complete Rhythm view pipe',len(raw)==int(np.prod(meta['shape']))*8)
        return np.frombuffer(raw,np.float64).reshape(meta['shape']),np.asarray(meta['boxes'],int)
    def close(self):
        p=self.process
        if p.poll() is None:
            ledger('rhythm_view_worker','closing',p.pid);p.stdin.write(b'{"close":true}\n');p.stdin.flush();p.stdin.close();p.wait(timeout=60)
        self.log.close();check('Rhythm CPU view worker clean exit',p.returncode==0)
