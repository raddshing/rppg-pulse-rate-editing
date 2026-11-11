"""Persistent official-environment RGB-view pipe; no geometry exports."""
from src.st9e.base import *
import numpy as np


class ViewWorker:
    def __init__(self):
        self.log=open(REPORTS/(NAMESPACE+'_model_view_worker.log'),'a')
        self.process=subprocess.Popen([str(Path(PATHS['environments']['repro'])/'bin/python'),'-B','-s',
            str(ROOT/'src/st9e/view_worker.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        ledger('model_view_worker','started',self.process.pid)
    def view(self,clip,backbone,recording_first=None,reuse_boxes=None,reuse_boxes_per_frame=False):
        extra=backbone=='PhysNet' and recording_first is not None and reuse_boxes is None
        frames=np.concatenate([recording_first[None],clip],axis=0) if extra else clip
        p=self.process
        p.stdin.write((json.dumps(dict(shape=list(frames.shape),backbone=backbone,extra_first=extra,reuse_boxes=None if reuse_boxes is None else np.asarray(reuse_boxes).tolist(),reuse_boxes_per_frame=reuse_boxes_per_frame))+'\n').encode())
        p.stdin.write(frames.tobytes());p.stdin.flush()
        line=p.stdout.readline()
        check('CPU official-view worker returned header',bool(line))
        meta=json.loads(line)
        check('Frozen model-view OpenCV4.5.2 CPU path',meta['opencv_version']=='4.5.2' and not meta['GPU_used'])
        count=int(np.prod(meta['shape']))*8
        raw=p.stdout.read(count)
        check('Complete model-view output pipe',len(raw)==count)
        return np.frombuffer(raw,np.float64).reshape(meta['shape']),np.asarray(meta['boxes'],int)
    def close(self):
        p=self.process
        if p.poll() is None:
            ledger('model_view_worker','closing',p.pid)
            p.stdin.write(b'{"close":true}\n');p.stdin.flush();p.stdin.close()
            p.wait(timeout=60)
        self.log.close()
        check('Owned CPU view worker clean exit',p.returncode==0)


def crop_native(frames,box):
    x,y,w,h=map(int,box)
    return frames[:,max(y,0):min(y+h,frames.shape[1]),max(x,0):min(x+w,frames.shape[2])]
