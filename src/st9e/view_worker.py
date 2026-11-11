"""CPU-only official crop/resize worker; all pixels/boxes stay in local pipes."""
import contextlib
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import cv2
import torch

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'third_party/rPPG-Toolbox'
os.chdir(REPO);sys.path.insert(0,str(REPO))
torch.set_num_threads(1)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only worker required')
with contextlib.redirect_stdout(sys.stderr):
    from dataset.data_loader.BaseLoader import BaseLoader
loader=object.__new__(BaseLoader)
output=sys.stdout.buffer
boxes=[]


def record_detection(self,*args,**kwargs):
    box=BaseLoader.face_detection(self,*args,**kwargs)
    boxes.append(np.asarray(box,dtype=int).tolist())
    return box


loader.face_detection=types.MethodType(record_detection,loader)
for line in sys.stdin.buffer:
    request=json.loads(line)
    if request.get('close'):break
    shape=tuple(request['shape']);nbytes=int(np.prod(shape))
    data=sys.stdin.buffer.read(nbytes)
    if len(data)!=nbytes:raise RuntimeError('Incomplete local frame pipe')
    frames=np.frombuffer(data,np.uint8).reshape(shape)
    backbone=request['backbone'];dynamic=backbone=='FactorizePhys'
    boxes.clear()
    with contextlib.redirect_stdout(sys.stderr):
        if request.get('reuse_boxes') is not None:
            boxes.extend(request['reuse_boxes'])
            result=np.zeros((len(frames),72,72,3),dtype=np.float64)
            for i,frame in enumerate(frames):
                x,y,w,h=boxes[i if request.get('reuse_boxes_per_frame') else min(i//30,len(boxes)-1)] if dynamic else boxes[0]
                result[i]=cv2.resize(frame[max(y,0):min(y+h,frame.shape[0]),max(x,0):min(x+w,frame.shape[1])],(72,72),interpolation=cv2.INTER_AREA)
        else:
            if dynamic and not hasattr(loader,'Y5FObj'):
                from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
                loader.Y5FObj=YOLO5Face(device='cpu')
            result=loader.crop_face_resize(frames,True,'Y5F' if dynamic else 'HC',True,1.5,
                dynamic,30 if dynamic else 32,False,72,72)
    if request.get('extra_first'):result=result[1:]
    result=np.ascontiguousarray(result,dtype=np.float64)
    header=dict(shape=list(result.shape),boxes=boxes,opencv_version=cv2.__version__,GPU_used=False)
    output.write((json.dumps(header)+'\n').encode());output.write(result.tobytes());output.flush()
