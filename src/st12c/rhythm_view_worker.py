"""CPU-only exact Stage1 static HC1.5 resize128; local pixel pipe only."""
import contextlib
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import cv2
import torch

ROOT=Path(__file__).resolve().parents[2];repo=ROOT/'third_party/rPPG-Toolbox'
os.chdir(repo);sys.path.insert(0,str(repo));torch.set_num_threads(1)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or cv2.__version__!='4.5.2':raise RuntimeError('Frozen CPU OpenCV required')
with contextlib.redirect_stdout(sys.stderr):from dataset.data_loader.BaseLoader import BaseLoader
loader=object.__new__(BaseLoader);boxes=[];output=sys.stdout.buffer


def detection(self,*args,**kwargs):
    box=BaseLoader.face_detection(self,*args,**kwargs);boxes.append(np.asarray(box,dtype=int).tolist());return box


loader.face_detection=types.MethodType(detection,loader)
for line in sys.stdin.buffer:
    request=json.loads(line)
    if request.get('close'):break
    shape=tuple(request['shape']);size=int(np.prod(shape));raw=sys.stdin.buffer.read(size)
    if len(raw)!=size:raise RuntimeError('Incomplete source pixel pipe')
    frames=np.frombuffer(raw,np.uint8).reshape(shape);boxes.clear()
    with contextlib.redirect_stdout(sys.stderr):
        if request.get('reuse_box') is None:
            result=loader.crop_face_resize(frames,True,'HC',True,1.5,False,30,False,128,128)
        else:
            boxes.append(request['reuse_box']);x,y,w,h=map(int,boxes[0]);result=np.empty((len(frames),128,128,3),np.float64)
            for i,frame in enumerate(frames):result[i]=cv2.resize(frame[max(y,0):min(y+h,frame.shape[0]),max(x,0):min(x+w,frame.shape[1])],(128,128),interpolation=cv2.INTER_AREA)
    result=np.ascontiguousarray(result,dtype=np.float64)
    output.write((json.dumps(dict(shape=list(result.shape),boxes=boxes,opencv_version=cv2.__version__,GPU_used=False))+'\n').encode())
    output.write(result.tobytes());output.flush()
