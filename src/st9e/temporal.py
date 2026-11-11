"""Unchanged instruction02B strict-support time mapping and test rounding."""
from src.st9e.generator import *

def shifted_start(requested,length,n,scale,tau):
    shift=float(tau)*(scale-1);span=length*scale;last=(length-1)*scale
    low=max(0.,-shift);high=min(n-span,n-span-shift,n-1-last,n-1-last-shift)
    if high<low:return None,dict(excluded=True,source_start=np.nan,label_shift_source_frames=shift,numeric_inward_ulp_steps=0)
    begin=max(low,min(float(requested),high))
    for steps in range(9):
        positions=begin+np.arange(length)*scale;label_positions=positions+shift
        below=min(positions[0],label_positions[0])<0
        above=max(positions[-1],label_positions[-1])>n-1 or begin+span>n or begin+shift+span>n
        if not below and not above:
            return (positions,label_positions),dict(excluded=False,source_start=begin,label_shift_source_frames=shift,numeric_inward_ulp_steps=steps)
        if below and above:break
        begin=float(np.nextafter(begin,np.inf if below else -np.inf))
        if begin<low or begin>high:break
    return None,dict(excluded=True,source_start=np.nan,label_shift_source_frames=shift,numeric_inward_ulp_steps=steps)


def resample_pixels(clip,positions,offset,rid,target):
    h=int(hashlib.sha256(f'ST7B|{rid}|i|{float(target)}'.encode()).hexdigest()[:8],16);rng=np.random.default_rng([0,h])
    output=np.empty((len(positions),)+clip.shape[1:],np.uint8)
    for i,p in enumerate(positions-offset):
        left=int(np.floor(p));right=int(np.ceil(p));fraction=p-left
        check('Video interpolation within loaded original support',0<=left<=right<len(clip))
        value=(1-fraction)*clip[left].astype(float)+fraction*clip[right]
        output[i]=np.floor(value+rng.random(value.shape)).astype(np.uint8)
    return output

