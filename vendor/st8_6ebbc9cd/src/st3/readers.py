"""Official POS/CHROM core equations on skin RGB, without extra normalization."""
import math
import numpy as np
from scipy.signal import butter,filtfilt,windows
from src.eval.har import detrend


def pos(rgb,fs=30,filtered=True):
    n=len(rgb);length=math.ceil(1.6*fs);h=np.zeros(n)
    for end in range(length,n):
        x=(rgb[end-length:end]/rgb[end-length:end].mean(axis=0)).T
        s=np.array([[0.,1.,-1.],[-2.,1.,1.]])@x
        wave=s[0]+s[0].std()/s[1].std()*s[1]
        h[end-length:end]+=wave-wave.mean()
    h=detrend(h,100)
    if not filtered:return h
    b,a=butter(1,[.75/(fs/2),3/(fs/2)],btype='bandpass')
    return filtfilt(b,a,h.astype(np.float64))


def chrom(rgb,fs=30):
    length=math.ceil(1.6*fs);length+=length%2;half=length//2
    nwin=math.floor((len(rgb)-half)/half);out=np.zeros(half*(nwin+1))
    b,a=butter(3,[.7/(fs/2),2.5/(fs/2)],btype='bandpass')
    for i in range(nwin):
        start=i*half;end=start+length;x=rgb[start:end]/rgb[start:end].mean(axis=0)
        xf=filtfilt(b,a,3*x[:,0]-2*x[:,1]);yf=filtfilt(b,a,1.5*x[:,0]+x[:,1]-1.5*x[:,2])
        wave=(xf-xf.std()/yf.std()*yf)*windows.hann(length)
        out[start:start+half]+=wave[:half];out[start+half:end]=wave[half:]
    return out


def signals(rgb):
    return {'POS':pos(rgb),'CHROM':chrom(rgb),'GREEN':rgb[:,1]/rgb[:,1].mean()-1}
