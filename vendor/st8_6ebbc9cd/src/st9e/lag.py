"""Fixed signed overlap correlation and all59-recording PURE lag estimation."""
from src.st9e.editor import *


def lag_value(y,p,search):
    y=np.asarray(y,float);p=np.asarray(p,float);scores=[]
    for lag in search:
        if lag>0:a,b=y[lag:],p[:-lag]
        elif lag<0:a,b=y[:lag],p[-lag:]
        else:a,b=y,p
        value=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else np.nan
        scores.append(value)
    finite=np.isfinite(scores)
    if not finite.any():return None,np.nan
    ix=int(np.argmax(np.where(finite,scores,-np.inf)))
    return int(search[ix]),float(scores[ix])


def estimate(clip,mask,ppg,search):
    rgb=weighted_rgb(clip,mask)
    return lag_value(editor_band(green(rgb)),editor_band(ppg),search)
