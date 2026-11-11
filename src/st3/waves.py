"""Fixed A-only saved-wave access and unchanged WP3 preparation/HR grid."""
from src.st3.base import *
import numpy as np
from scipy.signal import periodogram
from src.eval import grid,har


def prediction(run,rid,dataset='MMPD'):
    if dataset=='MMPD':
        check('A selected before prediction access',rid.split('/')[0] in SUBJECTS)
    path='outputs/st1_predictions/'+run+'/'+rid.replace('/','__')+'.npz'
    source_hash(path)
    source=har.load_prediction(ROOT/path)
    check('Stored prediction recording identity',source['recording_id'],rid)
    return source


def timeline(source,n):
    starts=np.asarray(source['chunk_starts'],dtype=int)
    lengths=np.asarray(source['chunk_len'])
    if lengths.ndim==0:lengths=np.full(len(starts),int(lengths))
    y=np.full(n,np.nan,dtype=source['pred'].dtype)
    g=np.full(n,np.nan,dtype=source['label'].dtype)
    covered=np.zeros(n,dtype=bool);offset=0
    for start,length in zip(starts,lengths):
        y[start:start+length]=source['pred'][offset:offset+length]
        g[start:start+length]=source['label'][offset:offset+length]
        covered[start:start+length]=True;offset+=length
    check('Packed stored waveform length',offset,len(source['pred']))
    return y,g,covered


def spectrum(wave,diff,layer):
    band=BANDS[layer]
    signal=grid._prepare(wave,diff,band,'default',30)
    f,p=periodogram(signal,fs=30,nfft=512,detrend=False)
    mask=(f>=band[0])&(f<=band[1])
    return f,p,mask


def hr(wave,diff,layer):
    if not np.isfinite(wave).all():return float('nan')
    if len(wave)==300:
        band=BANDS[layer]
        return grid._estimate(grid._prepare(wave,diff,band,'default',30),'periodogram',band,30)
    f,p,mask=spectrum(wave,diff,layer)
    return float(f[mask][np.argmax(p[mask])]*60)


def label_preprocess(raw,label_type):
    if label_type=='DiffNormalized':
        d=np.diff(raw)
        result=np.append(d/np.std(d),0.)
        result[np.isnan(result)]=0
    elif label_type=='Standardized':
        result=(raw-np.mean(raw))/np.std(raw)
        result[np.isnan(result)]=0
    elif label_type=='Raw':
        result=np.asarray(raw)
    else:
        raise ValueError('Unknown frozen label type')
    return result.astype(np.float32)
