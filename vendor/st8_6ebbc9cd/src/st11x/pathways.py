"""Fixed instruction06 pathways. This module performs no source discovery/inference.

Pure array functions are testable now. Dataset execution requires the future
explicit PI instruction06 receipt and a committed reference/population manifest.
"""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly

BANDS={'HAR':(.6,3.3),'L0':(.75,2.5)}
GRID=3.515625


def uniform_frame_grid(times,fs=30):
    t=np.asarray(times,dtype=np.float64)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or not (np.diff(t)>0).all():
        raise ValueError('Finite strictly increasing provided frame times are required')
    relative=t-t[0]
    n=int(np.floor(relative[-1]*fs+1e-9))+1
    return np.arange(n,dtype=np.float64)/fs


def interpolate_face_input(spatial_input,times,fs=30):
    """Crop/resize original frames before temporal interpolation; no fps fallback."""
    x=np.asarray(spatial_input)
    t=np.asarray(times,dtype=np.float64);grid=uniform_frame_grid(t,fs)
    if len(x)!=len(t):raise ValueError('One provided time per original frame required')
    relative=t-t[0]
    if len(grid)==len(relative) and np.array_equal(relative,grid):return x.copy(),grid
    # Avoid extrapolation; endpoint-only floating error is clipped by one ulp.
    grid=grid[grid<=np.nextafter(relative[-1],np.inf)]
    right=np.searchsorted(relative,grid,side='left');right=np.minimum(right,len(relative)-1)
    left=np.maximum(right-1,0);delta=relative[right]-relative[left]
    weight=np.divide(grid-relative[left],delta,out=np.zeros_like(grid),where=delta>0)
    weight=weight.reshape((len(grid),)+(1,)*(x.ndim-1))
    return (1-weight)*x[left].astype(np.float64)+weight*x[right],grid


def temporal_preprocess(values,model):
    """Same full-recording Stage1 transforms, then full nonoverlapping chunks."""
    from src.st9g.preprocess import diff_normalize_data
    x=np.asarray(values)
    if model=='PhysNet':x=diff_normalize_data(x);length=128
    elif model=='FactorizePhys':length=160
    else:raise ValueError('The authorized X5 model-input tests cover two structures')
    n=len(x)//length
    return np.asarray([x[i*length:(i+1)*length] for i in range(n)])


def timestamp_model_input(spatial_input,times,model):
    regular,grid=interpolate_face_input(spatial_input,times)
    return temporal_preprocess(regular,model),grid


def reference_30hz(contact,nominal_fs):
    ratio=Fraction(30/float(nominal_fs)).limit_denominator(1000000)
    return resample_poly(np.asarray(contact,dtype=np.float64),ratio.numerator,ratio.denominator,window=('kaiser',5.0),padtype='constant')


def complete_windows(grid,start_times):
    """Half-open10s windows begin at first grid sample>=t_k and need300 samples."""
    g=np.asarray(grid,dtype=np.float64)
    result=[]
    for t in start_times:
        start=int(np.searchsorted(g,float(t),side='left'))
        result.append(dict(start_seconds=float(t),grid_start=start,grid_end_exclusive=start+300,
                           available=start+300<=len(g)))
    return result


def mmpd_b_membership(defect,reference_class,h_har,h_l0):
    """Automatic-only B Qstar; no PI-confirmation branch or access here."""
    wide=not bool(defect) and reference_class=='agreement'
    narrow=wide and np.isfinite([h_har,h_l0]).all() and abs(h_l0-h_har)<=GRID
    return dict(Qstar_HAR=bool(wide),Qstar_L0=bool(narrow),PI_confirmation_path=False)


def vipl_duration_eligible(video_length,bvp_samples,nominal_fs=60):
    if not np.isfinite(video_length) or video_length<=0:return False
    return abs(bvp_samples/nominal_fs-video_length)/video_length<=.01


def cohort_selection(dataset,record):
    """AppendixC memberships from committed reference metadata only."""
    h=float(record['h_FFT']);agreement=record['reference_class']=='agreement'
    if dataset=='MCD-rPPG':
        confirmed=bool(record['mapping_confirmed']);irregular=bool(record['irregular_clock'])
        return dict(primary=confirmed and agreement and h>=100,confirmed_high=confirmed and h>=100,
                    irregular_high=irregular and h>=100,low=h<100)
    if dataset=='VIPL-HR':
        eligible=record['camera']=='source3' and vipl_duration_eligible(record['video_duration_seconds'],record['BVP_samples'],record['BVP_fs'])
        return dict(primary=eligible and agreement and h>=100,excluded_duration=not eligible)
    if dataset=='COHFACE':
        has_time=bool(record['provided_frame_times'])
        return dict(primary=has_time and agreement and h<100,excluded_missing_frame_times=not has_time)
    raise ValueError('External dataset name required')


def require_instruction06(receipt,reference_commit,already_evaluated):
    if receipt.get('instruction')!='ST4-PI-INSTR-06' or not receipt.get('explicit_user_authorization'):
        raise PermissionError('External and MMPD-B execution awaits PI instruction06')
    if not reference_commit:raise PermissionError('Commit references/windows/populations before model inference')
    if already_evaluated:raise PermissionError('Each model/dataset may be evaluated once only')


def numeric_subject_order(subjects):
    import re
    items=list(set(map(str,subjects)))
    def key(x):
        match=re.fullmatch(r'\D*(\d+)',x)
        return (0,int(match.group(1)),x) if match else (1,x)
    return sorted(items,key=key)


def paired_bootstrap_indices(subjects,seeds=3):
    subjects=numeric_subject_order(subjects);rng=np.random.default_rng(20261110)
    a=rng.integers(0,len(subjects),size=(10000,len(subjects)))
    b=rng.integers(0,seeds,size=(10000,seeds))
    return subjects,a,b
