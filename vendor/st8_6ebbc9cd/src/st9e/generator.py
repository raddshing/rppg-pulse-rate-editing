"""Native ED2W, temporal resampling and paired source streams for B preparation."""
from src.st9e.editor import *
from src.st9e.model_view import crop_native


def raw_target_label(gt,start,original_HR,target_HR):
    up,down=parameters(target_HR/original_HR)
    return resample(gt,start,up,down)[15:315]


def reflection_used(gt,start,original_HR,target_HR):
    up,down=parameters(target_HR/original_HR)
    margin=down*((30+down-1)//down)
    if margin*up//down<15:margin=down*((15+up-1)//up+max((30+down-1)//down,11))
    need=((330+15)*down+up-1)//up+2
    return start-margin<0 or start+need+margin>len(gt)


def ed2w_context(clip,face,roi,gt,start,original_HR,target_HR,tau,direction,physview,
                 physbox,window_id,condition,notched=None,K=1,seed=0,equivalence=False):
    """No detector on edits; raw contact follows the injection time map without tau."""
    if notched is None:notched=notch_all(clip,bands_for(original_HR))
    g0,gstar=ppg_vectors(gt,start,original_HR,target_HR,tau)
    pulse,w,beta=calibration(physview,crop_native(notched,physbox),crop_native(face,physbox),g0,direction)
    outputs=[]
    for k in range(K):
        key=window_id if equivalence else f'{window_id}|seed={seed}|augmentation={k}'
        rounding_rng=None
        if not equivalence:
            check('Only one seed0 test augmentation per context',K==1 and seed==0)
            h=int(hashlib.sha256(f'ST7B|{window_id}|ii-h|{float(target_HR)}'.encode()).hexdigest()[:8],16)
            rounding_rng=np.random.default_rng([0,h])
        output,diag=edited(clip,notched,face,roi,g0,gstar,direction,original_HR,'ED2W-S',pulse,w,key,condition,rounding_rng=rounding_rng)
        outputs.append(output)
    return outputs,raw_target_label(gt,start,original_HR,target_HR),dict(beta=beta,**diag,
        reflection_used=reflection_used(gt,start,original_HR,target_HR),target_HR=target_HR,
        injection_reference_HAR=common_hr(gstar,'HAR'),injection_reference_L0=common_hr(gstar,'L0'))


def temporal_resample(frames,labels,boxes,start,length,original_HR,target_HR,seed):
    scale=target_HR/original_HR;span=length*scale;n=len(frames)
    if span>n:return None,dict(excluded=True,reason='source_interval_exceeds_recording')
    begin=max(0,min(float(start),n-span));positions=begin+np.arange(length)*scale
    left=np.floor(positions).astype(int);right=np.minimum(left+1,n-1);fraction=positions-left
    rng=np.random.default_rng(seed);output=np.empty((length,)+frames.shape[1:],np.uint8)
    for i,(a,b,f) in enumerate(zip(left,right,fraction)):
        value=(1-f)*frames[a].astype(np.float64)+f*frames[b]
        output[i]=np.floor(value+rng.random(value.shape)).astype(np.uint8)
    label=np.interp(positions,np.arange(n),labels);mapped=np.asarray(boxes)[np.rint(positions).astype(int)]
    return (output,label,mapped),dict(excluded=False,source_start=begin,scale=scale,source_frame_span=span)
