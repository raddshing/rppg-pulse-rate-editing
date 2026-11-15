"""Prescribed whole-pixel notch and face-weighted S-only editors."""
from src.st9e.base import *
from src.st9e.native import *
from src.st9e.inherited import (ppg_vectors, green, power, ratio, interval_power,
    independent_validity, inherited_pos, inherited_chrom, band_reader_pos,
    band_reader_chrom, stochastic_round, edit_float)
from src.st9e.signals import filter4
from scipy.signal import butter, sosfiltfilt


def bands_for(original, target=None):
    frequencies=[original/60,2*original/60]
    if target is not None:frequencies += [target/60,2*target/60]
    bands=[]
    for low,high in sorted((f-.15,f+.15) for f in frequencies):
        check('Notch bands inside positive Nyquist support',0<low<high<15)
        if bands and low<=bands[-1][1]:bands[-1][1]=max(high,bands[-1][1])
        else:bands.append([low,high])
    return bands


def face_masks(clip, return_points=False):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    groups=json.loads((ROOT/'outputs/st5_face_mask_indices.json').read_text())
    source_hash('outputs/st3_roi_indices.json')
    roi=json.loads((ROOT/'outputs/st3_roi_indices.json').read_text())
    points=np.full((300,478,2),np.nan,dtype=np.float32)
    height,width=clip.shape[1:3]
    with vision.FaceLandmarker.create_from_options(tracking_options()) as tracker:
        for i,rgb in enumerate(clip):
            found=tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)),round(i*1000/30))
            if found.face_landmarks:
                points[i]=[[p.x*width,p.y*height] for p in found.face_landmarks[0]]
    detected=np.isfinite(points[:,0,0])
    check('Every fixed window has a face landmark observation',detected.any())
    flat=points.reshape(300,-1)
    for k in range(flat.shape[1]):
        good=np.isfinite(flat[:,k])
        flat[:,k]=np.interp(np.arange(300),np.flatnonzero(good),flat[good,k])
    masks=np.empty((300,height,width),np.float32)
    rois=np.empty((300,height,width),bool)
    for i,p in enumerate(points):
        mask=np.zeros((height,width),np.uint8)
        for name,value in [('FACE_OVAL',1),('LEFT_EYE',0),('RIGHT_EYE',0),
                           ('LEFT_EYEBROW',0),('RIGHT_EYEBROW',0),('INNER_LIPS',0)]:
            xy=np.rint(p[groups[name]]).astype(np.int32)
            cv2.fillConvexPoly(mask,cv2.convexHull(xy),value)
        masks[i]=np.clip(cv2.GaussianBlur(mask.astype(np.float32),(0,0),2.,sigmaY=2.,
                            borderType=cv2.BORDER_REFLECT_101),0,1)
        rois[i]=skin_mask(clip.shape[1:],p,roi)
    check('Nonempty face and ROI masks in all frames',np.all(masks.sum(axis=(1,2))>0) and np.all(rois.sum(axis=(1,2))>0))
    result=(masks,rois,float(1-detected.mean()))
    return result+(points,) if return_points else result


def notch_all(clip,bands):
    filters=[butter(4,band,btype='bandstop',fs=30,output='sos') for band in bands]
    old=clip.reshape(300,-1,3)
    result=np.empty(clip.shape,np.float32)
    flat=result.reshape(300,-1,3)
    for start in range(0,old.shape[1],2048):
        value=old[:,start:start+2048].astype(np.float64)
        for filt in filters:value=sosfiltfilt(filt,value,axis=0)
        flat[:,start:start+2048]=value.astype(np.float32)
    return result


def weighted_rgb(clip,masks):
    # One frame at a time avoids a second full float64 image stack.
    return np.asarray([(rgb.astype(np.float64)*mask[:,:,None]).sum(axis=(0,1))/mask.sum(dtype=np.float64)
                       for rgb,mask in zip(clip,masks)])


def read_signals(view,clip,face,roi,brange=False):
    rgb=view.mean(axis=(1,2),dtype=np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):
        return dict(POS=band_reader_pos(rgb) if brange else inherited_pos(rgb),
            CHROM=band_reader_chrom(rgb) if brange else inherited_chrom(rgb),
            FACE_GREEN=green(weighted_rgb(clip,face)),ROI_GREEN=green(weighted_rgb(clip,roi)))


def calibration(view,notched_view,mask_view,g0,direction):
    y=editor_band(green(view.mean(axis=(1,2),dtype=np.float64)))
    beta=float(np.dot(y,g0)/np.dot(g0,g0))
    pulse=abs(beta)*float(np.sqrt(np.mean(g0**2)))
    denominator=notched_view[:,:,:,1].sum(dtype=np.float64)
    numerator=sum(np.sum(mask*frame[:,:,1],dtype=np.float64) for mask,frame in zip(mask_view,notched_view))
    w=float(numerator/denominator)
    check('Finite positive W calibration support',np.isfinite([beta,pulse,w]).all() and denominator>0 and w>0 and direction[1]>0)
    return pulse,w,beta


def edited(clip,notched,face,roi,g0,goal,direction,original,combination,pulse,w,window_id,condition,rounding_rng=None):
    v=np.asarray(direction,float)
    mu=clip.mean(axis=0,dtype=np.float64)
    if combination=='ED2AN-S':
        original_roi=green(weighted_rgb(clip,roi))
        R=float(np.sqrt(np.mean(filter4(original_roi,[original/60-.15,original/60+.15])**2)))
        A=R/(v[1]*np.sqrt(np.mean(goal**2)))
    else:
        R=pulse
        A=R/(v[1]*np.sqrt(np.mean(goal**2))*w)
    check('Finite nonnegative editor amplitude',np.isfinite(A) and A>=0)
    value=np.empty(clip.shape,np.float32)
    n_clipped_values=n_clipped_pixels=0
    for i in range(300):
        if combination=='ED2W-S':
            x=notched[i].astype(np.float64)*(1+face[i,:,:,None]*A*v*goal[i])
        elif combination=='ED2Wadd-S':
            x=notched[i].astype(np.float64)+face[i,:,:,None]*mu*A*v*goal[i]
        else:
            x=notched[i].astype(np.float64)+roi[i,:,:,None]*mu*A*v*goal[i]
        clipped=(x<0)|(x>255)
        n_clipped_values+=int(clipped.sum());n_clipped_pixels+=int(clipped.any(axis=2).sum())
        value[i]=np.clip(x,0,255).astype(np.float32)
    if rounding_rng is None:result=stochastic_round(value,window_id,combination,condition)
    else:
        result=np.empty(value.shape,np.uint8)
        for i,frame in enumerate(value):result[i]=np.floor(frame.astype(np.float64)+rounding_rng.random(frame.shape)).astype(np.uint8)
    return result,dict(injection_A=float(A),R_pulse=float(pulse),injection_R=float(R),w=float(w),
        n_clipped_channel_values=n_clipped_values,n_clipped_pixel_times=n_clipped_pixels)


def amplitude_all(before,after):
    absolute=np.abs(after.astype(np.int16)-before.astype(np.int16))
    pixel=absolute.max(axis=3)
    result=dict(n_pixel_times=int(pixel.size),n_changed_pixel_times=int(np.count_nonzero(pixel)),
                changed_pixel_fraction=float(np.count_nonzero(pixel)/pixel.size))
    for name,array in [('channel_absolute',absolute),('pixel_max_absolute',pixel)]:
        q=np.quantile(array,[.25,.5,.75])
        result.update({name+'_p25':float(q[0]),name+'_median':float(q[1]),name+'_p75':float(q[2]),name+'_max':int(array.max())})
    return result


def nonpulsatile_all(before,after,original,g0,bands):
    f,p0=power(before);_,p1=power(after)
    inside=np.zeros(len(f),bool)
    for low,high in bands:inside|=(f>=low)&(f<=high)
    outside=(f>0)&~inside
    harmonic=filter4(g0,[2*original/60-.15,2*original/60+.15])
    design=np.column_stack([np.ones(len(g0)),g0,harmonic])
    r0=before-design@np.linalg.lstsq(design,before,rcond=None)[0]
    r1=after-design@np.linalg.lstsq(design,after,rcond=None)[0]
    _,e0=power(r0);_,e1=power(r1)
    broad=(f>=.6)&(f<=3.3)
    return dict(nu_out=ratio(p1[outside].sum(),p0[outside].sum()),
        nu_in=ratio(e1[inside].sum(),e0[inside].sum()),
        nonpulsatile_removal_fraction=ratio(e0[inside].sum()-e1[inside].sum(),p0[broad].sum()))


def measured_rows(base,before,after,original,goal_ref,condition):
    result=dict(readout=[],validity=[],sham=[],diagnostic=[])
    for reader in ['POS','CHROM','FACE_GREEN','ROI_GREEN']:
        for layer in BANDS:
            if 'range_low' in base and layer=='L0' and not 45<=base['nominal_target_HR']<=150:continue
            hb=common_hr(before[reader],layer);ha=common_hr(after[reader],layer)
            row=dict(**base,reader=reader,layer=layer,original_HR=original[layer],target_HR=goal_ref[layer],
                baseline_HR=hb,edited_HR=ha,baseline_hit=bool(abs(hb-original[layer])<=GRID+1e-6),
                target_hit=bool(abs(ha-goal_ref[layer])<=GRID+1e-6),
                original_hit_after=bool(abs(ha-original[layer])<=GRID+1e-6),
                target_absolute_error=abs(ha-goal_ref[layer]))
            result['readout' if reader in ['POS','CHROM'] else 'diagnostic'].append(row)
            if condition=='sham' and reader in ['POS','CHROM']:
                f,p0=power(before[reader]);_,p1=power(after[reader])
                result['sham'].append(dict(**base,reader=reader,layer=layer,
                    rho_sham=ratio(interval_power(f,p1,original[layer]),interval_power(f,p0,original[layer])),
                    original_hit_after=row['original_hit_after']))
        if condition=='tgt' and reader!='ROI_GREEN':
            result['validity'].append(dict(**base,signal=reader,
                **independent_validity(before[reader],after[reader],original['HAR'],goal_ref['HAR'])))
    return result
