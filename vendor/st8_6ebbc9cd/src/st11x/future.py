"""AppendixC array/reference/aggregation paths, dormant until instruction06.

No discovery, decoding, checkpoint loading or dataset execution occurs on import.
All dataset entry points check a future explicit authorization. Model adapters are
the retained Stage1 spatial and inference functions passed with their fixed config;
their entire uniform-grid recording is supplied once, never independently per10s.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter,sosfiltfilt,find_peaks,periodogram,detrend
from src.eval import grid
from src.st11x.pathways import (BANDS,GRID,interpolate_face_input,reference_30hz,
    complete_windows,mmpd_b_membership,cohort_selection,require_instruction06,paired_bootstrap_indices)

ROOT=Path(__file__).resolve().parents[2]


def frozen_reference_functions():
    """Execute only retained pure function definitions, not old operational imports."""
    path=ROOT/'src/st4/signals.py';tree=ast.parse(path.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    context=dict(np=np,BANDS=BANDS,GRID=GRID,grid=grid,CubicSpline=CubicSpline,butter=butter,
        sosfiltfilt=sosfiltfilt,find_peaks=find_peaks,periodogram=periodogram)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),context)
    return context


def stage1_temporal_input(spatial_input,times,preprocess_config,base_loader):
    """Every Stage1 channel transform and chunk length, including public15 configs."""
    regular,timeline=interpolate_face_input(spatial_input,times)
    blocks=[]
    for name in preprocess_config.DATA_TYPE:
        value=regular.copy()
        if name=='DiffNormalized':value=base_loader.diff_normalize_data(value)
        elif name=='Standardized':value=base_loader.standardized_data(value)
        elif name!='Raw':raise ValueError('Unsupported frozen Stage1 input transform')
        blocks.append(value)
    data=np.concatenate(blocks,axis=-1)
    length=int(preprocess_config.CHUNK_LENGTH)
    if preprocess_config.DO_CHUNK:
        count=len(data)//length;chunks=np.asarray([data[i*length:(i+1)*length] for i in range(count)]);starts=np.arange(count)*length
    else:chunks=data[None];starts=np.asarray([0]);length=len(data)
    return dict(input_chunks=chunks,chunk_starts=starts,chunk_length=length,frame_grid=timeline,
        original_frames=len(spatial_input),grid_samples=len(timeline),valid_prediction_samples=len(chunks)*length)


def frozen_spatial_input(frames,preprocess_config,base_loader_instance):
    cfg=preprocess_config;c=cfg.CROP_FACE;d=c.DETECTION
    return base_loader_instance.crop_face_resize(frames,c.DO_CROP_FACE,c.BACKEND,c.USE_LARGE_FACE_BOX,c.LARGE_BOX_COEF,
        d.DO_DYNAMIC_DETECTION,d.DYNAMIC_DETECTION_FREQUENCY,d.USE_MEDIAN_FACE_BOX,cfg.RESIZE.W,cfg.RESIZE.H)


def reference_rows(contact30,timeline,windows):
    helpers=frozen_reference_functions();out=[]
    for item in windows:
        bounds=complete_windows(timeline,[float(item['start_seconds'])])[0];start=bounds['grid_start']
        if not bounds['available'] or start+300>len(contact30):
            out.append(dict(item,available=False,exclusion='incomplete10s_contact_or_frame_support'));continue
        raw=np.asarray(contact30)[start:start+300];h={l:helpers['common_hr'](raw,l) for l in BANDS}
        diagnostic=helpers['reference_estimators'](raw,h['HAR']);classification=helpers['classify'](h['HAR'],diagnostic['h_ACF'],diagnostic['h_IBI'])
        out.append(dict(item,available=True,grid_start=start,h_FFT_HAR=h['HAR'],h_FFT_L0=h['L0'],h_FFT=h['HAR'],reference_class=classification,**diagnostic))
    return out


def mmpd_b_references(receipt,recordings,load_frame_aligned_contact):
    """Only future06: read B labels through original30Hz loader; automatic Qstar."""
    require_instruction06(receipt,'reference_preparation_stage',False)
    if len(recordings)!=435 or len({r['subject_id'] for r in recordings})!=22:raise ValueError('Fixed B435/22 population mismatch')
    helpers=frozen_reference_functions();out=[];segments=[]
    # Use the exact C4 defect function with its output hooks replaced locally.
    source=ROOT/'src/st3/c4.py';tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='defect_map')
    context=dict(np=np,pd=pd,write_rows=lambda *a,**k:None,update=lambda *a,**k:None)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),context)
    for rec in recordings:
        raw=np.asarray(load_frame_aligned_contact(rec));rows=[]
        for j in range(5):
            window=raw[j*300:j*300+300];h=helpers['common_hr'](window,'HAR')
            y=detrend(window.astype(np.float64),type='linear');f,p=periodogram(y,fs=30,window='hann',nfft=len(y),detrend=False,scaling='density')
            den=p[(f>=.6)&(f<=15)].sum();ratio=float(p[(f>3.5*h/60)&(f<=15)].sum()/den) if den>0 and np.isfinite(h) else np.nan
            rows.append(dict(window_id='MMPD|'+rec['recording_id']+'|'+str(j),recording_id=rec['recording_id'],subject_id=rec['subject_id'],window_index=j,
                start_seconds=10*j,T1=-1,T1b=-1,T1b_high_freq=int(ratio>=.10676) if np.isfinite(ratio) else -1,Q3_power_above_3p5_f0_ratio=ratio))
        frame=pd.DataFrame(rows);defects,parts=context['defect_map'](frame,{rec['recording_id']:raw});segments.extend(parts)
        refs=reference_rows(raw,np.arange(len(raw))/30,rows)
        for r in refs:
            d=defects.loc[r['window_id']].to_dict();r.update(d)
            r.update(mmpd_b_membership(d['defect_any'],r.get('reference_class','unavailable'),r.get('h_FFT_HAR',np.nan),r.get('h_FFT_L0',np.nan)))
            r['PI_confirmation_path']=False;r['T1b_high_frequency_A_threshold']=.10676;out.append(r)
    return out,segments


def external_references(receipt,dataset,records,read_provided_times,read_contact,read_fixed_mcd_windows):
    """Stage1 of future06; no video pixels, POS alignment or model outputs used."""
    require_instruction06(receipt,'reference_preparation_stage',False)
    result=[];exclusions=[]
    for rec in records:
        times=read_provided_times(rec)
        if times is None:
            exclusions.append(dict(recording_id=rec['recording_id'],reason='missing_provided_frame_times',container_fps_substituted=False));continue
        timeline=interpolate_face_input(np.zeros((len(times),1)),times)[1]
        if dataset=='MCD-rPPG':
            # Preserve S5-3M2 and04M window membership/contact values, including gaps.
            refs=read_fixed_mcd_windows(rec)
        else:
            contact,fs=read_contact(rec);duration=float(times[-1]-times[0])+float(np.median(np.diff(times)))
            if dataset=='VIPL-HR' and (rec['camera']!='source3' or abs(len(contact)/fs-duration)/duration>.01):
                exclusions.append(dict(recording_id=rec['recording_id'],reason='camera_or_duration_gt1percent'));continue
            contact30=reference_30hz(contact,fs)
            windows=[dict(window_id=dataset+'|'+rec['recording_id']+'|'+str(j),recording_id=rec['recording_id'],subject_id=rec['subject_id'],start_seconds=j*10)
                for j in range(int(np.floor(duration/10+1e-9)))]
            refs=reference_rows(contact30,timeline,windows)
        for r in refs:
            item=dict(rec,**{k:v for k,v in r.items() if k not in rec});item.update(r)
            if not item.get('available',True):exclusions.append(dict(recording_id=rec['recording_id'],window_id=r['window_id'],reason=item['exclusion']));continue
            if dataset!='MCD-rPPG':item.update(BVP_samples=len(contact),BVP_fs=fs,video_duration_seconds=duration,provided_frame_times=True)
            item.update(cohort_selection(dataset,item));result.append(item)
    if dataset=='VIPL-HR':
        primary=[r for r in result if r['primary']];sufficient=len(primary)>=50 and len({r['subject_id'] for r in primary})>=20
        for r in result:r['dataset_sufficiency_met']=sufficient
    return result,exclusions


def codec_diagnostics(receipt,window_rgb,container_keyframes):
    """Future MCD before model-output aggregation; RGB already follows fixed HC view."""
    require_instruction06(receipt,'reference_preparation_stage',False)
    from src.st3.readers import pos
    out=[]
    for item in window_rgb:
        rgb=np.asarray(item['rgb']);signals={'GREEN':rgb[:,1],'POS':pos(rgb,fs=30,filtered=False)}
        for name,wave in signals.items():
            f,p=periodogram(wave,fs=30,nfft=512);band=(f>=.6)&(f<=3.3);near=band&(np.abs(f-2.5)<=GRID/60)
            den=p[band].sum();peak=float(f[band][np.argmax(p[band])])
            out.append(dict(window_id=item['window_id'],reader=name,power_fraction_2p5hz=float(p[near].sum()/den) if den>0 else np.nan,
                band_peak_in_2p5hz_cell=bool(abs(peak-2.5)<=GRID/60),band_peak_Hz=peak))
    return dict(window_values=out,keyframe_intervals=np.diff(container_keyframes).tolist(),
        aggregate={name:dict(median_power_fraction=float(np.nanmedian([r['power_fraction_2p5hz'] for r in out if r['reader']==name])),
            band_peak_fraction=float(np.mean([r['band_peak_in_2p5hz_cell'] for r in out if r['reader']==name]))) for name in ['GREEN','POS']})


def common_prediction_hr(wave,layer,diff_flag,notch=False):
    band=BANDS[layer];value=grid._prepare(wave,diff_flag,band,'default',30)
    if not notch:return grid._estimate(value,'periodogram',band,30)
    f,p=periodogram(value,fs=30,nfft=512);p[(f>=2.45)&(f<=2.55)]=0;inside=(f>=band[0])&(f<=band[1])
    return float(f[inside][np.argmax(p[inside])]*60)


def evaluate_once(receipt,dataset,run,reference_commit,ledger_directory,recordings,read_frames_times,
                  base_loader,preprocess_config,stage1_infer_recording,reference_windows,output_directory):
    """Future06 runner uses existing Stage1 model adapter on one full recording.

    The supplied adapter accepts the exact config and chunks and returns the
    recording waveform; no external defaults or model roster are selected here.
    The caller must pass every authorized run from the committed06 roster.
    """
    key=hashlib.sha256((dataset+'|'+run['run_id']).encode()).hexdigest();ledger=Path(ledger_directory)/(key+'.json')
    require_instruction06(receipt,reference_commit,ledger.exists())
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('x') as f:json.dump(dict(status='started',dataset=dataset,run_id=run['run_id'],reference_commit=reference_commit),f)
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True);manifest=[];predictions=[]
    for rec in recordings:
        frames,times=read_frames_times(rec)
        spatial=frozen_spatial_input(frames,preprocess_config,base_loader)
        prepared=stage1_temporal_input(spatial,times,preprocess_config,type(base_loader))
        wave=np.asarray(stage1_infer_recording(run,prepared));diff=preprocess_config.LABEL_TYPE=='DiffNormalized'
        dest=directory/(rec['recording_id'].replace('/','__')+'.npz')
        with dest.open('xb') as f:np.savez(f,pred=wave,chunk_starts=prepared['chunk_starts'],chunk_len=prepared['chunk_length'],fs=30.,diff_flag=diff,frame_grid=prepared['frame_grid'])
        digest=hashlib.sha256(dest.read_bytes()).hexdigest();manifest.append(dict(recording_id=rec['recording_id'],path=str(dest),sha256=digest))
        for w in reference_windows[rec['recording_id']]:
            bounds=complete_windows(prepared['frame_grid'],[w['start_seconds']])[0];start=bounds['grid_start']
            if not bounds['available'] or start+300>len(wave):predictions.append(dict(window_id=w['window_id'],available=False,exclusion='incomplete_prediction_support'));continue
            for layer in BANDS:
                hr=common_prediction_hr(wave[start:start+300],layer,diff)
                predictions.append(dict(run_id=run['run_id'],window_id=w['window_id'],layer=layer,available=True,HR=hr,waveform_sha256=digest,
                    near150=bool(146.484375<=hr<=153.515625),notch_HR=common_prediction_hr(wave[start:start+300],layer,diff,True) if dataset=='MCD-rPPG' else None))
    ledger.write_text(json.dumps(dict(status='complete',dataset=dataset,run_id=run['run_id'],reference_commit=reference_commit,manifest=manifest),indent=2)+'\n')
    return manifest,predictions
