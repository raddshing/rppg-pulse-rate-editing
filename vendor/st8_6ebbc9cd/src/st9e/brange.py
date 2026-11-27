"""Preregistered 360-case ED2-W recording-lag gate with original model boxes."""
from src.st9e.editor import *
from src.st9e.model_view import ViewWorker,crop_native
from src.st9e.masks import face_from_points
from src.st9e.brange_tables import summarize


def view_signals(view):
    rgb=view.mean(axis=(1,2),dtype=np.float64)
    with np.errstate(divide='ignore',invalid='ignore'):
        return dict(POS=band_reader_pos(rgb),CHROM=band_reader_chrom(rgb))


def readout_rows(case,view,before,after,original,target_ref,tau,cell):
    output=[]
    base=dict(dataset='PURE',window_id=case['window_id'],population='fixed_PURE_training_Brange360',
        ground_truth_path='unchanged_Stage1_PURE_contact_common_HR;target=injected_shifted_waveform',
        task=case['task'],range_low=int(case['range_low']),range_high=int(case['range_high']),
        nominal_target_HR=float(case['target_HR']),harmonic_overlap=False,model_view=view,
        combination='ED2W-S',condition='tgt',tau_frames=int(tau),decomposition_cell=cell)
    for reader in ['POS','CHROM']:
        for layer in BANDS:
            if layer=='L0' and not 45<=base['nominal_target_HR']<=150:continue
            hb=common_hr(before[reader],layer);ha=common_hr(after[reader],layer)
            output.append(dict(**base,reader=reader,layer=layer,original_HR=original[layer],target_HR=target_ref[layer],
                baseline_HR=hb,edited_HR=ha,baseline_hit=bool(abs(hb-original[layer])<=GRID+1e-6),
                target_hit=bool(abs(ha-target_ref[layer])<=GRID+1e-6),
                original_hit_after=bool(abs(ha-original[layer])<=GRID+1e-6),target_absolute_error=abs(ha-target_ref[layer])))
    return output


def baseline_rows(wid,task,view,signals,original):
    return [dict(window_id=wid,task=task,population='fixed_PURE_B_sample_unique_windows',
        ground_truth_path='unchanged_Stage1_PURE_contact_common_HR',model_view=view,reader=reader,layer=layer,
        original_HR=original[layer],baseline_HR=common_hr(signals[reader],layer),
        baseline_hit=bool(abs(common_hr(signals[reader],layer)-original[layer])<=GRID+1e-6))
        for reader in ['POS','CHROM'] for layer in BANDS]


def comparison_with_s5(baselines,summary_path):
    old=pd.read_csv(ROOT/'outputs/st5_brange_baselines.csv',float_precision='round_trip')
    new=pd.DataFrame(baselines)
    keys=['window_id','model_view','reader','layer']
    joined=new.merge(old,on=keys,suffixes=('_new','_S5'),validate='one_to_one')
    joined['baseline_HR_equal']=joined.baseline_HR_new.eq(joined.baseline_HR_S5)
    joined['baseline_hit_equal']=joined.baseline_hit_new.eq(joined.baseline_hit_S5)
    joined.to_csv(ROOT/'outputs/st9e_brange_baseline_comparison.csv',index=False)
    current=pd.read_csv(summary_path,float_precision='round_trip')
    previous=pd.read_csv(ROOT/'outputs/st5_brange.csv',float_precision='round_trip')
    previous=previous[previous.combination.eq('ED2W-S')]
    columns=['model_view','combination','reader','layer','range_low','range_high','task_group']
    current.merge(previous,on=columns,suffixes=('_tau_rec_original_boxes','_S5'),validate='one_to_one').to_csv(
        ROOT/'outputs/st9e_brange_S5_comparison.csv',index=False)
    return dict(baseline_cells=len(joined),baseline_HR_equal=int(joined.baseline_HR_equal.sum()),
        baseline_hit_equal=int(joined.baseline_hit_equal.sum()))


def clean_consumed_cache():
    directory=CACHE/'brange'
    for path in sorted(directory.rglob('*')):
        if path.is_file():delete_temp(path)


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    settings_path=REPORTS/'st9e_brange_settings.json';settings=json.loads(settings_path.read_text())
    check('E3R-b actual settings committed',sha(settings_path) in (ROOT/'reports/ST4_INSTR03_PREFIX.md').read_text())
    for path,digest in settings['source_hashes'].items():check('E3R-b fixed dependency unchanged',sha(ROOT/path)==digest)
    check('Fixed landmark weight unchanged',sha(PATHS['face_landmarker_model'])==settings['landmark_weight_sha256'])
    check('Prescribed S7 timings finished before E3R-b',json.loads((ROOT/'reports/st9/gpu_accounts/st9s_S7_1c.json').read_text())['status']=='closed')
    for name in settings['frozen_inputs']:source_hash(name)
    cases=pd.read_csv(ROOT/'outputs/st4_editor_brange_sample.csv',dtype={'task':str})
    check('Unchanged360 B cases',len(cases)==360)
    _,pure,allowed=indices();roidef=json.loads((ROOT/'outputs/st3_roi_indices.json').read_text())
    direction=settings['direction_RGB'];taus={r['recording_id']:int(r['tau_rec']) for r in rows('outputs/st7b_tau_recordings.csv')}
    result=[];baselines=[];detection=[];context_records=[];started=time.monotonic();done=0
    directory=CACHE/'brange';directory.mkdir(exist_ok=True)
    check('No completed E3R-b is repeated',not (REPORTS/'st9e_brange_selection.json').exists())
    check('No partial E3R-b overwritten',not (ROOT/'outputs/st9e_brange_readout_windows.csv').exists())
    worker=ViewWorker()
    try:
        for rid,group in cases.groupby('recording_id',sort=True):
            row=pure[rid];files=frame_files(row);gt=contact(row,'PURE',allowed)
            first=read_pure(files,0,1)[0];wanted=set(group.window_index.astype(int))
            previous=None;frames=[];points=[];oldmasks=[]
            with vision.FaceLandmarker.create_from_options(tracking_options()) as tracker:
                for i in range((max(wanted)+1)*300):
                    rgb=cv2.cvtColor(cv2.imread(str(files[i])),cv2.COLOR_BGR2RGB)
                    value=tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),round(i*1000/30))
                    p=np.full((478,2),np.nan,np.float32)
                    if value.face_landmarks:
                        h,w=rgb.shape[:2]
                        observed=np.asarray([[v.x*w,v.y*h] for v in value.face_landmarks[0]],dtype=np.float64)
                        p[:]=observed;previous=skin_mask(rgb.shape,observed,roidef)
                    j=i//300
                    if j not in wanted:continue
                    check('Inherited B ROI has preceding observed mask',previous is not None and previous.any())
                    frames.append(rgb);points.append(p);oldmasks.append(previous)
                    if (i+1)%300:continue
                    clip=np.asarray(frames);roi=np.asarray(oldmasks);rawpoints=np.asarray(points)
                    face,failure=face_from_points(rawpoints,clip.shape[1:]);wid=f'PURE|{rid}|{j}';start=j*300
                    original={layer:common_hr(gt[start:start+300],layer) for layer in BANDS}
                    notched=notch_all(clip,bands_for(original['HAR']));before={};boxes={}
                    for view in ['PhysNet','FactorizePhys']:
                        resized,box=worker.view(clip,view,first if view=='PhysNet' else None)
                        before[view]=view_signals(resized);boxes[view]=box
                        if view=='PhysNet':physview=resized;physbox=box[0]
                        baselines.extend(baseline_rows(wid,group.task.iloc[0],view,before[view],original))
                    context_path=directory/f'{rid}_{j}_masks.npz'
                    np.savez(context_path,points=rawpoints,roi_bits=np.packbits(roi),roi_shape=np.asarray(roi.shape))
                    meta=dict(window_id=wid,recording_id=rid,window_index=j,mask_path=str(context_path),mask_sha256=sha(context_path),
                        boxes={view:box.tolist() for view,box in boxes.items()},original_HR=original,primary_j0_edits=[])
                    detection.append(dict(window_id=wid,recording_id=rid,W_landmark_failure_fraction=failure,
                        ED2_ROI_policy='previous_observed_mask_continuous_recording_tracker',
                        W_mask_policy='linear_interpolation_nearest_endpoints',tau_rec=taus[rid],
                        PhysNet_policy='original_recording_first_HC1.5_including_j0',FactorizePhys_policy='original_Y5F_dynamic30'))
                    for case in group[group.window_index.eq(j)].sort_values('range_low').to_dict('records'):
                        check('Frozen B original HR unchanged',abs(original['HAR']-case['original_HAR'])<1e-6)
                        tau=taus[rid];g0,gstar=ppg_vectors(gt,start,case['original_HAR'],case['target_HR'],tau)
                        pulse,w,beta=calibration(physview,crop_native(notched,physbox),crop_native(face,physbox),g0,direction)
                        target_ref={layer:common_hr(gstar,layer) for layer in BANDS}
                        seedcond=f'tgt-{case["range_low"]}-{case["range_high"]}'
                        output,_=edited(clip,notched,face,roi,g0,gstar,direction,original['HAR'],'ED2W-S',pulse,w,wid,seedcond)
                        temp=directory/f'{rid}_{j}_{int(case["range_low"])}_tau_rec_native.npy';np.save(temp,output)
                        for view in ['PhysNet','FactorizePhys']:
                            resized,_=worker.view(output,view,reuse_boxes=boxes[view])
                            result.extend(readout_rows(case,view,before[view],view_signals(resized),original,target_ref,tau,'tau_rec_original_boxes'))
                        if j==0:meta['primary_j0_edits'].append(dict(range_low=int(case['range_low']),path=str(temp),sha256=sha(temp)))
                        else:delete_temp(temp)
                        del output
                        done+=1;write_rows('outputs/st9e_brange_readout_windows.csv',result)
                        progress('brange',done,360,started)
                    context_records.append(meta)
                    save_json(REPORTS/'st9e_brange_contexts_server_only.json',dict(contexts=context_records,frame_geometry=True,package_exclude=True))
                    write_rows('outputs/st9e_brange_baselines.csv',baselines);write_rows('outputs/st9e_brange_detection.csv',detection)
                    frames=[];points=[];oldmasks=[];del clip,roi,face,notched,physview,rawpoints
                    budget_check((time.monotonic()-started)/done*(360-done)+14400)
    finally:worker.close()
    check('Every prescribed B target processed',done==360 and len(context_records)==226)
    selection=summarize(result,baselines,'st9e_brange','tau_rec_original_boxes')
    comparison=comparison_with_s5(baselines,ROOT/'outputs/st9e_brange.csv')
    if selection['gate_passed']:clean_consumed_cache()
    return dict(cases=360,unique_windows=len(context_records),gate_passed=selection['gate_passed'],
        conditional_decomposition_required=not selection['gate_passed'],**comparison,GPU_seconds=0,rPPG_model_inference=False)


if __name__=='__main__':finish('brange',main)
