"""One original recording/arm/view job; all face-derived scratch is hash-deleted."""
import os
os.environ['ST9_NAMESPACE']='st9g'
from src.st9e.temporal import *
from src.st9e.model_view import ViewWorker,crop_native
from src.st9g.preprocess import diff_normalize_data,diff_normalize_label,standardized_label
import resource
import shutil


def preprocess(view,label,model,offset,length):
    if model=='PhysNet':x=diff_normalize_data(view);y=diff_normalize_label(label)
    else:x=view;y=standardized_label(label)
    return np.ascontiguousarray(x[offset:offset+length],dtype=np.float32),np.ascontiguousarray(y[offset:offset+length],dtype=np.float32)


def generate_job(job):
    model,arm,rid=job['model'],job['arm'],job['recording_id'];began=time.monotonic()
    settings=json.loads((REPORTS/'st9g_pool_settings.json').read_text());items=job['items']
    marker=REPORTS/('st9g_stop_'+model+'_'+arm+'.json')
    if marker.exists():return dict(status='scope_stopped',job=job['id'],items=[])
    _,pure,allowed=indices();row=pure[rid];files=frame_files(row);gt=contact(row,'PURE',allowed)
    tau=next(int(r['tau_rec']) for r in rows('outputs/st7b_tau_recordings.csv') if r['recording_id']==rid)
    clip_all=read_pure(files,0,len(files));worker=ViewWorker();outputs=[];exceptions=[]
    directory=CACHE/'pool_scratch'/job['id'];directory.mkdir(parents=True,exist_ok=True)
    bulk=Path(PATHS['bulk'])/'st9g/augmented'/model/arm/rid;bulk.mkdir(parents=True,exist_ok=True)
    try:
        # Crop boxes are original-frame boxes on the full recording's fixed grid.
        phys_all,physboxes=worker.view(clip_all,'PhysNet')
        fpboxes=None
        if model=='FactorizePhys':
            discarded,fpboxes=worker.view(clip_all,'FactorizePhys');del discarded
        source_seconds=time.monotonic()-began
        chunks=sorted({int(r['chunk_order']) for r in items})
        for c in chunks:
            if marker.exists():break
            check('No disk resource stop before pool writes',not (ROOT/'reports/st9/st9_disk_limit.json').exists())
            check('Bulk free >=50GB',shutil.disk_usage(bulk).free>=50_000_000_000)
            current=[r for r in items if int(r['chunk_order'])==c];reference=current[0]
            context_start=int(reference['context_start']);length=int(reference['chunk_length']);offset=int(reference['context_offset'])
            original=float(reference['reference_HR']);context_setup=time.monotonic()
            if arm!='i':
                clip=clip_all[context_start:context_start+300]
                face,roi,failure=face_masks(clip);notched=notch_all(clip,bands_for(original))
                physview=phys_all[context_start:context_start+300]
                mapped=physboxes if model=='PhysNet' else fpboxes[(context_start+np.arange(300))//30]
            context_seconds=time.monotonic()-context_setup
            for r in current:
                tick=time.monotonic();k=int(r['augmentation_index']);target=float(r['target_HR'])
                base=dict(r);base.update(source_setup_seconds_allocated=source_seconds/len(items),context_setup_seconds_allocated=context_seconds/len(current))
                if str(r['target_excluded']).lower()=='true':
                    outputs.append(dict(**base,generation_excluded=True,exclusion_reason='iv_target100draws',CPU_seconds=time.monotonic()-tick));continue
                rng=np.random.default_rng([20261137,int(r['rounding_h'])]);diagnostics={}
                if arm=='i':
                    scale=target/original;positions,plan=shifted_start(int(r['start_frame']),length,len(files),scale,tau)
                    if positions is None:
                        outputs.append(dict(**base,generation_excluded=True,exclusion_reason='source_or_label_interval_exceeds_recording',CPU_seconds=time.monotonic()-tick));continue
                    sourcepos,labelpos=positions;output=np.empty((length,)+clip_all.shape[1:],np.uint8)
                    for i,p in enumerate(sourcepos):
                        left=int(np.floor(p));right=int(np.ceil(p));fraction=p-left
                        check('No edge hold/reflection/extrapolation in i video',0<=left<=right<len(clip_all),scope=model+'_'+arm)
                        value=(1-fraction)*clip_all[left].astype(np.float64)+fraction*clip_all[right]
                        output[i]=np.floor(value+rng.random(value.shape)).astype(np.uint8)
                    check('No edge hold/reflection/extrapolation in i label',labelpos.min()>=0 and labelpos.max()<=len(gt)-1,scope=model+'_'+arm)
                    label=np.interp(labelpos,np.arange(len(gt)),gt)
                    mapped=physboxes if model=='PhysNet' else fpboxes[np.rint(sourcepos).astype(int)//30]
                    view,_=worker.view(output,model,reuse_boxes=mapped,reuse_boxes_per_frame=model=='FactorizePhys')
                    x,y=preprocess(view,label,model,0,length)
                    ctx,ctxplan=shifted_start(plan['source_start']+(length-300)*scale/2,300,len(gt),scale,tau)
                    achieved=common_hr(np.interp(ctx[1],np.arange(len(gt)),gt),'HAR') if ctx is not None else np.nan
                    diagnostics.update(scale=scale,source_start=plan['source_start'],source_start_moved=plan['source_start']!=int(r['start_frame']),
                        label_shift_source_frames=plan['label_shift_source_frames'],achieved_context_HR=achieved,
                        target_difference=achieved-target,no_edge_hold_reflection_extrapolation=True,reflection_used=False,
                        numeric_inward_ulp_steps=plan['numeric_inward_ulp_steps'])
                else:
                    g0,gstar=ppg_vectors(gt,context_start,original,target,tau)
                    pulse,w,beta=calibration(physview,crop_native(notched,physboxes[0]),crop_native(face,physboxes[0]),g0,settings['direction_RGB'])
                    output,diag=edited(clip,notched,face,roi,g0,gstar,settings['direction_RGB'],original,'ED2W-S',pulse,w,
                        r['chunk_id'],arm,rounding_rng=rng,inject=injection_waveform(gt,context_start,original,target,tau,settings.get('injection','original')))
                    label=raw_target_label(gt,context_start,original,target)
                    view,_=worker.view(output,model,reuse_boxes=mapped,reuse_boxes_per_frame=model=='FactorizePhys')
                    x,y=preprocess(view,label,model,offset,length)
                    achieved=common_hr(label,'HAR')
                    diagnostics.update(scale=target/original,source_start=int(r['start_frame']),source_start_moved=False,
                        label_shift_source_frames=0.,achieved_context_HR=achieved,target_difference=achieved-target,
                        reflection_used=reflection_used(gt,context_start,original,target),mask_failure_fraction=failure,
                        injection_A=diag['injection_A'],beta=beta,w=w,no_edge_hold_reflection_extrapolation='',numeric_inward_ulp_steps=0)
                check('Exactly prescribed stored chunk shape',x.shape==(length,72,72,3) and y.shape==(length,))
                check('Finite stored input and label',np.isfinite(x).all() and np.isfinite(y).all())
                native=directory/f'c{c}_k{k}_consumed_native.npy';np.save(native,output);delete_temp(native)
                ip=bulk/f'c{c}_k{k}_input.npy';lp=bulk/f'c{c}_k{k}_label.npy'
                check('Never overwrite generated research tensors',not ip.exists() and not lp.exists())
                np.save(ip,x);np.save(lp,y)
                outputs.append(dict(**base,**diagnostics,generation_excluded=False,exclusion_reason='',tau_rec=tau,
                    input_path=str(ip),label_path=str(lp),input_bytes=ip.stat().st_size,label_bytes=lp.stat().st_size,
                    input_sha256=sha(ip),label_sha256=sha(lp),stored_format='THWC_float32',
                    CPU_seconds=time.monotonic()-tick+source_seconds/len(items)+context_seconds/len(current)))
                del output,view,x,y,label
            if arm!='i':del clip,face,roi,notched,physview
            write_rows(REPORTS/'pool_jobs'/(job['id']+'.csv'),outputs,sorted(set().union(*(r.keys() for r in outputs))))
    except Exception as error:
        import traceback
        problem=dict(job=job['id'],model=model,arm=arm,error_type=type(error).__name__,at_utc=now(),
            stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
        if isinstance(error,RequiredStop):problem.update(check=str(error),scope=error.scope)
        save_json(marker,problem);exceptions.append(problem)
    finally:worker.close()
    if outputs:write_rows(REPORTS/'pool_jobs'/(job['id']+'.csv'),outputs,sorted(set().union(*(r.keys() for r in outputs))))
    result=dict(status='complete' if not exceptions and len(outputs)==len(items) else 'scope_stopped',job=job['id'],model=model,arm=arm,
        planned_items=len(items),processed_items=len(outputs),generated_items=sum(not r['generation_excluded'] for r in outputs),
        wall_seconds=time.monotonic()-began,source_setup_seconds=locals().get('source_seconds'),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        manifest_path=str(REPORTS/'pool_jobs'/(job['id']+'.csv')),exceptions=exceptions)
    save_json(REPORTS/'pool_jobs'/(job['id']+'.json'),result);return result
