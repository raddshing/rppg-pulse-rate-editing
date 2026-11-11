"""One recording, fixed A targets, inherited generator and temporary cache only."""
from src.st30g.helpers import *
import resource


def _generate_job(job):
    settings=verify();began=time.monotonic();rid=job['recording_id'];model=job['model'];items=job['items']
    outcome=ROOT/'reports/st30g/A_jobs'/(job['id']+'.json');assert not outcome.exists()
    phase_path=ROOT/'reports/st30g/A_jobs'/(job['id']+'_phase.json')
    row=next(r for r in rows('data/index/pure_index_st1.csv') if r['recording_id']==rid)
    files=frame_files(row);guard={r['path']:r for r in rows('outputs/st12_input_hash_checks.csv')}
    def intact():
        for p in [*files,Path(row['label_path'])]:
            g=guard[str(p)];s=snapshot(p)
            assert all(int(g[k])==v for k,v in s.items()),str(p)
    intact();gt=contact(row,'PURE',set())
    tau=next(int(r['tau_rec']) for r in rows('outputs/st7b_tau_recordings.csv') if r['recording_id']==rid)
    native=read_pure(files,0,len(files));phys=ViewWorker();rhythm=RhythmViewWorker() if model=='RhythmFormer' else None
    outputs=[];error=None;checks=[];manifest='reports/st30g/A_jobs/'+job['id']+'.csv'
    try:
        phys_all,boxes=phys.view(native,'PhysNet');source_seconds=time.monotonic()-began
        for c in sorted({int(r['chunk_order']) for r in items}):
            capacity();current=[r for r in items if int(r['chunk_order'])==c];reference=current[0]
            start=int(reference['context_start']);offset=int(reference['context_offset']);length=int(reference['chunk_length']);h0=float(reference['reference_HR'])
            tick=time.monotonic();clip=native[start:start+300]
            needs_edit=any(r['arm']!='i' for r in current)
            if needs_edit:
                face,roi,failure=face_masks(clip);notched=notch_all(clip,bands_for(h0));physview=phys_all[start:start+300]
            context_seconds=time.monotonic()-tick
            for r in current:
                tick=time.monotonic();arm=r['arm'];k=int(r['augmentation_index']);target=float(r['target_HR'])
                rng=np.random.default_rng([20261202,int(r['rounding_h'])]);diag={}
                if arm=='i':
                    scale=target/h0;positions,plan=shifted_start(int(r['start_frame']),length,len(files),scale,tau)
                    if positions is None:
                        outputs.append(dict(r,generation_excluded=True,exclusion_reason='source_or_label_interval_exceeds_recording',
                            CPU_seconds=time.monotonic()-tick+source_seconds/len(items)));continue
                    sourcepos,labelpos=positions;output=np.empty((length,)+native.shape[1:],np.uint8)
                    for j,p in enumerate(sourcepos):
                        left=int(np.floor(p));right=int(np.ceil(p));frac=p-left
                        assert 0<=left<=right<len(native)
                        value=(1-frac)*native[left].astype(np.float64)+frac*native[right]
                        output[j]=np.floor(value+rng.random(value.shape)).astype(np.uint8)
                    assert labelpos.min()>=0 and labelpos.max()<=len(gt)-1
                    raw_label=np.interp(labelpos,np.arange(len(gt)),gt)
                    ctx,_=shifted_start(plan['source_start']+(length-300)*scale/2,300,len(gt),scale,tau)
                    achieved=common_hr(np.interp(ctx[1],np.arange(len(gt)),gt),'HAR') if ctx is not None else np.nan
                    cut=0;diag.update(scale=scale,source_start=plan['source_start'],source_start_moved=plan['source_start']!=int(r['start_frame']),
                        label_shift_source_frames=plan['label_shift_source_frames'],numeric_inward_ulp_steps=plan['numeric_inward_ulp_steps'],reflection_used=False)
                else:
                    g0,goal=ppg_vectors(gt,start,h0,target,tau)
                    pulse,w,beta=calibration(physview,crop_native(notched,boxes[0]),crop_native(face,boxes[0]),g0,settings['direction_RGB'])
                    output,ed=edited_with_fixed_A(clip,notched,face,roi,g0,goal,settings['direction_RGB'],h0,'ED2W-S',pulse,w,
                        r['chunk_id'],arm,rounding_rng=rng,fixed_A=settings['A_fix'] if arm=='ii-f' else None,
                        inject=injection_waveform(gt,start,h0,target,tau,settings.get('injection','original')))
                    if arm=='ii-f':assert ed['injection_A']==settings['A_fix']
                    raw_label=raw_target_label(gt,start,h0,target);achieved=common_hr(raw_label,'HAR');cut=offset
                    diag.update(scale=target/h0,source_start=int(r['start_frame']),source_start_moved=False,label_shift_source_frames=0,
                        numeric_inward_ulp_steps=0,reflection_used=reflection_used(gt,start,h0,target),mask_failure_fraction=failure,
                        injection_A=ed['injection_A'],beta=beta,w=w,n_clipped_channel_values=ed['n_clipped_channel_values'],n_clipped_pixel_times=ed['n_clipped_pixel_times'])
                view,_=rhythm.view(output,boxes[0]) if rhythm else phys.view(output,'PhysNet',reuse_boxes=boxes)
                if model=='PhysNet':x=diff_normalize_data(view);y=diff_normalize_label(raw_label)
                elif model=='FPm':x=view;y=standardized_label(raw_label)
                else:x=standardized_data(view);y=standardized_label(raw_label)
                x=np.ascontiguousarray(x[cut:cut+length],dtype=np.float32);y=np.ascontiguousarray(y[cut:cut+length],dtype=np.float32)
                edge=128 if model=='RhythmFormer' else 72
                assert x.shape==(length,edge,edge,3) and y.shape==(length,) and np.isfinite(x).all() and np.isfinite(y).all()
                capacity(add_cache=x.nbytes+y.nbytes+256)
                folder=Path(PATHS['cache'])/'st30g/pools'/model/arm/rid;folder.mkdir(parents=True,exist_ok=True)
                xp=folder/('c'+str(c)+'_k'+str(k)+'_input.npy');yp=folder/('c'+str(c)+'_k'+str(k)+'_label.npy')
                assert not xp.exists() and not yp.exists();np.save(xp,x);np.save(yp,y)
                result=dict(r,**diag,generation_excluded=False,exclusion_reason='',tau_rec=tau,achieved_context_HR=achieved,
                    target_difference=achieved-target,input_path=str(xp),label_path=str(yp),input_bytes=xp.stat().st_size,label_bytes=yp.stat().st_size,
                    input_sha256=sha(xp),label_sha256=sha(yp),stored_format='THWC_float32',temporary_under_PI_v1_3=True,
                    source_setup_seconds_allocated=source_seconds/len(items),context_setup_seconds_allocated=context_seconds/len(current),
                    CPU_seconds=time.monotonic()-tick+source_seconds/len(items)+context_seconds/len(current))
                outputs.append(result)
                for p,key in [(xp,'input'),(yp,'label')]:checks.append(dict(path=str(p),sha256=result[key+'_sha256'],**snapshot(p)))
                del output,view,x,y,raw_label
            if needs_edit:del face,roi,notched,physview
            write_rows(manifest,outputs)
            write_rows('reports/st30g/A_jobs/'+job['id']+'_guards.csv',checks)
            save('reports/st30g/A_jobs/'+job['id']+'_progress.json',dict(status='running',processed=len(outputs),total=len(items),updated_utc=now()))
        intact()
    except Exception as exc:
        error=dict(error_type=type(exc).__name__,reason=str(exc),at_utc=now(),traceback=traceback.format_exc())
        save('reports/st30g/A_jobs/'+job['id']+'_error.json',error)
    finally:
        phys.close()
        if rhythm:rhythm.close()
    if outputs:write_rows(manifest,outputs)
    if checks:write_rows('reports/st30g/A_jobs/'+job['id']+'_guards.csv',checks)
    result=dict(status='complete' if error is None and len(outputs)==len(items) else 'execution_inability',
        job=job['id'],model=model,recording_id=rid,planned=len(items),processed=len(outputs),
        generated=sum(not r['generation_excluded'] for r in outputs),excluded=sum(r['generation_excluded'] for r in outputs),
        manifest=manifest,wall_seconds=time.monotonic()-began,source_metadata_before_after_matches=error is None,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,error=error)
    save(outcome,result)
    phase_record=json.loads(phase_path.read_text());save(phase_path,dict(phase_record,status=result['status'],wall_seconds=result['wall_seconds'],finished_utc=now()))
    return result


def generate_job(job):
    began=time.monotonic();phase_path=ROOT/'reports/st30g/A_jobs'/(job['id']+'_phase.json')
    assert not phase_path.exists(),'Never repeat an attempted recording job'
    initial=dict(status='running',resource='ST6_CPU_worker',started_utc=now(),started_epoch=time.time(),**identity())
    save(phase_path,initial)
    try:result=_generate_job(job)
    except Exception as exc:
        result=dict(status='execution_inability',job=job['id'],model=job['model'],recording_id=job['recording_id'],
            error_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),planned=len(job['items']))
    result['full_worker_job_wall_seconds']=time.monotonic()-began
    save('reports/st30g/A_jobs/'+job['id']+'.json',result)
    save(phase_path,dict(initial,status=result['status'],wall_seconds=result['full_worker_job_wall_seconds'],finished_utc=now()))
    return result
