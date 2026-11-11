"""One recording of fixed ED2-W RhythmFormer-view pool preparation."""
import os
os.environ['ST12_NAMESPACE']='st12c'
from src.st12.compat import bind
bind()
from src.st9e.temporal import *
from src.st9e.model_view import ViewWorker,crop_native
from src.st9g.preprocess import standardized_data,standardized_label
from src.st12c.rhythm_view import RhythmViewWorker
from src.st12 import gpu
import resource


def generate_job(job):
    settings=json.loads((REPORTS/'st12c_a_settings.json').read_text());rid=job['recording_id'];items=job['items'];started=time.monotonic()
    _,pure,allowed=indices();files=frame_files(pure[rid]);gt=contact(pure[rid],'PURE',allowed)
    audit={r['path']:r for r in rows('outputs/st12_input_hash_checks.csv')}
    for p in files:
        st=p.stat();old=audit[path_key(p)]
        check('Native RF pool source unchanged',all(int(old[k])==v for k,v in dict(bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns).items()))
    tau=next(int(r['tau_rec']) for r in rows('outputs/st7b_tau_recordings.csv') if r['recording_id']==rid)
    native=read_pure(files,0,len(files));phys=ViewWorker();rhythm=RhythmViewWorker();outputs=[];error=None
    directory=Path(PATHS['bulk'])/'st12c/a/ii-h'/rid;directory.mkdir(parents=True,exist_ok=True)
    try:
        phys_all,physboxes=phys.view(native,'PhysNet');_,boxes=rhythm.view(native[:1]);source_seconds=time.monotonic()-started
        for c in sorted({int(r['chunk_order']) for r in items}):
            budget_check();gpu.capacity();current=[r for r in items if int(r['chunk_order'])==c];reference=current[0]
            context_start=int(reference['context_start']);offset=int(reference['context_offset']);h0=float(reference['reference_HR']);tick=time.monotonic()
            clip=native[context_start:context_start+300];face,roi,failure=face_masks(clip);notched=notch_all(clip,bands_for(h0))
            physview=phys_all[context_start:context_start+300];context_seconds=time.monotonic()-tick
            for r in current:
                tick=time.monotonic();k=int(r['augmentation_index']);target=float(r['target_HR']);rng=np.random.default_rng([20261137,int(r['rounding_h'])])
                g0,goal=ppg_vectors(gt,context_start,h0,target,tau)
                pulse,w,beta=calibration(physview,crop_native(notched,physboxes[0]),crop_native(face,physboxes[0]),g0,settings['direction_RGB'])
                edited_value,diag=edited(clip,notched,face,roi,g0,goal,settings['direction_RGB'],h0,'ED2W-S',pulse,w,r['chunk_id'],'ii-h',rounding_rng=rng,inject=injection_waveform(gt,context_start,h0,target,tau,settings.get('injection','original')))
                raw_label=raw_target_label(gt,context_start,h0,target);view,_=rhythm.view(edited_value,boxes[0])
                data=np.ascontiguousarray(standardized_data(view)[offset:offset+160],dtype=np.float32)
                label=np.ascontiguousarray(standardized_label(raw_label)[offset:offset+160],dtype=np.float32)
                check('Exact finite RhythmFormer augmented format',data.shape==(160,128,128,3) and label.shape==(160,) and np.isfinite(data).all() and np.isfinite(label).all())
                ip=directory/('c'+str(c)+'_k'+str(k)+'_input.npy');lp=directory/('c'+str(c)+'_k'+str(k)+'_label.npy')
                check('Never overwrite RhythmFormer pool research assets',not ip.exists() and not lp.exists());np.save(ip,data);np.save(lp,label)
                outputs.append(dict(r,tau_rec=tau,input_path=str(ip),label_path=str(lp),input_bytes=ip.stat().st_size,label_bytes=lp.stat().st_size,
                    input_sha256=sha(ip),label_sha256=sha(lp),stored_format='THWC_float32',generation_excluded=False,exclusion_reason='',
                    achieved_context_HR=common_hr(raw_label,'HAR'),mask_failure_fraction=failure,injection_A=diag['injection_A'],beta=beta,w=w,
                    reflection_used=reflection_used(gt,context_start,h0,target),CPU_seconds=time.monotonic()-tick+source_seconds/len(items)+context_seconds/len(current)))
                write_rows(REPORTS/'rhythm_pool_jobs'/(rid+'.csv'),outputs);del edited_value,view,data,label,raw_label
            del clip,face,roi,notched,physview
    except Exception as exc:
        import traceback
        error=dict(error_type=type(exc).__name__,at_utc=now(),stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(exc.__traceback__)])
        if isinstance(exc,RequiredStop):error.update(check=str(exc),scope=exc.scope)
        if isinstance(exc,ResourceWait):error.update(status='resource_wait',reason=str(exc))
    finally:phys.close();rhythm.close()
    result=dict(status='complete' if error is None and len(outputs)==len(items) else 'incomplete',recording_id=rid,
        planned_items=len(items),completed_items=len(outputs),manifest_path=str(REPORTS/'rhythm_pool_jobs'/(rid+'.csv')),
        wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,error=error)
    save_json(REPORTS/'rhythm_pool_jobs'/(rid+'.json'),result);return result
