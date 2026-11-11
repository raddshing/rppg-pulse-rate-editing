"""Single ST6 run with full-lifetime accounting and the prescribed frozen backend."""
from src.st30p.ops import *
configure('st30t')
import numpy as np
from collections import Counter


def delete_new_checkpoint(path,digest,run_id):
    path=Path(path);root=(Path(PATHS['cache'])/'st30t/checkpoint_tmp').resolve()
    assert root in path.resolve().parents and path.is_file() and not path.is_symlink()
    assert sha(path)==digest
    item=dict(run_id=run_id,path=str(path),bytes=path.stat().st_size,sha256=digest)
    append('outputs/st30t_intermediate_deletion_journal.csv',dict(at_utc=now(),event='verified_before_delete',**item))
    path.unlink()
    fd=os.open(str(path.parent),os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    append('outputs/st30t_intermediate_deletion_journal.csv',dict(at_utc=now(),event='deleted',**item))


def main(request):
    settings=json.loads((ROOT/request['settings']).read_text());run=request['run'];run_id=run['run_id']
    assert run in settings['runs'] and run in rows('reports/st30p/run_plan.csv')
    assert sha(request['settings']) in (ROOT/'reports/ST6_INSTR01_PREFIX.md').read_text()
    for p,h in settings['source_hashes'].items():assert sha(p)==h,p
    original=sorted(rows(settings['original_manifest']),key=lambda r:int(r['chunk_order']))
    guards={r['path']:r for r in rows(settings['input_guards'])}
    for r in original:
        for kind in ['input','label']:
            p=Path(r[kind+'_path']);s=p.stat();g=guards[str(p)]
            assert all(int(g[k])==v for k,v in dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),str(p)
    assert run['model']=='RhythmFormer' and run['arm']=='0B'
    from src.st12c.rhythm_backend_v2 import initialize,step,torch
    from src.st12c.rhythm_data import loader
    from src.st9s.backend import state_digest
    from src.st30r.author_augmentation import AuthorAugmentation
    from src.st11x.future import frozen_reference_functions
    common_hr=frozen_reference_functions()['common_hr']
    config=settings['model'];seed=int(run['seed']);total=int(run['steps'])
    assert len(original)==596 and total==4470 and config['steps_per_epoch']==149 and config['epochs']==30
    torch.set_num_threads(1)
    fixed=json.loads((ROOT/'reports/st30p/environment_lock.json').read_text())
    assert torch.__version__==fixed['torch'] and torch.version.cuda==fixed['cuda_build']
    assert torch.backends.cudnn.version()==fixed['cudnn_version']
    assert torch.backends.cuda.matmul.allow_tf32==fixed['matmul_allow_tf32']
    assert torch.backends.cudnn.allow_tf32==fixed['cudnn_allow_tf32']
    capacity(add_bulk=settings['checkpoint_reservation_bytes'],add_cache=settings['checkpoint_reservation_bytes'])
    model,criterion,optimizer,scheduler=initialize(config,seed)
    initial=state_digest(model)
    expected=json.loads((ROOT/'reports/st12c/st12c_initial_weights.json').read_text())['RhythmFormer|'+str(seed)]['sha256']
    assert initial==expected, 'Paired initial weights differ'
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    iterator=iter(loader(original,[],seed,'0',total,settings['data_workers']))
    augmenter=AuthorAugmentation()
    dest=Path(PATHS['bulk'])/'st30t/checkpoints'/run_id;dest.mkdir(parents=True,exist_ok=False)
    temporary=Path(PATHS['cache'])/'st30t/checkpoint_tmp'/run_id;temporary.mkdir(parents=True,exist_ok=False)
    final=dest/'final_epoch.pth'
    lossrows=[];epochs=[];exposures=[];hist=Counter();lossvalue=None;finalsha='';status='running';started=time.monotonic()
    effective_samples=effective_batches=temporal_samples=flipped_batches=0
    original_high=0;counts=Counter();completed=0
    torch.cuda.reset_peak_memory_stats()
    try:
        for k in range(total):
            tick=time.monotonic();x,y,kinds,indices,high=next(iterator)
            numbers=[int(i) for i in indices]
            # Inspect a clone, never advance or replace either author RNG stream.
            clone=np.random.RandomState();clone.set_state(np.random.get_state())
            r1=clone.random(len(x));r2=clone.random(len(x));old_y=y.numpy().copy()
            a=[original[i]['recording_id'] for i in numbers];b=[original[i]['chunk_index'] for i in numbers]
            x,y=augmenter.data_augmentation(x,y,a,b)
            temporal=np.zeros(len(x),bool)
            for j in range(len(x)):
                if r1[j]<.5 and (augmenter.data_dict[a[j]+b[j]]>90 or augmenter.data_dict[a[j]+b[j]]<75):temporal[:]=True
                else:temporal[j]=False
            flipped=bool(r2[-1]<.5);applied=temporal|flipped
            effective_samples+=int(applied.sum());effective_batches+=int(applied.any())
            temporal_samples+=int(temporal.sum());flipped_batches+=int(flipped)
            hrs=[float(common_hr(label,'HAR')) for label in y.numpy()]
            assert np.isfinite(hrs).all()
            for j,(i,h) in enumerate(zip(numbers,hrs)):
                hist[h]+=1;counts[i]+=1
                exposures.append(dict(run_id=run_id,step=k+1,position=j,original_index=i,chunk_id=original[i]['chunk_id'],
                    label_HR=h,high=h>=100,temporal_applied=bool(temporal[j]),flipped=flipped,
                    label_changed=bool(np.any(y[j].numpy()!=old_y[j]))))
            original_high+=int(high.sum())
            x=x.to('cuda:0',non_blocking=True);y=y.to('cuda:0',non_blocking=True)
            epoch=k//config['steps_per_epoch']+1
            loss=step(model,criterion,optimizer,scheduler,x,y,epoch-1);torch.cuda.synchronize()
            lossvalue=float(loss.detach());completed=k+1
            lossrows.append(dict(run_id=run_id,epoch=epoch,step=k+1,loss=lossvalue,finite=bool(np.isfinite(lossvalue)),
                step_wall_seconds=time.monotonic()-tick,learning_rate=optimizer.param_groups[0]['lr'],
                effective_augmented_samples=int(applied.sum()),temporal_samples=int(temporal.sum()),flipped=flipped,
                high_actual=sum(h>=100 for h in hrs),high_original=int(high.sum())))
            if not np.isfinite(lossvalue):status='stopped_nonfinite_loss';break
            if (k+1)%config['steps_per_epoch']==0:
                isfinal=k+1==total;checkpoint=final if isfinal else temporary/('epoch_'+str(epoch)+'.pth')
                capacity(add_bulk=settings['checkpoint_reservation_bytes'] if isfinal else 0,
                    add_cache=0 if isfinal else settings['checkpoint_reservation_bytes'])
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=epoch,run_id=run_id,model_name=run['model'],arm=run['arm'],seed=seed,
                    initial_weights_sha256=initial,source_settings_sha256=sha(request['settings'])),checkpoint)
                digest=sha(checkpoint)
                epochs.append(dict(run_id=run_id,epoch=epoch,mean_loss=float(np.mean([r['loss'] for r in lossrows[-149:]])),
                    checkpoint_sha256=digest,checkpoint_bytes=checkpoint.stat().st_size,final_epoch=isfinal))
                if isfinal:finalsha=digest
                else:delete_new_checkpoint(checkpoint,digest,run_id)
                write_rows('reports/st30t/loss/'+run_id+'_steps.csv',lossrows)
                write_rows('reports/st30t/loss/'+run_id+'_epochs.csv',epochs)
            if (k+1)%20==0 or k+1==total:
                capacity()
                save('reports/st30t/'+run_id+'_progress.json',dict(status='running',run_id=run_id,completed_steps=k+1,
                    total_steps=total,elapsed_seconds=time.monotonic()-started,last_loss=lossvalue,updated_utc=now()))
        if status=='running':status='complete'
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        if lossrows:write_rows('reports/st30t/loss/'+run_id+'_steps.csv',lossrows)
        if epochs:write_rows('reports/st30t/loss/'+run_id+'_epochs.csv',epochs)
        if exposures:write_rows('outputs/st30t_'+run_id+'_label_exposures.csv',exposures)
    result=dict(status=status,run_id=run_id,model=run['model'],arm=run['arm'],population=run['population'],seed=seed,
        completed_steps=completed,prescribed_steps=total,completed_epochs=len(epochs),final_loss=lossvalue,
        checkpoint_path=str(final) if finalsha else '',checkpoint_sha256=finalsha,
        checkpoint_bytes=final.stat().st_size if finalsha else 0,initial_weights_sha256=initial,
        original_exposures=completed*4,pool_exposures=0,original_high_exposures=original_high,
        augmentation_applied_batch_fraction=effective_batches/completed,augmentation_applied_sample_fraction=effective_samples/(4*completed),
        temporal_applied_sample_fraction=temporal_samples/(4*completed),flipped_batch_fraction=flipped_batches/completed,
        label_HR_high_fraction=sum(n for h,n in hist.items() if h>=100)/(4*completed),L_max=max(hist),
        actual_label_distribution=[dict(HR=h,exposures=n) for h,n in sorted(hist.items())],
        distinct_originals=len(counts),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        validation_calls=0,evaluation_calls=0,worker_body_seconds=time.monotonic()-started)
    save('reports/st30t/'+run_id+'_result.json',result)
    save('reports/st30t/'+run_id+'_progress.json',dict(status=status,run_id=run_id,completed_steps=completed,total_steps=total,updated_utc=now()))
    return result


if __name__=='__main__':
    req=json.loads(Path(sys.argv[1]).read_text())
    phase('st30t',req['run']['run_id'],lambda:main(req),resource='ST6_GPU')
