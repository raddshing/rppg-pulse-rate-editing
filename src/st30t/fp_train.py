"""FP-m original-only run using the inherited FactorizePhys backend unchanged."""
from src.st30p.ops import *
configure('st30t')
import numpy as np
from collections import Counter
from src.st30t.train import delete_new_checkpoint


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
            assert all(int(g[k])==v for k,v in dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),str(p)
    assert run['model']=='FPm' and run['arm']=='0'
    from src.st9s.backend import initialize,step,state_digest,torch
    from src.st9t.data import loader
    from src.st11x.future import frozen_reference_functions
    common_hr=frozen_reference_functions()['common_hr']
    config=settings['model'];seed=int(run['seed']);total=int(run['steps'])
    assert len(original)==596 and total==1490 and config['steps_per_epoch']==149 and config['epochs']==10
    torch.set_num_threads(1)
    fixed=json.loads((ROOT/'reports/st30p/environment_lock.json').read_text())
    assert torch.__version__==fixed['torch'] and torch.version.cuda==fixed['cuda_build']
    assert torch.backends.cudnn.version()==fixed['cudnn_version']
    assert torch.backends.cuda.matmul.allow_tf32==fixed['matmul_allow_tf32']
    assert torch.backends.cudnn.allow_tf32==fixed['cudnn_allow_tf32']
    capacity(add_bulk=settings['checkpoint_reservation_bytes'],add_cache=settings['checkpoint_reservation_bytes'])
    model,criterion,optimizer,scheduler=initialize('FactorizePhys',config,seed,'cuda:0')
    initial=state_digest(model)
    expected=json.loads((ROOT/'reports/st9t/st9t_initial_weights.json').read_text())['FactorizePhys|'+str(seed)]['sha256']
    assert initial==expected,'Paired initial weights differ'
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    iterator=iter(loader(original,[],seed,'0',total,settings['data_workers']))
    dest=Path(PATHS['bulk'])/'st30t/checkpoints'/run_id;dest.mkdir(parents=True,exist_ok=False)
    temporary=Path(PATHS['cache'])/'st30t/checkpoint_tmp'/run_id;temporary.mkdir(parents=True,exist_ok=False)
    final=dest/'final_epoch.pth';lossrows=[];epochs=[];exposures=[];hist=Counter();counts=Counter()
    labels={};lossvalue=None;finalsha='';status='running';started=time.monotonic();completed=0;original_high=0
    torch.cuda.reset_peak_memory_stats()
    try:
        for k in range(total):
            tick=time.monotonic();x,y,kinds,indices,high=next(iterator);numbers=[int(i) for i in indices]
            assert all(int(t)==0 for t in kinds)
            for j,i in enumerate(numbers):
                if i not in labels:labels[i]=float(common_hr(y[j].numpy(),'HAR'))
                h=labels[i];assert np.isfinite(h);hist[h]+=1;counts[i]+=1
                exposures.append(dict(run_id=run_id,step=k+1,position=j,original_index=i,
                    chunk_id=original[i]['chunk_id'],reference_HR=original[i]['reference_HR'],label_HR=h,high=h>=100))
            original_high+=int(high.sum())
            x=x.to('cuda:0',non_blocking=True);y=y.to('cuda:0',non_blocking=True)
            epoch=k//149+1;loss=step('FactorizePhys',model,criterion,optimizer,scheduler,x,y);torch.cuda.synchronize()
            lossvalue=float(loss.detach());completed=k+1
            lossrows.append(dict(run_id=run_id,epoch=epoch,step=k+1,loss=lossvalue,finite=bool(np.isfinite(lossvalue)),
                step_wall_seconds=time.monotonic()-tick,learning_rate=optimizer.param_groups[0]['lr'],
                original_exposures=4,pool_exposures=0,high_actual=sum(labels[i]>=100 for i in numbers),high_original=int(high.sum())))
            if not np.isfinite(lossvalue):status='stopped_nonfinite_loss';break
            if (k+1)%149==0:
                isfinal=k+1==total;checkpoint=final if isfinal else temporary/('epoch_'+str(epoch)+'.pth')
                capacity(add_bulk=settings['checkpoint_reservation_bytes'] if isfinal else 0,
                    add_cache=0 if isfinal else settings['checkpoint_reservation_bytes'])
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=epoch,run_id=run_id,model_name='FactorizePhys',arm=run['arm'],seed=seed,
                    initial_weights_sha256=initial,source_settings_sha256=sha(request['settings'])),checkpoint)
                digest=sha(checkpoint)
                epochs.append(dict(run_id=run_id,epoch=epoch,mean_loss=float(np.mean([r['loss'] for r in lossrows[-149:]])),
                    checkpoint_sha256=digest,checkpoint_bytes=checkpoint.stat().st_size,final_epoch=isfinal))
                if isfinal:finalsha=digest
                else:delete_new_checkpoint(checkpoint,digest,run_id)
                write_rows('reports/st30t/loss/'+run_id+'_steps.csv',lossrows)
                write_rows('reports/st30t/loss/'+run_id+'_epochs.csv',epochs)
            if (k+1)%20==0 or k+1==total:
                capacity();save('reports/st30t/'+run_id+'_progress.json',dict(status='running',run_id=run_id,
                    completed_steps=k+1,total_steps=total,elapsed_seconds=time.monotonic()-started,last_loss=lossvalue,updated_utc=now()))
        if status=='running':status='complete'
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        if lossrows:write_rows('reports/st30t/loss/'+run_id+'_steps.csv',lossrows)
        if epochs:write_rows('reports/st30t/loss/'+run_id+'_epochs.csv',epochs)
        if exposures:write_rows('outputs/st30t_'+run_id+'_label_exposures.csv',exposures)
    result=dict(status=status,run_id=run_id,model=run['model'],arm=run['arm'],population=run['population'],seed=seed,
        completed_steps=completed,prescribed_steps=total,completed_epochs=len(epochs),final_loss=lossvalue,
        checkpoint_path=str(final) if finalsha else '',checkpoint_sha256=finalsha,checkpoint_bytes=final.stat().st_size if finalsha else 0,
        initial_weights_sha256=initial,original_exposures=completed*4,pool_exposures=0,original_high_exposures=original_high,
        label_HR_high_fraction=sum(n for h,n in hist.items() if h>=100)/(4*completed),L_max=max(hist),
        frozen_reference_HR_L_max=max(float(original[i]['reference_HR']) for i in counts),
        actual_label_distribution=[dict(HR=h,exposures=n) for h,n in sorted(hist.items())],distinct_originals=len(counts),
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),validation_calls=0,evaluation_calls=0,
        worker_body_seconds=time.monotonic()-started)
    save('reports/st30t/'+run_id+'_result.json',result)
    save('reports/st30t/'+run_id+'_progress.json',dict(status=status,run_id=run_id,completed_steps=completed,total_steps=total,updated_utc=now()))
    return result


if __name__=='__main__':
    req=json.loads(Path(sys.argv[1]).read_text())
    phase('st30t',req['run']['run_id'],lambda:main(req),resource='ST6_GPU')
