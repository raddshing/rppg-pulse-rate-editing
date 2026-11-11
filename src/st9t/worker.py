"""One fixed core B run with no evaluation code path."""
import os
os.environ['ST9_NAMESPACE']='st9t'
from src.st9.ops import *
from src.st9.checkpoint_guard import require_checkpoint_capacity
import signal
import sys
import numpy as np


def main():
    launched=float(os.environ['ST9_GPU_STARTED_EPOCH']);cap=float(os.environ['ST9_GPU_ALLOWANCE'])
    request=json.loads(Path(sys.argv[1]).read_text());run_id=request['run_id'];model_name=request['model'];arm=request['arm'];seed=request['seed']
    def expire(*args):raise TimeoutError('Complete B-T child lifetime cap')
    signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,max(.1,cap-(time.time()-launched)-10))
    from src.st9s.backend import initialize,state_digest,step,torch
    from src.st9t.data import loader
    torch.set_num_threads(1)
    settings=json.loads((REPORTS/'st9t_settings.json').read_text());config=settings['models'][model_name]
    for path,digest in settings['source_hashes'].items():check('Fixed B source remains unchanged',sha(ROOT/path)==digest)
    check('Original manifest unchanged',sha(ROOT/settings['original_manifest'])==settings['original_manifest_sha256'])
    original=sorted([r for r in rows(settings['original_manifest']) if r['model']==model_name],key=lambda r:int(r['chunk_order']))
    check('All original chunks in exact frozen order',len(original)==config['chunks'] and [int(r['chunk_order']) for r in original]==list(range(config['chunks'])))
    pool=[]
    if arm not in ['0','vii']:
        check('Approved pool manifest unchanged',sha(request['pool_manifest'])==request['pool_manifest_sha256'])
        pool=sorted([r for r in rows(request['pool_manifest']) if r['model']==model_name and r['arm']==arm],key=lambda r:int(r['pool_index']))
        check('Actual augmented pool is nonempty and contiguous',len(pool)>0 and [int(r['pool_index']) for r in pool]==list(range(len(pool))))
    for r in original+pool:
        check('Exact stored input and label hashes before training',sha(r['input_path'])==r['input_sha256'] and sha(r['label_path'])==r['label_sha256'])
    model,criterion,optimizer,scheduler=initialize(model_name,config,seed,'cuda:0')
    initial=state_digest(model);initial_path=REPORTS/'st9t_initial_weights.json'
    initial_by=json.loads(initial_path.read_text()) if initial_path.exists() else {}
    key=model_name+'|'+str(seed)
    if key in initial_by:check('Same seed initial weights match across all arms',initial_by[key]['sha256']==initial)
    else:initial_by[key]=dict(sha256=initial,first_run=run_id);save_json(initial_path,initial_by)
    total=config['epochs']*config['steps_per_epoch']
    train_loader=loader(original,pool,seed,arm,total,settings['data_workers']);iterator=iter(train_loader)
    directory=CACHE/run_id;directory.mkdir(exist_ok=True)
    destination=Path(PATHS['bulk'])/'st9t'/run_id;destination.mkdir(parents=True,exist_ok=True)
    final_path=destination/'final_epoch.pth';check('Never overwrite a B research checkpoint',not final_path.exists())
    losses=[];epoch_rows=[];middle=[];exposure=dict(original=0,augmented=0,high_original=0,high_augmented=0)
    status='running';last_loss=None;completed_steps=0;final_sha='';epoch_start=time.monotonic();started=time.monotonic()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    try:
        for k in range(total):
            t=time.monotonic();x,y,kinds,numbers,high=next(iterator)
            x=x.to('cuda:0',non_blocking=True);y=y.to('cuda:0',non_blocking=True)
            loss=step(model_name,model,criterion,optimizer,scheduler,x,y);torch.cuda.synchronize()
            value=float(loss.detach());finite=bool(np.isfinite(value));epoch=k//config['steps_per_epoch']+1
            counts=dict(original=int((kinds==0).sum()),augmented=int((kinds==1).sum()),
                high_original=int((high*(kinds==0)).sum()),high_augmented=int((high*(kinds==1)).sum()))
            for name,n in counts.items():exposure[name]+=n
            losses.append(dict(run_id=run_id,epoch=epoch,step=k+1,loss=value,finite=finite,step_wall_seconds=time.monotonic()-t,
                learning_rate=optimizer.param_groups[0]['lr'],**counts))
            completed_steps=k+1;last_loss=value if finite else None
            if not finite:status='stopped_nonfinite_loss';break
            if (k+1)%config['steps_per_epoch']==0:
                epoch_loss=[r['loss'] for r in losses[-config['steps_per_epoch']:]]
                checkpoint=final_path if epoch==config['epochs'] else directory/('epoch_'+str(epoch)+'.pth')
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,
                    run_id=run_id,model_name=model_name,arm=arm,seed=seed,initial_weights_sha256=initial,
                    source_settings_sha256=sha(REPORTS/'st9t_settings.json'),original_manifest_sha256=settings['original_manifest_sha256'])
                require_checkpoint_capacity()
                torch.save(state,checkpoint);digest=sha(checkpoint);del state
                if epoch==config['epochs']:final_sha=digest
                else:middle.append(checkpoint)
                epoch_rows.append(dict(run_id=run_id,epoch=epoch,mean_loss=float(np.mean(epoch_loss)),last_loss=value,
                    epoch_wall_seconds=time.monotonic()-epoch_start,checkpoint_sha256=digest,final_epoch=epoch==config['epochs']))
                epoch_start=time.monotonic()
                write_rows(REPORTS/'loss'/(run_id+'_steps.csv'),losses);write_rows(REPORTS/'loss'/(run_id+'_epochs.csv'),epoch_rows)
            if (k+1)%20==0 or k+1==total:
                progress(run_id,k+1,total,started)
                remaining=(time.monotonic()-started)/(k+1)*(total-k-1)
                check('B run live projection within complete child allowance',time.time()-launched+1.10*remaining<cap,scope=run_id)
        if status=='running':status='complete'
    except Exception as exc:
        import traceback
        status='stopped_resource' if isinstance(exc,(RequiredStop,TimeoutError)) else 'implementation_error'
        save_json(REPORTS/(run_id+'_error.json'),dict(error_type=type(exc).__name__,stack=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(exc.__traceback__)]))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        del iterator,train_loader
        if losses:write_rows(REPORTS/'loss'/(run_id+'_steps.csv'),losses)
        if epoch_rows:write_rows(REPORTS/'loss'/(run_id+'_epochs.csv'),epoch_rows)
        for checkpoint in middle:delete_temp(checkpoint)
        result=dict(run_id=run_id,model=model_name,arm=arm,seed=seed,status=status,completed_steps=completed_steps,
            prescribed_steps=total,completed_epochs=len(epoch_rows),final_loss=last_loss,nonfinite_loss=status=='stopped_nonfinite_loss',
            initial_weights_sha256=initial,checkpoint_path=str(final_path) if final_sha else '',checkpoint_sha256=final_sha,
            checkpoint_bytes=final_path.stat().st_size if final_sha else 0,exposure=exposure,actual_pool_size=len(pool),
            CUDA_cudnn_deterministic=torch.backends.cudnn.deterministic,CUDA_cudnn_benchmark=torch.backends.cudnn.benchmark,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),validation_calls=0,evaluation_calls=0,
            worker_body_seconds=time.monotonic()-started)
        save_json(REPORTS/(run_id+'_result.json'),result)
    return 0 if status in ['complete','stopped_nonfinite_loss'] else 2


if __name__=='__main__':sys.exit(main())
