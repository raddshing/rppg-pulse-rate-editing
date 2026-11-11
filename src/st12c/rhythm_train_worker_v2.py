"""One conditional official RhythmFormer run; no validation or model selection."""
import os
os.environ['ST12_NAMESPACE']='st12c'
from src.st12.ops import *
from src.st12 import gpu
import sys
import numpy as np


def main():
    request=json.loads(Path(sys.argv[1]).read_text());run_id=request['run_id'];seed=int(request['seed']);arm=request['arm'];attempt=int(os.environ['ST12_GPU_ATTEMPT'])
    p=REPORTS/'st12c_a_training_settings.json';settings=json.loads(p.read_text());config=settings['model']
    check('Exact committed RF run',sha(p) in (ROOT/'reports/ST4_INSTR05B_PREFIX.md').read_text() and request in settings['runs'])
    for path,digest in settings['source_hashes'].items():check('Pinned RF training source',sha(ROOT/path)==digest)
    for name in ['original_manifest','pool_manifest','training_input_use_guard']:check('Exact RF training manifest',sha(ROOT/settings[name])==settings[name+'_sha256'])
    original=sorted(rows(settings['original_manifest']),key=lambda r:int(r['chunk_order']));pool=sorted(rows(settings['pool_manifest']),key=lambda r:int(r['pool_index'])) if arm=='ii' else []
    guards={r['path']:r for r in rows(settings['training_input_use_guard'])}
    for r in original+pool:
        for key in ['input','label']:
            path=Path(r[key+'_path']);actual=path.stat();g=guards[str(path)]
            check('Training input unchanged since all-file SHA check',all(int(g[k])==v for k,v in dict(bytes=actual.st_size,device=actual.st_dev,inode=actual.st_ino,mtime_ns=actual.st_mtime_ns,ctime_ns=actual.st_ctime_ns).items()))
    from src.st12c.rhythm_repair import verify_repair
    repair=verify_repair()
    from src.st12c.rhythm_backend_v2 import initialize,step,torch
    from src.st12c.rhythm_data import loader
    from src.st9s.backend import state_digest
    torch.set_num_threads(1);model,criterion,optimizer,scheduler=initialize(config,seed);digest=state_digest(model)
    key='RhythmFormer|'+str(seed)
    with (REPORTS/'.initial_weights.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=REPORTS/'st12c_initial_weights.json';known=json.loads(path.read_text()) if path.exists() else {}
        if key in known:check('Same initial RF weights across paired arms',known[key]['sha256']==digest)
        else:known[key]=dict(sha256=digest,first_run=run_id);save_json(path,known)
    total=config['epochs']*config['steps_per_epoch'];iterator=iter(loader(original,pool,seed,arm,total,settings['data_workers']))
    destination=Path(PATHS['bulk'])/'st12c/checkpoints'/run_id/('attempt'+str(attempt));destination.mkdir(parents=True,exist_ok=True)
    final=destination/'final_epoch.pth';check('Never overwrite RF checkpoint',not final.exists())
    losses=[];epochs=[];exposure=dict(original=0,augmented=0,high_original=0,high_augmented=0);started=time.monotonic();first_epoch=None;status='running';final_hash=None;done=0
    try:
        for k in range(total):
            tick=time.monotonic();x,y,kinds,indices,high=next(iterator);x=x.to('cuda:0',non_blocking=True);y=y.to('cuda:0',non_blocking=True)
            epoch=k//config['steps_per_epoch']+1;loss=step(model,criterion,optimizer,scheduler,x,y,epoch-1);torch.cuda.synchronize();value=float(loss.detach());done=k+1
            counts=dict(original=int((kinds==0).sum()),augmented=int((kinds==1).sum()),high_original=int((high*(kinds==0)).sum()),high_augmented=int((high*(kinds==1)).sum()))
            for key,value_count in counts.items():exposure[key]+=value_count
            finite=bool(np.isfinite(value));losses.append(dict(run_id=run_id,epoch=epoch,step=k+1,loss=value,finite=finite,step_wall_seconds=time.monotonic()-tick,learning_rate=optimizer.param_groups[0]['lr'],**counts))
            if not finite:
                status='stopped_nonfinite_loss';save_json(REPORTS/(run_id+'_execution_block.json'),dict(run_id=run_id,reason='Nonfinite official RF loss',step=k+1,at_utc=now()));break
            if (k+1)%config['steps_per_epoch']==0:
                checkpoint=final if epoch==config['epochs'] else destination/('epoch_'+str(epoch)+'.pth');gpu.capacity()
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,run_id=run_id,
                    model_name='RhythmFormer',arm=arm,seed=seed,initial_weights_sha256=digest,source_settings_sha256=sha(p),loss_repair_sha256=repair['receipt_sha256']),checkpoint)
                current=sha(checkpoint)
                if epoch==config['epochs']:final_hash=current
                epochs.append(dict(run_id=run_id,epoch=epoch,mean_loss=float(np.mean([r['loss'] for r in losses[-config['steps_per_epoch']:]])),
                    checkpoint_path=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=current,final_epoch=epoch==config['epochs']))
                if first_epoch is None:first_epoch=time.monotonic()
                elif epoch>=2:
                    rate=(time.monotonic()-first_epoch)/(k+1-config['steps_per_epoch']);remaining=total-k-1+sum(r['order']>request['order'] for r in settings['runs'])*total
                    used=gpu.total('C5_a');projected=used+rate*remaining;cap=gpu.caps()['C5_a']
                    forecast=dict(account='C5_a',run_id=run_id,epoch=epoch,used_seconds=used,seconds_per_step=rate,remaining_steps=remaining,
                        projected_seconds=projected,cap_seconds=cap,projection_exceeds=projected>cap,additional_seconds_with15percent_margin=max(0,1.15*projected-cap),updated_utc=now())
                    save_json(REPORTS/'st12c_a_training_projection.json',forecast)
                    if projected>cap:save_json(ROOT/'reports/st12/C5_a_projection_alert.json',forecast)
                write_rows(REPORTS/'loss'/(run_id+'_attempt'+str(attempt)+'_steps.csv'),losses);write_rows(REPORTS/'loss'/(run_id+'_attempt'+str(attempt)+'_epochs.csv'),epochs)
            if (k+1)%20==0 or k+1==total:progress(run_id,k+1,total,started)
        if status=='running':status='complete'
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        if losses:write_rows(REPORTS/'loss'/(run_id+'_attempt'+str(attempt)+'_steps.csv'),losses)
        if epochs:write_rows(REPORTS/'loss'/(run_id+'_attempt'+str(attempt)+'_epochs.csv'),epochs)
    return dict(status=status,run_id=run_id,model='RhythmFormer',arm=arm,seed=seed,completed_steps=done,completed_epochs=len(epochs),prescribed_steps=total,
        checkpoint_path=str(final) if final_hash else '',checkpoint_sha256=final_hash,checkpoint_bytes=final.stat().st_size if final_hash else 0,
        initial_weights_sha256=digest,exposure=exposure,actual_pool_size=len(pool),validation_calls=0,evaluation_calls=0,
        loss_repair_sha256=repair['receipt_sha256'],
        final_loss=float(losses[-1]['loss']) if losses and np.isfinite(losses[-1]['loss']) else None)


if __name__=='__main__':
    from src.st12.worker import run
    r=json.loads(Path(sys.argv[1]).read_text())
    run(main,REPORTS/(r['run_id']+'_attempt'+os.environ['ST12_GPU_ATTEMPT']+'_result.json'),r['run_id'])
