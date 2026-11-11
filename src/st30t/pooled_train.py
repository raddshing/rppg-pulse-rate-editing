"""New prescribed pooled/UBFC runs; immutable old workers and scientific sources."""
from src.st30p.ops import *
configure('st30t')
import numpy as np
from collections import Counter
from src.st30t.train import delete_new_checkpoint


def main(request):
    settings=json.loads((ROOT/request['settings']).read_text());run=request['run'];rid=run['run_id'];name=run['model']
    assert run in settings['runs'] and run in rows('reports/st30p/run_plan_v1_3.csv')
    assert sha(request['settings']) in (ROOT/'reports/ST6_INSTR01_PREFIX.md').read_text()
    for path,digest in settings['source_hashes'].items():assert sha(path)==digest,path
    group=settings['inputs'][run['population']+'|'+run['arm']]
    original=rows(group['original_manifest']);pool=rows(group['pool_manifest']) if group['pool_manifest'] else []
    original=sorted(original,key=lambda r:int(r['training_index']))
    pool=sorted(pool,key=lambda r:int(r['training_index']))
    assert [int(r['training_index']) for r in original]==list(range(len(original)))
    assert [int(r['training_index']) for r in pool]==list(range(len(pool)))
    assert len(original)==group['originals'] and len(pool)==group['pool_items']
    for c6path in group['C6_receipts']:
        c6=json.loads((ROOT/c6path).read_text());assert c6['status']=='complete' and c6['C6_complete']
        if c6['pool_id'] in group['BGL_required']:assert c6['BGL_complete']
    guard={r['path']:r for r in rows(settings['input_guards'])}
    def intact():
        for r in original+pool:
            for kind in ['input','label']:
                p=Path(r[kind+'_path']);s=p.stat();g=guard[str(p)]
                assert g['sha256']==r[kind+'_sha256']
                assert all(int(g[k])==v for k,v in dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items()),str(p)
    intact()
    from src.st30t.pooled_backend import initialize,step
    from src.st9s.backend import state_digest,torch
    if name=='RhythmFormer':from src.st12c.rhythm_data import loader
    else:from src.st9t.data import loader
    from src.st11x.future import frozen_reference_functions
    common_hr=frozen_reference_functions()['common_hr']
    config=settings['configs'][run['population']];seed=int(run['seed']);total=int(run['steps']);epoch_steps=config['steps_per_epoch']
    assert total==config['total_steps']
    assert (name,total) in [('PhysNet',5580),('RhythmFormer',4470),('FPm',1490)]
    torch.set_num_threads(1);fixed=json.loads((ROOT/'reports/st30p/environment_lock.json').read_text())
    assert torch.__version__==fixed['torch'] and torch.version.cuda==fixed['cuda_build'] and torch.backends.cudnn.version()==fixed['cudnn_version']
    assert torch.backends.cuda.matmul.allow_tf32==fixed['matmul_allow_tf32'] and torch.backends.cudnn.allow_tf32==fixed['cudnn_allow_tf32']
    capacity(add_bulk=settings['checkpoint_reservation_bytes'],add_cache=settings['checkpoint_reservation_bytes'])
    model,criterion,optimizer,scheduler=initialize(name,config,seed)
    initial=state_digest(model)
    initial_path='reports/st12c/st12c_initial_weights.json' if name=='RhythmFormer' else 'reports/st9t/st9t_initial_weights.json'
    expected=json.loads((ROOT/initial_path).read_text())[('FactorizePhys' if name=='FPm' else name)+'|'+str(seed)]['sha256']
    assert initial==expected and scheduler.total_steps==total
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    iterator=iter(loader(original,pool,seed,run['arm'],total,settings['data_workers']))
    dest=Path(PATHS['bulk'])/'st30t/checkpoints'/rid;dest.mkdir(parents=True,exist_ok=False)
    temporary=Path(PATHS['cache'])/'st30t/checkpoint_tmp'/rid;temporary.mkdir(parents=True,exist_ok=False)
    final=dest/'final_epoch.pth';lossrows=[];epochs=[];exposures=[];hist=Counter();manifest_hist=Counter();counts=Counter();labels={}
    lossvalue=None;finalsha='';status='running';started=time.monotonic();completed=0
    torch.cuda.reset_peak_memory_stats()
    try:
        for k in range(total):
            tick=time.monotonic();x,y,kinds,indices,high=next(iterator);batch_manifest=[];batch_stored=[]
            for j,(kind,number) in enumerate(zip(kinds,indices)):
                kind=int(kind);number=int(number);row=(original if kind==0 else pool)[number];key=(kind,number)
                if key not in labels:
                    signal=y[j].numpy().astype(np.float64)
                    labels[key]=float(common_hr(np.cumsum(signal) if name=='PhysNet' else signal,'HAR'))
                stored=labels[key];achieved=float(row['reference_HR'] if kind==0 else row['achieved_context_HR'])
                assert np.isfinite(stored) and np.isfinite(achieved)
                hist[stored]+=1;manifest_hist[achieved]+=1;counts[key]+=1;batch_stored.append(stored);batch_manifest.append(achieved)
                exposures.append(dict(run_id=rid,step=k+1,position=j,kind=kind,training_index=number,
                    chunk_id=row['chunk_id'],source_chunk_order=row['chunk_order'],source_pool_index=row.get('pool_index',''),
                    manifest_achieved_label_HR=achieved,stored_label_HR=stored,manifest_high=achieved>=100,stored_high=stored>=100))
            assert sum(int(t)==0 for t in kinds)==(4 if run['arm']=='0' else 2)
            x=x.to('cuda:0',non_blocking=True);y=y.to('cuda:0',non_blocking=True);epoch=k//epoch_steps+1
            loss=step(name,model,criterion,optimizer,scheduler,x,y,epoch-1);torch.cuda.synchronize()
            lossvalue=float(loss.detach());completed=k+1
            lossrows.append(dict(run_id=rid,epoch=epoch,step=k+1,loss=lossvalue,finite=bool(np.isfinite(lossvalue)),
                step_wall_seconds=time.monotonic()-tick,learning_rate=optimizer.param_groups[0]['lr'],
                original_exposures=int((kinds==0).sum()),pool_exposures=int((kinds==1).sum()),
                manifest_high=sum(h>=100 for h in batch_manifest),stored_high=sum(h>=100 for h in batch_stored)))
            if not np.isfinite(lossvalue):status='stopped_nonfinite_loss';break
            if (k+1)%epoch_steps==0 or k+1==total:
                isfinal=k+1==total;checkpoint=final if isfinal else temporary/('epoch_'+str(epoch)+'.pth')
                capacity(add_bulk=settings['checkpoint_reservation_bytes'] if isfinal else 0,add_cache=0 if isfinal else settings['checkpoint_reservation_bytes'])
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    epoch=epoch,step=k+1,run_id=rid,model_name=name,arm=run['arm'],seed=seed,population=run['population'],
                    initial_weights_sha256=initial,source_settings_sha256=sha(request['settings'])),checkpoint)
                digest=sha(checkpoint);epochrows=[r for r in lossrows if r['epoch']==epoch]
                epochs.append(dict(run_id=rid,epoch=epoch,steps=len(epochrows),mean_loss=float(np.mean([r['loss'] for r in epochrows])),
                    checkpoint_sha256=digest,checkpoint_bytes=checkpoint.stat().st_size,final_epoch=isfinal,partial_epoch=len(epochrows)<epoch_steps))
                if isfinal:finalsha=digest
                else:delete_new_checkpoint(checkpoint,digest,rid)
                write_rows('reports/st30t/loss/'+rid+'_steps.csv',lossrows);write_rows('reports/st30t/loss/'+rid+'_epochs.csv',epochs)
            if (k+1)%20==0 or k+1==total:
                capacity();save('reports/st30t/'+rid+'_progress.json',dict(status='running',run_id=rid,completed_steps=k+1,total_steps=total,
                    elapsed_seconds=time.monotonic()-started,last_loss=lossvalue,updated_utc=now()))
        if status=='running':status='complete'
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        if lossrows:write_rows('reports/st30t/loss/'+rid+'_steps.csv',lossrows)
        if epochs:write_rows('reports/st30t/loss/'+rid+'_epochs.csv',epochs)
        if exposures:write_rows('outputs/st30t_'+rid+'_label_exposures.csv',exposures)
    intact()
    result=dict(status=status,run_id=rid,model=name,arm=run['arm'],population=run['population'],seed=seed,
        completed_steps=completed,prescribed_steps=total,completed_epochs=len(epochs),final_loss=lossvalue,
        checkpoint_path=str(final) if finalsha else '',checkpoint_sha256=finalsha,checkpoint_bytes=final.stat().st_size if finalsha else 0,
        initial_weights_sha256=initial,original_exposures=sum(n for (kind,i),n in counts.items() if kind==0),
        pool_exposures=sum(n for (kind,i),n in counts.items() if kind==1),
        label_HR_high_fraction=sum(n for h,n in manifest_hist.items() if h>=100)/(4*completed),L_max=max(manifest_hist),
        L_max_definition='Actual exposed original reference_HR and pool achieved_context_HR; same300-frame manifest convention as S6-1 old55',
        stored_label_HR_high_fraction=sum(n for h,n in hist.items() if h>=100)/(4*completed),stored_label_L_max=max(hist),
        stored_label_definition='Auxiliary common_HR on actual stored label at chunk length; PhysNet difference labels cumulatively reconstructed',
        manifest_label_distribution=[dict(HR=h,exposures=n) for h,n in sorted(manifest_hist.items())],
        stored_label_distribution=[dict(HR=h,exposures=n) for h,n in sorted(hist.items())],
        actual_originals=len(original),actual_pool_size=len(pool),distinct_originals=sum(k==0 for k,i in counts),
        distinct_pool_items=sum(k==1 for k,i in counts),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        validation_calls=0,evaluation_calls=0,worker_body_seconds=time.monotonic()-started)
    save('reports/st30t/'+rid+'_result.json',result)
    save('reports/st30t/'+rid+'_progress.json',dict(status=status,run_id=rid,completed_steps=completed,total_steps=total,updated_utc=now()))
    return result


if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    phase('st30t',request['run']['run_id'],lambda:main(request),resource='ST6_GPU')
