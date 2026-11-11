"""One authorized GT-only training run and deterministic RePaint restoration."""
from src.st3.base import *
from src.st3.media import gt_aligned
from src.st3.r3_model import UNet1D,schedule
import numpy as np
import torch

FINAL=ROOT/'outputs/st3_r3_final.pt'
USAGE=ROOT/'reports/st3/st3_r3_gpu_usage.json'


def config():
    return json.loads((ROOT/'reports/st3/st3_prefix_settings.json').read_text())['r3']


def used_seconds():
    if not USAGE.exists():return config()['calibration_GPU_seconds']+120
    d=json.loads(USAGE.read_text())
    return d['completed_GPU_seconds']+(time.time()-d['active_started_epoch'] if d.get('active_started_epoch') else 0)


def add_usage(seconds):
    observed=used_seconds()
    d=json.loads(USAGE.read_text())
    d['observed_GPU_seconds']=observed
    USAGE.write_text(json.dumps(d)+'\n')
    check('r3 GPU ceiling',observed<=14400,scope='S3-1_r3')


def start_usage():
    prior=used_seconds()
    USAGE.write_text(json.dumps(dict(completed_GPU_seconds=prior,active_started_epoch=time.time(),
                                    cap_GPU_seconds=14400,includes_first_calibration_attempt_reserve_seconds=120))+'\n')


def close_usage():
    observed=used_seconds()
    USAGE.write_text(json.dumps(dict(completed_GPU_seconds=observed,observed_GPU_seconds=observed,
                                    active_started_epoch=None,cap_GPU_seconds=14400,
                                    includes_first_calibration_attempt_reserve_seconds=120))+'\n')
    torch.cuda.empty_cache()


def train():
    check('Explicit r3 GPU0',os.environ.get('CUDA_VISIBLE_DEVICES'),'0')
    cfg=config();check('Frozen r3 resource rung',cfg['selected'] is not None,scope='S3-1_r3')
    if FINAL.exists():
        data=torch.load(FINAL,map_location='cpu',weights_only=False)
        check('Only completed preregistered training is resumed',data['training_complete'],True,scope='S3-1_r3')
        start_usage()
        model=UNet1D().cuda();model.load_state_dict(data['model']);return model.eval()
    pieces=[]
    for dataset in ('UBFC','PURE'):
        rel='data/index/'+dataset.lower()+'_index_st1.csv';source_hash(rel)
        index=rows(rel);check('Fixed GT-only training cohort',len(index),42 if dataset=='UBFC' else 59)
        for row in sorted(index,key=lambda r:r['recording_id']):
            raw=gt_aligned(row,dataset)
            for start in range(0,len(raw)-299,30):
                w=raw[start:start+300];std=float(w.std())
                check('Finite nonconstant GT training piece',np.isfinite(w).all() and std>0,scope='S3-1_r3')
                pieces.append(((w-w.mean())/std).astype(np.float32))
    started=time.monotonic()
    start_usage()
    torch.manual_seed(20261021)
    model=UNet1D().cuda();model.train()
    data=torch.from_numpy(np.stack(pieces)).cuda().unsqueeze(1)
    ab=schedule('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    iterations=0;last_progress=started;last_save=started
    while time.monotonic()-started<cfg['training_GPU_seconds']:
        x0=data[torch.randint(len(data),(64,),device='cuda')]
        t=torch.randint(1000,(64,),device='cuda');noise=torch.randn_like(x0)
        a=ab[t,None,None];xt=a.sqrt()*x0+(1-a).sqrt()*noise
        optimizer.zero_grad(set_to_none=True)
        loss=(model(xt,t)-noise).square().mean()
        check('Finite diffusion training loss',bool(torch.isfinite(loss)),scope='S3-1_r3')
        loss.backward();optimizer.step();iterations+=1
        now=time.monotonic()
        if now-last_progress>=2:
            update('S3-1',done=min(int(now-started),cfg['training_GPU_seconds']),total=cfg['training_GPU_seconds'],label='GT-only diffusion training')
            wall_check();check('r3 projected training ceiling',used_seconds()<14400,scope='S3-1_r3')
            last_progress=now
        if now-last_save>=60:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iterations=iterations,
                            training_GPU_seconds=now-started,training_complete=False,config=cfg),FINAL)
            last_save=now
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iterations=iterations,
                    training_GPU_seconds=elapsed,training_complete=True,config=cfg),FINAL)
    add_usage(elapsed)
    save_json('reports/st3/st3_r3_training.json',dict(status='complete',training_GPU_seconds=elapsed,
                                                   iterations=iterations,checkpoint=str(FINAL),checkpoint_sha256=sha(FINAL),
                                                   GT_sources=['UBFC42','PURE59'],MMPD_training_rows=0))
    emit('S3-1',check='GT_only_training',check_status='passed')
    return model.eval()


def restore(model,cases):
    if not cases:return []
    cfg=config();rung=cfg['selected'];started=time.monotonic()
    ab=schedule('cuda');ticks=np.linspace(0,999,rung['respacing'],dtype=int)[::-1]
    output=[];size=cfg['restore_batch_size']
    for begin in range(0,len(cases),size):
        batch=cases[begin:begin+size];normalized=[];means=[];stds=[];generators=[]
        for wid,raw,mask,round_id in batch:
            known=raw[~mask];check('Known r3 context samples',len(known)>1 and known.std()>0,scope='S3-1_r3')
            mean=float(known.mean());std=float(known.std());means.append(mean);stds.append(std)
            normalized.append(((raw-mean)/std).astype(np.float32))
            fragment=wid if round_id=='actual' else wid+'|'+round_id
            seed=int(hashlib.sha256(fragment.encode()).hexdigest()[:16],16)%(2**32)
            generators.append(torch.Generator(device='cuda').manual_seed(seed))
        known=torch.from_numpy(np.stack(normalized)).cuda().unsqueeze(1)
        mask=torch.from_numpy(np.stack([c[2] for c in batch])).cuda().unsqueeze(1)
        def noise():
            return torch.stack([torch.randn((1,300),device='cuda',generator=g) for g in generators])
        x=noise()
        initial=ab[int(ticks[0])]
        x=torch.where(mask,x,initial.sqrt()*known+(1-initial).sqrt()*noise())
        with torch.inference_mode():
            for block_start in range(0,len(ticks),rung['jump']):
                block_end=min(len(ticks),block_start+rung['jump'])
                repetitions=rung['resample'] if block_end<len(ticks) else 1
                for repeat in range(repetitions):
                    for step in range(block_start,block_end):
                        t=int(ticks[step]);next_t=int(ticks[step+1]) if step+1<len(ticks) else -1
                        at=ab[t];an=ab[next_t] if next_t>=0 else torch.ones((),device='cuda')
                        eps=model(x,torch.full((len(batch),),t,device='cuda'))
                        x0=(x-(1-at).sqrt()*eps)/at.sqrt()
                        beta=1-at/an
                        mean=an.sqrt()*beta/(1-at)*x0+(at/an).sqrt()*(1-an)/(1-at)*x
                        variance=(beta*(1-an)/(1-at)).clamp_min(0)
                        x=mean+variance.sqrt()*noise() if next_t>=0 else x0
                        constrained=an.sqrt()*known+(1-an).sqrt()*noise() if next_t>=0 else known
                        x=torch.where(mask,x,constrained)
                    if repeat+1<repetitions:
                        low=int(ticks[block_end])
                        high=int(ticks[block_start])
                        ratio=ab[high]/ab[low]
                        x=ratio.sqrt()*x+(1-ratio).sqrt()*noise()
                        x=torch.where(mask,x,ab[high].sqrt()*known+(1-ab[high]).sqrt()*noise())
            y=x[:,0].cpu().numpy()
        for i,case in enumerate(batch):
            restored=y[i].astype(np.float64)*stds[i]+means[i]
            restored[~case[2]]=case[1][~case[2]]
            check('Finite r3 reconstruction',np.isfinite(restored).all(),scope='S3-1_r3')
            output.append(restored)
        torch.cuda.synchronize()
        check('r3 live projected ceiling',used_seconds()<14400,scope='S3-1_r3')
    add_usage(time.monotonic()-started)
    return output
