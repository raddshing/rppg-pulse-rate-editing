"""C4 fixed defect map, r0 reproduction gate, restoration and nulls."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.st3.base import *
from src.st3.media import gt_aligned
from src.st3.waves import prediction,timeline,hr,label_preprocess
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def inputs():
    for p in ('outputs/st2_window_categories.csv','outputs/st2_window_run_table.parquet','outputs/st3_rw_configs.csv'):
        if not p.startswith('outputs/st3_'):source_hash(p)
    cats=pd.read_csv(ROOT/'outputs/st2_window_categories.csv')
    p=cats.loc[cats.in_P].sort_values('window_id').reset_index(drop=True)
    check('P1100',len(p),1100)
    check('P T1',int((p.T1==1).sum()),315)
    check('P T1 unknown',int((p.T1==-1).sum()),1)
    check('P T1b',int((p.T1b==1).sum()),290)
    check('P T1 union T1b',int((p.T1_union_T1b==1).sum()),522)
    source_hash('data/index/mmpd_index_st1.csv')
    ix={r['recording_id']:r for r in rows('data/index/mmpd_index_st1.csv') if r['subject_id'] in SUBJECTS}
    gt={rid:gt_aligned(r,'MMPD') for rid,r in ix.items()}
    table=pd.read_parquet(ROOT/'outputs/st2_window_run_table.parquet')
    table=table[table.run_id.isin(RW15)&table.in_P]
    configs={r['run_id']:r for r in rows('outputs/st3_rw_configs.csv')}
    emit('S3-1',check='fixed_population_contrasts',check_status='passed')
    return p,gt,table,configs


def defect_map(p,gt):
    defects=[];segments=[]
    for i,r in enumerate(p.itertuples(),1):
        raw=gt[r.recording_id];j=int(r.window_index);begin=j*300;w=raw[begin:begin+300]
        blocks=np.array([np.percentile(raw[k:k+30],95)-np.percentile(raw[k:k+30],5) for k in range(0,len(raw)-29,30)])
        arec=float(np.median(blocks));length=0
        for k in range(10):
            if np.percentile(w[k*30:(k+1)*30],95)-np.percentile(w[k*30:(k+1)*30],5)<.1*arec:length+=1
            else:break
        d1=length>0;d3=j==0 and blocks[0]>3*arec
        if d1:segments.append(dict(window_id=r.window_id,type='d1',start_frame=begin,end_frame=begin+length*30))
        if d3:segments.append(dict(window_id=r.window_id,type='d3',start_frame=0,end_frame=30))
        q05,q25,q95=np.percentile(w,[5,25,95]);low=w<(q05-.5*(q95-q05))
        edges=np.diff(np.r_[False,low,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        count=0
        for a,b in zip(starts,ends):
            if b-a>30:continue
            recover=np.flatnonzero(w[b:min(300,b+31)]>q25)
            if len(recover):
                e=b+int(recover[0]);count+=1
                segments.append(dict(window_id=r.window_id,type='d2',start_frame=begin+int(a),end_frame=begin+e))
        defects.append(dict(window_id=r.window_id,d1=int(d1),d1_len_s=length,d2=int(count>0),d2_count=count,d3=int(d3),d4=int(r.T1b_high_freq),defect_any=int(d1 or d3 or count),T1=int(r.T1),T1b=int(r.T1b)))
        if i%25==0:update('S3-1',done=i,total=1100,label='GT defect map')
    d=pd.DataFrame(defects).set_index('window_id').reindex(p.window_id)
    write_rows('outputs/st3_gt_defect_map.csv',d.reset_index().to_dict('records'))
    write_rows('outputs/st3_gt_defect_segments.csv',segments,fields=['window_id','type','start_frame','end_frame'])
    return d,segments


def r0_gate(p,gt,table,configs):
    controls=[];pred=np.empty((15,2,1100));truth=np.empty_like(pred)
    for a,run in enumerate(RW15):
        cfg=configs[run];diff=cfg['label_type']=='DiffNormalized'
        processed={rid:label_preprocess(w,cfg['label_type']) for rid,w in gt.items()}
        for b,layer in enumerate(BANDS):
            ref=table[(table.run_id==run)&(table.layer==layer)].set_index('window_id').reindex(p.window_id)
            pred[a,b]=ref.pred_hr.to_numpy()
            values=np.array([hr(processed[r.recording_id][int(r.window_index)*300:(int(r.window_index)+1)*300],diff,layer) for r in p.itertuples()])
            delta=np.max(np.abs(values-ref.gt_hr.to_numpy()))
            controls.append(dict(run_id=run,layer=layer,max_absolute_delta=float(delta),n_cells=1100,status='passed' if delta<=1e-6 else 'failed'))
            truth[a,b]=values
            write_rows('outputs/st3_r0_controls.csv',controls)
            check('r0 prime reproduces every stored GT HR cell',float(delta),0.,1e-6,scope='S3-1')
        update('S3-1',done=a+1,total=15,label='r0 full GT-path reproduction')
    emit('S3-1',check='r0_all_cells',check_status='passed')
    return pred,truth


def interpolate(raw,mask):
    valid=~mask
    check('At least one known GT sample',valid.any(),scope='S3-1')
    result=raw.copy()
    result[mask]=np.interp(np.flatnonzero(mask),np.flatnonzero(valid),raw[valid])
    return result


def segment_masks(p,segments,gt):
    masks={rid:np.zeros(len(w),dtype=bool) for rid,w in gt.items()}
    wid_to_rid=p.set_index('window_id').recording_id.to_dict()
    for r in segments:masks[wid_to_rid[r['window_id']]][r['start_frame']:r['end_frame']]=True
    return masks


def eval_truth(p,gt,configs):
    result=np.empty((15,2,len(p)))
    # Only three fixed preprocessing label types; reuse their HR values exactly.
    by_type={}
    for kind in set(c['label_type'] for c in configs.values()):
        processed={rid:label_preprocess(w,kind) for rid,w in gt.items()}
        by_type[kind]=np.array([[hr(processed[r.recording_id][int(r.window_index)*300:(int(r.window_index)+1)*300],kind=='DiffNormalized',layer) for r in p.itertuples()] for layer in BANDS])
    for a,run in enumerate(RW15):result[a]=by_type[configs[run]['label_type']]
    return result


def validation_cases(p,d,gt):
    clean=p.loc[(d.defect_any.to_numpy()==0)&(d.T1.to_numpy()==0)&(d.T1b.to_numpy()==0)].sort_values('window_id')
    rng=np.random.default_rng(20261017)
    check('200 clean validation windows available',len(clean)>=200,scope='S3-1')
    selected=clean.iloc[rng.choice(len(clean),200,replace=False)]
    cases=[]
    for i,r in enumerate(selected.itertuples()):
        if i<100:a=0;b=[30,60,90,150][i//25]
        else:
            length=float(rng.uniform(.2,1.));start=float(rng.uniform(0,10-length));a=round(30*start);b=a+round(30*length)
        mask=np.zeros(300,dtype=bool);mask[a:b]=True
        raw=gt[r.recording_id][int(r.window_index)*300:(int(r.window_index)+1)*300].copy()
        cases.append((r.window_id,raw,mask,'validation_'+str(i)))
    return clean,cases


def main():
    p,gt,table,configs=inputs()
    d,segments=defect_map(p,gt)
    pred,truth0=r0_gate(p,gt,table,configs)
    masks=segment_masks(p,segments,gt)
    restored2={rid:interpolate(w,masks[rid]) for rid,w in gt.items()}
    truth2=eval_truth(p,restored2,configs)
    clean,cases=validation_cases(p,d,gt)
    # Model construction/training and restoration live in the fixed r3 helper.
    from src.st3.r3 import train,restore,used_seconds,close_usage
    model=train()
    update('S3-1',done=0,total=200,label='Restoration validation')
    restored3_cases=restore(model,cases)
    validation=[]
    for case,y3 in zip(cases,restored3_cases):
        wid,raw,mask,round_id=case
        row=p.set_index('window_id').loc[wid];a=int(row.window_index)*300
        full_mask=np.zeros(len(gt[row.recording_id]),dtype=bool);full_mask[a:a+300]=mask
        y2=interpolate(gt[row.recording_id],full_mask)[a:a+300]
        scale=float(raw.std())
        for treatment,y in [('r2',y2),('r3',y3)]:
            for layer in BANDS:
                validation.append(dict(window_id=wid,treatment=treatment,layer=layer,hr_abs_error=abs(hr(y,False,layer)-hr(raw,False,layer)),masked_rmse=float(np.sqrt(np.mean((y[mask]-raw[mask])**2))/scale),mask_samples=int(mask.sum())))
    vf=pd.DataFrame(validation);write_rows('outputs/st3_restore_validation.csv',validation)
    accepted=all(vf[(vf.layer==l)&(vf.treatment=='r3')].hr_abs_error.median()<=vf[(vf.layer==l)&(vf.treatment=='r2')].hr_abs_error.median() for l in BANDS)
    if not accepted:
        del model
        close_usage()
    defect_ids=p.loc[d.defect_any.to_numpy()==1,'window_id'].tolist()
    restored3=None;truth3=None
    if accepted:
        real_cases=[]
        for r in p[p.window_id.isin(defect_ids)].itertuples():
            j=int(r.window_index);a=j*300
            real_cases.append((r.window_id,gt[r.recording_id][a:a+300],masks[r.recording_id][a:a+300], 'actual'))
        replacements=restore(model,real_cases)
        restored3={rid:w.copy() for rid,w in gt.items()}
        mapping=p.set_index('window_id')
        for case,y in zip(real_cases,replacements):
            r=mapping.loc[case[0]];a=int(r.window_index)*300;restored3[r.recording_id][a:a+300]=y
        truth3=eval_truth(p,restored3,configs)
    treatments={'r0':truth0,'r1':truth0,'r2':truth2}
    if accepted:treatments['r3']=truth3
    keep=d.defect_any.to_numpy()==0
    evalrows=[];maes={}
    for treatment,t in treatments.items():
        select=keep if treatment=='r1' else np.ones(1100,dtype=bool)
        maes[treatment]=np.mean(np.abs(pred[:,:,select]-t[:,:,select]),axis=2)
        for a,run in enumerate(RW15):
            for b,layer in enumerate(BANDS):evalrows.append(dict(run_id=run,layer=layer,treatment=treatment,MAE=float(maes[treatment][a,b]),n_windows=int(select.sum())))
    write_rows('outputs/st3_restore_eval.csv',evalrows)
    null_rows=[];rng=np.random.default_rng(20261018)
    pmap=p.set_index('window_id');k=len(defect_ids)
    check('Clean windows sufficient for null transplant',len(clean)>=k,scope='S3-1')
    # Respect the two-heavy-CPU-process ceiling while C1/C3 are active.
    from src.st3.progress import snapshot
    def cpu_slots_full():
        statuses={r['stage']:r['status'] for r in snapshot()}
        return statuses['S3-2']=='running' and statuses['S3-4']=='running'
    if cpu_slots_full():
        if accepted:
            del model
            close_usage()
        while cpu_slots_full():
            update('S3-1',status='waiting_CPU_slot',label='Waiting for C1/C3 CPU slot; GPU released')
            time.sleep(2);wall_check()
        if accepted:model=train()
    null_start=time.monotonic();gpu_before_null=used_seconds()
    for iteration in range(1000):
        remove=rng.choice(1100,k,replace=False);select=np.ones(1100,dtype=bool);select[remove]=False
        random1=np.mean(np.abs(pred[:,:,select]-truth0[:,:,select]),axis=2)
        selected=clean.iloc[rng.choice(len(clean),k,replace=False)]
        null_masks={rid:np.zeros(len(w),dtype=bool) for rid,w in gt.items()};cases_null=[]
        for defect,r in zip(defect_ids,selected.itertuples()):
            source=pmap.loc[defect];sa=int(source.window_index)*300;da=int(r.window_index)*300
            mask=masks[source.recording_id][sa:sa+300].copy();raw=gt[r.recording_id][da:da+300].copy()
            # r2 uses full recording boundaries, not the local validation interpolation.
            null_masks[r.recording_id][da:da+300]|=mask
            cases_null.append((r.window_id,raw,mask,'null_'+str(iteration)))
        candidates={rid:interpolate(w,null_masks[rid]) for rid,w in gt.items()}
        indices=p.window_id.isin(selected.window_id).to_numpy()
        random_truth2=truth0.copy();calculated2=eval_truth(p,candidates,configs)
        random_truth2[:,:,indices]=calculated2[:,:,indices]
        random2=np.mean(np.abs(pred-random_truth2),axis=2)
        random_maes={'r1':random1,'r2':random2}
        if accepted:
            values=restore(model,cases_null)
            candidate3={rid:w.copy() for rid,w in gt.items()}
            for case,y in zip(cases_null,values):
                row=pmap.loc[case[0]];a=int(row.window_index)*300;candidate3[row.recording_id][a:a+300]=y
            random_truth3=truth0.copy();calculated3=eval_truth(p,candidate3,configs)
            random_truth3[:,:,indices]=calculated3[:,:,indices]
            random_maes['r3']=np.mean(np.abs(pred-random_truth3),axis=2)
        for treatment,m in random_maes.items():
            for b,layer in enumerate(BANDS):
                tau=float(kendalltau(maes['r0'][:,b],m[:,b]).statistic)
                change=float(np.median(np.abs(m[:,b]-maes['r0'][:,b])))
                null_rows.append(dict(iteration=iteration,treatment=treatment,layer=layer,tau_b=tau,median_abs_delta_MAE=change))
        if iteration%5==0 or iteration==999:
            write_rows('outputs/st3_restore_null.csv',null_rows)
            update('S3-1',done=iteration+1,total=1000,label='Shared restoration null controls')
            wall_check()
            if accepted:
                elapsed=used_seconds()-gpu_before_null
                check('r3 projected 1000-null GPU ceiling',used_seconds()+elapsed/(iteration+1)*(999-iteration)<14400,scope='S3-1_r3')
    if accepted:
        del model
        close_usage()
    nf=pd.DataFrame(null_rows);criteria=[]
    for b,layer in enumerate(BANDS):
        count=int((np.abs(maes['r3'][:,b]-maes['r2'][:,b])>.3).sum()) if accepted else 0
        criteria.append(dict(criterion='a',layer=layer,treatment='r3_vs_r2',value=count,threshold='>=8',passed=bool(accepted and count>=8),r3_adopted=bool(accepted)))
        for treatment in treatments:
            if treatment=='r0':continue
            null=nf[(nf.treatment==treatment)&(nf.layer==layer)]
            tau=float(kendalltau(maes['r0'][:,b],maes[treatment][:,b]).statistic);change=float(np.median(np.abs(maes[treatment][:,b]-maes['r0'][:,b])))
            lo=float(np.percentile(null.tau_b,2.5));hi=float(np.percentile(null.median_abs_delta_MAE,97.5))
            criteria.append(dict(criterion='b_tau',layer=layer,treatment=treatment,value=tau,threshold='<=0.9 and <null_p2.5='+str(lo),passed=bool(tau<=.9 and tau<lo),r3_adopted=bool(accepted)))
            criteria.append(dict(criterion='b_delta',layer=layer,treatment=treatment,value=change,threshold='>1 and >null_p97.5='+str(hi),passed=bool(change>1 and change>hi),r3_adopted=bool(accepted)))
    write_rows('outputs/st3_c4_criteria.csv',criteria)
    criteria_all=[dict(criterion='a_both_layers',passed=all(r['passed'] for r in criteria if r['criterion']=='a'))]
    criteria_all.append(dict(criterion='b_both_layers',passed=any(all(r['passed'] for r in criteria if r['criterion']==branch and r['treatment']==t) for t in treatments if t!='r0' for branch in ('b_tau','b_delta'))))
    write_rows('outputs/st3_c4_criteria_combined.csv',criteria_all)
    save_json('reports/st3/st3_c4_input_hashes.json',READ_HASHES)
    from src.st3.c4_description import describe
    describe(p,d,table)
    update('S3-1',done=1,total=1,status='complete',label='C4 computations complete; findings held')


if __name__=='__main__':
    finish('S3-1',main)
