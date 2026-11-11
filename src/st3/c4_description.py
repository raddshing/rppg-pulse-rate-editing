"""Requested descriptive cross-tabs and RG1 unit gains; no interpretation."""
from src.st3.base import *
from src.st3.waves import prediction,timeline,hr
import numpy as np
import pandas as pd
from src.eval import grid


def describe(p,d,unused_table):
    cross=[]
    for name in ('T1','T1b'):
        for state in (-1,0,1):
            selected=d[d[name]==state]
            cross.append(dict(source=name,group=str(state),n_windows=len(selected),n_defect_windows=int(selected.defect_any.sum())))
    for rel in ('blind/st2_blind_tags.csv','outputs/st2_private/st2_blind_key.csv'):
        source_hash(rel)
    tags=pd.read_csv(ROOT/'blind/st2_blind_tags.csv',keep_default_na=False)
    key=pd.read_csv(ROOT/'outputs/st2_private/st2_blind_key.csv')
    tags=tags.merge(key[['item_id','window_id']],on='item_id',validate='one_to_one')
    flags=pd.to_numeric(tags.gt_ppg_abnormal)
    check('Stage2 abnormal GT tag contrast',int((flags==1).sum()),11,scope='S3-1')
    for state in (0,1):
        chosen=tags[flags==state]
        count=sum(int(d.loc[w,'defect_any']) for w in chosen.window_id if w in d.index)
        cross.append(dict(source='Stage2_GT_tag',group=str(state),n_windows=len(chosen),n_defect_windows=count))
    # Verbatim memo groups avoid inventing scientific classifications.
    for memo,chosen in tags[flags==1].groupby('memo',sort=True):
        count=sum(int(d.loc[w,'defect_any']) for w in chosen.window_id if w in d.index)
        cross.append(dict(source='Stage2_GT_tag_memo_verbatim',group=memo,n_windows=len(chosen),n_defect_windows=count))
    write_rows('outputs/st3_gt_defect_crosstab.csv',cross)
    source_hash('outputs/st2_window_categories.csv')
    all_windows=pd.read_csv(ROOT/'outputs/st2_window_categories.csv')
    bad_recordings=set(p.loc[d.defect_any.to_numpy()==1,'recording_id'])
    attrs=all_windows.drop_duplicates('recording_id').set_index('recording_id')
    remaining=[rid for rid in attrs.index if rid not in bad_recordings]
    motions=['Stationary','Stationary (after exercise)','Rotation','Talking','Walking']
    per_motion={m:sum(attrs.loc[rid,'motion']==m for rid in remaining) for m in motions}
    allow_recording=all(n>=5 for n in per_motion.values())
    defect_ids=set(d.index[d.defect_any==1]);unit_rows=[]
    units=['recording','30s','10s'] if allow_recording else ['30s','10s']
    for i,run in enumerate(RG_A,1):
        arm='MAUBFC' if '_maubfc_' in run else 'UBFC';seed=int(run[-1])
        for rid in attrs.index:
            if allow_recording and rid in bad_recordings:continue
            source=prediction(run,rid);y,g,covered=timeline(source,1800)
            end=int(np.flatnonzero(covered)[-1])+1
            for unit in units:
                bounds=[(0,end)] if unit=='recording' else [(a,a+(900 if unit=='30s' else 300)) for a in range(0,end-(899 if unit=='30s' else 299),900 if unit=='30s' else 300)]
                for j,(a,b) in enumerate(bounds):
                    if not covered[a:b].all():continue
                    if not allow_recording and any('MMPD|'+rid+'|'+str(k) in defect_ids for k in range(a//300,(b-1)//300+1)):continue
                    for layer,band in BANDS.items():
                        if unit=='recording':
                            py=grid._estimate(grid._prepare(y[a:b],source['diff_flag'],band,'default',30),'periodogram',band,30)
                            gg=grid._estimate(grid._prepare(g[a:b],source['diff_flag'],band,'default',30),'periodogram',band,30)
                        else:
                            # 30s uses the original WP3 nfft rule (1024), exactly Stage2.
                            py=grid._estimate(grid._prepare(y[a:b],source['diff_flag'],band,'default',30),'periodogram',band,30)
                            gg=grid._estimate(grid._prepare(g[a:b],source['diff_flag'],band,'default',30),'periodogram',band,30)
                        unit_rows.append(dict(run_id=run,arm=arm,seed=seed,recording_id=rid,motion=attrs.loc[rid,'motion'],unit=unit,layer=layer,unit_index=j,error=abs(py-gg)))
        update('S3-1',done=i,total=6,label='C4 descriptive RG1 unit gains')
    data=pd.DataFrame(unit_rows);out=[]
    for unit in units:
        for layer in BANDS:
            for motion in ['all']+motions:
                x=data[(data.unit==unit)&(data.layer==layer)]
                if motion!='all':x=x[x.motion==motion]
                gains=[]
                for seed in range(3):
                    q=x[x.seed==seed]
                    u=q[q.arm=='UBFC'].groupby('recording_id').error.mean().mean()
                    m=q[q.arm=='MAUBFC'].groupby('recording_id').error.mean().mean()
                    gains.append(float(u-m))
                out.append(dict(unit=unit,layer=layer,motion=motion,gain_s0=gains[0],gain_s1=gains[1],gain_s2=gains[2],gain_mean=float(np.mean(gains)),gain_sd=float(np.std(gains,ddof=1)),n_recordings=x.recording_id.nunique(),n_units=len(x)//6,exclusion='whole_defect_recordings' if allow_recording else 'defect_windows_or_containing_30s_unit',recording_unit_available=allow_recording))
    write_rows('outputs/st3_c4_aug_gain.csv',out)
