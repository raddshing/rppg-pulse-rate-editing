"""Fixed S5-2 tables with instruction03 two-interval and zero-cell gate."""
from src.st9e.editor import *
GROUPS={'01':'stationary','02':'talking','03':'translation','04':'translation','05':'rotation','06':'rotation'}
RANGES=[(40,70),(70,100),(100,140),(140,180)]

def summarize(readouts,baselines,stem,cell):
    frame=pd.DataFrame(readouts);baseline=pd.DataFrame(baselines)
    frame['task_group']=frame.task.map(lambda v:GROUPS[str(v).zfill(2)])
    baseline['task_group']=baseline.task.map(lambda v:GROUPS[str(v).zfill(2)])
    result=[];cells=[]
    for view in ['PhysNet','FactorizePhys']:
        for combo in ['ED2W-S']:
            for reader in ['POS','CHROM']:
                for layer in BANDS:
                    b_all=baseline[baseline.model_view.eq(view)&baseline.reader.eq(reader)&baseline.layer.eq(layer)]
                    for low,high in RANGES:
                        q=frame[frame.model_view.eq(view)&frame.combination.eq(combo)&frame.reader.eq(reader)&
                            frame.layer.eq(layer)&frame.range_low.eq(low)]
                        applicable=layer=='HAR' or len(q)>=30
                        if layer=='HAR':check('B gate90 fixed windows per interval',len(q)==90)
                        for task_group in ['all','stationary','talking','translation','rotation']:
                            z=q if task_group=='all' else q[q.task_group.eq(task_group)]
                            b=b_all if task_group=='all' else b_all[b_all.task_group.eq(task_group)]
                            rate=float(b.baseline_hit.mean());target=float(z.target_hit.mean());retained=float(z.original_hit_after.mean())
                            result.append(dict(model_view=view,combination=combo,reader=reader,layer=layer,range_low=low,range_high=high,
                                task_group=task_group,n_windows=len(z),n_baseline_unique_windows=len(b),baseline_rate=rate,
                                target_hit_rate=target,original_hit_after_rate=retained,B1_applies=bool(applicable and task_group=='all'),
                                B1=bool(target>=rate-.12) if applicable and task_group=='all' else '',
                                B2=bool(retained<=rate/2) if applicable and task_group=='all' else '',
                                cell_below_threshold=bool(target<rate-.20) if layer=='HAR' and task_group!='all' else '',
                                decision_used=view=='PhysNet' and combo=='ED2W-S',n_nonfinite_readouts=int((~np.isfinite(z.edited_HR)).sum())))
    summary=pd.DataFrame(result)
    for view in ['PhysNet','FactorizePhys']:
        for combo in ['ED2W-S']:
            for low,high in RANGES:
                for group in ['stationary','talking','translation','rotation']:
                    q=summary[summary.model_view.eq(view)&summary.combination.eq(combo)&summary.layer.eq('HAR')&
                        summary.range_low.eq(low)&summary.task_group.eq(group)]
                    check('Two readers in each B exclusion cell',len(q)==2)
                    cells.append(dict(model_view=view,combination=combo,range_low=low,range_high=high,task_group=group,
                        excluded=bool(q.cell_below_threshold.eq(True).all()),decision_used=view=='PhysNet' and combo=='ED2W-S'))
    selected=summary[summary.model_view.eq('PhysNet')&summary.task_group.eq('all')&summary.B1_applies.eq(True)]
    intervals=[]
    for low,high in RANGES:
        z=selected[selected.range_low.eq(low)]
        intervals.append(bool(len(z)>0 and z.B1.eq(True).all() and z.B2.eq(True).all()))
    excluded=[c for c in cells if c['decision_used'] and c['excluded']]
    selected_excluded=[c for c in excluded if c['range_low'] in [70,100]]
    gate=intervals[1] and intervals[2] and len(selected_excluded)==0
    range_low=range_high=None
    if intervals[2]:
        left=right=2
        while left>0 and intervals[left-1]:left-=1
        while right<3 and intervals[right+1]:right+=1
        range_low=RANGES[left][0];range_high=RANGES[right][1]
    write_rows('outputs/'+stem+'.csv',result);write_rows('outputs/'+stem+'_cells.csv',cells)
    report=dict(cell=cell,gate_passed=bool(gate),interval_passed=intervals,n_excluded_cells_all_ranges=len(excluded),
        n_excluded_cells_70_140=len(selected_excluded),descriptive_range_low=range_low,descriptive_range_high=range_high,
        actual_training_target_range_fixed=[70,140],decision_view='PhysNet',decision_combination='ED2W-S',
        unchanged_B1='target_rate>=baseline_rate-0.12',unchanged_B2='original_retained<=baseline_rate/2',
        unchanged_cell_exclusion='both HAR readers below task baseline-0.20',scientific_interpretation=False)
    save_json(REPORTS/(stem+'_selection.json'),report)
    return report
