"""One cache-bounded A group, seven workers counted as one heavy CPU task."""
from src.st30g.helpers import *
from src.st30g.pool_A_worker import generate_job
import concurrent.futures
import multiprocessing


def main(group):
    settings=verify();selection=settings['groups'][group];model=selection['model'];arms=selection['arms']
    assert json.loads((REPORTS/'synthetic_A_check_completion.json').read_text())['status']=='complete'
    needed=sum(r['maximum_bytes'] for r in settings['plans'] if r['model']==model and r['arm'] in arms)
    limits=json.loads((ROOT/'reports/st30p/disk_authority_v1_3.json').read_text())
    capacity(add_cache=needed+limits['cache_projection_reserve_bytes'])
    assert limits['cache_limit_bytes']>=30*10**9
    selected=[r for r in rows(settings['target_file']) if r['model']==model and r['arm'] in arms]
    jobs=[dict(id=group+'_'+rid,model=model,recording_id=rid,items=[r for r in selected if r['recording_id']==rid]) for rid in sorted({r['recording_id'] for r in selected})]
    for job in jobs:assert not (REPORTS/'A_jobs'/(job['id']+'.json')).exists()
    (REPORTS/'A_jobs').mkdir(exist_ok=True);started=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
        futures={executor.submit(generate_job,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result()
            except Exception as exc:r=dict(status='execution_inability',job=futures[future]['id'],reason=str(exc),traceback=traceback.format_exc())
            results.append(r)
            save('reports/st30g/'+group+'_progress.json',dict(status='running',recordings=len(results),total=len(jobs),
                generated=sum(r.get('generated',0) for r in results),elapsed_seconds=time.monotonic()-started,updated_utc=now()))
            save('reports/st30g/'+group+'_jobs.json',dict(jobs=results))
    allrows=[];guards=[]
    for job in jobs:
        p=ROOT/'reports/st30g/A_jobs'/(job['id']+'.csv')
        if p.exists():allrows.extend(rows(p))
        p=ROOT/'reports/st30g/A_jobs'/(job['id']+'_guards.csv')
        if p.exists():guards.extend(rows(p))
    complete=all(r['status']=='complete' for r in results) and len(allrows)==len(selected)
    summaries=[]
    for arm in arms:
        good=sorted([r for r in allrows if r['arm']==arm and r['generation_excluded']=='False'],key=lambda r:(int(r['chunk_order']),int(r['augmentation_index'])))
        for i,r in enumerate(good):r['pool_index']=i
        output='outputs/st30g_'+model+'_'+arm+'_manifest.csv'
        assert not (ROOT/output).exists()
        if good:write_rows(output,good)
        bad=[r for r in allrows if r['arm']==arm and r['generation_excluded']=='True']
        if bad:write_rows('outputs/st30g_'+model+'_'+arm+'_exclusions.csv',bad)
        summary=dict(model=model,arm=arm,status='complete' if complete else 'execution_inability',planned=sum(r['arm']==arm for r in selected),
            generated=len(good),excluded=len(bad),manifest=output,manifest_sha256=sha(output) if good else '',
            bytes=sum(int(r['input_bytes'])+int(r['label_bytes']) for r in good),K=4,
            reflected_contexts=sum(r['reflection_used']=='True' for r in good),
            median_allocated_CPU_seconds=float(np.median([float(r['CPU_seconds']) for r in good])) if good else None,
            BGL_complete=False,C6_complete=False,training_released=False,temporary_cache=True)
        save('reports/st30g/'+model+'_'+arm+'_generation.json',summary);summaries.append(summary)
    write_rows('reports/st30g/'+group+'_input_guards.csv',guards)
    return dict(status='complete' if complete else 'execution_inability',group=group,worker_jobs=len(jobs),pools=summaries,
        source_metadata_before_after_matches=complete,new_model_calls=0,pixels_exported=0,geometry_exported=0)


if __name__=='__main__':
    group=sys.argv[1];assert group in ['PN_A','FPm_A','RF_A']
    phase('st30g','pool_'+group,lambda:main(group),heavy=True)
