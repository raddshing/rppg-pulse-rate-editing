"""Bounded seven-worker generation, immutable research outputs and stable pool order."""
import os
os.environ['ST9_NAMESPACE']='st9g'
from src.st9.ops import *
from src.st9.gpu import owned_cpu_roots
from src.st9g.pool_worker import generate_job
import argparse
import concurrent.futures
import multiprocessing
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True,choices=['i','h']);args=parser.parse_args()
    settings_path=REPORTS/'st9g_pool_settings.json';settings=json.loads(settings_path.read_text())
    prefix=(ROOT/'reports/ST4_INSTR03_PREFIX.md').read_text()
    check('Pool settings committed',sha(settings_path) in prefix)
    for p,digest in settings['source_hashes'].items():check('Frozen pool dependency unchanged',sha(ROOT/p)==digest)
    receipt_path=REPORTS/'st9g_targets_receipt.json';receipt=json.loads(receipt_path.read_text())
    check('Actual target table committed before generation',sha(receipt_path) in prefix and sha(receipt['targets_path'])==receipt['targets_sha256'])
    check('B-G-L calibration committed before generation',sha(REPORTS/'st9g_preflight.json') in prefix)
    check('S7 timings finished before B-G',json.loads((ROOT/'reports/st9/gpu_accounts/st9s_S7_1c.json').read_text())['status']=='closed')
    arms=['i'] if args.group=='i' else ['ii-h','iv-h']
    if args.group=='h':
        a=json.loads((ROOT/'reports/st9e/st9e_component_summary.json').read_text())['E3R_a_passed']
        b=json.loads((ROOT/'reports/st9e/st9e_brange_selection.json').read_text())['gate_passed']
        if not(a and b):
            result=dict(scope='ii-h_and_iv-h',status='not_started_prescribed_E3_stop',E3R_a=a,E3R_b=b,generated_items=0)
            save_json(REPORTS/'st9g_h_required_stop.json',result);return result
    # The bundle is one heavy phase; at most one other heavy CPU phase may run
    # while its seven bounded workers are active. Waiting launches no workers.
    while len(owned_cpu_roots())>1:
        save_json(REPORTS/('st9g_pool_'+args.group+'_waiting.json'),dict(status='waiting_for_at_most_one_other_heavy_CPU_phase',updated_utc=now()))
        budget_check(settings['parallel_wall_projection_seconds']);time.sleep(10)
    targets=[r for r in rows(receipt['targets_path']) if r['arm'] in arms]
    jobs=[]
    for model in ['PhysNet','FactorizePhys']:
        for arm in arms:
            group=[r for r in targets if r['model']==model and r['arm']==arm]
            for rid in sorted({r['recording_id'] for r in group}):
                job=dict(id=model+'_'+arm+'_'+rid,model=model,arm=arm,recording_id=rid,items=[r for r in group if r['recording_id']==rid])
                check('Never repeat a generated pool job',not (REPORTS/'pool_jobs'/(job['id']+'.json')).exists())
                jobs.append(job)
    (REPORTS/'pool_jobs').mkdir(exist_ok=True)
    started=time.monotonic();results=[];context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings['workers'],mp_context=context) as executor:
        futures={executor.submit(generate_job,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            save_json(REPORTS/('st9g_pool_'+args.group+'_jobs.json'),dict(jobs=results))
            progress('pool_'+args.group,len(results),len(jobs),started)
            if len(results)>=settings['workers']:
                projected=(time.monotonic()-started)/len(results)*(len(jobs)-len(results))
                budget_check(projected+7200)
    records=[]
    for r in results:
        path=Path(r.get('manifest_path',''))
        if path.is_file():records.extend(rows(path))
    summary={};valid=[];excluded=[]
    for model in ['PhysNet','FactorizePhys']:
        for arm in arms:
            planned=[r for r in targets if r['model']==model and r['arm']==arm]
            if not planned:continue
            group=[r for r in records if r['model']==model and r['arm']==arm]
            good=sorted([r for r in group if str(r['generation_excluded']).lower()=='false'],key=lambda r:(int(r['chunk_order']),int(r['augmentation_index'])))
            bad=[r for r in group if str(r['generation_excluded']).lower()=='true'];excluded.extend(bad)
            for i,r in enumerate(good):r['pool_index']=i
            path=ROOT/('outputs/st9g_pool_'+model+'_'+arm+'_manifest.csv')
            check('Never overwrite a finalized pool manifest',not path.exists())
            if good:write_rows(path,good,sorted(set().union(*(r.keys() for r in good))))
            valid.extend(good)
            stopped=(REPORTS/('st9g_stop_'+model+'_'+arm+'.json')).exists()
            scales=np.array([float(r['scale']) for r in good])
            summary[model+'|'+arm]=dict(model=model,arm=arm,planned=len(planned),processed=len(group),generated=len(good),excluded=len(bad),
                complete=not stopped and len(group)==len(planned),scope_stopped=stopped,
                source_start_moved=sum(str(r.get('source_start_moved')).lower()=='true' for r in good),
                reflected_contexts=sum(str(r.get('reflection_used')).lower()=='true' for r in good),
                redraws=sum(int(r['redraw_count']) for r in group),scale_median=float(np.median(scales)) if len(scales) else None,
                scale_over1p4_fraction=float(np.mean(scales>1.4)) if len(scales) else None,scale_over2_fraction=float(np.mean(scales>2)) if len(scales) else None,
                scale_max=float(scales.max()) if len(scales) else None,
                bytes=sum(int(r['input_bytes'])+int(r['label_bytes']) for r in good),
                manifest_path=str(path),manifest_sha256=sha(path) if path.exists() else None,
                mean_allocated_CPU_seconds=float(np.mean([float(r['CPU_seconds']) for r in good])) if good else None,
                all_research_tensors_retained=True)
            save_json(REPORTS/('st9g_'+model+'_'+arm+'_generation.json'),summary[model+'|'+arm])
    for name,records_out in [('manifest',valid),('exclusions',excluded)]:
        path=ROOT/('outputs/st9g_'+name+'.csv')
        previous=rows(path) if path.exists() else []
        merged=previous+records_out
        if merged:write_rows(path,merged,sorted(set().union(*(r.keys() for r in merged))))
    previous=json.loads((REPORTS/'st9g_summary.json').read_text()) if (REPORTS/'st9g_summary.json').exists() else dict(pools={})
    previous['pools'].update(summary);previous.update(K=settings['K'],workers=settings['workers'],updated_utc=now(),
        generation_only_no_training_release=True)
    save_json(REPORTS/'st9g_summary.json',previous)
    return dict(group=args.group,jobs=len(jobs),generated_items=len(valid),excluded_items=len(excluded),
        pools_complete=all(r['complete'] for r in summary.values()),max_worker_RSS_bytes=max(r.get('peak_RSS_bytes',0) for r in results))


if __name__=='__main__':
    import sys
    group=sys.argv[sys.argv.index('--group')+1]
    finish('pool_'+group,main)
