"""Bounded, fixed conditional ED2-W pool; B-G-L results are descriptive."""
import os
os.environ['ST12_NAMESPACE']='st12c'
from src.st12.ops import *
from src.st12c.rhythm_pool_worker import generate_job
import concurrent.futures
import multiprocessing


def main():
    p=REPORTS/'st12c_a_settings.json';settings=json.loads(p.read_text())
    check('Rhythm pool definition committed',sha(p) in (ROOT/'reports/ST4_INSTR05B_PREFIX.md').read_text())
    for name,digest in settings['source_hashes'].items():check('Frozen pool implementation',sha(ROOT/name)==digest)
    check('Exact fixed target plan',sha(ROOT/settings['targets_path'])==settings['targets_sha256'])
    preflight=REPORTS/'st12c_a_preflight.json'
    check('Original preflight recorded before augmented pool',preflight.exists() and sha(preflight) in (ROOT/'reports/ST4_INSTR05B_PREFIX.md').read_text())
    targets=rows(settings['targets_path']);jobs=[]
    for rid in settings['model']['training_recordings']:
        check('No implicit repeat of a pool recording job',not (REPORTS/'rhythm_pool_jobs'/(rid+'.json')).exists())
        jobs.append(dict(recording_id=rid,items=[r for r in targets if r['recording_id']==rid]))
    started=time.monotonic();results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=settings['pool_workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(generate_job,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());save_json(REPORTS/'st12c_a_pool_jobs.json',dict(jobs=results))
            progress('rhythm_pool',len(results),len(jobs),started)
            if len(results)>=settings['pool_workers']:budget_check((time.monotonic()-started)/len(results)*(len(jobs)-len(results))+3600)
    complete=all(r['status']=='complete' for r in results);records=[]
    for r in results:
        path=Path(r['manifest_path'])
        if path.exists():records.extend(rows(path))
    records.sort(key=lambda r:(int(r['chunk_order']),int(r['augmentation_index'])))
    for i,r in enumerate(records):r['pool_index']=i
    if records:write_rows('outputs/st12c_a_pool_manifest.csv',records)
    summary=dict(status='complete' if complete and len(records)==len(targets) else 'incomplete',jobs=len(results),planned_items=len(targets),completed_items=len(records),
        bytes=sum(int(r['input_bytes'])+int(r['label_bytes']) for r in records),manifest_sha256=sha(ROOT/'outputs/st12c_a_pool_manifest.csv') if records else None,
        B_G_L_as_gate=False,all_research_assets_retained=True,workers=settings['pool_workers'],
        heavy_CPU_scheduling='One bounded four-worker pool phase; at most one other heavy CPU phase while active')
    save_json(REPORTS/'st12c_a_pool_summary.json',summary)
    if any((r.get('error') or {}).get('status')=='resource_wait' for r in results):raise ResourceWait('RhythmFormer pool waiting with completed items retained')
    check('All planned pool items complete before training',summary['status']=='complete',scope='C5_a')
    return summary


if __name__=='__main__':finish('rhythm_pool',main)
