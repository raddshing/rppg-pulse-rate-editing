"""Freeze six runs and full input-use verification after original/pool completion."""
import os
os.environ['ST12_NAMESPACE']='st12c'
from src.st12.ops import *
from src.st12.supplement import register


def main():
    base=json.loads((REPORTS/'st12c_a_settings.json').read_text())
    completed=json.loads((REPORTS/'st12c_rhythm_pool_completion.json').read_text())
    check('Every fixed RF augmented item complete',completed['status']=='complete')
    original='outputs/st12c_a_original_manifest.csv';pool='outputs/st12c_a_pool_manifest.csv'
    original_items=rows(original);pool_items=rows(pool)
    check('Exact prescribed original andK4pool counts',len(original_items)==base['model']['chunks'] and len(pool_items)==4*base['model']['chunks'])
    guards=[];known=set();started=time.monotonic()
    for i,r in enumerate(original_items+pool_items,1):
        budget_check()
        for key in ['input','label']:
            path=Path(r[key+'_path'])
            if str(path) in known:continue
            before=path.stat();digest=sha(path);after=path.stat()
            check('Every new training tensor exact and stable before GPU use',digest==r[key+'_sha256'] and before.st_size==int(r[key+'_bytes']) and
                all(getattr(before,k)==getattr(after,k) for k in ['st_size','st_dev','st_ino','st_mtime_ns','st_ctime_ns']))
            known.add(str(path));guards.append(dict(path=str(path),sha256=digest,bytes=before.st_size,device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns))
        if i%100==0:progress('rhythm_training_input_checks',i,len(original_items)+len(pool_items),started)
    guard='outputs/st12c_a_training_use_guard.csv';write_rows(guard,guards)
    sources=['src/st12c/rhythm_train.py','src/st12c/rhythm_train_worker.py','src/st12c/rhythm_backend.py','src/st12c/rhythm_data.py',
        'src/st9s/streams.py','src/st9t/data.py','src/st9s/backend.py',
        'third_party/rPPG-Toolbox/neural_methods/model/RhythmFormer.py','third_party/rPPG-Toolbox/neural_methods/loss/RythmFormerLossComputer.py']
    register(sources+[original,pool],'C5',REPORTS/'st12c_a_training_source_checks.json')
    runs=[dict(run_id='st12c_a_RhythmFormer_'+arm+'_s'+str(seed),group='a',account='C5_a',model='RhythmFormer',arm=arm,seed=seed,order=i)
        for i,(arm,seed) in enumerate((a,s) for a in ['0','ii'] for s in base['seeds'])]
    settings=dict(group='a',account='C5_a',model=base['model'],seeds=base['seeds'],runs=runs,data_workers=base['data_workers'],
        source_hashes={**base['source_hashes'],**{p:sha(ROOT/p) for p in sources}},
        original_manifest=original,original_manifest_sha256=sha(ROOT/original),pool_manifest=pool,pool_manifest_sha256=sha(ROOT/pool),
        training_input_use_guard=guard,training_input_use_guard_sha256=sha(ROOT/guard),
        training='Exact official RF forward/per-sample normalization/criterion/AdamW/OneCycleLR; core paired streams; same fixed whole-batch step count for each arm; no validation; last epoch',
        checkpoint='Every completed epoch and timing checkpoint retained',
        per_use_verification='Full SHA/bytes before GPU phase; unchanged device/inode/bytes/mtime_ns/ctime_ns checked per run before loader')
    p=REPORTS/'st12c_a_training_settings.json';check('RF six-run definition once',not p.exists());save_json(p,settings)
    with (ROOT/'reports/ST4_INSTR05B_PREFIX.md').open('a') as f:f.write('\n## C5(a) exact six-run training inputs\n\n`'+path_key(p)+'` SHA-256 `'+sha(p)+'`. All tensor bytes verified before GPU training; same original/augmentation streams and official RF training math; all checkpoints retained.\n')
    return dict(runs=len(runs),verified_training_files=len(guards),settings_sha256=sha(p),commit_required_before_training=True)


if __name__=='__main__':finish('setup_rhythm_training',main)
