"""Freeze FP-m (0) inputs and the unchanged FactorizePhys training backend."""
from src.st30p.ops import *
configure('st30t')
import copy
import yaml


def main():
    complete=json.loads((ROOT/'reports/st30g/FPm_originals_completion.json').read_text())
    assert complete['status']=='complete' and complete['chunks']==596
    old_path='reports/st9t/st9t_settings.json'
    old=json.loads((ROOT/old_path).read_text());config=copy.deepcopy(old['models']['FactorizePhys'])
    official=ROOT/config['official_config'];assert sha(official)==config['config_sha256']
    value=yaml.safe_load(official.read_text());data=value['TRAIN']['DATA']
    pure_roots={str(Path(r['path']).parent) for r in rows('data/index/pure_index_st1.csv')}
    assert len(pure_roots)==1
    data['END']=.8;data['DATA_PATH']=next(iter(pure_roots))
    data['CACHED_PATH']=str(Path(PATHS['bulk'])/'st30g/FPm/original')
    crop=data['PREPROCESS']['CROP_FACE'];crop['BACKEND']='HC';crop['DETECTION']['DO_DYNAMIC_DETECTION']=False
    dest=ROOT/'configs/st30t/FPm_0.yaml';dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();dest.write_text(yaml.safe_dump(value,sort_keys=False))
    original=sorted(rows('outputs/st30g_FPm_original_manifest.csv'),key=lambda r:int(r['chunk_order']))
    assert len(original)==596 and [int(r['chunk_order']) for r in original]==list(range(596))
    assert sha('outputs/st30g_FPm_original_manifest.csv')==complete['manifest_sha256']
    guards={r['path']:r for r in rows('reports/st30g/FPm_original_guards.csv')}
    assert len(guards)==1192
    for row in original:
        for kind in ['input','label']:
            p=Path(row[kind+'_path']);g=guards[str(p)];s=p.stat()
            assert g['sha256']==row[kind+'_sha256']
            assert all(int(g[k])==v for k,v in dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns).items())
    config.update(official_config=str(dest.relative_to(ROOT)),config_sha256=sha(dest),
        training_subjects=sorted({r['subject_id'] for r in original}),
        training_recordings=sorted({r['recording_id'] for r in original}),chunks=596,steps_per_epoch=149,
        crop_face=copy.deepcopy(crop))
    sources=[old_path,str(official),str(dest),'outputs/st30g_FPm_original_manifest.csv',
        'reports/st30g/FPm_original_guards.csv','reports/st30g/FPm_original_source_bindings.json',
        'reports/st9t/st9t_initial_weights.json','src/st9s/backend.py','src/st9s/streams.py',
        'src/st9t/data.py','src/st30t/train.py','src/st30t/fp_train.py','src/st30t/queue_fp.py']
    sources.extend(old['source_hashes'])
    for p,h in old['source_hashes'].items():assert sha(p)==h,p
    runs=[r for r in rows('reports/st30p/run_plan.csv') if r['model']=='FPm' and r['arm']=='0']
    assert len(runs)==3 and all(int(r['steps'])==1490 for r in runs)
    settings=dict(status='prepared',model=config,model_name='FPm',source_settings=old_path,
        original_manifest='outputs/st30g_FPm_original_manifest.csv',input_guards='reports/st30g/FPm_original_guards.csv',
        data_workers=old['data_workers'],runs=runs,source_hashes={p:sha(p) for p in sources},
        config_copy=str(dest.relative_to(ROOT)),checkpoint_reservation_bytes=32*1024**2,
        changes=['PURE01-08; exact RF596 chunk boundaries','first-recording-frame HC1.5 fixed box;72INTER_AREA',
            '149 steps/epoch;10 epochs;1490 total as directly prescribed','new ST6 paths'],
        unchanged=['FactorizePhys model/loss/Adam/OneCycleLR','Raw input;Standardized labels;160 frames',
            'batch4;seed streams;16 data workers;prefetch2;persistent workers','float32 numerical environment'],
        label_exposure_rule='Frozen original reference_HR plus common_HR on stored160 Standardized label; L_max uses actually exposed stored labels',
        evaluation_hold='All nine prescribed FPm training runs must complete before FPm evaluation')
    path='reports/st30t/FPm0_settings.json';save(path,settings)
    with (ROOT/'reports/ST6_INSTR01_PREFIX.md').open('a') as f:
        f.write('\n## A4 FP-m (0) execution configuration\n\n')
        f.write('Three fixed seeds; unchanged FactorizePhys model, loss, optimizer and learning rate. Prescribed PURE01-08/RF596 boundaries and fixed HC1.5 72-pixel view;10×149 steps. Last epoch retained; no validation.\n\n')
        f.write('- `'+path+'` SHA-256 `'+sha(path)+'`\n')
        f.write('- `'+str(dest.relative_to(ROOT))+'` SHA-256 `'+sha(dest)+'`\n')
        f.write('- Original manifest SHA-256 `'+sha(settings['original_manifest'])+'`;1192 newly generated file metadata guards. FP-m evaluation waits all nine runs.\n')
    return dict(status='prepared',runs=3,input_files=len(guards),settings_sha256=sha(path))


if __name__=='__main__':phase('st30t','prepare_FPm0',main)
