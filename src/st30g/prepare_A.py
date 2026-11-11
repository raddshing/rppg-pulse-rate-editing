"""Fix A2/A1 replacement/FP-m targets and cache groups before pixel generation."""
from src.st30p.ops import *
configure('st30g')
import numpy as np


def main():
    assert not (ROOT/'reports/st30g/A_pool_settings.json').exists()
    amendment=json.loads((ROOT/'reports/st30p/amendment_v1_4.json').read_text())
    assert amendment['chosen_A1']=={'PhysNet':'iv-d','RhythmFormer':'iv-d'}
    core=json.loads((ROOT/'reports/st30r/core_facts.json').read_text())
    original={'PhysNet':sorted([r for r in rows('outputs/st9g_original_manifest.csv') if r['model']=='PhysNet'],key=lambda r:int(r['chunk_order'])),
        'RhythmFormer':sorted(rows('outputs/st12c_a_original_manifest.csv'),key=lambda r:int(r['chunk_order'])),
        'FPm':sorted(rows('outputs/st30g_FPm_original_manifest.csv'),key=lambda r:int(r['chunk_order']))}
    old_pn={(int(r['chunk_order']),int(r['augmentation_index'])):float(r['target_HR']) for r in rows('outputs/st9g_pool_PhysNet_ii-h_manifest.csv')}
    old_rf={(int(r['chunk_order']),int(r['augmentation_index'])):float(r['target_HR']) for r in rows('outputs/st12c_a_pool_manifest.csv')}
    targets=[];plans=[]
    for model,arms in [('PhysNet',['ii-f','iv-d']),('FPm',['i','ii']),('RhythmFormer',['iv-d'])]:
        originals=original[model];v=0 if model=='PhysNet' else 2
        hr=np.asarray([float(r['reference_HR']) for r in originals]);low=np.flatnonzero((hr>=70)&(hr<100));high=np.flatnonzero((hr>=100)&(hr<=140))
        assert len(low) and len(high)
        for arm in arms:
            for c,r in enumerate(originals):
                assert int(r['chunk_order'])==c
                for k in range(4):
                    donor='';u=''
                    if arm=='iv-d':
                        rng=np.random.default_rng([20261201,v,c,k]);u=float(rng.uniform());pool=low if u<3/7 else high
                        donor=int(pool[int(rng.integers(len(pool)))]);target=float(hr[donor])
                    else:
                        target=float(np.random.default_rng([20261127,v,c,k]).uniform(70,140))
                        assert target==(old_pn if model=='PhysNet' else old_rf)[c,k]
                    key='ST30G|'+model+'|'+r['chunk_id']+'|'+arm+'|'+str(k)
                    h=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
                    keep={name:r[name] for name in ['chunk_id','recording_id','subject_id','task','chunk_index','start_frame',
                        'chunk_length','context_start','context_offset','reference_HR','ground_truth_path','chunk_order']}
                    targets.append(dict(keep,model=model,view_index=v,arm=arm,augmentation_index=k,planned_pool_index=4*c+k,
                        target_HR=target,donor_chunk_order=donor,mixture_u=u,rounding_key=key,rounding_seed=20261202,rounding_h=h,
                        target_excluded=False,population='PURE01-08',A_fix=core['A_fix']['value'] if arm=='ii-f' else ''))
            length=128 if model=='PhysNet' else 160;edge=128 if model=='RhythmFormer' else 72
            plans.append(dict(model=model,arm=arm,items=4*len(originals),maximum_bytes=4*len(originals)*(length*edge*edge*3*4+128+length*4+128)))
    write_rows('outputs/st30g_A_targets.csv',targets)
    direction=json.loads((ROOT/'reports/st9g/st9g_pool_settings.json').read_text())['direction_RGB']
    sources=['src/st30g/helpers.py','src/st30g/prepare_A.py','src/st30g/pool_A.py','src/st30g/pool_A_worker.py',
        'src/st30g/synthetic_A_check.py','src/st9g/preprocess.py','src/st9e/native.py','src/st9e/inherited.py','src/st9e/editor.py',
        'src/st9e/generator.py','src/st9e/temporal.py','src/st9e/model_view.py','src/st9e/view_worker.py',
        'src/st12c/rhythm_view.py','src/st12c/rhythm_view_worker.py','src/st11x/future.py',
        'src/st11x/pathways.py','src/st3/readers.py','src/eval/har.py','outputs/st3_roi_indices.json','outputs/st5_face_mask_indices.json',
        'outputs/st7b_tau_recordings.csv','outputs/st9g_original_manifest.csv','outputs/st12c_a_original_manifest.csv',
        'outputs/st30g_FPm_original_manifest.csv','data/index/pure_index_st1.csv','outputs/st12_input_hash_checks.csv',
        'reports/st30r/core_facts.json','reports/st9g/st9g_pool_settings.json',str(PATHS['face_landmarker_model'])]
    settings=dict(status='prepared',workers=7,K=4,direction_RGB=direction,A_fix=core['A_fix']['value'],plans=plans,
        target_file='outputs/st30g_A_targets.csv',targets_sha256=sha('outputs/st30g_A_targets.csv'),
        source_hashes={p:sha(p) for p in sources},groups={'PN_A':{'model':'PhysNet','arms':['ii-f','iv-d']},
            'FPm_A':{'model':'FPm','arms':['i','ii']},'RF_A':{'model':'RhythmFormer','arms':['iv-d']}},
        lifecycle='Temporary cache only, float32 inherited view/shape; BGL and C6 then all3 seed consumers then SHA-journaled exact-file deletion',
        source='Original PURE raw frames and labels; prior full source hashes reused with exact before/after metadata checks',
        amplitude='Exact inherited edited function AST; fixed A insertion only for ii-f; synthetic byte parity before generation',
        view='Original first-frame HC1.5; PhysNet/FPm72 INTER_AREA, RF128 INTER_AREA; official repro OpenCV4.5.2 view pipe',
        transforms='PN diff input/diff label;FPm Raw/Standardized;RF Standardized/Standardized; context300 then original offset except i strict length160',
        execution_environment='Existing envs/main editor runtime; frozen envs/repro CPU resizing; CUDA disabled',
        BGL_C6_required_before_training=True,no_GU_attempt=True,new_model_calls=0)
    save('reports/st30g/A_pool_settings.json',settings)
    with (ROOT/'reports/ST6_INSTR01_PREFIX.md').open('a') as f:
        f.write('\n## A pools v1.3 — frozen targets and exact generator binding\n\n')
        f.write('- `reports/st30g/A_pool_settings.json` SHA-256 `'+sha('reports/st30g/A_pool_settings.json')+'`\n')
        f.write('- `outputs/st30g_A_targets.csv` SHA-256 `'+sha('outputs/st30g_A_targets.csv')+'`;'+str(len(targets))+' fixed target rows.\n')
        f.write('- iv-d samples the existing full original chunk-HR list within the prescribed two bands; duplicate HR entries retain their empirical multiplicity. No reference-audit eligibility filter is newly imposed on the old iv-h donor list.\n')
        f.write('- Each new rounding key is ST30G|view-name|chunk-ID|arm|k with seed20261202. View names are PhysNet, RhythmFormer, FPm. ii-f targets equal every old PN(ii) target;FPm(i)/(ii) targets equal every old RF(ii) target.\n')
    return dict(status='prepared',target_rows=len(targets),plans=plans,settings_sha256=sha('reports/st30g/A_pool_settings.json'))


if __name__=='__main__':phase('st30g','prepare_A',main)
