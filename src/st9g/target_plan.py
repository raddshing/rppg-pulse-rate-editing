"""Freeze every source chunk, paired target and independent rounding key."""
import os
os.environ['ST9_NAMESPACE']='st9g'
from src.st9.ops import *
import numpy as np


def main():
    settings=json.loads((REPORTS/'st9g_pool_settings.json').read_text())
    check('Pool target rules committed before draws',sha(REPORTS/'st9g_pool_settings.json') in (ROOT/'reports/ST4_INSTR03_PREFIX.md').read_text())
    for p,digest in settings['source_hashes'].items():check('Pool dependency unchanged',sha(ROOT/p)==digest)
    original=rows('outputs/st9g_original_manifest.csv');K=settings['K'];targets=[]
    for model,v,arms in [('PhysNet',0,['i','ii-h','iv-h']),('FactorizePhys',1,['i','ii-h'])]:
        chunks=sorted([r for r in original if r['model']==model],key=lambda r:int(r['chunk_order']))
        hrs=np.array([float(r['reference_HR']) for r in chunks]);N=len(chunks)
        for c,r in enumerate(chunks):
            check('Exact frozen chunk order',c==int(r['chunk_order']))
            for k in range(K):
                paired=float(np.random.default_rng([20261127,v,c,k]).uniform(70,140))
                for arm in arms:
                    target=paired;draws=1;excluded=False;donor=None
                    if arm=='iv-h':
                        rng=np.random.default_rng([20261128,v,c,k])
                        for draws in range(1,101):
                            donor=int(rng.integers(0,N));target=float(hrs[donor])
                            if 70<=target<=140:break
                        else:excluded=True
                    key=f'ST9G|{model}|{r["chunk_id"]}|{arm}|{k}'
                    targets.append(dict(model=model,view_index=v,arm=arm,chunk_order=c,chunk_id=r['chunk_id'],
                        augmentation_index=k,planned_pool_index=c*K+k,recording_id=r['recording_id'],subject_id=r['subject_id'],task=r['task'],
                        chunk_index=int(r['chunk_index']),start_frame=int(r['start_frame']),chunk_length=int(r['chunk_length']),
                        context_start=int(r['context_start']),context_offset=int(r['context_offset']),reference_HR=float(r['reference_HR']),
                        target_HR=target,donor_chunk_order=donor,draw_count=draws,redraw_count=draws-1,target_excluded=excluded,
                        rounding_h=int(hashlib.sha256(key.encode()).hexdigest()[:8],16),rounding_seed=20261137,
                        population='frozen_Stage1_PURE_chunks;K4_prespecified_targets',ground_truth_path=r['ground_truth_path']))
    destination=ROOT/'outputs/st9g_targets.csv';check('Never overwrite the target table',not destination.exists())
    write_rows(destination,targets)
    receipt=dict(K=K,planned_items=len(targets),target_exclusions=sum(r['target_excluded'] for r in targets),
        targets_sha256=sha(destination),targets_path=str(destination),matching_i_iih_targets=True,
        source_manifest_sha256=sha(ROOT/'outputs/st9g_original_manifest.csv'),commit_required_before_generation=True)
    save_json(REPORTS/'st9g_targets_receipt.json',receipt);return receipt


if __name__=='__main__':finish('targets',main)
