"""7.09b 나-1: UBFC-rPPG chunk and pool-item manifests for E1 (no pixels or model outputs are read).

Originals: the October UBFC chunk list (outputs/st30r_ubfc_chunks.csv), chunks that pass the reference check: PhysNet
555 (128 frames), RhythmFormer 436 (160 frames); FactorizePhys (matched) uses the RhythmFormer 160-frame chunks, as on PURE.
Chunk order and the October tensor SHA-256 (outputs/st30g_UBFC_<model>_original_manifest.csv) are kept so that regenerated
originals can be checked against them. LDS_weight = 1 (used only by arm vii).
Pool items, as on PURE: 4 per chunk (k = 0..3), target h* = default_rng([20261159, v, c, k]).uniform(70, 140) with
v = 0 (PhysNet) or 2 (RhythmFormer and FactorizePhys, which therefore share targets, as on PURE), c = chunk order; the PURE
first pool used the same form with 20261127, the second pool 20261158. Rounding generator of item h:
default_rng([20261159, j, h]), j = 1 PhysNet/RhythmFormer, 2 FactorizePhys, h = int(SHA-256('OPS2-709b|UBFC|<model>|<chunk_id>|<k>')[:8], 16).
tau_rec: October UBFC delay rule per recording (outputs/st10l_ubfc_tau.csv); (i-pc)/(ii-pc) use s = -1, d = 1 (UBFC audit, 7.09b).
Writes cache/ubfc709/plan/<model>_originals.csv and <model>_items.csv. Usage: python3 -B t709b_ubfc_manifests.py
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

import json  # 7.19
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.19: machine paths, see config/paths.example.json
OCT = Path(CFG['october_project']) / 'outputs'
OUT = Path(__file__).resolve().parents[1] / 'cache' / 'ubfc709' / 'plan'
KEY = 20261159
V = {'PhysNet': 0, 'RhythmFormer': 2, 'FPm': 2}
J = {'PhysNet': 1, 'RhythmFormer': 1, 'FPm': 2}
D_UBFC, S_UBFC = 1, -1


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def write(p, data):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, list(data[0]))
        w.writeheader()
        w.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tau = {r['recording_id']: int(r['tau_rec']) for r in rows(OCT / 'st10l_ubfc_tau.csv')}
    chunks = rows(OCT / 'st30r_ubfc_chunks.csv')
    idx = {r['recording_id']: r for r in rows(str(Path(CFG['october_project']) / 'data' / 'index' / 'ubfc_index_st1.csv'))}
    summary = []
    for model, view in [('PhysNet', 'PhysNet'), ('RhythmFormer', 'RhythmFormer'), ('FPm', 'RhythmFormer')]:
        octm = {r['chunk_id']: r for r in rows(OCT / f'st30g_UBFC_{view}_original_manifest.csv')}
        sel = [c for c in chunks if c['view'] == view and c['quality_pass'] == 'True']
        assert {c['chunk_id'] for c in sel} == set(octm)
        sel.sort(key=lambda c: int(octm[c['chunk_id']]['chunk_order']))
        originals, items = [], []
        for c in sel:
            o = octm[c['chunk_id']]
            assert int(c['tau_rec']) == tau[c['recording_id']] and float(c['h0']) == float(o['reference_HR'])
            base = dict(dataset='UBFC', model=model, view_index=c['view_index'], recording_id=c['recording_id'],
                        subject_id=c['subject_id'], chunk_id=c['chunk_id'], chunk_index=c['chunk_index'],
                        chunk_order=int(o['chunk_order']), start_frame=c['start_frame'], chunk_length=c['chunk_length'],
                        context_start=c['context_start'], context_offset=c['context_offset'], reference_HR=c['h0'],
                        h_ACF=c['h_ACF'], h_IBI=c['h_IBI'], reference_class=c['reference_class'], tau_rec=c['tau_rec'],
                        ground_truth_path=c['ground_truth_path'], source_video_path=c['input_path'],
                        frame_count=idx[c['recording_id']]['frame_count'])
            originals.append(dict(base, LDS_weight=1.0,
                                  october_input_sha256=o['input_sha256'] if model != 'FPm' else '',
                                  october_label_sha256=o['label_sha256'] if model != 'FPm' else ''))
            for k in range(4):
                t = float(np.random.default_rng([KEY, V[model], int(o['chunk_order']), k]).uniform(70, 140))
                h = int(hashlib.sha256(f'OPS2-709b|UBFC|{model}|{c["chunk_id"]}|{k}'.encode()).hexdigest()[:8], 16)
                items.append(dict(base, augmentation_index=k, target_HR=t, target_key=f'[{KEY},{V[model]},c,k]',
                                  rounding_h=h, rounding_key=f'[{KEY},{J[model]},h]', s_pc=S_UBFC, d_pc=D_UBFC,
                                  pool_index=len(items)))
        write(OUT / f'{model}_originals.csv', originals)
        write(OUT / f'{model}_items.csv', items)
        hr = np.array([float(r['reference_HR']) for r in originals])
        tg = np.array([r['target_HR'] for r in items])
        summary.append((model, len(originals), len({r['recording_id'] for r in originals}), int((hr >= 100).sum()),
                        len(items), float((tg >= 100).mean())))
    for s in summary:
        print('%s originals %d recordings %d originals>=100 %d items %d items>=100 share %.4f' % s)


if __name__ == '__main__':
    main()
