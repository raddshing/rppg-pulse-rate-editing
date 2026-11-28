"""OPS2 7.20 1: U-full-m drawn from the subjects of U-low (design correction before any training).

Per list (PhysNet: 555 chunks, v = 0; RhythmFormer, also used by FactorizePhys: 436 chunks, v = 2):
  S       = the subjects of U-low (chunks of the A9 item 1 list with reference HR < 100 bpm)
  C       = the chunk orders of all chunks of the list whose subject is in S, ascending (high-HR chunks included)
  n       = number of U-low chunks of the list
  U-full-m = sorted(C[i] for i in numpy.random.default_rng([20261160, 10, v]).choice(len(C), n, replace=False))
The first U-full-m lists (key [20261160, 9, v], drawn from all 42 subjects) are moved to cache/b1_719/plan/superseded_key9/
and not used. Writes cache/b1_719/plan/{PhysNet,RhythmFormer}_U_full_m_chunks.csv (same columns as before) and
outputs/719/A10/u_full_m_v2.csv (counts, SHA-256). Refuses to overwrite.
"""
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / 'tools' / 'a10'))
import b1_design as D  # noqa: E402

PLAN = W / 'cache' / 'b1_719' / 'plan'
OLD = PLAN / 'superseded_key9'
OUT = W / 'outputs' / '719' / 'A10'
K = D.K


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'u_full_m_v2.csv').exists():
        sys.exit('u_full_m_v2.csv exists')
    OLD.mkdir(exist_ok=True)
    rows = []
    for lname, v in (('PhysNet', 0), ('RhythmFormer', 2)):
        p = PLAN / f'{lname}_U_full_m_chunks.csv'
        moved = OLD / p.name
        assert not moved.exists()
        old_sha = sha(p)
        shutil.move(str(p), str(moved))
        ch = D.chunks(lname)
        low = [c for c, r in enumerate(ch) if float(r['reference_HR']) < 100]
        subj = {ch[c]['subject_id'] for c in low}
        cand = [c for c, r in enumerate(ch) if r['subject_id'] in subj]
        idx = np.random.default_rng([K, 10, v]).choice(len(cand), len(low), replace=False)
        fm = sorted(cand[int(i)] for i in idx)
        assert len(set(fm)) == len(low) and {ch[c]['subject_id'] for c in fm} <= subj
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['chunk_order', 'chunk_id', 'recording_id', 'subject_id', 'reference_HR'])
            w.writeheader()
            w.writerows(dict(chunk_order=c, chunk_id=ch[c]['chunk_id'], recording_id=ch[c]['recording_id'],
                             subject_id=ch[c]['subject_id'], reference_HR=ch[c]['reference_HR']) for c in fm)
        hr = np.array([float(ch[c]['reference_HR']) for c in fm])
        hi = [c for c in fm if float(ch[c]['reference_HR']) >= 100]
        rows.append(dict(list=lname, v=v, key=f'[{K},10,{v}]', U_low_chunks=len(low), U_low_subjects=len(subj),
                         candidate_chunks=len(cand), candidate_high_chunks=sum(float(ch[c]['reference_HR']) >= 100 for c in cand),
                         U_full_m_chunks=len(fm), U_full_m_subjects=len({ch[c]['subject_id'] for c in fm}),
                         U_full_m_high_chunks=len(hi), U_full_m_high_subjects=len({ch[c]['subject_id'] for c in hi}),
                         U_full_m_high_share=len(hi) / len(fm), U_full_m_ref_HR_median=float(np.median(hr)),
                         U_full_m_ref_HR_q25=float(np.quantile(hr, .25)), U_full_m_ref_HR_q75=float(np.quantile(hr, .75)),
                         sha256=sha(p), superseded_list_sha256=old_sha))
    with open(OUT / 'u_full_m_v2.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
