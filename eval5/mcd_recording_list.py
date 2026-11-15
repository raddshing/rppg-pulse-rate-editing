"""Step (a) (ACCESS_OPS2_BRIEF_02 6.69 가 2): the MCD-rPPG recording list of one half, made from the half's subject list of the
pre-registration manifest with the code used for the development half.
  1. Candidates: the rows of db.csv of the listed subjects (rows of other subjects are dropped before use), front view,
     FullHDwebcam camera, one recording per subject and state (recording_id = <subject>_<state>), as
     eval5/stage2_predict.mcd_recordings reads them.
  2. Section-3 recording selection, confirmed frame-to-PPG correspondence: the model-free pass of the development half
     (eval5/mcd_reference.job = b1/mcd_meta.job, unchanged) on every candidate; the list keeps the recordings with
     mapping_confirmed. (The development list of 404 recordings is the confirmed subset of its 476 candidates; the other 72
     have alignment_status raw_clock_irregular.) At least one high-HR window is applied by stage 1 (draft-rule population).
The subject list must have the SHA-256 of the given manifest row. Writes OUT.csv (recording_id) and OUT_candidates.csv
(candidate, alignment status, frame-count class, mapping_confirmed); prints the counts and the list's SHA-256.
Usage: mcd_recording_list.py VENDOR MCD_ROOT SUBJECT_LIST MANIFEST_CSV MANIFEST_ROW_KIND OUT.csv [WORKERS]
"""
import csv
import hashlib
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def init(vendor):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import stage1_reference
    stage1_reference.vendor_imports(vendor)


def meta_job(row):
    import mcd_reference
    rec, _ = mcd_reference.job(row)
    return dict(recording_id=row['recording_id'], alignment_status=rec.get('alignment_status', ''),
                frame_count_class=rec.get('frame_count_class', ''), mapping_confirmed=bool(rec['mapping_confirmed']))


def main():
    vendor, root, subj, manifest, kind, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
    workers = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    want = {r['kind']: r['sha256'] for r in csv.DictReader(open(manifest, newline=''))}
    sha = hashlib.sha256(subj.read_bytes()).hexdigest()
    assert want.get(kind) == sha, f'subject list SHA-256 {sha} is not the manifest row {kind}'
    keep = set(subj.read_text().split())
    cands = {}
    with (root / 'db.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['patient_id'] not in keep:
                continue  # dropped before use
            if r['view'] == 'front' and r['camera'] == 'FullHDwebcam':
                rid = f"{r['patient_id']}_{r['step']}"
                assert rid not in cands, rid
                cands[rid] = dict(recording_id=rid, subject_id=r['patient_id'], state=r['step'], video=str(root / r['video']),
                                  meta=str(root / r['meta']), ppg=str(root / r['ppg']), ppg_sync=str(root / r['ppg_sync']))
    rows = [cands[k] for k in sorted(cands)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init, initargs=(vendor,)) as ex:
        res = list(ex.map(meta_job, rows))
    cand_out = out.with_name(out.stem + '_candidates.csv')
    with cand_out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    kept = [r['recording_id'] for r in res if r['mapping_confirmed']]
    with out.open('w', newline='') as f:
        f.write('recording_id\n' + ''.join(r + '\n' for r in kept))
    print('candidates', len(rows), 'confirmed (listed)', len(kept), 'subjects', len({r.split('_')[0] for r in kept}),
          'list sha256', hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
