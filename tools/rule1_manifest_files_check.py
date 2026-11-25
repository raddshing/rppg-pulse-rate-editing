"""Brief 02 6.94 나 2: before the (ii-pc) one-step check, open the files of every row of the restored RhythmFormer
polarity_explicit manifest (the copy of the manifest of the standard (ii-pc) runs) through its three path columns.

Per row and path column (ops2_view_path, ops2_rawlabel_path, ops2_stats_path): the file exists; its size equals the
recorded byte column (ops2_view_bytes, ops2_rawlabel_bytes, ops2_stats_bytes). Recorded SHA-256 values recomputed:
  - ops2_view_file_sha256: SHA-256 of the view file;
  - ops2_input_sha256, ops2_label_sha256, ops2_input_data_sha256, ops2_label_data_sha256: SHA-256 of the model input and
    label built from the three files exactly as the training loader does (train/u8_store.py item_input, unchanged).
Byte total of the three columns' files is compared with 7,030,584,773 (the trained pool's server copy).
Usage: rule1_manifest_files_check.py MANIFEST OUT_JSON WORKERS
"""
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'train'))
import u8_store  # noqa: E402

COLS = (('ops2_view_path', 'ops2_view_bytes'), ('ops2_rawlabel_path', 'ops2_rawlabel_bytes'),
        ('ops2_stats_path', 'ops2_stats_bytes'))
EXPECTED_BYTES = 7030584773


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check(r):
    out = dict(key=(r['chunk_id'], r['augmentation_index']))
    for pc, bc in COLS:
        p = r[pc]
        ok = bool(p) and os.path.isfile(p)
        out[pc + '_exists'] = ok
        out[pc + '_size'] = os.path.getsize(p) if ok else None
        out[pc + '_size_equal'] = ok and out[pc + '_size'] == int(r[bc])
    if not all(out[pc + '_exists'] for pc, _ in COLS):
        return out
    out['view_file_sha_equal'] = file_sha(r['ops2_view_path']) == r['ops2_view_file_sha256']
    x, y = u8_store.item_input('RhythmFormer', r)
    out['input_sha_equal'] = u8_store.npy_sha(x) == r['ops2_input_sha256']
    out['label_sha_equal'] = u8_store.npy_sha(y) == r['ops2_label_sha256']
    out['input_data_sha_equal'] = u8_store.data_sha(x) == r['ops2_input_data_sha256']
    out['label_data_sha_equal'] = u8_store.data_sha(y) == r['ops2_label_data_sha256']
    return out


def main():
    man, dst, workers = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    rows = list(csv.DictReader(open(man, newline='')))
    with Pool(workers) as pool:
        res = pool.map(check, rows, chunksize=8)
    shas = ('view_file_sha_equal', 'input_sha_equal', 'label_sha_equal', 'input_data_sha_equal', 'label_data_sha_equal')
    s = dict(manifest_sha256=file_sha(man), rows=len(rows), unique_keys=len({r['key'] for r in res}))
    for pc, _ in COLS:
        s[pc] = dict(nonempty=sum(bool(r[pc]) for r in rows), exists=sum(r[pc + '_exists'] for r in res),
                     size_equal_recorded=sum(bool(r[pc + '_size_equal']) for r in res),
                     bytes=sum(r[pc + '_size'] or 0 for r in res))
    s['sha_recomputed'] = {k: dict(rows=sum(k in r for r in res), equal=sum(bool(r.get(k)) for r in res)) for k in shas}
    s['bytes_total'] = sum(s[pc]['bytes'] for pc, _ in COLS)
    s['bytes_expected'] = EXPECTED_BYTES
    s['files_total'] = sum(s[pc]['exists'] for pc, _ in COLS)
    s['all_pass'] = bool(
        s['unique_keys'] == len(rows)
        and all(s[pc]['exists'] == s[pc]['size_equal_recorded'] == len(rows) for pc, _ in COLS)
        and all(v['rows'] == v['equal'] == len(rows) for v in s['sha_recomputed'].values())
        and s['bytes_total'] == EXPECTED_BYTES)
    dst.write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    sys.exit(0 if s['all_pass'] else 1)


if __name__ == '__main__':
    main()
