"""7.02 다 preflight: for every trained arm, the pool item list as the trainer builds it (train/train_pc.py setup + manifest
merge) and the label file each item would be read from; counts how many of those files exist on the server (read only)."""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / 'train'))
os.environ.setdefault('OPS2_ARM', 'ii')
import train_pc as T  # noqa: E402  (chdir to vendor; read-only setup functions)

RUNS = {}
for f in sorted((WORK / 'cache' / 'train').glob('*_s20261131/result.json')):
    b, rest = f.parent.name.split('_', 1)
    RUNS[(b, rest.rsplit('_s', 1)[0])] = json.loads(f.read_text())['pool_manifest']


def items(b, arm):
    kind = 'i' if arm == 'i' or arm.startswith('i-') else 'ii'
    if arm in ('i', 'ii'):
        config, original, pool, *_ = T.setup(b, kind)
        return original, pool
    config, original, pool, *_ = T.setup(b, kind)
    new = {(r['chunk_id'], r['augmentation_index']): r for r in T.rows(RUNS[(b, arm)])}
    keys = ('ops2_view_path', 'ops2_rawlabel_path', 'ops2_stats_path', 'ops2_label_path', 'ops2_view_offset')
    pool = [dict(r, **{k: new[(r['chunk_id'], r['augmentation_index'])].get(k, '') for k in keys})
            if (r['chunk_id'], r['augmentation_index']) in new else r for r in pool]
    pool = [dict(r, target_HR=new[(r['chunk_id'], r['augmentation_index'])]['target_HR'])
            if new.get((r['chunk_id'], r['augmentation_index']), {}).get('target_HR_original') not in (None, '') else r for r in pool]
    return original, pool


def source(r):
    if r.get('ops2_view_path'):
        return 'ops2_rawlabel_path', r['ops2_rawlabel_path']
    if r.get('ops2_label_path'):
        return 'ops2_label_path', r['ops2_label_path']
    return 'label_path', r['label_path']


ARMS = {'PhysNet': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'i-pc-g80', 'sham-pc', 'roi-pc'],
        'RhythmFormer': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag'],
        'FPm': ['i', 'i-pc', 'i-simple', 'ii', 'ii-pc', 'ii-lag', 'ii-pc-g', 'ii-pc-g80', 'ii-pc-g2', 'sham-pc']}
for b, arms in ARMS.items():
    for arm in arms:
        original, pool = items(b, arm)
        c = Counter()
        for r in pool:
            col, p = source(r)
            c[(col, os.path.exists(p))] += 1
        oc = Counter(os.path.exists(r['label_path']) for r in original)
        print(b, arm, 'pool', len(pool), dict(c), '| originals', len(original), dict(oc), flush=True)
