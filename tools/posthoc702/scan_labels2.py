"""7.02 다 preflight 2: recorded processed-label SHA of every item, and whether a stored server file with that SHA exists."""
import csv, os, sys, hashlib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_labels as S  # noqa: E402  (prints the first scan again; harmless)

def recorded(b, arm, r, new):
    key = (r['chunk_id'], r['augmentation_index'])
    if r.get('ops2_view_path') or r.get('ops2_label_path'):
        return new[key].get('ops2_label_sha256', '')
    return r.get('label_sha256', '')

index = {}
avail = Counter()
plans = {}
for b, arms in S.ARMS.items():
    for arm in arms:
        original, pool = S.items(b, arm)
        new = {} if arm in ('i', 'ii') else {(r['chunk_id'], r['augmentation_index']): r for r in S.T.rows(S.RUNS[(b, arm)])}
        plans[(b, arm)] = (pool, new)
        for r in pool:
            col, p = S.source(r)
            sh = recorded(b, arm, r, new)
            if os.path.exists(p) and sh:
                index.setdefault(sh, p)
        for r in original:
            if os.path.exists(r['label_path']):
                index.setdefault(r.get('label_sha256', ''), r['label_path'])
print('index', len(index))
for (b, arm), (pool, new) in plans.items():
    c = Counter()
    for r in pool:
        col, p = S.source(r)
        sh = recorded(b, arm, r, new)
        c['exists' if os.path.exists(p) else ('same_sha_stored' if sh in index else ('no_sha' if not sh else 'missing'))] += 1
    print(b, arm, dict(c))
