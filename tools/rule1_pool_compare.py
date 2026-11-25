"""Diagnostic rule 1 (brief 02 6.90 라): compare the regenerated RhythmFormer (ii-pc) pool manifest with the manifest of
the pool the standard (ii-pc) runs were trained on (kept as a copy before regeneration).

Items are matched by (chunk_id, augmentation_index). Every column is compared as text except ops2_CPU_seconds (time taken).
The stored-file SHA-256 (ops2_view_file_sha256), the reloaded input and label tensor SHA-256 and the byte counts are among
the compared columns. Output: outputs/rule1/pool_regen_compare.json. Exit 1 if any compared value differs.
Usage: rule1_pool_compare.py OLD_MANIFEST NEW_MANIFEST
"""
import csv
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
SKIP = {'ops2_CPU_seconds'}


def load(p):
    rows = list(csv.DictReader(open(p, newline='')))
    return rows[0].keys() if rows else [], {(r['chunk_id'], r['augmentation_index']): r for r in rows}, len(rows)


def main():
    old_p, new_p = sys.argv[1], sys.argv[2]
    co, old, no = load(old_p)
    cn, new, nn = load(new_p)
    cols = [c for c in co if c not in SKIP]
    diff = {}
    for k in set(old) & set(new):
        for c in cols:
            if old[k].get(c) != new[k].get(c):
                diff[c] = diff.get(c, 0) + 1
    res = dict(old_rows=no, new_rows=nn, old_unique=len(old), new_unique=len(new), only_old=len(set(old) - set(new)),
               only_new=len(set(new) - set(old)), columns_equal=list(co) == list(cn), compared_columns=len(cols),
               skipped_columns=sorted(SKIP), differing_values=diff,
               input_sha_equals_original_new=sum(r['input_sha_equals_original'] == 'True' for r in new.values()),
               label_sha_equals_original_new=sum(r['label_sha_equals_original'] == 'True' for r in new.values()),
               view_bytes_new=sum(int(r['ops2_view_bytes']) + int(r['ops2_rawlabel_bytes']) + int(r['ops2_stats_bytes'] or 0)
                                  for r in new.values()))
    res['identical'] = (no == nn == len(old) == len(new) and not res['only_old'] and not res['only_new']
                        and res['columns_equal'] and not diff)
    out = WORK / 'outputs' / 'rule1' / 'pool_regen_compare.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    sys.exit(0 if res['identical'] else 1)


if __name__ == '__main__':
    main()
