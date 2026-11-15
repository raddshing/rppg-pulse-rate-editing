"""Brief 02 6.67 다 6: a second run of step (c) must reproduce every row of the first byte for byte. Compares two
eval5/stage3_analyze.py output folders of the same dataset and population: the hash lists (analysis_hashes.txt; every table
of the run) must be equal, every listed file must hash to its listed value in both folders, and the two folders must hold the
same files. run.json is compared without its start and end times. Writes OUT.csv (file, result) and prints a summary; exit 0
only if everything is identical. Usage: compare_stage3_runs.py FIRST_DIR SECOND_DIR OUT.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def listing(d):
    out = {}
    for line in (d / 'analysis_hashes.txt').read_text().splitlines():
        h, name = line.split('  ', 1)
        out[name] = h
    return out


def main():
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    la, lb = listing(a), listing(b)
    rows = []
    for name in sorted(set(la) | set(lb)):
        base = name.split('/')[-1]
        fa, fb = a / base, b / base
        if name not in la or name not in lb:
            rows.append(dict(file=base, result='listed in one run only'))
            continue
        ok_a = fa.exists() and sha(fa) == la[name]
        ok_b = fb.exists() and sha(fb) == lb[name]
        rows.append(dict(file=base, result='identical' if la[name] == lb[name] and ok_a and ok_b else
                         ('differs' if ok_a and ok_b else 'file does not match its listed hash')))
    files_a = {p.name for p in a.iterdir() if p.is_file()} - {'run.json', 'analysis_hashes.txt'}
    files_b = {p.name for p in b.iterdir() if p.is_file()} - {'run.json', 'analysis_hashes.txt'}
    listed = {n.split('/')[-1] for n in la}
    for n in sorted((files_a | files_b) - listed):
        rows.append(dict(file=n, result='not in the hash list'))
    ra, rb = json.loads((a / 'run.json').read_text()), json.loads((b / 'run.json').read_text())
    for k in ('started', 'ended'):
        ra.pop(k, None), rb.pop(k, None)
    rows.append(dict(file='run.json (without start and end times)', result='identical' if ra == rb else 'differs'))
    r = pd.DataFrame(rows)
    r.to_csv(out, index=False)
    bad = r[r.result.ne('identical')]
    print('files', len(r), 'identical', int(r.result.eq('identical').sum()), 'other', len(bad))
    if len(bad):
        print(bad.to_string(index=False))
        sys.exit(3)


if __name__ == '__main__':
    main()
